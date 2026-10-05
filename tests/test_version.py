"""Version records: deterministic source hashes and safe metadata fallbacks."""

from __future__ import annotations

import os
import re
import subprocess
from types import SimpleNamespace

import pytest

from runner import version as truco_version
from runner.version import _code_version, code_version, source_sha256


def _write(root: str, relative: str, text: str) -> None:
  """Write ``text`` beneath ``root``, creating parent directories."""
  filename = os.path.join(root, relative)
  os.makedirs(os.path.dirname(filename), exist_ok=True)
  with open(filename, "w", encoding="utf-8") as f:
    f.write(text)


def test_source_sha256_is_deterministic_and_independent_of_creation_order(tmp_path):
  files = [
      ("runner/z.py", "z = 1\n"),
      ("runner/nested/a.py", "a = 1\n"),
      ("truco/match.py", "match = 1\n"),
      ("openrouter_model.py", "model = 1\n"),
      ("kbench_model.py", "kbench = 1\n"),
  ]
  roots = [str(tmp_path / name) for name in ("first", "second")]
  for root, ordered in zip(roots, (files, list(reversed(files)))):
    for relative, text in ordered:
      _write(root, relative, text)
  assert source_sha256(roots[0]) == source_sha256(roots[0])
  assert source_sha256(roots[0]) == source_sha256(roots[1])


def test_source_sha256_changes_when_python_content_changes(tmp_path):
  root = str(tmp_path)
  _write(root, "runner/agent.py", "answer = 1\n")
  before = source_sha256(root)
  _write(root, "runner/agent.py", "answer = 2\n")
  assert source_sha256(root) != before


def test_source_sha256_ignores_pycache_and_non_python_files(tmp_path):
  root = str(tmp_path)
  _write(root, "runner/agent.py", "answer = 1\n")
  before = source_sha256(root)
  _write(root, "runner/__pycache__/cached.py", "cached = 1\n")
  _write(root, "runner/nested/__pycache__/cached.py", "cached = 2\n")
  _write(root, "runner/notes.txt", "not source\n")
  _write(root, "tests/test_agent.py", "not packaged\n")
  assert source_sha256(root) == before


@pytest.mark.parametrize("git_file", [False, True])
def test_code_version_without_git_directory_never_uses_parent_repo(tmp_path, monkeypatch, git_file):
  os.mkdir(os.path.join(tmp_path, ".git"))
  root = os.path.join(tmp_path, "installed")
  os.mkdir(root)
  if git_file:
    _write(root, ".git", "gitdir: ../.git\n")
  calls = []

  def run(*args, **kwargs):
    calls.append(args)
    raise AssertionError("Git must not run without a .git directory at the root")

  monkeypatch.setattr(truco_version.subprocess, "run", run)
  record = _code_version(root)
  assert record["git_commit"] is None and record["git_dirty"] is None
  assert calls == []


@pytest.mark.parametrize("status", ["", " M runner/agents.py\n"])
def test_code_version_records_git_commit_and_scoped_dirty_state(tmp_path, monkeypatch, status):
  os.mkdir(os.path.join(tmp_path, ".git"))
  calls = []

  def run(command, **kwargs):
    calls.append(command)
    assert kwargs == {
        "cwd": str(tmp_path), "timeout": 5, "capture_output": True, "text": True,
    }
    return subprocess.CompletedProcess(command, 0, "abc123\n" if command[1] == "rev-parse" else status)

  monkeypatch.setattr(truco_version.subprocess, "run", run)
  record = _code_version(str(tmp_path))
  assert record["git_commit"] == "abc123"
  assert record["git_dirty"] is bool(status)
  assert calls == [
      ["git", "rev-parse", "HEAD"],
      ["git", "status", "--porcelain", "--", "truco", "runner",
       "openrouter_model.py", "kbench_model.py"],
  ]


@pytest.mark.parametrize("failed_command", ["rev-parse", "status"])
@pytest.mark.parametrize("timeout", [False, True])
def test_git_failures_preserve_other_version_fields(tmp_path, monkeypatch, failed_command, timeout):
  os.mkdir(os.path.join(tmp_path, ".git"))

  def run(command, **kwargs):
    if command[1] == failed_command:
      if timeout:
        raise subprocess.TimeoutExpired(command, 5)
      return subprocess.CompletedProcess(command, 1, "error")
    return subprocess.CompletedProcess(command, 0, "abc123\n" if command[1] == "rev-parse" else "")

  monkeypatch.setattr(truco_version.subprocess, "run", run)
  record = _code_version(str(tmp_path))
  assert record["git_commit"] == (None if failed_command == "rev-parse" else "abc123")
  assert record["git_dirty"] is (None if failed_command == "status" else False)
  assert record["source_sha256"] == source_sha256(str(tmp_path))


@pytest.mark.parametrize("direct_url, commit", [
    ('{"vcs_info": {"commit_id": "harness123"}}', "harness123"),
    (None, None),
    ("not JSON", None),
    ('{"vcs_info": null}', None),
])
def test_metadata_failures_preserve_available_fields(tmp_path, monkeypatch, direct_url, commit):
  def unavailable_source(root):
    raise OSError("Cannot read source")

  def package_version(package):
    if package == "kaggle-benchmarks":
      raise truco_version.importlib.metadata.PackageNotFoundError(package)
    return "1.0"

  def distribution(package):
    assert package == "game-arena"
    return SimpleNamespace(read_text=lambda filename: direct_url)

  monkeypatch.setattr(truco_version, "source_sha256", unavailable_source)
  monkeypatch.setattr(truco_version.importlib.metadata, "version", package_version)
  monkeypatch.setattr(truco_version.importlib.metadata, "distribution", distribution)
  record = _code_version(str(tmp_path))
  assert record["source_sha256"] is None
  assert record["packages"] == {
      "truco-eval": "1.0", "kaggle-benchmarks": None, "game-arena": "1.0",
  }
  assert record["game_arena_commit"] == commit


def test_code_version_has_repo_source_hash_and_package_version():
  record = code_version()
  assert re.fullmatch(r"[0-9a-f]{64}", record["source_sha256"])
  assert record["packages"]["truco-eval"] is not None
