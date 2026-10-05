"""Record source and dependency versions; unavailable metadata stays null."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
from functools import lru_cache
from typing import Any

import runner

PACKAGE_FILES = ("truco", "runner", "openrouter_model.py", "kbench_model.py")


def source_sha256(root: str) -> str:
  """Hash the Python source files under the packaged project entries."""
  root = os.path.abspath(root)
  files: list[tuple[str, str]] = []
  for entry in PACKAGE_FILES:
    path = os.path.join(root, entry)
    if os.path.isdir(path):
      for directory, directories, names in os.walk(path):
        directories[:] = [name for name in directories if name != "__pycache__"]
        for name in names:
          if not name.endswith(".py"):
            continue
          filename = os.path.join(directory, name)
          relative = os.path.relpath(filename, root).replace(os.sep, "/")
          files.append((relative, filename))
    elif os.path.isfile(path) and path.endswith(".py"):
      relative = os.path.relpath(path, root).replace(os.sep, "/")
      files.append((relative, path))

  digest = hashlib.sha256()
  for relative, filename in sorted(files):
    with open(filename, "rb") as source:
      file_digest = hashlib.sha256(source.read()).hexdigest()
    digest.update(f"{relative}\0{file_digest}\n".encode("utf-8"))
  return digest.hexdigest()


def _git_output(root: str, args: list[str]) -> str | None:
  """Run Git only at a checkout root, returning ``None`` on failure."""
  try:
    if not os.path.isdir(os.path.join(root, ".git")):
      return None
    result = subprocess.run(
        ["git", *args], cwd=root, timeout=5,
        capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None
  except Exception:  # pylint: disable=broad-exception-caught
    return None


def _game_arena_commit() -> str | None:
  """Read the VCS commit recorded by the installed game-arena distribution."""
  try:
    direct_url = importlib.metadata.distribution("game-arena").read_text("direct_url.json")
    if not direct_url:
      return None
    metadata = json.loads(direct_url)
    vcs_info = metadata.get("vcs_info")
    if not isinstance(vcs_info, dict):
      return None
    commit = vcs_info.get("commit_id")
    return commit if isinstance(commit, str) and commit else None
  except Exception:  # pylint: disable=broad-exception-caught
    return None


def _code_version(root: str) -> dict[str, Any]:
  """Collect source, checkout, and installed-package versions without raising."""
  result: dict[str, Any] = {
      "source_sha256": None,
      "git_commit": None,
      "git_dirty": None,
      "packages": {
          "truco-eval": None,
          "kaggle-benchmarks": None,
          "game-arena": None,
      },
      "game_arena_commit": None,
  }
  try:
    result["source_sha256"] = source_sha256(root)
  except Exception:  # pylint: disable=broad-exception-caught
    pass
  result["git_commit"] = _git_output(root, ["rev-parse", "HEAD"]) or None
  status = _git_output(root, ["status", "--porcelain", "--", *PACKAGE_FILES])
  result["git_dirty"] = bool(status) if status is not None else None
  for package in result["packages"]:
    try:
      result["packages"][package] = importlib.metadata.version(package)
    except Exception:  # pylint: disable=broad-exception-caught
      pass
  result["game_arena_commit"] = _game_arena_commit()
  return result


@lru_cache(maxsize=1)
def code_version() -> dict[str, Any]:
  """Return the cached version record for this checkout or installed wheel."""
  try:
    root = os.path.dirname(os.path.dirname(os.path.abspath(runner.__file__)))
    return _code_version(root)
  except Exception:  # pylint: disable=broad-exception-caught
    return {
        "source_sha256": None,
        "git_commit": None,
        "git_dirty": None,
        "packages": {
            "truco-eval": None,
            "kaggle-benchmarks": None,
            "game-arena": None,
        },
        "game_arena_commit": None,
    }
