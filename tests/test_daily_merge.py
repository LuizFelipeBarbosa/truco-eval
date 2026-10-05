"""Tests for merging the daily Kaggle tournament outputs."""

from __future__ import annotations

import importlib.util
import os
import shutil

from runner import version as truco_version
from runner.config import KIND_RANDOM, ModelConfig
from runner.tournament import run_tournament


def _load_daily():
  path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "kaggle_task", "daily.py")
  spec = importlib.util.spec_from_file_location("daily_for_test", path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _run(root: str, label: str, seeds: list[int]) -> None:
  models = [ModelConfig(kind=KIND_RANDOM, label="a"), ModelConfig(kind=KIND_RANDOM, label="b")]
  run_tournament(models, seeds=seeds,
                 out_root=os.path.join(root, f"kaggle_cheap_rr_{label}", "truco_runs"),
                 progress=lambda *_: None)


def test_merge_reports_superseded_and_unpaired_runs(tmp_path):
  root = os.fspath(tmp_path)
  _run(root, "v2", [5])
  shutil.rmtree(os.path.join(root, "kaggle_cheap_rr_v2", "truco_runs", "a__vs__b",
                             "seed5_dup"))
  _run(root, "v3", [5, 6])
  _run(root, "v4", [7])
  shutil.rmtree(os.path.join(root, "kaggle_cheap_rr_v4", "truco_runs", "a__vs__b",
                             "seed7_dup"))

  report = _load_daily().merge_all(runs_root=root)

  assert report["superseded"] == [{
      "pairing": "a__vs__b",
      "match_id": "seed5_orig",
      "kept": "kaggle_cheap_rr_v3",
      "dropped": "kaggle_cheap_rr_v2",
  }]
  assert report["unpaired_excluded"] == [{
      "pairing": "a__vs__b",
      "match_id": "seed7_orig",
      "run": "kaggle_cheap_rr_v4",
  }]
  assert report["matches_played"] == 4
  assert [r["run"] for r in report["runs"]] == [
      "kaggle_cheap_rr_v2", "kaggle_cheap_rr_v3", "kaggle_cheap_rr_v4"]
  assert [r["code_version"] for r in report["runs"]] == [truco_version.code_version()] * 3

  with open(os.path.join(root, "kaggle_cheap_rr_leaderboard.txt"), encoding="utf-8") as f:
    leaderboard = f.read()
  assert "Superseded (later run kept):" in leaderboard
  assert "Unpaired, excluded:" in leaderboard
