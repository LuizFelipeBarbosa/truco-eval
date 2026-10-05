"""Tournament reports and resume behavior using only local bots and stubs."""

from __future__ import annotations

import json
import os
import threading
import time

from runner import cli
from runner import tournament
from runner.config import ModelConfig

BOTS = [ModelConfig(kind="random", label=f"bot{i}") for i in range(3)]


def test_pairings_and_dirs(tmp_path):
  assert [(a.display, b.display) for a, b in tournament.pairings(BOTS)] == [
      ("bot0", "bot1"), ("bot0", "bot2"), ("bot1", "bot2")]
  specs = tournament.tournament_specs(BOTS, seeds=[0, 1], duplicate=True, out_root=str(tmp_path))
  assert len(specs) == 3 * 2 * 2
  assert specs[0].out_dir == os.path.join(str(tmp_path), "bot0__vs__bot1", "seed0_orig")


def test_run_tournament_report_and_resume(tmp_path):
  calls = []
  from runner.match_runner import play_match
  def counting(spec):
    calls.append(spec.match_id)
    return play_match(spec)
  rep = tournament.run_tournament(BOTS, seeds=[0, 1], out_root=str(tmp_path), duplicate=True,
                                  parallel=2, progress=lambda m: None, play_fn=counting)
  assert rep["matches_played"] == 12 and len(calls) == 12
  assert {r["model"] for r in rep["standings"]} == {"bot0", "bot1", "bot2"}
  assert sum(r["wins"] for r in rep["standings"]) == 12
  h = rep["head_to_head"]
  assert h["bot0"]["bot1"]["matches"] == 4
  assert h["bot0"]["bot1"]["wins"] + h["bot1"]["bot0"]["wins"] == 4
  assert os.path.exists(os.path.join(str(tmp_path), "tournament.json"))
  with open(os.path.join(str(tmp_path), "tournament.json"), encoding="utf-8") as f:
    assert json.load(f)["code_version"] == rep["code_version"]
  assert len(rep["code_version"]["source_sha256"]) == 64
  assert "Head-to-head" in tournament.format_report(rep)
  assert "ratings" in rep and all("bt_elo" in r for r in rep["standings"])
  assert "BT Elo" in tournament.format_report(rep)
  # Resume: nothing is replayed, report is identical.
  calls.clear()
  rep2 = tournament.run_tournament(BOTS, seeds=[0, 1], out_root=str(tmp_path), duplicate=True,
                                   progress=lambda m: None, play_fn=counting)
  assert calls == [] and rep2["matches_played"] == 12
  assert rep2["head_to_head"] == rep["head_to_head"]
  # Adding a seed only plays the new matches.
  tournament.run_tournament(BOTS, seeds=[0, 1, 2], out_root=str(tmp_path), duplicate=True,
                            progress=lambda m: None, play_fn=counting)
  assert sorted(set(calls)) == ["seed2_dup", "seed2_orig"] and len(calls) == 6


def test_run_tournament_interleaves_each_pair_before_next_pair(tmp_path):
  models = [ModelConfig(kind="random", label=f"bot{i}") for i in range(3)]
  calls = []

  def recording(spec):
    calls.append((spec.seed, spec.team_a.display, spec.team_b.display, spec.swap))
    return {
        "match_id": spec.match_id,
        "seed": spec.seed,
        "swap": spec.swap,
        "winner_team": "A",
        "team_config": {"A": spec.config_for_team("A").to_dict(),
                         "B": spec.config_for_team("B").to_dict()},
        "winner_config": spec.team_a.display,
        "scores": {"A": 12, "B": 0},
        "hands_played": 1,
        "team_stats": {"A": {"cost_usd": 0.0}, "B": {"cost_usd": 0.0}},
    }

  tournament.run_tournament(models, seeds=[1, 2], out_root=str(tmp_path), parallel=1,
                            progress=lambda m: None, play_fn=recording)
  assert calls == [
      (1, "bot0", "bot1", False), (1, "bot0", "bot1", True),
      (1, "bot0", "bot2", False), (1, "bot0", "bot2", True),
      (1, "bot1", "bot2", False), (1, "bot1", "bot2", True),
      (2, "bot0", "bot1", False), (2, "bot0", "bot1", True),
      (2, "bot0", "bot2", False), (2, "bot0", "bot2", True),
      (2, "bot1", "bot2", False), (2, "bot1", "bot2", True),
  ]


def test_run_tournament_parallel_starts_duplicate_after_its_original_finishes(tmp_path):
  models = [ModelConfig(kind="random", label=f"bot{i}") for i in range(4)]
  events = []
  lock = threading.Lock()

  def recording(spec):
    key = (spec.seed, spec.team_a.display, spec.team_b.display)
    with lock:
      events.append(("start", key, spec.swap))
    time.sleep(0.005 if spec.swap else 0.02)  # slow originals invite a racing duplicate
    with lock:
      events.append(("finish", key, spec.swap))
    return {
        "match_id": spec.match_id,
        "seed": spec.seed,
        "swap": spec.swap,
        "winner_team": "A",
        "team_config": {"A": spec.config_for_team("A").to_dict(),
                         "B": spec.config_for_team("B").to_dict()},
        "winner_config": spec.team_a.display,
        "scores": {"A": 12, "B": 0},
        "hands_played": 1,
        "team_stats": {"A": {"cost_usd": 0.0}, "B": {"cost_usd": 0.0}},
    }

  rep = tournament.run_tournament(models, seeds=[1, 2, 3], out_root=str(tmp_path), parallel=4,
                                  progress=lambda m: None, play_fn=recording)
  assert rep["matches_played"] == 6 * 3 * 2
  keys = {key for _, key, _ in events}
  assert len(keys) == 6 * 3
  for key in keys:
    orig_finish = events.index(("finish", key, False))
    dup_start = events.index(("start", key, True))
    assert orig_finish < dup_start, key


def test_heuristic_anchors_ratings(tmp_path):
  models = [ModelConfig(kind="heuristic"), ModelConfig(kind="random", label="r1"),
            ModelConfig(kind="random", label="r2")]
  rep = tournament.run_tournament(models, seeds=[0, 1, 2], out_root=str(tmp_path),
                                  duplicate=True, progress=lambda m: None)
  assert rep["ratings"]["anchor"] == "heuristic"
  elo = {r["model"]: r["bt_elo"] for r in rep["standings"]}
  assert elo["heuristic"] == 0
  assert elo["r1"] < 0 and elo["r2"] < 0
  assert rep["standings"][0]["model"] == "heuristic"


def test_tournament_survives_failure(tmp_path):
  def flaky(spec):
    if spec.match_id == "seed0_dup":
      raise RuntimeError("boom")
    from runner.match_runner import play_match
    return play_match(spec)
  rep = tournament.run_tournament(BOTS[:2], seeds=[0], out_root=str(tmp_path),
                                  progress=lambda m: None, play_fn=flaky)
  assert rep["matches_played"] == 1
  assert rep["failures"] == [{"pairing": "bot0__vs__bot1", "match_id": "seed0_dup", "error": "RuntimeError: boom"}]


def test_cli_tournament(tmp_path, capsys):
  mf = tmp_path / "models.json"
  mf.write_text(json.dumps([{"slug": "random", "label": "r1"}, {"slug": "random", "label": "r2"},
                            {"slug": "random", "label": "r3"}]))
  rc = cli.main(["tournament", "--models-file", str(mf), "--seeds", "1", "--out", str(tmp_path / "rr")])
  assert rc == 0
  out = capsys.readouterr().out
  assert "Standings" in out and "Head-to-head" in out
  assert os.path.exists(tmp_path / "rr" / "r1__vs__r2" / "seed0_dup" / "summary.json")


def test_exclude_pairing(tmp_path):
  specs = tournament.tournament_specs(BOTS, seeds=[0], duplicate=False, out_root=str(tmp_path),
                                      exclude=[("bot1", "bot0")])
  assert [os.path.basename(os.path.dirname(s.out_dir)) for s in specs] == ["bot0__vs__bot2", "bot1__vs__bot2"]


def test_cli_exclude_and_preflight(tmp_path, capsys, monkeypatch):
  mf = tmp_path / "models.json"
  mf.write_text(json.dumps([{"slug": "random", "label": "r1"}, {"slug": "random", "label": "r2"},
                            {"slug": "random", "label": "r3"}]))
  rc = cli.main(["tournament", "--models-file", str(mf), "--seeds", "1", "--no-duplicate",
                 "--exclude", "r1/r2", "--out", str(tmp_path / "rr")])
  assert rc == 0
  assert not os.path.exists(tmp_path / "rr" / "r1__vs__r2")
  assert os.path.exists(tmp_path / "rr" / "r1__vs__r3" / "seed0_orig" / "summary.json")
  # Preflight failure aborts before any match is played.
  monkeypatch.setattr(tournament, "preflight", lambda models: {"r2": "DoNotRetryError: HTTP 403"})
  rc = cli.main(["tournament", "--models-file", str(mf), "--seeds", "1", "--out", str(tmp_path / "rr2")])
  assert rc == 2 and not os.path.exists(tmp_path / "rr2" / "r1__vs__r2")
  assert "PREFLIGHT FAILED for r2" in capsys.readouterr().out
