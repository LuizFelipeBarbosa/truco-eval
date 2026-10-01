import json
import os

from runner import cli
from runner import stats as truco_stats


def test_cli_run_random_bots_and_replay(tmp_path, capsys):
  out = str(tmp_path / "run")
  rc = cli.main(["run", "--team-a", "random", "--team-b", "random", "--seeds", "2",
                 "--out", out])
  assert rc == 0
  assert sorted(os.listdir(out)) == ["aggregate.json", "seed0_dup", "seed0_orig",
                                     "seed1_dup", "seed1_orig"]
  with open(os.path.join(out, "aggregate.json"), encoding="utf-8") as f:
    agg = json.load(f)
  assert agg["aggregate"]["matches"] == 4
  assert set(agg["aggregate"]["configs"]) == {"random#A", "random#B"}
  c = agg["aggregate"]["configs"]["random#A"]
  assert c["matches"] == 4 and c["match_win_rate"] is not None
  assert c["illegal_action_rate"] is None  # bots make no model decisions
  text = capsys.readouterr().out
  assert "match win rate" in text and "est. cost / match" in text
  rc = cli.main(["replay", os.path.join(out, "seed1_orig", "engine_events.jsonl")])
  assert rc == 0
  assert "byte-identical to the file: YES" in capsys.readouterr().out


def test_cli_no_duplicate_and_parallel(tmp_path):
  out = str(tmp_path / "run")
  rc = cli.main(["run", "--team-a", "random", "--team-b", "random", "--seeds", "3",
                 "--no-duplicate", "--parallel", "3", "--num-players", "2", "--out", out])
  assert rc == 0
  assert sorted(d for d in os.listdir(out) if d.startswith("seed")) == ["seed0_orig", "seed1_orig", "seed2_orig"]


def test_model_config_from_args_openrouter():
  cfg = cli.model_config_from_args("openai/gpt-5-mini", '{"temperature": 0.5}',
                                   '{"order": ["OpenAI"], "allow_fallbacks": false}',
                                   None, None, 1, None)
  assert cfg.kind == "openrouter" and cfg.slug == "openai/gpt-5-mini"
  assert cfg.model_options == {"temperature": 0.5}
  assert cfg.provider == {"order": ["OpenAI"], "allow_fallbacks": False}
  assert cfg.display == "openai/gpt-5-mini"


def test_format_table_handles_missing_values():
  agg = truco_stats.aggregate([])
  assert agg["matches"] == 0
  assert truco_stats.format_table(agg)


def test_cli_survives_a_failing_match(tmp_path, monkeypatch, capsys):
  from runner import cli as cli_mod
  real = cli_mod.play_match
  def flaky(spec, **kw):
    if spec.match_id == "seed1_orig":
      raise RuntimeError("provider down")
    return real(spec, **kw)
  monkeypatch.setattr(cli_mod, "play_match", flaky)
  out = str(tmp_path / "run")
  rc = cli_mod.main(["run", "--team-a", "random", "--team-b", "random", "--seeds", "2",
                     "--no-duplicate", "--out", out])
  assert rc == 1
  with open(os.path.join(out, "aggregate.json"), encoding="utf-8") as f:
    agg = json.load(f)
  assert agg["aggregate"]["matches"] == 1
  assert agg["failures"] == [{"match_id": "seed1_orig", "error": "RuntimeError: provider down"}]
  assert "seed1_orig: FAILED" in capsys.readouterr().out
