# %% [markdown]
# # Truco round-robin (pilot)
#
# Plays a Truco Paulista round-robin between Model Proxy models with the
# truco-eval runner: every pairing, each seed plus its duplicate (same deals,
# teams swapped across seats), 4 players, first to 12. The model the task is
# run with is not used; the contestants are listed in `MODELS`.
#
# Needs the `truco-eval-wheels` dataset attached (truco-eval + game_arena
# wheels). Per-match transcripts, replays and `tournament.json` are written
# under `truco_runs/` and come back with `kaggle b t download`.
#
# Local zero-cost check: `TRUCO_DRY_RUN=1 python kaggle_task/truco_pilot.py`
# (random bots in place of the models).

# %%
import glob
import os
import subprocess
import sys


def _ensure_truco() -> None:
  try:
    import runner.tournament  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import
    return
  except ImportError:
    pass
  wheels = sorted(glob.glob("/kaggle/input/**/*.whl", recursive=True))
  if not wheels:
    raise RuntimeError("truco-eval is not installed and no wheels were found under "
                       "/kaggle/input: attach the truco-eval-wheels dataset.")
  subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *wheels])


_ensure_truco()

# %%
import kaggle_benchmarks as kbench
from runner import tournament
from runner.config import KIND_KBENCH, KIND_RANDOM, ModelConfig

DRY_RUN = os.environ.get("TRUCO_DRY_RUN") == "1"
MODELS = [
    ModelConfig(kind=KIND_KBENCH, slug="google/gemini-3-flash-preview", label="gemini-3-flash"),
    ModelConfig(kind=KIND_KBENCH, slug="qwen/qwen3-next-80b-a3b-instruct", label="qwen3-next-80b"),
]
if DRY_RUN:
  MODELS = [ModelConfig(kind=KIND_RANDOM, label=m.display) for m in MODELS]
SEEDS = [1]
PARALLEL = 4
OUT_ROOT = "truco_runs"


# %%
@kbench.task(name="truco-round-robin-pilot",
             description="Truco Paulista round-robin between Model Proxy models (pilot).")
def truco_round_robin_pilot(llm) -> dict:  # pylint: disable=unused-argument
  errors = tournament.preflight(MODELS)
  kbench.assertions.assert_empty(errors, expectation="Every model answers a preflight request")
  if errors:
    return {"preflight_errors": errors}
  report = tournament.run_tournament(MODELS, seeds=SEEDS, out_root=OUT_ROOT,
                                     duplicate=True, parallel=PARALLEL)
  print(tournament.format_report(report))
  kbench.assertions.assert_empty(report["failures"], expectation="Every match completes")
  return {
      "matches_played": report["matches_played"],
      "standings": report["standings"],
      "head_to_head": report["head_to_head"],
      "failures": report["failures"],
  }


truco_round_robin_pilot.run(kbench.llm)
