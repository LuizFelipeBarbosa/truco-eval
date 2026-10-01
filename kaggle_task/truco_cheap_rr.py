# %% [markdown]
# # Truco round-robin: cheap models
#
# Truco Paulista round-robin between low-cost Model Proxy models with the
# truco-eval runner: every pairing, each seed plus its duplicate (same deals,
# teams swapped across seats), 4 players, first to 12. The model the task is
# run with is not used; the contestants are listed in `MODELS`.
#
# Models that fail a short preflight (e.g. HTTP 429 "heavy load") are dropped
# and listed in the result. A match only starts if spend plus the expected cost
# of the matches in flight stays under `BUDGET_USD`.
#
# Needs the `truco-eval-wheels` dataset attached. Transcripts, replays and
# `tournament.json` are written under `truco_runs/`.
#
# Local zero-cost check: `TRUCO_DRY_RUN=1 python kaggle_task/truco_cheap_rr.py`

# %%
import glob
import os
import subprocess
import sys


def _ensure_truco() -> None:
  try:
    import runner.kbench_task  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import
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
from runner import kbench_task
from runner import tournament
from runner.config import KIND_KBENCH, KIND_RANDOM, ModelConfig

DRY_RUN = os.environ.get("TRUCO_DRY_RUN") == "1"
MODELS = [
    ModelConfig(kind=KIND_KBENCH, slug="qwen3-next-80b-a3b-instruct", label="qwen3-next-80b"),
    ModelConfig(kind=KIND_KBENCH, slug="gemini-3.1-flash-lite-preview", label="gemini-3.1-flash-lite"),
    ModelConfig(kind=KIND_KBENCH, slug="gemini-3.5-flash-lite", label="gemini-3.5-flash-lite"),
    ModelConfig(kind=KIND_KBENCH, slug="gpt-5.4-nano-2026-03-17", label="gpt-5.4-nano"),
    ModelConfig(kind=KIND_KBENCH, slug="gpt-oss-20b", label="gpt-oss-20b"),
]
if DRY_RUN:
  MODELS = [ModelConfig(kind=KIND_RANDOM, label=m.display) for m in MODELS]
# v1 played seeds 1-2 ($4.27), v2 seeds 3-4 ($4.01), v3 seeds 5-9 plus seed-10
# originals ($9.52, 110 matches). v4 spends a full daily quota ($10): more seeds
# than it can afford, and the budget stops starting matches once spend plus
# in-flight matches would pass BUDGET_USD. The $0.40 headroom covers ~2 sd of
# 12 in-flight matches ($0.044 sd per match) plus the preflight.
SEEDS = list(range(11, 18))
PARALLEL = 12
BUDGET_USD = 9.60
EXPECTED_MATCH_USD = 0.10  # v3 averaged $0.087
OUT_ROOT = "truco_runs"


# %%
@kbench.task(name="truco-cheap-round-robin",
             description="Truco Paulista round-robin between low-cost Model Proxy models.")
def truco_cheap_round_robin(llm) -> dict:  # pylint: disable=unused-argument
  dropped = kbench_task.quick_preflight(MODELS)
  roster = [m for m in MODELS if m.display not in dropped]
  print(f"Roster: {[m.display for m in roster]}; dropped: {dropped}")
  kbench.assertions.assert_true(len(roster) >= 2,
                                expectation="At least two models answer the preflight")
  if len(roster) < 2:
    return {"dropped": dropped}
  budget = kbench_task.Budget(BUDGET_USD, initial_estimate_usd=EXPECTED_MATCH_USD)
  report = tournament.run_tournament(roster, seeds=SEEDS, out_root=OUT_ROOT, duplicate=True,
                                     parallel=PARALLEL, play_fn=budget.play_fn)
  print(tournament.format_report(report))
  print(f"Spent ${budget.spent_usd:.2f} of ${BUDGET_USD:.2f}")
  failures = [f for f in report["failures"] if not f["error"].startswith("BudgetExhausted")]
  kbench.assertions.assert_empty(failures, expectation="Every match the budget allowed completes")
  return {
      "dropped": dropped,
      "spent_usd": round(budget.spent_usd, 4),
      "matches_played": report["matches_played"],
      "standings": report["standings"],
      "head_to_head": report["head_to_head"],
      "skipped_for_budget": len(report["failures"]) - len(failures),
      "failures": failures,
  }


truco_cheap_round_robin.run(kbench.llm)
