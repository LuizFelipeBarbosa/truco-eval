"""Daily driver for the cheap-model Kaggle round-robin (run by launchd every 30 min).

The Model Proxy's daily quota is a rolling window: it refills 24 hours after
the previous window's first spend, so the job polls instead of firing at a
fixed hour. Each invocation:

1. If a pushed run has finished, downloads its outputs into
   ``runs/kaggle_cheap_rr_v<N>/`` (Kaggle keeps only the latest version's
   output, so this must happen before the next push), advances the next seed
   and rebuilds the merged leaderboard.
2. If no run is pending and the daily quota has refilled, renders
   ``truco_cheap_rr.py`` with the next seeds and pushes it (a push runs it).

State lives in ``runs/kaggle_daily/state.json``. After two consecutive failed
runs the job pauses itself (``"paused": true``); fix the cause, set it back to
false and the next invocation resumes.

  uv run python kaggle_task/daily.py            # one invocation
  uv run python kaggle_task/daily.py --dry-run  # report what it would do
  uv run python kaggle_task/daily.py --merge    # only rebuild the leaderboard
"""

from __future__ import annotations

import argparse
import collections
import datetime
import fcntl
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Any

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK = "truco-cheap-round-robin"
TASK_FILE = os.path.join(REPO, "kaggle_task", "truco_cheap_rr.py")
DATASET = "lfpmb1/truco-eval-wheels"
KERNEL = "lfpmb1/new-benchmark-task-2cbca"  # backing notebook of TASK (same across versions)
STATE_DIR = os.path.join(REPO, "runs", "kaggle_daily")
STATE_FILE = os.path.join(STATE_DIR, "state.json")
RUNS = os.path.join(REPO, "runs")
SEEDS_PER_RUN = 7  # more than a day's quota buys (~5.5 seeds); the task's budget cuts it off
REFILLED_BELOW_USD = 0.5
MAX_CONSECUTIVE_FAILURES = 2
KAGGLE = os.path.join(os.path.dirname(sys.executable), "kaggle")


def log(msg: str) -> None:
  print(f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M:%S}Z {msg}", flush=True)


def load_state() -> dict[str, Any]:
  if os.path.exists(STATE_FILE):
    with open(STATE_FILE, encoding="utf-8") as f:
      return json.load(f)
  # v1-v3 were run by hand: seeds 1-9 complete, seed-10 originals only.
  return {"next_seed": 11, "next_label": 4, "pending": None, "consecutive_failures": 0,
          "paused": False, "history": []}


def save_state(state: dict[str, Any]) -> None:
  os.makedirs(STATE_DIR, exist_ok=True)
  tmp = STATE_FILE + ".tmp"
  with open(tmp, "w", encoding="utf-8") as f:
    json.dump(state, f, indent=2)
  os.replace(tmp, STATE_FILE)


def kaggle(*args: str, timeout: int = 600) -> str:
  out = subprocess.run([KAGGLE, *args], cwd=REPO, capture_output=True, text=True, timeout=timeout)
  if out.returncode != 0:
    raise RuntimeError(f"kaggle {' '.join(args)} failed ({out.returncode}): "
                       f"{(out.stderr or out.stdout)[-1000:]}")
  return out.stdout


def daily_quota() -> dict[str, Any]:
  from kaggle.api.kaggle_api_extended import KaggleApi  # pylint: disable=import-outside-toplevel

  api = KaggleApi()
  api.authenticate()
  with api.build_kaggle_client() as k:
    balances = k.models.model_proxy_api_client.get_model_proxy_quotas().to_dict()["quotaBalances"]
  daily = next(b for b in balances if b.get("refillPeriod") == "DAILY")
  monthly = next((b for b in balances if b.get("refillPeriod") == "MONTHLY"), {})
  return {"used": float(daily.get("quotaUsed", 0.0)), "allowed": float(daily["totalQuotaAllowed"]),
          "refill": daily.get("refillTime"), "monthly_used": float(monthly.get("quotaUsed", 0.0)),
          "monthly_allowed": float(monthly.get("totalQuotaAllowed", 0.0))}


def task_status() -> tuple[int | None, str | None]:
  text = kaggle("b", "t", "status", TASK)
  version = re.search(r"^Version:\s+(\d+)", text, re.M)
  status = re.search(r"^Status:\s+(\w+)", text, re.M)
  return (int(version.group(1)) if version else None, status.group(1) if status else None)


def render_task(seeds: list[int], label: str) -> str:
  with open(TASK_FILE, encoding="utf-8") as f:
    src = f.read()
  src, n = re.subn(r"^SEEDS = .*$", f"SEEDS = {seeds}  # rendered by kaggle_task/daily.py ({label})",
                   src, count=1, flags=re.M)
  if n != 1:
    raise RuntimeError(f"no 'SEEDS = ...' line in {TASK_FILE}")
  path = os.path.join(STATE_DIR, f"truco_cheap_rr_{label}.py")
  os.makedirs(STATE_DIR, exist_ok=True)
  with open(path, "w", encoding="utf-8") as f:
    f.write(src)
  return path


def run_dir(label: str) -> str:
  return os.path.join(RUNS, f"kaggle_cheap_rr_{label}")


def played_seeds(directory: str) -> set[int]:
  return {int(m.group(1)) for f in glob.glob(os.path.join(directory, "truco_runs", "*", "*", "summary.json"))
          if (m := re.search(r"/seed(\d+)_(?:orig|dup)/summary\.json$", f))}


def collect(state: dict[str, Any], dry_run: bool) -> None:
  pending = state["pending"]
  version, status = task_status()
  if version != pending["task_version"]:
    log(f"pending {pending['label']} expects task version {pending['task_version']}, "
        f"Kaggle shows {version} ({status}); waiting")
    return
  if status not in ("Completed", "Errored", "Failed", "Cancelled"):
    log(f"{pending['label']} (task v{version}) still {status}")
    return
  out = run_dir(pending["label"])
  log(f"{pending['label']} finished ({status}); downloading outputs to {out}")
  if dry_run:
    return
  if os.path.exists(out):
    shutil.rmtree(out)
  os.makedirs(out)
  kaggle("kernels", "output", KERNEL, "-p", out, timeout=1800)
  seeds = played_seeds(out)
  ok = bool(seeds & set(pending["seeds"]))
  report_path = os.path.join(out, "truco_runs", "tournament.json")
  matches = json.load(open(report_path, encoding="utf-8"))["matches_played"] if os.path.exists(report_path) else 0
  state["history"].append({**pending, "status": status, "matches": matches,
                           "seeds_played": sorted(seeds), "collected_at": now_iso()})
  state["pending"] = None
  if ok:
    state["next_seed"] = max(seeds) + 1
    state["consecutive_failures"] = 0
    log(f"{pending['label']}: {matches} matches, seeds {min(seeds)}-{max(seeds)}; next seed {state['next_seed']}")
    merge_all()
  else:
    state["consecutive_failures"] += 1
    log(f"{pending['label']}: no matches for its seeds {pending['seeds']} (status {status}); "
        f"failure {state['consecutive_failures']}/{MAX_CONSECUTIVE_FAILURES}. Kernel log is in {out}")
    if state["consecutive_failures"] >= MAX_CONSECUTIVE_FAILURES:
      state["paused"] = True
      log(f"PAUSED after {MAX_CONSECUTIVE_FAILURES} consecutive failures; set \"paused\": false "
          f"in {STATE_FILE} once fixed")


def launch(state: dict[str, Any], dry_run: bool) -> None:
  quota = daily_quota()
  if quota["used"] >= REFILLED_BELOW_USD:
    log(f"daily quota ${quota['used']:.2f}/${quota['allowed']:.0f} used, refills {quota['refill']}; nothing to do")
    return
  if quota["monthly_allowed"] and quota["monthly_allowed"] - quota["monthly_used"] < quota["allowed"]:
    log(f"monthly quota ${quota['monthly_used']:.2f}/${quota['monthly_allowed']:.0f}: less than a full day "
        f"left; the task's budget cap may exceed it, so not launching")
    return
  label = f"v{state['next_label']}"
  seeds = list(range(state["next_seed"], state["next_seed"] + SEEDS_PER_RUN))
  log(f"daily quota refilled (${quota['used']:.2f} used); launching {label} with seeds {seeds}")
  if dry_run:
    return
  path = render_task(seeds, label)
  before, _ = task_status()
  expected = (before or 0) + 1
  push_error = None
  try:
    kaggle("b", "t", "push", TASK, "-f", path, "-d", DATASET)
  except RuntimeError as e:
    push_error = e
  # Status can lag the push by a few seconds; a run that exists must be
  # recorded as pending even if the CLI errored, or the next invocation would
  # push a second run onto the same quota.
  after, status = before, None
  for _ in range(18):
    after, status = task_status()
    if after is not None and after >= expected:
      break
    time.sleep(10)
  if after is None or after < expected:
    raise RuntimeError(f"push did not create task version {expected} (Kaggle shows {after}): {push_error}")
  state["pending"] = {"label": label, "seeds": seeds, "task_version": after, "pushed_at": now_iso()}
  state["next_label"] += 1
  log(f"pushed {label} as task version {after} ({status})"
      + (f"; the CLI reported {push_error}" if push_error else ""))


def now_iso() -> str:
  return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def merge_all() -> None:
  """Merged leaderboard over every run; later runs win, only complete orig+dup pairs count."""
  from runner import tournament  # pylint: disable=import-outside-toplevel
  from runner.config import ModelConfig  # pylint: disable=import-outside-toplevel

  def order(d: str) -> int:
    m = re.search(r"kaggle_cheap_rr(?:_v(\d+))?$", d)
    return int(m.group(1) or 1) if m else 0

  dirs = sorted((d for d in glob.glob(os.path.join(RUNS, "kaggle_cheap_rr*")) if os.path.isdir(d)), key=order)
  merged: dict[tuple[str, str], dict[str, Any]] = {}
  models = None
  for d in dirs:
    report = os.path.join(d, "truco_runs", "tournament.json")
    if os.path.exists(report):
      models = json.load(open(report, encoding="utf-8"))["models"]
    for f in glob.glob(os.path.join(d, "truco_runs", "*", "*", "summary.json")):
      merged[(f.split("/")[-3], f.split("/")[-2])] = json.load(open(f, encoding="utf-8"))
  halves = collections.defaultdict(set)
  for pairing, match_id in merged:
    seed, half = match_id.split("_")
    halves[(pairing, seed)].add(half)
  balanced = [s for (pairing, match_id), s in merged.items()
              if halves[(pairing, match_id.split("_")[0])] == {"orig", "dup"}]
  configs = [ModelConfig(**{k: v for k, v in m.items() if k != "display"}) for m in models]
  report = tournament.build_report(configs, balanced)
  with open(os.path.join(RUNS, "kaggle_cheap_rr_merged_tournament.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, sort_keys=True, ensure_ascii=False)
  lines = [f"Merged Kaggle cheap round-robin, {now_iso()}: {len(balanced)} matches in complete "
           f"orig+dup pairs ({len(merged) - len(balanced)} unpaired excluded), runs: "
           f"{', '.join(os.path.basename(d) for d in dirs)}", "", tournament.format_report(report), "",
           "Win rate 95% CI (normal approximation):"]
  for r in report["standings"]:
    n, w = r["matches"], r["wins"]
    if n:
      p = w / n
      se = math.sqrt(p * (1 - p) / n)
      lines.append(f"  {r['model']:<22} {w}/{n} = {p:.3f}  [{p - 1.96 * se:.2f}, {p + 1.96 * se:.2f}]")
  with open(os.path.join(RUNS, "kaggle_cheap_rr_leaderboard.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
  log(f"merged leaderboard: {len(balanced)} matches -> runs/kaggle_cheap_rr_leaderboard.txt")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--merge", action="store_true")
  args = parser.parse_args()
  if args.merge:
    merge_all()
    return 0
  os.makedirs(STATE_DIR, exist_ok=True)
  with open(os.path.join(STATE_DIR, "daily.lock"), "w", encoding="utf-8") as lock:
    try:
      fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
      log("another invocation is running; exiting")
      return 0
    state = load_state()
    if state.get("paused"):
      log(f"paused (see {STATE_FILE}); exiting")
      return 0
    try:
      if state["pending"]:
        collect(state, args.dry_run)
      if not state["pending"] and not state.get("paused"):
        launch(state, args.dry_run)
    except Exception as e:  # pylint: disable=broad-exception-caught
      log(f"ERROR {type(e).__name__}: {e}")
      return 1
    finally:
      if not args.dry_run:
        save_state(state)
  return 0


if __name__ == "__main__":
  sys.exit(main())
