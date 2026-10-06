"""State-machine tests for the unattended Kaggle daily driver."""

from __future__ import annotations

import copy
import fcntl
import importlib.util
import json
import os

import pytest


def _load_daily():
  path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "kaggle_task", "daily.py")
  spec = importlib.util.spec_from_file_location("daily_for_state_test", path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _state(*, pending=None, next_seed=11, next_label=4, failures=0, paused=False,
           retry_queue=None):
  return {"next_seed": next_seed, "next_label": next_label, "pending": pending,
          "consecutive_failures": failures, "paused": paused, "history": [],
          "retry_queue": [] if retry_queue is None else retry_queue}


def _pending(label="v4", seeds=None, task_version=4):
  return {"label": label, "seeds": list(range(11, 18) if seeds is None else seeds),
          "task_version": task_version, "pushed_at": "BEFORE"}


def _patch_paths(module, monkeypatch, tmp_path):
  root = os.fspath(tmp_path)
  state_dir = os.path.join(root, "kaggle_daily")
  monkeypatch.setattr(module, "STATE_DIR", state_dir)
  monkeypatch.setattr(module, "STATE_FILE", os.path.join(state_dir, "state.json"))
  monkeypatch.setattr(module, "RUNS", os.path.join(root, "runs"))
  monkeypatch.setattr(module, "now_iso", lambda: "NOW")


def _quota(module, monkeypatch, *, used=0.0, allowed=10.0, monthly_used=0.0,
           monthly_allowed=100.0):
  monkeypatch.setattr(module, "daily_quota", lambda: {
      "used": used, "allowed": allowed, "refill": "LATER",
      "monthly_used": monthly_used, "monthly_allowed": monthly_allowed,
  })


def _write_outputs(out, seeds, *, kinds=("orig", "dup"), pairing="a__vs__b", pairings=None,
                   matches=0, failures=None, skipped=None, report=True, summaries=True):
  for name in (pairing,) if pairings is None else pairings:
    for seed in seeds:
      for kind in kinds:
        directory = os.path.join(out, "truco_runs", name, f"seed{seed}_{kind}")
        os.makedirs(directory, exist_ok=True)
        if summaries:  # a failed match leaves its directory without a summary
          with open(os.path.join(directory, "summary.json"), "w", encoding="utf-8") as f:
            json.dump({}, f)
  report_dir = os.path.join(out, "truco_runs")
  os.makedirs(report_dir, exist_ok=True)
  if not report:
    return
  report = {"matches_played": matches}
  if failures is not None:
    report["failures"] = failures
  if skipped is not None:
    report["skipped"] = skipped
  with open(os.path.join(report_dir, "tournament.json"), "w", encoding="utf-8") as f:
    json.dump(report, f)


PAIRINGS = ("a__vs__b", "a__vs__c", "b__vs__c")


def _refused(seeds_by_pairing, *, legacy):
  """Budget-refused entries in the legacy ``failures`` shape or the ``skipped`` shape."""
  entries = []
  for pairing, seeds in seeds_by_pairing.items():
    for seed in seeds:
      for kind in ("orig", "dup"):
        entry = {"pairing": pairing, "match_id": f"seed{seed}_{kind}"}
        message = ("budget: $8.07 spent + 11 in flight + 8 reserved + 2 needed would exceed "
                   "$9.60")
        if legacy:
          entry["error"] = f"BudgetExhausted: {message}"
        else:
          entry["reason"] = message
        entries.append(entry)
  return entries


def _write_truncated_run(out, *, complete, partial, refused, legacy):
  """Seeds ``complete`` in every pairing, ``partial`` only in the first, then refusals."""
  entries = _refused(refused, legacy=legacy)
  report = {"failures": entries} if legacy else {"failures": [], "skipped": entries}
  _write_outputs(out, complete, pairings=PAIRINGS, **report)
  _write_outputs(out, partial, pairing=PAIRINGS[0], **report)


def _fake_kaggle(module, calls, monkeypatch, *, output_writer=None, push_error=None):
  def fake(*args, timeout=600):
    calls.append({"args": args, "timeout": timeout})
    if args[:3] == ("b", "t", "push"):
      assert len(args) == 8
      assert args[3] == module.TASK
      assert args[4] == "-f"
      assert args[6:] == ("-d", module.DATASET)
      if push_error is not None:
        raise RuntimeError(push_error)
      return ""
    if args[:2] == ("kernels", "output"):
      assert args[2] == module.KERNEL
      assert args[3] == "-p"
      assert len(args) == 5
      if output_writer is not None:
        output_writer(args[4])
      return ""
    raise AssertionError(f"unexpected kaggle args: {args!r}")

  monkeypatch.setattr(module, "kaggle", fake)


def _push_calls(calls):
  return [call for call in calls if call["args"][:3] == ("b", "t", "push")]


def test_launch_skips_when_daily_quota_is_still_used(tmp_path, monkeypatch):
  module = _load_daily()
  _patch_paths(module, monkeypatch, tmp_path)
  _quota(module, monkeypatch, used=module.REFILLED_BELOW_USD)
  state = _state()
  before = copy.deepcopy(state)
  calls = []
  _fake_kaggle(module, calls, monkeypatch)
  monkeypatch.setattr(module, "render_task", lambda *_: pytest.fail("should not render"))

  module.launch(state, dry_run=False)

  assert state == before
  assert calls == []


def test_launch_skips_when_monthly_quota_cannot_cover_a_day(tmp_path, monkeypatch):
  module = _load_daily()
  _patch_paths(module, monkeypatch, tmp_path)
  _quota(module, monkeypatch, monthly_used=95.0, monthly_allowed=100.0)
  state = _state()
  before = copy.deepcopy(state)
  calls = []
  _fake_kaggle(module, calls, monkeypatch)
  monkeypatch.setattr(module, "render_task", lambda *_: pytest.fail("should not render"))

  module.launch(state, dry_run=False)

  assert state == before
  assert calls == []


def test_launch_records_pending_run_and_advances_label(tmp_path, monkeypatch):
  module = _load_daily()
  _patch_paths(module, monkeypatch, tmp_path)
  _quota(module, monkeypatch)
  calls = []
  _fake_kaggle(module, calls, monkeypatch)
  statuses = iter([(3, "Completed"), (4, "Running")])
  monkeypatch.setattr(module, "task_status", lambda: next(statuses))
  monkeypatch.setattr(module, "render_task", lambda seeds, label: os.path.join(
      os.fspath(tmp_path), f"{label}.py"))
  monkeypatch.setattr(module.time, "sleep", lambda _: None)
  state = _state()

  module.launch(state, dry_run=False)

  assert state["pending"] == {"label": "v4", "seeds": list(range(11, 18)),
                               "task_version": 4, "pushed_at": "NOW"}
  assert state["next_label"] == 5
  assert len(_push_calls(calls)) == 1


def test_launch_takes_queued_seeds_first_without_mutating_the_queue(tmp_path, monkeypatch):
  module = _load_daily()
  _patch_paths(module, monkeypatch, tmp_path)
  _quota(module, monkeypatch)
  calls = []
  _fake_kaggle(module, calls, monkeypatch)
  statuses = iter([(6, "Completed"), (7, "Running")])
  monkeypatch.setattr(module, "task_status", lambda: next(statuses))
  rendered = []
  monkeypatch.setattr(module, "render_task", lambda seeds, label: rendered.append(seeds) or "t.py")
  monkeypatch.setattr(module.time, "sleep", lambda _: None)
  queue = [{"seed": 20, "attempts": 1}, {"seed": 25, "attempts": 0}]
  state = _state(next_seed=26, next_label=7, retry_queue=copy.deepcopy(queue))

  module.launch(state, dry_run=False)

  assert rendered == [[20, 25, 26, 27, 28, 29, 30]]
  assert state["pending"]["seeds"] == [20, 25, 26, 27, 28, 29, 30]
  assert state["retry_queue"] == queue
  assert state["next_seed"] == 26


def test_launch_keeps_pending_when_push_cli_errors_but_version_appears(tmp_path, monkeypatch):
  module = _load_daily()
  _patch_paths(module, monkeypatch, tmp_path)
  _quota(module, monkeypatch)
  calls = []
  _fake_kaggle(module, calls, monkeypatch, push_error="transient CLI failure")
  statuses = iter([(3, "Completed"), (4, "Running")])
  monkeypatch.setattr(module, "task_status", lambda: next(statuses))
  monkeypatch.setattr(module, "render_task", lambda *_: os.path.join(os.fspath(tmp_path), "task.py"))
  monkeypatch.setattr(module.time, "sleep", lambda _: None)
  state = _state()

  module.launch(state, dry_run=False)

  assert state["pending"] == {"label": "v4", "seeds": list(range(11, 18)),
                               "task_version": 4, "pushed_at": "NOW"}
  assert state["next_label"] == 5
  assert len(_push_calls(calls)) == 1


def test_launch_fails_without_a_new_task_version_and_leaves_state_unchanged(tmp_path, monkeypatch):
  module = _load_daily()
  _patch_paths(module, monkeypatch, tmp_path)
  _quota(module, monkeypatch)
  calls = []
  _fake_kaggle(module, calls, monkeypatch, push_error="push failed")
  status_calls = []

  def status():
    status_calls.append(None)
    return (3, "Completed") if len(status_calls) == 1 else (3, "Running")

  monkeypatch.setattr(module, "task_status", status)
  monkeypatch.setattr(module, "render_task", lambda *_: os.path.join(os.fspath(tmp_path), "task.py"))
  monkeypatch.setattr(module.time, "sleep", lambda _: None)
  state = _state(retry_queue=[{"seed": 9, "attempts": 1}])
  before = copy.deepcopy(state)

  with pytest.raises(RuntimeError, match="push did not create task version 4"):
    module.launch(state, dry_run=False)

  assert state == before
  assert len(status_calls) == 19
  assert len(_push_calls(calls)) == 1


def test_collect_waits_for_matching_terminal_task(tmp_path, monkeypatch):
  for response in ((99, "Completed"), (4, "Running")):
    module = _load_daily()
    _patch_paths(module, monkeypatch, tmp_path)
    state = _state(pending=_pending())
    before = copy.deepcopy(state)
    out = module.run_dir("v4")
    os.makedirs(out, exist_ok=True)
    marker = os.path.join(out, "stale.txt")
    with open(marker, "w", encoding="utf-8") as f:
      f.write("keep")
    calls = []
    _fake_kaggle(module, calls, monkeypatch)
    monkeypatch.setattr(module, "task_status", lambda response=response: response)

    module.collect(state, dry_run=False)

    assert state == before
    assert os.path.exists(marker)
    assert calls == []


def test_collect_records_complete_pairs_and_merges(tmp_path, monkeypatch):
  module = _load_daily()
  _patch_paths(module, monkeypatch, tmp_path)
  pending = _pending(seeds=[11, 12, 13])
  state = _state(pending=pending)
  out = module.run_dir("v4")
  os.makedirs(out, exist_ok=True)
  stale = os.path.join(out, "stale.txt")
  with open(stale, "w", encoding="utf-8") as f:
    f.write("remove")
  calls = []
  _fake_kaggle(module, calls, monkeypatch,
               output_writer=lambda path: _write_outputs(path, [11, 12, 13], matches=6))
  monkeypatch.setattr(module, "task_status", lambda: (4, "Completed"))
  merged = []
  monkeypatch.setattr(module, "merge_all", lambda: merged.append(True))

  module.collect(state, dry_run=False)

  assert state["pending"] is None
  assert state["next_seed"] == 14
  assert state["consecutive_failures"] == 0
  assert state["history"] == [{**pending, "status": "Completed", "matches": 6,
                               "seeds_played": [11, 12, 13], "collected_at": "NOW"}]
  assert merged == [True]
  assert not os.path.exists(stale)
  assert len([call for call in calls if call["args"][:2] == ("kernels", "output")]) == 1


def _collect(module, monkeypatch, tmp_path, writer, *, pending=None, **state_kwargs):
  _patch_paths(module, monkeypatch, tmp_path)
  state = _state(pending=pending or _pending(), **state_kwargs)
  calls = []
  _fake_kaggle(module, calls, monkeypatch, output_writer=writer)
  monkeypatch.setattr(module, "task_status", lambda: (state["pending"]["task_version"],
                                                      "Completed"))
  merged = []
  monkeypatch.setattr(module, "merge_all", lambda: merged.append(True))
  module.collect(state, dry_run=False)
  assert state["pending"] is None
  assert state["consecutive_failures"] == 0
  assert merged == [True]
  return state


def _queue(*items):
  return [{"seed": seed, "attempts": attempts} for seed, attempts in items]


def test_collect_requeues_untouched_seeds_free_including_interior_ones(tmp_path, monkeypatch):
  module = _load_daily()
  # Parallel admission refused seed 13 (no directory, refusal in ``skipped``) and admitted
  # later seeds; 16 and 17 were refused at the tail.
  skipped = _refused({"a__vs__b": [13, 16, 17]}, legacy=False)
  state = _collect(module, monkeypatch, tmp_path, lambda path: _write_outputs(
      path, [11, 12, 14, 15], failures=[], skipped=skipped, matches=8))

  assert state["retry_queue"] == _queue((13, 0), (16, 0), (17, 0))
  assert state["next_seed"] == 18
  assert state["history"][0]["seeds_played"] == [11, 12, 14, 15]


def test_collect_retries_an_incomplete_first_seed(tmp_path, monkeypatch):
  module = _load_daily()
  failures = [{"pairing": "a__vs__b", "match_id": "seed12_orig", "error": "RuntimeError: boom"}]

  def writer(path):
    _write_outputs(path, range(13, 19), failures=failures, skipped=[], matches=12)
    _write_outputs(path, [12], kinds=("orig",), summaries=False, failures=failures, skipped=[],
                   matches=12)

  state = _collect(module, monkeypatch, tmp_path, writer,
                   pending=_pending(seeds=list(range(12, 19))), next_seed=12)

  assert state["retry_queue"] == _queue((12, 1))
  assert state["next_seed"] == 19


def test_collect_drops_a_seed_whose_started_matches_always_fail(tmp_path, monkeypatch, capsys):
  module = _load_daily()
  _patch_paths(module, monkeypatch, tmp_path)
  monkeypatch.setattr(module, "merge_all", lambda: None)
  calls = []
  # Seed 12's matches start and fail in every run (directories, no summaries); the rest complete.
  failures = [{"pairing": "a__vs__b", "match_id": f"seed12_{kind}", "error": "RuntimeError: boom"}
              for kind in ("orig", "dup")]

  def writer(path):
    _write_outputs(path, [seed for seed in range(11, 25) if seed != 12], failures=failures,
                   skipped=[], matches=2)
    _write_outputs(path, [12], summaries=False, failures=failures, skipped=[], matches=2)

  _fake_kaggle(module, calls, monkeypatch, output_writer=writer)
  state = _state()

  launched = []
  for number in (4, 5):
    retries = [item["seed"] for item in state["retry_queue"]]
    seeds = retries + list(range(state["next_seed"],
                                 state["next_seed"] + module.SEEDS_PER_RUN - len(retries)))
    launched.append(seeds)
    state["pending"] = _pending(label=f"v{number}", seeds=seeds, task_version=number)
    monkeypatch.setattr(module, "task_status", lambda number=number: (number, "Completed"))
    module.collect(state, dry_run=False)
    if number == 4:
      assert state["retry_queue"] == _queue((12, 1))

  assert launched == [list(range(11, 18)), [12, *range(18, 24)]]
  assert state["retry_queue"] == []
  assert state["next_seed"] == 24
  assert state["consecutive_failures"] == 0
  assert "WARNING: v5: seed 12 is still incomplete after 2 attempts; dropped" in (
      capsys.readouterr().out)


def test_collect_charges_every_incomplete_seed_without_a_report(tmp_path, monkeypatch):
  module = _load_daily()
  # The kernel crashed or timed out before writing tournament.json.
  state = _collect(module, monkeypatch, tmp_path,
                   lambda path: _write_outputs(path, [11, 12, 13], report=False),
                   retry_queue=_queue((5, 1)), next_seed=11)

  assert state["history"][0]["matches"] == 0
  assert state["retry_queue"] == _queue((5, 1), (14, 1), (15, 1), (16, 1), (17, 1))
  assert state["next_seed"] == 18


def test_collect_advances_past_last_seed_when_all_played(tmp_path, monkeypatch):
  module = _load_daily()
  state = _collect(module, monkeypatch, tmp_path,
                   lambda path: _write_outputs(path, range(11, 18), matches=14))

  assert state["next_seed"] == 18
  assert state["retry_queue"] == []


def test_collect_retried_seeds_leave_the_queue_and_keep_next_seed(tmp_path, monkeypatch):
  module = _load_daily()
  pending = _pending(seeds=[20, 25, 26, 27, 28, 29, 30])
  state = _collect(module, monkeypatch, tmp_path,
                   lambda path: _write_outputs(path, [20, 25, 26, 27, 28, 29], matches=12),
                   pending=pending, next_seed=26, retry_queue=_queue((20, 1), (25, 1), (40, 0)))

  # 30 was refused untouched: requeued free behind the still-queued 40.
  assert state["retry_queue"] == _queue((30, 0), (40, 0))
  assert state["next_seed"] == 31


def test_collect_requeues_seed_cut_short_by_budget_in_legacy_failures(tmp_path, monkeypatch):
  module = _load_daily()
  refused = {PAIRINGS[0]: [26, 27], PAIRINGS[1]: [25, 26, 27], PAIRINGS[2]: [25, 26, 27]}
  state = _collect(module, monkeypatch, tmp_path, lambda path: _write_truncated_run(
      path, complete=range(21, 25), partial=[25], refused=refused, legacy=True),
      pending=_pending(seeds=list(range(21, 28))), next_seed=21)

  assert state["history"][0]["seeds_played"] == [21, 22, 23, 24, 25]
  assert state["retry_queue"] == _queue((25, 1), (26, 0), (27, 0))
  assert state["next_seed"] == 28


def test_collect_requeues_seed_cut_short_by_budget_in_skipped(tmp_path, monkeypatch):
  module = _load_daily()
  refused = {PAIRINGS[0]: [26, 27], PAIRINGS[1]: [25, 26, 27], PAIRINGS[2]: [25, 26, 27]}
  state = _collect(module, monkeypatch, tmp_path, lambda path: _write_truncated_run(
      path, complete=range(21, 25), partial=[25], refused=refused, legacy=False),
      pending=_pending(seeds=list(range(21, 28))), next_seed=21)

  assert state["retry_queue"] == _queue((25, 1), (26, 0), (27, 0))
  assert state["next_seed"] == 28


def test_collect_retries_budget_cut_at_runs_first_seed(tmp_path, monkeypatch):
  module = _load_daily()
  later = list(range(22, 28))
  refused = {PAIRINGS[0]: later, PAIRINGS[1]: [21, *later], PAIRINGS[2]: [21, *later]}
  state = _collect(module, monkeypatch, tmp_path, lambda path: _write_truncated_run(
      path, complete=[], partial=[21], refused=refused, legacy=False),
      pending=_pending(seeds=list(range(21, 28))), next_seed=21)

  assert state["retry_queue"] == _queue((21, 1), *((seed, 0) for seed in later))
  assert state["next_seed"] == 28


def test_collect_does_not_retry_seed_with_non_budget_failure(tmp_path, monkeypatch):
  module = _load_daily()
  # Partial model failures stay excluded, as before: only seeds the cap cut short or that have
  # no complete pair in any pairing are retried.
  failures = [{"pairing": PAIRINGS[2], "match_id": f"seed23_{kind}", "error": "RuntimeError: boom"}
              for kind in ("orig", "dup")]

  def writer(path):
    _write_outputs(path, [seed for seed in range(21, 28) if seed != 23], pairings=PAIRINGS,
                   failures=failures, skipped=[])
    _write_outputs(path, [23], pairings=PAIRINGS[:2], failures=failures, skipped=[])
    _write_outputs(path, [23], pairing=PAIRINGS[2], summaries=False, failures=failures,
                   skipped=[])

  state = _collect(module, monkeypatch, tmp_path, writer,
                   pending=_pending(seeds=list(range(21, 28))), next_seed=21)

  assert state["retry_queue"] == []
  assert state["next_seed"] == 28


def test_collect_dup_refused_after_its_original_failed_is_not_a_cap_refusal(tmp_path,
                                                                           monkeypatch):
  module = _load_daily()
  failures = [{"pairing": PAIRINGS[1], "match_id": "seed22_orig", "error": "RuntimeError: boom"}]
  skipped = [{"pairing": PAIRINGS[1], "match_id": "seed22_dup",
              "reason": "budget: original for a vs c seed 22 failed; duplicate is not started"}]

  def writer(path):
    _write_outputs(path, [21, 22], pairings=(PAIRINGS[0], PAIRINGS[2]), failures=failures,
                   skipped=skipped)
    _write_outputs(path, [21], pairing=PAIRINGS[1], failures=failures, skipped=skipped)
    _write_outputs(path, [22], kinds=("orig",), pairing=PAIRINGS[1], summaries=False,
                   failures=failures, skipped=skipped)

  state = _collect(module, monkeypatch, tmp_path, writer,
                   pending=_pending(seeds=[21, 22]), next_seed=21)

  assert state["retry_queue"] == []
  assert state["next_seed"] == 23


def _write_legacy_history(module):
  """v5 (17..23) cut short at 20 and v6 (21..27) cut short at 25, as on Kaggle."""
  later = {PAIRINGS[0]: [21, 22, 23], PAIRINGS[1]: [20, 21, 22, 23], PAIRINGS[2]: [20, 21, 22, 23]}
  _write_truncated_run(module.run_dir("v5"), complete=range(17, 20), partial=[20],
                       refused=later, legacy=True)
  later = {PAIRINGS[0]: [26, 27], PAIRINGS[1]: [25, 26, 27], PAIRINGS[2]: [25, 26, 27]}
  _write_truncated_run(module.run_dir("v6"), complete=range(21, 25), partial=[25],
                       refused=later, legacy=True)
  return [{**_pending(label="v5", seeds=list(range(17, 24)), task_version=5),
           "status": "Completed", "seeds_played": [17, 18, 19, 20]},
          {**_pending(label="v6", seeds=list(range(21, 28)), task_version=6),
           "status": "Completed", "seeds_played": [21, 22, 23, 24, 25]}]


def test_migrate_retry_queue_recovers_seeds_cut_short_by_old_checkpoint(tmp_path, monkeypatch,
                                                                        capsys):
  module = _load_daily()
  _patch_paths(module, monkeypatch, tmp_path)
  history = _write_legacy_history(module)
  missing = {**_pending(label="v3", seeds=[1, 2], task_version=3), "status": "Completed"}
  state = _state(next_seed=26, next_label=7)
  state["history"] = [missing, *history]
  del state["retry_queue"]

  assert module.migrate_retry_queue(state, module.RUNS) == [20, 25]
  assert state["retry_queue"] == _queue((20, 1), (25, 1))
  assert "WARNING: migration: no run directory" in capsys.readouterr().out

  # Pending seeds are left to collect.
  state = _state(next_seed=26, next_label=7, pending=_pending(seeds=[25, *range(26, 32)]))
  state["history"] = history
  assert module.migrate_retry_queue(state, module.RUNS) == [20]

  empty = _state()
  del empty["retry_queue"]
  assert module.migrate_retry_queue(empty, module.RUNS) == []
  assert empty["retry_queue"] == []


def test_collect_failures_pause_after_two_orphan_runs(tmp_path, monkeypatch):
  module = _load_daily()
  _patch_paths(module, monkeypatch, tmp_path)
  calls = []
  _fake_kaggle(module, calls, monkeypatch,
               output_writer=lambda path: _write_outputs(path, [11], kinds=("orig",), matches=1))
  monkeypatch.setattr(module, "merge_all", lambda: pytest.fail("failed runs must not merge"))
  queue = [{"seed": 9, "attempts": 1}]
  state = _state(retry_queue=copy.deepcopy(queue))

  for number in (4, 5):
    state["pending"] = _pending(label=f"v{number}", seeds=[9, *range(11, 17)],
                                task_version=number)
    monkeypatch.setattr(module, "task_status", lambda number=number: (number, "Errored"))
    module.collect(state, dry_run=False)

  # A run that is not ok changes neither the queue nor next_seed, so the same window relaunches.
  assert state["retry_queue"] == queue
  assert state["next_seed"] == 11
  assert state["consecutive_failures"] == 2
  assert state["paused"] is True
  assert state["pending"] is None
  assert len(state["history"]) == 2
  assert len([call for call in calls if call["args"][:2] == ("kernels", "output")]) == 2


def test_dry_run_collect_and_launch_do_not_mutate_or_download(tmp_path, monkeypatch):
  module = _load_daily()
  _patch_paths(module, monkeypatch, tmp_path)
  calls = []
  _fake_kaggle(module, calls, monkeypatch)
  monkeypatch.setattr(module, "task_status", lambda: (4, "Completed"))
  out = module.run_dir("v4")
  os.makedirs(out, exist_ok=True)
  marker = os.path.join(out, "stale.txt")
  with open(marker, "w", encoding="utf-8") as f:
    f.write("keep")
  state = _state(pending=_pending())
  before = copy.deepcopy(state)

  module.collect(state, dry_run=True)

  assert state == before
  assert os.path.exists(marker)
  assert calls == []

  _quota(module, monkeypatch)
  launch_state = _state()
  launch_before = copy.deepcopy(launch_state)
  monkeypatch.setattr(module, "render_task", lambda *_: pytest.fail("dry run should not render"))
  module.launch(launch_state, dry_run=True)
  assert launch_state == launch_before
  assert calls == []


def test_main_persists_pending_collects_and_handles_failures(tmp_path, monkeypatch):
  module = _load_daily()
  _patch_paths(module, monkeypatch, tmp_path)
  _quota(module, monkeypatch)
  monkeypatch.setattr(module, "render_task", lambda seeds, label: os.path.join(
      os.fspath(tmp_path), f"{label}.py"))
  monkeypatch.setattr(module.time, "sleep", lambda _: None)
  calls = []
  _fake_kaggle(module, calls, monkeypatch,
               output_writer=lambda path: _write_outputs(path, range(11, 18), matches=14))
  statuses = iter([
      (3, "Completed"), (4, "Running"),  # invocation 1: launch v4
      (4, "Running"),                    # invocation 2: wait for v4
      (4, "Completed"), (4, "Completed"), (5, "Running"),  # invocation 3
  ])
  monkeypatch.setattr(module, "task_status", lambda: next(statuses))
  merged = []
  monkeypatch.setattr(module, "merge_all", lambda: merged.append(True))

  monkeypatch.setattr(module.sys, "argv", ["daily.py"])
  assert module.main() == 0
  with open(module.STATE_FILE, encoding="utf-8") as f:
    first = json.load(f)
  assert first["pending"]["label"] == "v4"
  assert first["pending"]["seeds"] == list(range(11, 18))

  assert module.main() == 0
  with open(module.STATE_FILE, encoding="utf-8") as f:
    second = json.load(f)
  assert second == first

  assert module.main() == 0
  with open(module.STATE_FILE, encoding="utf-8") as f:
    third = json.load(f)
  assert third["pending"]["label"] == "v5"
  assert third["pending"]["seeds"] == list(range(18, 25))
  assert third["next_seed"] == 18
  assert third["history"][0]["seeds_played"] == list(range(11, 18))
  assert merged == [True]
  assert len(_push_calls(calls)) == 2

  third["paused"] = True
  with open(module.STATE_FILE, "w", encoding="utf-8") as f:
    json.dump(third, f)
  calls_before_pause = copy.deepcopy(calls)
  assert module.main() == 0
  assert calls == calls_before_pause

  third["paused"] = False
  third["pending"] = None
  with open(module.STATE_FILE, "w", encoding="utf-8") as f:
    json.dump(third, f)
  saved_before_error = copy.deepcopy(third)
  monkeypatch.setattr(module, "daily_quota", lambda: (_ for _ in ()).throw(RuntimeError("quota down")))
  assert module.main() == 1
  with open(module.STATE_FILE, encoding="utf-8") as f:
    assert json.load(f) == saved_before_error

  os.unlink(module.STATE_FILE)
  _quota(module, monkeypatch)
  monkeypatch.setattr(module.sys, "argv", ["daily.py", "--dry-run"])
  assert module.main() == 0
  assert not os.path.exists(module.STATE_FILE)


def test_main_migrates_once_then_launches_queued_seeds_first(tmp_path, monkeypatch, capsys):
  module = _load_daily()
  _patch_paths(module, monkeypatch, tmp_path)
  _quota(module, monkeypatch)
  monkeypatch.setattr(module, "render_task", lambda seeds, label: os.path.join(
      os.fspath(tmp_path), f"{label}.py"))
  monkeypatch.setattr(module.time, "sleep", lambda _: None)
  calls = []
  _fake_kaggle(module, calls, monkeypatch)
  state = _state(next_seed=26, next_label=7)
  state["history"] = _write_legacy_history(module)
  del state["retry_queue"]
  os.makedirs(module.STATE_DIR, exist_ok=True)
  with open(module.STATE_FILE, "w", encoding="utf-8") as f:
    json.dump(state, f)
  statuses = iter([
      (6, "Completed"), (7, "Running"),  # invocation 1: migrate, launch v7
      (7, "Running"),                    # invocation 2: wait for v7
  ])
  monkeypatch.setattr(module, "task_status", lambda: next(statuses))
  monkeypatch.setattr(module.sys, "argv", ["daily.py"])

  assert module.main() == 0
  with open(module.STATE_FILE, encoding="utf-8") as f:
    first = json.load(f)
  assert first["retry_queue"] == _queue((20, 1), (25, 1))
  assert first["next_seed"] == 26
  assert first["pending"]["seeds"] == [20, 25, 26, 27, 28, 29, 30]

  assert module.main() == 0
  with open(module.STATE_FILE, encoding="utf-8") as f:
    assert json.load(f) == first
  assert capsys.readouterr().out.count("migration: retry queue") == 1
  assert len(_push_calls(calls)) == 1


def test_main_exits_when_another_process_holds_lock(tmp_path, monkeypatch):
  module = _load_daily()
  _patch_paths(module, monkeypatch, tmp_path)
  os.makedirs(module.STATE_DIR, exist_ok=True)
  lock_path = os.path.join(module.STATE_DIR, "daily.lock")
  calls = []
  _fake_kaggle(module, calls, monkeypatch)
  monkeypatch.setattr(module.sys, "argv", ["daily.py"])

  with open(lock_path, "w", encoding="utf-8") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    assert module.main() == 0

  assert calls == []
