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


def _state(*, pending=None, next_seed=11, next_label=4, failures=0, paused=False):
  return {"next_seed": next_seed, "next_label": next_label, "pending": pending,
          "consecutive_failures": failures, "paused": paused, "history": []}


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


def _write_outputs(out, seeds, *, kinds=("orig", "dup"), pairing="a__vs__b", matches=0):
  for seed in seeds:
    for kind in kinds:
      directory = os.path.join(out, "truco_runs", pairing, f"seed{seed}_{kind}")
      os.makedirs(directory, exist_ok=True)
      with open(os.path.join(directory, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({}, f)
  report_dir = os.path.join(out, "truco_runs")
  os.makedirs(report_dir, exist_ok=True)
  with open(os.path.join(report_dir, "tournament.json"), "w", encoding="utf-8") as f:
    json.dump({"matches_played": matches}, f)


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
  state = _state()
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


def test_collect_resumes_at_first_seed_gap(tmp_path, monkeypatch):
  module = _load_daily()
  _patch_paths(module, monkeypatch, tmp_path)
  state = _state(pending=_pending())
  calls = []
  _fake_kaggle(module, calls, monkeypatch,
               output_writer=lambda path: _write_outputs(path, [11, 13], matches=4))
  monkeypatch.setattr(module, "task_status", lambda: (4, "Completed"))
  merged = []
  monkeypatch.setattr(module, "merge_all", lambda: merged.append(True))

  module.collect(state, dry_run=False)

  assert state["next_seed"] == 12
  assert state["consecutive_failures"] == 0
  assert state["history"][0]["seeds_played"] == [11, 13]
  assert merged == [True]


def test_collect_moves_past_a_gap_at_the_runs_first_seed(tmp_path, monkeypatch):
  module = _load_daily()
  _patch_paths(module, monkeypatch, tmp_path)
  state = _state(pending=_pending(seeds=list(range(12, 19))), next_seed=12)
  calls = []
  _fake_kaggle(module, calls, monkeypatch,
               output_writer=lambda path: _write_outputs(path, range(13, 19), matches=12))
  monkeypatch.setattr(module, "task_status", lambda: (4, "Completed"))
  merged = []
  monkeypatch.setattr(module, "merge_all", lambda: merged.append(True))

  module.collect(state, dry_run=False)

  assert state["next_seed"] == 19
  assert state["consecutive_failures"] == 0
  assert merged == [True]


def test_collect_retries_a_later_gap_when_the_runs_first_seed_is_also_missing(tmp_path,
                                                                             monkeypatch):
  module = _load_daily()
  _patch_paths(module, monkeypatch, tmp_path)
  state = _state(pending=_pending(seeds=list(range(12, 19))), next_seed=12)
  calls = []
  _fake_kaggle(module, calls, monkeypatch, output_writer=lambda path: _write_outputs(
      path, [seed for seed in range(12, 19) if seed not in (12, 14)], matches=10))
  monkeypatch.setattr(module, "task_status", lambda: (4, "Completed"))
  monkeypatch.setattr(module, "merge_all", lambda: None)

  module.collect(state, dry_run=False)

  assert state["next_seed"] == 14
  assert state["consecutive_failures"] == 0


def test_collect_retries_a_persistently_failing_seed_once(tmp_path, monkeypatch):
  module = _load_daily()
  _patch_paths(module, monkeypatch, tmp_path)
  monkeypatch.setattr(module, "merge_all", lambda: None)
  calls = []
  # Seed 12 never completes; every other seed does.
  _fake_kaggle(module, calls, monkeypatch, output_writer=lambda path: _write_outputs(
      path, [seed for seed in range(11, 26) if seed != 12], matches=2))
  state = _state()

  launched = []
  for number in (4, 5):
    seeds = list(range(state["next_seed"], state["next_seed"] + module.SEEDS_PER_RUN))
    launched.append(seeds)
    state["pending"] = _pending(label=f"v{number}", seeds=seeds, task_version=number)
    monkeypatch.setattr(module, "task_status", lambda number=number: (number, "Completed"))
    module.collect(state, dry_run=False)

  assert launched == [list(range(11, 18)), list(range(12, 19))]
  assert state["next_seed"] == 19
  assert state["consecutive_failures"] == 0


def test_collect_advances_past_last_seed_when_all_played(tmp_path, monkeypatch):
  module = _load_daily()
  _patch_paths(module, monkeypatch, tmp_path)
  state = _state(pending=_pending())
  calls = []
  _fake_kaggle(module, calls, monkeypatch,
               output_writer=lambda path: _write_outputs(path, range(11, 18), matches=14))
  monkeypatch.setattr(module, "task_status", lambda: (4, "Completed"))
  merged = []
  monkeypatch.setattr(module, "merge_all", lambda: merged.append(True))

  module.collect(state, dry_run=False)

  assert state["next_seed"] == 18
  assert state["consecutive_failures"] == 0
  assert merged == [True]


def test_collect_failures_pause_after_two_orphan_runs(tmp_path, monkeypatch):
  module = _load_daily()
  _patch_paths(module, monkeypatch, tmp_path)
  calls = []
  _fake_kaggle(module, calls, monkeypatch,
               output_writer=lambda path: _write_outputs(path, [11], kinds=("orig",), matches=1))
  monkeypatch.setattr(module, "merge_all", lambda: pytest.fail("failed runs must not merge"))
  state = _state()

  for number in (4, 5):
    state["pending"] = _pending(label=f"v{number}", task_version=number)
    monkeypatch.setattr(module, "task_status", lambda number=number: (number, "Errored"))
    module.collect(state, dry_run=False)

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
