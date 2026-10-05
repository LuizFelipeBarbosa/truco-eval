"""Kaggle Benchmarks helpers: slug resolution and the tournament budget guard."""

import random
import runpy
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from kbench_model import resolve_slug
from runner import kbench_task
from runner import tournament
from runner.config import KIND_HEURISTIC, KIND_KBENCH, KIND_RANDOM, ModelConfig

AVAILABLE = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "anthropic/claude-sonnet-5@default"]


@pytest.mark.parametrize("slug, key", [
    ("openai/gpt-oss-20b", "openai/gpt-oss-20b"),
    ("gpt-oss-20b", "openai/gpt-oss-20b"),
    ("claude-sonnet-5", "anthropic/claude-sonnet-5@default"),
    ("claude-sonnet-5-default", "anthropic/claude-sonnet-5@default"),
])
def test_resolve_slug(slug, key):
  assert resolve_slug(slug, AVAILABLE) == key


def test_resolve_slug_unknown():
  with pytest.raises(ValueError, match="not available"):
    resolve_slug("gpt-oss", AVAILABLE)


def test_budget_stops_new_matches_after_cap(tmp_path):
  def fake_play(spec):
    return {"team_stats": {"A": {"cost_usd": 0.4}, "B": {"cost_usd": 0.1}},
            "team_config": {"A": spec.team_a.to_dict(), "B": spec.team_b.to_dict()},
            "winner_config": spec.team_a.display, "scores": {"A": 12, "B": 0},
            "hands_played": 1}

  models = [ModelConfig(kind=KIND_RANDOM, label=l) for l in ("a", "b")]
  budget = kbench_task.Budget(1.0, play=fake_play)
  specs = tournament.tournament_specs(models, seeds=[1, 2, 3], duplicate=True,
                                      out_root=str(tmp_path))
  played, refused = 0, 0
  for spec in specs:
    try:
      budget.play_fn(spec)
      played += 1
    except kbench_task.BudgetExhausted:
      refused += 1
  assert (played, refused) == (2, 4)
  assert budget.spent_usd == pytest.approx(1.0)


def test_budget_counts_matches_in_flight():
  models = [ModelConfig(kind=KIND_RANDOM, label=l) for l in ("a", "b")]
  spec = tournament.tournament_specs(models, seeds=[1], duplicate=True,
                                     out_root="/tmp/truco-budget-test")[0]
  budget = kbench_task.Budget(1.0, initial_estimate_usd=0.3, paired=False)
  budget._in_flight = 3  # three matches running at the estimated $0.30 each
  with pytest.raises(kbench_task.BudgetExhausted):
    budget.play_fn(spec)
  budget._in_flight = 2
  budget._play = lambda spec: {"team_stats": {"A": {"cost_usd": 0.2}, "B": {"cost_usd": 0.1}}}
  budget.play_fn(spec)
  assert (budget.spent_usd, budget.finished, budget._in_flight) == (pytest.approx(0.3), 1, 2)


def _budget_summary(spec, cost=0.3):
  return {
      "match_id": spec.match_id, "seed": spec.seed, "swap": spec.swap,
      "winner_team": "A",
      "team_stats": {"A": {"cost_usd": cost / 2}, "B": {"cost_usd": cost / 2}},
      "team_config": {"A": spec.team_a.to_dict(), "B": spec.team_b.to_dict()},
      "winner_config": spec.team_a.display, "scores": {"A": 12, "B": 0},
      "hands_played": 1,
  }


def _budget_specs(tmp_path, *, seeds=(1,)):
  models = [ModelConfig(kind=KIND_RANDOM, label=l) for l in ("a", "b", "c")]
  return tournament.tournament_specs(models, seeds=seeds, duplicate=True,
                                     out_root=str(tmp_path))


def test_budget_refuses_original_when_only_one_match_is_affordable(tmp_path):
  spec = _budget_specs(tmp_path)[0]
  budget = kbench_task.Budget(1.0, initial_estimate_usd=0.6,
                              play=lambda spec: _budget_summary(spec, 0.6))
  with pytest.raises(kbench_task.BudgetExhausted):
    budget.play_fn(spec)


def test_budget_admits_reserved_duplicate_over_cap(tmp_path):
  orig, dup = _budget_specs(tmp_path)[0:2]
  budget = kbench_task.Budget(1.0, initial_estimate_usd=0.5,
                              play=lambda spec: _budget_summary(spec, 0.7))
  budget.play_fn(orig)
  assert budget.spent_usd == pytest.approx(0.7)
  budget.play_fn(dup)
  assert budget.spent_usd == pytest.approx(1.4)


def test_budget_reservation_counts_against_later_original(tmp_path):
  first_orig, first_dup, second_orig = _budget_specs(tmp_path)[0:3]
  budget = kbench_task.Budget(0.8, initial_estimate_usd=0.3,
                              play=lambda spec: _budget_summary(spec, 0.3))
  budget.play_fn(first_orig)
  with pytest.raises(kbench_task.BudgetExhausted):
    budget.play_fn(second_orig)
  budget.play_fn(first_dup)


def test_budget_charges_failed_matches_and_refuses_failed_duplicate(tmp_path):
  orig, dup = _budget_specs(tmp_path)[0:2]

  def fails_with_cost(spec):
    error = RuntimeError("failed")
    error.cost_usd_so_far = 0.7
    raise error

  budget = kbench_task.Budget(2.0, initial_estimate_usd=0.4, play=fails_with_cost)
  with pytest.raises(RuntimeError) as caught:
    budget.play_fn(orig)
  assert caught.value.cost_usd_so_far == pytest.approx(0.7)
  assert (budget.spent_usd, budget.finished, budget.failed) == (pytest.approx(0.7), 0, 1)
  assert budget.expected_match_usd() == pytest.approx(0.4)
  with pytest.raises(kbench_task.BudgetExhausted, match="original.*failed"):
    budget.play_fn(dup)

  def succeeds(spec):
    return _budget_summary(spec, 0.2)

  def fails_without_cost(spec):
    raise RuntimeError("failed")

  budget = kbench_task.Budget(2.0, paired=False, play=succeeds)
  budget.play_fn(orig)
  budget._play = fails_without_cost
  with pytest.raises(RuntimeError):
    budget.play_fn(dup)
  assert budget.spent_usd == pytest.approx(0.4)
  assert budget.expected_match_usd() == pytest.approx(0.2)

  budget = kbench_task.Budget(2.0, initial_estimate_usd=0.4, play=fails_without_cost)
  with pytest.raises(RuntimeError):
    budget.play_fn(orig)
  assert budget.spent_usd == pytest.approx(0.4)
  assert budget.expected_match_usd() == pytest.approx(0.4)


@pytest.mark.parametrize("cost_usd_so_far", [0.7, None])
def test_budget_failed_match_releases_in_flight_once(tmp_path, cost_usd_so_far):
  spec, following = _budget_specs(tmp_path)[0:2]

  def fails(spec):
    error = RuntimeError("failed")
    if cost_usd_so_far is not None:
      error.cost_usd_so_far = cost_usd_so_far
    raise error

  budget = kbench_task.Budget(1.0, initial_estimate_usd=0.6, paired=False,
                              play=fails)
  with pytest.raises(RuntimeError):
    budget.play_fn(spec)
  assert budget._in_flight == 0
  assert budget.failed == 1

  budget._play = lambda spec: _budget_summary(spec, 0.2)
  with pytest.raises(kbench_task.BudgetExhausted):
    budget.play_fn(following)


def test_budget_refused_original_blocks_its_duplicate(tmp_path):
  orig, dup = _budget_specs(tmp_path)[0:2]
  played = []

  def fake_play(spec):
    played.append(spec.match_id)
    return _budget_summary(spec, 0.6)

  budget = kbench_task.Budget(1.0, initial_estimate_usd=0.6, play=fake_play)
  with pytest.raises(kbench_task.BudgetExhausted):
    budget.play_fn(orig)
  with pytest.raises(kbench_task.BudgetExhausted, match="original.*not started"):
    budget.play_fn(dup)
  assert played == []
  assert (budget.spent_usd, budget.finished, budget.failed) == (0.0, 0, 0)


def test_budget_admits_duplicate_whose_original_it_never_saw(tmp_path):
  dup = _budget_specs(tmp_path)[1]
  budget = kbench_task.Budget(1.0, initial_estimate_usd=0.6,
                              play=lambda spec: _budget_summary(spec, 0.6))
  budget.play_fn(dup)  # resume: the original's summary already exists
  assert budget.finished == 1


@pytest.mark.parametrize("estimate", [0.3, 0.7])
def test_budget_failed_match_charges_at_least_expected_cost(tmp_path, estimate):
  spec = _budget_specs(tmp_path)[0]

  def fails_without_tallied_cost(spec):
    error = RuntimeError("failed")
    error.cost_usd_so_far = 0.0
    raise error

  budget = kbench_task.Budget(2.0, initial_estimate_usd=estimate,
                              play=fails_without_tallied_cost)
  with pytest.raises(RuntimeError):
    budget.play_fn(spec)
  assert budget.spent_usd == pytest.approx(estimate)
  assert budget.expected_match_usd() == pytest.approx(estimate)


def test_budget_paired_false_keeps_one_match_admission(tmp_path):
  orig, dup = _budget_specs(tmp_path)[0:2]
  budget = kbench_task.Budget(1.0, initial_estimate_usd=0.6, paired=False,
                              play=lambda spec: _budget_summary(spec, 0.6))
  budget.play_fn(orig)
  with pytest.raises(kbench_task.BudgetExhausted):
    budget.play_fn(dup)


def test_budget_tournament_cutoff_keeps_played_pairs_complete(tmp_path):
  models = [ModelConfig(kind=KIND_RANDOM, label=l) for l in ("a", "b", "c")]
  played = []

  def fake_play(spec):
    played.append((spec.team_a.display, spec.team_b.display, spec.seed, spec.swap))
    return _budget_summary(spec, 0.5)

  budget = kbench_task.Budget(1.5, initial_estimate_usd=0.5, play=fake_play)
  tournament.run_tournament(models, seeds=[1, 2], out_root=str(tmp_path), parallel=1,
                            progress=lambda m: None, play_fn=budget.play_fn)
  origs = {(a, b, seed) for a, b, seed, swap in played if not swap}
  dups = {(a, b, seed) for a, b, seed, swap in played if swap}
  assert origs <= dups


def test_budget_parallel_tournament_cutoff_leaves_no_orphans(tmp_path):
  models = [ModelConfig(kind=KIND_RANDOM, label=l) for l in ("a", "b", "c")]
  for i in range(5):
    rng = random.Random(i)
    played = []
    lock = threading.Lock()

    def fake_play(spec):
      with lock:
        delay = rng.uniform(0.0, 0.01)
      time.sleep(delay)
      with lock:
        played.append((spec.team_a.display, spec.team_b.display, spec.seed, spec.swap))
      return _budget_summary(spec, 0.5)

    budget = kbench_task.Budget(3.5, initial_estimate_usd=0.5, play=fake_play)
    tournament.run_tournament(models, seeds=[1, 2, 3], out_root=str(tmp_path / str(i)),
                              parallel=4, progress=lambda m: None, play_fn=budget.play_fn)
    origs = {(a, b, seed) for a, b, seed, swap in played if not swap}
    dups = {(a, b, seed) for a, b, seed, swap in played if swap}
    assert 0 < len(played) < 18  # the cap cuts the run part-way
    assert origs == dups


def test_quick_preflight_skips_non_proxy_models():
  assert kbench_task.quick_preflight([ModelConfig(kind=KIND_RANDOM, label="r"),
                                     ModelConfig(kind=KIND_HEURISTIC)]) == {}


@pytest.mark.parametrize("dry_run", [False, True])
def test_cheap_task_keeps_heuristic_and_returns_ratings(monkeypatch, dry_run):
  import kaggle_benchmarks as kbench

  calls = []

  def task(**kwargs):
    def decorate(fn):
      fn.run = lambda llm: calls.append(llm)
      return fn
    return decorate

  monkeypatch.setattr(kbench, "task", task)
  monkeypatch.setenv("TRUCO_DRY_RUN", "1" if dry_run else "0")
  monkeypatch.setattr(kbench_task, "quick_preflight", lambda models: {})
  report = {"ratings": {"anchor": "heuristic"}, "standings": [],
            "head_to_head": {}, "failures": [], "matches_played": 0}
  monkeypatch.setattr(tournament, "run_tournament", lambda *args, **kwargs: report)
  monkeypatch.setattr(tournament, "format_report", lambda report: "test report")
  monkeypatch.setattr(kbench, "assertions", SimpleNamespace(
      assert_true=lambda value, **kwargs: pytest.fail("roster unavailable") if not value else None,
      assert_empty=lambda value, **kwargs: pytest.fail("match failures") if value else None))
  path = Path(__file__).resolve().parents[1] / "kaggle_task" / "truco_cheap_rr.py"
  namespace = runpy.run_path(str(path))
  models = namespace["MODELS"]
  assert len(calls) == 1
  assert len(models) == 6
  assert [(m.kind, m.display) for m in models if m.kind == KIND_HEURISTIC] == [
      (KIND_HEURISTIC, "heuristic")]
  expected = KIND_RANDOM if dry_run else KIND_KBENCH
  assert sum(m.kind == expected for m in models) == 5
  result = namespace["truco_cheap_round_robin"](None)
  assert result["ratings"] is report["ratings"]


class _FailingLLM:
  def __init__(self, error):
    self._error = error

  def respond(self, **kwargs):
    raise self._error


def _kbench_model_raising(error):
  import kaggle_benchmarks as kbench
  from kbench_model import KbenchModel

  model = KbenchModel.__new__(KbenchModel)  # skip the Model Proxy lookup
  model._model_name = "fake/model"
  model._model_options = {}
  model._kbench = kbench
  model._llm = _FailingLLM(error)
  return model


def _status_error(code):
  import httpx
  import openai

  response = httpx.Response(code, request=httpx.Request("POST", "https://proxy.invalid"))
  return openai.APIStatusError("boom", response=response, body=None)


@pytest.mark.parametrize("code, expected", [(401, "DoNotRetryError"), (429, "APIStatusError")])
def test_proxy_errors_propagate_in_kaggle_batch_mode(monkeypatch, code, expected):
  """Kaggle batch runs swallow exceptions inside kbench chat contexts; ours must surface."""
  import concurrent.futures
  from kaggle_benchmarks._config import config

  monkeypatch.setattr(config, "continue_with_exceptions", True)
  model = _kbench_model_raising(_status_error(code))
  with concurrent.futures.ThreadPoolExecutor(1) as ex:  # worker thread = root context
    future = ex.submit(model._generate, "prompt", None)
    with pytest.raises(Exception) as info:
      future.result()
  assert type(info.value).__name__ == expected
