"""Kaggle Benchmarks helpers: slug resolution and the tournament budget guard."""

import pytest

from kbench_model import resolve_slug
from runner import kbench_task
from runner import tournament
from runner.config import KIND_RANDOM, ModelConfig

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
  budget = kbench_task.Budget(1.0, initial_estimate_usd=0.3)
  budget._in_flight = 3  # three matches running at the estimated $0.30 each
  with pytest.raises(kbench_task.BudgetExhausted):
    budget.play_fn(None)
  budget._in_flight = 2
  budget._play = lambda spec: {"team_stats": {"A": {"cost_usd": 0.2}, "B": {"cost_usd": 0.1}}}
  budget.play_fn(None)
  assert (budget.spent_usd, budget.finished, budget._in_flight) == (pytest.approx(0.3), 1, 2)


def test_quick_preflight_skips_non_proxy_models():
  assert kbench_task.quick_preflight([ModelConfig(kind=KIND_RANDOM, label="r")]) == {}


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
