"""Helpers for running a tournament inside a Kaggle Benchmarks task.

Two guards that matter on the Model Proxy's free quota:

* ``quick_preflight``: the harness retries a failed request for up to 60
  minutes, so a model that is overloaded (HTTP 429 "heavy load") would stall
  a whole task run. This probes each model with a short retry budget instead,
  so the task can drop it from the roster and play the rest.
* ``Budget``: only starts a match if the reported spend plus the expected
  cost of every match in flight (including this one) stays under the cap, so
  a run can use up a quota without the proxy cutting matches off midway.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
from typing import Any, Callable, Sequence

from runner import match_runner
from runner.config import KIND_KBENCH, MatchSpec, ModelConfig

_PROBE = "Reply with exactly: Final Answer: FOLD"


class BudgetExhausted(RuntimeError):
  pass


def quick_preflight(models: Sequence[ModelConfig], *, attempts: int = 4,
                    wait_secs: float = 20.0) -> dict[str, str]:
  """One request per Model Proxy model, in parallel; returns {label: error}."""
  from game_arena.harness import model_generation  # pylint: disable=import-outside-toplevel

  def probe(m: ModelConfig) -> str | None:
    try:
      model = m.build_agent()._sampler._model  # pylint: disable=protected-access
    except Exception as e:  # pylint: disable=broad-exception-caught
      return f"{type(e).__name__}: {str(e)[:3000]}"  # long: lists the available models
    last: Exception | None = None
    for attempt in range(attempts):
      try:
        model._generate(_PROBE, None)  # pylint: disable=protected-access  (undecorated: no harness retry)
        return None
      except model_generation.DoNotRetryError as e:
        return f"{type(e).__name__}: {str(e)[:300]}"
      except Exception as e:  # pylint: disable=broad-exception-caught
        last = e
        if attempt + 1 < attempts:
          time.sleep(wait_secs)
    return f"{type(last).__name__}: {str(last)[:300]}"

  kbench_models = [m for m in models if m.kind == KIND_KBENCH]
  if not kbench_models:
    return {}
  with concurrent.futures.ThreadPoolExecutor(max_workers=len(kbench_models)) as ex:
    results = list(ex.map(probe, kbench_models))
  return {m.display: err for m, err in zip(kbench_models, results) if err is not None}


class Budget:
  """A ``play_fn`` for ``run_tournament`` that keeps spend plus in-flight matches under a cap.

  The expected cost of a match is the running mean of finished matches, or
  ``initial_estimate_usd`` before any has finished.
  """

  def __init__(self, cap_usd: float, *, initial_estimate_usd: float = 0.0,
               play: Callable[[MatchSpec], dict[str, Any]] = match_runner.play_match):
    self.cap_usd = cap_usd
    self.spent_usd = 0.0
    self.finished = 0
    self._initial_estimate = initial_estimate_usd
    self._in_flight = 0
    self._play = play
    self._lock = threading.Lock()

  def expected_match_usd(self) -> float:
    return self.spent_usd / self.finished if self.finished else self._initial_estimate

  def play_fn(self, spec: MatchSpec) -> dict[str, Any]:
    with self._lock:
      committed = self.spent_usd + (self._in_flight + 1) * self.expected_match_usd()
      if self.spent_usd >= self.cap_usd or committed > self.cap_usd:
        raise BudgetExhausted(f"budget: ${self.spent_usd:.2f} spent + {self._in_flight} in flight "
                              f"would exceed ${self.cap_usd:.2f}")
      self._in_flight += 1
    try:
      summary = self._play(spec)
    finally:
      with self._lock:
        self._in_flight -= 1
    cost = sum(t.get("cost_usd") or 0.0 for t in summary["team_stats"].values())
    with self._lock:
      self.spent_usd += cost
      self.finished += 1
    return summary
