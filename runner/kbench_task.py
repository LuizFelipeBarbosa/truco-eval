"""Helpers for running a tournament inside a Kaggle Benchmarks task.

Two guards that matter on the Model Proxy's free quota:

* ``quick_preflight``: the harness retries a failed request for up to 60
  minutes, so a model that is overloaded (HTTP 429 "heavy load") would stall
  a whole task run. This probes each model with a short retry budget instead,
  so the task can drop it from the roster and play the rest.
* ``Budget``: reserves each original/duplicate pair's expected cost, including
  matches in flight and pending duplicates in admission checks. A reserved
  duplicate starts even over the cap, unless its original has already failed;
  a duplicate whose original was refused is refused too. Failed matches are
  charged at least the expected match cost (more if they report higher partial
  spend), without changing the mean cost of completed matches.
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
  """Admit matches against a cap, reserving duplicates by default.

  The expected cost is the running mean of completed matches, or
  ``initial_estimate_usd`` before any has completed. Failed matches are
  charged at least the expected match cost, because requests inside a failing
  decision may never be tallied; that charge does not change the estimate.
  When paired, a duplicate is refused if its original failed or was refused
  here; a duplicate whose original this budget never saw (e.g. on resume) gets
  the normal one-match check. ``paired=False`` admits each match independently.
  """

  def __init__(self, cap_usd: float, *, initial_estimate_usd: float = 0.0,
               paired: bool = True,
               play: Callable[[MatchSpec], dict[str, Any]] = match_runner.play_match):
    self.cap_usd = cap_usd
    self.spent_usd = 0.0
    self.finished = 0
    self.failed = 0
    self._completed_spend_usd = 0.0
    self._initial_estimate = initial_estimate_usd
    self._paired = paired
    self._in_flight = 0
    self._reserved: set[tuple[str, str, int]] = set()
    self._orig_failed: set[tuple[str, str, int]] = set()
    self._blocked: set[tuple[str, str, int]] = set()
    self._play = play
    self._lock = threading.Lock()

  def expected_match_usd(self) -> float:
    """Return the mean completed-match cost, or the initial estimate."""
    return (self._completed_spend_usd / self.finished
            if self.finished else self._initial_estimate)

  def play_fn(self, spec: MatchSpec) -> dict[str, Any]:
    """Admit and play ``spec``, charging successful and failed matches."""
    key = (spec.team_a.display, spec.team_b.display, spec.seed)
    with self._lock:
      if self._paired and spec.swap and key in self._reserved:
        self._reserved.remove(key)
      elif self._paired and spec.swap and key in self._orig_failed:
        raise BudgetExhausted(
            f"budget: original for {key[0]} vs {key[1]} seed {key[2]} failed; "
            "duplicate is not started")
      elif self._paired and spec.swap and key in self._blocked:
        # A duplicate without its original is excluded from the leaderboard.
        raise BudgetExhausted(
            f"budget: original for {key[0]} vs {key[1]} seed {key[2]} was not started "
            "(budget); duplicate is not started")
      else:
        need = 2 if self._paired and not spec.swap else 1
        committed = self.spent_usd + (
            self._in_flight + len(self._reserved) + need) * self.expected_match_usd()
        if self.spent_usd >= self.cap_usd or committed > self.cap_usd:
          if self._paired and not spec.swap:
            self._blocked.add(key)
          raise BudgetExhausted(
              f"budget: ${self.spent_usd:.2f} spent + {self._in_flight} in flight + "
              f"{len(self._reserved)} reserved + {need} needed would exceed "
              f"${self.cap_usd:.2f}")
        if self._paired and not spec.swap:
          self._reserved.add(key)
          self._blocked.discard(key)
      self._in_flight += 1
    try:
      summary = self._play(spec)
    except Exception as e:  # pylint: disable=broad-exception-caught
      with self._lock:
        # Partial spend under-reports: requests in a failing decide() are not tallied.
        cost = max(getattr(e, "cost_usd_so_far", None) or 0.0, self.expected_match_usd())
        self.spent_usd += cost
        self.failed += 1
        if self._paired and not spec.swap:
          self._reserved.discard(key)
          self._orig_failed.add(key)
      raise
    else:
      cost = sum(t.get("cost_usd") or 0.0 for t in summary["team_stats"].values())
      with self._lock:
        self.spent_usd += cost
        self._completed_spend_usd += cost
        self.finished += 1
      return summary
    finally:
      # Record spend before releasing the estimate for this in-flight match.
      with self._lock:
        self._in_flight -= 1
