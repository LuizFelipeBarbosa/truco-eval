"""Read OpenRouter-shaped usage fields emitted by both model layers.

The model wrappers expose ``usage.cost`` in USD and ``provider`` through their
``response_for_logging`` dictionaries. These helpers keep match statistics
independent of either model implementation.
"""

from __future__ import annotations

from typing import Any


def served_provider(generate_return: Any) -> str | None:
  """Return the provider that served a response, when it is recorded."""
  response = generate_return.response_for_logging or {}
  return response.get("provider")


def served_cost(generate_return: Any) -> float | None:
  """Return the recorded response cost in USD, when present."""
  response = generate_return.response_for_logging or {}
  usage = response.get("usage") or {}
  cost = usage.get("cost")
  return float(cost) if cost is not None else None
