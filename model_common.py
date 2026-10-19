"""Code shared by the model layers (``openrouter_model``, ``kbench_model``).

Both layers return ``GenerateReturn`` objects whose ``response_for_logging`` is
OpenRouter-shaped: ``provider`` names the backend that served the request and
``usage.cost`` is the cost in USD. The match stats read those two fields
through :func:`served_provider` and :func:`served_cost`, whichever layer made
the request.

The policy both layers apply:

* ``max_tokens`` defaults to :data:`DEFAULT_MAX_TOKENS` (a benchmark default).
* An HTTP 4xx other than 408/429 is never retried (``DoNotRetryError``);
  everything else is left to the harness's retry decorator.
* Reasoning text returned out of band is wrapped in ``<think>`` tags in
  ``main_response_and_thoughts``; without it, inline ``<think>`` tags in the
  reply are split off the main response.

Imports only the harness, so the orchestrator can use it without importing a
model layer's SDK.
"""

from __future__ import annotations

from typing import Any, Mapping

from game_arena.harness import model_generation

# High enough for reasoning models, which spend output tokens before the
# visible answer.
DEFAULT_MAX_TOKENS = 16384
RETRYABLE_4XX = (408, 429)


def max_tokens(model_options: Mapping[str, Any]) -> int:
  """The request's ``max_tokens``: the configured value or the default."""
  return int(model_options.get("max_tokens", DEFAULT_MAX_TOKENS))


def is_non_retryable_status(status: int) -> bool:
  """Whether an HTTP status is a 4xx that can never succeed on retry."""
  return 400 <= status < 500 and status not in RETRYABLE_4XX


def text_content(prompt_text: str) -> list[dict[str, Any]]:
  """Chat content parts for a text-only prompt."""
  return [{"type": "text", "text": prompt_text}]


def split_thoughts(content: str, reasoning: str | None) -> tuple[str, str]:
  """Return ``(main_response, main_response_and_thoughts)`` for a reply."""
  if reasoning:
    return content, f"<think>{reasoning}</think>{content}"
  main_response = content
  separated = model_generation.separate_main_response_and_thoughts(content)
  if separated is not None:
    main_response = separated[0]
  return main_response, content


def served_provider(generate_return: model_generation.GenerateReturn) -> str | None:
  """Return the provider that served a response, when it is recorded."""
  response = generate_return.response_for_logging or {}
  return response.get("provider")


def served_cost(generate_return: model_generation.GenerateReturn) -> float | None:
  """Return the recorded response cost in USD, when present."""
  response = generate_return.response_for_logging or {}
  usage = response.get("usage") or {}
  cost = usage.get("cost")
  return float(cost) if cost is not None else None
