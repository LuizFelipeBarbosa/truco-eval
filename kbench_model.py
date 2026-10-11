"""Kaggle Benchmarks model class for the Game Arena harness.

Wraps a ``kaggle_benchmarks`` LLM (``kbench.llms[slug]``, served through the
Kaggle Model Proxy) as a ``MultimodalModel``, so the Truco runner can play
matches inside a Kaggle Benchmarks task exactly as it does through
``OpenRouterModel``. Differences from the OpenRouter path:

* Each request runs in its own orphan ``kbench.chats.new()`` chat, so no
  history leaks between decisions and the run file does not accumulate every
  prompt of the match.
* The proxy reports usage and cost (nanodollars); they are exposed in
  ``response_for_logging`` in the same shape OpenRouter returns
  (``usage.cost`` in USD, ``provider``), so ``runner.usage.served_cost`` and
  ``served_provider`` read them in the match stats.
* ``model_options["reasoning"] = {"effort": "low"|"medium"|"high"}`` maps to
  the SDK's ``reasoning`` level. Unset means the model's default, which is
  what the OpenRouter runs used.
"""

from __future__ import annotations

from typing import Any, Mapping

from game_arena.harness import model_generation
import openai

PROVIDER = "kaggle-model-proxy"
DEFAULT_MAX_TOKENS = 16384
_RETRYABLE_4XX = (408, 429)


def resolve_slug(slug: str, available) -> str:
  """Maps a bare slug (``gpt-oss-20b``) to the proxy key (``openai/gpt-oss-20b``)."""
  if slug in available:
    return slug
  def bare(key: str) -> set[str]:  # "anthropic/x@default" -> {"x", "x-default"}
    name = key.split("/", 1)[-1]
    return {name.split("@", 1)[0], name.replace("@", "-")}

  matches = [k for k in available if slug in bare(k)]
  if len(matches) == 1:
    return matches[0]
  raise ValueError(f"{slug!r} is not available on the Model Proxy (matches: {matches}). "
                   f"Available: {sorted(available)}")


class KbenchModel(model_generation.MultimodalModel):
  """A model served through the Kaggle Model Proxy via ``kaggle_benchmarks``."""

  def __init__(
      self,
      model_name: str,
      *,
      model_options: Mapping[str, Any] | None = None,
      api_options: Mapping[str, Any] | None = None,
  ):
    super().__init__(model_name, model_options=model_options, api_options=api_options)
    if self._model_options is None:
      self._model_options = {}
    import kaggle_benchmarks as kbench  # pylint: disable=import-outside-toplevel

    self._kbench = kbench
    self._llm = kbench.llms[resolve_slug(model_name, kbench.llms)]
    # Streaming (on in interactive notebooks) drops reasoning traces.
    self._llm.stream_responses = False

  def _generate(self, prompt_text: str,
                system_instruction: str | None) -> model_generation.GenerateReturn:
    kwargs: dict[str, Any] = {
        "max_tokens": int(self._model_options.get("max_tokens", DEFAULT_MAX_TOKENS)),
    }
    effort = (self._model_options.get("reasoning") or {}).get("effort")
    if effort:
      kwargs["reasoning"] = effort
    request = {"model": self._model_name, "system": system_instruction,
               "prompt": prompt_text, **kwargs}
    # The error is re-raised after the chat context closes: in Kaggle batch
    # runs kbench sets continue_with_exceptions, and the context swallows
    # exceptions raised inside it when there is no enclosing run (worker
    # threads start from the root context).
    error: Exception | None = None
    with self._kbench.chats.new(f"truco:{self._model_name}", orphan=True):
      self._kbench.user.send(prompt_text)
      try:
        message = self._llm.respond(system=system_instruction, **kwargs)
      except Exception as e:  # pylint: disable=broad-exception-caught
        error = e
    if error is not None:
      # Auth, bad-request and unknown-model errors never succeed on retry
      # (an expired proxy key would otherwise retry forever).
      if (isinstance(error, openai.APIStatusError)
          and 400 <= error.status_code < 500 and error.status_code not in _RETRYABLE_4XX):
        raise model_generation.DoNotRetryError(
            f"Model Proxy returned HTTP {error.status_code} for {self._model_name}",
            info=str(error)[:2000]) from error
      raise error

    content = message.content or ""
    reasoning = message.reasoning_traces
    if reasoning:
      main_response, with_thoughts = content, f"<think>{reasoning}</think>{content}"
    else:
      main_response = with_thoughts = content
      separated = model_generation.separate_main_response_and_thoughts(content)
      if separated is not None:
        main_response = separated[0]

    usage = message.usage
    cost_nd = usage.total_cost_nanodollars
    total = None
    if usage.input_tokens is not None and usage.output_tokens is not None:
      total = usage.input_tokens + usage.output_tokens
    return model_generation.GenerateReturn(
        main_response=main_response,
        main_response_and_thoughts=with_thoughts,
        request_for_logging=request,
        response_for_logging={
            "provider": PROVIDER,
            "usage": {
                "prompt_tokens": usage.input_tokens,
                "completion_tokens": usage.output_tokens,
                "cost": None if cost_nd is None else cost_nd / 1e9,
                "backend_latency_ms": usage.total_backend_latency_ms,
            },
        },
        generation_tokens=usage.output_tokens,
        prompt_tokens=usage.input_tokens,
        total_tokens=total,
    )

  def generate_with_text_input(
      self, model_input: model_generation.ModelTextInput
  ) -> model_generation.GenerateReturn:
    return self._generate(model_input.prompt_text, model_input.system_instruction)

  def generate_with_image_text_input(
      self, model_input: model_generation.ModelImageTextInput
  ) -> model_generation.GenerateReturn:
    raise model_generation.UnsupportedCapabilityError("Truco prompts are text-only.")
