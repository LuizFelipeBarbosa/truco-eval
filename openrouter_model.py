"""OpenRouter model class for the Game Arena harness.

Follows the ``TogetherAIModel`` pattern from
``game_arena.harness.model_generation_http``: an OpenAI-compatible
chat-completions POST. Differences:

* ``base_url`` is a constructor parameter (default OpenRouter) so the same
  class can point at a self-hosted vLLM / Ollama endpoint.
* ``model_options["provider"]`` is forwarded verbatim as the OpenRouter
  ``provider`` routing object (provider pinning). The provider that actually
  served the request is read back from the response
  (``model_common.served_provider``).
* ``model_options["reasoning"]`` is forwarded verbatim (e.g. ``{"effort":
  "high"}``); reasoning text returned by the API is captured into
  ``main_response_and_thoughts`` wrapped in ``<think>`` tags.
* ``usage.include`` is requested so the response carries token usage and the
  OpenRouter-computed cost.
* HTTP 4xx (other than 408/429) raises ``DoNotRetryError``; everything else
  is raised as a plain exception and retried by the harness's retry decorator
  (which the base class installs through ``__init_subclass__``).

The token default, retry codes and reasoning handling live in ``model_common``.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

from absl import logging
from game_arena.harness import model_generation, model_generation_http
import requests

import model_common

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT_SECS = 600
_PASSTHROUGH_OPTIONS = ("temperature", "top_p", "top_k", "provider", "reasoning",
                        "seed", "stop", "presence_penalty", "frequency_penalty",
                        "repetition_penalty", "min_p", "top_a", "response_format")


class OpenRouterModel(model_generation.MultimodalModel):
  """Wrapper for models served through OpenRouter (or any OpenAI-compatible URL)."""

  def __init__(
      self,
      model_name: str,
      *,
      model_options: Mapping[str, Any] | None = None,
      api_options: Mapping[str, Any] | None = None,
      api_key: str | None = None,
      base_url: str = DEFAULT_BASE_URL,
  ):
    super().__init__(
        model_name, model_options=model_options, api_options=api_options
    )
    if self._model_options is None:
      self._model_options = {}
    if self._api_options is None:
      self._api_options = {}
    if api_key is None:
      api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
      if base_url == DEFAULT_BASE_URL:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable not set. Set it to use "
            f"{model_name} through OpenRouter."
        )
      api_key = "none"  # Self-hosted endpoints usually ignore the token.
    self._base_url = base_url
    self._headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": self._api_options.get("referer", "https://github.com/truco-eval"),
        "X-Title": self._api_options.get("title", "truco-eval"),
    }

  @property
  def base_url(self) -> str:
    return self._base_url

  def build_request(
      self,
      content: Sequence[Mapping[str, Any]],
      system_instruction: str | None = None,
  ) -> dict[str, Any]:
    """The JSON body that will be POSTed (exposed for logging and tests)."""
    messages: list[dict[str, Any]] = []
    if system_instruction is not None:
      messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": list(content)})
    request: dict[str, Any] = {
        "model": self._model_name,
        "messages": messages,
        "max_tokens": model_common.max_tokens(self._model_options),
    }
    for option in _PASSTHROUGH_OPTIONS:
      if option in self._model_options:
        request[option] = self._model_options[option]
    if self._api_options.get("include_usage", True):
      request["usage"] = {"include": True}
    extra_body = self._model_options.get("extra_body")
    if extra_body:
      request.update(dict(extra_body))
    return request

  def _post(self, request: Mapping[str, Any]) -> dict[str, Any]:
    timeout = self._api_options.get("timeout", DEFAULT_TIMEOUT_SECS)
    try:
      response = requests.post(
          self._base_url, json=request, headers=self._headers, timeout=timeout
      )
    except requests.exceptions.RequestException as e:
      # Connection errors / timeouts are transient: let the harness retry.
      raise RuntimeError(f"OpenRouter request failed: {e}") from e
    status = response.status_code
    if model_common.is_non_retryable_status(status):
      raise model_generation.DoNotRetryError(
          f"OpenRouter returned HTTP {status} for model {self._model_name}",
          info=response.text[:2000],
      )
    if status != 200:
      raise RuntimeError(
          f"OpenRouter returned HTTP {status} for model {self._model_name}: "
          f"{response.text[:500]}"
      )
    try:
      completion = response.json()
    except ValueError as e:
      raise RuntimeError(f"OpenRouter returned non-JSON body: {response.text[:500]}") from e
    if "error" in completion and not completion.get("choices"):
      err = completion["error"]
      code = err.get("code") if isinstance(err, dict) else None
      if isinstance(code, int) and model_common.is_non_retryable_status(code):
        raise model_generation.DoNotRetryError(
            f"OpenRouter error {code} for model {self._model_name}", info=err
        )
      raise RuntimeError(f"OpenRouter error for model {self._model_name}: {err}")
    return completion

  def _generate(
      self,
      content: Sequence[Mapping[str, Any]],
      system_instruction: str | None = None,
  ) -> model_generation.GenerateReturn:
    request = self.build_request(content, system_instruction)
    completion = self._post(request)

    choices = completion.get("choices") or []
    if not choices:
      raise RuntimeError(f"OpenRouter response has no choices: {completion}")
    message = choices[0].get("message") or {}
    content_text = message.get("content")
    if content_text is None:
      logging.warning(
          "OpenRouter completion content is None for %s. Returning empty string.",
          self._model_name,
      )
      content_text = ""
    if isinstance(content_text, list):  # Some providers return content parts.
      content_text = "".join(
          part.get("text", "") for part in content_text if isinstance(part, dict)
      )

    reasoning = message.get("reasoning")
    if not reasoning and message.get("reasoning_details"):
      parts = []
      for detail in message["reasoning_details"]:
        if isinstance(detail, dict):
          text = detail.get("text") or detail.get("summary")
          if text:
            parts.append(str(text))
      reasoning = "\n".join(parts) or None

    main_response, main_response_and_thoughts = model_common.split_thoughts(
        content_text, reasoning)

    usage = completion.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    provider = completion.get("provider")
    logging.info(
        "OpenRouter served %s via provider %s (usage=%s)",
        self._model_name, provider, usage,
    )
    return model_generation.GenerateReturn(
        main_response=main_response,
        main_response_and_thoughts=main_response_and_thoughts,
        request_for_logging=request,
        response_for_logging=completion,
        generation_tokens=usage.get("completion_tokens"),
        prompt_tokens=usage.get("prompt_tokens"),
        reasoning_tokens=details.get("reasoning_tokens"),
        total_tokens=usage.get("total_tokens"),
    )

  def generate_text_once(
      self, prompt_text: str, system_instruction: str | None = None
  ) -> model_generation.GenerateReturn:
    """One request without the harness retry that wraps ``generate_with_text_input``."""
    return self._generate(model_common.text_content(prompt_text), system_instruction)

  def generate_with_text_input(
      self, model_input: model_generation.ModelTextInput
  ) -> model_generation.GenerateReturn:
    return self.generate_text_once(model_input.prompt_text, model_input.system_instruction)

  def generate_with_image_text_input(
      self, model_input: model_generation.ModelImageTextInput
  ) -> model_generation.GenerateReturn:
    content = model_generation_http._create_image_text_content(model_input)  # pylint: disable=protected-access
    return self._generate(content, model_input.system_instruction)
