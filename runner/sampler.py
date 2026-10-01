"""Custom sampler implementing the benchmark's illegal-action policy.

Policy (overrides the harness rethink defaults):
  1. Prompt the model. Parse with the rule-based parser, then soft-match
     against the legal list.
  2. If the result is not exactly one legal action, re-prompt ONCE with the
     legal list shown again and the previous reply quoted.
  3. If the second reply is also illegal, the action is replaced by a
     uniformly random legal action drawn from the engine's seed-derived
     fallback stream. Every illegal reply is recorded as a structured event.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Sequence

from game_arena.harness import model_generation
from game_arena.harness import parsers
from game_arena.harness import prompt_generation
from game_arena.harness import samplers

from runner import prompts as truco_prompts
from runner import parsers as truco_parsers

GAME_SHORT_NAME = "truco"


@dataclasses.dataclass(frozen=True, kw_only=True)
class IllegalActionRecord:
  attempt: int
  raw_response: str
  extracted_action: str | None
  legal_actions: list[str]
  prompt_text: str


class TrucoSampler(samplers.Sampler):
  """Samples one legal Truco action following the policy above."""

  def __init__(
      self,
      model: model_generation.Model,
      *,
      max_reprompts: int = 1,
      prompt_generator: prompt_generation.PromptGeneratorText | None = None,
      system_instruction: str = truco_prompts.SYSTEM_INSTRUCTION,
      rethink_template: str = truco_prompts.RETHINK,
  ):
    super().__init__(model)
    self._max_reprompts = max_reprompts
    self._prompt_generator = prompt_generator or prompt_generation.PromptGeneratorText()
    self._system_instruction = system_instruction
    self._rethink_template = rethink_template
    self._rule_parser = truco_parsers.TrucoRuleParser()

  @property
  def system_instruction(self) -> str:
    return self._system_instruction

  def build_prompt(
      self,
      *,
      observation: dict[str, Any],
      legal_actions: Sequence[str],
      readable_state_str: str,
      rethink_prompt: str = "",
  ) -> model_generation.ModelTextInput:
    decision_type = truco_prompts.decision_type_for(observation)
    template = truco_prompts.TEMPLATE_BY_DECISION_TYPE[decision_type]
    subs = truco_prompts.prompt_substitutions(observation, legal_actions, readable_state_str)
    model_input = self._prompt_generator.generate_prompt_with_text_only(
        template, GAME_SHORT_NAME, rethink_prompt=rethink_prompt, **subs
    )
    return dataclasses.replace(model_input, system_instruction=self._system_instruction)

  def sample_action_with_text_input(
      self, model_input: model_generation.ModelTextInput
  ) -> samplers.SamplerOutput:
    raise NotImplementedError("Use sample_action_with_state_input for Truco.")

  def sample_action_with_state_input(
      self,
      *,
      observation: dict[str, Any],
      legal_actions: Sequence[str],
      readable_state_str: str,
      fallback_fn: Callable[[], str],
  ) -> samplers.SamplerOutput:
    legal = list(legal_actions)
    soft_parser = truco_parsers.TrucoSoftParser(
        truco_parsers.aliases_for(observation, legal)
    )
    legality_parser = parsers.ChainedMoveParser([soft_parser])

    generate_returns: list[model_generation.GenerateReturn] = []
    attempts: list[dict[str, Any]] = []
    illegal: list[IllegalActionRecord] = []
    rethink_prompt = ""
    quoted_generation: str | None = None
    matched: str | None = None
    extracted: str | None = None
    talk: str | None = None

    for attempt in range(self._max_reprompts + 1):
      model_input = self.build_prompt(
          observation=observation, legal_actions=legal,
          readable_state_str=readable_state_str, rethink_prompt=rethink_prompt,
      )
      generate_return = self._model.generate_with_text_input(model_input)
      generate_returns.append(generate_return)
      raw = generate_return.main_response or ""
      if not raw.strip() and generate_return.main_response_and_thoughts:
        # Some endpoints return the whole reply inside the reasoning field with
        # empty visible content. Only then is the reasoning text parsed instead.
        raw = generate_return.main_response_and_thoughts
      extracted = self._rule_parser.parse(parsers.TextParserInput(text=raw))
      matched = legality_parser.parse(parsers.TextParserInput(
          text="" if extracted is None else extracted,
          state_str=readable_state_str,
          legal_moves=legal,
          player_number=int(observation["seat"]),
      ))
      attempts.append({
          "attempt": attempt,
          "prompt_text": model_input.prompt_text,
          "extracted_action": extracted,
          "matched_action": matched,
          # The previous reply quoted inside a re-prompt (None on attempt 0). It
          # is the model's own text, so isolation checks must not grep it.
          "quoted_generation": quoted_generation,
      })
      if matched is not None:
        talk = truco_parsers.extract_talk(raw)
        break
      illegal.append(IllegalActionRecord(
          attempt=attempt, raw_response=raw, extracted_action=extracted,
          legal_actions=legal, prompt_text=model_input.prompt_text,
      ))
      quoted_generation = raw.strip()[-4000:]
      rethink_prompt = self._rethink_template.format(
          generation=quoted_generation,
          legal_actions=truco_prompts.format_legal_actions(legal),
      )

    if matched is not None:
      action, move_type, fallback = matched, samplers.MoveType.LEGAL, False
    else:
      action, move_type, fallback = fallback_fn(), samplers.MoveType.ILLEGAL, True
      assert action in legal
      talk = None  # AMBIGUITIES #17: no talk from an illegal reply.

    return samplers.SamplerOutput(
        action=action,
        extracted_action=extracted,
        matched_action=matched,
        generate_returns=generate_returns,
        auxiliary_outputs={
            "attempts": attempts,
            "illegal": [dataclasses.asdict(r) for r in illegal],
            "fallback": fallback,
            "talk": talk,
        },
        move_type=move_type,
    )
