"""Seat agents: LLM-backed (via the sampler), random bot, and scripted (tests)."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Protocol, Sequence

from game_arena.harness import model_generation
from game_arena.harness import samplers

from runner import render
from runner.sampler import TrucoSampler


@dataclasses.dataclass(frozen=True, kw_only=True)
class Decision:
  action: str
  talk: str | None = None
  source: str = "model"  # "model" | "fallback" | "random_bot" | "scripted"
  prompts: list[dict[str, Any]] = dataclasses.field(default_factory=list)
  generate_returns: list[model_generation.GenerateReturn] = dataclasses.field(default_factory=list)
  illegal: list[dict[str, Any]] = dataclasses.field(default_factory=list)
  extracted_action: str | None = None


class Agent(Protocol):
  name: str

  def decide(
      self,
      observation: dict[str, Any],
      legal_actions: Sequence[str],
      fallback_fn: Callable[[], str],
  ) -> Decision:
    ...


class RandomBotAgent:
  """Plays a uniformly random legal action from the engine's fallback stream."""

  name = "random"

  def decide(self, observation, legal_actions, fallback_fn) -> Decision:
    return Decision(action=fallback_fn(), source="random_bot")


class ScriptedAgent:
  """Returns canned model responses (for tests); falls back to a policy fn."""

  name = "scripted"

  def __init__(self, responses: Sequence[str]):
    self._responses = list(responses)

  def decide(self, observation, legal_actions, fallback_fn) -> Decision:
    action = self._responses.pop(0) if self._responses else fallback_fn()
    return Decision(action=action, source="scripted")


class LLMAgent:
  """One seat driven by a harness ``Model`` through ``TrucoSampler``."""

  def __init__(self, model: model_generation.Model, name: str, *, max_reprompts: int = 1):
    self.name = name
    self._sampler = TrucoSampler(model, max_reprompts=max_reprompts)

  @property
  def system_instruction(self) -> str:
    return self._sampler.system_instruction

  def decide(self, observation, legal_actions, fallback_fn) -> Decision:
    readable = render.render_observation(observation)
    out: samplers.SamplerOutput = self._sampler.sample_action_with_state_input(
        observation=observation, legal_actions=legal_actions,
        readable_state_str=readable, fallback_fn=fallback_fn,
    )
    aux = out.auxiliary_outputs
    return Decision(
        action=out.action,
        talk=aux["talk"],
        source="fallback" if aux["fallback"] else "model",
        prompts=aux["attempts"],
        generate_returns=list(out.generate_returns),
        illegal=aux["illegal"],
        extracted_action=out.extracted_action,
    )
