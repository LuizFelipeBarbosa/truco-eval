"""Configuration dataclasses for model configurations and match specs."""

from __future__ import annotations

import dataclasses
from typing import Any

from runner import agents as truco_agents

KIND_OPENROUTER = "openrouter"
KIND_RANDOM = "random"
KIND_KBENCH = "kbench"  # Kaggle Model Proxy, only inside kaggle_benchmarks


@dataclasses.dataclass(frozen=True, kw_only=True)
class ModelConfig:
  """One 'model configuration' that fills a team."""

  kind: str = KIND_OPENROUTER
  slug: str = ""
  provider: dict[str, Any] | None = None
  model_options: dict[str, Any] = dataclasses.field(default_factory=dict)
  api_options: dict[str, Any] = dataclasses.field(default_factory=dict)
  base_url: str | None = None
  label: str = ""
  max_reprompts: int = 1

  @property
  def display(self) -> str:
    if self.label:
      return self.label
    return "random" if self.kind == KIND_RANDOM else self.slug

  def to_dict(self) -> dict[str, Any]:
    d = dataclasses.asdict(self)
    d["display"] = self.display
    return d

  def to_dict_no_display(self) -> dict[str, Any]:
    return dataclasses.asdict(self)

  def build_agent(self) -> truco_agents.Agent:
    if self.kind == KIND_RANDOM:
      return truco_agents.RandomBotAgent()
    if self.kind == KIND_KBENCH:
      from kbench_model import KbenchModel  # pylint: disable=import-outside-toplevel

      model = KbenchModel(self.slug, model_options=dict(self.model_options),
                          api_options=dict(self.api_options))
      return truco_agents.LLMAgent(model, self.display, max_reprompts=self.max_reprompts)
    if self.kind != KIND_OPENROUTER:
      raise ValueError(f"Unknown model kind {self.kind!r}")
    # Imported lazily so the engine/runner tests never need the harness's SDKs.
    from openrouter_model import OpenRouterModel  # pylint: disable=import-outside-toplevel

    options = dict(self.model_options)
    if self.provider is not None:
      options["provider"] = self.provider
    kwargs: dict[str, Any] = {}
    if self.base_url:
      kwargs["base_url"] = self.base_url
    model = OpenRouterModel(self.slug, model_options=options,
                            api_options=dict(self.api_options), **kwargs)
    return truco_agents.LLMAgent(model, self.display, max_reprompts=self.max_reprompts)


@dataclasses.dataclass(frozen=True, kw_only=True)
class MatchSpec:
  match_id: str
  seed: int
  num_players: int = 4
  team_a: ModelConfig
  team_b: ModelConfig
  swap: bool = False  # True: team_a config sits on the odd seats (duplicate match).
  out_dir: str | None = None

  def config_for_team(self, team: str) -> ModelConfig:
    if (team == "A") != self.swap:
      return self.team_a
    return self.team_b

  def seat_configs(self) -> dict[int, ModelConfig]:
    return {s: self.config_for_team("A" if s % 2 == 0 else "B") for s in range(self.num_players)}
