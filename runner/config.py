"""Configuration dataclasses for model configurations and match specs.

``ModelConfig.from_entry`` is the one place a model setting is built from a
loose mapping (CLI flags, a models file, a ``tournament.json`` record), and
``build_model`` the one place a config becomes a harness model. ``match_id``
and ``parse_match_id`` define the ``seed{N}_{orig|dup}`` naming.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any, Mapping

from game_arena.harness import model_generation

from runner import agents as truco_agents
from truco.match import TEAMS, team_of

KIND_OPENROUTER = "openrouter"
KIND_RANDOM = "random"
KIND_HEURISTIC = "heuristic"
KIND_KBENCH = "kbench"  # Kaggle Model Proxy, only inside kaggle_benchmarks
BOT_KINDS = (KIND_RANDOM, KIND_HEURISTIC)

_MATCH_ID = re.compile(r"seed(-?\d+)_(orig|dup)")


def match_id(seed: int, swap: bool) -> str:
  """The id of a match: ``seed{N}_orig``, or ``seed{N}_dup`` for the duplicate."""
  return f"seed{seed}_{'dup' if swap else 'orig'}"


def parse_match_id(text: str) -> tuple[int, bool] | None:
  """Return ``(seed, swap)`` for a match id, or ``None`` if ``text`` is not one."""
  m = _MATCH_ID.fullmatch(text)
  return (int(m.group(1)), m.group(2) == "dup") if m else None


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

  @classmethod
  def from_entry(cls, entry: Mapping[str, Any], *,
                 max_reprompts: int | None = None) -> ModelConfig:
    """Build a config from a models-file entry, CLI values or a ``to_dict()`` record.

    Without ``kind``, slug ``random`` or ``heuristic`` (any case) selects that bot and any
    other slug is an OpenRouter model. A bot keeps only its label (default: the kind) and
    ``max_reprompts``. ``max_reprompts``, when given, overrides the entry's. Missing or
    ``None`` fields take the defaults; other keys, such as ``display``, are ignored.
    """
    slug = entry.get("slug") or ""
    kind = entry.get("kind") or (slug.lower() if slug.lower() in BOT_KINDS else KIND_OPENROUTER)
    if max_reprompts is None:
      max_reprompts = entry.get("max_reprompts")
    if max_reprompts is None:
      max_reprompts = 1
    if kind in BOT_KINDS:
      return cls(kind=kind, label=entry.get("label") or kind, max_reprompts=max_reprompts)
    return cls(
        kind=kind, slug=slug, provider=entry.get("provider"),
        model_options=dict(entry.get("model_options") or {}),
        api_options=dict(entry.get("api_options") or {}),
        base_url=entry.get("base_url"), label=entry.get("label") or "",
        max_reprompts=max_reprompts,
    )

  @property
  def display(self) -> str:
    if self.label:
      return self.label
    if self.kind in BOT_KINDS:
      return self.kind
    return self.slug

  def to_dict(self) -> dict[str, Any]:
    d = dataclasses.asdict(self)
    d["display"] = self.display
    return d

  def to_dict_no_display(self) -> dict[str, Any]:
    return dataclasses.asdict(self)

  def build_model(self) -> model_generation.Model:
    """The harness model behind an LLM config; raises ``ValueError`` for a bot."""
    # Model layers are imported lazily so the engine/runner tests never need their SDKs.
    if self.kind == KIND_KBENCH:
      from kbench_model import KbenchModel  # pylint: disable=import-outside-toplevel

      return KbenchModel(self.slug, model_options=dict(self.model_options),
                         api_options=dict(self.api_options))
    if self.kind != KIND_OPENROUTER:
      raise ValueError(f"Model kind {self.kind!r} has no model")
    from openrouter_model import OpenRouterModel  # pylint: disable=import-outside-toplevel

    options = dict(self.model_options)
    if self.provider is not None:
      options["provider"] = self.provider
    kwargs: dict[str, Any] = {}
    if self.base_url:
      kwargs["base_url"] = self.base_url
    return OpenRouterModel(self.slug, model_options=options,
                           api_options=dict(self.api_options), **kwargs)

  def build_agent(self) -> truco_agents.Agent:
    if self.kind == KIND_RANDOM:
      return truco_agents.RandomBotAgent()
    if self.kind == KIND_HEURISTIC:
      from runner.heuristic import HeuristicBotAgent  # pylint: disable=import-outside-toplevel

      return HeuristicBotAgent()
    if self.kind not in (KIND_OPENROUTER, KIND_KBENCH):
      raise ValueError(f"Unknown model kind {self.kind!r}")
    return truco_agents.LLMAgent(self.build_model(), self.display,
                                 max_reprompts=self.max_reprompts)


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
    if (team == TEAMS[0]) != self.swap:
      return self.team_a
    return self.team_b

  def seat_configs(self) -> dict[int, ModelConfig]:
    return {s: self.config_for_team(team_of(s)) for s in range(self.num_players)}
