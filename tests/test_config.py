"""Model settings: ``ModelConfig.from_entry``, ``build_model`` and match ids."""

from __future__ import annotations

import pytest

from openrouter_model import OpenRouterModel
from runner import cli, tournament
from runner.config import (
    KIND_HEURISTIC,
    KIND_KBENCH,
    KIND_OPENROUTER,
    KIND_RANDOM,
    ModelConfig,
    match_id,
    parse_match_id,
)


@pytest.mark.parametrize("slug, kind", [("random", KIND_RANDOM), ("Heuristic", KIND_HEURISTIC)])
def test_from_entry_bot_slug_keeps_only_label_and_reprompts(slug, kind):
  cfg = ModelConfig.from_entry({"slug": slug, "model_options": {"temperature": 1.0},
                                "provider": {"order": ["X"]}}, max_reprompts=2)
  assert cfg == ModelConfig(kind=kind, label=kind, max_reprompts=2)
  assert ModelConfig.from_entry({"slug": slug, "label": "bot"}).label == "bot"


def test_from_entry_openrouter_defaults_and_none_values():
  cfg = ModelConfig.from_entry({"slug": "openai/gpt-5-mini", "label": None,
                                "model_options": None, "provider": None})
  assert cfg == ModelConfig(kind=KIND_OPENROUTER, slug="openai/gpt-5-mini")
  assert cfg.display == "openai/gpt-5-mini"


def test_from_entry_round_trips_to_dict_and_ignores_display():
  cfg = ModelConfig(kind=KIND_KBENCH, slug="gpt-oss-20b", label="oss", max_reprompts=0,
                    model_options={"reasoning": {"effort": "low"}}, api_options={"timeout": 5})
  assert ModelConfig.from_entry(cfg.to_dict()) == cfg
  bot = ModelConfig(kind=KIND_HEURISTIC, label="heuristic")
  assert ModelConfig.from_entry(bot.to_dict()) == bot


def test_from_entry_max_reprompts_argument_overrides_entry():
  entry = {"slug": "x/y", "max_reprompts": 3}
  assert ModelConfig.from_entry(entry).max_reprompts == 3
  assert ModelConfig.from_entry(entry, max_reprompts=0).max_reprompts == 0
  assert ModelConfig.from_entry({"slug": "x/y"}).max_reprompts == 1


def test_cli_run_and_models_file_build_the_same_config():
  from_flags = cli.model_config_from_args("x/y", '{"temperature": 0.5}', '{"order": ["A"]}',
                                          "lbl", "http://localhost:1/v1", 2, '{"timeout": 9}')
  from_file = ModelConfig.from_entry(
      {"slug": "x/y", "label": "lbl", "model_options": {"temperature": 0.5},
       "provider": {"order": ["A"]}, "api_options": {"timeout": 9},
       "base_url": "http://localhost:1/v1"}, max_reprompts=2)
  assert from_flags == from_file


def test_build_model_merges_provider_and_base_url(monkeypatch):
  monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
  cfg = ModelConfig(slug="x/y", provider={"order": ["A"]}, model_options={"temperature": 0.1},
                    base_url="http://localhost:1/v1")
  model = cfg.build_model()
  assert isinstance(model, OpenRouterModel)
  assert model.base_url == "http://localhost:1/v1"
  assert model.build_request([])["provider"] == {"order": ["A"]}
  assert cfg.build_agent().name == "x/y"


@pytest.mark.parametrize("kind", [KIND_RANDOM, KIND_HEURISTIC, "bogus"])
def test_build_model_rejects_kinds_without_a_model(kind):
  with pytest.raises(ValueError):
    ModelConfig(kind=kind, label="b").build_model()


def test_preflight_probes_the_built_model(monkeypatch):
  prompts = []

  class Probe:
    def generate_with_text_input(self, model_input):
      prompts.append(model_input.prompt_text)

  monkeypatch.setattr(ModelConfig, "build_model", lambda self: Probe())
  models = [ModelConfig(slug="x/y"), ModelConfig(kind=KIND_RANDOM)]
  assert tournament.preflight(models) == {}
  assert prompts == ["Reply with exactly: Final Answer: FOLD"]


@pytest.mark.parametrize("seed, swap, text", [(0, False, "seed0_orig"), (17, True, "seed17_dup")])
def test_match_id_round_trips(seed, swap, text):
  assert match_id(seed, swap) == text
  assert parse_match_id(text) == (seed, swap)


@pytest.mark.parametrize("text", ["seed1", "seed1_orig.json", "x_seed1_dup", "seedA_orig", ""])
def test_parse_match_id_rejects_other_names(text):
  assert parse_match_id(text) is None
