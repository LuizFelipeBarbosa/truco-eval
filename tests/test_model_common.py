"""Shared model-layer policy and the canonical event encoding."""

from __future__ import annotations

from game_arena.harness import model_generation
import pytest

import model_common
from truco.match import TrucoMatch, canonical_json


def test_default_max_tokens_is_the_benchmark_value():
  assert model_common.DEFAULT_MAX_TOKENS == 16384
  assert model_common.max_tokens({}) == 16384
  assert model_common.max_tokens({"max_tokens": "512"}) == 512


@pytest.mark.parametrize("status, never_retry", [
    (400, True), (401, True), (404, True), (408, False), (429, False), (500, False), (200, False),
])
def test_non_retryable_statuses(status, never_retry):
  assert model_common.is_non_retryable_status(status) is never_retry


def test_split_thoughts_wraps_out_of_band_reasoning():
  assert model_common.split_thoughts("Final Answer: FOLD", "why") == (
      "Final Answer: FOLD", "<think>why</think>Final Answer: FOLD")


def test_split_thoughts_separates_inline_think_tags():
  main, full = model_common.split_thoughts("<think>why</think>Final Answer: FOLD", None)
  assert main == "Final Answer: FOLD"
  assert full == "<think>why</think>Final Answer: FOLD"
  assert model_common.split_thoughts("plain", None) == ("plain", "plain")


def test_served_cost_and_provider_read_the_logged_response():
  gr = model_generation.GenerateReturn(
      main_response="", main_response_and_thoughts="",
      response_for_logging={"provider": "P", "usage": {"cost": "0.25"}})
  assert model_common.served_provider(gr) == "P"
  assert model_common.served_cost(gr) == 0.25
  empty = model_generation.GenerateReturn(main_response="", main_response_and_thoughts="")
  assert model_common.served_provider(empty) is None and model_common.served_cost(empty) is None


def test_canonical_json_encoding_is_pinned():
  # Replay verification compares bytes against files written by earlier code: this encoding
  # must never change.
  assert canonical_json({"b": [1, 2], "a": "Q♠", "c": None}) == '{"a":"Q♠","b":[1,2],"c":null}'
  match = TrucoMatch(3)
  assert match.serialize_events() == "".join(canonical_json(e) + "\n" for e in match.events)
