"""Illegal-action policy: first -> re-prompt, second -> random legal, recorded."""

from game_arena.harness import samplers

from runner import render
from runner.sampler import TrucoSampler
from tests.fake_model import FakeModel
from truco.match import TrucoMatch


def _ctx():
  m = TrucoMatch(3)
  seat = m.current_actor()
  obs = m.observation(seat)
  legal = m.legal_actions(seat)
  return m, seat, obs, legal, render.render_observation(obs)


def _sample(model, m, seat, obs, legal, readable):
  return TrucoSampler(model).sample_action_with_state_input(
      observation=obs, legal_actions=legal, readable_state_str=readable,
      fallback_fn=lambda: m.random_legal_action(seat))


def test_legal_first_try_no_reprompt():
  m, seat, obs, legal, readable = _ctx()
  model = FakeModel(responses=[f"Talk: hello table\nFinal Answer: {legal[0]}"])
  out = _sample(model, m, seat, obs, legal, readable)
  assert out.action == legal[0] and out.move_type == samplers.MoveType.LEGAL
  assert len(model.calls) == 1
  assert out.auxiliary_outputs["illegal"] == [] and not out.auxiliary_outputs["fallback"]
  assert out.auxiliary_outputs["talk"] == "hello table"
  assert model.calls[0].system_instruction.startswith("You are playing the card game Truco")
  assert "{rethink_prompt}" not in model.calls[0].prompt_text


def test_illegal_then_legal_reprompts_once_with_legal_list_and_quote():
  m, seat, obs, legal, readable = _ctx()
  model = FakeModel(responses=["Final Answer: PLAY 9♣", f"Final Answer: {legal[1]}"])
  out = _sample(model, m, seat, obs, legal, readable)
  assert out.action == legal[1] and out.move_type == samplers.MoveType.LEGAL
  assert len(model.calls) == 2
  reprompt = model.calls[1].prompt_text
  assert "did not contain exactly one legal action" in reprompt
  assert "Final Answer: PLAY 9♣" in reprompt
  for a in legal:
    assert f"- {a}" in reprompt
  illegal = out.auxiliary_outputs["illegal"]
  assert len(illegal) == 1
  assert illegal[0]["attempt"] == 0
  assert illegal[0]["raw_response"] == "Final Answer: PLAY 9♣"
  assert illegal[0]["extracted_action"] == "PLAY9♣"
  assert illegal[0]["legal_actions"] == legal


def test_illegal_twice_falls_back_to_seeded_random_legal_action():
  m, seat, obs, legal, readable = _ctx()
  model = FakeModel(responses=["no idea", "Final Answer: DANCE"])
  out = _sample(model, m, seat, obs, legal, readable)
  expected = TrucoMatch(3).random_legal_action(seat)  # same seed -> same fallback draw
  assert out.action == expected and out.action in legal
  assert out.move_type == samplers.MoveType.ILLEGAL
  assert out.auxiliary_outputs["fallback"] is True
  assert len(out.auxiliary_outputs["illegal"]) == 2
  assert out.auxiliary_outputs["illegal"][1]["attempt"] == 1
  assert out.auxiliary_outputs["talk"] is None
  assert len(model.calls) == 2  # never a third prompt


def test_unparsable_first_response_counts_as_illegal():
  m, seat, obs, legal, readable = _ctx()
  model = FakeModel(responses=["", f"Final Answer: {legal[0]}"])
  out = _sample(model, m, seat, obs, legal, readable)
  assert out.action == legal[0]
  assert out.auxiliary_outputs["illegal"][0]["extracted_action"] is None


def test_soft_match_accepts_ascii_suit_in_final_answer():
  m, seat, obs, legal, readable = _ctx()
  card = legal[0][5:]
  ascii_card = card[0] + {"♣": "C", "♥": "H", "♠": "S", "♦": "D"}[card[1]]
  model = FakeModel(responses=[f"Final Answer: play {ascii_card}"])
  out = _sample(model, m, seat, obs, legal, readable)
  assert out.action == legal[0] and out.auxiliary_outputs["illegal"] == []


def test_sampler_accepts_compressed_suit_word_card():
  m, seat, obs, _, readable = _ctx()
  legal = ["PLAY K♦", "PLAY 3♣", "TRUCO", "FOLD"]
  model = FakeModel(responses=["Final Answer: K of diamonds"])
  out = _sample(model, m, seat, obs, legal, readable)
  assert out.action == "PLAY K♦"
  assert out.auxiliary_outputs["attempts"][0]["attempt"] == 0
  assert out.auxiliary_outputs["illegal"] == []
  assert out.auxiliary_outputs["fallback"] is False
  assert len(model.calls) == 1


def test_sampler_accepts_markdown_play_with_compressed_suit_word_card():
  m, seat, obs, _, readable = _ctx()
  legal = ["PLAY K♦", "PLAY 3♣", "TRUCO", "FOLD"]
  model = FakeModel(responses=["Final Answer: **play the 3 of clubs**"])
  out = _sample(model, m, seat, obs, legal, readable)
  assert out.action == "PLAY 3♣"
  assert out.auxiliary_outputs["illegal"] == []


def test_sampler_accepts_portuguese_suit_word_card():
  m, seat, obs, _, readable = _ctx()
  legal = ["PLAY Q♠", "PLAY 3♣", "TRUCO", "FOLD"]
  model = FakeModel(responses=["Final Answer: Q de espadas"])
  out = _sample(model, m, seat, obs, legal, readable)
  assert out.action == "PLAY Q♠"
  assert out.auxiliary_outputs["illegal"] == []


def test_sampler_reprompts_for_illegal_suit_word_card():
  m, seat, obs, _, readable = _ctx()
  legal = ["PLAY K♦", "PLAY 3♣", "TRUCO", "FOLD"]
  model = FakeModel(responses=["Final Answer: K of hearts", "Final Answer: PLAY K♦"])
  out = _sample(model, m, seat, obs, legal, readable)
  assert out.action == "PLAY K♦"
  assert out.auxiliary_outputs["attempts"][0]["matched_action"] is None
  assert out.auxiliary_outputs["illegal"][0]["attempt"] == 0
  assert out.auxiliary_outputs["illegal"][0]["extracted_action"] == "Kofhearts"
  assert len(model.calls) == 2


def test_sampler_maps_mao_forfeit_without_corrupting_of_text():
  m, seat, obs, _, readable = _ctx()
  legal = ["MAO_PLAY", "MAO_FORFEIT"]
  model = FakeModel(responses=["Final Answer: MAO_FORFEIT"])
  out = _sample(model, m, seat, obs, legal, readable)
  assert out.action == "MAO_FORFEIT"
  assert out.auxiliary_outputs["illegal"] == []


def test_prompt_templates_cover_every_decision_type():
  from runner import prompts
  # Raise response
  m = TrucoMatch(0)
  m.apply_action(0, "TRUCO")
  obs = m.observation(1)
  assert prompts.decision_type_for(obs) == prompts.DECISION_TYPE_RAISE_RESPONSE
  s = TrucoSampler(FakeModel())
  text = s.build_prompt(observation=obs, legal_actions=m.legal_actions(1),
                        readable_state_str=render.render_observation(obs)).prompt_text
  assert "has called TRUCO" in text and "- RAISE" in text
  # Mão de onze
  m = TrucoMatch(0, initial_scores={"A": 11, "B": 2})
  obs = m.observation(0)
  assert prompts.decision_type_for(obs) == prompts.DECISION_TYPE_MAO_DE_ONZE
  text = s.build_prompt(observation=obs, legal_actions=m.legal_actions(0),
                        readable_state_str=render.render_observation(obs)).prompt_text
  assert "MÃO DE ONZE" in text and "Partner seat 2's hand" in text
  # Mão de ferro
  m = TrucoMatch(0, initial_scores={"A": 11, "B": 11})
  obs = m.observation(0)
  assert prompts.decision_type_for(obs) == prompts.DECISION_TYPE_MAO_DE_FERRO
  text = s.build_prompt(observation=obs, legal_actions=m.legal_actions(0),
                        readable_state_str=render.render_observation(obs)).prompt_text
  assert "MÃO DE FERRO" in text and "- PLAY 1" in text and "Your cards are face down" in text
  # Card play with raise clause
  m = TrucoMatch(0)
  obs = m.observation(0)
  text = s.build_prompt(observation=obs, legal_actions=m.legal_actions(0),
                        readable_state_str=render.render_observation(obs)).prompt_text
  assert "call TRUCO (raise the stake to 3)" in text


def test_empty_content_falls_back_to_reasoning_text():
  from game_arena.harness import model_generation
  m, seat, obs, legal, readable = _ctx()

  class ReasoningOnlyModel(FakeModel):
    def generate_with_text_input(self, model_input):
      self.calls.append(model_input)
      return model_generation.GenerateReturn(
          main_response="", main_response_and_thoughts=f"<think>ok\nFinal Answer: {legal[0]}</think>")

  out = _sample(ReasoningOnlyModel(), m, seat, obs, legal, readable)
  assert out.action == legal[0] and out.auxiliary_outputs["illegal"] == []
  # With non-empty content, the reasoning is never consulted.
  class BothModel(FakeModel):
    def generate_with_text_input(self, model_input):
      self.calls.append(model_input)
      return model_generation.GenerateReturn(
          main_response="no move here", main_response_and_thoughts=f"<think>Final Answer: {legal[0]}</think>no move here")
  out = _sample(BothModel(), m, seat, obs, legal, readable)
  assert out.auxiliary_outputs["fallback"] is True
