"""Mão de onze (§9), mão de ferro (§9), folding (§8)."""

import pytest

from tests.conftest import act, deal
from truco.match import IllegalActionError, TrucoMatch

D = deal("3♠ 3♦ 4♣", "K♠ K♦ 7♣", "3♥ 3♣ 4♥", "Q♠ Q♦ 7♦", vira="4♦")


def test_mao_de_onze_decider_and_visibility_team_a():
  m = TrucoMatch(0, scripted_deals=[D], initial_scores={"A": 11, "B": 6})
  assert m.hand_type == "mao_de_onze"
  assert m.current_actor() == 0
  assert m.legal_actions(0) == ["MAO_PLAY", "MAO_FORFEIT"]
  o0, o2 = m.observation(0), m.observation(2)
  assert o0["partner_hand"] == {"2": ["3♥", "3♣", "4♥"]}
  assert o2["partner_hand"] == {"0": ["3♠", "3♦", "4♣"]}
  for s in (1, 3):
    o = m.observation(s)
    assert "partner_hand" not in o and "partner_hand_at_deal" not in o


def test_mao_de_onze_decider_when_team_b_has_eleven():
  m = TrucoMatch(0, scripted_deals=[D], initial_scores={"A": 4, "B": 11})
  assert m.current_actor() == 1  # first B player in play order from mão (seat 0)
  assert "partner_hand" in m.observation(3)


def test_mao_de_onze_play_fixed_stake_3_no_raises_fold_allowed():
  m = TrucoMatch(0, scripted_deals=[D], initial_scores={"A": 11, "B": 6})
  act(m, "MAO_PLAY")
  assert m.stake == 3 and not m.raises_allowed
  assert m.legal_actions(0) == ["PLAY 3♠", "PLAY 3♦", "PLAY 4♣", "FOLD"]
  # Memory of the partner's cards persists for the hand for both partners.
  assert m.observation(0)["partner_hand_at_deal"] == {"2": ["3♥", "3♣", "4♥"]}
  assert m.observation(2)["partner_hand_at_deal"] == {"0": ["3♠", "3♦", "4♣"]}
  assert "partner_hand" not in m.observation(0)
  act(m, "PLAY 3♠")
  assert m.legal_actions(1) == ["PLAY K♠", "PLAY K♦", "PLAY 7♣", "FOLD"]
  act(m, "PLAY K♠", "PLAY 4♥", "PLAY Q♠", "PLAY 3♦", "PLAY K♦", "PLAY 3♥", "PLAY Q♦")
  assert m.scores == {"A": 14, "B": 6}
  assert m.is_terminal() and m.winner == "A"
  assert m.returns() == {"A": 1.0, "B": -1.0}


def test_mao_de_onze_forfeit_gives_opponents_one_point():
  m = TrucoMatch(0, scripted_deals=[D], initial_scores={"A": 11, "B": 6})
  act(m, "MAO_FORFEIT")
  assert m.scores == {"A": 11, "B": 7}
  assert m.hand_history[0]["reason"] == "mao_de_onze_forfeit"
  assert m.hand_index == 1 and m.dealer == 0


def test_mao_de_onze_fold_gives_opponents_three():
  m = TrucoMatch(0, scripted_deals=[D], initial_scores={"A": 11, "B": 6})
  act(m, "MAO_PLAY", "FOLD")
  assert m.scores == {"A": 11, "B": 9}


def test_mao_de_onze_two_players():
  m = TrucoMatch(0, num_players=2, initial_scores={"A": 11, "B": 0})
  assert m.current_actor() == 0
  assert m.observation(0)["partner_hand"] == {}
  act(m, "MAO_PLAY")
  assert m.stake == 3


def test_mao_de_ferro_blind_positions_and_match_end():
  m = TrucoMatch(0, scripted_deals=[D], initial_scores={"A": 11, "B": 11})
  assert m.hand_type == "mao_de_ferro"
  assert m.stake == 1 and not m.raises_allowed
  assert m.legal_actions(0) == ["PLAY 1", "PLAY 2", "PLAY 3", "FOLD"]
  o = m.observation(0)
  assert o["hand"] is None and o["blind_positions"] == [1, 2, 3]
  for card in ("3♠", "3♦", "4♣"):
    assert card not in str(o)
  act(m, "PLAY 2")  # reveals 3♦
  assert m.plays[-1] == {"trick": 0, "seat": 0, "card": "3♦", "position": 2}
  assert m.observation(0)["blind_positions"] == [1, 3]
  assert m.legal_actions(1) == ["PLAY 1", "PLAY 2", "PLAY 3", "FOLD"]
  act(m, "PLAY 1", "PLAY 3", "PLAY 1")  # K♠, 4♥, Q♠ -> A wins trick 1
  assert m.trick_results[0]["winner_team"] == "A"
  # Position 2 already used by seat 0.
  assert m.legal_actions(0) == ["PLAY 1", "PLAY 3", "FOLD"]
  with pytest.raises(IllegalActionError):
    m.apply_action(0, "PLAY 2")
  act(m, "PLAY 1", "PLAY 2", "PLAY 1", "PLAY 2")  # 3♠, K♦, 3♥, Q♦ -> A wins
  assert m.is_terminal() and m.winner == "A"
  assert m.scores == {"A": 12, "B": 11}


def test_mao_de_ferro_pardas_resolved_normally():
  d = deal("3♠ 2♠ K♠", "3♦ 2♦ K♦", "4♣ 4♥ 6♣", "4♦ 4♠ 6♦", vira="7♦")
  m = TrucoMatch(0, scripted_deals=[d], initial_scores={"A": 11, "B": 11})
  act(m, *(["PLAY 1"] * 4))
  assert m.trick_results[0]["winner_team"] is None
  act(m, *(["PLAY 2"] * 4))
  act(m, *(["PLAY 3"] * 4))
  assert m.hand_history[0]["tricks"] == [None, None, None]
  assert m.winner == "A"  # mão's team


def test_mao_de_ferro_fold_ends_match():
  m = TrucoMatch(0, scripted_deals=[D], initial_scores={"A": 11, "B": 11})
  act(m, "FOLD")
  assert m.winner == "B" and m.scores == {"A": 11, "B": 12}


def test_fold_scoring_in_every_context():
  # Stake 1, no call.
  m = TrucoMatch(0, scripted_deals=[D])
  act(m, "PLAY 3♠", "FOLD")
  assert m.scores == {"A": 1, "B": 0} and m.hand_history[0]["by_seat"] == 1
  # After accepted Truco (stake 3).
  m = TrucoMatch(0, scripted_deals=[D])
  act(m, "TRUCO", "ACCEPT", "PLAY 3♠", "FOLD")
  assert m.scores == {"A": 3, "B": 0}
  # After accepted raise-back chain (stake 6), caller's team folds.
  m = TrucoMatch(0, scripted_deals=[D])
  act(m, "TRUCO", "RAISE", "ACCEPT")  # A calls, B raises, A accepts -> 6
  assert m.stake == 6 and m.current_actor() == 0
  act(m, "FOLD")
  assert m.scores == {"A": 0, "B": 6}
  # While a raise is pending: DECLINE is the fold and pays the pre-call stake.
  m = TrucoMatch(0, scripted_deals=[D])
  act(m, "TRUCO", "ACCEPT", "PLAY 3♠", "SEIS", "DECLINE")
  assert m.scores == {"A": 0, "B": 3}


def test_match_ends_at_twelve_or_more():
  m = TrucoMatch(0, scripted_deals=[D], initial_scores={"A": 10, "B": 0})
  act(m, "TRUCO", "RAISE", "ACCEPT")  # stake 6
  act(m, "PLAY 3♠", "PLAY K♠", "PLAY 4♥", "PLAY Q♠", "PLAY 3♦", "PLAY K♦", "PLAY 3♥", "PLAY Q♦")
  assert m.scores["A"] == 16 and m.is_terminal() and m.winner == "A"
  assert m.current_actor() == -4
  assert m.legal_actions(0) == []
  with pytest.raises(IllegalActionError):
    m.apply_action(0, "FOLD")


def test_talk_is_recorded_public_and_truncated():
  m = TrucoMatch(0, scripted_deals=[D])
  m.apply_action(0, "PLAY 3♠", talk="I have the club manilha!")
  assert m.observation(3)["talk"] == [
      {"hand": 0, "trick": 0, "seat": 0, "text": "I have the club manilha!"}
  ]
  m.apply_action(1, "PLAY K♠", talk="x" * 250)
  assert len(m.talk[-1]["text"]) == 200
  assert any(e["type"] == "talk_truncated" and e["original_length"] == 250 for e in m.events)
  m.apply_action(2, "PLAY 4♥", talk="   ")
  assert len(m.talk) == 2
