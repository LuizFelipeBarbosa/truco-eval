"""Ladder mechanics (§7): calls, responses, raise-backs, raise right."""

import pytest

from tests.conftest import act, deal
from truco.match import IllegalActionError, TrucoMatch

D = deal("3♠ 3♦ 4♣", "K♠ K♦ 7♣", "3♥ 3♣ 4♥", "Q♠ Q♦ 7♦", vira="4♦")


def fresh(**kw):
  return TrucoMatch(seed=0, scripted_deals=[D], **kw)


def test_initial_legal_actions_include_truco_and_fold():
  m = fresh()
  assert m.legal_actions(0) == ["PLAY 3♠", "PLAY 3♦", "PLAY 4♣", "TRUCO", "FOLD"]
  assert m.legal_actions(1) == []


def test_decline_scores_pre_call_stake_example_b():
  m = fresh()
  act(m, "PLAY 3♠", "PLAY K♠")
  assert m.current_actor() == 2
  act(m, "TRUCO")
  assert m.current_actor() == 3
  assert m.legal_actions(3) == ["ACCEPT", "DECLINE", "RAISE"]
  act(m, "DECLINE")
  assert m.hand_history[0]["winner_team"] == "A"
  assert m.hand_history[0]["points"] == 1
  assert m.hand_history[0]["reason"] == "decline"
  assert m.scores == {"A": 1, "B": 0}


def test_accept_sets_stake_and_transfers_raise_right():
  m = fresh()
  act(m, "TRUCO")
  assert m.pending_call["name"] == "TRUCO" and m.pending_call["value"] == 3
  assert m.stake == 1
  act(m, "ACCEPT")
  assert m.stake == 3
  assert m.raise_right == {"B"}
  assert m.current_actor() == 0  # caller resumes
  assert "SEIS" not in m.legal_actions(0)
  act(m, "PLAY 3♠")
  assert "SEIS" in m.legal_actions(1)


def test_raise_back_chain_example_c():
  m = fresh()
  act(m, "PLAY 3♠")            # seat 0 plays
  act(m, "TRUCO")              # seat 1 (B) calls Truco
  assert m.current_actor() == 2
  act(m, "RAISE")              # seat 2 answers Seis
  assert m.pending_call["name"] == "SEIS" and m.pending_call["value"] == 6
  assert m.pending_call["pre_call_stake"] == 3
  assert m.current_actor() == 3
  act(m, "ACCEPT")
  assert m.stake == 6
  assert m.raise_right == {"B"}
  assert m.current_actor() == 1  # seat 1 plays their card
  act(m, "PLAY K♠", "PLAY 4♥")
  assert "NOVE" in m.legal_actions(3)
  act(m, "PLAY Q♠")
  # Trick 1 won by A (3♠); seat 0 leads and may not raise.
  assert m.current_actor() == 0
  assert "NOVE" not in m.legal_actions(0)


def test_decline_of_raise_back_scores_implicitly_accepted_stake():
  m = fresh()
  act(m, "PLAY 3♠", "TRUCO", "RAISE")   # B calls Truco, A raises Seis
  act(m, "DECLINE")                    # seat 3 declines Seis
  assert m.hand_history[0]["winner_team"] == "A"
  assert m.hand_history[0]["points"] == 3


def test_full_ladder_to_doze_and_no_raises_at_12():
  m = fresh()
  act(m, "TRUCO", "RAISE", "RAISE", "RAISE")  # 3 -> 6 -> 9 -> 12 pending
  assert m.pending_call["name"] == "DOZE"
  assert m.legal_actions(m.current_actor()) == ["ACCEPT", "DECLINE"]
  act(m, "ACCEPT")
  assert m.stake == 12
  assert not m.raises_allowed
  for seat in range(4):
    assert not any(a in ("TRUCO", "SEIS", "NOVE", "DOZE") for a in m.legal_actions(seat))
  act(m, "PLAY 3♠")
  assert m.legal_actions(1) == ["PLAY K♠", "PLAY K♦", "PLAY 7♣", "FOLD"]


def test_decline_scores_at_each_ladder_step():
  for chain, expected_points, expected_winner in [
      (["TRUCO"], 1, "A"),
      (["TRUCO", "RAISE"], 3, "B"),
      (["TRUCO", "RAISE", "RAISE"], 6, "A"),
      (["TRUCO", "RAISE", "RAISE", "RAISE"], 9, "B"),
  ]:
    m = fresh()
    act(m, *chain)
    act(m, "DECLINE")
    assert m.hand_history[0]["points"] == expected_points
    assert m.hand_history[0]["winner_team"] == expected_winner


def test_raise_right_prohibition_until_opponents_raise():
  m = fresh()
  act(m, "TRUCO", "ACCEPT")           # A called, B accepted -> only B may raise
  act(m, "PLAY 3♠")
  assert "SEIS" in m.legal_actions(1)
  act(m, "PLAY K♠")
  assert "SEIS" not in m.legal_actions(2)
  act(m, "PLAY 4♥")
  act(m, "SEIS", "ACCEPT")            # B raises, A accepts -> only A may raise
  assert m.raise_right == {"A"}
  assert m.current_actor() == 3
  assert "NOVE" not in m.legal_actions(3)


def test_raise_only_before_playing_card_in_turn():
  m = fresh()
  act(m, "PLAY 3♠")
  # Seat 0 has played; it is seat 1's turn. Seat 0 cannot act.
  with pytest.raises(IllegalActionError):
    m.apply_action(0, "TRUCO")


def test_raise_allowed_in_trick_3():
  d = deal("3♠ 4♣ 7♠", "K♦ 3♦ 7♦", "6♣ 6♥ Q♥", "6♦ 6♠ Q♦", vira="4♦")
  m = TrucoMatch(0, scripted_deals=[d])
  act(m, "PLAY 3♠", "PLAY K♦", "PLAY 6♣", "PLAY 6♦")
  act(m, "PLAY 4♣", "PLAY 3♦", "PLAY 6♥", "PLAY 6♠")
  assert m.trick_index == 2
  assert "TRUCO" in m.legal_actions(1)
  act(m, "TRUCO", "ACCEPT")
  assert m.stake == 3
  act(m, "PLAY 7♦", "PLAY Q♥", "PLAY Q♦", "PLAY 7♠")
  assert m.hand_history[0]["points"] == 3


def test_illegal_actions_rejected():
  m = fresh()
  with pytest.raises(IllegalActionError):
    m.apply_action(0, "PLAY K♠")   # not in hand
  with pytest.raises(IllegalActionError):
    m.apply_action(1, "PLAY K♠")   # not your turn
  with pytest.raises(IllegalActionError):
    m.apply_action(0, "SEIS")      # wrong ladder step
  with pytest.raises(IllegalActionError):
    m.apply_action(0, "ACCEPT")    # nothing pending
  act(m, "TRUCO")
  with pytest.raises(IllegalActionError):
    m.apply_action(1, "PLAY K♠")   # must respond first
  with pytest.raises(IllegalActionError):
    m.apply_action(1, "FOLD")      # DECLINE is the fold while responding


def test_hand_winner_scores_current_stake():
  m = fresh()
  act(m, "TRUCO", "ACCEPT")
  act(m, "PLAY 3♠", "PLAY K♠", "PLAY 4♥", "PLAY Q♠", "PLAY 3♦", "PLAY K♦", "PLAY 3♥", "PLAY Q♦")
  assert m.scores == {"A": 3, "B": 0}


def test_raise_responder_is_next_seat_with_six_players():
  m = TrucoMatch(5, num_players=6)
  s = m.current_actor()
  act(m, "TRUCO")
  assert m.current_actor() == (s + 1) % 6
  act(m, "RAISE")
  assert m.current_actor() == (s + 2) % 6
  act(m, "ACCEPT")
  assert m.current_actor() == s
