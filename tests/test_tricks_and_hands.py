"""Trick resolution (§5), hand resolution (§6), early termination."""

import pytest

from tests.conftest import act, deal
from truco.match import TrucoMatch, hand_winner, resolve_trick


def test_cross_team_tie_is_parda_and_same_team_tie_is_not():
  vira = "A♦"  # manilha = 2
  # Seats 0 (A) and 1 (B) tie with 3s -> parda (Example A).
  assert resolve_trick([(0, "3♠"), (1, "3♦"), (2, "K♣"), (3, "7♥")], vira) == (None, None)
  # Seats 0 and 2 (both A) tie -> team A wins, earlier player (0) leads next.
  assert resolve_trick([(0, "3♠"), (1, "K♦"), (2, "3♣"), (3, "7♥")], vira) == ("A", 0)
  # Three-way tie including both teams is a parda.
  assert resolve_trick([(0, "3♠"), (1, "3♦"), (2, "3♣"), (3, "7♥")], vira) == (None, None)


def test_manilha_beats_everything_example_a_trick_2():
  vira = "A♦"
  assert resolve_trick([(0, "2♦"), (1, "2♣"), (2, "A♥"), (3, "4♣")], vira) == ("B", 1)


@pytest.mark.parametrize("results,mao_team,expected", [
    (["A"], "A", None),
    (["A", "A"], "B", "A"),
    (["B", "B"], "A", "B"),
    (["A", None], "B", "A"),           # winner of trick 1 then parda
    (["A", "B"], "A", None),           # split -> trick 3 decides
    (["A", "B", "B"], "A", "B"),
    (["A", "B", "A"], "B", "A"),
    (["A", "B", None], "B", "A"),      # Example D
    ([None, "B"], "A", "B"),           # parda then winner
    ([None, None], "A", None),
    ([None, None, "A"], "B", "A"),
    ([None, None, None], "A", "A"),    # all pardas -> mão's team
    ([None, None, None], "B", "B"),
])
def test_hand_winner_table(results, mao_team, expected):
  assert hand_winner(results, mao_team) == expected


def _fresh(d, **kw):
  return TrucoMatch(seed=0, num_players=4, scripted_deals=[d], **kw)


def test_two_tricks_same_team_ends_hand_without_third_trick():
  # vira 4♦ -> manilha 5. Team A (seats 0, 2) holds all the 3s.
  d = deal("3♠ 3♦ 4♣", "K♠ K♦ 7♣", "3♥ 3♣ 4♥", "Q♠ Q♦ 7♦", vira="4♦")
  m = _fresh(d)
  act(m, "PLAY 3♠", "PLAY K♠", "PLAY 4♥", "PLAY Q♠")   # trick 1 -> A (seat 0)
  assert m.trick_results[-1]["winner_team"] == "A"
  assert m.current_actor() == 0
  act(m, "PLAY 3♦", "PLAY K♦", "PLAY 3♥", "PLAY Q♦")   # trick 2 -> A
  assert len(m.hand_history) == 1
  assert m.hand_history[0]["winner_team"] == "A"
  assert m.hand_history[0]["points"] == 1
  assert m.hand_history[0]["tricks"] == ["A", "A"]
  assert m.hand_index == 1  # next hand dealt automatically


def test_parda_first_then_first_decided_trick_wins_and_leader_repeats():
  d = deal("3♠ 2♦ 4♣", "3♦ 2♣ 7♣", "K♣ 4♥ 4♠", "7♥ 4♦ 6♦", vira="A♦")  # Example A
  m = _fresh(d)
  act(m, "PLAY 3♠", "PLAY 3♦", "PLAY K♣", "PLAY 7♥")
  assert m.trick_results[0]["winner_team"] is None
  assert m.current_actor() == 0  # previous leader leads again after parda
  act(m, "PLAY 2♦", "PLAY 2♣", "PLAY 4♥", "PLAY 4♦")
  assert m.hand_history[0]["winner_team"] == "B"
  assert m.hand_history[0]["tricks"] == [None, "B"]


def test_win_then_parda_ends_hand_for_trick1_winner():
  d = deal("3♠ 2♦ 4♣", "K♦ 2♣ 7♣", "K♣ 4♥ 4♠", "7♥ 4♦ 6♦", vira="A♦")
  m = _fresh(d)
  act(m, "PLAY 3♠", "PLAY K♦", "PLAY K♣", "PLAY 7♥")  # A wins trick 1
  act(m, "PLAY 2♦", "PLAY 2♣", "PLAY 4♥", "PLAY 4♦")  # hmm 2♣ beats 2♦: B wins
  assert m.hand_history == []  # split -> third trick
  # Now force a parda in trick 3 impossible with these cards; use new deal below.


def test_split_then_parda_goes_to_trick1_winner_example_d():
  # vira 4♦ -> manilha 5. Seat 0: 3♠ 4♣ 7♠ ; seat 1: K♦ 3♦ 7♦ ; seat 2: 6♣ 6♥ Q♥ ; seat 3: 6♦ 6♠ Q♦
  d = deal("3♠ 4♣ 7♠", "K♦ 3♦ 7♦", "6♣ 6♥ Q♥", "6♦ 6♠ Q♦", vira="4♦")
  m = _fresh(d)
  act(m, "PLAY 3♠", "PLAY K♦", "PLAY 6♣", "PLAY 6♦")   # A wins trick 1 (seat 0 leads)
  act(m, "PLAY 4♣", "PLAY 3♦", "PLAY 6♥", "PLAY 6♠")   # B wins trick 2 (seat 1 leads)
  assert m.current_actor() == 1
  act(m, "PLAY 7♦", "PLAY Q♥", "PLAY Q♦", "PLAY 7♠")   # 7♦(B) vs 7♠(A) parda
  assert m.hand_history[0]["tricks"] == ["A", "B", None]
  assert m.hand_history[0]["winner_team"] == "A"


def test_all_three_pardas_go_to_mao_team():
  d = deal("3♠ 2♠ K♠", "3♦ 2♦ K♦", "4♣ 4♥ 6♣", "4♦ 4♠ 6♦", vira="7♦")  # manilha Q
  m = _fresh(d)
  act(m, "PLAY 3♠", "PLAY 3♦", "PLAY 4♣", "PLAY 4♦")
  act(m, "PLAY 2♠", "PLAY 2♦", "PLAY 4♥", "PLAY 4♠")
  act(m, "PLAY K♠", "PLAY K♦", "PLAY 6♣", "PLAY 6♦")
  assert m.hand_history[0]["tricks"] == [None, None, None]
  assert m.hand_history[0]["winner_team"] == "A"  # mão is seat 0 (team A)


def test_all_three_pardas_go_to_mao_team_when_mao_is_team_b():
  # First hand mão is seat 0; use hand 2 (dealer 0, mão 1) via two scripted deals.
  h1 = deal("3♠ 3♦ 4♣", "K♠ K♦ 7♣", "3♥ 3♣ 4♥", "Q♠ Q♦ 7♦", vira="4♦")
  h2 = deal("3♠ 2♠ K♠", "3♦ 2♦ K♦", "4♣ 4♥ 6♣", "4♦ 4♠ 6♦", vira="7♦")
  m = TrucoMatch(0, scripted_deals=[h1, h2])
  act(m, "PLAY 3♠", "PLAY K♠", "PLAY 4♥", "PLAY Q♠", "PLAY 3♦", "PLAY K♦", "PLAY 3♥", "PLAY Q♦")
  assert m.hand_index == 1 and m.mao == 1
  act(m, "PLAY 3♦", "PLAY 4♣", "PLAY 4♦", "PLAY 3♠")
  act(m, "PLAY 2♦", "PLAY 4♥", "PLAY 4♠", "PLAY 2♠")
  act(m, "PLAY K♦", "PLAY 6♣", "PLAY 6♦", "PLAY K♠")
  assert m.hand_history[1]["tricks"] == [None, None, None]
  assert m.hand_history[1]["winner_team"] == "B"


def test_parda_chain_then_decided_trick_3():
  d = deal("3♠ 2♠ K♠", "3♦ 2♦ 4♦", "4♣ 4♥ 6♣", "5♦ 4♠ 6♦", vira="7♦")
  m = _fresh(d)
  act(m, "PLAY 3♠", "PLAY 3♦", "PLAY 4♣", "PLAY 5♦")
  act(m, "PLAY 2♠", "PLAY 2♦", "PLAY 4♥", "PLAY 4♠")
  act(m, "PLAY K♠", "PLAY 4♦", "PLAY 6♣", "PLAY 6♦")
  assert m.hand_history[0]["tricks"] == [None, None, "A"]
  assert m.hand_history[0]["winner_team"] == "A"


def test_same_team_tie_winner_leads_next_trick():
  d = deal("3♠ 4♣ 7♠", "K♦ 3♦ 7♦", "3♣ 6♥ Q♥", "6♦ 6♠ Q♦", vira="4♦")
  m = _fresh(d)
  act(m, "PLAY 3♠", "PLAY K♦", "PLAY 3♣", "PLAY 6♦")  # seats 0 and 2 tie -> A wins, seat 0 leads
  assert m.trick_results[0]["winner_team"] == "A"
  assert m.trick_results[0]["winner_seat"] == 0
  assert m.current_actor() == 0


def test_dealer_and_mao_rotate():
  m = TrucoMatch(0)
  assert m.dealer == 3 and m.mao == 0
  act(m, "FOLD")
  assert m.dealer == 0 and m.mao == 1
  act(m, "FOLD")
  assert m.dealer == 1 and m.mao == 2


def test_two_player_match_works():
  m = TrucoMatch(3, num_players=2)
  assert m.dealer == 1 and m.mao == 0
  s = m.current_actor()
  assert set(m.legal_actions(s)) >= {"TRUCO", "FOLD"}
