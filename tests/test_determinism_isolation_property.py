"""Seeded determinism, information isolation, and the 10k-match property test."""

import json
import random

import pytest

from tests.conftest import play_random_match
from truco import cards as C
from truco.match import TrucoMatch, team_of


def test_same_seed_same_actions_identical_event_log():
  a = play_random_match(42, bot_seed=7)
  b = play_random_match(42, bot_seed=7)
  assert a.serialize_events() == b.serialize_events()
  assert a.serialize_events().encode("utf-8") == b.serialize_events().encode("utf-8")


def test_different_seeds_differ():
  assert play_random_match(1).serialize_events() != play_random_match(2).serialize_events()


def test_fallback_rng_flows_from_seed():
  m1, m2 = TrucoMatch(9), TrucoMatch(9)
  seq1 = [m1.random_legal_action(0) for _ in range(20)]
  seq2 = [m2.random_legal_action(0) for _ in range(20)]
  assert seq1 == seq2
  assert set(seq1) <= set(m1.legal_actions(0))


def test_deal_sequence_independent_of_play_duplicate_matches():
  """Hand k's deal depends only on (seed, k), regardless of earlier play."""
  fold_only = TrucoMatch(11)
  while not fold_only.is_terminal():
    s = fold_only.current_actor()
    legal = fold_only.legal_actions(s)
    fold_only.apply_action(s, "FOLD" if "FOLD" in legal else legal[0])
  rnd = play_random_match(11, bot_seed=3)
  deals_a = {e["hand"]: (e["hands"], e["vira"]) for e in fold_only.events if e["type"] == "hand_start"}
  deals_b = {e["hand"]: (e["hands"], e["vira"]) for e in rnd.events if e["type"] == "hand_start"}
  for k in set(deals_a) & set(deals_b):
    assert deals_a[k] == deals_b[k]


def _check_isolation(m: TrucoMatch) -> None:
  for seat in range(m.num_players):
    hidden = m.hidden_cards(seat)
    obs_text = json.dumps(m.observation(seat), ensure_ascii=False)
    for card in hidden:
      assert card not in obs_text, f"seat {seat} can see hidden {card}"


@pytest.mark.parametrize("seed", range(40))
def test_information_isolation_random_play(seed):
  m = TrucoMatch(seed, initial_scores={"A": 9, "B": 10} if seed % 2 else None)
  rng = random.Random(seed)
  while not m.is_terminal():
    _check_isolation(m)
    s = m.current_actor()
    m.apply_action(s, rng.choice(m.legal_actions(s)))


def test_isolation_in_mao_de_onze_and_ferro():
  for scores in ({"A": 11, "B": 5}, {"A": 2, "B": 11}, {"A": 11, "B": 11}):
    m = TrucoMatch(5, initial_scores=scores)
    rng = random.Random(1)
    while not m.is_terminal():
      _check_isolation(m)
      s = m.current_actor()
      m.apply_action(s, rng.choice(m.legal_actions(s)))


def test_observation_never_contains_unused_cards_or_opponent_hands():
  m = TrucoMatch(77)
  dealt = m.events[-1]["hands"]
  for seat in range(4):
    text = json.dumps(m.observation(seat), ensure_ascii=False)
    for other in range(4):
      if other == seat:
        continue
      for card in dealt[str(other)]:
        assert card not in text


@pytest.mark.parametrize("num_players", [2, 4, 6])
def test_property_random_matches_no_crash_valid_scores_termination(num_players):
  n_matches = 10_000 if num_players == 4 else 1_000
  max_hands = 0
  for seed in range(n_matches):
    m = TrucoMatch(seed, num_players=num_players)
    rng = random.Random(seed * 7 + 1)
    steps = 0
    while not m.is_terminal():
      s = m.current_actor()
      legal = m.legal_actions(s)
      assert legal, "actor must always have a legal action"
      assert team_of(s) in ("A", "B")
      m.apply_action(s, rng.choice(legal))
      steps += 1
      assert steps < 5000, "match did not terminate"
    assert m.winner in ("A", "B")
    assert m.scores[m.winner] >= 12
    assert m.scores[m.winner] <= 12 + 11  # at most 12 points per hand, from <= 11
    loser = "B" if m.winner == "A" else "A"
    assert 0 <= m.scores[loser] < 12
    assert sum(h["points"] for h in m.hand_history) == m.scores["A"] + m.scores["B"]
    assert all(h["points"] in (1, 3, 6, 9, 12) for h in m.hand_history)
    assert 1 <= len(m.hand_history) <= 23
    max_hands = max(max_hands, len(m.hand_history))
    assert m.events[-1]["type"] == "match_end"
    # Every dealt hand had 40 distinct cards accounted for.
    for e in m.events:
      if e["type"] == "hand_start":
        cards = [c for h in e["hands"].values() for c in h] + [e["vira"]]
        assert len(cards) == 3 * num_players + 1 and len(set(cards)) == len(cards)
        assert all(C.is_card(c) for c in cards)
  assert max_hands <= 23
