import itertools

import pytest

from truco import cards as C


@pytest.mark.parametrize("vira_rank,manilha", [
    ("4", "5"), ("5", "6"), ("6", "7"), ("7", "Q"), ("Q", "J"),
    ("J", "K"), ("K", "A"), ("A", "2"), ("2", "3"), ("3", "4"),
])
def test_manilha_for_every_vira_rank(vira_rank, manilha):
  for suit in C.SUITS:
    assert C.manilha_rank(vira_rank + suit) == manilha


@pytest.mark.parametrize("vira_rank", C.RANKS)
def test_manilha_suit_order_clubs_hearts_spades_diamonds(vira_rank):
  vira = vira_rank + "♦"
  mr = C.manilha_rank(vira)
  order = [mr + s for s in ("♣", "♥", "♠", "♦")]
  strengths = [C.strength(c, vira) for c in order]
  assert strengths == sorted(strengths, reverse=True)
  assert len(set(strengths)) == 4
  # Every manilha beats every non-manilha card.
  weakest_manilha = C.strength(mr + "♦", vira)
  for card in C.make_deck():
    if C.rank(card) != mr:
      assert C.strength(card, vira) < weakest_manilha


def test_ordinary_order_including_J_over_Q():
  vira = "4♦"  # manilha = 5; the ordinary list is 3 2 A K J Q 7 6 (no 5) 4
  expected = ["3", "2", "A", "K", "J", "Q", "7", "6", "4"]
  strengths = [C.strength(r + "♠", vira) for r in expected]
  assert strengths == sorted(strengths, reverse=True)
  assert C.strength("J♦", vira) > C.strength("Q♣", vira)


def test_suits_irrelevant_for_ordinary_cards_equal_ranks_tie():
  vira = "7♥"  # manilha = Q
  for r in C.RANKS:
    if r == "Q":
      continue
    vals = {C.strength(r + s, vira) for s in C.SUITS}
    assert len(vals) == 1


def test_manilha_rank_removed_from_ordinary_list():
  vira = "K♠"  # manilha = A
  order = C.describe_strength_order(vira)
  assert order.startswith("A manilhas (suit order ♣ > ♥ > ♠ > ♦) > 3 > 2 > K")
  assert " A " not in order


def test_full_order_example_vira_7():
  assert C.describe_strength_order("7♦") == (
      "Q manilhas (suit order ♣ > ♥ > ♠ > ♦) > 3 > 2 > A > K > J > 7 > 6 > 5 > 4"
  )


def test_deck_has_40_unique_cards():
  deck = C.make_deck()
  assert len(deck) == 40 and len(set(deck)) == 40
  for r, s in itertools.product(C.RANKS, C.SUITS):
    assert r + s in deck
