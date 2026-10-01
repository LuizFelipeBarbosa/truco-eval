"""Cards, manilhas and card strength for Truco Paulista."""

from __future__ import annotations

# Cyclic rank order (§4.1). It is also the ordinary strength order (§4.2):
# 4 < 5 < 6 < 7 < Q < J < K < A < 2 < 3.
RANKS: tuple[str, ...] = ("4", "5", "6", "7", "Q", "J", "K", "A", "2", "3")

# Suits in ascending manilha strength: ♦ < ♠ < ♥ < ♣ (§4.1).
SUITS: tuple[str, ...] = ("♦", "♠", "♥", "♣")

SUIT_NAMES = {"♣": "clubs", "♥": "hearts", "♠": "spades", "♦": "diamonds"}
SUIT_ASCII = {"♣": "C", "♥": "H", "♠": "S", "♦": "D"}

_MANILHA_BASE = 100


def make_deck() -> list[str]:
  """The 40-card deck in a fixed, deterministic order."""
  return [r + s for r in RANKS for s in SUITS]


def rank(card: str) -> str:
  return card[:-1]


def suit(card: str) -> str:
  return card[-1]


def is_card(text: str) -> bool:
  return len(text) == 2 and text[0] in RANKS and text[1] in SUITS


def manilha_rank(vira: str) -> str:
  """Rank immediately after the vira's rank in the cycle (§4.1)."""
  return RANKS[(RANKS.index(rank(vira)) + 1) % len(RANKS)]


def strength(card: str, vira: str) -> int:
  """Numeric strength of ``card`` given the ``vira``. Higher is stronger.

  Manilhas are strictly ordered by suit and beat everything else. Ordinary
  cards compare by rank only, so equal ranks give equal strength (a tie).
  """
  if rank(card) == manilha_rank(vira):
    return _MANILHA_BASE + SUITS.index(suit(card))
  return RANKS.index(rank(card))


def describe_strength_order(vira: str) -> str:
  """Human-readable full order for the hand, e.g. for prompts (§4.3)."""
  mr = manilha_rank(vira)
  manilhas = f"{mr} manilhas (suit order ♣ > ♥ > ♠ > ♦)"
  ordinary = [r for r in reversed(RANKS) if r != mr]
  return " > ".join([manilhas] + ordinary)
