"""Shared helpers for engine tests."""

from __future__ import annotations

import random

from truco.match import Deal, TrucoMatch


def deal(*hands: str, vira: str) -> Deal:
  """Build a Deal from strings like "3♠ 2♦ K♣" (one per seat)."""
  return Deal(hands=tuple(tuple(h.split()) for h in hands), vira=vira)


def play_random_match(seed: int, num_players: int = 4, bot_seed: int | None = None,
                      **kwargs) -> TrucoMatch:
  """Play a full match with a uniformly random bot; returns the finished match."""
  m = TrucoMatch(seed, num_players, **kwargs)
  rng = random.Random(seed if bot_seed is None else bot_seed)
  while not m.is_terminal():
    s = m.current_actor()
    m.apply_action(s, rng.choice(m.legal_actions(s)))
  return m


def act(m: TrucoMatch, *actions: str, talk: str | None = None) -> None:
  """Apply actions in order for whichever seat is to act."""
  for a in actions:
    m.apply_action(m.current_actor(), a, talk=talk)
