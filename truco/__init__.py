"""Standalone deterministic Truco Paulista engine.

Pure Python. No LLM or harness imports. See ``truco.match.TrucoMatch``.
"""

from truco.cards import (
    RANKS,
    SUITS,
    make_deck,
    manilha_rank,
    strength,
)
from truco.match import (
    TERMINAL_PLAYER,
    Deal,
    IllegalActionError,
    TrucoMatch,
    hand_winner,
    resolve_trick,
    team_of,
)

__all__ = [
    "RANKS",
    "SUITS",
    "make_deck",
    "manilha_rank",
    "strength",
    "TERMINAL_PLAYER",
    "Deal",
    "IllegalActionError",
    "TrucoMatch",
    "hand_winner",
    "resolve_trick",
    "team_of",
]
