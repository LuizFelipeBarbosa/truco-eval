"""Deterministic rule-based baseline bot ("heuristic").

Uses only the seat's observation and legal actions; never the engine, never talk,
never the fallback stream. Thresholds are versioned: changing any of them changes
the rating anchor of every result that includes this bot.

Definitions
  s(c)       truco strength of card c under the current vira.
  v(c)       card value: manilha 3; else rank "3" 2, "2" 1.5, "A" 1, "K" 0.5, others 0.
  hand_score sum(v(c) for c in hand) + 2 * tricks my team won - 2 * tricks opponents won
             (a parda counts 0).
  steps(x)   ladder steps above truco: {3: 0, 6: 1, 9: 2, 12: 3}.

Rules
  A. MAO_DE_ONZE: team_cards = own hand + partner hand(s). MAO_PLAY iff
     sum(v) >= 2.0 * len(team_cards) / 3, else MAO_FORFEIT.
  B. RAISE_RESPONSE (call value V, pre-call stake P):
     - ACCEPT if opponent score + P >= target (declining would lose the match).
     - T = 2.5 + 0.5 * steps(V). RAISE if legal and hand_score >= T + 2.5;
       ACCEPT if hand_score >= T; otherwise DECLINE.
  C. PLAY in a mao de ferro (blind hand): play the lowest blind position.
  D. PLAY otherwise:
     1. If a raise R is legal and hand_score >= 4.0 + 1.0 * steps(value of R): call R.
     2. Otherwise play a card (cards ordered by (strength, hand index) ascending):
        - Not leading: let best be the strongest card on the table.
          * every seat holding a best card is my team -> play my lowest card;
          * else if I can beat best -> play the lowest card that beats it;
          * else if I hold a card tying best, it is trick 2 or 3, and my team won
            trick 1 -> play that tying card (aiming for a parda, which my team wins);
          * else play my lowest card.
        - Leading: trick 1 with 3 cards -> middle card; otherwise the strongest.
     3. Never FOLD voluntarily.
"""

from __future__ import annotations

from typing import Any, Sequence

from truco import cards as C
from truco.match import RAISE_VALUE_BY_NAME

from runner.agents import Decision

_RANK_VALUE = {"3": 2.0, "2": 1.5, "A": 1.0, "K": 0.5}
_STEPS = {3: 0, 6: 1, 9: 2, 12: 3}


def _card_value(card: str, manilha_rank: str) -> float:
  rank = C.rank(card)
  if rank == manilha_rank:
    return 3.0
  return _RANK_VALUE.get(rank, 0.0)


def _tricks_won_lost(obs: dict[str, Any]) -> tuple[int, int]:
  won = lost = 0
  for r in obs["trick_results"]:
    w = r["winner_team"]
    if w is None:
      continue
    if w == obs["team"]:
      won += 1
    else:
      lost += 1
  return won, lost


def _hand_score(obs: dict[str, Any]) -> float:
  won, lost = _tricks_won_lost(obs)
  base = sum(_card_value(c, obs["manilha_rank"]) for c in obs["hand"])
  return base + 2 * won - 2 * lost


class HeuristicBotAgent:
  """Fixed-threshold rule-based player (see module docstring for the spec)."""

  name = "heuristic"

  def decide(self, observation, legal_actions, fallback_fn) -> Decision:
    action = self._choose(observation, list(legal_actions))
    assert action in legal_actions, (action, legal_actions)
    return Decision(action=action, source="heuristic_bot")

  def _choose(self, obs: dict[str, Any], legal: Sequence[str]) -> str:
    phase = obs["phase"]
    if phase == "MAO_DE_ONZE":
      return self._mao_de_onze(obs)
    if phase == "RAISE_RESPONSE":
      return self._raise_response(obs, legal)
    if obs["hand_type"] == "mao_de_ferro":
      return f"PLAY {min(obs['blind_positions'])}"
    return self._play(obs, legal)

  def _mao_de_onze(self, obs: dict[str, Any]) -> str:
    team_cards = list(obs["hand"])
    for cards in obs["partner_hand"].values():
      team_cards.extend(cards)
    total = sum(_card_value(c, obs["manilha_rank"]) for c in team_cards)
    return "MAO_PLAY" if total >= 2.0 * len(team_cards) / 3 else "MAO_FORFEIT"

  def _raise_response(self, obs: dict[str, Any], legal: Sequence[str]) -> str:
    call = obs["pending_call"]
    opp = "B" if obs["team"] == "A" else "A"
    if obs["scores"][opp] + call["pre_call_stake"] >= obs["target"]:
      return "ACCEPT"
    threshold = 2.5 + 0.5 * _STEPS[call["value"]]
    score = _hand_score(obs)
    if "RAISE" in legal and score >= threshold + 2.5:
      return "RAISE"
    if score >= threshold:
      return "ACCEPT"
    return "DECLINE"

  def _play(self, obs: dict[str, Any], legal: Sequence[str]) -> str:
    score = _hand_score(obs)
    for name, value in RAISE_VALUE_BY_NAME.items():
      if name in legal and score >= 4.0 + 1.0 * _STEPS[value]:
        return name
    return f"PLAY {self._card(obs)}"

  def _card(self, obs: dict[str, Any]) -> str:
    vira = obs["vira"]
    hand = obs["hand"]
    ordered = sorted(
        ((C.strength(c, vira), i, c) for i, c in enumerate(hand)),
        key=lambda t: (t[0], t[1]),
    )
    trick = obs["current_trick"]
    if not trick:
      if obs["trick_index"] == 0 and len(hand) == 3:
        return ordered[1][2]
      return ordered[-1][2]

    table = [(p["seat"], C.strength(p["card"], vira)) for p in trick]
    best = max(st for _, st in table)
    my_parity = obs["seat"] % 2
    if all(seat % 2 == my_parity for seat, st in table if st == best):
      return ordered[0][2]
    for st, _, c in ordered:
      if st > best:
        return c
    results = obs["trick_results"]
    if obs["trick_index"] >= 1 and results and results[0]["winner_team"] == obs["team"]:
      for st, _, c in ordered:
        if st == best:
          return c
    return ordered[0][2]
