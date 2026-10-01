"""Truco Paulista match engine with an OpenSpiel-flavored interface.

One ``TrucoMatch`` object is one full match. Hands are sub-episodes inside it;
the match ends when a team reaches ``target`` (12) points at the end of a hand.

All randomness flows from the constructor ``seed``:
  * the deal of hand *k* is drawn from ``Random(f"truco:{seed}:hand:{k}")`` so
    the deal sequence never depends on how earlier hands were played
    (duplicate matches), and
  * the uniformly random legal-action fallback is drawn from
    ``Random(f"truco:{seed}:fallback")``.

The event log is a list of plain dicts with no timestamps; the same seed and
the same action sequence produce a byte-identical serialized log.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import random
from typing import Any, Iterable, Sequence

from truco import cards as C

# OpenSpiel's PlayerId.TERMINAL value.
TERMINAL_PLAYER = -4

TEAMS = ("A", "B")
LADDER = (1, 3, 6, 9, 12)
RAISE_NAME_BY_VALUE = {3: "TRUCO", 6: "SEIS", 9: "NOVE", 12: "DOZE"}
RAISE_VALUE_BY_NAME = {v: k for k, v in RAISE_NAME_BY_VALUE.items()}
TALK_MAX_CHARS = 200

PHASE_MAO_DE_ONZE = "MAO_DE_ONZE"
PHASE_PLAY = "PLAY"
PHASE_RAISE_RESPONSE = "RAISE_RESPONSE"
PHASE_TERMINAL = "TERMINAL"

HAND_NORMAL = "normal"
HAND_MAO_DE_ONZE = "mao_de_onze"
HAND_MAO_DE_FERRO = "mao_de_ferro"


class IllegalActionError(ValueError):
  """Raised when ``apply_action`` is given a seat/action that is not legal."""


def team_of(seat: int) -> str:
  return TEAMS[seat % 2]


def other_team(team: str) -> str:
  return "B" if team == "A" else "A"


def next_stake(stake: int) -> int | None:
  """Next ladder step above ``stake``, or None at 12."""
  idx = LADDER.index(stake)
  if idx + 1 >= len(LADDER):
    return None
  return LADDER[idx + 1]


@dataclasses.dataclass(frozen=True)
class Deal:
  """A scripted deal: ``hands[seat]`` is that seat's three cards, in order."""

  hands: tuple[tuple[str, str, str], ...]
  vira: str

  def validate(self, num_players: int) -> None:
    if len(self.hands) != num_players:
      raise ValueError(f"Deal has {len(self.hands)} hands, expected {num_players}")
    seen: set[str] = set()
    for hand in self.hands:
      if len(hand) != 3:
        raise ValueError("Each hand must have exactly 3 cards")
      for card in hand:
        if not C.is_card(card):
          raise ValueError(f"Invalid card {card!r}")
        if card in seen:
          raise ValueError(f"Duplicate card {card}")
        seen.add(card)
    if not C.is_card(self.vira) or self.vira in seen:
      raise ValueError(f"Invalid vira {self.vira!r}")


def resolve_trick(
    plays: Sequence[tuple[int, str]], vira: str
) -> tuple[str | None, int | None]:
  """Resolve one trick (§5).

  Args:
    plays: (seat, card) pairs in play order.
    vira: the vira card.

  Returns:
    (winning_team, winning_seat). ``(None, None)`` for a parda. On a same-team
    tie for the strongest card the team wins and the earlier of the tied
    players is the winning seat (AMBIGUITIES #7).
  """
  best = max(C.strength(card, vira) for _, card in plays)
  top = [(seat, card) for seat, card in plays if C.strength(card, vira) == best]
  teams = {team_of(seat) for seat, _ in top}
  if len(teams) > 1:
    return None, None
  seat, _ = top[0]
  return team_of(seat), seat


def hand_winner(results: Sequence[str | None], mao_team: str) -> str | None:
  """Winner of the hand given trick results so far, or None if undecided (§6).

  ``results[i]`` is the team that won trick i, or None for a parda.
  """
  n = len(results)
  if n < 2:
    return None
  r0 = results[0]
  if r0 is None:
    for later in results[1:]:
      if later is not None:
        return later
    return mao_team if n == 3 else None
  r1 = results[1]
  if r1 is None or r1 == r0:
    return r0
  if n == 3:
    r2 = results[2]
    return r2 if r2 is not None else r0
  return None


class TrucoMatch:
  """One full Truco match (see module docstring)."""

  def __init__(
      self,
      seed: int,
      num_players: int = 4,
      *,
      target: int = 12,
      initial_scores: dict[str, int] | None = None,
      scripted_deals: Iterable[Deal] | None = None,
  ):
    if num_players not in (2, 4, 6):
      raise ValueError("num_players must be 2, 4 or 6")
    self.seed = int(seed)
    self.num_players = num_players
    self.target = target
    self.scores: dict[str, int] = {"A": 0, "B": 0}
    if initial_scores:
      self.scores.update({k: int(v) for k, v in initial_scores.items()})
    self._scripted_deals: list[Deal] = list(scripted_deals or [])
    for d in self._scripted_deals:
      d.validate(num_players)
    self._fallback_rng = random.Random(f"truco:{self.seed}:fallback")

    self.events: list[dict[str, Any]] = []
    self.hand_history: list[dict[str, Any]] = []
    self.talk: list[dict[str, Any]] = []
    self.phase = PHASE_TERMINAL
    self.hand_index = -1
    self.dealer = num_players - 1  # Seat N-1 deals the first hand (§3).
    self.winner: str | None = None

    # Per-hand state (initialized in _start_hand).
    self.mao = 0
    self.vira = ""
    self.manilha_rank = ""
    self.hand_type = HAND_NORMAL
    self._dealt: dict[int, tuple[str, str, str]] = {}
    self._hands: dict[int, list[str]] = {}
    self._positions: dict[int, list[int]] = {}
    self.stake = 1
    self.raise_right: set[str] = {"A", "B"}
    self.pending_call: dict[str, Any] | None = None
    self.trick_index = 0
    self.trick_leader = 0
    self.trick_plays: list[tuple[int, str]] = []
    self.trick_results: list[dict[str, Any]] = []
    self.plays: list[dict[str, Any]] = []  # all plays this hand
    self.turn_seat = 0
    self.onze_team: str | None = None
    self.onze_decider: int | None = None
    self.onze_revealed: dict[int, tuple[str, ...]] = {}

    self._log("match_start", seed=self.seed, num_players=num_players,
              target=target, scores=dict(self.scores))
    self._maybe_end_match_or_deal()

  # ---------------------------------------------------------------- logging

  def _log(self, event_type: str, **fields: Any) -> None:
    event = {"i": len(self.events), "type": event_type, "hand": self.hand_index}
    event.update(fields)
    self.events.append(event)

  def serialize_events(self) -> str:
    """JSON Lines, deterministic key order."""
    return "".join(
        json.dumps(e, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        + "\n"
        for e in self.events
    )

  # --------------------------------------------------------------- helpers

  def _deal_for_hand(self, hand_index: int) -> Deal:
    if hand_index < len(self._scripted_deals):
      return self._scripted_deals[hand_index]
    rng = random.Random(f"truco:{self.seed}:hand:{hand_index}")
    deck = C.make_deck()
    rng.shuffle(deck)
    hands = []
    order = [(self.mao + i) % self.num_players for i in range(self.num_players)]
    by_seat: dict[int, tuple[str, str, str]] = {}
    for i, seat in enumerate(order):
      by_seat[seat] = tuple(deck[3 * i : 3 * i + 3])  # type: ignore[assignment]
    for seat in range(self.num_players):
      hands.append(by_seat[seat])
    vira = deck[3 * self.num_players]
    return Deal(hands=tuple(hands), vira=vira)

  def _maybe_end_match_or_deal(self) -> None:
    for team in TEAMS:
      if self.scores[team] >= self.target:
        self.winner = team
        self.phase = PHASE_TERMINAL
        self._log("match_end", winner=team, scores=dict(self.scores),
                  hands_played=len(self.hand_history))
        return
    self._start_hand()

  def _start_hand(self) -> None:
    n = self.num_players
    self.hand_index += 1
    if self.hand_index > 0:
      self.dealer = (self.dealer + 1) % n
    self.mao = (self.dealer + 1) % n
    deal = self._deal_for_hand(self.hand_index)
    self.vira = deal.vira
    self.manilha_rank = C.manilha_rank(self.vira)
    self._dealt = {seat: tuple(deal.hands[seat]) for seat in range(n)}
    self._hands = {seat: list(deal.hands[seat]) for seat in range(n)}
    self._positions = {seat: [1, 2, 3] for seat in range(n)}
    self.stake = 1
    self.raise_right = {"A", "B"}
    self.pending_call = None
    self.trick_index = 0
    self.trick_leader = self.mao
    self.trick_plays = []
    self.trick_results = []
    self.plays = []
    self.turn_seat = self.mao
    self.onze_team = None
    self.onze_decider = None
    self.onze_revealed = {}

    a11 = self.scores["A"] == self.target - 1
    b11 = self.scores["B"] == self.target - 1
    if a11 and b11:
      self.hand_type = HAND_MAO_DE_FERRO
    elif a11 or b11:
      self.hand_type = HAND_MAO_DE_ONZE
    else:
      self.hand_type = HAND_NORMAL

    self._log(
        "hand_start",
        dealer=self.dealer,
        mao=self.mao,
        vira=self.vira,
        manilha_rank=self.manilha_rank,
        hands={str(s): list(self._dealt[s]) for s in range(n)},
        hand_type=self.hand_type,
        scores=dict(self.scores),
    )

    if self.hand_type == HAND_MAO_DE_FERRO:
      self.stake = 1
      self.phase = PHASE_PLAY
      self._log("mao_de_ferro", stake=1)
    elif self.hand_type == HAND_MAO_DE_ONZE:
      self.onze_team = "A" if a11 else "B"
      # AMBIGUITIES #3: first player of that team in play order from the mão.
      for i in range(n):
        seat = (self.mao + i) % n
        if team_of(seat) == self.onze_team:
          self.onze_decider = seat
          break
      self.onze_revealed = {
          seat: self._dealt[seat]
          for seat in range(n)
          if team_of(seat) == self.onze_team
      }
      self.phase = PHASE_MAO_DE_ONZE
      self._log("mao_de_onze", team=self.onze_team, decider=self.onze_decider)
    else:
      self.phase = PHASE_PLAY

  @property
  def raises_allowed(self) -> bool:
    return self.hand_type == HAND_NORMAL and self.stake < LADDER[-1]

  def _team_seats(self, team: str) -> list[int]:
    return [s for s in range(self.num_players) if team_of(s) == team]

  # ----------------------------------------------------------- public API

  def is_terminal(self) -> bool:
    return self.phase == PHASE_TERMINAL

  def current_actor(self) -> int:
    if self.phase == PHASE_TERMINAL:
      return TERMINAL_PLAYER
    if self.phase == PHASE_MAO_DE_ONZE:
      assert self.onze_decider is not None
      return self.onze_decider
    if self.phase == PHASE_RAISE_RESPONSE:
      assert self.pending_call is not None
      return int(self.pending_call["responder_seat"])
    return self.turn_seat

  def returns(self) -> dict[str, float]:
    if self.winner is None:
      return {"A": 0.0, "B": 0.0}
    return {t: (1.0 if t == self.winner else -1.0) for t in TEAMS}

  def legal_actions(self, seat: int) -> list[str]:
    """Exact action strings legal for ``seat`` right now (§13)."""
    if self.phase == PHASE_TERMINAL or seat != self.current_actor():
      return []
    if self.phase == PHASE_MAO_DE_ONZE:
      return ["MAO_PLAY", "MAO_FORFEIT"]
    if self.phase == PHASE_RAISE_RESPONSE:
      assert self.pending_call is not None
      actions = ["ACCEPT", "DECLINE"]
      if next_stake(int(self.pending_call["value"])) is not None:
        actions.append("RAISE")
      return actions
    # PHASE_PLAY
    actions: list[str] = []
    if self.hand_type == HAND_MAO_DE_FERRO:
      actions.extend(f"PLAY {p}" for p in self._positions[seat])
    else:
      actions.extend(f"PLAY {card}" for card in self._hands[seat])
    if self.raises_allowed and team_of(seat) in self.raise_right:
      nxt = next_stake(self.stake)
      if nxt is not None:
        actions.append(RAISE_NAME_BY_VALUE[nxt])
    actions.append("FOLD")
    return actions

  def random_legal_action(self, seat: int) -> str:
    """Uniformly random legal action from the seed-derived fallback stream."""
    legal = self.legal_actions(seat)
    if not legal:
      raise IllegalActionError(f"Seat {seat} has no legal actions now")
    return self._fallback_rng.choice(legal)

  def apply_action(self, seat: int, action: str, talk: str | None = None) -> None:
    """Apply ``action`` for ``seat``; optional public table talk (§11)."""
    legal = self.legal_actions(seat)
    if action not in legal:
      raise IllegalActionError(
          f"Action {action!r} is not legal for seat {seat} "
          f"(actor={self.current_actor()}, legal={legal})"
      )
    talk_text = self._record_talk(seat, talk)
    self._log("action", seat=seat, action=action, talk=talk_text,
              phase=self.phase, trick=self.trick_index)

    if self.phase == PHASE_MAO_DE_ONZE:
      self._apply_mao_de_onze(seat, action)
    elif self.phase == PHASE_RAISE_RESPONSE:
      self._apply_raise_response(seat, action)
    elif action == "FOLD":
      self._end_hand(other_team(team_of(seat)), self.stake, reason="fold",
                     by_seat=seat)
    elif action in RAISE_VALUE_BY_NAME:
      self._call_raise(seat, RAISE_VALUE_BY_NAME[action])
    else:
      assert action.startswith("PLAY ")
      self._play(seat, action[5:])

  # ----------------------------------------------------------- internals

  def _record_talk(self, seat: int, talk: str | None) -> str | None:
    if talk is None:
      return None
    text = str(talk).strip()
    if not text:
      return None
    if len(text) > TALK_MAX_CHARS:
      self._log("talk_truncated", seat=seat, original_length=len(text))
      text = text[:TALK_MAX_CHARS]
    self.talk.append({"hand": self.hand_index, "trick": self.trick_index,
                      "seat": seat, "text": text})
    return text

  def _apply_mao_de_onze(self, seat: int, action: str) -> None:
    assert self.onze_team is not None
    if action == "MAO_FORFEIT":
      self._log("mao_de_onze_decision", seat=seat, decision="forfeit")
      self._end_hand(other_team(self.onze_team), 1, reason="mao_de_onze_forfeit",
                     by_seat=seat)
      return
    self._log("mao_de_onze_decision", seat=seat, decision="play")
    self.stake = 3
    self.phase = PHASE_PLAY

  def _call_raise(self, seat: int, value: int) -> None:
    team = team_of(seat)
    self.pending_call = {
        "name": RAISE_NAME_BY_VALUE[value],
        "value": value,
        "caller_seat": seat,
        "calling_team": team,
        "responder_seat": (seat + 1) % self.num_players,
        "pre_call_stake": self.stake,
    }
    self.phase = PHASE_RAISE_RESPONSE
    self._log("raise_called", seat=seat, team=team,
              name=self.pending_call["name"], value=value,
              pre_call_stake=self.stake,
              responder_seat=self.pending_call["responder_seat"])

  def _accept_pending(self, seat: int) -> None:
    assert self.pending_call is not None
    call = self.pending_call
    self.stake = int(call["value"])
    self.raise_right = {team_of(seat)}
    self._log("raise_accepted", seat=seat, team=team_of(seat),
              name=call["name"], stake=self.stake,
              raise_right=sorted(self.raise_right))
    self.pending_call = None

  def _apply_raise_response(self, seat: int, action: str) -> None:
    assert self.pending_call is not None
    call = self.pending_call
    if action == "DECLINE":
      self._log("raise_declined", seat=seat, team=team_of(seat),
                name=call["name"], points=call["pre_call_stake"])
      self.pending_call = None
      self._end_hand(str(call["calling_team"]), int(call["pre_call_stake"]),
                     reason="decline", by_seat=seat)
      return
    # ACCEPT and RAISE both accept the pending call (§7.3, AMBIGUITIES #9).
    self._accept_pending(seat)
    if action == "ACCEPT":
      self.phase = PHASE_PLAY  # The interrupted seat (turn_seat) resumes.
      return
    nxt = next_stake(self.stake)
    assert nxt is not None
    self._call_raise(seat, nxt)

  def _play(self, seat: int, what: str) -> None:
    if self.hand_type == HAND_MAO_DE_FERRO:
      position = int(what)
      self._positions[seat].remove(position)
      card = self._dealt[seat][position - 1]
      self._hands[seat].remove(card)
      self._log("card_played", seat=seat, card=card, position=position,
                trick=self.trick_index)
      self.plays.append({"trick": self.trick_index, "seat": seat, "card": card,
                         "position": position})
    else:
      card = what
      self._hands[seat].remove(card)
      self._log("card_played", seat=seat, card=card, trick=self.trick_index)
      self.plays.append({"trick": self.trick_index, "seat": seat, "card": card})
    self.trick_plays.append((seat, card))
    if len(self.trick_plays) == self.num_players:
      self._finish_trick()
    else:
      self.turn_seat = (seat + 1) % self.num_players

  def _finish_trick(self) -> None:
    winner_team, winner_seat = resolve_trick(self.trick_plays, self.vira)
    result = {
        "trick": self.trick_index,
        "leader": self.trick_leader,
        "plays": [{"seat": s, "card": c} for s, c in self.trick_plays],
        "winner_team": winner_team,
        "winner_seat": winner_seat,
    }
    self.trick_results.append(result)
    self._log("trick_result", **result)
    results = [r["winner_team"] for r in self.trick_results]
    mao_team = team_of(self.mao)
    hw = hand_winner(results, mao_team)
    if hw is not None:
      self._end_hand(hw, self.stake, reason="tricks")
      return
    if self.trick_index == 2:
      # Cannot happen: three tricks always decide (§6).
      raise AssertionError("Three tricks played without a decision")
    self.trick_index += 1
    if winner_seat is not None:
      self.trick_leader = winner_seat
    # After a parda the previous leader leads again (§5).
    self.trick_plays = []
    self.turn_seat = self.trick_leader

  def _end_hand(self, winner_team: str, points: int, *, reason: str,
                by_seat: int | None = None) -> None:
    self.scores[winner_team] += points
    record = {
        "hand": self.hand_index,
        "hand_type": self.hand_type,
        "dealer": self.dealer,
        "mao": self.mao,
        "manilha_rank": self.manilha_rank,
        "winner_team": winner_team,
        "points": points,
        "reason": reason,
        "by_seat": by_seat,
        "final_stake": self.stake,
        "tricks": [r["winner_team"] for r in self.trick_results],
        "scores": dict(self.scores),
    }
    self.hand_history.append(record)
    self._log("hand_result", **{k: v for k, v in record.items() if k != "hand"})
    self.pending_call = None
    self._maybe_end_match_or_deal()

  # ----------------------------------------------------------- observation

  def observation(self, seat: int) -> dict[str, Any]:
    """Everything ``seat`` may see right now per §12 — and nothing more."""
    if not 0 <= seat < self.num_players:
      raise ValueError(f"Bad seat {seat}")
    team = team_of(seat)
    obs: dict[str, Any] = {
        "seat": seat,
        "team": team,
        "num_players": self.num_players,
        "partner_seats": [s for s in self._team_seats(team) if s != seat],
        "phase": self.phase,
        "terminal": self.is_terminal(),
        "winner": self.winner,
        "scores": dict(self.scores),
        "target": self.target,
        "hand_index": self.hand_index,
        "hand_type": self.hand_type,
        "dealer": self.dealer,
        "mao": self.mao,
        "vira": self.vira,
        "manilha_rank": self.manilha_rank,
        "strength_order": C.describe_strength_order(self.vira) if self.vira else "",
        "stake": self.stake,
        "pending_call": copy.deepcopy(self.pending_call),
        "raise_right": sorted(self.raise_right),
        "raises_allowed": self.raises_allowed,
        "trick_index": self.trick_index,
        "trick_leader": self.trick_leader,
        "current_trick": [{"seat": s, "card": c} for s, c in self.trick_plays],
        "plays": copy.deepcopy(self.plays),
        "trick_results": copy.deepcopy(self.trick_results),
        "to_act": self.current_actor(),
        "talk": copy.deepcopy(self.talk),
        "hand_history": copy.deepcopy(self.hand_history),
        "legal_actions": self.legal_actions(seat),
    }
    if self.is_terminal():
      obs["hand"] = None
      obs["blind_positions"] = None
      return obs
    if self.hand_type == HAND_MAO_DE_FERRO:
      obs["hand"] = None
      obs["blind_positions"] = list(self._positions[seat])
    else:
      obs["hand"] = list(self._hands[seat])
      obs["blind_positions"] = None
    # Mão de onze visibility (§9, §12, AMBIGUITIES #4).
    if self.hand_type == HAND_MAO_DE_ONZE and team == self.onze_team:
      partner_cards = {
          str(s): list(cards) for s, cards in self.onze_revealed.items() if s != seat
      }
      if self.phase == PHASE_MAO_DE_ONZE:
        obs["partner_hand"] = partner_cards
        obs["mao_de_onze_decider"] = self.onze_decider
      else:
        obs["partner_hand_at_deal"] = partner_cards
    return obs

  # ------------------------------------------------------ private inspect

  def hidden_cards(self, seat: int) -> set[str]:
    """Cards that ``seat`` is NOT entitled to see right now (for tests)."""
    if self.is_terminal():
      return set()
    n = self.num_players
    dealt = set()
    for s in range(n):
      dealt.update(self._dealt[s])
    dealt.add(self.vira)
    visible: set[str] = {self.vira}
    visible.update(p["card"] for p in self.plays)
    if self.hand_type != HAND_MAO_DE_FERRO:
      visible.update(self._hands[seat])
    if self.hand_type == HAND_MAO_DE_ONZE and team_of(seat) == self.onze_team:
      for s, cards in self.onze_revealed.items():
        visible.update(cards)
    unseen_deck = set(C.make_deck()) - dealt
    return (dealt - visible) | unseen_deck

  def snapshot_hands(self) -> dict[int, list[str]]:
    """Full-information view of remaining cards (logging/replay only)."""
    return {s: list(cs) for s, cs in self._hands.items()}
