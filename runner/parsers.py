"""Truco action parsing: harness rule-based extraction + soft matching."""

from __future__ import annotations

import re
from typing import Mapping, Sequence

from game_arena.harness import parsers

from truco import cards as C

_FINAL_ANSWER_RE = re.compile(r"final\s*answer\s*[:\-–]?", re.IGNORECASE)
_TALK_RE = re.compile(r"^[\s>*_#\-]*talk[ \t*_]*:[ \t*_]*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)
_SUIT_WORDS = {
    "CLUBS": "♣", "CLUB": "♣", "PAUS": "♣",
    "HEARTS": "♥", "HEART": "♥", "COPAS": "♥",
    "SPADES": "♠", "SPADE": "♠", "ESPADAS": "♠",
    "DIAMONDS": "♦", "DIAMOND": "♦", "OUROS": "♦",
}
_SUIT_LETTERS = {"C": "♣", "H": "♥", "S": "♠", "D": "♦"}
_CARD_RE = re.compile(r"^([4567QJKA23])([♣♥♠♦CHSD])$")
_SUIT_WORD_PATTERN = "|".join(re.escape(word) for word in _SUIT_WORDS)
_CARD_WORD_RE = re.compile(
    rf"^([4567QJKA23])(?:OF|DE)?({_SUIT_WORD_PATTERN})$"
)
_STRIP_RE = re.compile(r"[^A-Z0-9♣♥♠♦_]")


def extract_talk(response: str) -> str | None:
  """Last ``Talk:`` line of a response, or None if absent/empty/'none'."""
  matches = _TALK_RE.findall(response or "")
  if not matches:
    return None
  text = matches[-1].strip(" \t\"'*_")
  if not text or text.lower() in ("none", "-", "(none)", "n/a", "no talk", "nothing"):
    return None
  return text


class TrucoRuleParser(parsers.TextParser):
  """Extracts the action from the last ``Final Answer:`` line.

  Isolates that line first (so a trailing ``Talk:`` line is never mistaken
  for the action), then applies the harness ``RuleBasedMoveParser`` to it.
  The harness parser strips whitespace and markdown before soft matching, so
  compressed word-form cards such as ``Kofdiamonds`` are canonicalized too.
  """

  def __init__(self):
    self._rule_parser = parsers.RuleBasedMoveParser(
        action_tag="Final Answer:", additional_tags=()
    )

  def parse(self, parser_input: parsers.TextParserInput) -> str | None:
    text = parser_input.text or ""
    matches = list(_FINAL_ANSWER_RE.finditer(text))
    if not matches:
      return None
    segment = text[matches[-1].end():].split("\n", 1)[0]
    return self._rule_parser.parse(
        parsers.TextParserInput(text="Final Answer:" + segment)
    )


def normalize(text: str) -> str:
  t = (text or "").upper()
  for word, sym in _SUIT_WORDS.items():
    t = re.sub(rf"\b{word}\b", sym, t)
  t = re.sub(r"\bOF\b", "", t)
  t = _STRIP_RE.sub("", t)
  return t


def _canonical_card(token: str) -> str | None:
  m = _CARD_RE.match(token)
  if not m:
    word_match = _CARD_WORD_RE.match(token)
    if word_match is None:
      return None
    rank, suit_word = word_match.groups()
    return rank + _SUIT_WORDS[suit_word]
  rank, suit = m.group(1), m.group(2)
  suit = _SUIT_LETTERS.get(suit, suit)
  return rank + suit


def soft_match(
    text: str,
    legal_actions: Sequence[str],
    aliases: Mapping[str, str] | None = None,
) -> str | None:
  """Map free-form ``text`` to exactly one legal action, else None."""
  if not text:
    return None
  norm = normalize(text)
  if not norm:
    return None
  by_norm = {normalize(a): a for a in legal_actions}
  if norm in by_norm:
    return by_norm[norm]
  # Aliases (e.g. SEIS -> RAISE while responding to TRUCO), only if legal.
  if aliases:
    for alias, target in aliases.items():
      if norm == normalize(alias) and target in legal_actions:
        return target
  # Card plays: "PLAY QS", "QS", "Q of spades", "play the Q♠".
  body = norm[4:] if norm.startswith("PLAY") else norm
  body = re.sub(r"^(THE|CARD|MY)+", "", body)
  # RuleBasedMoveParser strips whitespace before soft_match, yielding e.g. KOFDIAMONDS.
  card = _canonical_card(body)
  plays = [a for a in legal_actions if a.startswith("PLAY ")]
  if card is not None:
    if f"PLAY {card}" in legal_actions:
      return f"PLAY {card}"
    return None
  # Rank-only card ("PLAY 3") when exactly one legal card has that rank.
  if body in C.RANKS:
    candidates = [a for a in plays if C.is_card(a[5:]) and C.rank(a[5:]) == body]
    if len(candidates) == 1:
      return candidates[0]
  # Positions in mão de ferro: "PLAY POSITION 2", "POSITION2", "2".
  pos = re.sub(r"^(POSITION|POS|SLOT)", "", body)
  if pos in ("1", "2", "3") and f"PLAY {pos}" in legal_actions:
    return f"PLAY {pos}"
  # Unique legal action that the text starts with / contains (e.g. "TRUCO!!").
  contains = [a for n, a in by_norm.items() if n and n in norm]
  if len(contains) == 1:
    return contains[0]
  return None


class TrucoSoftParser(parsers.SoftMoveParser):
  """Harness ``SoftMoveParser`` specialised for Truco action strings."""

  def __init__(self, aliases: Mapping[str, str] | None = None):
    self._aliases = dict(aliases or {})

  def _parse_selected_action(self, parser_input: parsers.TextParserInput) -> str | None:
    assert parser_input.legal_moves is not None
    return soft_match(parser_input.text, parser_input.legal_moves, self._aliases)


def aliases_for(observation: dict, legal_actions: Sequence[str]) -> dict[str, str]:
  """Context-dependent aliases that map to exactly one legal action."""
  aliases: dict[str, str] = {
      "RUN": "FOLD", "CORRER": "FOLD", "CORRO": "FOLD", "FUGIR": "FOLD", "GIVE UP": "FOLD",
      "ACEITO": "ACCEPT", "ACEITAR": "ACCEPT", "YES": "ACCEPT", "OK": "ACCEPT",
      "NO": "DECLINE", "NAO": "DECLINE", "NÃO": "DECLINE", "REJECT": "DECLINE", "REFUSE": "DECLINE",
      "PLAY": "MAO_PLAY", "JOGAR": "MAO_PLAY", "MAO PLAY": "MAO_PLAY", "MAOPLAY": "MAO_PLAY",
      "FORFEIT": "MAO_FORFEIT", "MAO FORFEIT": "MAO_FORFEIT", "MAOFORFEIT": "MAO_FORFEIT",
  }
  call = observation.get("pending_call")
  if call and "RAISE" in legal_actions:
    next_name = {3: "SEIS", 6: "NOVE", 9: "DOZE"}.get(int(call["value"]))
    if next_name:
      aliases[next_name] = "RAISE"
      aliases["RAISE BACK"] = "RAISE"
  return aliases
