"""Rule-based extraction, talk extraction, and soft matching."""

import pytest
from game_arena.harness import parsers

from runner.parsers import TrucoRuleParser, TrucoSoftParser, aliases_for, extract_talk, soft_match

LEGAL = ["PLAY 3♠", "PLAY 2♦", "PLAY K♣", "TRUCO", "FOLD"]


def rule(text):
  return TrucoRuleParser().parse(parsers.TextParserInput(text=text))


def test_rule_parser_takes_last_final_answer_line_only():
  assert rule("thinking...\nFinal Answer: PLAY 3♠") == "PLAY3♠"
  assert rule("Final Answer: FOLD\nActually no.\nFinal Answer: TRUCO") == "TRUCO"
  assert rule("**Final Answer:** PLAY K♣\nTalk: I have manilhas") == "PLAYK♣"
  assert rule("final answer - fold") == "fold"
  assert rule("Final Answer: \\boxed{PLAY 2♦}") == "PLAY2♦"
  assert rule("no answer here") is None
  assert rule("") is None


def test_talk_extraction():
  assert extract_talk("Talk: Truco is coming!\nFinal Answer: PLAY 3♠") == "Truco is coming!"
  assert extract_talk("Final Answer: PLAY 3♠\n**Talk:** \"careful\"") == "careful"
  assert extract_talk("Final Answer: FOLD") is None
  assert extract_talk("Talk: none\nFinal Answer: FOLD") is None
  assert extract_talk("Talk:\nFinal Answer: FOLD") is None


@pytest.mark.parametrize("text,expected", [
    ("PLAY3♠", "PLAY 3♠"), ("PLAY 3♠", "PLAY 3♠"), ("play 3s", "PLAY 3♠"),
    ("PLAY 3S", "PLAY 3♠"), ("3♠", "PLAY 3♠"), ("3 of spades", "PLAY 3♠"),
    ("Play the 2 of Diamonds", "PLAY 2♦"), ("PLAY K♣.", "PLAY K♣"), ("PLAY KC", "PLAY K♣"),
    ("PLAY 3", "PLAY 3♠"),          # rank-only, unique in hand
    ("TRUCO!", "TRUCO"), ("truco", "TRUCO"), ("Fold", "FOLD"), ("I fold", "FOLD"),
    ("PLAY 4♠", None),              # not in hand
    ("PLAY 3♦", None),              # wrong suit
    ("SEIS", None),                 # wrong ladder step
    ("ACCEPT", None), ("", None), ("???", None),
    ("PLAY 3♠ or PLAY 2♦", None),   # ambiguous
])
def test_soft_match_card_play_context(text, expected):
  assert soft_match(text, LEGAL) == expected


def test_soft_match_rank_only_ambiguous_is_none():
  assert soft_match("PLAY 3", ["PLAY 3♠", "PLAY 3♦", "FOLD"]) is None


def test_soft_match_raise_response_aliases():
  legal = ["ACCEPT", "DECLINE", "RAISE"]
  obs = {"pending_call": {"value": 3}}
  aliases = aliases_for(obs, legal)
  assert soft_match("SEIS", legal, aliases) == "RAISE"
  assert soft_match("NOVE", legal, aliases) is None  # not the next step
  assert soft_match("raise back", legal, aliases) == "RAISE"
  assert soft_match("Accept!", legal, aliases) == "ACCEPT"
  assert soft_match("no", legal, aliases) == "DECLINE"
  assert soft_match("FOLD", legal, aliases) is None  # DECLINE is the fold; must be explicit
  # At Doze, RAISE is not legal and no alias may create it.
  legal12 = ["ACCEPT", "DECLINE"]
  assert soft_match("RAISE", legal12, aliases_for({"pending_call": {"value": 12}}, legal12)) is None


def test_soft_match_mao_de_onze_and_ferro():
  legal = ["MAO_PLAY", "MAO_FORFEIT"]
  al = aliases_for({"pending_call": None}, legal)
  assert soft_match("MAO_PLAY", legal, al) == "MAO_PLAY"
  assert soft_match("mao play", legal, al) == "MAO_PLAY"
  assert soft_match("Play", legal, al) == "MAO_PLAY"
  assert soft_match("forfeit", legal, al) == "MAO_FORFEIT"
  ferro = ["PLAY 1", "PLAY 3", "FOLD"]
  assert soft_match("PLAY 3", ferro) == "PLAY 3"
  assert soft_match("position 1", ferro) == "PLAY 1"
  assert soft_match("PLAY 2", ferro) is None  # already played


def test_soft_parser_class_requires_state_and_returns_only_legal():
  p = TrucoSoftParser()
  with pytest.raises(ValueError):
    p.parse(parsers.TextParserInput(text="FOLD"))
  out = p.parse(parsers.TextParserInput(text="PLAY3♠", state_str="s", legal_moves=LEGAL, player_number=0))
  assert out == "PLAY 3♠"
  out = p.parse(parsers.TextParserInput(text="PLAY 9♠", state_str="s", legal_moves=LEGAL, player_number=0))
  assert out is None
