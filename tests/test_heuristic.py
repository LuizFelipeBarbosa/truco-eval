"""Heuristic baseline bot: exact rule positions, determinism, legality, strength."""

import json
import random

import pytest

from runner import cli
from runner.config import KIND_HEURISTIC, KIND_RANDOM, ModelConfig
from runner.heuristic import HeuristicBotAgent
from runner.match_runner import make_spec, play_match
from runner.tournament import preflight
from tests.conftest import act, deal
from truco.match import TrucoMatch

VIRA = "4♦"  # manilha rank is 5
FILLER = ("4♥ 6♥ 7♥", "4♠ 6♠ 7♠", "Q♦ Q♠ J♦")


def _no_fallback():
  pytest.fail("heuristic must not use the fallback")


def decide(m, seat):
  return HeuristicBotAgent().decide(
      m.observation(seat), m.legal_actions(seat), _no_fallback).action


def _match(*hands, **kwargs):
  return TrucoMatch(0, scripted_deals=[deal(*hands, vira=VIRA)], **kwargs)


def test_decision_source_and_name():
  m = _match("4♣ 6♣ 7♣", *FILLER)
  d = HeuristicBotAgent().decide(m.observation(0), m.legal_actions(0), _no_fallback)
  assert d.source == "heuristic_bot" and d.talk is None
  assert HeuristicBotAgent.name == "heuristic"


def test_partner_winning_trick_plays_lowest_card():
  m = _match("2♣ 7♦ 6♦", "4♥ 6♥ 7♥", "4♠ A♠ 3♥", "Q♦ Q♠ J♦")
  act(m, "PLAY 2♣", "PLAY 4♥")
  assert m.current_actor() == 2
  assert decide(m, 2) == "PLAY 4♠"


def test_plays_lowest_card_that_beats_current_best():
  m = _match("J♦ 7♦ 6♦", "6♥ K♥ A♥", "4♠ 6♠ 7♠", "Q♦ Q♠ 4♣")
  act(m, "PLAY J♦")
  assert decide(m, 1) == "PLAY K♥"


def test_cannot_beat_plays_lowest_card():
  m = _match("3♣ 7♦ 6♦", "6♥ K♥ A♥", "4♠ 6♠ 7♠", "Q♦ Q♠ 4♣")
  act(m, "PLAY 3♣")
  assert decide(m, 1) == "PLAY 6♥"


def test_tie_for_parda_after_winning_trick_one():
  m = _match("A♦ 4♠ 6♠", "4♥ 7♥ 6♥", "6♣ 7♣ 4♣", "Q♦ J♦ J♠")
  act(m, "PLAY A♦", "PLAY 4♥", "PLAY 4♣", "PLAY Q♦", "PLAY 4♠", "PLAY 7♥")
  assert m.trick_index == 1 and m.current_actor() == 2
  assert decide(m, 2) == "PLAY 7♣"


def test_tie_not_used_after_losing_trick_one():
  m = _match("4♠ 6♠ J♣", "A♥ 7♥ 6♥", "6♣ 7♣ 4♣", "Q♦ J♦ J♠")
  act(m, "PLAY 4♠", "PLAY A♥", "PLAY 4♣", "PLAY Q♦", "PLAY 7♥")
  assert m.trick_index == 1 and m.current_actor() == 2
  assert decide(m, 2) == "PLAY 6♣"


def test_tie_not_used_in_trick_one():
  m = _match("7♦ 4♠ 6♠", "4♥ 7♥ 6♥", "4♣ 6♣ 7♣", "Q♦ Q♠ J♦")
  act(m, "PLAY 7♦")
  assert decide(m, 1) == "PLAY 4♥"


def test_trick_one_lead_plays_middle_card():
  m = _match("2♣ 7♦ 6♦", "4♥ 6♥ 7♥", "4♠ 6♠ 7♠", "6♣ Q♠ J♠")
  assert decide(m, 0) == "PLAY 7♦"


def test_later_lead_plays_strongest_card():
  m = _match("2♣ 7♦ 6♦", "4♥ 6♥ 7♥", "4♠ 6♠ 7♠", "6♣ Q♠ J♠")
  act(m, "PLAY 7♦", "PLAY 4♥", "PLAY 4♠", "PLAY 6♣")
  assert m.trick_index == 1 and m.current_actor() == 0
  assert decide(m, 0) == "PLAY 2♣"


def test_calls_truco_with_strong_hand_and_plays_card_with_weak_hand():
  strong = _match("5♣ 5♥ 4♠", "4♥ 6♥ 7♥", "6♠ 7♠ Q♠", "Q♦ J♦ J♠")
  assert decide(strong, 0) == "TRUCO"
  weak = _match("4♠ 6♠ 7♠", "4♥ 6♥ 7♥", "6♣ 7♣ Q♠", "Q♦ J♦ J♠")
  assert decide(weak, 0).startswith("PLAY ")


def _after_truco(responder_hand, scores=None):
  m = _match("5♣ 3♣ 2♣", responder_hand, "4♠ 6♠ 7♠", "Q♦ Q♠ J♦",
             **({"initial_scores": scores} if scores else {}))
  act(m, "TRUCO")
  assert m.current_actor() == 1
  return m


@pytest.mark.parametrize("hand,expected", [
    ("3♥ 4♥ 6♥", "DECLINE"),  # score 2.0 < 2.5
    ("3♥ K♥ A♥", "ACCEPT"),  # 3.5, just under raise
    ("5♥ 3♥ 2♥", "RAISE"),  # 6.5
])
def test_raise_response(hand, expected):
  assert decide(_after_truco(hand), 1) == expected


def test_accepts_when_declining_would_lose_the_match():
  def seis_response(scores):
    m = _match("5♣ 3♣ 2♣", "4♥ 6♥ 7♥", "4♠ 6♠ 7♠", "Q♦ Q♠ J♦", initial_scores=scores)
    act(m, "TRUCO", "ACCEPT", "PLAY 5♣", "SEIS")
    assert m.current_actor() == 2
    return decide(m, 2)

  assert seis_response({"B": 0}) == "DECLINE"
  assert seis_response({"B": 9}) == "ACCEPT"  # 9 + pre_call_stake 3 >= 12


def test_mao_de_onze_threshold():
  at_threshold = _match("3♣ 3♥ 4♠", "4♥ 6♥ 7♥", "6♠ 7♠ Q♠", "Q♦ J♦ J♠",
                        initial_scores={"A": 11})
  assert at_threshold.observation(0)["phase"] == "MAO_DE_ONZE"
  assert decide(at_threshold, 0) == "MAO_PLAY"  # 4.0 >= 2.0 * 6 / 3
  below = _match("3♣ 2♥ 4♠", "4♥ 6♥ 7♥", "6♠ 7♠ Q♠", "Q♦ J♦ J♠",
                 initial_scores={"A": 11})
  assert decide(below, 0) == "MAO_FORFEIT"  # 3.5


def test_mao_de_ferro_plays_lowest_position():
  m = _match("3♣ 3♥ 4♠", "4♥ 6♥ 7♥", "6♠ 7♠ Q♠", "Q♦ J♦ J♠",
             initial_scores={"A": 11, "B": 11})
  assert m.observation(0)["hand_type"] == "mao_de_ferro"
  assert decide(m, 0) == "PLAY 1"
  act(m, "PLAY 3")
  assert decide(m, 1) == "PLAY 1"


def test_deterministic():
  m = _match("5♣ 3♣ 2♣", "3♥ 2♥ 4♥", "4♠ 6♠ 7♠", "Q♦ Q♠ J♦")
  obs, legal = m.observation(0), m.legal_actions(0)
  bot = HeuristicBotAgent()
  first = bot.decide(obs, legal, _no_fallback)
  assert all(bot.decide(obs, legal, _no_fallback) == first for _ in range(5))
  assert HeuristicBotAgent().decide(obs, legal, _no_fallback) == first


def _play_out(seed, n, heuristic_seats):
  m = TrucoMatch(seed, n)
  bot = HeuristicBotAgent()
  rng = random.Random(seed)
  while not m.is_terminal():
    s = m.current_actor()
    legal = m.legal_actions(s)
    if s in heuristic_seats:
      action = bot.decide(m.observation(s), legal, _no_fallback).action
    else:
      action = rng.choice(legal)
    assert action in legal
    m.apply_action(s, action)
  return m


@pytest.mark.parametrize("n", [2, 4, 6])
def test_self_play_matches_terminate_with_legal_actions(n):
  for seed in range(500):
    m = _play_out(seed, n, set(range(n)))
    assert m.winner in ("A", "B")


@pytest.mark.parametrize("n", [2, 4, 6])
def test_heuristic_vs_random_matches_terminate_with_legal_actions(n):
  for seed in range(500):
    m = _play_out(seed, n, set(range(0, n, 2)))
    assert m.winner in ("A", "B")


HEURISTIC = ModelConfig(kind=KIND_HEURISTIC)
RANDOM = ModelConfig(kind=KIND_RANDOM)
MIN_WIN_RATE_VS_RANDOM = 0.85


def test_heuristic_beats_random():
  wins = games = 0
  for seed in range(100):
    for swap in (False, True):
      summary = play_match(make_spec(seed, HEURISTIC, RANDOM, swap=swap))
      wins += summary["winner_config"] == "heuristic"
      games += 1
      stats = summary["team_stats"]
      assert all(s["model_decisions"] == 0 for s in stats.values())
  rate = wins / games
  print(f"heuristic vs random win rate: {rate:.3f} ({wins}/{games})")
  assert rate >= MIN_WIN_RATE_VS_RANDOM


def test_heuristic_config_display_and_agent():
  cfg = ModelConfig(kind=KIND_HEURISTIC)
  assert cfg.display == "heuristic"
  assert isinstance(cfg.build_agent(), HeuristicBotAgent)


def test_cli_run_heuristic_vs_random(tmp_path):
  rc = cli.main(["run", "--team-a", "heuristic", "--team-b", "random",
                 "--seeds", "2", "--out", str(tmp_path)])
  assert rc == 0


def test_cli_tournament_with_heuristic_entry(tmp_path):
  models = tmp_path / "models.json"
  models.write_text(json.dumps([{"slug": "heuristic"}, {"slug": "random"}]))
  rc = cli.main(["tournament", "--models-file", str(models), "--seeds", "2",
                 "--out", str(tmp_path / "rr")])
  assert rc == 0


def test_preflight_skips_bots():
  assert preflight([HEURISTIC, RANDOM]) == {}
