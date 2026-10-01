"""Full matches through the orchestrator with fake LLMs: isolation, logging, duplicates."""

import json
import os

import pytest

from runner import agents as truco_agents
from runner.config import ModelConfig
from runner.match_runner import IsolationViolation, make_spec, play_match, redact_talk
from runner.replay import replay_file
from tests.fake_model import FakeModel
from truco.match import TrucoMatch, team_of

RANDOM_A = ModelConfig(kind="random", label="botA")
RANDOM_B = ModelConfig(kind="random", label="botB")


def _llm_agents(seed=0, illegal_prob=0.0):
  fa = FakeModel(seed=seed, illegal_prob=illegal_prob)
  fb = FakeModel(seed=seed + 1000, illegal_prob=illegal_prob)
  a, b = truco_agents.LLMAgent(fa, "fakeA"), truco_agents.LLMAgent(fb, "fakeB")
  return {0: a, 1: b, 2: a, 3: b}, fa, fb


def _events(out_dir, name="transcript.jsonl"):
  with open(os.path.join(out_dir, name), encoding="utf-8") as f:
    return [json.loads(l) for l in f if l.strip()]


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_every_prompt_is_information_isolated(tmp_path, seed):
  """Grep every prompt for cards the seat is not entitled to see."""
  spec = make_spec(seed, RANDOM_A, RANDOM_B, out_root=str(tmp_path))
  agents, _, _ = _llm_agents(seed, illegal_prob=0.3)
  summary = play_match(spec, agents, assert_isolation=True)  # runtime guard on too
  events = _events(spec.out_dir)
  # Independently recompute what each seat was entitled to see at each prompt by
  # replaying the engine alongside the transcript.
  engine = TrucoMatch(spec.seed, spec.num_players)
  checked = 0
  for e in events:
    if e["source"] == "engine" and e["type"] == "action":
      engine.apply_action(e["seat"], e["action"], e.get("talk"))
    elif e["type"] == "prompt":
      seat = e["seat"]
      assert engine.current_actor() == seat
      hidden = engine.hidden_cards(seat)
      scan = e["prompt_text"]
      if e.get("quoted_generation"):
        scan = scan.replace(e["quoted_generation"], "<previous reply>")
      scan = redact_talk(scan, engine.observation(seat)["talk"])
      for card in hidden:
        assert card not in scan, f"seed {seed}: seat {seat} prompt shows {card}"
      # The seat's own cards (when visible) must be in the prompt.
      obs = engine.observation(seat)
      if obs["hand"]:
        for card in obs["hand"]:
          assert card in e["prompt_text"]
      checked += 1
  assert checked > 0 and summary["winner_team"] in ("A", "B")


def test_runtime_isolation_guard_raises_on_leak():
  spec = make_spec(0, RANDOM_A, RANDOM_B)
  leaky_agents, fa, _ = _llm_agents(0)

  class LeakyAgent:
    name = "leaky"

    def decide(self, observation, legal_actions, fallback_fn):
      d = leaky_agents[0].decide(observation, legal_actions, fallback_fn)
      d.prompts[0]["prompt_text"] += " " + " ".join(sorted(hidden))
      return d

  probe = TrucoMatch(0)
  hidden = probe.hidden_cards(0)
  agents = dict(leaky_agents)
  agents[0] = LeakyAgent()
  with pytest.raises(IsolationViolation):
    play_match(spec, agents)


def test_transcript_contains_illegal_and_fallback_events_and_usage(tmp_path):
  spec = make_spec(7, RANDOM_A, RANDOM_B, out_root=str(tmp_path))
  agents, _, _ = _llm_agents(7, illegal_prob=0.5)
  summary = play_match(spec, agents)
  events = _events(spec.out_dir)
  types = {e["type"] for e in events}
  assert {"match_config", "prompt", "response", "illegal_action", "fallback_action",
          "decision", "hand_start", "action", "hand_result", "match_end"} <= types
  cfg = events[0]
  assert cfg["type"] == "match_config" and cfg["seats"]["0"]["display"] == "botA"
  assert "Truco Paulista" in cfg["system_instruction"]
  ill = [e for e in events if e["type"] == "illegal_action"]
  assert all({"seat", "hand", "trick", "attempt", "raw_response", "legal_actions"} <= set(e) for e in ill)
  # Every fallback action was preceded by two illegal attempts for that decision.
  for fb in [e for e in events if e["type"] == "fallback_action"]:
    same = [e for e in ill if e["seat"] == fb["seat"] and e["hand"] == fb["hand"]]
    assert len(same) >= 2
  # Stats
  ts = summary["team_stats"]["A"]
  assert ts["illegal_responses"] > 0 and ts["fallbacks"] > 0
  assert ts["prompt_tokens"] == 100 * ts["requests"]
  assert ts["cost_usd"] == pytest.approx(0.001 * ts["requests"])
  assert ts["providers"] == {"Fake": ts["requests"]}
  resp = next(e for e in events if e["type"] == "response")
  assert resp["served_provider"] == "Fake" and resp["usage"]["prompt_tokens"] == 100
  # Files
  for name in ("transcript.jsonl", "engine_events.jsonl", "replay.txt", "summary.json"):
    assert os.path.exists(os.path.join(spec.out_dir, name))
  match, identical = replay_file(os.path.join(spec.out_dir, "engine_events.jsonl"))
  assert identical and match.scores == summary["scores"]


def test_talk_flows_from_model_to_engine_and_other_seats(tmp_path):
  spec = make_spec(2, RANDOM_A, RANDOM_B, out_root=str(tmp_path))
  agents, _, _ = _llm_agents(2)
  play_match(spec, agents)
  events = _events(spec.out_dir)
  talk_actions = [e for e in events if e["type"] == "action" and e.get("talk")]
  assert talk_actions
  assert talk_actions[0]["talk"] == "watch out, I'm strong!"
  later_prompts = [e for e in events if e["type"] == "prompt" and e["i" if "i" in e else "seat"] is not None]
  assert any("watch out, I'm strong!" in e["prompt_text"] for e in later_prompts)


def test_duplicate_match_swaps_teams_and_keeps_deals(tmp_path):
  orig = make_spec(11, RANDOM_A, RANDOM_B, out_root=str(tmp_path))
  dup = make_spec(11, RANDOM_A, RANDOM_B, swap=True, out_root=str(tmp_path))
  assert orig.match_id == "seed11_orig" and dup.match_id == "seed11_dup"
  assert orig.seat_configs()[0].label == "botA" and dup.seat_configs()[0].label == "botB"
  assert dup.seat_configs()[1].label == "botA"
  s1 = play_match(orig)
  s2 = play_match(dup)
  d1 = [e for e in _events(orig.out_dir, "engine_events.jsonl") if e["type"] == "hand_start"]
  d2 = [e for e in _events(dup.out_dir, "engine_events.jsonl") if e["type"] == "hand_start"]
  for a, b in zip(d1, d2):
    assert a["hands"] == b["hands"] and a["vira"] == b["vira"]
  assert s1["team_config"]["A"]["display"] == "botA" and s2["team_config"]["A"]["display"] == "botB"


def test_random_bot_matches_are_byte_identical_across_runs(tmp_path):
  a = make_spec(5, RANDOM_A, RANDOM_B, out_root=str(tmp_path / "a"))
  b = make_spec(5, RANDOM_A, RANDOM_B, out_root=str(tmp_path / "b"))
  play_match(a)
  play_match(b)
  for name in ("engine_events.jsonl", "transcript.jsonl", "replay.txt"):
    with open(os.path.join(a.out_dir, name), "rb") as fa, open(os.path.join(b.out_dir, name), "rb") as fb:
      assert fa.read() == fb.read()


def test_seat_map_records_slug_provider_and_options(tmp_path):
  cfg_a = ModelConfig(slug="openai/gpt-5-mini", provider={"order": ["OpenAI"]},
                      model_options={"temperature": 0.2}, label="mini")
  cfg_b = ModelConfig(slug="deepseek/deepseek-chat-v3.1", label="ds")
  spec = make_spec(0, cfg_a, cfg_b, out_root=str(tmp_path))
  agents, _, _ = _llm_agents(0)
  summary = play_match(spec, agents)
  cfg = _events(spec.out_dir)[0]
  assert cfg["seats"]["0"]["slug"] == "openai/gpt-5-mini"
  assert cfg["seats"]["0"]["provider"] == {"order": ["OpenAI"]}
  assert cfg["seats"]["0"]["model_options"] == {"temperature": 0.2}
  assert cfg["seats"]["1"]["slug"] == "deepseek/deepseek-chat-v3.1"
  assert summary["team_config"]["A"]["display"] == "mini"
  for seat in range(4):
    assert cfg["seats"][str(seat)]["team"] == team_of(seat)


def test_talk_naming_hidden_cards_is_not_an_isolation_violation(tmp_path):
  """A player may bluff about any card in public talk; that is not a leak."""
  from truco import cards as C

  class BluffingFakeModel(FakeModel):
    def generate_with_text_input(self, model_input):
      legal = [l for l in __import__("tests.fake_model", fromlist=["x"]).legal_from_prompt(
          model_input.prompt_text)]
      all_cards = " ".join(C.make_deck())  # names every card, hidden ones included
      text = f"Talk: I hold {all_cards}\nFinal Answer: {legal[0]}"
      self.calls.append(model_input)
      from game_arena.harness import model_generation
      return model_generation.GenerateReturn(main_response=text, main_response_and_thoughts=text)

  bluff = truco_agents.LLMAgent(BluffingFakeModel(), "bluffer")
  spec = make_spec(3, RANDOM_A, RANDOM_B, out_root=str(tmp_path))
  summary = play_match(spec, {0: bluff, 1: bluff, 2: bluff, 3: bluff}, assert_isolation=True)
  assert summary["winner_team"] in ("A", "B")
  # The bluff reached later prompts without tripping the runtime guard.
  events = _events(spec.out_dir)
  assert any(e["type"] == "prompt" and "I hold" in e["prompt_text"] for e in events)


def test_reprompt_quoting_an_illegal_card_is_not_a_violation(tmp_path):
  """A model naming a card it does not hold, quoted back in the re-prompt, is its own text."""
  spec = make_spec(4, RANDOM_A, RANDOM_B, out_root=str(tmp_path))
  agents, _, _ = _llm_agents(4, illegal_prob=0.6)
  summary = play_match(spec, agents, assert_isolation=True)
  events = _events(spec.out_dir)
  reprompts = [e for e in events if e["type"] == "prompt" and e["attempt"] == 1]
  assert reprompts and all(e["quoted_generation"] for e in reprompts)
  assert summary["team_stats"]["A"]["illegal_responses"] > 0


def test_illegal_reply_that_quotes_talk_and_names_hidden_cards(tmp_path):
  """Reasoning that repeats a talk line and counts hidden cards, quoted in a re-prompt."""
  from game_arena.harness import model_generation
  from tests.fake_model import legal_from_prompt
  from truco import cards as C

  class Model(FakeModel):
    def __init__(self):
      super().__init__(); self.n = 0
    def generate_with_text_input(self, model_input):
      self.calls.append(model_input); self.n += 1
      legal = legal_from_prompt(model_input.prompt_text)
      if self.n % 3 == 1:  # every third decision: a legal move that also talks
        text = f"Talk: partner, go low behind me and keep your best card\nFinal Answer: {legal[0]}"
        return model_generation.GenerateReturn(main_response=text, main_response_and_thoughts=text)
      if self.n % 3 == 2:  # then an empty-content reply whose reasoning quotes that talk + counts cards
        thoughts = ("They said 'partner, go low behind me and keep your best card'. Remaining cards: "
                    + ", ".join(C.make_deck()) + ". No decision yet.")
        return model_generation.GenerateReturn(main_response="", main_response_and_thoughts=f"<think>{thoughts}</think>")
      text = f"Final Answer: {legal[0]}"
      return model_generation.GenerateReturn(main_response=text, main_response_and_thoughts=text)

  a = truco_agents.LLMAgent(Model(), "quoter")
  spec = make_spec(26, RANDOM_A, RANDOM_B, out_root=str(tmp_path))
  summary = play_match(spec, {0: a, 1: a, 2: a, 3: a}, assert_isolation=True)
  assert summary["team_stats"]["A"]["illegal_responses"] > 0


def test_replay_reproduces_truncated_talk(tmp_path):
  from runner.replay import replay_events
  from truco.match import TrucoMatch
  m = TrucoMatch(3)
  m.apply_action(0, m.legal_actions(0)[0], talk="y" * 350)
  m.apply_action(1, m.legal_actions(1)[0], talk="short")
  events = [json.loads(l) for l in m.serialize_events().splitlines()]
  assert any(e["type"] == "talk_truncated" for e in events)
  rebuilt, identical = replay_events(events)
  assert identical
