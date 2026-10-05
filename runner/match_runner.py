"""The match loop: engine + per-seat agents + logging + per-match stats."""

from __future__ import annotations

import collections
import dataclasses
from typing import Any, Mapping

import openrouter_model
from runner import agents as truco_agents
from runner import prompts as truco_prompts
from runner import version as truco_version
from runner.config import MatchSpec, ModelConfig
from runner.match_log import MatchLogger
from truco.match import TrucoMatch, team_of

RAISE_CALLS = ("TRUCO", "SEIS", "NOVE", "DOZE")


class IsolationViolation(AssertionError):
  """A prompt contained a card the seat is not entitled to see."""


def _new_stats() -> dict[str, Any]:
  return {
      "decisions": 0, "model_decisions": 0, "illegal_responses": 0, "fallbacks": 0,
      "folds": 0, "fold_opportunities": 0,
      "raise_calls": 0, "raise_opportunities": 0,
      "responses": 0, "accepts": 0, "declines": 0, "raise_backs": 0,
      "mao_play": 0, "mao_forfeit": 0, "talk_lines": 0,
      "requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
      "cost_usd": 0.0, "cost_responses": 0, "cost_known": True, "generation_secs": 0.0,
      "providers": collections.Counter(),
  }


def _tally(stats: dict[str, Any], action: str, legal: list[str], decision: truco_agents.Decision) -> None:
  stats["decisions"] += 1
  if decision.source in ("model", "fallback"):
    stats["model_decisions"] += 1
    stats["illegal_responses"] += len(decision.illegal)
    if decision.source == "fallback":
      stats["fallbacks"] += 1
  if "FOLD" in legal:
    stats["fold_opportunities"] += 1
    if action == "FOLD":
      stats["folds"] += 1
  if any(a in legal for a in RAISE_CALLS):
    stats["raise_opportunities"] += 1
    if action in RAISE_CALLS:
      stats["raise_calls"] += 1
  if "ACCEPT" in legal:
    stats["responses"] += 1
    stats[{"ACCEPT": "accepts", "DECLINE": "declines", "RAISE": "raise_backs"}[action]] += 1
  if action == "MAO_PLAY":
    stats["mao_play"] += 1
  if action == "MAO_FORFEIT":
    stats["mao_forfeit"] += 1
  if decision.talk:
    stats["talk_lines"] += 1
  for gr in decision.generate_returns:
    stats["requests"] += 1
    stats["prompt_tokens"] += gr.prompt_tokens or 0
    stats["completion_tokens"] += gr.generation_tokens or 0
    stats["reasoning_tokens"] += gr.reasoning_tokens or 0
    stats["generation_secs"] += gr.duration_success_only_secs or 0.0
    cost = openrouter_model.served_cost(gr)
    if cost is None:
      stats["cost_known"] = False
    else:
      stats["cost_responses"] += 1
      stats["cost_usd"] += cost
    provider = openrouter_model.served_provider(gr)
    if provider:
      stats["providers"][provider] += 1


def _finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
  out = dict(stats)
  out["providers"] = dict(stats["providers"])
  if not stats["cost_known"] and stats["requests"] == 0:
    out["cost_known"] = True
  return out


def redact_talk(text: str, talk: list[dict[str, Any]]) -> str:
  """Remove public table-talk text from a prompt before grepping for cards.

  Talk is unverified public speech (§11): a player may name any card, so a card
  string inside a talk line is not information the engine leaked.
  """
  for entry in sorted(talk, key=lambda t: -len(t["text"])):
    if entry["text"]:
      text = text.replace(entry["text"], "<talk>")
  return text


def check_isolation(engine: TrucoMatch, seat: int, text: str,
                    talk: list[dict[str, Any]] | None = None,
                    quoted_generation: str | None = None) -> None:
  """Raise if ``text`` names a card ``seat`` may not see.

  Public talk and the seat's own previous reply (quoted in a re-prompt) are
  redacted first: both are player-authored text, not engine information.
  """
  # Order matters: the quoted reply may itself contain talk lines, and talk
  # redaction would then break the exact match on the quoted block.
  scan = text
  if quoted_generation:
    scan = scan.replace(quoted_generation, "<previous reply>")
  scan = redact_talk(scan, talk or [])
  for card in engine.hidden_cards(seat):
    if card in scan:
      raise IsolationViolation(f"Prompt for seat {seat} contains hidden card {card}")


def play_match(
    spec: MatchSpec,
    agents: Mapping[int, truco_agents.Agent] | None = None,
    *,
    assert_isolation: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
  """Play one full match and return its summary dict."""
  logger: MatchLogger | None = None
  engine: TrucoMatch | None = None
  team_stats: dict[str, dict[str, Any]] = {}
  try:
    seat_configs = spec.seat_configs()
    if agents is None:
      by_config: dict[int, truco_agents.Agent] = {}
      agents = {}
      for seat, cfg in seat_configs.items():
        key = id(cfg)
        if key not in by_config:
          by_config[key] = cfg.build_agent()
        agents[seat] = by_config[key]

    engine = TrucoMatch(spec.seed, spec.num_players)
    logger = MatchLogger(spec.out_dir)
    seat_map = {
        str(seat): {"team": team_of(seat), "agent": agents[seat].name, **cfg.to_dict()}
        for seat, cfg in seat_configs.items()
    }
    logger.event({
        "source": "runner", "type": "match_config", "match_id": spec.match_id,
        "seed": spec.seed, "num_players": spec.num_players, "swap": spec.swap,
        "seats": seat_map,
        "system_instruction": truco_prompts.SYSTEM_INSTRUCTION,
        "code_version": truco_version.code_version(),
    })
    logger.replay(f"=== Match {spec.match_id} — seed {spec.seed}, N={spec.num_players}, swap={spec.swap} ===")
    for seat, cfg in seat_configs.items():
      logger.replay(f"seat {seat} (Team {team_of(seat)}): {cfg.display} provider={cfg.provider} "
                    f"options={cfg.model_options}")
    logger.flush_engine_events(engine)

    team_stats = {"A": _new_stats(), "B": _new_stats()}
    seat_stats = {s: _new_stats() for s in range(spec.num_players)}

    while not engine.is_terminal():
      seat = engine.current_actor()
      obs = engine.observation(seat)
      legal = engine.legal_actions(seat)
      decision = agents[seat].decide(obs, legal, lambda s=seat: engine.random_legal_action(s))

      for i, prompt in enumerate(decision.prompts):
        if assert_isolation:
          check_isolation(engine, seat, prompt["prompt_text"], obs["talk"],
                          prompt.get("quoted_generation"))
        gr = decision.generate_returns[i] if i < len(decision.generate_returns) else None
        logger.event({
            "source": "runner", "type": "prompt", "seat": seat, "hand": engine.hand_index,
            "trick": engine.trick_index, "attempt": prompt["attempt"],
            "decision_type": truco_prompts.decision_type_for(obs),
            "model": seat_configs[seat].display, "prompt_text": prompt["prompt_text"],
            "legal_actions": legal, "quoted_generation": prompt.get("quoted_generation"),
        })
        if gr is not None:
          logger.event({
              "source": "runner", "type": "response", "seat": seat, "hand": engine.hand_index,
              "trick": engine.trick_index, "attempt": prompt["attempt"],
              "model": seat_configs[seat].display,
              "main_response": gr.main_response,
              "main_response_and_thoughts": gr.main_response_and_thoughts,
              "extracted_action": prompt["extracted_action"],
              "matched_action": prompt["matched_action"],
              "served_provider": openrouter_model.served_provider(gr),
              "request_model": (gr.request_for_logging or {}).get("model"),
              "request_provider": (gr.request_for_logging or {}).get("provider"),
              "usage": {
                  "prompt_tokens": gr.prompt_tokens, "completion_tokens": gr.generation_tokens,
                  "reasoning_tokens": gr.reasoning_tokens, "total_tokens": gr.total_tokens,
                  "cost_usd": openrouter_model.served_cost(gr),
              },
              "duration_secs": gr.duration_success_only_secs,
          })
      for rec in decision.illegal:
        logger.event({
            "source": "runner", "type": "illegal_action", "seat": seat, "hand": engine.hand_index,
            "trick": engine.trick_index, "attempt": rec["attempt"],
            "raw_response": rec["raw_response"], "extracted_action": rec["extracted_action"],
            "legal_actions": rec["legal_actions"],
        })
        logger.replay(f"    !! illegal reply from seat {seat} (attempt {rec['attempt']}): "
                      f"extracted={rec['extracted_action']!r}")
      if decision.source == "fallback":
        logger.event({
            "source": "runner", "type": "fallback_action", "seat": seat, "hand": engine.hand_index,
            "trick": engine.trick_index, "action": decision.action, "legal_actions": legal,
        })
        logger.replay(f"    !! random fallback action for seat {seat}: {decision.action}")
      logger.event({
          "source": "runner", "type": "decision", "seat": seat, "hand": engine.hand_index,
          "trick": engine.trick_index, "action": decision.action, "talk": decision.talk,
          "decision_source": decision.source,
      })
      _tally(team_stats[team_of(seat)], decision.action, legal, decision)
      _tally(seat_stats[seat], decision.action, legal, decision)

      engine.apply_action(seat, decision.action, decision.talk)
      logger.flush_engine_events(engine)
      if verbose:
        print(f"[{spec.match_id}] hand {engine.hand_index} seat {seat} -> {decision.action}"
              f" ({decision.source}) score {engine.scores}")

    summary = {
        "match_id": spec.match_id,
        "seed": spec.seed,
        "num_players": spec.num_players,
        "swap": spec.swap,
        "winner_team": engine.winner,
        "winner_config": spec.config_for_team(engine.winner).display,
        "scores": dict(engine.scores),
        "hands_played": len(engine.hand_history),
        "hand_history": engine.hand_history,
        "team_config": {t: spec.config_for_team(t).to_dict() for t in ("A", "B")},
        "team_stats": {t: _finalize_stats(s) for t, s in team_stats.items()},
        "seat_stats": {str(s): _finalize_stats(st) for s, st in seat_stats.items()},
        "engine_event_count": len(engine.events),
        "out_dir": spec.out_dir,
        "code_version": truco_version.code_version(),
    }
    logger.close(engine, summary)
    return summary
  except Exception as e:  # pylint: disable=broad-exception-caught
    # Only tallied decisions are counted: requests made inside a ``decide()`` that raised are
    # not visible here, so callers must treat this as a lower bound (the Kaggle ``Budget``
    # charges a failed match at least its expected match cost).
    e.cost_usd_so_far = sum((s["cost_usd"] for s in team_stats.values()), 0.0)
    if logger is not None:
      error_event: dict[str, Any] = {
          "source": "runner", "type": "match_error", "match_id": spec.match_id,
          "error": f"{type(e).__name__}: {e}", "cost_usd_so_far": e.cost_usd_so_far,
      }
      if engine is not None:
        error_event["hand"] = engine.hand_index
      try:
        logger.event(error_event)
      except Exception:  # pylint: disable=broad-exception-caught
        pass  # e.g. the failure came from ``logger.close`` after the transcript was closed
      logger.close_transcript()
    raise


def make_spec(seed: int, team_a: ModelConfig, team_b: ModelConfig, *, num_players: int = 4,
              swap: bool = False, out_root: str | None = None) -> MatchSpec:
  match_id = f"seed{seed}_{'dup' if swap else 'orig'}"
  out_dir = f"{out_root}/{match_id}" if out_root else None
  return MatchSpec(match_id=match_id, seed=seed, num_players=num_players,
                   team_a=team_a, team_b=team_b, swap=swap, out_dir=out_dir)
