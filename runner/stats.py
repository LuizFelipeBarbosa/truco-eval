"""Aggregate statistics across match summaries, per model configuration."""

from __future__ import annotations

import collections
from typing import Any, Iterable

_SUM_KEYS = ("decisions", "model_decisions", "illegal_responses", "fallbacks", "folds",
             "fold_opportunities", "raise_calls", "raise_opportunities", "responses",
             "accepts", "declines", "raise_backs", "mao_play", "mao_forfeit", "talk_lines",
             "requests", "cost_responses", "prompt_tokens", "completion_tokens", "reasoning_tokens",
             "cost_usd", "generation_secs")


def _rate(num: float, den: float) -> float | None:
  return None if not den else num / den


def aggregate(summaries: Iterable[dict[str, Any]]) -> dict[str, Any]:
  per: dict[str, dict[str, Any]] = {}
  matches = 0
  for s in summaries:
    matches += 1
    for team in ("A", "B"):
      label = s["team_config"][team]["display"]
      agg = per.setdefault(label, {
          "matches": 0, "wins": 0, "points": 0, "net_points": 0, "hands": 0,
          "providers": collections.Counter(), **{k: 0 for k in _SUM_KEYS},
      })
      agg["matches"] += 1
      agg["wins"] += int(s["winner_team"] == team)
      agg["points"] += s["scores"][team]
      other_team = "B" if team == "A" else "A"
      agg["net_points"] += s["scores"][team] - s["scores"][other_team]
      agg["hands"] += s["hands_played"]
      ts = s["team_stats"][team]
      for k in _SUM_KEYS:
        if k not in ("cost_responses", "cost_usd"):
          agg[k] += ts.get(k, 0) or 0
      if "cost_responses" in ts:
        cost_responses = ts.get("cost_responses", 0) or 0
        agg["cost_usd"] += ts.get("cost_usd", 0) or 0
      elif ts.get("cost_known", True):
        cost_responses = ts.get("requests", 0) or 0
        agg["cost_usd"] += ts.get("cost_usd", 0) or 0
      else:
        # A legacy summary with an unknown cost does not tell us which responses were priced.
        cost_responses = 0
      agg["cost_responses"] += cost_responses
      agg["providers"].update(ts.get("providers", {}))

  out: dict[str, Any] = {"matches": matches, "configs": {}}
  for label, a in per.items():
    cost_per_request = _rate(a["cost_usd"], a["cost_responses"])
    estimated_cost = None if cost_per_request is None else _rate(
        cost_per_request * a["requests"], a["matches"])
    out["configs"][label] = {
        "matches": a["matches"],
        "wins": a["wins"],
        "match_win_rate": _rate(a["wins"], a["matches"]),
        "points_per_hand": _rate(a["points"], a["hands"]),
        "net_points_per_hand": _rate(a["net_points"], a["hands"]),
        "hands_per_match": _rate(a["hands"], a["matches"]),
        "illegal_action_rate": _rate(a["illegal_responses"], a["model_decisions"]),
        "fallback_rate": _rate(a["fallbacks"], a["model_decisions"]),
        "fold_rate": _rate(a["folds"], a["fold_opportunities"]),
        "raise_call_rate": _rate(a["raise_calls"], a["raise_opportunities"]),
        "accept_rate": _rate(a["accepts"], a["responses"]),
        "decline_rate": _rate(a["declines"], a["responses"]),
        "raise_back_rate": _rate(a["raise_backs"], a["responses"]),
        "mao_forfeit_rate": _rate(a["mao_forfeit"], a["mao_play"] + a["mao_forfeit"]),
        "talk_rate": _rate(a["talk_lines"], a["decisions"]),
        "prompt_tokens_per_match": _rate(a["prompt_tokens"], a["matches"]),
        "completion_tokens_per_match": _rate(a["completion_tokens"], a["matches"]),
        "reasoning_tokens_per_match": _rate(a["reasoning_tokens"], a["matches"]),
        "requests_per_match": _rate(a["requests"], a["matches"]),
        "cost_coverage": _rate(a["cost_responses"], a["requests"]),
        "cost_per_request_usd": cost_per_request,
        "estimated_cost_per_match_usd": estimated_cost,
        "total_cost_usd": a["cost_usd"] if a["cost_responses"] else None,
        "generation_secs_per_match": _rate(a["generation_secs"], a["matches"]),
        "providers": dict(a["providers"]),
        "totals": {k: a[k] for k in _SUM_KEYS},
    }
  return out


def _fmt(v: Any) -> str:
  if v is None:
    return "n/a"
  if isinstance(v, float):
    return f"{v:.3f}" if abs(v) < 100 else f"{v:,.0f}"
  return str(v)


def format_table(agg: dict[str, Any]) -> str:
  rows = [
      ("matches", "matches"), ("match win rate", "match_win_rate"),
      ("net points per hand", "net_points_per_hand"), ("hands per match", "hands_per_match"),
      ("illegal-action rate", "illegal_action_rate"), ("fallback rate", "fallback_rate"),
      ("fold rate", "fold_rate"), ("raise call rate", "raise_call_rate"),
      ("accept rate", "accept_rate"), ("decline rate", "decline_rate"),
      ("raise-back rate", "raise_back_rate"), ("mão de onze forfeit rate", "mao_forfeit_rate"),
      ("talk rate", "talk_rate"), ("requests / match", "requests_per_match"),
      ("prompt tokens / match", "prompt_tokens_per_match"),
      ("completion tokens / match", "completion_tokens_per_match"),
      ("reasoning tokens / match", "reasoning_tokens_per_match"),
      ("priced responses", "cost_coverage"),
      ("est. cost / match (USD)", "estimated_cost_per_match_usd"),
      ("known cost (USD)", "total_cost_usd"),
  ]
  labels = list(agg["configs"])
  width = max([len(r[0]) for r in rows] + [10])
  col = max([len(l) for l in labels] + [12])
  lines = [f"{'':<{width}}  " + "  ".join(f"{l:>{col}}" for l in labels)]
  for name, key in rows:
    lines.append(f"{name:<{width}}  " + "  ".join(
        f"{_fmt(agg['configs'][l][key]):>{col}}" for l in labels))
  for l in labels:
    prov = agg["configs"][l]["providers"]
    if prov:
      lines.append(f"providers for {l}: {prov}")
  return "\n".join(lines)
