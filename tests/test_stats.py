"""Aggregate statistics and human-facing stat tables."""

from __future__ import annotations

from typing import Any

from runner import stats


def summary(*, scores: dict[str, int], hands: int = 1,
            team_stats: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
  """Build a minimal match summary for aggregate-stat tests."""
  return {
      "team_config": {"A": {"display": "alpha"}, "B": {"display": "beta"}},
      "winner_team": "A" if scores["A"] > scores["B"] else "B",
      "scores": scores,
      "hands_played": hands,
      "team_stats": team_stats or {"A": {}, "B": {}},
  }


def test_aggregate_reports_net_points_per_hand():
  agg = stats.aggregate([summary(scores={"A": 9, "B": 3}, hands=3)])

  assert agg["configs"]["alpha"]["points_per_hand"] == 3
  assert agg["configs"]["alpha"]["net_points_per_hand"] == 2
  assert agg["configs"]["beta"]["net_points_per_hand"] == -2


def test_legacy_unknown_cost_is_excluded_from_priced_sum():
  ts = {"requests": 2, "cost_usd": 5.0, "cost_known": False}
  config = stats.aggregate([summary(scores={"A": 7, "B": 5}, team_stats={"A": ts, "B": {}})])[
      "configs"]["alpha"]

  assert config["cost_coverage"] == 0
  assert config["cost_per_request_usd"] is None
  assert config["estimated_cost_per_match_usd"] is None
  assert config["total_cost_usd"] is None
  assert config["totals"]["cost_responses"] == 0
  assert config["totals"]["cost_usd"] == 0


def test_partial_cost_coverage_extrapolates_over_all_requests():
  ts = {"requests": 4, "cost_responses": 2, "cost_usd": 3.0}
  config = stats.aggregate([summary(scores={"A": 7, "B": 5}, team_stats={"A": ts, "B": {}})])[
      "configs"]["alpha"]

  assert config["cost_coverage"] == 0.5
  assert config["cost_per_request_usd"] == 1.5
  assert config["estimated_cost_per_match_usd"] == 6.0
  assert config["total_cost_usd"] == 3.0


def test_format_table_has_net_points_and_priced_cost_rows():
  text = stats.format_table(stats.aggregate([summary(scores={"A": 7, "B": 5})]))

  assert "net points per hand" in text
  assert "priced responses" in text
  assert "est. cost / match (USD)" in text
  assert "known cost (USD)" in text
