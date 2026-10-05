"""Round-robin tournament: every pairing of N model configurations, resumable."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import itertools
import json
import os
import re
import traceback
from typing import Any, Callable, Sequence

from runner import ratings
from runner import stats as truco_stats
from runner import version as truco_version
from runner.config import KIND_HEURISTIC, KIND_RANDOM, MatchSpec, ModelConfig
from runner.match_runner import play_match


_MISSING = object()


class MatchSkipped(RuntimeError):
  """Raised by a play_fn to decline a match without it being a failure."""


def _config_mismatches(prior: dict[str, Any],
                       expected: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
  """Return expected model-config keys whose prior values do not match."""
  fields = {field.name: field for field in dataclasses.fields(ModelConfig)}
  mismatches: dict[str, tuple[Any, Any]] = {}
  for key, new_value in expected.items():
    if key in prior:
      old_value = prior[key]
    elif key == "display":
      old_value = _MISSING
    else:
      field = fields[key]
      if field.default is not dataclasses.MISSING:
        old_value = field.default
      elif field.default_factory is not dataclasses.MISSING:
        old_value = field.default_factory()
      else:
        old_value = _MISSING
    if old_value is _MISSING or old_value != new_value:
      mismatches[key] = (old_value, new_value)
  return mismatches


def _prior_mismatches(spec: MatchSpec,
                      prior: dict[str, Any]) -> dict[str, dict[str, tuple[Any, Any]]]:
  """Compare a prior summary's match identity and team configs with the current match spec.

  The out_dir does not encode the player count, so the match-level fields are checked too.
  Every summary has recorded them since the first commit, so an absent one is a mismatch.
  ``code_version`` is deliberately not compared: resume exists to replay failed matches after
  a code fix.
  """
  mismatches: dict[str, dict[str, tuple[Any, Any]]] = {}
  expected_match = {"num_players": spec.num_players, "seed": spec.seed, "swap": spec.swap,
                    "match_id": spec.match_id}
  match_differences = {
      key: (prior.get(key, _MISSING), new_value) for key, new_value in expected_match.items()
      if prior.get(key, _MISSING) != new_value}
  if match_differences:
    mismatches["match"] = match_differences
  prior_teams = prior.get("team_config", {})
  for team in ("A", "B"):
    expected = spec.config_for_team(team).to_dict()
    old = prior_teams.get(team, {}) if isinstance(prior_teams, dict) else {}
    differences = _config_mismatches(old if isinstance(old, dict) else {}, expected)
    if differences:
      mismatches[f"team {team}"] = differences
  return mismatches


def _format_old(value: Any) -> str:
  """Format a prior config value for a resume mismatch message."""
  return "<missing>" if value is _MISSING else repr(value)


def _slug(label: str) -> str:
  return re.sub(r"[^A-Za-z0-9_.-]+", "_", label)


def pairings(models: Sequence[ModelConfig]) -> list[tuple[ModelConfig, ModelConfig]]:
  labels = [m.display for m in models]
  if len(set(labels)) != len(labels):
    raise ValueError(f"Model labels must be unique: {labels}")
  return list(itertools.combinations(models, 2))


def pairing_dir(out_root: str, a: ModelConfig, b: ModelConfig) -> str:
  return os.path.join(out_root, f"{_slug(a.display)}__vs__{_slug(b.display)}")


def preflight(models: Sequence[ModelConfig]) -> dict[str, str]:
  """One tiny request per model; returns {label: error} for models that fail."""
  from game_arena.harness import model_generation  # pylint: disable=import-outside-toplevel

  errors: dict[str, str] = {}
  for m in models:
    if m.kind in (KIND_RANDOM, KIND_HEURISTIC):
      continue
    try:
      agent = m.build_agent()
      model = agent._sampler._model  # pylint: disable=protected-access
      model.generate_with_text_input(model_generation.ModelTextInput(
          prompt_text="Reply with exactly: Final Answer: FOLD"))
    except Exception as e:  # pylint: disable=broad-exception-caught
      errors[m.display] = f"{type(e).__name__}: {str(e)[:300]}"
  return errors


def tournament_specs(models: Sequence[ModelConfig], *, seeds: Sequence[int], duplicate: bool,
                     out_root: str, num_players: int = 4,
                     exclude: Sequence[tuple[str, str]] = ()) -> list[MatchSpec]:
  excluded = {frozenset(pair) for pair in exclude}
  specs = []
  for a, b in pairings(models):
    if frozenset((a.display, b.display)) in excluded:
      continue
    root = pairing_dir(out_root, a, b)
    for seed in seeds:
      for swap in ((False, True) if duplicate else (False,)):
        match_id = f"seed{seed}_{'dup' if swap else 'orig'}"
        specs.append(MatchSpec(match_id=match_id, seed=seed, num_players=num_players,
                               team_a=a, team_b=b, swap=swap,
                               out_dir=os.path.join(root, match_id)))
  return specs


def _load_summary(spec: MatchSpec) -> dict[str, Any] | None:
  path = os.path.join(spec.out_dir, "summary.json")
  if not os.path.exists(path):
    return None
  with open(path, encoding="utf-8") as f:
    return json.load(f)


def run_tournament(
    models: Sequence[ModelConfig],
    *,
    seeds: Sequence[int],
    out_root: str,
    duplicate: bool = True,
    parallel: int = 1,
    num_players: int = 4,
    resume: bool = True,
    exclude: Sequence[tuple[str, str]] = (),
    progress: Callable[[str], None] = print,
    play_fn: Callable[..., dict[str, Any]] = play_match,
) -> dict[str, Any]:
  """Play every pairing; skip matches whose summary.json already exists."""
  os.makedirs(out_root, exist_ok=True)
  specs = tournament_specs(models, seeds=seeds, duplicate=duplicate, out_root=out_root,
                           num_players=num_players, exclude=exclude)
  summaries: list[dict[str, Any]] = []
  todo: list[MatchSpec] = []
  resume_mismatches: list[str] = []
  for spec in specs:
    prior = _load_summary(spec) if resume else None
    if prior is not None:
      mismatches = _prior_mismatches(spec, prior)
      if mismatches:
        details = []
        for scope, differences in mismatches.items():
          details.extend(
              f"{scope} {key}: old={_format_old(old)}, new={_format_old(new)}"
              for key, (old, new) in differences.items())
        resume_mismatches.append(f"{spec.out_dir}: " + "; ".join(details))
      else:
        summaries.append(prior)
    else:
      todo.append(spec)
  if resume_mismatches:
    raise ValueError(
        "Existing match summaries do not match the current match or team settings:\n"
        + "\n".join(f"  {mismatch}" for mismatch in resume_mismatches)
        + "\nUse a new output directory (or --no-resume) to run with different settings.")
  progress(f"Tournament: {len(models)} models, {len(pairings(models))} pairings, "
           f"{len(specs)} matches total, {len(summaries)} already done, {len(todo)} to play "
           f"(parallel={parallel}) -> {out_root}")
  failures: list[dict[str, str]] = []
  skipped: list[dict[str, str]] = []
  done = [len(summaries)]

  def _play(spec: MatchSpec):
    pairing = os.path.basename(os.path.dirname(spec.out_dir))
    try:
      s = play_fn(spec)
    except MatchSkipped as e:
      skipped.append({"pairing": pairing, "match_id": spec.match_id, "reason": str(e)})
      progress(f"  [{pairing}] {spec.match_id}: SKIPPED ({str(e)[:160]})")
      return None
    except Exception as e:  # pylint: disable=broad-exception-caught
      traceback.print_exc()
      failures.append({"pairing": pairing, "match_id": spec.match_id,
                       "error": f"{type(e).__name__}: {e}"})
      progress(f"  [{pairing}] {spec.match_id}: FAILED ({type(e).__name__}: {str(e)[:160]})")
      return None
    done[0] += 1
    progress(f"  [{pairing}] {spec.match_id}: {s['winner_config']} wins "
             f"{s['scores']['A']}-{s['scores']['B']} in {s['hands_played']} hands "
             f"({done[0]}/{len(specs)} done)")
    return s

  def _play_unit(unit: list[MatchSpec]):
    return [_play(sp) for sp in unit]

  # Interleave pairings so one slow model does not serialize the run. Each
  # pending orig/dup pair is one unit that a single worker plays in order, so
  # the original finishes before its duplicate starts: a pair-aware budget
  # then never admits a duplicate ahead of an original it would refuse.
  pair_key = lambda sp: (sp.seed, sp.team_a.display, sp.team_b.display)
  todo.sort(key=lambda sp: (*pair_key(sp), sp.swap))
  units = [list(g) for _, g in itertools.groupby(todo, key=pair_key)]
  if parallel > 1 and units:
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as ex:
      results = [r for rs in ex.map(_play_unit, units) for r in rs]
  else:
    results = [r for unit in units for r in _play_unit(unit)]
  summaries.extend(r for r in results if r is not None)
  report = build_report(models, summaries, failures, skipped)
  report["code_version"] = truco_version.code_version()
  with open(os.path.join(out_root, "tournament.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, sort_keys=True, ensure_ascii=False)
  return report


def build_report(models: Sequence[ModelConfig], summaries: list[dict[str, Any]],
                 failures: list[dict[str, str]] | None = None,
                 skipped: list[dict[str, str]] | None = None) -> dict[str, Any]:
  labels = [m.display for m in models]
  extra = {s["team_config"][t]["display"] for s in summaries for t in ("A", "B")} - set(labels)
  labels.extend(sorted(extra))
  anchor = next((m.display for m in models if m.kind == KIND_HEURISTIC), None)
  rat = ratings.compute_ratings(summaries, labels, anchor=anchor)
  by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
  for s in summaries:
    a, b = s["team_config"]["A"]["display"], s["team_config"]["B"]["display"]
    key = tuple(sorted((a, b)))
    by_pair.setdefault(key, []).append(s)
  head_to_head: dict[str, dict[str, Any]] = {l: {} for l in labels}
  pairing_stats: dict[str, Any] = {}
  for (x, y), ss in by_pair.items():
    wins_x = sum(1 for s in ss if s["winner_config"] == x)
    n = len(ss)
    head_to_head[x][y] = {"wins": wins_x, "matches": n, "win_rate": wins_x / n}
    head_to_head[y][x] = {"wins": n - wins_x, "matches": n, "win_rate": (n - wins_x) / n}
    pairing_stats[f"{x} vs {y}"] = truco_stats.aggregate(ss)
  for x, row in head_to_head.items():
    for y, cell in row.items():
      cell.update(rat["head_to_head"].get(x, {}).get(y, {}))
  overall = truco_stats.aggregate(summaries)
  standings = []
  for l in labels:
    cfg = overall["configs"].get(l)
    rm = rat["models"].get(l, {})
    standings.append({
        **{k: rm.get(k) for k in ("bt_elo", "bt_elo_ci", "margin_strength",
                                  "margin_strength_ci", "win_rate_ci", "tier")},
        "model": l,
        "matches": cfg["matches"] if cfg else 0,
        "wins": cfg["wins"] if cfg else 0,
        "win_rate": cfg["match_win_rate"] if cfg else None,
        "net_points_per_hand": cfg["net_points_per_hand"] if cfg else None,
        "illegal_action_rate": cfg["illegal_action_rate"] if cfg else None,
        "cost_coverage": cfg["cost_coverage"] if cfg else None,
        "cost_per_match_usd": cfg["estimated_cost_per_match_usd"] if cfg else None,
    })
  standings.sort(key=lambda r: (r["bt_elo"] is None, -(r["bt_elo"] or 0), -(r["win_rate"] or 0)))
  return {
      "ratings": rat,
      "models": [m.to_dict() for m in models],
      "matches_played": len(summaries),
      "standings": standings,
      "head_to_head": head_to_head,
      "pairings": pairing_stats,
      "overall": overall,
      "failures": failures or [],
      "skipped": skipped or [],
  }


def format_report(report: dict[str, Any]) -> str:
  labels = [r["model"] for r in report["standings"]]
  w = max([len(l) for l in labels] + [8])
  lines = ["Standings (all matches):"]
  f = lambda v, fmt: ("n/a" if v is None else fmt.format(v))
  ci = lambda v, fmt: ("n/a" if v is None else f"[{fmt.format(v[0])}, {fmt.format(v[1])}]")
  lines.append(f"  {'tier':>4}  {'model':<{w}}  matches  wins  win rate  win-rate 95% CI  BT Elo  BT 95% CI"
               "          margin/pair  net/hand  illegal  $/match  priced")
  for r in report["standings"]:
    tier = "n/a" if r.get("tier") is None else str(r["tier"])
    lines.append(f"  {tier:>4}  {r['model']:<{w}}  {r['matches']:>7}  {r['wins']:>4}  {f(r['win_rate'], '{:8.3f}')}  "
                 f"{ci(r.get('win_rate_ci'), '{:.2f}'):>14}  {f(r.get('bt_elo'), '{:+.0f}'):>6}  "
                 f"{ci(r.get('bt_elo_ci'), '{:+.0f}'):>16}  {f(r.get('margin_strength'), '{:+.1f}'):>11}  "
                 f"{f(r['net_points_per_hand'], '{:+7.3f}')}  {f(r['illegal_action_rate'], '{:7.3f}')}  "
                 f"{f(r['cost_per_match_usd'], '{:7.3f}')}  {f(r['cost_coverage'], '{:.0%}')}")
  rat = report.get("ratings")
  if rat:
    status = rat.get("fit_status", "converged" if rat["connected"] else "disconnected")
    if status == "converged":
      anchor_text = (f"anchor: {rat['anchor']}" if rat.get("anchor") is not None else
                     "zero-centred (mean of rated models)")
      if rat.get("anchor_skipped"):
        anchor_text += (f"; {rat['anchor_candidate']} not used as anchor: "
                        f"{rat['anchor_skipped']}")
      lines.append(
          f"Ratings: Bradley–Terry Elo ({anchor_text}, "
          f"+{rat['prior_pseudo_wins']} pseudo-wins per side per pairing); 95% CIs from "
          f"{rat['bootstrap']['resamples']} bootstrap resamples of whole seeds across pairings; "
          f"margin = points per duplicate pair; {rat['unpaired_matches']} unpaired match(es) "
          "excluded from margin.")
      if rat["bootstrap"].get("bt_failed_resamples", 0):
        lines.append(
            f"BT confidence intervals unavailable: {rat['bootstrap']['bt_failed_resamples']} "
            "bootstrap fit(s) did not converge.")
      if rat.get("tiers") is not None:
        tier_text = " | ".join(
            f"{number}: {', '.join(tier)}" for number, tier in enumerate(rat["tiers"], start=1))
        lines.append("Tiers (each tier ahead of the next in ≥95% of bootstrap resamples; "
                     f"order within a tier is not significant): {tier_text}")
    elif status == "no_matches":
      lines.append("Ratings unavailable: no matches.")
    elif status == "not_converged":
      lines.append("Ratings unavailable: Bradley–Terry solver did not converge.")
    else:
      lines.append("Ratings unavailable: comparison graph is disconnected.")
    if rat.get("bootstrap", {}).get("ms_failed_resamples", 0):
      lines.append(
          f"Margin confidence intervals unavailable: {rat['bootstrap']['ms_failed_resamples']} "
          "bootstrap fit(s) did not converge or were disconnected.")
  lines.append("")
  lines.append("Head-to-head win rate (row beats column), matches in parentheses:")
  cw = max(w, 12)
  lines.append(f"  {'':<{w}}  " + "  ".join(f"{l[:cw]:>{cw}}" for l in labels))
  for a in labels:
    cells = []
    for b in labels:
      if a == b:
        cells.append(f"{'—':>{cw}}")
      else:
        h = report["head_to_head"].get(a, {}).get(b)
        cells.append(f"{'n/a':>{cw}}" if h is None else f"{h['win_rate']:.2f} ({h['matches']})".rjust(cw))
    lines.append(f"  {a:<{w}}  " + "  ".join(cells))
  if report["failures"]:
    lines.append(f"\n{len(report['failures'])} match(es) failed and are excluded.")
  if report.get("skipped"):
    lines.append(f"\n{len(report['skipped'])} match(es) skipped (not started) and are excluded.")
  return "\n".join(lines)
