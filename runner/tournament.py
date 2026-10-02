"""Round-robin tournament: every pairing of N model configurations, resumable."""

from __future__ import annotations

import concurrent.futures
import itertools
import json
import os
import re
import traceback
from typing import Any, Callable, Sequence

from runner import ratings
from runner import stats as truco_stats
from runner.config import KIND_HEURISTIC, KIND_RANDOM, MatchSpec, ModelConfig
from runner.match_runner import play_match


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
  for spec in specs:
    prior = _load_summary(spec) if resume else None
    if prior is not None:
      summaries.append(prior)
    else:
      todo.append(spec)
  progress(f"Tournament: {len(models)} models, {len(pairings(models))} pairings, "
           f"{len(specs)} matches total, {len(summaries)} already done, {len(todo)} to play "
           f"(parallel={parallel}) -> {out_root}")
  failures: list[dict[str, str]] = []
  done = [len(summaries)]

  def _play(spec: MatchSpec):
    pairing = os.path.basename(os.path.dirname(spec.out_dir))
    try:
      s = play_fn(spec)
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

  # Interleave pairings so one slow model does not serialize the run.
  todo.sort(key=lambda sp: (sp.seed, sp.swap))
  if parallel > 1 and todo:
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as ex:
      results = list(ex.map(_play, todo))
  else:
    results = [_play(sp) for sp in todo]
  summaries.extend(r for r in results if r is not None)
  report = build_report(models, summaries, failures)
  with open(os.path.join(out_root, "tournament.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, sort_keys=True, ensure_ascii=False)
  return report


def build_report(models: Sequence[ModelConfig], summaries: list[dict[str, Any]],
                 failures: list[dict[str, str]] | None = None) -> dict[str, Any]:
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
                                  "margin_strength_ci", "win_rate_ci")},
        "model": l,
        "matches": cfg["matches"] if cfg else 0,
        "wins": cfg["wins"] if cfg else 0,
        "win_rate": cfg["match_win_rate"] if cfg else None,
        "points_per_hand": cfg["points_per_hand"] if cfg else None,
        "illegal_action_rate": cfg["illegal_action_rate"] if cfg else None,
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
  }


def format_report(report: dict[str, Any]) -> str:
  labels = [r["model"] for r in report["standings"]]
  w = max([len(l) for l in labels] + [8])
  lines = ["Standings (all matches):"]
  f = lambda v, fmt: ("n/a" if v is None else fmt.format(v))
  ci = lambda v, fmt: ("n/a" if v is None else f"[{fmt.format(v[0])}, {fmt.format(v[1])}]")
  lines.append(f"  {'model':<{w}}  matches  wins  win rate  win-rate 95% CI  BT Elo  BT 95% CI"
               "          margin/pair  pts/hand  illegal  $/match")
  for r in report["standings"]:
    lines.append(f"  {r['model']:<{w}}  {r['matches']:>7}  {r['wins']:>4}  {f(r['win_rate'], '{:8.3f}')}  "
                 f"{ci(r.get('win_rate_ci'), '{:.2f}'):>14}  {f(r.get('bt_elo'), '{:+.0f}'):>6}  "
                 f"{ci(r.get('bt_elo_ci'), '{:+.0f}'):>16}  {f(r.get('margin_strength'), '{:+.1f}'):>11}  "
                 f"{f(r['points_per_hand'], '{:8.3f}')}  {f(r['illegal_action_rate'], '{:7.3f}')}  "
                 f"{f(r['cost_per_match_usd'], '{:7.3f}')}")
  rat = report.get("ratings")
  if rat:
    status = rat.get("fit_status", "converged" if rat["connected"] else "disconnected")
    if status == "converged":
      lines.append(
          f"Ratings: Bradley–Terry Elo (anchor: {rat['anchor'] or 'mean of rated models'}, "
          f"+{rat['prior_pseudo_wins']} pseudo-wins per side per pairing); 95% CIs from "
          f"{rat['bootstrap']['resamples']} bootstrap resamples of (pairing, seed) units; "
          f"margin = points per duplicate pair; {rat['unpaired_matches']} unpaired match(es) "
          "excluded from margin.")
      if rat["bootstrap"].get("bt_failed_resamples", 0):
        lines.append(
            f"BT confidence intervals unavailable: {rat['bootstrap']['bt_failed_resamples']} "
            "bootstrap fit(s) did not converge.")
    elif status == "no_matches":
      lines.append("Ratings unavailable: no matches.")
    elif status == "not_converged":
      lines.append("Ratings unavailable: Bradley–Terry solver did not converge.")
    else:
      lines.append("Ratings unavailable: comparison graph is disconnected.")
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
  return "\n".join(lines)
