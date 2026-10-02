"""Match runner CLI.

Examples:
  python -m runner.cli run --team-a openai/gpt-5-mini --team-b deepseek/deepseek-chat-v3.1 \
      --seeds 2 --duplicate --out runs/demo
  python -m runner.cli run --team-a random --team-b random --seeds 5 --out runs/bots
  python -m runner.cli replay runs/demo/seed0_orig/engine_events.jsonl
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import traceback
from typing import Any

from runner import stats as truco_stats
from runner.config import KIND_HEURISTIC, KIND_OPENROUTER, KIND_RANDOM, ModelConfig
from runner.match_runner import make_spec, play_match
from runner.replay import replay_file


def _json_arg(text: str | None) -> dict[str, Any] | None:
  if text is None:
    return None
  if os.path.exists(text):
    with open(text, encoding="utf-8") as f:
      return json.load(f)
  return json.loads(text)


def model_config_from_args(slug: str, options: str | None, provider: str | None,
                           label: str | None, base_url: str | None,
                           max_reprompts: int, api_options: str | None) -> ModelConfig:
  if slug.lower() == "random":
    return ModelConfig(kind=KIND_RANDOM, label=label or "random", max_reprompts=max_reprompts)
  if slug.lower() == "heuristic":
    return ModelConfig(kind=KIND_HEURISTIC, label=label or "heuristic", max_reprompts=max_reprompts)
  return ModelConfig(
      kind=KIND_OPENROUTER, slug=slug, model_options=_json_arg(options) or {},
      provider=_json_arg(provider), label=label or "", base_url=base_url,
      max_reprompts=max_reprompts, api_options=_json_arg(api_options) or {},
  )


def build_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(prog="runner.cli", description="Truco LLM benchmark runner")
  sub = p.add_subparsers(dest="command", required=True)

  run = sub.add_parser("run", help="play seeded matches between two model configurations")
  run.add_argument("--team-a", required=True, help="OpenRouter slug, 'random', or 'heuristic'")
  run.add_argument("--team-b", required=True, help="OpenRouter slug, 'random', or 'heuristic'")
  for t in ("a", "b"):
    run.add_argument(f"--team-{t}-options", help="JSON dict or path: temperature, top_p, top_k, max_tokens, reasoning, ...")
    run.add_argument(f"--team-{t}-provider", help='JSON dict or path, e.g. {"order":["DeepInfra"],"allow_fallbacks":false}')
    run.add_argument(f"--team-{t}-label", help="display label for aggregate stats")
    run.add_argument(f"--team-{t}-api-options", help="JSON dict: timeout, referer, title")
  run.add_argument("--num-players", type=int, default=4, choices=(2, 4, 6))
  run.add_argument("--seeds", type=int, default=1, help="number of seeds to play")
  run.add_argument("--start-seed", type=int, default=0)
  run.add_argument("--duplicate", action=argparse.BooleanOptionalAction, default=True,
                   help="also play each seed with teams swapped across seats")
  run.add_argument("--out", required=True, help="output directory")
  run.add_argument("--parallel", type=int, default=1, help="matches to play concurrently")
  run.add_argument("--max-reprompts", type=int, default=1)
  run.add_argument("--base-url", help="override the chat-completions URL (self-hosted endpoints)")
  run.add_argument("--verbose", action="store_true")
  run.add_argument("--no-isolation-check", action="store_true",
                   help="skip the runtime hidden-card check on every prompt")

  tr = sub.add_parser("tournament", help="round-robin: every pairing of the given models")
  tr.add_argument("--models-file", required=True,
                  help='JSON list of {"slug","label","provider","model_options"} (slug "random" or "heuristic" for a bot)')
  tr.add_argument("--seeds", type=int, default=100)
  tr.add_argument("--start-seed", type=int, default=0)
  tr.add_argument("--duplicate", action=argparse.BooleanOptionalAction, default=True)
  tr.add_argument("--num-players", type=int, default=4, choices=(2, 4, 6))
  tr.add_argument("--out", required=True)
  tr.add_argument("--parallel", type=int, default=1)
  tr.add_argument("--max-reprompts", type=int, default=1)
  tr.add_argument("--no-resume", action="store_true", help="replay matches that already have a summary")
  tr.add_argument("--exclude", action="append", default=[], metavar="LABEL_A/LABEL_B",
                  help="skip this pairing (repeatable), e.g. a pairing already run elsewhere")
  tr.add_argument("--no-preflight", action="store_true",
                  help="skip the one-request-per-model check before starting")

  rp = sub.add_parser("replay", help="re-drive an engine_events.jsonl and verify byte-identity")
  rp.add_argument("path")
  rp.add_argument("--print", action="store_true", help="print the rebuilt replay events")
  return p


def cmd_run(args: argparse.Namespace) -> int:
  team_a = model_config_from_args(args.team_a, args.team_a_options, args.team_a_provider,
                                  args.team_a_label, args.base_url, args.max_reprompts,
                                  args.team_a_api_options)
  team_b = model_config_from_args(args.team_b, args.team_b_options, args.team_b_provider,
                                  args.team_b_label, args.base_url, args.max_reprompts,
                                  args.team_b_api_options)
  if team_a.display == team_b.display:
    team_a = ModelConfig(**{**team_a.to_dict_no_display(), "label": team_a.display + "#A"})
    team_b = ModelConfig(**{**team_b.to_dict_no_display(), "label": team_b.display + "#B"})
  os.makedirs(args.out, exist_ok=True)
  specs = []
  for seed in range(args.start_seed, args.start_seed + args.seeds):
    specs.append(make_spec(seed, team_a, team_b, num_players=args.num_players,
                           swap=False, out_root=args.out))
    if args.duplicate:
      specs.append(make_spec(seed, team_a, team_b, num_players=args.num_players,
                             swap=True, out_root=args.out))
  print(f"Playing {len(specs)} match(es): {team_a.display} vs {team_b.display} "
        f"(N={args.num_players}, duplicate={args.duplicate}) -> {args.out}")

  failures: list[dict[str, str]] = []

  def _play(spec):
    try:
      summary = play_match(spec, assert_isolation=not args.no_isolation_check, verbose=args.verbose)
    except Exception as e:  # pylint: disable=broad-exception-caught
      # One broken match (provider outage, hard 4xx, ...) must not sink the run.
      traceback.print_exc()
      failures.append({"match_id": spec.match_id, "error": f"{type(e).__name__}: {e}"})
      print(f"  {spec.match_id}: FAILED ({type(e).__name__}: {str(e)[:200]})", flush=True)
      return None
    print(f"  {spec.match_id}: Team {summary['winner_team']} ({summary['winner_config']}) wins "
          f"{summary['scores']['A']}-{summary['scores']['B']} in {summary['hands_played']} hands",
          flush=True)
    return summary

  if args.parallel > 1:
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as ex:
      results = list(ex.map(_play, specs))
  else:
    results = [_play(s) for s in specs]
  summaries = [r for r in results if r is not None]

  agg = truco_stats.aggregate(summaries)
  with open(os.path.join(args.out, "aggregate.json"), "w", encoding="utf-8") as f:
    json.dump({"aggregate": agg, "matches": summaries, "failures": failures}, f, indent=2,
              sort_keys=True, ensure_ascii=False)
  print()
  print(truco_stats.format_table(agg))
  if failures:
    print(f"\n{len(failures)} match(es) FAILED and are excluded from the aggregate:")
    for f in failures:
      print(f"  {f['match_id']}: {f['error'][:300]}")
  print(f"\nWrote {os.path.join(args.out, 'aggregate.json')}")
  return 1 if failures else 0


def cmd_tournament(args: argparse.Namespace) -> int:
  from runner import tournament  # pylint: disable=import-outside-toplevel

  with open(args.models_file, encoding="utf-8") as f:
    entries = json.load(f)
  models = []
  for e in entries:
    slug = e["slug"]
    if slug.lower() == "random":
      models.append(ModelConfig(kind=KIND_RANDOM, label=e.get("label", "random"),
                                max_reprompts=args.max_reprompts))
    elif slug.lower() == "heuristic":
      models.append(ModelConfig(kind=KIND_HEURISTIC, label=e.get("label", "heuristic"),
                                max_reprompts=args.max_reprompts))
    else:
      models.append(ModelConfig(kind=KIND_OPENROUTER, slug=slug, label=e.get("label", ""),
                                provider=e.get("provider"), model_options=e.get("model_options", {}),
                                api_options=e.get("api_options", {}), base_url=e.get("base_url"),
                                max_reprompts=args.max_reprompts))
  exclude = []
  for item in args.exclude:
    if "/" not in item:
      raise SystemExit(f"--exclude expects LABEL_A/LABEL_B, got {item!r}")
    exclude.append(tuple(item.split("/", 1)))
  if not args.no_preflight:
    errors = tournament.preflight(models)
    if errors:
      for label, err in errors.items():
        print(f"PREFLIGHT FAILED for {label}: {err}", flush=True)
      return 2
    print("Preflight OK: every model answered one request.", flush=True)
  report = tournament.run_tournament(
      models, seeds=range(args.start_seed, args.start_seed + args.seeds), out_root=args.out,
      duplicate=args.duplicate, parallel=args.parallel, num_players=args.num_players,
      resume=not args.no_resume, exclude=exclude, progress=lambda m: print(m, flush=True),
  )
  print()
  print(tournament.format_report(report))
  print(f"\nWrote {os.path.join(args.out, 'tournament.json')}")
  return 1 if report["failures"] else 0


def cmd_replay(args: argparse.Namespace) -> int:
  match, identical = replay_file(args.path)
  print(f"Rebuilt match: winner Team {match.winner}, scores {match.scores}, "
        f"{len(match.hand_history)} hands, {len(match.events)} events")
  print("Event log byte-identical to the file: " + ("YES" if identical else "NO"))
  if args.print:
    sys.stdout.write(match.serialize_events())
  return 0 if identical else 1


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  if args.command == "run":
    return cmd_run(args)
  if args.command == "tournament":
    return cmd_tournament(args)
  return cmd_replay(args)


if __name__ == "__main__":
  sys.exit(main())
