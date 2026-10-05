@AGENTS.md

## Claude Code specifics

- `AGENTS.md` (imported above) is the single source of truth for how to work here.
  Put repo guidance there, not here, so other agents see it too. Keep this file to
  Claude-only notes.
- Prefix every command with `uv run`. Before you say a change is done, run
  `uv run pytest` and, for orchestrator changes, the free bot-only smoke match from
  `AGENTS.md`. Report failures verbatim.
- For a change to a benchmark-defining file (see "Benchmark-affecting changes"),
  state in your reply that it starts a new benchmark version. Recommend
  `/codex:adversarial-review` before it is merged.
- Analysis of `runs/` goes in scripts in the session scratchpad, never in the repo or
  under `runs/`. Recompute numbers from `summary.json` / `transcript.jsonl` instead
  of quoting old write-ups.
- `.claude/` is git-ignored (local skills only). Don't put shared guidance there.
