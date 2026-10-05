# Working in truco-eval

Guide for anyone, human or coding agent, who changes this repository. `README.md`
explains what the benchmark is and how to run it; this file explains how to change
it without breaking its results.

## What the repository is

A Truco Paulista LLM-vs-LLM benchmark: 4 players, 2 teams, first to 12. It has
three layers, and dependencies only point downward:

| Layer | Files | May import |
|---|---|---|
| Engine | `truco/cards.py`, `truco/match.py` | stdlib only |
| Model layers | `openrouter_model.py` (OpenRouter), `kbench_model.py` (Kaggle Model Proxy) | `game_arena` harness, `requests` |
| Orchestrator | `runner/` | `truco`, model layers, harness |
| Kaggle tasks and ops | `kaggle_task/` | `runner` (installed from wheels on Kaggle) |

`runner/` module map:

- `rules_text`, `prompts`, `render`: what a seat is told.
- `parsers`, `sampler`, `agents`: how a reply becomes an action, including re-prompts and the random fallback.
- `heuristic`: the deterministic baseline bot.
- `config`: `ModelConfig` and `MatchSpec`.
- `match_runner`, `match_log`: the match loop, per-match stats and output files.
- `stats`, `ratings`, `tournament`: aggregates, Bradley–Terry ratings and round-robins.
- `replay`: re-drives a logged match.
- `kbench_task`: Kaggle preflight and budget.
- `cli`: the entry point.

`AMBIGUITIES.md` records every rule edge case and how it was resolved. `docs/` is a
static snapshot of the explainer page. Treat it as an export and don't edit it by hand.

## Commands

Always go through `uv`. The system `python3` is too old, and bare `python` is not on PATH.

```bash
uv sync                                   # install, including dev tools
uv run pytest                             # full suite, about 15 s, no network
uv run pytest tests/test_ratings.py -k anchor   # one area

# Free end-to-end check: bots only, no API key, no cost
uv run python -m runner.cli run --team-a heuristic --team-b random --seeds 2 --out /tmp/truco-smoke
uv run python -m runner.cli replay /tmp/truco-smoke/seed0_orig/engine_events.jsonl

# Kaggle daily driver: --dry-run only reads Kaggle status and logs; --merge rewrites the
# leaderboard files under runs/. Without either flag it can push a paid run.
uv run python kaggle_task/daily.py --dry-run
# Kaggle task with bots in place of models (a few seconds). It writes truco_runs/ and
# *.run.json / *.task.json into the current directory, so run it from a scratch dir:
REPO=$PWD; mkdir -p /tmp/truco-kaggle-dry && cd /tmp/truco-kaggle-dry && \
  TRUCO_DRY_RUN=1 uv run --project "$REPO" python "$REPO/kaggle_task/truco_cheap_rr.py"
```

No linter or formatter is configured. Match the surrounding style by hand (see
"Code conventions").

## Invariants: break these and published results stop meaning anything

1. **The engine is pure and deterministic.**
   - `truco/` imports nothing from `runner`, the harness, or the network.
   - The deal of hand *k* depends only on `(seed, k)`.
   - `engine_events.jsonl` is a pure function of `(seed, actions)`. `runner.cli replay` checks this byte for byte against every existing run.
   - Changing an engine event's fields, their order or their JSON encoding breaks replay of old runs. Only do it deliberately, and say so.
2. **Information isolation.**
   - A prompt may contain only `TrucoMatch.observation(seat)` plus the legal list.
   - `match_runner.check_isolation` raises if a hidden card appears in a prompt. Never weaken it, and never default `--no-isolation-check` to on.
   - `tests/test_runner_match.py::test_every_prompt_is_information_isolated` must keep passing.
3. **The illegal-action policy.** One re-prompt (`max_reprompts=1`), then a random legal action drawn from the engine's seed-derived fallback stream. That stream is separate from the deal stream, so fallbacks never change the cards.
4. **Duplicate matches.**
   - `seed{N}_orig` and `seed{N}_dup` see identical deals with the teams swapped across seats.
   - Even seats (team A) lead hand 1 and win about 56% of matches. Only complete orig+dup pairs are seat-fair.
   - Keep pair-level logic, such as margin, pair-aware; the bootstrap resamples whole seeds across pairings (orig and dup of a seed still move together).

### Benchmark-affecting changes

Some files define the benchmark itself, not just the code that runs it:

- `truco/` game rules and deals
- `runner/rules_text.py`, `prompts.py`, `render.py`: the system instruction and observation text
- `runner/parsers.py`, `sampler.py`: what counts as a legal reply
- `runner/heuristic.py`: its thresholds are the rating anchor
- model defaults (`DEFAULT_MAX_TOKENS = 16384`, retry policy)
- stat or rating definitions in `stats.py` and `ratings.py`

A change to any of these is a new benchmark version. Results produced before and
after it must not be merged into one leaderboard.

When you make such a change:
- say so in the commit message;
- update `AMBIGUITIES.md` if it touches the rules;
- update the README sections that describe the behavior;
- add a test that pins the new behavior.

Outputs do not yet record a code version. Until they do, the commit date is the only
way to tell versions apart, so keep these changes in commits of their own.

## How to make common changes

**Rule fix or new edge case**
- Change `truco/match.py`.
- Add an entry to `AMBIGUITIES.md` citing the rules section, and reference it in the code (`# AMBIGUITIES #n`).
- Add a scripted-deal test (`tests/conftest.py`: `deal(...)`, `act(...)`).
- Run the 10,000-match property test (`test_determinism_isolation_property.py`).

**CLI flag**
- `runner/cli.py:build_parser` → `cmd_run` / `cmd_tournament`.
- A per-model flag must also go through `model_config_from_args` and the models-file loop in `cmd_tournament`.
- Update the README flags table.
- Add a `cli.main([...])` test that uses the random or heuristic bots.

**Model kind**
- Add `KIND_X` in `runner/config.py`, and a branch in `ModelConfig.build_agent` that lazy-imports any SDK.
- Update both CLI constructors and the preflight skip lists (`tournament.preflight`, `kbench_task.quick_preflight`).
- The model class subclasses the harness `MultimodalModel`. Its `response_for_logging` must include `provider` and `usage.cost` (USD), or cost shows as n/a.

**Per-match stat**
- `match_runner._new_stats` and `_tally` → `stats._SUM_KEYS` → a rate in `stats.aggregate` via `_rate`, which returns `None` on a zero denominator.
- Add a row in `format_table`, optionally the tournament report, then the README "Interpreting the outputs" section.

**Ratings**
- `runner/ratings.py` is fully specified in its module docstring. Keep the docstring, the README "Ratings" section and the code in agreement.
- The RNG seed is fixed so that reports are deterministic.

**Kaggle task or roster**
- Edit `kaggle_task/truco_cheap_rr.py` (the template). `daily.py` renders the copies, so don't edit the rendered ones in `runs/kaggle_daily/`.
- Kaggle runs execute the **wheels in the `lfpmb1/truco-eval-wheels` dataset**, not this working tree. A change to `truco/`, `runner/` or the model layers only reaches Kaggle after you:
  - rebuild with `uv build --wheel` plus `pip wheel --no-deps` of the pinned `game_arena` commit into `kaggle_task/dataset/`;
  - push a new dataset version.
  Both steps are outward-facing, so ask first.

## Code conventions (de facto; follow them)

- **Layout:** 2-space indentation, 4-space hanging continuation (Google/pyink style, as in `game_arena`), about 100 columns, double quotes.
- **Imports:**
  - Put `from __future__ import annotations` at the top of every module.
  - Use absolute imports only.
  - Alias a module when its name collides with a harness module: `from runner import agents as truco_agents`, `from truco import cards as C`.
  - Lazy-import optional SDKs inside the function, with `# pylint: disable=import-outside-toplevel`.
- **Typing:**
  - Use PEP 604/585 syntax (`X | None`, `list[str]`, `dict[str, Any]`), with `Any`, `Sequence`, `Mapping` and `Callable` from `typing`.
  - Type JSON-shaped data as `dict[str, Any]`.
- **Data:**
  - Configs and value records are frozen `kw_only` dataclasses (`ModelConfig`, `MatchSpec`, `Decision`, `Deal`).
  - Anything serialized stays a plain dict: observations, events, summaries, reports.
- **Vocabulary:**
  - `seat` is an int from 0 to N−1.
  - `team` is `"A"` or `"B"`, taken from `truco.match.team_of(seat)` (even seats are A).
  - `label` / `display` is a model's name in every aggregate.
  - `slug` is the provider's model id.
  - `swap=True` marks the duplicate match, and `match_id` is `seed{N}_{orig|dup}`.
  - Actions are exact uppercase strings (`PLAY Q♠`, `TRUCO`, `ACCEPT`, `MAO_PLAY`, `FOLD`).
  - Import raise names, phases and hand types from `truco.match` instead of retyping the literals. Some older modules still retype them; don't add more.
- **Docstrings:**
  - Every module starts with a docstring that states its purpose and the policy it implements; `heuristic.py` and `ratings.py` are the models to follow.
  - Functions get a one-line summary, with ``double backticks`` for code.
  - Comments explain *why*, and cite rule sections (`§12`) or `AMBIGUITIES #n`.
- **Errors:**
  - Raise `ValueError` for bad input. Domain exceptions subclass builtins (`IllegalActionError(ValueError)`, `IsolationViolation(AssertionError)`, `BudgetExhausted(RuntimeError)`).
  - Raise `DoNotRetryError` in model layers for non-retryable 4xx.
  - Catch broad exceptions only at per-match or per-model boundaries, and record them as `f"{type(e).__name__}: {e}"`.
- **I/O:**
  - Use `os.path` (not pathlib) and `open(..., encoding="utf-8")`.
  - Write JSONL with `sort_keys=True, ensure_ascii=False, separators=(",", ":")`, and readable JSON with `indent=2, sort_keys=True, ensure_ascii=False`.
  - Progress output is `print(..., flush=True)`.
- **Tests:**
  - Write plain pytest functions named `test_<behavior>`.
  - Engine helpers come from `tests/conftest.py`, imported explicitly.
  - Use `tests/fake_model.FakeModel` in place of LLMs, `tmp_path` for outputs, and `monkeypatch` or `play_fn=` injection for stubbing. The OpenRouter tests use a local mock HTTP server.
  - Tests never touch the network, and never assume `runs/` exists.

## Data in `runs/` (git-ignored, about 1.2 GB, not reproducible)

- **Never delete, move or rewrite anything under `runs/`.** Each match there cost real money and cannot be re-queried identically, because LLM sampling is not deterministic.
- Write new experiments to a new directory.
- `runs/flash200_full/` is the 200-match DeepSeek vs GLM study. `runs/flash200/` is an older, strict subset of it, so never count both.
- `runs/kaggle_cheap_rr*/`, `runs/kaggle_daily/state.json` and `runs/kaggle_cheap_rr_leaderboard.txt` are owned by the daily job. Read them; don't edit them.
- `compare/`, `demo/`, `free_pilot/` and `kaggle_pilot/` are smoke tests and are not part of any leaderboard.

## Things that cost money or act outside the repo: ask first

- **OpenRouter runs** (`runner.cli run|tournament` with model slugs) bill the user's key. A preflight request is made per model. Pass the key inline (`OPENROUTER_API_KEY=... uv run ...`), because it is not exported in agent shells.
- **Kaggle:**
  - `kaggle b t push` runs a task as soon as it is pushed, and spends the Model Proxy quota.
  - `kaggle_task/daily.py` with no flag can push; `--merge` doesn't push, but it rewrites the leaderboard files under `runs/`.
  - A launchd agent (`com.lfpmb.truco-kaggle-daily`) already runs `daily.py` every 30 minutes, so don't run it alongside, and don't load or unload it, unless asked.
- **Never** commit, push, push a Kaggle dataset version, or publish artifacts unless explicitly asked.

## Git

- `main` is the default branch, and work lands through PRs.
- Commit subjects are imperative and describe the outcome ("Make tournament comparisons schedule-adjusted and baseline-anchored").
- The body explains why, and lists any behavior or benchmark change.
- Leave unrelated working-tree changes alone.

## Known issues (as of 2026-10-04)

Fix these on purpose, not as a side effect of other work:

- The live parser never matches suit words ("K of diamonds"): the harness strips spaces before `soft_match` runs. Parser tests call `soft_match` directly, so they miss this. Test through `TrucoSampler`.
- Isolation is checked after the model replies, not before the prompt is sent (the README says "before").
- `runner/match_runner.py` imports `openrouter_model` eagerly, which pulls in `requests`.
- `kaggle_task/daily.py` (unattended, and it deletes directories) and `kaggle_task/truco_pilot.py` (stale) have no tests.
