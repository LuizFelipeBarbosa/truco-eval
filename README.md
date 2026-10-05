# Truco LLM benchmark

Reproducible LLM-vs-LLM matches of **Truco Paulista** (4-player, 40-card deck,
first to 12), built from three fixed pieces:

| Piece | Where | What |
|---|---|---|
| Game engine | `truco/` | Standalone deterministic Python engine with an OpenSpiel-flavored interface. No LLM or harness imports. |
| Model layer | `openrouter_model.py` | `OpenRouterModel`, a `game_arena.harness.model_generation.MultimodalModel` following the harness's `TogetherAIModel` pattern. |
| Orchestrator | `runner/` | Per-seat prompts with strict information isolation, a custom sampler implementing the illegal-action policy, JSONL logging, aggregate stats, CLI. |

`AMBIGUITIES.md` lists every rule edge case that the rules document leaves open,
with the resolution the engine implements.

## Write-ups

Two companion pages explain the benchmark and analyse its results. Both are
hosted on claude.ai and open only for people they have been shared with.

| Page | What it covers |
|---|---|
| [Inside Truco Eval](https://claude.ai/artifact/PpCSejFFUEf8iPG1UJegRL) | Interactive, diagram-led explainer: the game, a step-through of one full hand (`seed0_orig`), the harness, and results of the 200-match run. Each figure can be copied as SVG. |
| [Truco Eval Overview](https://claude.ai/artifact/T6VKyMp9hG5PUEn6LRRerf) | A doc with four tabs. **Overview**: project structure and how a run works. **Interview deep dive**: architecture, design trade-offs, limitations, rehearsal questions. **Potential vs efficiency**: quiz-style intelligence versus in-game efficiency, read through the 200-match `deepseek-v4.1-flash` vs `glm-5.3-flash` run (risk profiles, bluffing, reasoning-overflow failures, cost per win). **Play by play**: full replays of the demo matches. |

A snapshot of Inside Truco Eval is kept in `docs/inside-truco-eval.html`. It is
one self-contained file with its data embedded, so it opens in any browser.
The hosted page is the live version; refresh the snapshot after editing it.

The 200-match analysis is computed from the transcripts in
`runs/flash200_full/` (git-ignored, not in this repository).

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/). Python 3.11+ is
required by the Game Arena harness; `.python-version` pins 3.12.

```bash
uv sync                      # creates .venv, installs the harness (pinned git commit) and dev tools
export OPENROUTER_API_KEY=sk-or-...
```

Game Arena is declared in `pyproject.toml` as a git dependency pinned to a
commit (`[tool.uv.sources]`), so `uv.lock` makes the harness version
reproducible. It is used as a dependency only (samplers, parsers, model
interface, retry decorator, prompt-generation conventions); it is not forked
and Truco is not registered as an OpenSpiel game. To read or step through the
harness source, `git clone https://github.com/google-deepmind/game_arena
vendor/game_arena` (the directory is git-ignored); to use that clone instead of
the pinned commit, change the source to
`game_arena = { path = "vendor/game_arena", editable = true }` and re-run
`uv sync`.

Run the tests (about 5 seconds, including the 10,000-match property test):

```bash
uv run pytest
```

Without uv: create a 3.11+ virtualenv, `pip install "game_arena @ git+https://github.com/google-deepmind/game_arena" absl-py requests tenacity pytest`, and use `python` in place of `uv run python` below.

## Running a demo match

Two OpenRouter model slugs, one per team (even seats vs odd seats), one seed,
plus its duplicate (same deals, teams swapped across seats):

```bash
uv run python -m runner.cli run \
  --team-a openai/gpt-5-mini \
  --team-b deepseek/deepseek-chat-v3.1 \
  --seeds 1 --duplicate --out runs/demo
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--team-a-options '{"temperature":0.7,"max_tokens":8000,"reasoning":{"effort":"low"}}'` | Forwarded to the request body (`temperature`, `top_p`, `top_k`, `max_tokens`, `reasoning`, ...). Default `max_tokens` is 16384 so reasoning models have room. |
| `--team-a-provider '{"order":["OpenAI"],"allow_fallbacks":false}'` | OpenRouter provider pinning, sent verbatim as the `provider` object. The provider that actually served each request is logged. |
| `--team-a-label mini` | Display label in the stats table. |
| `--team-a random` | A uniformly random legal-action bot instead of a model (deterministic, useful as a baseline and for reproducibility checks). |
| `--team-a heuristic` | A deterministic rule-based baseline (fixed card-strength thresholds for playing, raising, answering calls and mão de onze) instead of a model. See below. |
| `--seeds N --start-seed S` | Seeds `S .. S+N-1`. |
| `--duplicate / --no-duplicate` | Also play each seed with the two configurations swapped across seats. Default on. |
| `--num-players {2,4,6}` | Default 4. |
| `--parallel K` | Play K matches concurrently. |
| `--max-reprompts` | Re-prompts after an illegal reply before the random fallback. Default 1 (the benchmark rule). |
| `--base-url URL` | Point `OpenRouterModel` at a self-hosted OpenAI-compatible endpoint (vLLM, Ollama). |

Both teams may use the same slug; the labels get `#A` / `#B` suffixes.

A JSON argument may also be a path to a JSON file.

### The heuristic baseline

`heuristic` is a deterministic rule-based player that sees only what a model
would see. It scores its hand (manilha 3, then 3 → 2, 2 → 1.5, A → 1, K → 0.5,
plus or minus 2 per trick won or lost), calls and answers raises by comparing
that score to fixed thresholds that rise with the stake, never folds
voluntarily, plays the lowest card that wins the trick (or the lowest card when
its partner is already winning), leads the middle card in trick 1 and its
strongest later, and decides mão de onze from the average card value of its
team's hands. The full rule list is the docstring of `runner/heuristic.py`. The
thresholds are versioned: changing any of them changes the bot, and therefore
the rating anchor of every result that includes it.

## Round-robin tournaments

`tournament` plays every pairing of a list of models and is resumable: matches
that already have a `summary.json` are skipped, so a run can be stopped and
relaunched, extended with more seeds, or have failed matches replayed after a
fix.

```bash
uv run python -m runner.cli tournament --models-file models.json --seeds 100 --duplicate \
    --parallel 20 --out runs/rr [--exclude labelA/labelB] [--no-preflight]
```

`models.json` is a list of `{"slug", "label", "provider", "model_options"}`
objects (`"slug": "random"` or `"slug": "heuristic"` for the bots). A
one-request preflight per model (skipped for the bots) runs first. The report (`tournament.json` and the printed table) has standings
over all matches with Bradley–Terry Elo ratings and bootstrap confidence intervals
(see [Ratings](#ratings)), and a head-to-head win-rate matrix. `tournament.json`
also records the code version that produced it.

With duplicates enabled, matches are ordered by seed and pairing, and each
original/duplicate pair runs as one unit in a single worker (the duplicate
starts after its original finishes). The Kaggle `Budget` admits a pair as a
unit: it starts an original only if the expected cost of both halves fits,
always starts the duplicate of an admitted original, refuses the duplicate when
its original was refused or failed, and charges failed matches at least the
expected match cost.

## Interpreting the outputs

`runs/demo/` contains one directory per match (`seed0_orig`, `seed0_dup`, ...)
and `aggregate.json`.

Per match:

| File | Contents |
|---|---|
| `transcript.jsonl` | Every event in order. Engine events (`source: engine`): deal (`hand_start`, all hands and the vira), `mao_de_onze`, `mao_de_ferro`, every `action` with its talk, `card_played`, `trick_result`, `raise_called` / `raise_accepted` / `raise_declined`, `hand_result` (winner, points, reason, stake history), `match_end`. Runner events (`source: runner`): `match_config` (seed, seat → exact slug, pinned provider, model options, and the full system instruction), `prompt` (the exact prompt text a seat received, decision type, legal list), `response` (main response, reasoning where the API exposes it, served provider, token usage, OpenRouter cost, latency), `illegal_action` (seat, hand, trick, attempt, raw response, legal list), `fallback_action` (the random replacement), `decision` (action, talk, source), and `match_error` (the error and the partial known cost, a lower bound, when a match fails). `match_config.code_version` records the source hash, Git state, package versions, and harness commit. |
| `engine_events.jsonl` | Engine events only, no timestamps. Byte-identical for the same seed and action sequence. `uv run python -m runner.cli replay <file>` rebuilds the match from the seed plus logged actions and verifies this. |
| `replay.txt` | Human-readable replay: deals, actions, talk, trick and hand results, illegal replies, fallbacks. |
| `summary.json` | Winner, scores, hand-by-hand history, per-team and per-seat counters (decisions, illegal responses, fallbacks, folds, raise calls / opportunities, accept / decline / raise-back, mão de onze choices, tokens, cost, providers seen), and `code_version` (same shape as `match_config.code_version`). |

`aggregate.json` and the table printed at the end give, per model configuration:
match win rate, net points per hand (final score margin divided by hands
played), hands per match, illegal-action rate (illegal
replies per model decision), fallback rate, fold rate (folds per turn where a
fold was legal), raise call rate (per turn where a raise was legal), accept /
decline / raise-back rates (per raise response), mão de onze forfeit rate, talk
rate, tokens and requests per match, priced-response coverage, and cost per match
extrapolated from OpenRouter's reported `usage.cost`. Known cost is the sum of
priced responses (a lower bound); values are n/a when no response carried a cost.
The JSON keeps the gross `points_per_hand` field for compatibility; displayed
tables use the net value.

### Ratings

Tournament standings are sorted by a Bradley–Terry Elo fitted on match wins,
which corrects for unequal schedules (not every model plays every other).
The anchor is the heuristic bot at 0 when it is in the tournament, otherwise the
mean of the rated models; each played pairing adds +0.5 pseudo-wins per side so
sweeps stay finite. Margin strength is a least-squares fit of points per
duplicate pair (the orig and dup score margins summed). 95% CIs come from a
1000-resample bootstrap that resamples (pairing, seed) units within each pairing
and completeness stratum (orig and dup share deals, so they move together;
complete/incomplete counts stay fixed so both comparison graphs are preserved),
with a fixed RNG seed so reports are deterministic. A disconnected graph makes
BT ratings unavailable, but does not suppress raw win-rate CIs. The
Bradley–Terry solver is a safeguarded Newton iteration and reports
`fit_status`: if it does not converge, no unfinished point estimate is
published and the report says so instead. If any bootstrap replicate fails to
fit, the BT CIs are suppressed and `ratings.bootstrap.bt_failed_resamples`
records the count; win rates and margin strength are still reported.
`tournament.json` holds these under `ratings`
(method, anchor, connectivity, bootstrap settings, per-model fields), plus
`bt_elo`, `bt_elo_ci`, `margin_strength`, `margin_strength_ci` and `win_rate_ci`
on each standings row and `pairs`, `pair_margin_mean`, `pair_record` on each
`head_to_head` cell.

### Auditing a prompt

Every `prompt` event in `transcript.jsonl` can be paired with the `match_config`
event (which gives the seat's slug, pinned provider and model options) and its
`response` event (which gives the provider that actually served it). The prompt
text contains only the rendering of `TrucoMatch.observation(seat)` plus the
legal action list; the runner re-checks every prompt against
`TrucoMatch.hidden_cards(seat)` before it is sent and raises if a hidden card
appears. `tests/test_runner_match.py::test_every_prompt_is_information_isolated`
replays a transcript alongside a fresh engine and greps every prompt the same way.

Prompts are stateless single turns (system instruction = the rules text minus
§13 plus the answer format; user message = observation + legal list). The
observation already carries the full public history of the hand, all table
talk, and the results of previous hands, so nothing is lost by not accumulating
chat history. A model that answers illegally is re-prompted once with its own
previous reply quoted and the legal list repeated; a second illegal reply is
replaced by a uniformly random legal action drawn from the engine's
seed-derived fallback stream.

## Reproducibility

* The deal of hand *k* depends only on `(seed, k)`, so duplicate matches see the
  same deals regardless of how earlier hands went, and the random fallback uses
  a separate seed-derived stream.
* `TrucoMatch` is deterministic given the action sequence: the engine event log
  is a pure function of `(seed, actions)`. With random bots on both sides, two
  runs produce byte-identical `transcript.jsonl`, `engine_events.jsonl` and
  `replay.txt`. With LLMs, sampling is not deterministic even at temperature 0,
  so reproduce a match from its log with `runner.cli replay` rather than by
  re-querying the models.

## Engine API (OpenSpiel-flavored)

```python
from truco import TrucoMatch

m = TrucoMatch(seed=0, num_players=4)
while not m.is_terminal():
    seat = m.current_actor()          # includes raise responses and mão de onze deciders
    legal = m.legal_actions(seat)     # exact strings, e.g. "PLAY Q♠", "TRUCO", "ACCEPT", "MAO_PLAY", "PLAY 2"
    obs = m.observation(seat)         # only what §12 allows this seat to see
    m.apply_action(seat, legal[0], talk="optional ≤200-char table talk")
print(m.returns(), m.scores, m.serialize_events())
```

`current_actor()` returns `-4` (`TERMINAL_PLAYER`, OpenSpiel's terminal id)
when the match is over. `random_legal_action(seat)` draws from the seeded
fallback stream. Constructor extras for tests: `initial_scores={"A": 11, "B": 6}`,
`scripted_deals=[Deal(...)]`.

## Layout

```
pyproject.toml    uv project: dependencies (harness pinned by git commit), dev group, pytest config
uv.lock           locked environment
truco/            cards.py (ranks, manilhas, strength), match.py (TrucoMatch, resolve_trick, hand_winner)
openrouter_model.py
runner/           rules_text.py, prompts.py, render.py, parsers.py, sampler.py, agents.py,
                  config.py, match_log.py, match_runner.py, stats.py, replay.py, cli.py
tests/            engine, model-layer (mock HTTP server), parser, sampler, orchestrator, CLI tests
AMBIGUITIES.md    rule edge cases and the resolutions implemented
docs/             inside-truco-eval.html, a standalone snapshot of the explainer page
```
