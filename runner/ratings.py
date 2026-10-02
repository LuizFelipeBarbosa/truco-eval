"""Schedule-adjusted ratings from match summaries.

Raw win rate pooled over opponents is confounded by who each model happened to
play (models dropped for a day, budget cut-offs mid-seed). This module fits:

* Bradley–Terry on match wins (primary), reported on the Elo scale
  (``400·log10`` of the strength). Every pairing that was played gets
  ``PRIOR_PSEUDO_WINS`` virtual wins on each side, so perfect records stay
  finite. Fitted with Hunter's (2004) MM algorithm.
* Margin strength: least squares ``pair_margin ≈ s_x − s_y`` over complete
  duplicate pairs, where a pair's margin is X's final-score margin in the
  ``orig`` match plus X's margin in the ``dup`` match (same deals, seats
  swapped). Unit: points per duplicate pair.

Ratings are anchored at ``anchor`` (rating 0) when given and rated, otherwise
mean-zero over the rated models.

Uncertainty comes from a cluster bootstrap. The unit is (pairing, seed): the
``orig`` and ``dup`` matches of a seed share their deals, so they are resampled
together. Resampling is stratified by pairing and completeness (each stratum
keeps its number of units), preserving both the win and margin comparison graphs.
The bootstrap
RNG is seeded, so a report is a pure function of its summaries.
"""

from __future__ import annotations

import collections
import math
import random
from typing import Any, Iterable, Sequence

PRIOR_PSEUDO_WINS = 0.5
ELO_PER_LOG = 400.0 / math.log(10.0)
DEFAULT_RESAMPLES = 1000
DEFAULT_BOOTSTRAP_SEED = 0

Pair = tuple[str, str]


# ------------------------------------------------------------------- units

def _team_of(summary: dict[str, Any], label: str) -> str:
  teams = [t for t in ("A", "B") if summary["team_config"][t]["display"] == label]
  if len(teams) != 1:
    raise ValueError(f"label {label!r} must fill exactly one team in match "
                     f"{summary.get('match_id')!r}")
  return teams[0]


def build_units(summaries: Iterable[dict[str, Any]]) -> dict[Pair, list[dict[str, Any]]]:
  """Groups matches into (pairing, seed) units, keyed by the sorted label pair.

  Each match is recorded from the perspective of ``x`` (the first label of the
  sorted pair): ``x_won`` and ``margin_x`` (x's final score minus y's).
  """
  grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = collections.defaultdict(list)
  for s in summaries:
    x, y = sorted((s["team_config"]["A"]["display"], s["team_config"]["B"]["display"]))
    if x == y:
      raise ValueError(f"both teams have label {x!r} in match {s.get('match_id')!r}")
    tx, ty = _team_of(s, x), _team_of(s, y)
    grouped[(x, y, int(s["seed"]))].append({
        "swap": bool(s["swap"]),
        "x_won": s["winner_config"] == x,
        "margin_x": int(s["scores"][tx]) - int(s["scores"][ty]),
    })
  units: dict[Pair, list[dict[str, Any]]] = collections.defaultdict(list)
  for (x, y, seed) in sorted(grouped):
    matches = sorted(grouped[(x, y, seed)], key=lambda m: m["swap"])
    complete = [m["swap"] for m in matches] == [False, True]
    units[(x, y)].append({
        "seed": seed,
        "matches": matches,
        "complete": complete,
        "pair_margin_x": sum(m["margin_x"] for m in matches) if complete else None,
    })
  return dict(units)


# ---------------------------------------------------------------- tallies

def _tally(units_by_pair: dict[Pair, list[dict[str, Any]]]):
  wins: dict[Pair, float] = collections.defaultdict(float)  # (i, j) -> matches i beat j
  record: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])  # label -> [wins, matches]
  margins: list[tuple[str, str, float]] = []
  for (x, y), units in units_by_pair.items():
    for u in units:
      for m in u["matches"]:
        winner, loser = (x, y) if m["x_won"] else (y, x)
        wins[(winner, loser)] += 1
        record[winner][0] += 1
        record[x][1] += 1
        record[y][1] += 1
      if u["complete"]:
        margins.append((x, y, float(u["pair_margin_x"])))
  return wins, record, margins


def _connected(nodes: Sequence[str], edges: Iterable[Pair]) -> bool:
  if not nodes:
    return False
  adj: dict[str, set[str]] = {n: set() for n in nodes}
  for a, b in edges:
    adj[a].add(b)
    adj[b].add(a)
  seen, stack = {nodes[0]}, [nodes[0]]
  while stack:
    for nb in adj[stack.pop()]:
      if nb not in seen:
        seen.add(nb)
        stack.append(nb)
  return len(seen) == len(nodes)


# ----------------------------------------------------------- Bradley–Terry

def fit_bradley_terry(
    wins: dict[Pair, float],
    *,
    prior: float = PRIOR_PSEUDO_WINS,
    init: dict[str, float] | None = None,
    tol: float = 1e-9,
    max_iter: int = 5000,
) -> dict[str, float] | None:
  """Bradley–Terry log-strengths (natural log, mean zero) from pairwise win counts.

  ``wins[(i, j)]`` is the number of matches i beat j. Every pairing with at
  least one match gets ``prior`` virtual wins on each side. Returns None when
  the comparison graph is disconnected (strengths are then not identifiable).
  """
  pairs = {tuple(sorted(k)) for k, v in wins.items() if v > 0}
  players = sorted({p for pair in pairs for p in pair})
  if not players or not _connected(players, pairs):
    return None
  games: dict[str, list[tuple[str, float]]] = {p: [] for p in players}  # i -> [(j, n_ij)]
  won = {p: 0.0 for p in players}
  for i, j in sorted(pairs):
    w_ij, w_ji = wins.get((i, j), 0.0) + prior, wins.get((j, i), 0.0) + prior
    games[i].append((j, w_ij + w_ji))
    games[j].append((i, w_ij + w_ji))
    won[i] += w_ij
    won[j] += w_ji
  logp = {p: (init or {}).get(p, 0.0) for p in players}
  for _ in range(max_iter):
    p = {k: math.exp(v) for k, v in logp.items()}
    new = {i: math.log(won[i] / sum(n / (p[i] + p[j]) for j, n in games[i])) for i in players}
    mean = sum(new.values()) / len(new)
    new = {k: v - mean for k, v in new.items()}
    delta = max(abs(new[k] - logp[k]) for k in players)
    logp = new
    if delta < tol:
      break
  return logp


# -------------------------------------------------------- margin strength

def _solve(a: list[list[float]], b: list[float]) -> list[float]:
  """Gaussian elimination with partial pivoting (small dense systems)."""
  n = len(b)
  m = [row[:] + [b[i]] for i, row in enumerate(a)]
  for c in range(n):
    piv = max(range(c, n), key=lambda r: abs(m[r][c]))
    if abs(m[piv][c]) < 1e-12:
      raise ValueError("singular system")
    m[c], m[piv] = m[piv], m[c]
    for r in range(c + 1, n):
      f = m[r][c] / m[c][c]
      for k in range(c, n + 1):
        m[r][k] -= f * m[c][k]
  x = [0.0] * n
  for r in range(n - 1, -1, -1):
    x[r] = (m[r][n] - sum(m[r][k] * x[k] for k in range(r + 1, n))) / m[r][r]
  return x


def fit_margin_strength(margins: Sequence[tuple[str, str, float]]) -> dict[str, float] | None:
  """Least-squares strengths (mean zero) with ``margin ≈ s_x − s_y``; None if disconnected."""
  players = sorted({p for x, y, _ in margins for p in (x, y)})
  if not players or not _connected(players, [(x, y) for x, y, _ in margins]):
    return None
  idx = {p: k for k, p in enumerate(players)}
  n = len(players)
  lap = [[0.0] * n for _ in range(n)]
  rhs = [0.0] * n
  for x, y, m in margins:
    i, j = idx[x], idx[y]
    lap[i][i] += 1
    lap[j][j] += 1
    lap[i][j] -= 1
    lap[j][i] -= 1
    rhs[i] += m
    rhs[j] -= m
  # Pin players[0] at 0 (drop its row and column), then centre.
  sol = [0.0] + (_solve([row[1:] for row in lap[1:]], rhs[1:]) if n > 1 else [])
  mean = sum(sol) / n
  return {p: sol[idx[p]] - mean for p in players}


# -------------------------------------------------------------- reporting

def _anchored(values: dict[str, float] | None, anchor: str | None,
              scale: float = 1.0) -> dict[str, float] | None:
  if values is None:
    return None
  shift = values[anchor] if anchor in values else 0.0  # values are mean-zero already
  return {k: (v - shift) * scale for k, v in values.items()}


def _percentile(sorted_vals: list[float], q: float) -> float:
  pos = q * (len(sorted_vals) - 1)
  lo = math.floor(pos)
  hi = min(lo + 1, len(sorted_vals) - 1)
  return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def _ci(samples: list[float]) -> list[float] | None:
  if not samples:
    return None
  s = sorted(samples)
  return [_percentile(s, 0.025), _percentile(s, 0.975)]


def _pairing_stats(units_by_pair: dict[Pair, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
  """Duplicate-pair stats for both directions: out[x][y] is from x's perspective."""
  out: dict[str, dict[str, Any]] = collections.defaultdict(dict)
  for (x, y), units in units_by_pair.items():
    complete = [u for u in units if u["complete"]]
    won = sum(1 for u in complete if all(m["x_won"] for m in u["matches"]))
    lost = sum(1 for u in complete if not any(m["x_won"] for m in u["matches"]))
    mean = (sum(u["pair_margin_x"] for u in complete) / len(complete)) if complete else None
    out[x][y] = {"pairs": len(complete), "pair_margin_mean": mean,
                 "pair_record": {"won": won, "split": len(complete) - won - lost, "lost": lost}}
    out[y][x] = {"pairs": len(complete), "pair_margin_mean": None if mean is None else -mean,
                 "pair_record": {"won": lost, "split": len(complete) - won - lost, "lost": won}}
  return dict(out)


def compute_ratings(
    summaries: Iterable[dict[str, Any]],
    labels: Sequence[str] = (),
    *,
    anchor: str | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
  """BT Elo, margin strength and win rate, each with a 95% cluster-bootstrap CI.

  ``labels`` fixes the reported models (and their order); labels seen only in
  the summaries are appended. Models without matches get None everywhere.
  """
  units_by_pair = build_units(summaries)
  seen = sorted({p for pair in units_by_pair for p in pair})
  all_labels = list(dict.fromkeys([*labels, *seen]))
  wins, record, margins = _tally(units_by_pair)
  logp = fit_bradley_terry(wins)
  if anchor is not None and (logp is None or anchor not in logp):
    anchor = None
  bt = _anchored(logp, anchor, ELO_PER_LOG)
  ms = _anchored(fit_margin_strength(margins), anchor)

  bt_s: dict[str, list[float]] = collections.defaultdict(list)
  ms_s: dict[str, list[float]] = collections.defaultdict(list)
  wr_s: dict[str, list[float]] = collections.defaultdict(list)
  rng = random.Random(seed)
  strata = {k: [[u for u in units_by_pair[k] if u["complete"] == complete]
                for complete in (True, False)] for k in sorted(units_by_pair)}
  for _ in range(resamples):
    sample = {k: [rng.choice(group) for group in groups for _ in group]
              for k, groups in strata.items()}
    w, rec, mg = _tally(sample)
    for k, v in (_anchored(fit_bradley_terry(w, init=logp, tol=1e-7), anchor, ELO_PER_LOG) or {}).items():
      bt_s[k].append(v)
    for k, v in (_anchored(fit_margin_strength(mg), anchor) or {}).items():
      ms_s[k].append(v)
    for k, (wn, n) in rec.items():
      wr_s[k].append(wn / n)

  complete_pairs = collections.Counter()
  matches = collections.Counter()
  for (x, y), units in units_by_pair.items():
    for u in units:
      matches[x] += len(u["matches"])
      matches[y] += len(u["matches"])
      if u["complete"]:
        complete_pairs[x] += 1
        complete_pairs[y] += 1

  models = {}
  for l in all_labels:
    models[l] = {
        "matches": matches[l],
        "complete_pairs": complete_pairs[l],
        "win_rate": (record[l][0] / record[l][1]) if record.get(l) and record[l][1] else None,
        "win_rate_ci": _ci(wr_s.get(l, [])),
        "bt_elo": None if bt is None else bt.get(l),
        "bt_elo_ci": _ci(bt_s.get(l, [])) if bt is not None and l in bt else None,
        "margin_strength": None if ms is None else ms.get(l),
        "margin_strength_ci": _ci(ms_s.get(l, [])) if ms is not None and l in ms else None,
    }
  return {
      "method": "bradley_terry",
      "scale": "elo",
      "anchor": anchor,
      "prior_pseudo_wins": PRIOR_PSEUDO_WINS,
      "connected": logp is not None,
      "unpaired_matches": sum(len(u["matches"]) for us in units_by_pair.values()
                              for u in us if not u["complete"]),
      "bootstrap": {"resamples": resamples, "unit": "pairing x seed",
                    "strata": "pairing x completeness", "seed": seed},
      "models": models,
      "head_to_head": _pairing_stats(units_by_pair),
  }
