"""Schedule-adjusted ratings from match summaries.

Raw win rate pooled over opponents is confounded by who each model happened to
play (models dropped for a day, budget cut-offs mid-seed). This module fits:

* Bradley–Terry on match wins (primary), reported on the Elo scale
  (``400·log10`` of the strength). Every pairing that was played gets
  ``PRIOR_PSEUDO_WINS`` virtual wins on each side, so perfect records stay
  finite. Fitted with safeguarded Newton iteration in log-strength space.
* Margin strength: least squares ``pair_margin ≈ s_x − s_y`` over complete
  duplicate pairs, where a pair's margin is X's final-score margin in the
  ``orig`` match plus X's margin in the ``dup`` match (same deals, seats
  swapped). Unit: points per duplicate pair.

An ``anchor`` candidate is used at rating 0 only when it has at least
``MIN_ANCHOR_PAIRS`` complete duplicate pairs against every other rated model;
otherwise ratings are mean-zero over the rated models. Uncertainty comes from a
cluster bootstrap that resamples whole seeds across pairings: one sampled seed
brings every pairing's unit for that seed, including incomplete units. A draw is
redone when it drops a model or disconnects a connected full win graph or a
connected full complete-pair margin graph, up to ``MAX_BOOTSTRAP_REDRAWS`` redraws
per replicate; if a replicate's BT or margin fit still fails or omits a model, that
statistic's CIs are suppressed for every model. The bootstrap RNG is seeded, so
a report is a pure function of its summaries. Tiers are boundaries whose joint
BT Elo ordering is separated with confidence ``TIER_CONFIDENCE``: for each cut,
the minimum Elo in the upper segment must exceed the maximum Elo in the lower
segment. A replicate missing a rated model is not separated, and tiers are
unavailable when the point BT fit is unavailable or any bootstrap fit fails.
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
MIN_ANCHOR_PAIRS = 20
MAX_BOOTSTRAP_REDRAWS = 100
TIER_CONFIDENCE = 0.95

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
  the comparison graph is disconnected or the solver cannot converge.
  """
  pairs = {tuple(sorted(k)) for k, v in wins.items() if v > 0}
  players = sorted({p for pair in pairs for p in pair})
  if not players or not _connected(players, pairs):
    return None
  games: dict[str, list[tuple[str, float]]] = {p: [] for p in players}  # i -> [(j, n_ij)]
  for i, j in sorted(pairs):
    n = wins.get((i, j), 0.0) + wins.get((j, i), 0.0) + 2 * prior
    games[i].append((j, n))
    games[j].append((i, n))
  idx = {p: i for i, p in enumerate(players)}
  logp = {p: (init or {}).get(p, 0.0) for p in players}
  if not all(math.isfinite(v) for v in logp.values()):
    return None
  shift = logp[players[0]]
  logp = {p: v - shift for p, v in logp.items()}

  def state(values):
    loss = 0.0
    gradient = [0.0] * len(players)
    hessian = [[0.0] * len(players) for _ in players]
    for i, player in enumerate(players):
      for opponent, n in games[player]:
        j = idx[opponent]
        d = values[player] - values[opponent]
        q = math.exp(-abs(d))
        p, complement = ((1 / (1 + q), q / (1 + q)) if d >= 0
                         else (q / (1 + q), 1 / (1 + q)))
        w_ij = wins.get((player, opponent), 0.0) + prior
        w_ji = wins.get((opponent, player), 0.0) + prior
        gradient[i] += w_ij * complement - w_ji * p
        curvature = n * q / (1 + q) ** 2
        hessian[i][i] += curvature
        hessian[i][j] -= curvature
        if i < j:
          loss += w_ij * max(-d, 0) + w_ji * max(d, 0) + n * math.log1p(q)
    return loss, gradient, hessian

  for _ in range(max_iter):
    loss, gradient, hessian = state(logp)
    try:
      step = [0.0] + _solve([row[1:] for row in hessian[1:]], gradient[1:])
    except ValueError:
      return None
    if not math.isfinite(loss) or not all(math.isfinite(v) for v in step):
      return None
    if max(abs(v) for v in step) < tol:
      mean = sum(logp.values()) / len(logp)
      return {p: v - mean for p, v in logp.items()}
    improvement = sum(g * s for g, s in zip(gradient, step))
    rate = 1.0
    for _ in range(50):
      candidate = {p: logp[p] + rate * step[i] for i, p in enumerate(players)}
      if state(candidate)[0] <= loss - 1e-4 * rate * improvement + 1e-12 * max(1.0, loss):
        logp = candidate
        break
      rate *= 0.5
    else:
      return None
  return None


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
    min_anchor_pairs: int = MIN_ANCHOR_PAIRS,
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
  connected = _connected(seen, units_by_pair)
  logp = fit_bradley_terry(wins)
  fit_status = ("converged" if logp is not None else "no_matches" if not seen
                else "disconnected" if not connected else "not_converged")
  anchor_candidate = anchor
  anchor_skipped: str | None = None
  if anchor is None:
    used_anchor = None
  elif anchor not in seen:
    used_anchor = None
    anchor_skipped = "no matches"
  else:
    pair_counts = []
    for opponent in seen:
      if opponent == anchor:
        continue
      pair = tuple(sorted((anchor, opponent)))
      count = sum(1 for unit in units_by_pair.get(pair, ()) if unit["complete"])
      pair_counts.append((count, opponent))
    if not pair_counts:
      used_anchor = None
      anchor_skipped = "no matches"
    else:
      fewest, opponent = min(pair_counts, key=lambda item: (item[0], item[1]))
      if fewest < min_anchor_pairs:
        used_anchor = None
        anchor_skipped = (f"fewer than {min_anchor_pairs} complete pairs vs "
                          f"{opponent} ({fewest})")
      else:
        used_anchor = anchor
  bt = _anchored(logp, used_anchor, ELO_PER_LOG)
  ms = _anchored(fit_margin_strength(margins), used_anchor)

  bt_s: dict[str, list[float]] = collections.defaultdict(list)
  ms_s: dict[str, list[float]] = collections.defaultdict(list)
  wr_s: dict[str, list[float]] = collections.defaultdict(list)
  bt_replicates: list[dict[str, float]] = []
  rng = random.Random(seed)
  bt_failed_resamples = 0
  ms_failed_resamples = 0
  # Margin graph over complete pairs only; incomplete units keep a pairing in the
  # win graph but not in the margin fit, so it needs its own connectivity check.
  ms_nodes = sorted({p for x, y, _ in margins for p in (x, y)})
  ms_connected = _connected(ms_nodes, [(x, y) for x, y, _ in margins])

  all_seeds = sorted({unit["seed"] for units in units_by_pair.values() for unit in units})
  units_by_seed: dict[int, list[tuple[Pair, dict[str, Any]]]] = collections.defaultdict(list)
  for pair, units in units_by_pair.items():
    for unit in units:
      units_by_seed[unit["seed"]].append((pair, unit))

  def draw_seed_sample() -> dict[Pair, list[dict[str, Any]]]:
    sample: dict[Pair, list[dict[str, Any]]] = collections.defaultdict(list)
    if not all_seeds:
      return {}
    for _ in all_seeds:
      drawn_seed = rng.choice(all_seeds)
      for pair, unit in units_by_seed[drawn_seed]:
        sample[pair].append(unit)
    return dict(sample)

  redraws = 0
  for _ in range(resamples):
    sample = draw_seed_sample()
    redraws_for_replicate = 0
    while True:
      sample_seen = {label for pair, units in sample.items() if units for label in pair}
      sample_connected = _connected(
          seen, [pair for pair, units in sample.items() if units])
      sample_ms_connected = _connected(
          ms_nodes, [pair for pair, units in sample.items() if any(u["complete"] for u in units)])
      valid = (set(seen).issubset(sample_seen) and (not connected or sample_connected)
               and (not ms_connected or sample_ms_connected))
      if valid or redraws_for_replicate >= MAX_BOOTSTRAP_REDRAWS:
        break
      sample = draw_seed_sample()
      redraws += 1
      redraws_for_replicate += 1
    w, rec, mg = _tally(sample)
    if logp is not None:
      sample_fit = fit_bradley_terry(w, init=logp, tol=1e-7)
      if sample_fit is None or not set(logp) <= set(sample_fit):
        bt_failed_resamples += 1
      sample_bt = _anchored(sample_fit, used_anchor, ELO_PER_LOG)
      if sample_bt is not None:
        bt_replicates.append(sample_bt)
      for k, v in (sample_bt or {}).items():
        bt_s[k].append(v)
    sample_ms = _anchored(fit_margin_strength(mg), used_anchor)
    if ms is not None and (sample_ms is None or not set(ms) <= set(sample_ms)):
      ms_failed_resamples += 1
    for k, v in (sample_ms or {}).items():
      ms_s[k].append(v)
    for k, (wn, n) in rec.items():
      wr_s[k].append(wn / n)

  complete_pairs = collections.Counter()
  for (x, y), units in units_by_pair.items():
    for u in units:
      if u["complete"]:
        complete_pairs[x] += 1
        complete_pairs[y] += 1

  point_bt = bt or {}
  tier_order = sorted(point_bt, key=lambda label: (-point_bt[label], label))
  tier_separation: list[dict[str, Any]] | None = None
  tiers: list[list[str]] | None = None
  if bt is not None and bt_failed_resamples == 0:
    tier_separation = []
    boundaries: set[int] = set()
    for cut in range(1, len(tier_order)):
      top = set(tier_order[:cut])
      rest = set(tier_order[cut:])
      separated = 0
      for replicate in bt_replicates:
        if not all(label in replicate for label in tier_order):
          continue
        if min(replicate[label] for label in top) > max(replicate[label] for label in rest):
          separated += 1
      separation = separated / len(bt_replicates) if bt_replicates else 0.0
      tier_separation.append({"after": tier_order[cut - 1], "separation": separation})
      if separation >= TIER_CONFIDENCE:
        boundaries.add(cut)
    tiers = []
    start = 0
    for end in sorted((*boundaries, len(tier_order))):
      tiers.append(tier_order[start:end])
      start = end

  tier_by_model: dict[str, int] = {}
  for tier, tier_labels in enumerate(tiers or (), start=1):
    tier_by_model.update({label: tier for label in tier_labels})

  models = {}
  for l in all_labels:
    models[l] = {
        "matches": record[l][1] if l in record else 0,
        "complete_pairs": complete_pairs[l],
        "win_rate": (record[l][0] / record[l][1]) if record.get(l) and record[l][1] else None,
        "win_rate_ci": _ci(wr_s.get(l, [])),
        "bt_elo": None if bt is None else bt.get(l),
        "bt_elo_ci": (_ci(bt_s.get(l, [])) if bt is not None and l in bt
                      and bt_failed_resamples == 0 else None),
        "margin_strength": None if ms is None else ms.get(l),
        "margin_strength_ci": (_ci(ms_s.get(l, [])) if ms is not None and l in ms
                               and ms_failed_resamples == 0 else None),
        "tier": tier_by_model.get(l),
    }
  return {
      "method": "bradley_terry",
      "scale": "elo",
      "anchor": used_anchor,
      "anchor_candidate": anchor_candidate,
      "anchor_min_pairs": min_anchor_pairs,
      "anchor_skipped": anchor_skipped,
      "prior_pseudo_wins": PRIOR_PSEUDO_WINS,
      "connected": connected,
      "fit_status": fit_status,
      "unpaired_matches": sum(len(u["matches"]) for us in units_by_pair.values()
                              for u in us if not u["complete"]),
      "bootstrap": {"resamples": resamples, "unit": "seed", "strata": None,
                    "seed": seed, "bt_failed_resamples": bt_failed_resamples,
                    "ms_failed_resamples": ms_failed_resamples, "redraws": redraws},
      "tier_confidence": TIER_CONFIDENCE,
      "tiers": tiers,
      "tier_separation": tier_separation,
      "models": models,
      "head_to_head": _pairing_stats(units_by_pair),
  }
