"""Bradley–Terry, margin strength and the cluster bootstrap on synthetic summaries."""

import math
import time

import pytest

from runner import ratings
from runner import tournament
from runner.config import ModelConfig


def match(x, y, seed, swap, x_score, y_score):
  """A minimal summary: x sits on Team A unless swap (duplicate half)."""
  tx, ty = ("B", "A") if swap else ("A", "B")
  return {
      "match_id": f"seed{seed}_{'dup' if swap else 'orig'}", "seed": seed, "swap": swap,
      "team_config": {tx: {"display": x}, ty: {"display": y}},
      "scores": {tx: x_score, ty: y_score},
      "winner_config": x if x_score > y_score else y,
      "winner_team": tx if x_score > y_score else ty,
      "hands_played": 1, "team_stats": {"A": {}, "B": {}},
  }


def wins_only(x, y, x_wins, y_wins, start_seed=0):
  """x_wins + y_wins single matches (all orig, so no complete pairs)."""
  out, seed = [], start_seed
  for _ in range(x_wins):
    out.append(match(x, y, seed, False, 12, 5)); seed += 1
  for _ in range(y_wins):
    out.append(match(x, y, seed, False, 5, 12)); seed += 1
  return out


def pair(x, y, seed, orig, dup):
  """A complete duplicate pair; orig/dup are (x_score, y_score)."""
  return [match(x, y, seed, False, *orig), match(x, y, seed, True, *dup)]


def test_two_model_closed_form():
  r = ratings.compute_ratings(wins_only("a", "b", 7, 3), resamples=50)
  gap = r["models"]["a"]["bt_elo"] - r["models"]["b"]["bt_elo"]
  assert gap == pytest.approx(400 * math.log10(7.5 / 3.5), abs=1e-6)
  assert r["models"]["a"]["bt_elo"] == pytest.approx(-r["models"]["b"]["bt_elo"])  # mean zero
  assert r["connected"] and r["anchor"] is None


def test_three_model_matches_likelihood_equations():
  ss = wins_only("a", "b", 6, 4) + wins_only("b", "c", 7, 3, 100) + wins_only("a", "c", 2, 1, 200)
  wins = {("a", "b"): 6, ("b", "a"): 4, ("b", "c"): 7, ("c", "b"): 3, ("a", "c"): 2, ("c", "a"): 1}
  logp = ratings.fit_bradley_terry(wins)
  p = {k: math.exp(v) for k, v in logp.items()}
  # Score equations with the 0.5 prior: W_i = sum_j n_ij p_i / (p_i + p_j).
  for i in p:
    w_i = sum(v + 0.5 for (a, _), v in wins.items() if a == i)
    n = {j: wins[(i, j)] + wins[(j, i)] + 1.0 for j in p if j != i}
    assert w_i == pytest.approx(sum(n[j] * p[i] / (p[i] + p[j]) for j in n), rel=1e-8)
  r = ratings.compute_ratings(ss, resamples=20)
  assert r["models"]["a"]["bt_elo"] == pytest.approx(ratings.ELO_PER_LOG * logp["a"])


@pytest.mark.parametrize("count", [10_000, 1_000_000])
def test_large_transitive_sweep_converges_to_symmetric_score_equations(count):
  wins = {("a", "b"): count, ("b", "c"): count, ("a", "c"): count}
  logp = ratings.fit_bradley_terry(wins)
  assert logp is not None
  assert logp["b"] * ratings.ELO_PER_LOG == pytest.approx(0, abs=1e-6)
  assert logp["a"] == pytest.approx(-logp["c"], abs=1e-9)
  p = {k: math.exp(v) for k, v in logp.items()}
  for i in p:
    observed = sum(wins.get((i, j), 0) + 0.5 for j in p if j != i)
    expected = sum((wins.get((i, j), 0) + wins.get((j, i), 0) + 1)
                   * p[i] / (p[i] + p[j]) for j in p if j != i)
    assert observed == pytest.approx(expected, abs=1e-8, rel=0)


def test_iteration_exhaustion_does_not_return_an_unfinished_fit():
  wins = {("a", "b"): 10_000, ("b", "c"): 10_000, ("a", "c"): 10_000}
  assert ratings.fit_bradley_terry(wins, max_iter=1) is None
  assert ratings.fit_bradley_terry(wins, max_iter=0) is None


def test_point_fit_failure_preserves_other_statistics(monkeypatch):
  ss = pair("a", "b", 0, (12, 6), (6, 12))
  monkeypatch.setattr(ratings, "fit_bradley_terry", lambda *args, **kwargs: None)
  r = ratings.compute_ratings(ss, resamples=10)
  assert r["connected"] is True
  assert r["fit_status"] == "not_converged"
  for m in r["models"].values():
    assert m["bt_elo"] is None and m["bt_elo_ci"] is None
    assert m["win_rate_ci"] == [0.5, 0.5]
    assert m["margin_strength"] == 0
    assert m["margin_strength_ci"] == [0.0, 0.0]


def test_failed_bootstrap_fit_suppresses_bt_intervals(monkeypatch):
  ss = pair("a", "b", 0, (12, 6), (6, 12)) + pair("a", "b", 1, (12, 6), (12, 6))
  real_fit = ratings.fit_bradley_terry
  calls = 0

  def fail_one_sample(*args, **kwargs):
    nonlocal calls
    calls += 1
    return None if calls == 3 else real_fit(*args, **kwargs)

  monkeypatch.setattr(ratings, "fit_bradley_terry", fail_one_sample)
  r = ratings.compute_ratings(ss, resamples=10)
  assert r["fit_status"] == "converged"
  assert r["bootstrap"]["bt_failed_resamples"] == 1
  for m in r["models"].values():
    assert m["bt_elo"] is not None and m["bt_elo_ci"] is None
    assert m["win_rate_ci"] is not None and m["margin_strength_ci"] is not None


def test_perfect_record_is_finite():
  r = ratings.compute_ratings(wins_only("a", "b", 10, 0), resamples=20)
  gap = r["models"]["a"]["bt_elo"] - r["models"]["b"]["bt_elo"]
  assert gap == pytest.approx(400 * math.log10(10.5 / 0.5), abs=1e-6)


def test_bt_corrects_for_schedule_where_raw_win_rate_misleads():
  # strong beats mid 7-3 and weak 9-1 (16/20 = .80). farmer mostly played weak
  # (19-1) and lost both to mid (19/22 = .86): higher raw win rate, lower BT.
  ss = (wins_only("strong", "mid", 7, 3) + wins_only("strong", "weak", 9, 1, 100)
        + wins_only("farmer", "weak", 19, 1, 200) + wins_only("farmer", "mid", 0, 2, 300)
        + wins_only("mid", "weak", 7, 3, 400))
  r = ratings.compute_ratings(ss, resamples=50)["models"]
  assert r["farmer"]["win_rate"] > r["strong"]["win_rate"]
  assert r["strong"]["bt_elo"] > r["farmer"]["bt_elo"]


def test_margin_strength_closed_form_two_models():
  ss = pair("a", "b", 0, (12, 5), (12, 9)) + pair("a", "b", 1, (12, 8), (10, 12))
  # pair margins from a's side: (7 + 3) = 10 and (4 - 2) = 2 -> mean 6 -> s_a - s_b = 6
  r = ratings.compute_ratings(ss, resamples=20)
  assert r["models"]["a"]["margin_strength"] == pytest.approx(3.0)
  assert r["models"]["b"]["margin_strength"] == pytest.approx(-3.0)
  h = r["head_to_head"]
  assert h["a"]["b"]["pair_margin_mean"] == pytest.approx(6.0)
  assert h["b"]["a"]["pair_margin_mean"] == pytest.approx(-6.0)
  assert h["a"]["b"]["pair_record"] == {"won": 1, "split": 1, "lost": 0}
  assert h["b"]["a"]["pair_record"] == {"won": 0, "split": 1, "lost": 1}


def test_margin_strength_recovers_consistent_three_model_system():
  true = {"a": 5.0, "b": 1.0, "c": -6.0}
  ss, seed = [], 0
  for x, y in (("a", "b"), ("a", "c"), ("b", "c")):
    m = true[x] - true[y]  # split the pair margin over the two halves
    for _ in range(3):
      ss += pair(x, y, seed, (12, 12 - int(m)), (12, 12)) if m >= 0 else pair(x, y, seed, (12 + int(m), 12), (12, 12))
      seed += 1
  r = ratings.compute_ratings(ss, resamples=20)["models"]
  for k, v in true.items():
    assert r[k]["margin_strength"] == pytest.approx(v, abs=1e-9)


def test_units_pair_orig_and_dup_and_count_unpaired():
  ss = pair("a", "b", 0, (12, 5), (12, 9)) + [match("a", "b", 1, False, 12, 3)]
  units = ratings.build_units(ss)
  assert [u["complete"] for u in units[("a", "b")]] == [True, False]
  r = ratings.compute_ratings(ss, resamples=20)
  assert r["unpaired_matches"] == 1
  assert r["models"]["a"]["matches"] == 3 and r["models"]["a"]["complete_pairs"] == 1
  assert r["models"]["a"]["margin_strength"] == pytest.approx(5.0)  # complete pair only: 7 + 3, centred
  assert r["head_to_head"]["a"]["b"]["pairs"] == 1
  # BT uses all three matches (a won all three).
  assert r["models"]["a"]["bt_elo"] - r["models"]["b"]["bt_elo"] == pytest.approx(
      400 * math.log10(3.5 / 0.5), abs=1e-6)


def test_seat_swap_is_read_from_team_config():
  # In the dup half a sits on Team B; its margin must still be a's score minus b's.
  units = ratings.build_units(pair("a", "b", 0, (12, 5), (3, 12)))
  assert [m["margin_x"] for m in units[("a", "b")][0]["matches"]] == [7, -9]
  assert units[("a", "b")][0]["pair_margin_x"] == -2


def test_anchor_is_zero():
  ss = wins_only("a", "b", 7, 3) + wins_only("b", "bot", 6, 4, 100)
  r = ratings.compute_ratings(ss, anchor="bot", min_anchor_pairs=0, resamples=50)
  assert r["anchor"] == "bot"
  assert r["models"]["bot"]["bt_elo"] == 0.0 and r["models"]["bot"]["bt_elo_ci"] == [0.0, 0.0]
  assert r["models"]["a"]["bt_elo"] > r["models"]["b"]["bt_elo"] > 0
  # An anchor with no matches is ignored (mean-zero instead).
  assert ratings.compute_ratings(wins_only("a", "b", 7, 3), labels=["a", "b", "bot"],
                                 anchor="bot", resamples=10)["anchor"] is None


def test_disconnected_graph_gives_no_ratings_but_keeps_win_rate_uncertainty():
  ss = wins_only("a", "b", 3, 2) + wins_only("c", "d", 4, 1, 100)
  r = ratings.compute_ratings(ss, resamples=100)
  assert r["connected"] is False
  for m in r["models"].values():
    assert m["bt_elo"] is None and m["bt_elo_ci"] is None
    assert m["win_rate_ci"] is not None
    lo, hi = m["win_rate_ci"]
    assert lo < m["win_rate"] < hi


def test_bootstrap_incomplete_pairs_preserve_margin_reference():
  ss = (pair("a", "b", 0, (12, 11), (12, 11))
        + pair("b", "c", 0, (12, 10), (12, 10))
        + [match("b", "c", 1, False, 12, 10)])
  for anchor, expected in (("c", {"a": 6.0, "b": 4.0, "c": 0.0}),
                           (None, {"a": 8 / 3, "b": 2 / 3, "c": -10 / 3})):
    r = ratings.compute_ratings(ss, anchor=anchor, min_anchor_pairs=0, resamples=100)
    for label, value in expected.items():
      assert r["models"][label]["margin_strength"] == pytest.approx(value)
      lo, hi = r["models"][label]["margin_strength_ci"]
      assert lo - 1e-9 <= value <= hi + 1e-9


def _margin_bridge_only_on_seed_zero():
  # a-b and c-d have a complete pair on every seed; the b-c bridge is complete
  # only on seed 0 and orig-only elsewhere. A draw without seed 0 keeps b-c in
  # the win graph but splits the complete-pair margin graph in two.
  ss = []
  for s in range(5):
    ss += pair("a", "b", s, (12, 6 + s), (12, 8))
    ss += pair("c", "d", s, (12, 9), (10 - s, 12))
  ss += pair("b", "c", 0, (12, 9), (12, 10))
  for s in range(1, 5):
    ss.append(match("b", "c", s, False, 12, 7))
  return ss


def test_bootstrap_redraws_draws_that_disconnect_the_margin_graph():
  ss = _margin_bridge_only_on_seed_zero()
  r = ratings.compute_ratings(ss, resamples=200)
  assert r["bootstrap"]["ms_failed_resamples"] == 0
  assert r["bootstrap"]["bt_failed_resamples"] == 0
  assert r["bootstrap"]["redraws"] > 0  # only the margin guard can trigger a redraw here
  for label in ("a", "b", "c", "d"):
    assert r["models"][label]["margin_strength"] is not None
    assert r["models"][label]["margin_strength_ci"] is not None
  assert ratings.compute_ratings(ss, resamples=200) == r


def test_failed_margin_bootstrap_fit_suppresses_margin_intervals(monkeypatch):
  monkeypatch.setattr(ratings, "MAX_BOOTSTRAP_REDRAWS", 0)
  r = ratings.compute_ratings(_margin_bridge_only_on_seed_zero(), resamples=200)
  # About a third of 5-seed draws miss seed 0, so their margin fits are disconnected.
  assert r["bootstrap"]["ms_failed_resamples"] > 0
  assert r["bootstrap"]["redraws"] == 0
  assert r["bootstrap"]["bt_failed_resamples"] == 0
  for label in ("a", "b", "c", "d"):
    m = r["models"][label]
    assert m["margin_strength"] is not None and m["margin_strength_ci"] is None
    assert m["bt_elo_ci"] is not None and m["win_rate_ci"] is not None
  models = [ModelConfig(kind="random", label=l) for l in ("a", "b", "c", "d")]
  rep = tournament.build_report(models, _margin_bridge_only_on_seed_zero())
  failed = rep["ratings"]["bootstrap"]["ms_failed_resamples"]
  assert failed > 0
  assert (f"Margin confidence intervals unavailable: {failed} bootstrap fit(s) did not "
          "converge or were disconnected.") in tournament.format_report(rep)


def test_bootstrap_replicate_missing_a_model_suppresses_bt_intervals(monkeypatch):
  # c plays only on seed 0; a draw without seed 0 drops c while a-b stays connected,
  # so its BT fit succeeds on a subset of the rated models.
  ss = []
  for s in range(5):
    ss += pair("a", "b", s, (12, 6 + s), (8, 12))
  ss += pair("b", "c", 0, (12, 9), (12, 10))
  monkeypatch.setattr(ratings, "MAX_BOOTSTRAP_REDRAWS", 0)
  r = ratings.compute_ratings(ss, resamples=200)
  assert r["fit_status"] == "converged"
  assert r["bootstrap"]["redraws"] == 0
  assert r["bootstrap"]["bt_failed_resamples"] > 0
  assert r["bootstrap"]["ms_failed_resamples"] == r["bootstrap"]["bt_failed_resamples"]
  assert r["tiers"] is None and r["tier_separation"] is None
  for label in ("a", "b", "c"):
    m = r["models"][label]
    assert m["bt_elo"] is not None and m["bt_elo_ci"] is None
    assert m["margin_strength"] is not None and m["margin_strength_ci"] is None


def test_labels_without_matches_are_reported_as_none():
  r = ratings.compute_ratings(wins_only("a", "b", 3, 2), labels=["z", "a", "b"], resamples=10)
  assert list(r["models"]) == ["z", "a", "b"]
  assert r["models"]["z"] == {"matches": 0, "complete_pairs": 0, "win_rate": None, "win_rate_ci": None,
                              "bt_elo": None, "bt_elo_ci": None, "margin_strength": None,
                              "margin_strength_ci": None, "tier": None}


def test_anchor_candidate_requires_complete_pairs_against_every_opponent():
  ss = []
  for s in range(20):
    ss += pair("anchor", "many", s, (12, 8), (12, 8))
  for s in range(3):
    ss += pair("anchor", "few", 100 + s, (12, 8), (12, 8))
  used = ratings.compute_ratings(ss, anchor="anchor", min_anchor_pairs=3, resamples=20)
  assert used["anchor"] == "anchor"
  assert used["anchor_candidate"] == "anchor"
  assert used["anchor_skipped"] is None
  skipped = ratings.compute_ratings(ss, anchor="anchor", min_anchor_pairs=4, resamples=20)
  assert skipped["anchor"] is None
  assert skipped["anchor_candidate"] == "anchor"
  assert skipped["anchor_skipped"] == "fewer than 4 complete pairs vs few (3)"


def test_bootstrap_clusters_shared_seeds_across_pairings():
  correlated = []
  for s in range(20):
    scores = (12, 6) if s % 2 == 0 else (6, 12)
    correlated += pair("a", "b", s, scores, scores)
    correlated += pair("a", "c", s, scores, scores)
  disjoint = []
  for s in range(20):
    scores = (12, 6) if s % 2 == 0 else (6, 12)
    disjoint += pair("a", "b", s, scores, scores)
    # The second pairing has separate seeds, so its outcomes are independent
    # units in the whole-seed bootstrap.
    disjoint += pair("a", "c", 1000 + s, scores, scores)
  correlated_result = ratings.compute_ratings(correlated, resamples=300)
  shared_width = (correlated_result["models"]["a"]["win_rate_ci"][1]
                  - correlated_result["models"]["a"]["win_rate_ci"][0])
  disjoint_result = ratings.compute_ratings(disjoint, resamples=300)
  disjoint_width = (disjoint_result["models"]["a"]["win_rate_ci"][1]
                    - disjoint_result["models"]["a"]["win_rate_ci"][0])
  assert shared_width > disjoint_width


def test_bootstrap_tiers_are_derived_from_joint_replicates():
  separated = []
  for s in range(30):
    separated += pair("a", "b", s, (12, 8) if s < 16 else (8, 12),
                      (12, 8) if s < 16 else (8, 12))
    separated += pair("c", "d", s, (12, 8) if s < 16 else (8, 12),
                      (12, 8) if s < 16 else (8, 12))
    for strong, weak in (("a", "c"), ("a", "d"), ("b", "c"), ("b", "d")):
      separated += pair(strong, weak, s, (12, 5), (12, 5))
  result = ratings.compute_ratings(separated, resamples=300)
  assert result["tiers"] == [["a", "b"], ["c", "d"]]
  assert [result["models"][label]["tier"] for label in ("a", "b", "c", "d")] == [1, 1, 2, 2]

  noisy = []
  for s in range(30):
    scores = (12, 8) if s % 2 == 0 else (8, 12)
    for x, y in (("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"),
                 ("b", "d"), ("c", "d")):
      noisy += pair(x, y, s, scores, scores)
  noisy_result = ratings.compute_ratings(noisy, resamples=300)
  assert noisy_result["tiers"] == [["a", "b", "c", "d"]]


def _many_pairs(n_seeds, start=0):
  ss = []
  for s in range(start, start + n_seeds):
    ss += pair("a", "b", s, (12, 6) if s % 3 else (7, 12), (12, 9) if s % 4 else (8, 12))
  return ss


def test_bootstrap_deterministic_contains_estimate_and_narrows():
  small, big = _many_pairs(10), _many_pairs(40)
  r1 = ratings.compute_ratings(small, resamples=300)
  r2 = ratings.compute_ratings(small, resamples=300)
  assert r1 == r2
  for key in ("bt_elo", "margin_strength", "win_rate"):
    lo, hi = r1["models"]["a"][f"{key}_ci"]
    assert lo <= r1["models"]["a"][key] <= hi
  width = lambda r: r["models"]["a"]["bt_elo_ci"][1] - r["models"]["a"]["bt_elo_ci"][0]
  assert width(ratings.compute_ratings(big, resamples=300)) < width(r1)


def test_bootstrap_resamples_whole_seeds():
  # Every seed is a split (a wins orig, loses dup): any resample of whole seeds
  # has a win rate of exactly 0.5, so the CI collapses. Resampling single
  # matches would not.
  ss = []
  for s in range(20):
    ss += pair("a", "b", s, (12, 6), (6, 12))
  assert ratings.compute_ratings(ss, resamples=200)["models"]["a"]["win_rate_ci"] == [0.5, 0.5]


def test_speed_six_models():
  labels = [f"m{i}" for i in range(6)]
  ss = []
  for i, x in enumerate(labels):
    for y in labels[i + 1:]:
      for s in range(10):
        ss += pair(x, y, s, (12, (s * 7 + i) % 12), ((s * 5 + i) % 12, 12))
  t = time.perf_counter()
  r = ratings.compute_ratings(ss, labels=labels)
  assert time.perf_counter() - t < 3.0
  assert r["bootstrap"]["resamples"] == ratings.DEFAULT_RESAMPLES
  assert all(r["models"][l]["bt_elo_ci"] is not None for l in labels)


def test_build_report_tolerates_label_missing_from_models():
  ss = wins_only("a", "b", 3, 2) + wins_only("a", "dropped", 2, 2, 100)
  models = [ModelConfig(kind="random", label="a"), ModelConfig(kind="random", label="b")]
  rep = tournament.build_report(models, ss)
  assert {r["model"] for r in rep["standings"]} == {"a", "b", "dropped"}


def test_build_report_shows_solver_failure_without_disconnected(monkeypatch):
  ss = pair("a", "b", 0, (12, 6), (6, 12))
  models = [ModelConfig(kind="random", label="a"), ModelConfig(kind="random", label="b")]
  monkeypatch.setattr(ratings, "fit_bradley_terry", lambda *args, **kwargs: None)
  rep = tournament.build_report(models, ss)
  assert rep["ratings"]["connected"] is True
  out = tournament.format_report(rep)
  assert "solver did not converge" in out
  assert "graph is disconnected" not in out
