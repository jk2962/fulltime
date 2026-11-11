"""Round-10 checkpoint evaluation (MODEL_NOTES.md, decision log 2026-11-10).

    python -m pipeline.checkpoint dc    # candidate A, judged under point 2
    python -m pipeline.checkpoint sim   # candidate S, judged under point 7

Prints evidence only; it changes no setting and writes no file. The 2025-26
holdout is scored for candidate A only if it passes the first two conditions.
"""

import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

from pipeline import ingest, model, scoring, simulate
from pipeline.run import walk_forward_blocks

COMPARISON_SEASONS = ["2023-24", "2024-25"]
HOLDOUT_SEASON = "2025-26"
SIM_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]
SIM_ROUNDS = [6, 15, 25]
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 2026


# ---- Candidate A: Dixon-Coles ------------------------------------------------


def tau(home_goals, away_goals, lam, mu, rho):
    """Dixon-Coles factor for each scoreline; 1 outside 0-0, 1-0, 0-1, 1-1."""
    out = np.ones(np.broadcast(home_goals, away_goals, lam).shape)
    out = np.where((home_goals == 0) & (away_goals == 0), 1 - lam * mu * rho, out)
    out = np.where((home_goals == 0) & (away_goals == 1), 1 + lam * rho, out)
    out = np.where((home_goals == 1) & (away_goals == 0), 1 + mu * rho, out)
    return np.where((home_goals == 1) & (away_goals == 1), 1 - rho, out)


def fit_rho(fitted, train):
    """Maximum-likelihood rho on the training window, expected goals held fixed."""
    latest = train["kickoff_utc"].max()
    window = train[train["kickoff_utc"] > latest - pd.Timedelta(days=model.TRAINING_WINDOW_DAYS)]
    preds = [model.predict(fitted, h, a) for h, a in zip(window["home"], window["away"])]
    lam = np.array([p["exp_home_goals"] for p in preds])
    mu = np.array([p["exp_away_goals"] for p in preds])
    x = window["home_goals"].to_numpy()
    y = window["away_goals"].to_numpy()
    # Every tau must stay positive: 1 - lam*mu*rho > 0, 1 + lam*rho > 0, 1 + mu*rho > 0, 1 - rho > 0.
    low = max(-1 / lam.max(), -1 / mu.max()) + 1e-6
    high = min(1 / (lam * mu).max(), 1.0) - 1e-6
    result = minimize_scalar(
        lambda rho: -np.log(tau(x, y, lam, mu, rho)).sum(), bounds=(low, high), method="bounded"
    )
    return float(result.x)


def predict_dc(fitted, rho, home, away):
    """(p_home, p_draw, p_away) with the Dixon-Coles correction on the 0-10 grid."""
    p = model.predict(fitted, home, away)
    goals = np.arange(model.MAX_GOALS + 1)
    grid = np.outer(poisson.pmf(goals, p["exp_home_goals"]), poisson.pmf(goals, p["exp_away_goals"]))
    grid = grid * tau(goals[:, None], goals[None, :], p["exp_home_goals"], p["exp_away_goals"], rho)
    grid = grid / grid.sum()
    return [np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()]


def walk_forward(history, season):
    """Current and candidate-A forecasts for one season, refitted before each week.

    Returns (current probs, candidate probs, outcomes, rhos), one fit per week
    serving both, so the only difference is the correction."""
    current, candidate, outcomes, rhos = [], [], [], []
    for block in walk_forward_blocks(history[history["season"] == season]):
        train = ingest.get_training_data(history, block["kickoff_utc"].min())
        fitted = model.fit(train)
        rho = fit_rho(fitted, train)
        rhos.append(rho)
        for h, a in zip(block["home"], block["away"]):
            p = model.predict(fitted, h, a)
            current.append([p["p_home"], p["p_draw"], p["p_away"]])
            candidate.append(predict_dc(fitted, rho, h, a))
        outcomes += list(scoring.match_outcomes(block))
    return np.array(current), np.array(candidate), np.array(outcomes), rhos


def bootstrap_se(current, candidate, outcomes, statistic):
    """Bootstrap SE of statistic(candidate) - statistic(current), matches resampled."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(outcomes)
    diffs = []
    for _ in range(N_BOOTSTRAP):
        i = rng.integers(0, n, n)
        diffs.append(statistic(candidate[i], outcomes[i]) - statistic(current[i], outcomes[i]))
    return float(np.std(diffs, ddof=1))


def evaluate_dc(history):
    runs = {s: walk_forward(history, s) for s in COMPARISON_SEASONS}
    print("Candidate A: Dixon-Coles. Walk-forward on " + " + ".join(COMPARISON_SEASONS))
    for season, (cur, cand, out, rhos) in runs.items():
        print(
            f"  {season}: RPS current {scoring.rps_many(cur, out).mean():.4f}, "
            f"candidate {scoring.rps_many(cand, out).mean():.4f}; "
            f"rho median {np.median(rhos):+.3f} (range {min(rhos):+.3f} to {max(rhos):+.3f})"
        )
    cur = np.concatenate([r[0] for r in runs.values()])
    cand = np.concatenate([r[1] for r in runs.values()])
    out = np.concatenate([r[2] for r in runs.values()])
    diff = scoring.rps_many(cand, out) - scoring.rps_many(cur, out)
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    by_season = {s: (scoring.rps_many(r[1], r[2]) - scoring.rps_many(r[0], r[2])).mean() for s, r in runs.items()}

    calib = {k: scoring.calibration_error(p, out) for k, p in [("current", cur), ("candidate", cand)]}
    draw = {k: scoring.draw_rate_gap(p, out) for k, p in [("current", cur), ("candidate", cand)]}
    calib_se = bootstrap_se(cur, cand, out, scoring.calibration_error)
    draw_se = bootstrap_se(cur, cand, out, scoring.draw_rate_gap)

    print(f"\n  {len(out)} matches. RPS difference (candidate - current) {diff.mean():+.5f}, SE {se:.5f}, "
          f"{diff.mean() / se:+.2f} SE")
    print(f"  calibration error: current {calib['current']:.4f}, candidate {calib['candidate']:.4f}, "
          f"bootstrap SE of difference {calib_se:.4f}")
    print(f"  draw-rate gap: current {draw['current']:.4f}, candidate {draw['candidate']:.4f}, "
          f"bootstrap SE of difference {draw_se:.4f}")
    print(f"  predicted draw rate: current {cur[:, 1].mean():.4f}, candidate {cand[:, 1].mean():.4f}, "
          f"observed {(out == 'D').mean():.4f}")

    c1 = diff.mean() <= -2 * se
    c2 = all(d < 0 for d in by_season.values())
    c4 = calib["candidate"] - calib["current"] <= calib_se and draw["candidate"] - draw["current"] <= draw_se
    print(f"\n  1. lower by >= 2 SE:           {'PASS' if c1 else 'FAIL'}")
    print("  2. lower in each season:       " + ("PASS" if c2 else "FAIL")
          + "  (" + ", ".join(f"{s} {d:+.5f}" for s, d in by_season.items()) + ")")
    if c1 and c2:
        h_cur, h_cand, h_out, _ = walk_forward(history, HOLDOUT_SEASON)
        r_cur, r_cand = scoring.rps_many(h_cur, h_out).mean(), scoring.rps_many(h_cand, h_out).mean()
        c3 = r_cand <= r_cur
        print(f"  3. holdout {HOLDOUT_SEASON} not worse:  {'PASS' if c3 else 'FAIL'}  "
              f"(current {r_cur:.4f}, candidate {r_cand:.4f})  HOLDOUT USED: log it")
    else:
        c3 = False
        print("  3. holdout:                    NOT RUN (1 and 2 must pass first)")
    print(f"  4. calibration not worse:      {'PASS' if c4 else 'FAIL'}")
    print(f"\n  Outcome: {'ADOPT' if c1 and c2 and c3 and c4 else 'REJECT'}")


# ---- Candidate S: promoted-team prior variance in the simulation -------------


def promoted_precision(history, seasons):
    """(attack, defense) precision: 1 / variance of promoted teams' first-season
    gaps, pooled over `seasons`, around their pooled mean (ddof 1)."""
    gaps = model.promoted_gaps(history[history["season"] <= max(seasons)])
    attack = [g for s in seasons for g in gaps[s][0]]
    defense = [g for s in seasons for g in gaps[s][1]]
    return 1 / np.var(attack, ddof=1), 1 / np.var(defense, ddof=1)


def final_positions(season_df):
    """Actual finishing position per team: points, goal difference, goals scored."""
    table = simulate.current_table(season_df)
    order = table.sort_values(["points", "gd", "gf"], ascending=False).index
    return pd.Series(range(1, len(order) + 1), index=order), table["points"]


def sim_snapshot(history, season, round_, precision):
    """(RPS over positions per team, inside 80% points interval per team)."""
    season_df = history[history["season"] == season].sort_values("kickoff_utc", kind="stable")
    cutoff = season_df["kickoff_utc"].iloc[(round_ - 1) * 10]
    train = ingest.get_training_data(history, cutoff)
    table = simulate.current_table(train[train["season"] == season])
    fixtures = season_df[season_df["kickoff_utc"] >= cutoff]

    fitted = model.fit(train)
    teams, cov = simulate.laplace_covariance(fitted, train, precision)
    rng = np.random.default_rng(simulate.SEED)
    deviations = rng.multivariate_normal(np.zeros(len(cov)), cov, size=simulate.N_RUNS)
    _, positions, points = simulate.simulate_positions(
        table, fixtures, fitted, teams, deviations, rng, return_points=True
    )

    actual_pos, actual_points = final_positions(season_df)
    names = list(table.index)
    cum_pred = positions.loc[names].cumsum(axis=1).to_numpy()[:, :-1]
    cum_actual = (np.arange(1, len(names))[None, :] >= actual_pos[names].to_numpy()[:, None]).astype(float)
    rps = ((cum_pred - cum_actual) ** 2).sum(axis=1) / (len(names) - 1)
    low, high = np.percentile(points, [10, 90], axis=0)
    inside = (actual_points[names].to_numpy() >= low) & (actual_points[names].to_numpy() <= high)
    return rps, inside


def evaluate_sim(history):
    in_use = promoted_precision(history, SIM_SEASONS)
    print("Candidate S: promoted-team prior variance in the simulation")
    print(f"  value from {SIM_SEASONS[0]} to {SIM_SEASONS[-1]}: attack precision {in_use[0]:.1f} "
          f"(SD {in_use[0] ** -0.5:.3f}), defense precision {in_use[1]:.1f} (SD {in_use[1] ** -0.5:.3f}); "
          f"current {model.PRIOR_STRENGTH} (SD {model.PRIOR_STRENGTH ** -0.5:.3f})\n")
    results = {"current": ([], []), "candidate": ([], [])}
    for season in SIM_SEASONS:
        loso = promoted_precision(history, [s for s in SIM_SEASONS if s != season])
        for round_ in SIM_ROUNDS:
            line = f"  {season} round {round_:>2}:"
            for name, precision in [("current", None), ("candidate", loso)]:
                rps, inside = sim_snapshot(history, season, round_, precision)
                results[name][0].append(rps)
                results[name][1].append(inside)
                line += f"  {name} RPS {rps.mean():.4f} coverage {inside.mean():.2f}"
            print(line)
    summary = {k: (np.concatenate(r).mean(), np.concatenate(c).mean()) for k, (r, c) in results.items()}
    print()
    for name, (rps, cov) in summary.items():
        print(f"  {name:<9} RPS over positions {rps:.4f}, 80% interval coverage {cov:.3f}")
    ok_rps = summary["candidate"][0] <= summary["current"][0]
    ok_cov = abs(summary["candidate"][1] - 0.8) < abs(summary["current"][1] - 0.8)
    print(f"\n  RPS not higher:          {'PASS' if ok_rps else 'FAIL'}")
    print(f"  coverage closer to 80%:  {'PASS' if ok_cov else 'FAIL'}")
    print(f"\n  Outcome: {'ADOPT' if ok_rps and ok_cov else 'REJECT'}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which not in ("dc", "sim"):
        raise SystemExit("usage: python -m pipeline.checkpoint dc|sim")
    history = ingest.load_history()
    evaluate_dc(history) if which == "dc" else evaluate_sim(history)
