"""Ranked Probability Score (RPS) for home/draw/away forecasts. Lower is better.

RPS compares cumulative probabilities, so it respects the order
home < draw < away: a forecast that put its weight on "draw" is punished less
for an away win than one that put its weight on "home".
"""

import numpy as np

OUTCOMES = ["H", "D", "A"]  # the order matters


def match_outcomes(matches):
    """'H', 'D' or 'A' for each match, from home_goals and away_goals."""
    home = matches["home_goals"].to_numpy(dtype=float, na_value=np.nan)
    away = matches["away_goals"].to_numpy(dtype=float, na_value=np.nan)
    if np.isnan(home).any() or np.isnan(away).any():
        raise ValueError("Cannot compute outcomes: some matches have no score")
    return np.where(home > away, "H", np.where(home == away, "D", "A"))


def rps(probs_hda, outcome):
    """RPS of one forecast (p_home, p_draw, p_away) for outcome 'H', 'D' or 'A'."""
    return float(rps_many([probs_hda], [outcome])[0])


def rps_many(probs_hda, outcomes):
    """Vectorized RPS: probs_hda has shape (n, 3); outcomes has n of 'H'/'D'/'A'."""
    probs = np.asarray(probs_hda, dtype=float)
    outcomes = np.asarray(outcomes)
    if probs.ndim != 2 or probs.shape[1] != 3 or len(probs) != len(outcomes):
        raise ValueError("probs_hda must have shape (n, 3) with one outcome per row")
    if (probs < 0).any() or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("Each forecast must be non-negative and sum to 1")
    if not np.isin(outcomes, OUTCOMES).all():
        raise ValueError(f"Outcomes must be one of {OUTCOMES}")

    # One-hot actual result, e.g. 'D' -> (0, 1, 0).
    actual = (outcomes[:, None] == np.array(OUTCOMES)[None, :]).astype(float)
    # Cumulative difference at "home" and at "home or draw". The third
    # cumulative term is always 1 - 1 = 0, so it is left out.
    cum_diff = np.cumsum(probs - actual, axis=1)[:, :2]
    return (cum_diff**2).sum(axis=1) / 2


def _pooled(probs_hda, outcomes):
    """Every home, draw and away forecast as one (probability, happened) pair."""
    probs = np.asarray(probs_hda, dtype=float)
    actual = np.asarray(outcomes)[:, None] == np.array(OUTCOMES)[None, :]
    return probs.ravel(), actual.ravel().astype(float)


def calibration_bins(probs_hda, outcomes, n_bins=10):
    """(count, mean predicted, observed frequency) per equal-width bin, with
    home, draw and away forecasts pooled. Empty bins have count 0 and NaNs."""
    p, hit = _pooled(probs_hda, outcomes)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)  # p = 1.0 goes in the top bin
    count = np.bincount(bins, minlength=n_bins)
    with np.errstate(invalid="ignore"):
        predicted = np.bincount(bins, weights=p, minlength=n_bins) / count
        observed = np.bincount(bins, weights=hit, minlength=n_bins) / count
    return count, predicted, observed


def calibration_error(probs_hda, outcomes, n_bins=10):
    """Mean absolute gap between predicted and observed, weighted by bin count
    (MODEL_NOTES.md, change policy point 2)."""
    count, predicted, observed = calibration_bins(probs_hda, outcomes, n_bins)
    used = count > 0
    return float((count[used] * np.abs(predicted[used] - observed[used])).sum() / count.sum())


def draw_rate_gap(probs_hda, outcomes):
    """Absolute gap between mean predicted draw probability and observed draw rate."""
    probs = np.asarray(probs_hda, dtype=float)
    return float(abs(probs[:, 1].mean() - (np.asarray(outcomes) == "D").mean()))
