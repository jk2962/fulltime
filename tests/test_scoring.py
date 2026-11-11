import numpy as np
import pytest

from pipeline.scoring import rps, rps_many


def test_perfect_forecast_scores_zero():
    assert rps([1.0, 0.0, 0.0], "H") == 0.0
    assert rps([0.0, 1.0, 0.0], "D") == 0.0
    assert rps([0.0, 0.0, 1.0], "A") == 0.0


def test_uniform_forecast_with_home_win():
    assert rps([1 / 3, 1 / 3, 1 / 3], "H") == pytest.approx(5 / 18)


def test_order_matters():
    # Certain of a home win, but it ends as a draw (near miss) or away (far miss).
    assert rps([1.0, 0.0, 0.0], "D") == pytest.approx(0.5)
    assert rps([1.0, 0.0, 0.0], "A") == pytest.approx(1.0)


def test_vectorized_matches_scalar():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet([1, 1, 1], size=200)
    outcomes = rng.choice(["H", "D", "A"], size=200)
    scalar = [rps(p, o) for p, o in zip(probs, outcomes)]
    assert rps_many(probs, outcomes) == pytest.approx(scalar)


def test_bad_input_raises():
    with pytest.raises(ValueError):
        rps([0.5, 0.2, 0.2], "H")  # does not sum to 1
    with pytest.raises(ValueError):
        rps([1 / 3, 1 / 3, 1 / 3], "X")  # unknown outcome


def test_calibration_error_by_hand():
    from pipeline.scoring import calibration_error, draw_rate_gap

    # Two matches, same forecast. Pooled: 0.5 x2 (bin 5, 1 hit), 0.3 x2 (bin 3,
    # 1 hit), 0.2 x2 (bin 2, 0 hits). Gaps 0.0, 0.2, 0.2, each bin 2 of 6.
    probs = [[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]]
    assert calibration_error(probs, ["H", "D"]) == pytest.approx((0 + 0.2 * 2 + 0.2 * 2) / 6)
    assert draw_rate_gap(probs, ["H", "D"]) == pytest.approx(0.2)
    # A certain, correct forecast lands in the top bin and is perfectly calibrated.
    assert calibration_error([[1.0, 0.0, 0.0]], ["H"]) == pytest.approx(0.0)
