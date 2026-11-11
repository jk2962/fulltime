import numpy as np
import pytest

from pipeline import checkpoint, ingest, model, simulate
from tests.test_simulate import toy_league


def test_dixon_coles_with_rho_zero_is_the_current_model():
    fitted, _, _ = toy_league()
    p = model.predict(fitted, "A", "B")
    assert checkpoint.predict_dc(fitted, 0.0, "A", "B") == pytest.approx([p["p_home"], p["p_draw"], p["p_away"]])


def test_negative_rho_raises_low_scoring_draws():
    fitted, _, _ = toy_league()
    flat = checkpoint.predict_dc(fitted, 0.0, "B", "C")
    corrected = checkpoint.predict_dc(fitted, -0.1, "B", "C")
    assert sum(corrected) == pytest.approx(1.0)
    assert corrected[1] > flat[1]


def test_tau_only_touches_the_four_low_scorelines():
    goals = np.arange(4)
    t = checkpoint.tau(goals[:, None], goals[None, :], 1.4, 1.1, 0.1)
    assert t[0, 0] == pytest.approx(1 - 1.4 * 1.1 * 0.1)
    assert t[1, 1] == pytest.approx(0.9)
    t[:2, :2] = 1.0
    assert (t == 1.0).all()


def test_promoted_precision_narrows_only_promoted_teams():
    history = ingest.load_history()
    train = ingest.get_training_data(history, history[history["season"] == "2024-25"]["kickoff_utc"].iloc[60])
    fitted = model.fit(train)
    before = model.predict(fitted, "Arsenal", "Ipswich")
    teams, base = simulate.laplace_covariance(fitted, train)
    _, same = simulate.laplace_covariance(fitted, train, (model.PRIOR_STRENGTH, model.PRIOR_STRENGTH))
    _, tight = simulate.laplace_covariance(fitted, train, (50.0, 50.0))
    assert same == pytest.approx(base)
    sd = lambda cov, team: np.sqrt(cov[2 + teams.index(team), 2 + teams.index(team)])
    # Ipswich (promoted in 2024-25) narrows a lot; Arsenal only through the
    # level uncertainty every attack shares with the intercept.
    assert sd(tight, "Ipswich") < 0.5 * sd(base, "Ipswich")
    assert sd(tight, "Arsenal") > 0.8 * sd(base, "Arsenal")
    # Match forecasts are untouched: the simulation candidate never feeds predict.
    assert model.predict(fitted, "Arsenal", "Ipswich") == before
