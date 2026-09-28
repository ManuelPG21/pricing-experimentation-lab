import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pricing_lab import elasticity, experiments, pareto  # noqa: E402


def test_welch_detects_known_difference():
    rng = np.random.default_rng(0)
    c = rng.normal(10, 2, 5000)
    t = rng.normal(10.5, 2, 5000)
    res = experiments.welch_ttest(c, t)
    assert res.significant
    assert res.ci_low < 0.5 < res.ci_high


def test_two_proportion_matches_statsmodels():
    from statsmodels.stats.proportion import proportions_ztest

    res = experiments.two_proportion_ztest(120, 1000, 150, 1000)
    _, p_ref = proportions_ztest([150, 120], [1000, 1000])
    assert np.isclose(res.p_value, p_ref)


def test_sample_size_textbook_value():
    # sd = 1, mde = 0.2 -> about 393 per arm at 80% power, alpha 0.05
    assert experiments.sample_size_per_arm(1.0, 0.2) in (392, 393, 394)


def test_cuped_reduces_variance_and_keeps_mean():
    rng = np.random.default_rng(1)
    x = rng.gamma(2, 50, 4000)
    y = 0.8 * x + rng.normal(0, 20, 4000)
    y_adj, theta = experiments.cuped(y, x)
    assert 0.7 < theta < 0.9
    assert y_adj.var() < 0.3 * y.var()
    assert np.isclose(y_adj.mean(), y.mean())


def test_aa_false_positive_rate_near_alpha():
    rng = np.random.default_rng(2)
    x = rng.gamma(2, 50, 2000)
    y = x + rng.normal(0, 30, 2000)
    fpr = experiments.simulate_power(y, x, lift=0.0, n_sims=400, use_cuped=True, seed=3)
    assert 0.02 < fpr < 0.09


def test_elasticity_recovers_simulated_beta():
    rng = np.random.default_rng(3)
    rows = []
    weeks = pd.date_range("2010-01-04", periods=80, freq="W-MON")
    for i in range(30):
        beta = rng.normal(-1.5, 0.3)
        price = np.exp(rng.normal(0, 0.2, len(weeks))) * (1 + i / 10)
        units = np.exp(4 + beta * np.log(price) + rng.normal(0, 0.1, len(weeks)))
        rows.append(pd.DataFrame({"stock_code": f"P{i}", "week": weeks, "units": units, "avg_price": price}))
    panel = elasticity.prepare_panel(pd.concat(rows))
    pooled = elasticity.pooled_elasticity(panel)
    assert abs(pooled["beta"] + 1.5) < 0.15
    shrunk, eb = elasticity.empirical_bayes_shrink(elasticity.per_product_elasticities(panel))
    assert abs(eb["mu"] + 1.5) < 0.2


def test_pareto_items_for_share():
    df = pd.DataFrame({"k": list("abcde"), "v": [50, 30, 10, 5, 5]})
    t = pareto.pareto_table(df, "v", "k")
    assert pareto.items_for_share(t, 0.8) == 2
