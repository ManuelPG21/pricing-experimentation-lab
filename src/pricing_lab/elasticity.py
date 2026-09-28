"""Price-elasticity estimation on a product x week panel.

Model (per product i, week t):
    log(units_it) = a_i + beta_i * log(price_it) + seasonality_t + e_it

beta_i is the own-price elasticity. Per-product estimates are noisy, so they are
shrunk toward the pooled estimate with a normal-normal empirical-Bayes model
(DerSimonian-Laird estimate of between-product variance).

Caveat: this is observational data. Prices here move mostly with order size,
promotions and catalogue changes, so estimates are associations under the stated
controls, not causal effects. A randomized price test (see experiments.py) is the
way to validate a pricing decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def prepare_panel(panel: pd.DataFrame, min_weeks: int = 40, min_log_price_sd: float = 0.05,
                  price_col: str = "avg_price") -> pd.DataFrame:
    """Keep products with enough weeks and enough price variation to identify an elasticity."""
    df = panel[(panel["units"] > 0) & (panel[price_col] > 0)].copy()
    df["log_units"] = np.log(df["units"])
    df["log_price"] = np.log(df[price_col])
    df["month"] = pd.to_datetime(df["week"]).dt.month
    df["t"] = (pd.to_datetime(df["week"]) - pd.to_datetime(df["week"]).min()).dt.days / 365.25
    stats = df.groupby("stock_code").agg(n=("week", "size"), sd=("log_price", "std"))
    keep = stats[(stats["n"] >= min_weeks) & (stats["sd"] >= min_log_price_sd)].index
    return df[df["stock_code"].isin(keep)]


def pooled_elasticity(df: pd.DataFrame) -> dict:
    """Within-product (fixed-effects) elasticity with month seasonality, clustered by product."""
    d = df.copy()
    for col in ("log_units", "log_price", "t"):
        d[col + "_w"] = d[col] - d.groupby("stock_code")[col].transform("mean")
    model = smf.ols("log_units_w ~ log_price_w + t_w + C(month)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d["stock_code"])[0]}
    )
    lo, hi = model.conf_int().loc["log_price_w"]
    return {"beta": model.params["log_price_w"], "se": model.bse["log_price_w"], "ci_low": lo, "ci_high": hi,
            "n_obs": int(model.nobs), "n_products": d["stock_code"].nunique()}


def per_product_elasticities(df: pd.DataFrame) -> pd.DataFrame:
    """OLS elasticity for each product with heteroskedasticity-robust (HC3) standard errors."""
    rows = []
    for code, g in df.groupby("stock_code"):
        X = sm.add_constant(pd.DataFrame({"log_price": g["log_price"], "t": g["t"]}))
        fit = sm.OLS(g["log_units"], X).fit(cov_type="HC3")
        rows.append({"stock_code": code, "beta": fit.params["log_price"], "se": fit.bse["log_price"], "n_weeks": len(g)})
    return pd.DataFrame(rows)


def empirical_bayes_shrink(est: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Shrink noisy per-product betas toward the precision-weighted mean (random-effects meta-analysis)."""
    b, v = est["beta"].to_numpy(), est["se"].to_numpy() ** 2
    w = 1 / v
    mu_fe = np.sum(w * b) / np.sum(w)
    q = np.sum(w * (b - mu_fe) ** 2)
    k = len(b)
    tau2 = max(0.0, (q - (k - 1)) / (np.sum(w) - np.sum(w**2) / np.sum(w)))
    w_re = 1 / (v + tau2)
    mu = np.sum(w_re * b) / np.sum(w_re)
    shrink = tau2 / (tau2 + v)
    out = est.copy()
    out["beta_shrunk"] = mu + shrink * (b - mu)
    out["shrinkage"] = 1 - shrink
    return out, {"mu": mu, "tau": float(np.sqrt(tau2)), "k": k}


def revenue_curve(beta: float, price_changes: np.ndarray) -> np.ndarray:
    """Relative revenue change for a constant-elasticity demand curve: (1+dp)^(1+beta) - 1."""
    return (1 + price_changes) ** (1 + beta) - 1
