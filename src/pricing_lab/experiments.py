"""A/B testing utilities: significance tests, power analysis, bootstrap and CUPED."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    effect: float
    ci_low: float
    ci_high: float
    p_value: float

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


def welch_ttest(control: np.ndarray, treatment: np.ndarray, alpha: float = 0.05) -> TestResult:
    """Difference in means (treatment - control) with a Welch t-test and confidence interval."""
    c, t = np.asarray(control, float), np.asarray(treatment, float)
    diff = t.mean() - c.mean()
    vc, vt = c.var(ddof=1) / len(c), t.var(ddof=1) / len(t)
    se = np.sqrt(vc + vt)
    dof = (vc + vt) ** 2 / (vc**2 / (len(c) - 1) + vt**2 / (len(t) - 1))
    q = stats.t.ppf(1 - alpha / 2, dof)
    p = 2 * stats.t.sf(abs(diff / se), dof)
    return TestResult(diff, diff - q * se, diff + q * se, p)


def two_proportion_ztest(conv_c: int, n_c: int, conv_t: int, n_t: int, alpha: float = 0.05) -> TestResult:
    """Difference in conversion rates with a pooled z-test and an unpooled (Wald) interval."""
    pc, pt = conv_c / n_c, conv_t / n_t
    pooled = (conv_c + conv_t) / (n_c + n_t)
    se0 = np.sqrt(pooled * (1 - pooled) * (1 / n_c + 1 / n_t))
    p = 2 * stats.norm.sf(abs((pt - pc) / se0))
    se = np.sqrt(pc * (1 - pc) / n_c + pt * (1 - pt) / n_t)
    z = stats.norm.ppf(1 - alpha / 2)
    return TestResult(pt - pc, pt - pc - z * se, pt - pc + z * se, p)


def sample_size_per_arm(sd: float, mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Units per arm to detect an absolute difference `mde` in means with a two-sided test."""
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return int(np.ceil(2 * ((z_a + z_b) * sd / mde) ** 2))


def bootstrap_diff_ci(control, treatment, stat=np.mean, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI for stat(treatment) - stat(control); robust to skewed revenue data."""
    rng = np.random.default_rng(seed)
    c, t = np.asarray(control, float), np.asarray(treatment, float)
    diffs = np.array([stat(rng.choice(t, len(t))) - stat(rng.choice(c, len(c))) for _ in range(n_boot)])
    return float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))


def cuped(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    """CUPED adjustment: y - theta * (x - mean(x)), theta = cov(y, x) / var(x).

    Using a pre-experiment covariate x removes the variance in y it explains, which
    shrinks confidence intervals without biasing the treatment effect.
    """
    y, x = np.asarray(y, float), np.asarray(x, float)
    theta = np.cov(y, x, ddof=1)[0, 1] / x.var(ddof=1)
    return y - theta * (x - x.mean()), float(theta)


def simulate_power(y: np.ndarray, x: np.ndarray, lift: float, n_sims: int = 1000, use_cuped: bool = False, seed: int = 0) -> float:
    """Share of simulated randomized experiments that detect a multiplicative `lift` (lift=0 gives the false-positive rate)."""
    rng = np.random.default_rng(seed)
    y, x = np.asarray(y, float), np.asarray(x, float)
    hits = 0
    for _ in range(n_sims):
        assign = rng.random(len(y)) < 0.5
        outcome = np.where(assign, y * (1 + lift), y)
        if use_cuped:
            outcome, _ = cuped(outcome, x)
        hits += welch_ttest(outcome[~assign], outcome[assign]).significant
    return hits / n_sims
