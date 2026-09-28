"""Pareto (80/20) analysis to prioritize where analytical effort pays off."""

from __future__ import annotations

import pandas as pd


def pareto_table(df: pd.DataFrame, value_col: str, key_col: str) -> pd.DataFrame:
    """Sort by value and add cumulative share columns."""
    out = df.sort_values(value_col, ascending=False).reset_index(drop=True)
    out["share"] = out[value_col] / out[value_col].sum()
    out["cum_share"] = out["share"].cumsum()
    out["cum_items_share"] = (out.index + 1) / len(out)
    return out[[key_col, *[c for c in out.columns if c != key_col]]]


def items_for_share(table: pd.DataFrame, share: float = 0.8) -> int:
    """Number of top items needed to reach `share` of the total."""
    return int((table["cum_share"] < share).sum() + 1)
