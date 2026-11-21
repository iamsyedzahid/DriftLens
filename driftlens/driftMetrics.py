"""Feature-drift distances implemented with numpy only."""
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _clean(values) -> np.ndarray:
    """Sorted finite values as floats."""
    array = np.asarray(values, dtype=float)
    return np.sort(array[np.isfinite(array)])


def ksStatistic(reference, current) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (max CDF gap)."""
    a, b = _clean(reference), _clean(current)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    grid = np.concatenate([a, b])
    cdfA = np.searchsorted(a, grid, side="right") / len(a)
    cdfB = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(cdfA - cdfB)))


def wassersteinDistance(reference, current) -> float:
    """1-D Wasserstein-1 distance: integral of |CDF_a - CDF_b|."""
    a, b = _clean(reference), _clean(current)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    grid = np.sort(np.concatenate([a, b]))
    cdfA = np.searchsorted(a, grid[:-1], side="right") / len(a)
    cdfB = np.searchsorted(b, grid[:-1], side="right") / len(b)
    return float(np.sum(np.abs(cdfA - cdfB) * np.diff(grid)))


def _klDivergence(p: np.ndarray, q: np.ndarray) -> float:
    support = p > 0
    return float(np.sum(p[support] * np.log2(p[support] / q[support])))


def jensenShannon(reference: pd.Series, current: pd.Series) -> float:
    """Jensen-Shannon divergence (base 2, range 0..1) between category frequencies."""
    refValues, curValues = reference.dropna().astype(str), current.dropna().astype(str)
    if len(refValues) == 0 or len(curValues) == 0:
        return float("nan")
    categories = sorted(set(refValues) | set(curValues))
    p = refValues.value_counts(normalize=True).reindex(categories, fill_value=0.0).to_numpy()
    q = curValues.value_counts(normalize=True).reindex(categories, fill_value=0.0).to_numpy()
    m = 0.5 * (p + q)
    return 0.5 * _klDivergence(p, m) + 0.5 * _klDivergence(q, m)


def driftReport(reference: pd.DataFrame, current: pd.DataFrame,
                numericCols: List[str], categoricalCols: List[str]) -> List[Dict[str, Any]]:
    """One row per feature. Wasserstein is expressed in reference-std units."""
    numericRows = [
        {
            "feature": c, "kind": "numeric",
            "ks": ksStatistic(reference[c], current[c]),
            "wasserstein": wassersteinDistance(reference[c], current[c]) / (reference[c].std() or 1.0),
            "jensenShannon": float("nan"),
        }
        for c in numericCols
    ]
    categoricalRows = [
        {"feature": c, "kind": "categorical", "ks": float("nan"),
         "wasserstein": float("nan"), "jensenShannon": jensenShannon(reference[c], current[c])}
        for c in categoricalCols
    ]
    return numericRows + categoricalRows
