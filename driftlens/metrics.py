"""Predictive, calibration and selective-prediction metrics."""
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from sklearn.metrics import average_precision_score

from driftlens.config import ExperimentConfig


def prAuc(probs: np.ndarray, labels: np.ndarray) -> float:
    """Average precision (PR-AUC); NaN when the window has no positives."""
    return float(average_precision_score(labels, probs)) if labels.sum() > 0 else float("nan")


def brierScore(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def logLoss(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    clipped = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))


def reliabilityTable(probs: np.ndarray, labels: np.ndarray, numBins: int) -> List[Dict[str, Any]]:
    """Equal-width binning: mean predicted vs observed positive rate per bin."""
    edges = np.linspace(0.0, 1.0, numBins + 1)
    binIds = np.clip(np.digitize(probs, edges[1:-1]), 0, numBins - 1)
    return [
        {
            "bin": b,
            "binLow": float(edges[b]),
            "binHigh": float(edges[b + 1]),
            "count": int(mask.sum()),
            "meanPredicted": float(probs[mask].mean()),
            "observedRate": float(labels[mask].mean()),
        }
        for b in range(numBins)
        for mask in [binIds == b]
        if mask.any()
    ]


def expectedCalibrationError(probs: np.ndarray, labels: np.ndarray, numBins: int) -> float:
    """Count-weighted mean |predicted - observed| over equal-width bins."""
    table = reliabilityTable(probs, labels, numBins)
    return float(sum([r["count"] / len(probs) * abs(r["meanPredicted"] - r["observedRate"]) for r in table]))


def precisionAtTop(probs: np.ndarray, labels: np.ndarray, fraction: float) -> float:
    """Precision among the highest-scored `fraction` of cases."""
    k = max(1, int(round(len(probs) * fraction)))
    topIdx = np.argsort(-probs, kind="stable")[:k]
    return float(labels[topIdx].mean())


def riskCoverage(probs: np.ndarray, labels: np.ndarray, levels) -> List[Dict[str, float]]:
    """Error among accepted cases when abstaining on the least confident ones.

    Confidence = max(p, 1-p). Acceptance depends only on the confidence rank,
    never on the labels, so no threshold is tuned on evaluation labels.
    """
    confidence = np.maximum(probs, 1 - probs)
    order = np.argsort(-confidence, kind="stable")
    errors = ((probs >= 0.5).astype(int) != labels)[order]
    counts = [max(1, int(round(len(probs) * level))) for level in levels]
    return [
        {"coverage": float(level), "accepted": int(k), "error": float(errors[:k].mean())}
        for level, k in zip(levels, counts)
    ]


def evaluateProbabilities(probs: np.ndarray, labels: np.ndarray, config: ExperimentConfig) -> Dict[str, float]:
    """The standard metric bundle reported for every window."""
    return {
        "prAuc": prAuc(probs, labels),
        "brier": brierScore(probs, labels),
        "logLoss": logLoss(probs, labels),
        "ece": expectedCalibrationError(probs, labels, config.calibrationBins),
        "precisionTop": precisionAtTop(probs, labels, config.topFraction),
    }


def pairedBootstrapDiff(metricFn: Callable, probsA: np.ndarray, probsB: np.ndarray,
                        labels: np.ndarray, numSamples: int, seed: int) -> Dict[str, float]:
    """95% paired bootstrap interval of metric(A) - metric(B) on the same rows."""
    rng = np.random.default_rng(seed)
    resamples = [rng.integers(0, len(labels), len(labels)) for _ in range(numSamples)]
    diffs = np.array([metricFn(probsA[i], labels[i]) - metricFn(probsB[i], labels[i]) for i in resamples])
    finite = diffs[np.isfinite(diffs)]
    low, high = np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan, np.nan)
    return {
        "observedDiff": float(metricFn(probsA, labels) - metricFn(probsB, labels)),
        "ciLow": float(low),
        "ciHigh": float(high),
    }


def safeCorrelation(a, b) -> Optional[float]:
    """Pearson r, or None when it is undefined (too few / constant values)."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return None
    return float(np.corrcoef(x[valid], y[valid])[0, 1])
