"""Controlled synthetic perturbations (clearly separate from natural drift)."""
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

from driftlens.metrics import evaluateProbabilities
from driftlens.models import positiveProba
from driftlens.strategies import DataBundle

RARE_LABEL = "__newCategory__"


def shiftNumeric(frame: pd.DataFrame, numericCols: List[str], categoricalCols: List[str],
                 severity: float, seed: int) -> pd.DataFrame:
    """Shift each numeric column by `severity` standard deviations."""
    return frame.assign(**{c: frame[c] + severity * frame[c].std() for c in numericCols})


def injectMissing(frame: pd.DataFrame, numericCols: List[str], categoricalCols: List[str],
                  severity: float, seed: int) -> pd.DataFrame:
    """Blank out values completely at random; missing rate = 0.3 * severity."""
    rng = np.random.default_rng(seed)
    return frame.assign(**{c: frame[c].mask(rng.random(len(frame)) < 0.3 * severity)
                           for c in numericCols + categoricalCols})


def injectRareCategories(frame: pd.DataFrame, numericCols: List[str], categoricalCols: List[str],
                         severity: float, seed: int) -> pd.DataFrame:
    """Replace categories with an unseen label; replacement rate = 0.2 * severity."""
    rng = np.random.default_rng(seed)
    return frame.assign(**{c: frame[c].mask(rng.random(len(frame)) < 0.2 * severity, RARE_LABEL)
                           for c in categoricalCols})


# name -> perturbation function (all pure: they return new frames).
PERTURBATIONS: Dict[str, Callable[..., pd.DataFrame]] = {
    "numericShift": shiftNumeric,
    "missingValues": injectMissing,
    "unseenCategories": injectRareCategories,
}


class StressTester:
    """Evaluate a fitted model on perturbed copies of the evaluation region."""

    def __init__(self, bundle: DataBundle, model) -> None:
        self.bundle = bundle
        self.model = model
        manifest = bundle.manifest
        self.evalRange = (manifest.evalRanges[0][0], manifest.evalRanges[-1][1])

    def evaluatePerturbation(self, name: str, severity: float, offset: int) -> Dict[str, Any]:
        frame = self.bundle.featureSlice(self.evalRange)
        perturbed = PERTURBATIONS[name](frame, self.bundle.numericCols, self.bundle.categoricalCols,
                                        severity, self.bundle.config.seed + offset)
        probs = positiveProba(self.model, perturbed)
        metrics = evaluateProbabilities(probs, self.bundle.targetSlice(self.evalRange), self.bundle.config)
        return {"perturbation": name, "severity": float(severity), **metrics}

    def run(self, severities) -> List[Dict[str, Any]]:
        names = list(PERTURBATIONS)
        return [self.evaluatePerturbation(name, s, i * 100 + j)
                for i, name in enumerate(names) for j, s in enumerate(severities)]
