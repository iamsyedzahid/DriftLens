"""Model factories, a Platt-scaling calibrator and small timing helpers."""
import time
from typing import Callable, Dict, List, Tuple

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Registry of CPU-friendly baselines (functional: name -> constructor).
MODEL_FACTORIES: Dict[str, Callable[[int], object]] = {
    "logistic": lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
    "histGradientBoosting": lambda seed: HistGradientBoostingClassifier(max_iter=100, random_state=seed),
}


def buildPreprocessor(numericCols: List[str], categoricalCols: List[str]) -> ColumnTransformer:
    """Impute + scale numerics, impute + one-hot categoricals (fit on training rows only)."""
    numericPipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categoricalPipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", numericPipe, numericCols), ("cat", categoricalPipe, categoricalCols)])


def buildModel(modelName: str, seed: int, numericCols: List[str], categoricalCols: List[str]) -> Pipeline:
    """Create an unfitted preprocessing + classifier pipeline."""
    if modelName not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model '{modelName}'. Options: {sorted(MODEL_FACTORIES)}")
    return Pipeline([
        ("prep", buildPreprocessor(numericCols, categoricalCols)),
        ("clf", MODEL_FACTORIES[modelName](seed)),
    ])


def fitTimed(model: Pipeline, features, target) -> Tuple[Pipeline, float]:
    """Fit a model and return (model, wall-clock seconds)."""
    startTime = time.perf_counter()
    model.fit(features, target)
    return model, time.perf_counter() - startTime


def positiveProba(model: Pipeline, features) -> np.ndarray:
    """P(y = 1), robust to a model that only saw one class."""
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(features))
    return model.predict_proba(features)[:, classes.index(1)]


class SigmoidCalibrator:
    """Platt scaling: a logistic map fitted on logit(probability).

    Fitted on a labelled window that is disjoint from the model-fit rows.
    """

    def __init__(self) -> None:
        self.model = None
        self.isIdentity = False

    @staticmethod
    def toLogit(probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
        return np.log(clipped / (1 - clipped)).reshape(-1, 1)

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> "SigmoidCalibrator":
        # With a single class there is nothing to learn: keep scores unchanged.
        self.isIdentity = len(np.unique(labels)) < 2
        self.model = None if self.isIdentity else LogisticRegression(C=1e4).fit(self.toLogit(probabilities), labels)
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        if self.isIdentity:
            return probabilities
        return self.model.predict_proba(self.toLogit(probabilities))[:, 1]
