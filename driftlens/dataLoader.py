"""Data loading, auditing and a synthetic drifting dataset for demos/tests."""
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from driftlens.config import ExperimentConfig

# Columns known to be unavailable at decision time (post-outcome information).
LEAKAGE_COLUMNS: Tuple[str, ...] = ("duration",)


def computeChecksum(path: str) -> str:
    """SHA-256 of a file on disk (recorded for reproducibility)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def computeChecksumFromBytes(content: bytes) -> str:
    """SHA-256 of an in-memory upload."""
    return hashlib.sha256(content).hexdigest()


def loadBankCsv(source: Any) -> pd.DataFrame:
    """Load a CSV path or file-like object; the separator (';' or ',') is sniffed."""
    return pd.read_csv(source, sep=None, engine="python")


def encodeTarget(series: pd.Series) -> np.ndarray:
    """Map 'yes'/'no' (or numeric) targets to 1/0."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int).to_numpy()
    return (series.astype(str).str.strip().str.lower() == "yes").astype(int).to_numpy()


def splitFeaturesTarget(frame: pd.DataFrame, config: ExperimentConfig) -> Tuple[pd.DataFrame, np.ndarray]:
    """Separate the target and drop leakage columns while preserving row order."""
    droppable = [c for c in config.dropColumns if c in frame.columns]
    features = frame.drop(columns=[config.targetColumn] + droppable).reset_index(drop=True)
    return features, encodeTarget(frame[config.targetColumn])


def inferColumnTypes(features: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return (numericColumns, categoricalColumns)."""
    numericCols = [c for c in features.columns if pd.api.types.is_numeric_dtype(features[c])]
    categoricalCols = [c for c in features.columns if c not in numericCols]
    return numericCols, categoricalCols


def auditData(frame: pd.DataFrame, config: ExperimentConfig) -> Dict[str, Any]:
    """Summarise size, class balance, missingness and feature-availability risks."""
    target = encodeTarget(frame[config.targetColumn])
    flagged = [c for c in frame.columns if c in LEAKAGE_COLUMNS]
    excluded = [c for c in flagged if c in config.dropColumns]
    columnRows = [
        {
            "column": c,
            "dtype": str(frame[c].dtype),
            "missingRate": float(frame[c].isna().mean()),
            "unknownRate": float((frame[c].astype(str) == "unknown").mean()),
            "nUnique": int(frame[c].nunique()),
            "leakageRisk": c in LEAKAGE_COLUMNS,
        }
        for c in frame.columns
    ]
    return {
        "numRows": int(len(frame)),
        "numColumns": int(frame.shape[1]),
        "positiveRate": float(target.mean()),
        "leakageColumnsFlagged": flagged,
        "leakageColumnsExcluded": excluded,
        "unexcludedLeakage": [c for c in flagged if c not in excluded],
        "columnRows": columnRows,
    }


def makeSyntheticBankData(numRows: int = 8000, seed: int = 7) -> pd.DataFrame:
    """Bank-marketing-like data with covariate AND concept drift over row order.

    Used for demos and tests when the UCI file is not available. It is NOT
    real data and must never be reported as a real-world result.
    """
    rng = np.random.default_rng(seed)
    progress = np.linspace(0.0, 1.0, numRows)                      # position in the sequence
    age = np.clip(rng.normal(40, 10, numRows), 18, 90).round()
    campaign = rng.poisson(1.5, numRows) + 1
    euribor3m = 4.5 - 3.0 * progress + rng.normal(0, 0.2, numRows)  # covariate drift
    isRetired = rng.random(numRows) < (0.03 + 0.20 * progress)      # category-mix drift
    baseJobs = rng.choice(["admin", "technician", "services", "management", "student"], numRows)
    job = np.where(isRetired, "retired", baseJobs)
    contact = np.where(rng.random(numRows) < 0.4 + 0.3 * progress, "cellular", "telephone")
    # Age effect flips sign over time -> concept drift.
    logit = (-2.3 - 0.35 * (euribor3m - 2.0) + 0.025 * (age - 40) * (1 - 2 * progress)
             + 0.9 * isRetired + 0.5 * (contact == "cellular") - 0.1 * (campaign - 2))
    subscribed = rng.random(numRows) < 1.0 / (1.0 + np.exp(-logit))
    duration = rng.exponential(180, numRows) + 250 * subscribed      # leakage demo column
    return pd.DataFrame({
        "age": age, "campaign": campaign, "euribor3m": euribor3m, "job": job,
        "contact": contact, "duration": duration, "y": np.where(subscribed, "yes", "no"),
    })
