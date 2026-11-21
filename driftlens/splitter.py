"""Order-based split manifest (pure functions + an immutable manifest)."""
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np

from driftlens.config import ExperimentConfig

Range = Tuple[int, int]  # half-open [start, end) row indices


@dataclass(frozen=True)
class SplitManifest:
    """Saved split boundaries: train | calibration | eval blocks (all disjoint)."""

    numRows: int
    trainRange: Range
    calibrationRange: Range
    evalRanges: Tuple[Range, ...]

    @property
    def numBlocks(self) -> int:
        return len(self.evalRanges)

    def toDict(self) -> Dict[str, Any]:
        return asdict(self)

    def toRows(self) -> List[Dict[str, Any]]:
        """Flat table form used for CSV export and the UI."""
        named = [("train", self.trainRange), ("calibration", self.calibrationRange)] + [
            (f"eval{i}", r) for i, r in enumerate(self.evalRanges)
        ]
        return [{"segment": n, "start": r[0], "end": r[1], "size": r[1] - r[0]} for n, r in named]


def rangesOverlap(a: Range, b: Range) -> bool:
    """True when two half-open ranges share at least one row."""
    return a[0] < b[1] and b[0] < a[1]


def computeSplit(numRows: int, config: ExperimentConfig) -> SplitManifest:
    """Build the chronological manifest (50% / 10% / 4 x 10% by default)."""
    trainEnd = int(numRows * config.trainFraction)
    calEnd = trainEnd + int(numRows * config.calibrationFraction)
    edges = np.linspace(calEnd, numRows, config.numEvalBlocks + 1).astype(int)
    evalRanges = tuple([(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])])
    if not all([b > a for a, b in evalRanges]) or trainEnd < 1 or calEnd <= trainEnd:
        raise ValueError("Dataset too small for the requested split.")
    return SplitManifest(numRows, (0, trainEnd), (trainEnd, calEnd), evalRanges)
