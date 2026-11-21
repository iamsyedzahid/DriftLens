"""Maintenance strategies (OOP): frozen, recalibrated and rolling retraining."""
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from driftlens.config import ExperimentConfig
from driftlens.models import SigmoidCalibrator, buildModel, fitTimed, positiveProba
from driftlens.splitter import Range, SplitManifest, rangesOverlap


@dataclass(eq=False)
class DataBundle:
    """Everything a strategy may read. Slicing is by row range only."""

    features: pd.DataFrame
    target: np.ndarray
    manifest: SplitManifest
    numericCols: List[str]
    categoricalCols: List[str]
    config: ExperimentConfig

    def featureSlice(self, rowRange: Range) -> pd.DataFrame:
        return self.features.iloc[rowRange[0]:rowRange[1]]

    def targetSlice(self, rowRange: Range) -> np.ndarray:
        return self.target[rowRange[0]:rowRange[1]]

    def fitModel(self, fitRange: Range):
        """Fit a fresh pipeline on a row range; returns (model, seconds)."""
        model = buildModel(self.config.modelName, self.config.seed, self.numericCols, self.categoricalCols)
        return fitTimed(model, self.featureSlice(fitRange), self.targetSlice(fitRange))


@dataclass
class StrategyOutput:
    """Predictions for one block plus an audit trail of the rows that were used."""

    strategyName: str
    blockIndex: int
    probabilities: np.ndarray
    updateSeconds: float            # maintenance cost only (initial fit excluded)
    newLabelsUsed: int              # labels newly consumed for this update
    modelFitRange: Range
    calibrationRange: Optional[Range]


class MaintenanceStrategy(ABC):
    """Template: a strategy turns a block index into leakage-audited predictions."""

    name = "base"

    def __init__(self, bundle: DataBundle) -> None:
        self.bundle = bundle

    def blockFeatures(self, blockIndex: int) -> pd.DataFrame:
        return self.bundle.featureSlice(self.bundle.manifest.evalRanges[blockIndex])

    @abstractmethod
    def predictBlock(self, blockIndex: int) -> StrategyOutput:
        """Predict a block using only labels from strictly earlier rows."""


class FrozenStrategy(MaintenanceStrategy):
    """Fit once on the initial block and never update."""

    name = "frozen"

    def __init__(self, bundle: DataBundle) -> None:
        super().__init__(bundle)
        self.baseModel, self.baseFitSeconds = bundle.fitModel(bundle.manifest.trainRange)

    def predictBlock(self, blockIndex: int) -> StrategyOutput:
        probs = positiveProba(self.baseModel, self.blockFeatures(blockIndex))
        return StrategyOutput(self.name, blockIndex, probs, 0.0, 0, self.bundle.manifest.trainRange, None)


class RecalibratedStrategy(MaintenanceStrategy):
    """Keep the base model; refit a sigmoid calibrator on the latest observed window."""

    name = "recalibrated"

    def __init__(self, bundle: DataBundle) -> None:
        super().__init__(bundle)
        self.baseModel, self.baseFitSeconds = bundle.fitModel(bundle.manifest.trainRange)

    def calibrationRangeFor(self, blockIndex: int) -> Range:
        """Initial calibration window first, then the previous (now observed) block."""
        manifest = self.bundle.manifest
        return manifest.calibrationRange if blockIndex == 0 else manifest.evalRanges[blockIndex - 1]

    def predictBlock(self, blockIndex: int) -> StrategyOutput:
        calRange = self.calibrationRangeFor(blockIndex)
        startTime = time.perf_counter()
        calProbs = positiveProba(self.baseModel, self.bundle.featureSlice(calRange))
        calibrator = SigmoidCalibrator().fit(calProbs, self.bundle.targetSlice(calRange))
        updateSeconds = time.perf_counter() - startTime
        probs = calibrator.transform(positiveProba(self.baseModel, self.blockFeatures(blockIndex)))
        return StrategyOutput(self.name, blockIndex, probs, updateSeconds, calRange[1] - calRange[0],
                              self.bundle.manifest.trainRange, calRange)


class RollingRetrainStrategy(MaintenanceStrategy):
    """Retrain from scratch on every row that precedes the evaluation block."""

    name = "rolling"

    def predictBlock(self, blockIndex: int) -> StrategyOutput:
        manifest = self.bundle.manifest
        blockStart = manifest.evalRanges[blockIndex][0]
        fitRange = (0, blockStart)
        model, seconds = self.bundle.fitModel(fitRange)
        previousStart = manifest.trainRange[1] if blockIndex == 0 else manifest.evalRanges[blockIndex - 1][0]
        probs = positiveProba(model, self.blockFeatures(blockIndex))
        return StrategyOutput(self.name, blockIndex, probs, seconds, blockStart - previousStart, fitRange, None)


# Registry (functional style): add a class here to add a strategy.
STRATEGY_CLASSES = [FrozenStrategy, RecalibratedStrategy, RollingRetrainStrategy]


def buildStrategies(bundle: DataBundle) -> List[MaintenanceStrategy]:
    return [strategyClass(bundle) for strategyClass in STRATEGY_CLASSES]


def isLeakageFree(output: StrategyOutput, manifest: SplitManifest) -> bool:
    """Audit: fit/calibration rows end before the block, and never overlap each other."""
    blockStart = manifest.evalRanges[output.blockIndex][0]
    fitOk = output.modelFitRange[1] <= blockStart
    calOk = output.calibrationRange is None or (
        output.calibrationRange[1] <= blockStart
        and not rangesOverlap(output.modelFitRange, output.calibrationRange)
    )
    return bool(fitOk and calOk)
