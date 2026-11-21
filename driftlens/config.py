"""Configuration objects for DriftLens experiments."""
from dataclasses import dataclass, asdict
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable, serialisable description of one experiment run."""

    seed: int = 42
    trainFraction: float = 0.5          # first 50% of rows fit the initial model
    calibrationFraction: float = 0.1    # next 10% is the initial adaptation window
    numEvalBlocks: int = 4              # remaining rows split into sequential blocks
    modelName: str = "logistic"         # "logistic" or "histGradientBoosting"
    topFraction: float = 0.1            # precision-at-top-k fraction
    coverageLevels: Tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
    calibrationBins: int = 10
    bootstrapSamples: int = 200
    dropColumns: Tuple[str, ...] = ("duration",)   # unavailable before the call
    targetColumn: str = "y"
    stressSeverities: Tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)

    def __post_init__(self) -> None:
        """Reject configurations that would create empty or leaky windows."""
        if not (0.0 < self.trainFraction < 1.0 and 0.0 < self.calibrationFraction < 1.0):
            raise ValueError("Train and calibration fractions must lie in (0, 1).")
        if self.trainFraction + self.calibrationFraction >= 0.95:
            raise ValueError("Train + calibration must leave room for evaluation blocks.")
        if self.numEvalBlocks < 1:
            raise ValueError("At least one evaluation block is required.")
        if not 0.0 < self.topFraction <= 1.0:
            raise ValueError("topFraction must lie in (0, 1].")

    def toDict(self) -> Dict[str, Any]:
        """Return a plain dictionary so the config can be saved as JSON."""
        return asdict(self)
