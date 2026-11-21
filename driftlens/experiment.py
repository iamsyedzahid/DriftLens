"""Experiment engine: runs every strategy over every window and collects tables."""
import platform
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import sklearn

from driftlens.config import ExperimentConfig
from driftlens.dataLoader import auditData, inferColumnTypes, splitFeaturesTarget
from driftlens.driftMetrics import driftReport
from driftlens.metrics import (brierScore, evaluateProbabilities, pairedBootstrapDiff, prAuc,
                     reliabilityTable, riskCoverage, safeCorrelation)
from driftlens.splitter import computeSplit
from driftlens.strategies import DataBundle, StrategyOutput, buildStrategies, isLeakageFree
from driftlens.stressTest import StressTester

# Metrics compared against the frozen baseline with paired bootstrap intervals.
BOOTSTRAP_METRICS = [("prAuc", prAuc), ("brier", brierScore)]


@dataclass
class ExperimentResult:
    """Machine-readable output of one run (tables are DataFrames)."""

    tables: Dict[str, pd.DataFrame]
    metadata: Dict[str, Any]


def collectVersions() -> Dict[str, str]:
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikitLearn": sklearn.__version__}


class ExperimentRunner:
    """Coordinates data split, strategies, metrics, drift, stress tests and audits."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def scoreOutput(self, bundle: DataBundle, output: StrategyOutput) -> Dict[str, Any]:
        """One row of the per-window metric table."""
        blockRange = bundle.manifest.evalRanges[output.blockIndex]
        labels = bundle.targetSlice(blockRange)
        return {
            "strategy": output.strategyName, "block": output.blockIndex,
            "start": blockRange[0], "end": blockRange[1],
            "positiveRate": float(labels.mean()),
            **evaluateProbabilities(output.probabilities, labels, self.config),
            "updateSeconds": output.updateSeconds, "newLabelsUsed": output.newLabelsUsed,
        }

    def run(self, frame: pd.DataFrame, sourceChecksum: Optional[str] = None) -> ExperimentResult:
        startTime = time.perf_counter()
        config = self.config
        features, target = splitFeaturesTarget(frame, config)
        numericCols, categoricalCols = inferColumnTypes(features)
        manifest = computeSplit(len(frame), config)
        bundle = DataBundle(features, target, manifest, numericCols, categoricalCols, config)
        strategies = buildStrategies(bundle)
        blockIds = list(range(manifest.numBlocks))

        # Every strategy predicts every block in order, using only past labels.
        outputs = [s.predictBlock(b) for s in strategies for b in blockIds]
        outputIndex = {(o.strategyName, o.blockIndex): o for o in outputs}
        blockLabels = [bundle.targetSlice(r) for r in manifest.evalRanges]
        strategyNames = [s.name for s in strategies]

        windowFrame = pd.DataFrame([self.scoreOutput(bundle, o) for o in outputs])

        riskFrame = pd.DataFrame([
            {"strategy": o.strategyName, "block": o.blockIndex, **row}
            for o in outputs for row in riskCoverage(o.probabilities, blockLabels[o.blockIndex], config.coverageLevels)
        ])

        pooledLabels = np.concatenate(blockLabels)
        calibrationFrame = pd.DataFrame([
            {"strategy": name, **row}
            for name in strategyNames
            for pooled in [np.concatenate([outputIndex[(name, b)].probabilities for b in blockIds])]
            for row in reliabilityTable(pooled, pooledLabels, config.calibrationBins)
        ])

        reference = bundle.featureSlice(manifest.trainRange)
        driftFrame = pd.DataFrame([
            {"block": b, **row}
            for b in blockIds
            for row in driftReport(reference, bundle.featureSlice(manifest.evalRanges[b]), numericCols, categoricalCols)
        ])
        blockDrift = driftFrame.groupby("block").agg(meanKs=("ks", "mean"), meanJs=("jensenShannon", "mean")).reset_index()
        frozenScores = (windowFrame[windowFrame["strategy"] == "frozen"][["block", "prAuc", "brier"]]
                        .rename(columns={"prAuc": "frozenPrAuc", "brier": "frozenBrier"}))
        driftPerformance = blockDrift.merge(frozenScores, on="block")

        bootstrapFrame = pd.DataFrame([
            {"strategy": name, "block": b, "metric": metricName,
             **pairedBootstrapDiff(metricFn, outputIndex[(name, b)].probabilities,
                                   outputIndex[("frozen", b)].probabilities,
                                   blockLabels[b], config.bootstrapSamples, config.seed + b)}
            for name in strategyNames if name != "frozen"
            for b in blockIds
            for metricName, metricFn in BOOTSTRAP_METRICS
        ])

        frozenStrategy = [s for s in strategies if s.name == "frozen"][0]
        stressFrame = pd.DataFrame(StressTester(bundle, frozenStrategy.baseModel).run(config.stressSeverities))

        auditFrame = pd.DataFrame([
            {"strategy": o.strategyName, "block": o.blockIndex,
             "blockStart": manifest.evalRanges[o.blockIndex][0],
             "modelFitStart": o.modelFitRange[0], "modelFitEnd": o.modelFitRange[1],
             "calibrationStart": o.calibrationRange[0] if o.calibrationRange else None,
             "calibrationEnd": o.calibrationRange[1] if o.calibrationRange else None,
             "leakageFree": isLeakageFree(o, manifest)}
            for o in outputs
        ])

        dataAudit = auditData(frame, config)
        tables = {
            "windowMetrics": windowFrame, "riskCoverage": riskFrame, "calibration": calibrationFrame,
            "featureDrift": driftFrame, "driftPerformance": driftPerformance,
            "bootstrap": bootstrapFrame, "stressTests": stressFrame, "leakageAudit": auditFrame,
            "splitManifest": pd.DataFrame(manifest.toRows()),
            "dataAudit": pd.DataFrame(dataAudit["columnRows"]),
        }
        metadata = {
            "config": config.toDict(),
            "manifest": manifest.toDict(),
            "versions": collectVersions(),
            "sourceChecksum": sourceChecksum,
            "numRows": dataAudit["numRows"],
            "positiveRate": dataAudit["positiveRate"],
            "leakageColumnsExcluded": dataAudit["leakageColumnsExcluded"],
            "unexcludedLeakage": dataAudit["unexcludedLeakage"],
            "numericColumns": numericCols,
            "categoricalColumns": categoricalCols,
            "allLeakageFree": bool(auditFrame["leakageFree"].all()),
            "corrMeanKsVsFrozenPrAuc": safeCorrelation(driftPerformance["meanKs"], driftPerformance["frozenPrAuc"]),
            "runSeconds": time.perf_counter() - startTime,
        }
        return ExperimentResult(tables, metadata)
