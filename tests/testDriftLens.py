"""Unit tests: split boundaries, leakage, metrics, drift, stress tests, persistence."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftlens.config import ExperimentConfig
from driftlens.dataLoader import makeSyntheticBankData, inferColumnTypes, splitFeaturesTarget
from driftlens.driftMetrics import jensenShannon, ksStatistic, wassersteinDistance
from driftlens.experiment import ExperimentRunner
from driftlens.metrics import brierScore, expectedCalibrationError, precisionAtTop, riskCoverage
from driftlens.splitter import computeSplit, rangesOverlap
from driftlens.reporting import buildMarkdownReport
from driftlens.store import ResultStore
from driftlens.strategies import DataBundle, buildStrategies
from driftlens.stressTest import injectMissing, shiftNumeric

CONFIG = ExperimentConfig(bootstrapSamples=20)
FRAME = makeSyntheticBankData(3000)


class SplitTests(unittest.TestCase):
    def testWindowsAreContiguousAndDisjoint(self):
        manifest = computeSplit(1000, CONFIG)
        self.assertEqual(manifest.trainRange, (0, 500))
        self.assertEqual(manifest.calibrationRange, (500, 600))
        self.assertEqual(manifest.evalRanges[0][0], 600)
        self.assertEqual(manifest.evalRanges[-1][1], 1000)
        allRanges = [manifest.trainRange, manifest.calibrationRange] + list(manifest.evalRanges)
        pairs = [(a, b) for i, a in enumerate(allRanges) for b in allRanges[i + 1:]]
        self.assertFalse(any([rangesOverlap(a, b) for a, b in pairs]))

    def testInvalidConfigRejected(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(trainFraction=0.9, calibrationFraction=0.1)


class LeakageTests(unittest.TestCase):
    def testAuditPassesForEveryStrategyAndBlock(self):
        result = ExperimentRunner(CONFIG).run(FRAME)
        self.assertTrue(result.metadata["allLeakageFree"])
        self.assertTrue(result.tables["leakageAudit"]["leakageFree"].all())

    def testDurationColumnIsExcluded(self):
        features, _ = splitFeaturesTarget(FRAME, CONFIG)
        self.assertNotIn("duration", features.columns)

    def testCurrentBlockLabelsDoNotChangePredictions(self):
        """Flipping labels inside the last block must not change its predictions."""
        manifest = computeSplit(len(FRAME), CONFIG)
        lastStart, lastEnd = manifest.evalRanges[-1]
        flipped = FRAME.copy()
        flipped.loc[lastStart:lastEnd, "y"] = np.where(flipped.loc[lastStart:lastEnd, "y"] == "yes", "no", "yes")

        def predictLast(frame):
            features, target = splitFeaturesTarget(frame, CONFIG)
            numericCols, categoricalCols = inferColumnTypes(features)
            bundle = DataBundle(features, target, manifest, numericCols, categoricalCols, CONFIG)
            return [s.predictBlock(manifest.numBlocks - 1).probabilities for s in buildStrategies(bundle)]

        originalPreds, flippedPreds = predictLast(FRAME), predictLast(flipped)
        _ = [np.testing.assert_allclose(a, b) for a, b in zip(originalPreds, flippedPreds)]


class MetricTests(unittest.TestCase):
    def testBrierPerfectIsZero(self):
        self.assertEqual(brierScore(np.array([1.0, 0.0]), np.array([1, 0])), 0.0)

    def testPrecisionAtTop(self):
        probs, labels = np.array([0.9, 0.8, 0.1, 0.2]), np.array([1, 0, 0, 1])
        self.assertEqual(precisionAtTop(probs, labels, 0.5), 0.5)

    def testEceZeroWhenCalibrated(self):
        self.assertAlmostEqual(expectedCalibrationError(np.full(4, 0.5), np.array([1, 0, 1, 0]), 10), 0.0)

    def testRiskCoverage(self):
        probs, labels = np.array([0.99, 0.9, 0.6, 0.4]), np.array([1, 1, 0, 1])
        rows = riskCoverage(probs, labels, [0.5, 1.0])
        self.assertEqual(rows[0]["error"], 0.0)
        self.assertEqual(rows[1]["error"], 0.5)


class DriftTests(unittest.TestCase):
    def testKs(self):
        a = np.arange(100.0)
        self.assertEqual(ksStatistic(a, a), 0.0)
        self.assertEqual(ksStatistic(a, a + 1000), 1.0)

    def testWassersteinEqualsShift(self):
        a = np.random.default_rng(0).normal(size=500)
        self.assertAlmostEqual(wassersteinDistance(a, a + 2.0), 2.0, places=6)

    def testJensenShannon(self):
        same = pd.Series(["a", "b", "a", "b"])
        self.assertAlmostEqual(jensenShannon(same, same), 0.0)
        self.assertAlmostEqual(jensenShannon(pd.Series(["a"] * 5), pd.Series(["b"] * 5)), 1.0)


class StressTests(unittest.TestCase):
    def testPerturbationsDoNotMutateInput(self):
        features, _ = splitFeaturesTarget(FRAME, CONFIG)
        numericCols, categoricalCols = inferColumnTypes(features)
        before = features.copy()
        shiftNumeric(features, numericCols, categoricalCols, 1.0, 1)
        injectMissing(features, numericCols, categoricalCols, 1.0, 1)
        pd.testing.assert_frame_equal(before, features)

    def testMissingRateApproximate(self):
        features, _ = splitFeaturesTarget(FRAME, CONFIG)
        numericCols, categoricalCols = inferColumnTypes(features)
        perturbed = injectMissing(features, numericCols, categoricalCols, 1.0, 1)
        self.assertAlmostEqual(float(perturbed["age"].isna().mean()), 0.3, delta=0.05)

    def testZeroSeverityIsIdentity(self):
        features, _ = splitFeaturesTarget(FRAME, CONFIG)
        numericCols, categoricalCols = inferColumnTypes(features)
        pd.testing.assert_frame_equal(shiftNumeric(features, numericCols, categoricalCols, 0.0, 1), features, check_dtype=False)


class PersistenceTests(unittest.TestCase):
    def testSaveLoadRoundTrip(self):
        result = ExperimentRunner(CONFIG).run(FRAME, "synthetic")
        with tempfile.TemporaryDirectory() as tempDir:
            store = ResultStore(tempDir)
            store.save(result)
            loaded = store.load()
        self.assertEqual(set(loaded.tables), set(result.tables))
        pd.testing.assert_frame_equal(loaded.tables["windowMetrics"], result.tables["windowMetrics"],
                                      check_dtype=False, check_exact=False)
        self.assertEqual(loaded.metadata["sourceChecksum"], "synthetic")

    def testEndToEndCoversAllStrategies(self):
        result = ExperimentRunner(CONFIG).run(FRAME)
        self.assertEqual(set(result.tables["windowMetrics"]["strategy"]), {"frozen", "recalibrated", "rolling"})
        self.assertEqual(len(result.tables["windowMetrics"]), 3 * CONFIG.numEvalBlocks)


class ReportingTests(unittest.TestCase):
    def testReportContainsEverySection(self):
        report = buildMarkdownReport(ExperimentRunner(CONFIG).run(FRAME, "synthetic"))
        headings = ["Split manifest", "per-window metrics", "paired bootstrap", "risk-coverage",
                    "Calibration", "drift vs. frozen", "stress tests", "Limitations"]
        self.assertTrue(all([h in report for h in headings]))
        self.assertNotIn("LEAKAGE DEMONSTRATION", report)

    def testLeakageDemoIsFlagged(self):
        demoConfig = ExperimentConfig(bootstrapSamples=20, dropColumns=())
        result = ExperimentRunner(demoConfig).run(FRAME, "synthetic")
        self.assertEqual(result.metadata["unexcludedLeakage"], ["duration"])
        self.assertIn("LEAKAGE DEMONSTRATION", buildMarkdownReport(result))


if __name__ == "__main__":
    unittest.main()
