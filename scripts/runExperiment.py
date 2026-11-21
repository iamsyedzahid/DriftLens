"""Command-line entry point.

    python scripts/runExperiment.py --dataPath data/raw/bank-additional-full.csv --model logistic
    python scripts/runExperiment.py --synthetic
    python scripts/runExperiment.py --dataPath ... --leakageDemo   # keeps `duration` (NOT a primary result)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftlens.config import ExperimentConfig
from driftlens.dataLoader import computeChecksum, loadBankCsv, makeSyntheticBankData
from driftlens.experiment import ExperimentRunner
from driftlens.reporting import buildMarkdownReport, summarizeByStrategy
from driftlens.store import ResultStore


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DriftLens experiment.")
    parser.add_argument("--dataPath", default=None, help="Path to bank-additional-full.csv")
    parser.add_argument("--synthetic", action="store_true", help="Use the synthetic drifting dataset")
    parser.add_argument("--model", default="logistic", choices=["logistic", "histGradientBoosting"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=200, help="Paired bootstrap resamples")
    parser.add_argument("--leakageDemo", action="store_true",
                        help="Keep the post-outcome `duration` column to demonstrate leakage")
    parser.add_argument("--outDir", default="results/latest")
    return parser.parse_args()


def main() -> None:
    args = parseArgs()
    if args.dataPath:
        frame, checksum = loadBankCsv(args.dataPath), computeChecksum(args.dataPath)
    elif args.synthetic:
        frame, checksum = makeSyntheticBankData(), "synthetic"
    else:
        raise SystemExit("Provide --dataPath <csv> or --synthetic.")
    dropColumns = () if args.leakageDemo else ("duration",)
    config = ExperimentConfig(seed=args.seed, modelName=args.model,
                              bootstrapSamples=args.bootstrap, dropColumns=dropColumns)
    result = ExperimentRunner(config).run(frame, checksum)
    store = ResultStore(args.outDir)
    store.save(result)
    (store.directory / "report.md").write_text(buildMarkdownReport(result), encoding="utf-8")
    print(summarizeByStrategy(result.tables["windowMetrics"]).to_string(index=False))
    print(f"\nLeakage-free windows: {result.metadata['allLeakageFree']} | "
          f"unexcluded leakage columns: {result.metadata['unexcludedLeakage']} | saved to {args.outDir}")


if __name__ == "__main__":
    main()
