"""Pure helpers that turn result tables into chart-ready frames and Markdown."""
import numpy as np
import pandas as pd

from driftlens.experiment import ExperimentResult

SUMMARY_COLUMNS = ["prAuc", "brier", "logLoss", "ece", "precisionTop", "updateSeconds", "newLabelsUsed"]


def pivotMetric(windowFrame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Rows = block, columns = strategy."""
    return windowFrame.pivot(index="block", columns="strategy", values=metric)


def summarizeByStrategy(windowFrame: pd.DataFrame) -> pd.DataFrame:
    """Mean of each metric across evaluation blocks."""
    return windowFrame.groupby("strategy")[SUMMARY_COLUMNS].mean().reset_index()


def averageRiskCoverage(riskFrame: pd.DataFrame) -> pd.DataFrame:
    """Rows = coverage, columns = strategy, values = mean error over blocks."""
    return riskFrame.groupby(["coverage", "strategy"])["error"].mean().unstack()


def formatCell(value) -> str:
    return f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)


def frameToMarkdown(frame: pd.DataFrame) -> str:
    """Small Markdown table writer (avoids the optional `tabulate` dependency)."""
    header = "| " + " | ".join([str(c) for c in frame.columns]) + " |"
    divider = "| " + " | ".join(["---" for _ in frame.columns]) + " |"
    rows = ["| " + " | ".join([formatCell(v) for v in row]) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, divider] + rows)


def bootstrapVerdict(row) -> str:
    """Label a paired interval: does it exclude zero?"""
    if not (np.isfinite(row["ciLow"]) and np.isfinite(row["ciHigh"])):
        return "undefined"
    if row["ciLow"] > 0:
        return "above frozen"
    if row["ciHigh"] < 0:
        return "below frozen"
    return "no clear difference"


def bootstrapSummary(bootstrapFrame: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap table with an explicit verdict column."""
    return bootstrapFrame.assign(verdict=[bootstrapVerdict(r) for _, r in bootstrapFrame.iterrows()])


def stressSummary(stressFrame: pd.DataFrame) -> pd.DataFrame:
    return stressFrame[["perturbation", "severity", "prAuc", "brier", "ece", "precisionTop"]]


def buildMarkdownReport(result: ExperimentResult) -> str:
    """Self-contained summary of settings, safeguards and every result table."""
    meta, tables = result.metadata, result.tables
    corr = meta.get("corrMeanKsVsFrozenPrAuc")
    windowColumns = ["strategy", "block", "start", "end", "positiveRate", "prAuc", "brier",
                     "logLoss", "ece", "precisionTop", "updateSeconds", "newLabelsUsed"]
    leakWarning = ([
        "> **LEAKAGE DEMONSTRATION.** This run keeps post-outcome columns "
        f"({', '.join(meta['unexcludedLeakage'])}). Its numbers are NOT valid primary results.", ""]
        if meta.get("unexcludedLeakage") else [])
    lines = [
        "# DriftLens experiment report", "", *leakWarning,
        "## Run settings", "",
        f"- Rows: {meta['numRows']} (positive rate {meta['positiveRate']:.3f})",
        f"- Model: {meta['config']['modelName']}, seed {meta['config']['seed']}, "
        f"bootstrap resamples {meta['config']['bootstrapSamples']}",
        f"- Excluded leakage columns: {', '.join(meta['leakageColumnsExcluded']) or 'none'}",
        f"- All windows leakage-free (temporal audit): {meta['allLeakageFree']}",
        f"- Source checksum (SHA-256): {meta.get('sourceChecksum')}",
        f"- Versions: {meta['versions']}",
        f"- Runtime: {meta.get('runSeconds', float('nan')):.1f} s", "",
        "## Split manifest", "", frameToMarkdown(tables["splitManifest"]), "",
        "## RQ1/RQ2: mean metrics across evaluation blocks", "",
        frameToMarkdown(summarizeByStrategy(tables["windowMetrics"])), "",
        "## RQ1/RQ2: per-window metrics", "",
        frameToMarkdown(tables["windowMetrics"][windowColumns]), "",
        "## RQ2: paired bootstrap 95% intervals (strategy minus frozen)", "",
        "For PR-AUC higher is better; for Brier lower is better.", "",
        frameToMarkdown(bootstrapSummary(tables["bootstrap"])), "",
        "## RQ3: risk-coverage (mean error among accepted cases over blocks)", "",
        frameToMarkdown(averageRiskCoverage(tables["riskCoverage"]).reset_index()), "",
        "## Calibration (reliability, pooled over evaluation blocks)", "",
        frameToMarkdown(tables["calibration"]), "",
        "## RQ4: drift vs. frozen-model quality", "", frameToMarkdown(tables["driftPerformance"]), "",
        f"Correlation(meanKS, frozen PR-AUC) = {corr if corr is None else round(corr, 3)} "
        "(only a few blocks: interpret with caution).", "",
        "## Synthetic stress tests (frozen model, all evaluation rows)", "",
        frameToMarkdown(stressSummary(tables["stressTests"])), "",
        "## Limitations", "",
        "- Row order is a proxy for time, not a record-level timestamp.",
        "- Stress tests are synthetic perturbations, not observed drift.",
        "- Few evaluation blocks: correlations and trends are indicative only.",
        "- Educational prototype; not a validated banking decision system.",
    ]
    return "\n".join(lines)
