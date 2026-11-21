"""Streamlit front end. Run with:  streamlit run app.py

The UI only calls tested experiment functions and plots values that were
saved to disk, so what you see always matches the exported files.
"""
import io
from pathlib import Path

import pandas as pd
import streamlit as st

from driftlens.config import ExperimentConfig
from driftlens.dataLoader import computeChecksumFromBytes, loadBankCsv, makeSyntheticBankData
from driftlens.experiment import ExperimentResult, ExperimentRunner
from driftlens.reporting import averageRiskCoverage, bootstrapSummary, buildMarkdownReport, pivotMetric, summarizeByStrategy
from driftlens.store import ResultStore

RESULT_DIR = Path("results/latest")
BUNDLED_DATA = Path("data/raw/bank-additional-full.csv")


def readSidebarConfig() -> ExperimentConfig:
    """Collect experiment settings; invalid combinations stop the app politely."""
    st.sidebar.header("Experiment settings")
    modelName = st.sidebar.selectbox("Model", ["logistic", "histGradientBoosting"])
    seed = int(st.sidebar.number_input("Random seed", value=42, step=1))
    trainFraction = st.sidebar.slider("Initial training fraction", 0.2, 0.7, 0.5, 0.05)
    calibrationFraction = st.sidebar.slider("Initial calibration fraction", 0.05, 0.2, 0.1, 0.05)
    numBlocks = st.sidebar.slider("Evaluation blocks", 2, 8, 4)
    bootstrapSamples = st.sidebar.slider("Bootstrap resamples", 20, 500, 100, 20)
    leakageDemo = st.sidebar.checkbox("Leakage demo: keep `duration`", value=False,
                                      help="Post-outcome column; results are NOT valid primary results.")
    try:
        return ExperimentConfig(seed=seed, modelName=modelName, trainFraction=trainFraction,
                                calibrationFraction=calibrationFraction, numEvalBlocks=numBlocks,
                                bootstrapSamples=bootstrapSamples,
                                dropColumns=() if leakageDemo else ("duration",))
    except ValueError as error:
        st.sidebar.error(str(error))
        st.stop()


def loadSelectedData():
    """Return (frame, checksum) for the chosen source, or (None, None)."""
    source = st.sidebar.radio("Data source", ["Bundled UCI file", "Synthetic demo", "Upload CSV", "Local path"])
    if source == "Bundled UCI file" and BUNDLED_DATA.exists():
        return loadBankCsv(BUNDLED_DATA), computeChecksumFromBytes(BUNDLED_DATA.read_bytes())
    if source == "Synthetic demo":
        return makeSyntheticBankData(), "synthetic"
    if source == "Upload CSV":
        upload = st.sidebar.file_uploader("bank-additional-full.csv", type="csv")
        if upload is None:
            return None, None
        content = upload.getvalue()
        return loadBankCsv(io.BytesIO(content)), computeChecksumFromBytes(content)
    path = st.sidebar.text_input("CSV path")
    if path and Path(path).exists():
        return loadBankCsv(path), computeChecksumFromBytes(Path(path).read_bytes())
    return None, None


def renderAudit(result: ExperimentResult) -> None:
    meta = result.metadata
    st.subheader("Data audit")
    st.write(f"Rows: **{meta['numRows']}** | positive rate: **{meta['positiveRate']:.3f}**")
    if meta["unexcludedLeakage"]:
        st.error("LEAKAGE DEMONSTRATION run - these numbers are not valid primary results.")
    st.write(f"Excluded as unavailable at decision time: `{meta['leakageColumnsExcluded']}`")
    if meta["unexcludedLeakage"]:
        st.warning(f"Leakage-risk columns still in the model: {meta['unexcludedLeakage']}")
    (st.success if meta["allLeakageFree"] else st.error)(
        "All windows pass the temporal leakage audit." if meta["allLeakageFree"]
        else "Leakage audit FAILED - results are invalid.")
    st.dataframe(result.tables["dataAudit"])
    st.caption("Row order is an order-based proxy for time, not record-level timestamps.")
    st.dataframe(result.tables["splitManifest"])
    st.dataframe(result.tables["leakageAudit"])


def renderWindows(result: ExperimentResult) -> None:
    windowFrame = result.tables["windowMetrics"]
    st.subheader("Strategy comparison across sequential windows")
    st.dataframe(summarizeByStrategy(windowFrame))
    metric = st.selectbox("Metric", ["prAuc", "brier", "logLoss", "ece", "precisionTop"])
    st.line_chart(pivotMetric(windowFrame, metric))
    st.caption("PR-AUC / precision@top: higher is better. Brier, log loss, ECE: lower is better.")
    st.subheader("Paired bootstrap: strategy minus frozen")
    st.dataframe(bootstrapSummary(result.tables["bootstrap"]))
    st.dataframe(windowFrame)


def renderCalibration(result: ExperimentResult) -> None:
    st.subheader("Reliability (pooled over evaluation blocks)")
    st.line_chart(result.tables["calibration"], x="meanPredicted", y="observedRate", color="strategy")
    st.dataframe(result.tables["calibration"])


def renderDrift(result: ExperimentResult) -> None:
    st.subheader("Feature drift vs. the training window")
    driftPerformance = result.tables["driftPerformance"]
    st.bar_chart(driftPerformance.set_index("block")[["meanKs", "meanJs"]])
    st.line_chart(driftPerformance.set_index("block")[["frozenPrAuc"]])
    st.write(f"Correlation(meanKS, frozen PR-AUC): `{result.metadata['corrMeanKsVsFrozenPrAuc']}`")
    st.caption("With few blocks this correlation is only indicative; drift is not proof of failure.")
    featureDrift = result.tables["featureDrift"]
    block = st.selectbox("Block", sorted(featureDrift["block"].unique()))
    st.dataframe(featureDrift[featureDrift["block"] == block])


def renderAbstention(result: ExperimentResult) -> None:
    st.subheader("Risk-coverage (mean error among accepted cases)")
    st.line_chart(averageRiskCoverage(result.tables["riskCoverage"]))
    st.dataframe(result.tables["riskCoverage"])


def renderStress(result: ExperimentResult) -> None:
    st.subheader("Synthetic stress tests (frozen model)")
    st.caption("Perturbations are artificial and not evidence of real-world drift.")
    stress = result.tables["stressTests"]
    st.line_chart(stress.pivot_table(index="severity", columns="perturbation", values="prAuc"))
    st.dataframe(stress)


def renderConfig(result: ExperimentResult) -> None:
    st.subheader("Saved configuration and environment")
    st.json(result.metadata)


def renderExport(result: ExperimentResult) -> None:
    st.subheader("Downloads")
    st.download_button("report.md", buildMarkdownReport(result), "report.md", "text/markdown", key="report")
    # Side-effect comprehension: one download button per saved table.
    _ = [st.download_button(f"{name}.csv", frame.to_csv(index=False), f"{name}.csv", "text/csv", key=name)
         for name, frame in result.tables.items()]


def renderInTab(tab, render, result: ExperimentResult) -> None:
    """Draw one renderer inside its tab container."""
    with tab:
        render(result)


def main() -> None:
    st.set_page_config(page_title="DriftLens", layout="wide")
    st.title("DriftLens")
    st.caption("Educational ML-reliability auditor under temporal data shift - not a banking decision system.")
    config = readSidebarConfig()
    frame, checksum = loadSelectedData()
    store = ResultStore(RESULT_DIR)

    if st.sidebar.button("Run experiment", type="primary"):
        if frame is None:
            st.sidebar.error("Provide a dataset first.")
        else:
            with st.spinner("Running all strategies..."):
                store.save(ExperimentRunner(config).run(frame, checksum))

    if not store.exists():
        st.info("Choose settings and press **Run experiment**.")
        return
    result = store.load()   # always display what was saved on disk
    tabs = st.tabs(["Audit", "Windows", "Calibration", "Drift", "Abstention", "Stress", "Config", "Export"])
    renderers = [renderAudit, renderWindows, renderCalibration, renderDrift,
                 renderAbstention, renderStress, renderConfig, renderExport]
    # Side-effect comprehension: render each tab with its matching function.
    _ = [renderInTab(tab, render, result) for tab, render in zip(tabs, renderers)]

main()
