"""Streamlit front end. Run with:  streamlit run app.py

The UI only calls tested experiment functions and plots values that were
saved to disk, so what you see always matches the exported files.
"""
import io
from pathlib import Path
from typing import List, Optional

import altair as alt
import pandas as pd
import streamlit as st

from driftlens.config import ExperimentConfig
from driftlens.dataLoader import computeChecksumFromBytes, loadBankCsv, makeSyntheticBankData
from driftlens.experiment import ExperimentResult, ExperimentRunner
from driftlens.reporting import averageRiskCoverage, bootstrapSummary, buildMarkdownReport, summarizeByStrategy
from driftlens.store import ResultStore

RESULTS_ROOT = Path("results")
LATEST_DIR = RESULTS_ROOT / "latest"
BUNDLED_DATA = Path("data/raw/bank-additional-full.csv")

STRATEGIES = ["frozen", "recalibrated", "rolling"]
STRATEGY_COLORS = ["#6B7785", "#E8A33D", "#0F6E6E"]
STRATEGY_SCALE = alt.Scale(domain=STRATEGIES, range=STRATEGY_COLORS)
METRICS = {  # key -> (label, higher is better)
    "prAuc": ("PR-AUC", True), "brier": ("Brier score", False), "logLoss": ("Log loss", False),
    "ece": ("Calibration error (ECE)", False), "precisionTop": ("Precision @ top 10%", True),
}
RUN_LABELS = {"bank_logistic": "UCI Bank · Logistic Regression", "bank_hgb": "UCI Bank · Gradient Boosting",
              "bank_logistic_leakageDemo": "UCI Bank · Logistic + duration (leakage demo)", "latest": "Latest run from this app"}

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1300px;}
</style>
"""


# ---------- data sources ----------

def savedRunDirs() -> List[Path]:
    """Every results folder that contains a saved run."""
    return sorted([p for p in RESULTS_ROOT.glob("*") if (p / "metadata.json").exists()],
                  key=lambda p: list(RUN_LABELS).index(p.name) if p.name in RUN_LABELS else 99)


@st.cache_data(show_spinner=False)
def loadRun(directory: str) -> ExperimentResult:
    return ResultStore(directory).load()


def readNewRunConfig() -> ExperimentConfig:
    """Settings for a fresh run; invalid combinations stop the app politely."""
    modelName = st.sidebar.selectbox("Model", ["logistic", "histGradientBoosting"],
                                     format_func=lambda m: {"logistic": "Logistic Regression", "histGradientBoosting": "Gradient Boosting"}[m])
    with st.sidebar.expander("Advanced settings"):
        seed = int(st.number_input("Random seed", value=42, step=1))
        trainFraction = st.slider("Initial training share", 0.2, 0.7, 0.5, 0.05)
        calibrationFraction = st.slider("Initial calibration share", 0.05, 0.2, 0.1, 0.05)
        numBlocks = st.slider("Evaluation blocks", 2, 8, 4)
        bootstrapSamples = st.slider("Bootstrap resamples", 20, 500, 100, 20)
        leakageDemo = st.checkbox("Leakage demo: keep `duration`", value=False,
                                  help="Post-outcome column; results are NOT valid primary results.")
    try:
        return ExperimentConfig(seed=seed, modelName=modelName, trainFraction=trainFraction,
                                calibrationFraction=calibrationFraction, numEvalBlocks=numBlocks,
                                bootstrapSamples=bootstrapSamples, dropColumns=() if leakageDemo else ("duration",))
    except ValueError as error:
        st.sidebar.error(str(error))
        st.stop()


def loadSelectedData():
    """Return (frame, checksum) for the chosen source, or (None, None)."""
    options = (["UCI Bank file (bundled)"] if BUNDLED_DATA.exists() else []) + ["Synthetic demo data", "Upload CSV"]
    source = st.sidebar.radio("Dataset", options)
    if source.startswith("UCI"):
        return loadBankCsv(BUNDLED_DATA), computeChecksumFromBytes(BUNDLED_DATA.read_bytes())
    if source.startswith("Synthetic"):
        return makeSyntheticBankData(), "synthetic"
    upload = st.sidebar.file_uploader("CSV with a `y` column", type="csv")
    if upload is None:
        return None, None
    content = upload.getvalue()
    return loadBankCsv(io.BytesIO(content)), computeChecksumFromBytes(content)


def chooseRun() -> Optional[ExperimentResult]:
    """Sidebar: open a saved run, or run a new experiment."""
    mode = st.sidebar.radio("Mode", ["Open a saved run", "Run a new experiment"], horizontal=False)
    if mode == "Run a new experiment":
        config = readNewRunConfig()
        frame, checksum = loadSelectedData()
        if st.sidebar.button("Run experiment", type="primary", width="stretch"):
            if frame is None:
                st.sidebar.error("Provide a dataset first.")
            else:
                with st.spinner("Training and evaluating all three strategies (about 20 s)..."):
                    ResultStore(LATEST_DIR).save(ExperimentRunner(config).run(frame, checksum))
                loadRun.clear()
        return loadRun(str(LATEST_DIR)) if (LATEST_DIR / "metadata.json").exists() else None
    runs = savedRunDirs()
    if not runs:
        st.sidebar.info("No saved runs yet. Switch to 'Run a new experiment'.")
        return None
    chosen = st.sidebar.selectbox("Saved run", runs, format_func=lambda p: RUN_LABELS.get(p.name, p.name))
    return loadRun(str(chosen))


# ---------- helpers ----------

def hint(text: str) -> None:
    st.caption(text)


def bestStrategy(summary: pd.DataFrame, metric: str) -> str:
    higher = METRICS[metric][1]
    row = summary.loc[summary[metric].idxmax() if higher else summary[metric].idxmin()]
    return row["strategy"]


def strategyLegend() -> alt.Color:
    return alt.Color("strategy:N", scale=STRATEGY_SCALE, title="Strategy", legend=alt.Legend(orient="bottom"))


# ---------- tabs ----------

def renderHeader(result: ExperimentResult) -> None:
    meta = result.metadata
    model = {"logistic": "Logistic Regression", "histGradientBoosting": "Gradient Boosting"}[meta["config"]["modelName"]]
    source = "synthetic demo data" if meta.get("sourceChecksum") == "synthetic" else "UCI Bank Marketing"
    st.title("DriftLens")
    st.markdown(f"**{model}** on **{source}** · {meta['numRows']:,} rows · positive rate {meta['positiveRate']:.1%} · "
                f"{meta['config']['numEvalBlocks']} evaluation blocks · seed {meta['config']['seed']}")
    if meta["unexcludedLeakage"]:
        st.error(f"Leakage demonstration: this run keeps {meta['unexcludedLeakage']}, which is only known after the "
                 "outcome. Its scores are inflated and are not valid results.")
    if not meta["allLeakageFree"]:
        st.error("Temporal leakage audit FAILED: results are invalid.")


def renderOverview(result: ExperimentResult) -> None:
    windowFrame = result.tables["windowMetrics"]
    summary = summarizeByStrategy(windowFrame)
    byName = summary.set_index("strategy")
    st.caption("Mean over the evaluation blocks. Bold marks the best strategy for each row.")
    st.dataframe(comparisonTable(byName), width="stretch")

    st.subheader("Results")
    boot = bootstrapSummary(result.tables["bootstrap"])
    prBoot = boot[boot["metric"] == "prAuc"]
    risk = averageRiskCoverage(result.tables["riskCoverage"])
    best = bestStrategy(summary, "prAuc")
    corr = result.metadata.get("corrMeanKsVsFrozenPrAuc")
    findings = [
        f"**{best.capitalize()}** has the best mean PR-AUC ({byName.loc[best, 'prAuc']:.3f} vs frozen {byName.loc['frozen', 'prAuc']:.3f}).",
        "Paired bootstrap, PR-AUC vs frozen: " + "; ".join([
            f"{name} better in {int((prBoot[prBoot['strategy'] == name]['verdict'] == 'above frozen').sum())}, "
            f"worse in {int((prBoot[prBoot['strategy'] == name]['verdict'] == 'below frozen').sum())} of "
            f"{int((prBoot['strategy'] == name).sum())} blocks" for name in ["recalibrated", "rolling"]]) + ".",
        "Abstaining on the least confident half changes error by: " + ", ".join([
            f"{name} {risk.loc[1.0, name]:.3f} → {risk.loc[0.5, name]:.3f}" for name in STRATEGIES if 0.5 in risk.index]) + ".",
        f"Feature drift vs frozen PR-AUC correlation: r = {corr:.2f} (few blocks, indicative only)." if corr is not None
        else "Drift/performance correlation is undefined for this run.",
    ]
    st.markdown("\n".join([f"{k + 1}. {text}" for k, text in enumerate(findings)]))
    with st.expander("Summary table"):
        st.dataframe(summary.round(3), hide_index=True, width="stretch")


def comparisonTable(byName: pd.DataFrame):
    """Metrics as rows, strategies as columns, best value in bold."""
    rows = list(METRICS) + ["updateSeconds", "newLabelsUsed"]
    labels = {**{k: f"{v[0]} ({'higher' if v[1] else 'lower'} is better)" for k, v in METRICS.items()},
              "updateSeconds": "Update time per block (s)", "newLabelsUsed": "New labels per block"}
    table = byName.loc[STRATEGIES, rows].T.rename(index=labels, columns=str.capitalize)
    bestOf = {labels[k]: (max if METRICS[k][1] else min)(table.loc[labels[k]]) for k in METRICS}
    return table.style.format(lambda v: f"{v:,.0f}" if v >= 100 else f"{v:.3f}").apply(
        lambda row: ["font-weight:700" if row.name in bestOf and v == bestOf[row.name] else "" for v in row], axis=1)


def renderOverTime(result: ExperimentResult) -> None:
    windowFrame = result.tables["windowMetrics"]
    metric = st.segmented_control("Metric", list(METRICS), format_func=lambda k: METRICS[k][0], default="prAuc") or "prAuc"
    label, higher = METRICS[metric]
    hint(f"{'Higher' if higher else 'Lower'} is better. Dashed grey line: the block's positive rate.")
    lines = alt.Chart(windowFrame).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X("block:O", title="Evaluation block (later →)", axis=alt.Axis(labelAngle=0)), y=alt.Y(f"{metric}:Q", title=label),
        color=strategyLegend(), tooltip=["strategy", "block", alt.Tooltip(f"{metric}:Q", format=".3f")])
    prevalence = alt.Chart(windowFrame[windowFrame["strategy"] == "frozen"]).mark_line(strokeDash=[6, 4], color="#9AA5B1").encode(
        x="block:O", y="positiveRate:Q", tooltip=[alt.Tooltip("positiveRate:Q", format=".3f", title="positive rate")])
    st.altair_chart((lines + prevalence) if metric in ("prAuc", "precisionTop") else lines, width="stretch")

    st.subheader("Paired bootstrap vs frozen")
    hint("95% interval of strategy minus frozen on the same rows, 200 resamples.")
    st.dataframe(bootstrapView(result.tables["bootstrap"]).style.map(
        lambda v: {"better": "color:#0F6E6E;font-weight:600", "worse": "color:#C8553D;font-weight:600"}.get(v, ""),
        subset=["Result"]).format(precision=3), hide_index=True, width="stretch")
    with st.expander("All per-window numbers"):
        st.dataframe(windowFrame.round(4), hide_index=True, width="stretch")


def bootstrapView(bootstrapFrame: pd.DataFrame) -> pd.DataFrame:
    """Readable bootstrap table whose verdict respects each metric's direction."""
    boot = bootstrapSummary(bootstrapFrame)
    higherBetter = boot["metric"].map(lambda m: METRICS[m][1])
    flip = {"above frozen": "better", "below frozen": "worse"}
    flipDown = {"above frozen": "worse", "below frozen": "better"}
    result = [(flip if h else flipDown).get(v, "no clear difference") for h, v in zip(higherBetter, boot["verdict"])]
    return pd.DataFrame({"Strategy": boot["strategy"], "Block": boot["block"],
                         "Metric": boot["metric"].map(lambda m: METRICS[m][0]), "Difference": boot["observedDiff"],
                         "95% CI low": boot["ciLow"], "95% CI high": boot["ciHigh"], "Result": result})


def renderCalibration(result: ExperimentResult) -> None:
    calibration = result.tables["calibration"]
    hint("Dashed line = perfect calibration. Pooled over evaluation blocks; point size is the bin count.")
    diagonal = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color="#9AA5B1").encode(x="x:Q", y="y:Q")
    curves = alt.Chart(calibration).mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X("meanPredicted:Q", title="Mean predicted probability", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("observedRate:Q", title="Observed positive rate", scale=alt.Scale(domain=[0, 1])),
        color=strategyLegend(), size=alt.Size("count:Q", legend=None),
        tooltip=["strategy", alt.Tooltip("meanPredicted:Q", format=".3f"), alt.Tooltip("observedRate:Q", format=".3f"), "count"])
    left, right = st.columns([3, 2])
    left.altair_chart(diagonal + curves, width="stretch")
    eceChart = alt.Chart(result.tables["windowMetrics"]).mark_bar().encode(
        x=alt.X("block:O", title="Block", axis=alt.Axis(labelAngle=0)), xOffset="strategy:N", y=alt.Y("ece:Q", title="ECE (lower is better)"),
        color=strategyLegend(), tooltip=["strategy", "block", alt.Tooltip("ece:Q", format=".3f")])
    right.altair_chart(eceChart, width="stretch")


def renderAbstention(result: ExperimentResult) -> None:
    hint("The model answers only its most confident cases. With a useful confidence score, error drops as coverage drops.")
    risk = averageRiskCoverage(result.tables["riskCoverage"]).reset_index().melt("coverage", var_name="strategy", value_name="error")
    chart = alt.Chart(risk).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X("coverage:Q", title="Coverage (share of cases the model answers)", scale=alt.Scale(domain=[1, 0.45]),
                axis=alt.Axis(format="%")),
        y=alt.Y("error:Q", title="Error among answered cases"), color=strategyLegend(),
        tooltip=["strategy", alt.Tooltip("coverage:Q", format=".0%"), alt.Tooltip("error:Q", format=".3f")])
    st.altair_chart(chart, width="stretch")
    with st.expander("Per-block numbers"):
        st.dataframe(result.tables["riskCoverage"].round(4), hide_index=True, width="stretch")


def renderDrift(result: ExperimentResult) -> None:
    featureDrift = result.tables["featureDrift"]
    driftPerformance = result.tables["driftPerformance"]
    corr = result.metadata.get("corrMeanKsVsFrozenPrAuc")
    left, right = st.columns([3, 2])
    numeric = featureDrift.dropna(subset=["ks"])
    heat = alt.Chart(numeric).mark_rect().encode(
        x=alt.X("block:O", title="Block", axis=alt.Axis(labelAngle=0)), y=alt.Y("feature:N", title=None, sort="-color"),
        color=alt.Color("ks:Q", title="KS", scale=alt.Scale(scheme="teals", domain=[0, 1])),
        tooltip=["feature", "block", alt.Tooltip("ks:Q", format=".3f"), alt.Tooltip("wasserstein:Q", format=".3f")])
    left.markdown("**Numeric features: KS distance**")
    left.altair_chart(heat, width="stretch")
    right.metric("Correlation: mean KS vs frozen PR-AUC", "n/a" if corr is None else f"{corr:.2f}")
    right.caption(f"Computed from only {len(driftPerformance)} blocks.")
    right.dataframe(driftPerformance.round(3), hide_index=True, width="stretch")
    categorical = featureDrift.dropna(subset=["jensenShannon"])
    if len(categorical):
        with st.expander("Categorical features: Jensen-Shannon divergence"):
            st.dataframe(categorical.pivot(index="feature", columns="block", values="jensenShannon").round(3),
                         width="stretch")


def renderStress(result: ExperimentResult) -> None:
    hint("Frozen model on deliberately perturbed copies of the evaluation rows. Synthetic, not observed drift.")
    stress = result.tables["stressTests"]
    metric = st.segmented_control("Metric ", ["prAuc", "brier", "ece"], format_func=lambda k: METRICS[k][0], default="prAuc") or "prAuc"
    chart = alt.Chart(stress).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X("severity:Q", title="Severity"), y=alt.Y(f"{metric}:Q", title=METRICS[metric][0]),
        color=alt.Color("perturbation:N", title="Perturbation", legend=alt.Legend(orient="bottom"),
                        scale=alt.Scale(range=["#0F6E6E", "#E8A33D", "#C8553D"])),
        tooltip=["perturbation", "severity", alt.Tooltip(f"{metric}:Q", format=".3f")])
    st.altair_chart(chart, width="stretch")


def renderAudit(result: ExperimentResult) -> None:
    meta = result.metadata
    (st.success if meta["allLeakageFree"] else st.error)(
        "Leakage audit passed: every strategy used only labels from before each block." if meta["allLeakageFree"]
        else "Leakage audit FAILED.")
    st.markdown(f"Excluded as unavailable at decision time: `{', '.join(meta['leakageColumnsExcluded']) or 'none'}` · "
                "row order is used as a proxy for time (no per-row timestamps).")
    manifest = result.tables["splitManifest"]
    bar = alt.Chart(manifest).mark_bar(height=40).encode(
        x=alt.X("start:Q", title="Row index"), x2="end:Q",
        color=alt.Color("segment:N", legend=alt.Legend(orient="bottom", title=None),
                        scale=alt.Scale(range=["#0F6E6E", "#E8A33D", "#3D5A80", "#5C7AA3", "#7E99BE", "#A3B8D6", "#C4D3E8", "#DCE5F2"])),
        tooltip=["segment", "start", "end", "size"])
    st.altair_chart(bar.properties(height=90), width="stretch")
    with st.expander("Leakage audit (one row per strategy and block)"):
        st.dataframe(result.tables["leakageAudit"], hide_index=True, width="stretch")
    with st.expander("Data audit (one row per column)"):
        st.dataframe(result.tables["dataAudit"].round(4), hide_index=True, width="stretch")
    with st.expander("Saved configuration and environment"):
        st.json(meta)


def renderExport(result: ExperimentResult) -> None:
    st.download_button("Download report (Markdown)", buildMarkdownReport(result), "report.md", "text/markdown",
                       key="report", type="primary")
    columns = st.columns(3)
    _ = [columns[k % 3].download_button(f"{name}.csv", frame.to_csv(index=False), f"{name}.csv", "text/csv",
                                        key=name, width="stretch")
         for k, (name, frame) in enumerate(result.tables.items())]


def renderInTab(tab, render, result: ExperimentResult) -> None:
    with tab:
        render(result)


def renderWelcome() -> None:
    st.title("DriftLens")
    st.markdown("An educational tool for checking how a machine-learning model holds up as data change over time. "
                "It compares three ways of maintaining a model (**frozen**, **recalibrated**, **rolling retraining**) "
                "and shows what abstaining on uncertain cases buys.")
    st.info("Pick a saved run in the sidebar, or switch to *Run a new experiment*.")


def main() -> None:
    st.set_page_config(page_title="DriftLens", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    result = chooseRun()
    st.sidebar.caption("Educational prototype, not a banking decision system.")
    if result is None:
        renderWelcome()
        return
    renderHeader(result)
    names = ["Overview", "Over time", "Calibration", "Abstention", "Drift", "Stress tests", "Data & audit", "Export"]
    renderers = [renderOverview, renderOverTime, renderCalibration, renderAbstention,
                 renderDrift, renderStress, renderAudit, renderExport]
    _ = [renderInTab(tab, render, result) for tab, render in zip(st.tabs(names), renderers)]


main()
