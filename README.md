# DriftLens

A reproducible study and Streamlit tool that compares **frozen**, **recalibrated** and **rolling-retrained**
models, plus confidence-based abstention, under order-based data shift (UCI Bank Marketing).
It is an educational prototype, not a banking decision system.

## Project layout
```
driftLens/
├── driftlens/                 # Python package (all ML logic, tested)
│   ├── config.py              # frozen ExperimentConfig
│   ├── dataLoader.py          # loading, audit, leakage flags, synthetic data
│   ├── splitter.py            # SplitManifest: 50% / 10% / 4 x 10% order-based split
│   ├── models.py              # LogisticRegression / HistGradientBoosting pipelines, SigmoidCalibrator
│   ├── strategies.py          # Frozen / Recalibrated / Rolling + leakage audit
│   ├── metrics.py             # PR-AUC, Brier, log loss, ECE, precision@top, risk-coverage, bootstrap
│   ├── driftMetrics.py        # KS, Wasserstein, Jensen-Shannon
│   ├── stressTest.py          # synthetic perturbations
│   ├── experiment.py          # ExperimentRunner -> result tables + metadata
│   ├── store.py               # CSV/JSON persistence
│   └── reporting.py           # Markdown report builder
├── app.py                     # Streamlit dashboard
├── scripts/
│   ├── downloadData.py        # fetch UCI data + provenance (date, SHA-256)
│   └── runExperiment.py       # CLI experiment runner
├── tests/testDriftLens.py     # unit + end-to-end tests
├── data/                      # raw data (not committed); see data/README.md
├── results/                   # saved experiment outputs (one folder per run)
├── docs/
│   ├── report.md              # research-style report (draft)
│   ├── literatureMatrix.md    # annotated literature matrix
│   ├── reproducibility.md     # clean-run checklist
│   └── AI_USE.md              # AI-use disclosure
├── requirements.txt, pyproject.toml
└── DriftLens_Project_Proposal.pdf
```

## Setup
    python -m venv .venv && .venv\Scripts\activate      # (Linux/macOS: source .venv/bin/activate)
    pip install -r requirements.txt
    python scripts/downloadData.py

## Run
    python scripts/runExperiment.py --dataPath data/raw/bank-additional-full.csv --model logistic --outDir results/bank_logistic
    python scripts/runExperiment.py --dataPath data/raw/bank-additional-full.csv --model histGradientBoosting --outDir results/bank_hgb
    python scripts/runExperiment.py --dataPath data/raw/bank-additional-full.csv --leakageDemo --outDir results/bank_logistic_leakageDemo
    python scripts/runExperiment.py --synthetic --outDir results/synthetic     # demo data, not real results
    streamlit run app.py
    python -m unittest discover -s tests -p "test*.py"

Each run folder contains one CSV per table, `metadata.json` (config, split manifest, versions, checksum)
and `report.md`.

## Conventions
- camelCase names (PascalCase classes); comments and docstrings throughout.
- No `for`/`while` statements: comprehensions, pure functions and registries only.
- Dependencies: numpy, pandas, scikit-learn, streamlit. Distances and calibration are implemented by hand.

## Safeguards
- `duration` is excluded. It is kept only in `--leakageDemo` runs, which are flagged in the report and the UI.
- No strategy sees labels from its current or future block. A test checks this by flipping those labels.
- Calibration rows never overlap model-fit rows.
- Every run saves its config, split manifest, versions and data checksum.
- Row order stands in for time, and the stress tests are synthetic.

## Limitations and AI use
See [docs/report.md](docs/report.md) §6 and [docs/AI_USE.md](docs/AI_USE.md).
