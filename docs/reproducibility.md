# Clean-run reproducibility checklist (O7)

Run these steps from a fresh clone on a standard laptop. No GPU, paid service or API key is needed.

| # | Step | Command | Expected |
|---|---|---|---|
| 1 | Create an environment | `python -m venv .venv` and activate it | - |
| 2 | Install | `pip install -r requirements.txt` | no errors |
| 3 | Fetch data | `python scripts/downloadData.py` | `data/raw/bank-additional-full.csv`, 41,188 rows; SHA-256 `74adfc57...afb4d8` |
| 4 | Tests | `python -m unittest discover -s tests -p "test*.py"` | `Ran 19 tests ... OK` |
| 5 | Logistic run | `python scripts/runExperiment.py --dataPath data/raw/bank-additional-full.csv --model logistic --outDir results/bank_logistic` | `Leakage-free windows: True` |
| 6 | HGB run | same as step 5 with `--model histGradientBoosting --outDir results/bank_hgb` | `Leakage-free windows: True` |
| 7 | Leakage demo | add `--leakageDemo --outDir results/bank_logistic_leakageDemo` | `unexcluded leakage columns: ['duration']` |
| 8 | Compare | diff the new `windowMetrics.csv` against the committed copies | identical up to float rounding (fixed seed 42) |
| 9 | App | `streamlit run app.py` → "Bundled UCI file" → Run experiment | 8 tabs; Audit tab shows the green leakage-audit message |
| 10 | Export | Export tab → download `report.md` and the CSVs | files open; values match `results/` |

Reference environment for the committed results: Python 3.15.0, numpy 2.5.5, pandas 3.1.1,
scikit-learn 1.9.1 (Windows 11). Run times: about 17-21 s per run.

Known source of variation: `updateSeconds` (wall-clock time) will differ between machines. All other metrics are deterministic.
