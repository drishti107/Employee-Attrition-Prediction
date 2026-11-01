# Employee Attrition Prediction — upgraded mini project

Estimate an attrition-label probability, produce a class, and explain the principal individual factors. The research question is whether domain-guided feature engineering can improve prediction performance and interpretability together.

**Start with `docs/PROJECT_REPORT.docx`.** Measured conclusions are in `docs/RESULTS_AND_CONCLUSION.md`; the app displays the same saved results. The fictional IBM data have no defined future turnover horizon.

## Run the complete supplied project

Python 3.12 is recommended. Extract the ZIP before running commands, then open a terminal in the folder containing `app.py`:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python verify.py
python -m streamlit run app.py
```

Open the localhost address shown by Streamlit. The trained model is included; retraining is not needed to demonstrate the app. Complete Windows/Conda guidance is in `docs/RUN_WINDOWS.md`.

For a command-line explanation:

```bash
python predict.py --csv data/sample_employees.csv --row 0 --output outputs/my_explanation.json
```

Open `notebooks/Attrition_Project_Walkthrough.ipynb` in Jupyter or VS Code. Its stored outputs are executed, and its cells can be rerun against the included results and model files.

## What the upgrade adds

- Five repeats × five outer stratified folds; five-fold tuning inside each training fold.
- Three model families with original, original +8 and original +7 features on identical folds.
- Four engineered-block ablations and an age sensitivity analysis.
- Fold mean/SD, corrected paired intervals, win counts and Holm-adjusted AP comparisons.
- Full-data final refit selected from original/+7 candidates; sigmoid calibration and training-CV thresholds.
- Named domain contributions with formulas, plus six-group probability contributions and side-by-side explanations.
- Every held-out employee evaluated for K80 and top-three stability; 50 bootstrap refits per fold, including forests.
- TreeSHAP, permutation-ranking cross-checks, reliability curves and ECE.
- A human clarity rubric and an empty rating template. **No human study has been performed.**

`all` keeps eight engineered features for the original research comparison. `no_log` removes LogMonthlyIncome to avoid duplicate raw/log income attribution; it is the seven-feature candidate. The four ablations start from `no_log`, not `all`. Age is experimental only. Gender and marital status are excluded throughout.

## Reproduce the experiment

The complete study fits 350 outer models (each with inner tuning/calibration) plus 7,500 bootstrap models. It is substantially more expensive than running the app. Use a new destination so the supplied results remain intact:

```bash
python -m pip install -r requirements-report.txt
python reproduce.py --destination ../Attrition_Reproduced --jobs 6
```

Or, in a fresh project copy:

```bash
python -c "from pathlib import Path; Path('FILE_CHECKSUMS.json').unlink(missing_ok=True)"
python train.py --jobs 6
python interpretability.py --jobs 6
python scripts/build_outputs.py
python scripts/summarize_study.py
python -m pip install -r requirements-report.txt
python scripts/build_report.py
python scripts/build_notebook.py
python scripts/execute_notebook.py
python predict.py --csv data/sample_employees.csv --row 0 --output outputs/cli_example.json
python verify.py --with-cache --save
python tests/test_app.py
python scripts/checksums.py
```

The fit checkpoints in `.cache/` are regenerated during reproduction and fingerprinted; they are omitted from the distributable to keep it small. All held-out predictions, fold identities, hyperparameters, summary tables and bootstrap summary measurements are included. `models/example_fold.joblib` preserves held-out models for the worked example. The old single-split outputs are archived separately and are not primary evidence.

The checksum manifest validates the supplied files before editing. Rebuilding intentionally changes generated files, so the manual workflow removes the old manifest first and creates a new one after verification.

## Files to present

| File | Purpose |
|---|---|
| docs/PROJECT_REPORT.docx | Editable submission report |
| notebooks/Attrition_Project_Walkthrough.ipynb | Executed walkthrough |
| app.py | Local demonstration interface |
| protocol.json | Frozen upgrade design and selection rules |
| results/nested_cv_summary.csv | Mean/SD of held-out metrics |
| results/paired_comparisons.csv | Corrected paired differences and wins |
| results/interpretability_summary.csv | Compactness, stability and rank agreement intervals |
| results/calibration_summary.json | Held-out ECE summary |
| docs/UPGRADE_CHECKLIST.md | Plan-to-deliverable mapping and limitations |
| docs/HUMAN_EVALUATION_RUBRIC.md | Optional study protocol; no fabricated findings |

The study is exploratory on one previously examined fictional dataset. Selecting a winning family from outer-CV comparisons adds selection optimism. Interpretability intervals describe conditional split variation, not population uncertainty. Explanations are model associations, not causes or HR instructions. See `docs/METHODOLOGY.md` for precise definitions and limitations.

The supplied explanation run uses reference SHAP and a verified exact FastTreeSHAP accelerator where available. `results/explanation_backend.json` records the equivalence checks. The standard workflow above needs only SHAP. An optional Unix accelerator is documented in `requirements-fast.txt` and `scripts/fast_interpretability.py`; it is not required for running the app.

## Gap closure (version 2.1 additions)

These additions answer the gaps found in review. Each one reports what was run. Anything that needs data or people the project lacks is marked as not performed.

| Gap | Action | Files | Status |
|---|---|---|---|
| No independent test | Locked holdout: 294 employees split off with a fixed seed before any comparison; selection on the other 1,176 rows; winner scored once | `scripts/locked_holdout.py`, `results/holdout/` | Run; still the same fictional data |
| Fairness not measured | Subgroup audit by Gender, MaritalStatus and age band on saved held-out predictions, with employee-level bootstrap | `scripts/fairness_audit.py`, `results/fairness/` | Run; descriptive |
| No human interpretability evidence | Blinded study packet (10 participant cards, counterbalanced A/B, key, analysis script) | `scripts/human_study_kit.py`, `scripts/analyze_human_responses.py`, `human_study/` | Materials ready; **no ratings collected** |
| No dated data for a turnover-horizon test | Forward-in-time validation script, ready for a dated extract with a stated horizon | `scripts/temporal_validation.py` | Not performed; no dates in the dataset |
| Windows execution not verified | Ran the pinned environment and `verify.py` on Windows | `docs/WINDOWS_VERIFICATION.md` | See that file |

Run the new analyses (after the original `python train.py` and `python interpretability.py` results exist):

```bash
python scripts/locked_holdout.py --jobs 6
python scripts/fairness_audit.py
python scripts/human_study_kit.py
python scripts/analyze_human_responses.py      # reports "not performed" until responses.csv exists
python -m pytest -q tests/test_human_analysis.py   # or: python tests/test_human_analysis.py
```

`scripts/temporal_validation.py` requires a dated CSV with `EmployeeNumber`, `observation_date` and `Attrition`. It is not run on the IBM file.
