"""Create a walkthrough that executes against the supplied measured artifacts."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s))
def code(s):cells.append(nbf.v4.new_code_cell(s))
md('''# Employee attrition prediction: upgraded study

**Objective:** estimate an individual attrition-label probability, explain its major associated factors, and investigate whether domain-informed features improve prediction and interpretability together.

This notebook analyzes the actually executed repeated nested-CV experiment. Full training is a separate script because it includes 350 outer fits with tuning and 7,500 bootstrap refits. The data are fictional, have no defined forecast horizon, and have already been examined in an earlier single-split study. No human comprehension study was performed.''')
code('''from pathlib import Path
import sys, json
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'attrition.py').exists())
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import joblib
from IPython.display import display, Image
from attrition import DomainFeatures, linear_contributions, calibration_bins
from explanations import FEATURE_GUIDE, make_report, compare_reports
from train import load_data, outer_splits
RESULTS = ROOT / 'results'
pd.set_option('display.max_columns', 12)
''')
md('## 1. Audit data and target\nAll 1,470 employees participate in repeated outer holdouts. Age, Gender and MaritalStatus are excluded from main models; only the explicit sensitivity model includes Age.')
code('''data, X, y = load_data()
display(pd.DataFrame({'count': data.Attrition.value_counts()}))
assert data.shape == (1470, 35) and int(y.sum()) == 237
print('Shape:', data.shape, 'Positive prevalence:', round(y.mean(), 4))
json.loads((RESULTS / 'data_audit.json').read_text())''')
md('## 2. Domain-informed representations\nThe original eight-feature design is preserved. A seven-feature candidate removes LogMonthlyIncome to avoid duplicate raw/log income attribution. Every learned role-income median is fit inside training folds.')
code('''display(pd.DataFrame(FEATURE_GUIDE, columns=['Block', 'Feature', 'Formula', 'Interpretation']))
for mode in ['raw', 'all', 'no_log']:
    transformer = DomainFeatures(mode).fit(X.iloc[:1000])
    print(mode, transformer.transform(X.iloc[[1001]]).shape[1], 'source attributes')''')
md('## 3. Matched repeated nested cross-validation\nOuter: five repeats × five stratified folds. Inner: five folds, original eight-setting grid per family. Hyperparameters, preprocessing, sigmoid and threshold use outer-training data only. Calibration and threshold development reuse internal training scores; the untouched outer fold evaluates the complete procedure.')
code('''manifest = json.loads((RESULTS / 'split_manifest.json').read_text())
for (tr, te), fold in zip(outer_splits(X, y), manifest):
    assert list(tr) == fold['train_indices'] and list(te) == fold['test_indices']
    assert not set(tr) & set(te)
predictions = pd.read_csv(RESULTS / 'outer_predictions.csv.gz', float_precision='round_trip')
assert len(predictions) == 102900
assert predictions.groupby(['model', 'repeat']).row_index.nunique().eq(1470).all()
print('25 matched outer folds, 14 specifications, 7,350 held-out observations per specification.')''')
md('## 4. Predictive performance\nAP and ROC-AUC: higher is better. Brier: lower is better. Values are outer-fold means and sample SDs, not independent-repeat estimates. The selected model is refitted on all development rows.')
code('''summary = pd.read_csv(RESULTS / 'nested_cv_summary.csv')
core = summary[summary.feature_mode.isin(['raw', 'all', 'no_log'])]
core[['model', 'average_precision_mean', 'average_precision_std', 'roc_auc_mean', 'roc_auc_std', 'brier_mean', 'brier_std']].round(4)''')
code('''display(Image(filename=str(RESULTS / 'feature_comparison.png')))''')
md('## 5. Paired uncertainty\nThe corrected SE is sqrt((1/25 + 1/4) × variance of 25 paired differences). Intervals and tests use t(24); Holm adjustment covers three AP family tests separately within each specification. Overlapping folds are not independent. Selecting the winner across outer comparisons adds selection optimism; no new independent test is available.')
code('''paired = pd.read_csv(RESULTS / 'paired_comparisons.csv')
paired[(paired.metric == 'average_precision') & paired.comparison.isin(['all', 'no_log'])][['candidate', 'difference', 'ci_low', 'ci_high', 'wins', 'p_two_sided', 'p_holm_AP']].round(4)''')
md('## 6. Block ablations and age sensitivity\nAblations remove one engineered block from the seven-feature model; original predictors remain. Age sensitivity does not establish fairness. These diagnostics are exploratory and excluded from deployment selection.')
code('''display(summary[summary.feature_mode.str.startswith('without_') | summary.feature_mode.isin(['no_log', 'no_log_age'])][['model', 'average_precision_mean', 'roc_auc_mean', 'brier_mean']].round(4))
paired[paired.comparison.isin(['ablation', 'age']) & (paired.metric == 'average_precision')][['candidate', 'difference', 'ci_low', 'ci_high']].round(4)''')
md('## 7. Calibration\nTen equal-width bins, ECE computed separately on each repeat’s held-out predictions. Pooled bin counts repeat the same employees. ECE depends on bins and sample size, and does not guarantee the full-data refit is calibrated.')
code('''calibration = json.loads((RESULTS / 'calibration_summary.json').read_text())
print('Mean repeat ECE:', calibration['mean_repeat_ECE'])
print('SD across repeats:', calibration['sd_repeat_ECE'])
display(Image(filename=str(RESULTS / 'calibration_curve.png')))
pd.read_csv(RESULTS / 'reliability_pooled.csv').round(4)''')
md('## 8. Interpretation proxies on all held-out employees\nK80 counts attributes covering 80% of absolute contribution mass. Top-three Jaccard measures agreement after 50 training bootstrap refits per outer fold. Logistic explanations use calibrated log-odds; forest TreeSHAP explains base probability before calibration. Intervals bootstrap five repeat means and describe conditional split variation, not population uncertainty.')
code('''interpretation = pd.read_csv(RESULTS / 'interpretability_summary.csv')
assert interpretation.unique_employees.eq(1470).all()
assert interpretation.refits_per_fold.eq(50).all()
interpretation[['model', 'metric', 'estimate', 'ci_low', 'ci_high']].round(4)''')
code('''display(Image(filename=str(RESULTS / 'interpretability_comparison.png')))
pd.read_csv(RESULTS / 'interpretability_paired.csv').round(4)''')
md('## 9. Explanation ranking cross-check\nSpearman compares mean absolute attribution ranks with held-out permutation AP-drop ranks. All one-hot levels of an attribute are permuted together. Derived columns are perturbed independently; inconsistent synthetic combinations can result. Ranking agreement is distinct from mathematical reconstruction and does not establish causal validity.')
code('''importance = pd.read_csv(RESULTS / 'global_attributions.csv')
importance[importance.model == 'logistic_no_log'].sort_values('mean_absolute_contribution', ascending=False).head(12).round(4)''')
md('## 10. Same employee, original versus engineered\nThe prior illustrative employee 478 is reused. These example models did not train on that employee. It was previously chosen as a high-risk illustration, not a representative random example. The app’s final full-data model is separate.')
code('''example_bundle = joblib.load(ROOT / 'models/example_fold.joblib')
row = example_bundle['row']
comparison = compare_reports(example_bundle['models']['logistic_raw'], example_bundle['models']['logistic_no_log'], row)
for report in comparison.values():
    if isinstance(report, dict):
        assert report['group_additivity_error'] < 1e-9
        assert abs(report['feature_explanation']['reconstructed_probability'] - report['probability']) < 1e-9
pd.DataFrame([{'model': key, 'probability': comparison[key]['probability'], 'threshold': comparison[key]['threshold'], 'class': comparison[key]['predicted_label']} for key in ['original', 'engineered']])''')
code('''details = pd.DataFrame(comparison['engineered']['feature_explanation']['factors'])
display(details[details.is_engineered][['display_name', 'value', 'contribution_calibrated_log_odds', 'definition']])
display(Image(filename=str(RESULTS / 'example_factors.png')))
(ROOT / 'outputs/notebook_employee_explanation.json').write_text(json.dumps(comparison, indent=2, allow_nan=False))
print('Exported the computed comparison. Log-odds and probability points have different baselines and must not be mixed.')''')
md('## 11. Final model and measured answer\nCandidate selection uses mean outer AP among original/seven-feature models, then lower Brier and model key. A full-data refit uses training CV calibration and threshold selection. Demonstration predictions on those training records are not evaluation results.')
code('''bundle = joblib.load(ROOT / 'models/bundle.joblib')
model = bundle['models'][bundle['selected']]
print('Selected model:', bundle['selected'], 'Final threshold:', model.threshold)
conclusion = json.loads((RESULTS / 'conclusion.json').read_text())
for field in ['answer', 'comparison_statement', 'eight_feature_interpretability']:
    print(conclusion[field])''')
md('## 12. Reproduce and extend\nRun `python reproduce.py --destination ../Attrition_Reproduced --jobs 6` from the project root to repeat the complete experiment in a new folder. Run `python verify.py` to validate the supplied artifacts. The optional human-study rubric uses 5–10 classmates and anonymous, counterbalanced 1–5 ratings. **It has not been performed.**\n\nLimitations: fictional cross-sectional data, unknown timing and horizon, previously examined data, winner-selection optimism, dependent folds, five-repeat descriptive interpretation intervals, correlated features and no external, human-understanding or fairness validation. Read `docs/METHODOLOGY.md` for definitions and primary references.')
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python','version':'3.12'}})
nbf.write(nb,ROOT/'notebooks/Attrition_Project_Walkthrough.ipynb')
print('Created walkthrough:',sum(c.cell_type=='code' for c in cells),'code cells')
