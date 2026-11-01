"""Run locally with: python -m streamlit run app.py"""
from pathlib import Path
import hashlib
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from attrition import GROUPS, validate_input
from explanations import FEATURE_GUIDE, display_name, make_report, compare_reports

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Employee Attrition Lab', page_icon='📊', layout='wide')
st.title('Employee Attrition Lab')
st.write('Estimate attrition probability, inspect the contributing factors, and test whether domain features help.')
st.caption('Educational prototype · Fictional IBM HR data · No defined prediction time horizon')


@st.cache_resource
def load_bundle():
    return joblib.load(ROOT / 'models/bundle.joblib')


def compact_metrics(frame):
    return frame[['model', 'average_precision', 'roc_auc', 'brier', 'precision', 'recall', 'f1']].rename(
        columns={'model': 'Model', 'average_precision': 'Average precision ↑', 'roc_auc': 'ROC-AUC ↑',
                 'brier': 'Brier ↓', 'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1'})


def signed_plot(factors):
    frame = pd.DataFrame(factors).sort_values('contribution_probability_points')
    values = frame.contribution_probability_points
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.barh(frame.factor_group, values, color=['#C4662B' if v > 0 else '#257985' for v in values])
    ax.axvline(0, color='#718096', linewidth=.8)
    ax.set_xlabel('Contribution to estimated probability (percentage points)')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    fig.tight_layout()
    return fig


def show_report(report):
    a, b, c = st.columns(3)
    a.metric('Estimated attrition probability', f"{report['probability']:.1%}")
    b.metric('Predicted class', report['predicted_label'])
    c.metric('Decision threshold', f"{report['threshold']:.1%}")
    st.caption('The label is Attrition when probability is at or above the threshold selected from training-only cross-validated scores. It is not a certainty.')
    st.subheader('What contributes to this prediction?')
    cols = st.columns(2)
    for col, title, field in [(cols[0], 'Raises the prediction', 'principal_increasing_groups'),
                               (cols[1], 'Lowers the prediction', 'principal_decreasing_groups')]:
        with col:
            st.markdown('**' + title + '**')
            if not report[field]:
                st.write('No group contributes in this direction for this profile.')
            for factor in report[field]:
                st.write(f"{factor['factor_group']}: {factor['contribution_probability_points']:+.2f} percentage points")
    fig = signed_plot(report['factors'])
    st.pyplot(fig)
    plt.close(fig)
    total = sum(f['contribution_probability_points'] for f in report['factors'])
    st.write(f"Reference {report['reference_probability']:.2%} + total contribution {total:+.2f} percentage points = {report['probability']:.2%}.")
    st.caption('Displayed values are rounded. Reference: average prediction for 32 fixed training employees. These explanations describe model associations, not causes or promised effects of HR actions.')
    if 'feature_explanation' in report:
        details = pd.DataFrame(report['feature_explanation']['factors'])
        st.subheader('Individual employee attributes')
        st.write('The strongest individual contributions are shown below, with the values you submitted.')
        for factor in report['feature_explanation']['factors'][:3]:
            st.write(f"• {factor['display_name']} = {factor['value']}: {factor['direction']} the model score relative to its training reference ({factor['contribution_calibrated_log_odds']:+.3f} log-odds).")
        st.dataframe(details[['display_name', 'value', 'contribution_calibrated_log_odds', 'is_engineered', 'definition']].head(10).astype({'value': str}),
                     hide_index=True, width='stretch',
                     column_config={'display_name': 'Attribute', 'value': 'Submitted value',
                                    'contribution_calibrated_log_odds': st.column_config.NumberColumn('Log-odds contribution', format='%+.3f'),
                                    'is_engineered': 'Domain feature', 'definition': 'Definition'})
        with st.expander('All contributions and mathematical reconstruction'):
            st.dataframe(details.astype({'value': str}), hide_index=True, width='stretch')
            st.write('Feature contributions are in calibrated log-odds. They cannot be added directly to probability percentages.')
            st.code('p = sigmoid(baseline_log_odds + sum(feature_contributions))')
            st.write(f"Reconstructed probability: {report['feature_explanation']['reconstructed_probability']:.8f}")
            st.write('This feature reference differs from the group explanation reference above; each decomposition is internally additive.')
    st.download_button('Download complete explanation', json.dumps(report, indent=2, allow_nan=False),
                       file_name='attrition_explanation.json', mime='application/json', key='download_report')


if not (ROOT / 'models/bundle.joblib').exists():
    st.error('The trained model is missing. Extract the complete project ZIP.'); st.stop()
try:
    bundle = load_bundle()
except Exception as error:
    st.error('Cannot load the model. Install requirements.txt in Python 3.12. Details: ' + str(error)); st.stop()
model = bundle['models'][bundle['selected']]
reference = model.reference_X
st.caption('Final model: ' + bundle['selected'] + ' · Selected by repeated nested-CV average precision')
tab_predict, tab_experiment, tab_calibration, tab_global, tab_guide = st.tabs(
    ['Predict and explain', 'Feature experiment', 'Calibration', 'Interpretability', 'Project guide'])

with tab_predict:
    st.subheader('Employee profile')
    upload = st.file_uploader('Optional CSV upload', type=['csv'], help='Use data/sample_employees.csv as a schema template. All 27 predictors are required.')
    if upload is None:
        profiles = reference.reset_index(drop=True)
        source_key = 'training'
        st.caption('Start with a fictional training employee and edit the inputs. Demo records do not provide an evaluation score.')
    else:
        try:
            profiles = validate_input(pd.read_csv(upload), reference).reset_index(drop=True)
            source_key = hashlib.sha256(upload.getvalue()).hexdigest()[:12]
        except (ValueError, pd.errors.ParserError) as error:
            st.error(str(error)); st.stop()
    index = int(st.number_input('Record index (starts at zero)', min_value=0, max_value=len(profiles)-1,
                                value=0, step=1, key='record_' + source_key))
    selected_row = profiles.iloc[[index]].copy()
    profile_key = source_key + '_' + str(index)
    st.download_button('Download a five-row CSV template', (ROOT / 'data/sample_employees.csv').read_bytes(),
                       'sample_employees.csv', 'text/csv')
    with st.form('employee_profile'):
        edited = {}
        for group, columns in GROUPS.items():
            with st.expander(group, expanded=group == 'Workload and commute'):
                cols = st.columns(3)
                for i, column in enumerate(columns):
                    with cols[i % 3]:
                        value = selected_row.iloc[0][column]
                        key = profile_key + '_' + column
                        if pd.api.types.is_numeric_dtype(reference[column]):
                            edited[column] = st.number_input(display_name(column), min_value=0.0,
                                value=float(value), step=1.0, key=key, help='Source column: ' + column)
                        else:
                            choices = sorted(reference[column].dropna().unique().tolist())
                            edited[column] = st.selectbox(display_name(column), choices, index=choices.index(value), key=key)
        st.caption('Satisfaction, involvement and balance: 1–4. Education and job level: 1–5. Stock options: 0–3. Career durations are in years.')
        submitted = st.form_submit_button('Estimate probability and explain', type='primary')
    if submitted:
        for column, value in edited.items():
            selected_row[column] = value
        try:
            with st.spinner('Calculating probability and explanations…'):
                report = make_report(model, selected_row, bundle['selected'])
            st.session_state['last_report'] = report
            st.session_state['last_profile_key'] = profile_key
            st.session_state['last_input'] = selected_row
        except ValueError as error:
            st.session_state.pop('last_report', None)
            st.error(str(error))
    if st.session_state.get('last_profile_key') == profile_key and 'last_report' in st.session_state:
        st.divider()
        st.caption('Results for the most recently submitted profile. Submit again after editing any input.')
        show_report(st.session_state['last_report'])
        with st.expander('Same employee: original versus domain features', expanded=True):
            row = st.session_state['last_input']
            comparison = compare_reports(bundle['models']['logistic_raw'], bundle['models']['logistic_no_log'], row)
            left, right = st.columns(2)
            for column, key, title in [(left, 'original', 'Original predictors'), (right, 'engineered', 'Original + seven domain features')]:
                r = comparison[key]
                with column:
                    st.markdown('**' + title + '**')
                    st.metric(title + ' probability', f"{r['probability']:.1%}")
                    top = pd.DataFrame(r['feature_explanation']['factors']).head(6)
                    st.dataframe(top[['display_name', 'contribution_calibrated_log_odds', 'is_engineered']].round(3), hide_index=True, width='stretch')
            st.caption(comparison['note'])
            domains = [f for f in comparison['engineered']['feature_explanation']['factors'] if f['is_engineered']]
            st.markdown('**Named domain features in this prediction**')
            for f in domains:
                st.write(f"{f['display_name']}: {f['direction']} the score by {f['contribution_calibrated_log_odds']:+.3f} log-odds · value {float(f['value']):.3f}")
                st.caption(f['definition'])
            st.download_button('Download side-by-side explanation', json.dumps(comparison, indent=2, allow_nan=False),
                file_name='attrition_comparison.json', mime='application/json')

with tab_experiment:
    st.subheader('Can domain-guided features improve both outcomes?')
    if (ROOT/'results/conclusion.json').exists():
        conclusion = json.loads((ROOT/'results/conclusion.json').read_text())
        st.write(conclusion['answer'])
        st.write(conclusion['comparison_statement'])
        st.write(conclusion['eight_feature_interpretability'])
    summary = pd.read_csv(ROOT/'results/nested_cv_summary.csv')
    st.write('Five repeats of five-fold stratified outer CV; five-fold tuning inside each training fold. All 1,470 employees are held out once per repeat. The 25 folds overlap in training data.')
    st.markdown('**Predictive comparison: mean and standard deviation across 25 outer folds**')
    core = summary[summary.feature_mode.isin(['raw', 'all', 'no_log'])]
    st.dataframe(core[['model', 'average_precision_mean', 'average_precision_std', 'roc_auc_mean', 'roc_auc_std', 'brier_mean', 'brier_std']].round(4), hide_index=True, width='stretch')
    st.caption('raw = 27 original predictors; all = original + 8; no_log = original + 7, excluding LogMonthlyIncome. AP/ROC-AUC higher is better; Brier lower is better. Final eligibility is raw/no_log, fixed before this run.')
    st.image(str(ROOT/'results/feature_comparison.png'))
    paired = pd.read_csv(ROOT/'results/paired_comparisons.csv')
    st.markdown('**Paired AP changes, candidate minus original**')
    st.dataframe(paired[(paired.metric=='average_precision') & paired.comparison.isin(['all','no_log'])][['candidate','difference','ci_low','ci_high','wins','folds','p_two_sided','p_holm_AP']].round(4), hide_index=True, width='stretch')
    st.caption('Approximate Nadeau–Bengio corrected 95% intervals account for overlapping training folds. Holm adjustment covers three model-family AP tests within each feature specification. These are exploratory comparisons on one previously examined dataset.')
    with st.expander('Block ablations and age sensitivity'):
        st.dataframe(summary[~summary.feature_mode.isin(['raw','all'])][['model','average_precision_mean','roc_auc_mean','brier_mean']].round(4), hide_index=True, width='stretch')
        st.dataframe(paired[paired.comparison.isin(['ablation','age']) & (paired.metric=='average_precision')][['candidate','difference','ci_low','ci_high']].round(4), hide_index=True, width='stretch')
        st.write('without_* removes an engineered block from the seven-feature version; every original predictor remains. Age is added only for a sensitivity analysis. Gender and marital status are excluded. Exclusion alone does not establish fairness.')
    st.dataframe(pd.DataFrame(FEATURE_GUIDE, columns=['Block','Feature','Formula','Rationale']), hide_index=True, width='stretch')
    st.caption('LogMonthlyIncome is retained only in the eight-feature research comparator, preventing duplicate MonthlyIncome/log-income attribution in the deployed engineered candidate.')

with tab_calibration:
    st.subheader('Do predicted probabilities match observed frequencies?')
    calibration = json.loads((ROOT/'results/calibration_summary.json').read_text())
    st.metric('Mean repeat ECE', f"{calibration['mean_repeat_ECE']:.4f}")
    st.caption(f"Across five repeats: SD {calibration['sd_repeat_ECE']:.4f}; ten equal-width probability bins. Lower ECE is better.")
    st.image(str(ROOT/'results/calibration_curve.png'))
    st.dataframe(pd.read_csv(ROOT/'results/reliability_pooled.csv').round(4), hide_index=True, width='stretch')
    st.write('Light lines show individual repeats. Bin counts pool 7,350 predictions of 1,470 distinct employees. ECE depends on binning and sample size. This evaluates the selected training procedure using held-out predictions; it does not guarantee calibration of the final full-data model.')
    st.write('Each sigmoid and decision threshold uses its outer-training fold only. The final model is refitted on all development data, with its threshold chosen from training CV scores. Its displayed threshold is not optimized on outer test labels.')

with tab_global:
    st.subheader('Explanation compactness, stability and ranking agreement')
    interpretation = pd.read_csv(ROOT/'results/interpretability_summary.csv')
    st.dataframe(interpretation[['model','metric','estimate','ci_low','ci_high']].round(4), hide_index=True, width='stretch')
    st.image(str(ROOT/'results/interpretability_comparison.png'))
    st.write('K80 is the number of attributes covering 80% of absolute contribution mass (lower is more compact). Jaccard measures top-three overlap after training bootstrap refits (higher is more stable). Each model covers every held-out employee and uses 50 refits per outer fold.')
    st.caption('Intervals bootstrap five repeat-level means and describe split variability conditional on this dataset. They are not population confidence intervals. Bootstrap refits keep tuned hyperparameters fixed. Forest TreeSHAP explains the base forest probability, before sigmoid calibration; logistic contributions explain calibrated log-odds.')
    st.write('Spearman compares mean absolute explanation ranks with held-out permutation AP-drop ranks. Agreement is a cross-check between different notions of importance, not proof of causality or human clarity. Independently permuting engineered columns can create unrealistic combinations.')
    importance = pd.read_csv(ROOT/'results/global_attributions.csv')
    choice = st.selectbox('Inspect a model', sorted(importance.model.unique()), index=sorted(importance.model.unique()).index('logistic_no_log'))
    imp = importance[importance.model==choice].sort_values('mean_absolute_contribution',ascending=False)
    st.dataframe(imp[['feature','mean_absolute_contribution','permutation_AP_drop']].head(15).round(4), hide_index=True, width='stretch')
    st.info('No human clarity study has been performed. docs/HUMAN_EVALUATION_RUBRIC.md provides a blinded 1–5 clarity/usefulness exercise for 5–10 classmates, with an empty response template.')

with tab_guide:
    st.subheader('Objective and implementation')
    st.markdown('''1. Estimate an individual attrition-label probability.
2. Apply a training-CV threshold to produce a class.
3. Explain signed group contributions and individual domain attributes.
4. Compare original, eight-feature and seven-feature variants on identical nested folds.
5. Report uncertainty, calibration, feature ablations and interpretation proxies.''')
    st.write('The upgraded study replaces the single 70/20/10 split. Earlier results are archived separately. Inner tuning, imputation, scaling, encoding and income reference medians are fitted using training data only. Selecting the winning family from outer-CV comparisons can still make its reported score optimistic; an external or temporal test is needed for confirmation.')
    st.info('Fictional IBM HR data with no dated turnover horizon. This coursework prototype is not validated for employment decisions. Model explanations are associations, not causes. No human understanding or fairness claim is established.')
    st.write('Read docs/PROJECT_REPORT.docx for the report, docs/METHODOLOGY.md for formulas, and docs/VIVA_GUIDE.md for the demonstration script.')
