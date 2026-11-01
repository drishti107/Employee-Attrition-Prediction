"""Regenerate the upgraded report from saved measured results."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from explanations import FEATURE_GUIDE
OUT=ROOT/'docs';RESULTS=ROOT/'results'
S=pd.read_csv(RESULTS/'nested_cv_summary.csv').set_index('model')
P=pd.read_csv(RESULTS/'paired_comparisons.csv')
I=pd.read_csv(RESULTS/'interpretability_summary.csv')
IP=pd.read_csv(RESULTS/'interpretability_paired.csv')
C=json.loads((RESULTS/'calibration_summary.json').read_text())
E=json.loads((RESULTS/'example_comparison.json').read_text())
SEL=json.loads((RESULTS/'selection.json').read_text())
CON=json.loads((RESULTS/'conclusion.json').read_text())
selected=SEL['selected'];threshold=SEL['fit_info'][selected]['threshold']
def fmt(value):return f'{value:.4f}'
def interval(row,estimate='difference',digits=4):
    return f"{getattr(row,estimate):.{digits}f} [{row.ci_low:.{digits}f}, {row.ci_high:.{digits}f}]"
def metric_cell(key,metric):return f"{S.loc[key,metric+'_mean']:.4f} ± {S.loc[key,metric+'_std']:.4f}"
D=Document(); sec=D.sections[0]
sec.page_width=Inches(8.27);sec.page_height=Inches(11.69)
sec.top_margin=sec.bottom_margin=Inches(.65)
sec.left_margin=sec.right_margin=Inches(.7)
sec.header_distance=sec.footer_distance=Inches(.3)
styles=D.styles
for border in D.styles.element.xpath('.//w:pBdr'):
 border.getparent().remove(border)
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3']:
 st=styles[name];st.font.name='Calibri';st.font.color.rgb=RGBColor(0,0,0)
styles['Normal'].font.size=Pt(10.5)
styles['Normal'].paragraph_format.space_after=Pt(7)
styles['Normal'].paragraph_format.line_spacing=1.1
styles['Title'].font.size=Pt(24);styles['Title'].paragraph_format.space_after=Pt(8)
styles['Heading 1'].font.size=Pt(16);styles['Heading 1'].paragraph_format.space_after=Pt(10)
styles['Heading 2'].font.size=Pt(12);styles['Heading 2'].paragraph_format.space_before=Pt(10)
styles['Subtitle'].font.size=Pt(12)
# Page numbers are useful in this multi-page academic report.
f=sec.footer.paragraphs[0];f.alignment=WD_ALIGN_PARAGRAPH.RIGHT
r=f.add_run('Employee Attrition Mini Project  |  ');r.font.size=Pt(8);r.font.color.rgb=RGBColor.from_string('59636B')
fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');f._p.append(fld)
md=[]

def title(t):
 D.add_paragraph(t,'Title');md.append('# '+t+'\n')
def h(t,level=1):
 D.add_heading(t,level=level);md.append('#'*(level+1)+' '+t+'\n')
def p(t):
 D.add_paragraph(t);md.append(t+'\n')
def page(t):
 D.add_page_break();h(t)
def table(headers, rows, widths, fs=9.4):
 t=D.add_table(rows=1, cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
 for cell,w in zip(t.columns,widths):cell.width=Inches(w)
 for c,text in zip(t.rows[0].cells,headers):c.text=str(text)
 for row in rows:
  for c,text in zip(t.add_row().cells,row):c.text=str(text)
 for ri,row in enumerate(t.rows):
  for ci,cell in enumerate(row.cells):
   cell.width=Inches(widths[ci]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
   pr=cell._tc.get_or_add_tcPr()
   sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'223E50' if ri==0 else ('F0F4F6' if ri%2==0 else 'FFFFFF'));pr.append(sh)
   borders=OxmlElement('w:tcBorders')
   for side in ['top','left','bottom','right']:
    item=OxmlElement('w:'+side);item.set(qn('w:val'),'single');item.set(qn('w:sz'),'4');item.set(qn('w:color'),'D9D9D9');borders.append(item)
   pr.append(borders)
   margins=OxmlElement('w:tcMar')
   for side in ['top','bottom','left','right']:
    item=OxmlElement('w:'+side);item.set(qn('w:w'),'85');item.set(qn('w:type'),'dxa');margins.append(item)
   pr.append(margins)
   for para in cell.paragraphs:
    para.paragraph_format.space_after=Pt(2);para.paragraph_format.space_before=Pt(2);para.paragraph_format.line_spacing=1.03
    if ci>0 and len(str(cell.text))<14:para.alignment=WD_ALIGN_PARAGRAPH.CENTER
    for run in para.runs:
     run.font.name='Calibri';run.font.size=Pt(fs);run.bold=ri==0
     run.font.color.rgb=RGBColor.from_string('FFFFFF' if ri==0 else '000000')
  trpr=row._tr.get_or_add_trPr();trpr.append(OxmlElement('w:cantSplit'))
 t.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
 D.add_paragraph().paragraph_format.space_after=Pt(2)
 md.append('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(str(x).replace('\n',' ') for x in row)+' |' for row in rows)+'\n')
def picture(name,width=6.7):
 D.add_picture(str(RESULTS/name),width=Inches(width));md.append(f'![{name}](../results/{name})\n')
def caption(text):
 q=D.add_paragraph(text);q.paragraph_format.space_after=Pt(8)
 for r in q.runs:r.font.size=Pt(9);r.italic=True
 md.append(text+'\n')

title('Employee Attrition Prediction with Domain Guided Features')
D.add_paragraph('Upgraded mini project with repeated nested validation','Subtitle')
p('Research question: Can domain-guided feature engineering simultaneously improve predictive performance and interpretability?')
h('Abstract',2)
p('This system predicts the fictional IBM HR dataset’s attrition label and explains the major factors associated with an individual prediction. The upgraded experiment uses all 1,470 employees in five repeats of five-fold nested stratified cross-validation. Logistic regression, decision trees and random forests are compared under original, eight-feature and seven-feature representations. Four block ablations, an age sensitivity analysis, sigmoid probability calibration and training-CV threshold selection are included. Explanation diagnostics cover every held-out employee, with 50 bootstrap refits per outer fold, and include random forests.')
p(CON['answer'])
h('Objective to evidence mapping',2)
table(['Professor’s objective','Implemented evidence'],[
 ['Estimate individual turnover probability','Calibrated probability plus a training-CV thresholded class'],
 ['Identify principal associated factors','Signed group contributions and named domain attributes'],
 ['Investigate feature engineering','Matched nested folds, original +8 and original +7 variants, ablations'],
 ['Evaluate interpretability with performance','K80, top-three stability, ranking agreement and uncertainty']],[2.35,4.52],9.7)
p(f"The final deployment model is {selected}, refitted on all development records. Its threshold is {threshold:.3f}. Reported outer-CV scores evaluate the training procedure; they are not an independent test of this final refit.")
caption('Educational scope: fictional, cross-sectional data with no defined forecast horizon. No human clarity study has been performed.')

page('Data and experimental design')
p('The source dataset contains 1,470 records and 35 columns, including Attrition. There are 237 Yes labels (16.12%) and 1,233 No labels. The source describes the data as fictional [1]. The supplied CSV is fingerprinted; missing cells and employee-ID duplicates are checked before training. No dated departure or observation window is provided.')
h('Predictor policy',2)
p('The main models use 27 original predictors. EmployeeNumber is an identifier; EmployeeCount, Over18 and StandardHours are constant. These are removed. Age, Gender and MaritalStatus are excluded from the principal study. Age is added only in a separate sensitivity analysis. Excluding demographics does not establish fairness because other inputs may proxy them.')
h('Matched repeated nested cross-validation',2)
table(['Stage','Design','Role'],[
 ['Outer evaluation','5 repeats × 5 stratified folds','25 matched test folds; each has 294 test and 1,176 training rows'],
 ['Inner tuning','5 stratified training folds','Select one of 8 parameter settings per family by mean AP'],
 ['Calibration','5-fold training OOF scores','Fit a sigmoid without outer-test labels'],
 ['Threshold','Training OOF probabilities','Maximize F1 on the fixed 0.05–0.80 grid'],
 ['Deployment','Refit on all 1,470 records','Selected raw/seven-feature family; no independent final test']],[1.35,1.95,3.57],9.5)
p('Every employee is held out exactly once in each repeat: 7,350 held-out predictions per specification, not 7,350 independent people. The same outer folds are used across all 14 specifications. All learned preprocessing and role-income references are refitted inside training folds [2].')
h('What changed from the original project',2)
p('The 70/20/10 split is replaced as the primary evaluation. Its results are preserved only in a historical archive. The upgraded protocol was fixed before this run, but the dataset and earlier results had already been examined. This is therefore an exploratory extension, not prospective preregistration or an independent replication.')
p('The final family is selected from outer-CV comparisons. Nested CV protects each candidate’s tuning/evaluation separation; selecting the best candidate across those results can still make the winning score optimistic. A new external or temporal test is needed for confirmation.')

page('Domain guided representations')
p('The original representation is compared with original + eight domain features (“all”). A second version (“no_log”) removes LogMonthlyIncome, retaining seven features to avoid splitting attribution between raw income and its log transform. All three versions are run for every model family; only original and seven-feature candidates are deployment-eligible, as fixed before training.')
table(['Feature','Definition','Block'],[[name,formula,block] for block,name,formula,rationale in FEATURE_GUIDE],[2.18,3.69,1.0],9.1)
p('IncomeToRoleMedian uses only the training fold’s job-role medians, with its global income median as a fallback for unseen roles. Career denominators add one to handle zero tenure. The sentiment average is a linear combination of existing survey ratings, so it does not create independent information. Distance units and rate-field meanings are not invented.')
h('Domain rationale and interpretability',2)
p('Interactions make overtime-plus-balance or overtime-plus-commute relationships explicit to a linear model. Ratios describe career time and pay relative to a reference. These names can aid explanation, but are hypotheses rather than established causes. A high role-tenure share need not mean stagnation, and overtime pressure is not a burnout diagnosis.')
p('Adding correlated features can redistribute credit and increase explanation length. Retaining MonthlyIncome with IncomeToRoleMedian also leaves shared information, because the ratio provides role context. Removing the log-income duplicate reduces one redundant representation; it does not make all inputs independent.')
p('Four logistic ablations remove workload, career, reward or sentiment engineered blocks from the seven-feature version while retaining every original predictor. They are evaluated on the same nested folds and do not enter final selection.')

page('Models calibration and uncertainty')
table(['Family','Unchanged tuning grid'],[
 ['Logistic regression','C = 0.05, 0.2, 1, 5; class weight = none or balanced'],
 ['Decision tree','Depth = 3 or 5; minimum leaf = 10 or 25; class weight = none or balanced'],
 ['Random forest','150 trees; depth = 5 or unrestricted; minimum leaf = 3 or 10; class weight = none or balanced']],[1.55,5.32],9.8)
p('Numeric inputs use median imputation and standardization; categorical inputs use most-frequent imputation and one-hot encoding. Ordinal ratings are treated as equally spaced. Pipelines prevent outer-test data from fitting preprocessing statistics. Tuning compares equal eight-setting budgets, not identical fitted hyperparameters.')
h('Probabilities and threshold',2)
p('After inner tuning, five-fold out-of-fold scores from the chosen pipeline fit a sigmoid calibrator. The base pipeline is refitted on the whole outer-training fold. Logistic scores are decision-function values; tree and forest scores are clipped probability logits. Tuning and calibration reuse outer-training data, so internal calibration/threshold development is not fully nested. The complete procedure is scored only on its untouched outer-test fold [3].')
p('Thresholds maximize F1 on calibrated training OOF scores over 0.05–0.80 at 0.005 steps, with higher thresholds breaking ties. This can overfit the development scores; outer evaluation measures the resulting procedure. No 294-row validation pool or outer-test threshold tuning is used.')
h('Metrics and paired intervals',2)
p('Average precision (AP) is primary because attrition is the minority class; it is not trapezoidal PR-AUC. ROC-AUC assesses ranking. Brier assesses probability error; it is not a pure calibration measure. ECE and reliability curves assess calibration descriptively. Precision, recall, F1 and accuracy use each training-selected threshold.')
p('For 25 matched differences d, corrected SE = sqrt((1/25 + 294/1176) × sample variance(d)). Use Student t with 24 degrees of freedom for approximate 95% intervals and two-sided p-values [4,5]. The correction reflects overlapping training folds. Holm-adjust AP p-values over the three families separately for the eight- and seven-feature comparisons. Ablations, age and other metrics are exploratory.')
p('Report fold means and sample SDs, paired changes, intervals and win counts. Do not interpret the 25 overlapping folds as independent experiments or an interval as a probability that the hypothesis is true.')

page('Predictive results')
keys=[f'{family}_{mode}' for family in ['logistic','tree','forest'] for mode in ['raw','all','no_log']]
labels={f'{family}_{mode}':f"{family.title()} {'original' if mode=='raw' else '+8' if mode=='all' else '+7'}" for family in ['logistic','tree','forest'] for mode in ['raw','all','no_log']}
table(['Model','AP ↑ mean ± SD','ROC-AUC ↑','Brier ↓'],[[labels[k],metric_cell(k,'average_precision'),metric_cell(k,'roc_auc'),metric_cell(k,'brier')] for k in keys],[1.55,1.78,1.78,1.76],9.2)
picture('feature_comparison.png')
caption('Figure 1. Matched repeated nested-CV average precision. “+8” is the original engineered design; “+7” removes LogMonthlyIncome.')
p(f"Selected among raw/+7 candidates: {selected}. Mean outer precision {S.loc[selected,'precision_mean']:.4f}, recall {S.loc[selected,'recall_mean']:.4f}, F1 {S.loc[selected,'f1_mean']:.4f}, and accuracy {S.loc[selected,'accuracy_mean']:.4f}. These means use fold-specific training-selected thresholds; the final full-data threshold is {threshold:.3f}.")
p('The positive prevalence is 16.12%, illustrating why accuracy alone is insufficient. A training-prior baseline is included separately in results/prior_baseline.csv. The main tables and app are generated from the same saved outer predictions.')

page('Paired changes and ablations')
p('Each value below is engineered minus original AP on the same 25 outer folds. A positive difference favors engineering. An interval crossing zero indicates unresolved direction under the corrected approximation; a win count alone is not a significance test.')
ap=P[(P.metric=='average_precision') & P.comparison.isin(['all','no_log'])]
table(['Candidate','AP change [95% interval]','Wins /25','Holm p'],[[labels[r.candidate],interval(r),str(r.wins),fmt(r.p_holm_AP)] for r in ap.itertuples()],[1.55,3.12,.9,1.3],9.3)
p('Seven-feature logistic versus original: '+CON['uncertainty_statement'])
p(CON['comparison_statement'])
h('Remove one engineered block from seven features',2)
abl=P[(P.metric=='average_precision') & (P.comparison=='ablation')]
table(['Removed block','Mean AP','AP change vs +7 [95% interval]'],[[r.candidate.replace('logistic_without_','').title(),fmt(S.loc[r.candidate,'average_precision_mean']),interval(r)] for r in abl.itertuples()],[1.4,1.1,4.37],9.4)
p('Each ablation is retuned within each outer-training fold. Original predictors remain. A negative change suggests that the removed block helped the tuned procedure, but these exploratory intervals are unadjusted and can overlap zero.')
h('Age sensitivity and fairness scope',2)
age=P[(P.metric=='average_precision') & (P.comparison=='age')].iloc[0]
p(f"Adding Age to seven-feature logistic gives mean AP {S.loc['logistic_no_log_age','average_precision_mean']:.4f}; change {age.difference:+.4f}, corrected interval [{age.ci_low:+.4f}, {age.ci_high:+.4f}]. This is an information sensitivity check, not a fairness test. Age remains excluded from the deployed main model, along with Gender and MaritalStatus.")
p('The full paired table also reports ROC-AUC and Brier changes. Its Brier direction is reversed: lower is better. Differences between engineered variants must not be cherry-picked to imply a universal benefit.')

page('Interpretability results')
p('All six logistic and forest representations cover 1,470 employees, each held out once per repeat. Fifty stratified bootstrap refits per outer fold, with tuned parameters fixed, yield 7,500 refits.')
rows=[]
for key in [f'{f}_{m}' for f in ['logistic','forest'] for m in ['raw','all','no_log']]:
 r=I[I.model==key].set_index('metric'); rows.append([labels[key],interval(r.loc['mean_k80'],'estimate',2),interval(r.loc['top3_jaccard'],'estimate',3)])
table(['Representation','Mean K80 ↓ [95% interval]','Top-3 Jaccard ↑ [95% interval]'],rows,[1.55,2.66,2.66],9.2)
p('K80 counts attributes covering 80% of absolute contribution mass. Jaccard measures top-three set overlap after bootstrap refitting. One-hot levels are grouped by source attribute. These proxies measure compactness and stability, not human comprehension.')
seven_changes=IP[IP.candidate=='logistic_no_log'].set_index('metric')
p(f"Logistic +7: stability change {seven_changes.loc['top3_jaccard','difference']:+.4f}; K80 change {seven_changes.loc['mean_k80','difference']:+.2f} attributes (lower is better).")
p(CON['eight_feature_interpretability'])
h('Intervals and random-forest coverage',2)
p('Intervals use 10,000 bootstrap draws of five repeat means, matched for paired changes. They describe conditional split variability, not population uncertainty: repeats share data and only five means are available. Adding attributes can mechanically increase K80.')
p('Logistic contributions use calibrated log-odds. Forest TreeSHAP uses uncalibrated class-1 probability with a tree-path-dependent background [6]. Their magnitudes have different units. The app separately explains calibrated probabilities through six groups.')
h('Ranking agreement and fidelity',2)
rho=I[I.metric=='spearman']
table(['Model','Spearman vs permutation ranks [95% interval]'],[[labels[r.model],interval(r,'estimate',3)] for r in rho.itertuples()],[2.0,4.87],9.1)
p('Five-shuffle permutation AP drops cross-check representation reliance [7]. Independent derived-column permutations can create inconsistent combinations. Spearman agreement does not establish causality or human clarity; additivity is checked separately. No human study was performed.')

page('Probability calibration')
p(f"For the selected {selected} procedure, ten-bin ECE averages {C['mean_repeat_ECE']:.4f} across five repeats (SD {C['sd_repeat_ECE']:.4f}). Mean outer Brier is {S.loc[selected,'brier_mean']:.4f} (SD {S.loc[selected,'brier_std']:.4f}). Lower is better for both metrics.")
picture('calibration_curve.png',6.1)
caption('Figure 2. Reliability curves from held-out predictions. Pale lines show individual repeats; the stronger line pools predictions descriptively. Perfect calibration follows the diagonal.')
table(['Repeat','ECE'],[[str(i+1),fmt(v)] for i,v in enumerate(C['repeat_ECE'])],[2,4.87],9.6)
p('ECE is the weighted absolute difference between mean prediction and observed positive rate within ten equal-width probability bins. Empty bins are omitted. Values depend on binning and finite sample size, particularly in high-probability bins with few employees. Brier also reflects discrimination and uncertainty, so neither metric alone proves calibrated individual risk.')
p('The plot pools 7,350 held-out predictions from 1,470 employees. Its bin counts are repeated observations, not independent employees. These curves evaluate the selected training procedure; they are not an external calibration guarantee for the final model refitted on all records.')

page('Individual prediction and group explanation')
raw=E['original'];eng=E['engineered']
p(f"The same fictional employee, ID 478, is evaluated by original and seven-feature logistic models from outer fold {E['fold_id']+1} of the first repeat. This employee is excluded from both models’ training and reference data. It was already used as a high-risk illustration in the original project, so it is not a representative or randomly selected case.")
table(['Model','Attrition probability','Threshold','Class'],[['Original',f"{raw['probability']:.2%}",f"{raw['threshold']:.3f}",raw['predicted_label']],['Original +7',f"{eng['probability']:.2%}",f"{eng['threshold']:.3f}",eng['predicted_label']]],[1.8,1.9,1.37,1.8],10)
h('Six concept groups in probability units',2)
p('Enumerate all 64 combinations of six raw-input groups against 32 fixed training-background records, recomputing engineered features for every hybrid. Exact Shapley weights allocate the prediction difference to workload, satisfaction, career, pay, education and role groups. Each contribution is in probability percentage points.')
picture('example_factors.png')
caption(f"Figure 3. +7 model: reference {eng['reference_probability']:.2%}, prediction {eng['probability']:.2%}. Rounded values are shown; the unrounded contributions reconstruct the probability exactly.")
p('These group explanations describe model associations relative to a reference, not confirmed causes of leaving. Combining correlated groups from different employees can create unrealistic profiles. Changing a displayed factor does not establish a real-world treatment effect.')
p('The app uses final models refitted on all development records. Its probabilities for this employee can differ from the held-out illustration, and in-sample demo profiles are not evaluation evidence.')

page('Same employee and named domain attributes')
p('The detailed logistic explanation uses a training-mean baseline. For transformed attribute j, contribution = a × beta_j × (z_j − mean_training_j). Baseline plus all contributions, followed by the sigmoid, exactly reconstructs the calibrated probability. One-hot categories are combined into the original attribute.')
top=[]
for title_,report in [('Original',raw),('Original +7',eng)]:
 for f in report['feature_explanation']['factors'][:3]:top.append([title_,f['display_name'],f"{f['contribution_calibrated_log_odds']:+.3f}"])
table(['Model','Top attribute','Log-odds'],top,[1.2,4.37,1.3],9.5)
h('Engineered factors in the seven feature explanation',2)
domains=[f for f in eng['feature_explanation']['factors'] if f['is_engineered']]
table(['Named domain factor','Value','Log-odds'],[[f['display_name'],f"{float(f['value']):.3f}",f"{f['contribution_calibrated_log_odds']:+.3f}"] for f in domains],[4.17,1.35,1.35],9.3)
for f in domains[:2]:p(f"{f['display_name']}: {f['definition']}. For this employee it {f['direction']} the calibrated model score by {abs(f['contribution_calibrated_log_odds']):.3f} log-odds relative to the transformed training mean.")
p('A positive sign raises the score and a negative sign lowers it. Log-odds are not percentage points and cannot be added directly to the probability. The feature baseline differs from the six-group background-probability reference; each decomposition is internally additive. The app displays every domain definition next to its contribution.')
p('The seven-feature version contains MonthlyIncome once as its raw scale, plus the role-relative ratio. LogMonthlyIncome remains only in the original eight-feature comparator. Correlation can still distribute contribution among pay, role and tenure inputs.')

page('Implementation verification and demonstration')
table(['Component','Purpose'],[['train.py','25 outer folds, inner tuning, ablations, CV thresholds, final refit'],['attrition.py / explanations.py','Fold-local features, probabilities, validation, additive explanations'],['interpretability.py','All-held-out K80, 50 bootstrap refits per fold, TreeSHAP and rank checks'],['app.py / predict.py','Interactive and CLI prediction, side-by-side explanations and export'],['notebooks / scripts','Executed walkthrough; shared data-driven charts and report'],['verify.py / tests','Fold coverage, metrics, medians, calibration, explanations and UI']],[2.1,4.77],9.5)
h('Run the supplied system',2)
p('Extract the entire ZIP, create a Python 3.12 environment, then run: pip install -r requirements.txt; python verify.py; python -m streamlit run app.py. The trained bundle is included, so full retraining is not needed for a classroom demonstration. Complete Windows commands are in docs/RUN_WINDOWS.md.')
p('For full reproduction, first install requirements-report.txt, then run python reproduce.py --destination ../Attrition_Reproduced --jobs 6. This recreates 350 outer model fits and 7,500 bootstrap refits in a new directory. Runtime depends on the machine. The supplied notebook explores and verifies saved results without repeating this expensive stage.')
h('Demonstration sequence',2)
p('Select a fictional employee, estimate its probability, and explain the thresholded class. Read one increasing and one decreasing factor with the correct units. Compare original versus engineered explanations for that same profile. Show nested-CV AP intervals, block ablations, calibration and interpretation diagnostics. End with the measured research answer and its limits.')
h('Verification evidence',2)
p('Checks recompute saved fold metrics from all outer predictions, verify matched fold coverage and training-only income medians, check final model selection, reproduce calibration bins, inspect bootstrap counts and interval calculations, and test explanation additivity. UI tests cover valid prediction, invalid input rejection, recovery and changing profiles. The notebook is executed. Verification output is supplied in results; runtime checks were performed on Linux. verify.py was also run on Windows 11 in the pinned Python 3.13 environment (see docs/WINDOWS_VERIFICATION.md).')
h('Optional human evaluation',2)
p('The supplied rubric specifies a blinded, counterbalanced 1–5 clarity/usefulness exercise with 5–10 classmates, five fixed fictional profiles, and comprehension questions. The response CSV is empty. No responses were collected or fabricated; numerical proxies cannot substitute for that study.')

page('Gap closure: locked holdout, subgroup audit, human-study readiness')
HO=RESULTS/'holdout'; FA=RESULTS/'fairness'; HS=RESULTS/'human_study'
p('This section was added after the first review. It reports only what was run. Items that need data or people that the project does not have are marked as not performed.')
if (HO/'holdout_test_metrics.csv').exists():
    ht=pd.read_csv(HO/'holdout_test_metrics.csv');hp=json.loads((HO/'holdout_paired_ap.json').read_text());hs=json.loads((HO/'holdout_selection.json').read_text())
    h('Locked holdout test (independent of selection)',2)
    p(f"A fixed stratified split (seed 20260101) set aside 294 employees before any model comparison. Selection used only the remaining 1,176 development rows, with the same nested-CV, calibration, threshold and winner rules. The selected model was {hs['selected']}. It was refitted on all development rows and scored once on the holdout. This removes selection optimism from the holdout number. It does not make the test external: it is still the same fictional dataset.")
    table(['Model','Holdout AP','ROC-AUC','Brier','F1 (dev threshold)','Selected'],
        [[r.model,f'{r.average_precision:.4f}',f'{r.roc_auc:.4f}',f'{r.brier:.4f}',f'{r.f1:.4f}','yes' if r.selected else ''] for r in ht.itertuples()],
        [1.5,0.95,0.95,0.95,1.35,0.9])
    lo,hi=hp['bootstrap_95_percentile_interval']
    p(f"Holdout AP difference, selected minus logistic_raw: {hp['difference_observed']:+.4f}, bootstrap 95% percentile interval [{lo:+.4f}, {hi:+.4f}] over 294 employees. The holdout is 20% of the data, so this interval is wide and should be read descriptively.")
else:
    p('Locked holdout: not run.')
if (FA/'fairness_gap_summary.json').exists():
    fg=json.loads((FA/'fairness_gap_summary.json').read_text())
    h('Subgroup audit: Gender, marital status, age band',2)
    p('The saved held-out predictions for every employee were grouped by attributes that the model excluded. Gaps are the largest minus smallest group value. Intervals resample employees, keeping each employee\'s five held-out predictions together.')
    rows=[]
    m=fg['models']['logistic_no_log']
    for attr in ['Gender','MaritalStatus','AgeBand']:
        g=m['point_gaps'][attr]; b=m['bootstrap_95_percentile'][attr]
        rows.append([attr,f"{g['flagged_rate_gap']:.3f} {b['flagged_rate_gap'][0]:.3f} to {b['flagged_rate_gap'][1]:.3f}",
            f"{g['true_positive_rate_gap']:.3f} {b['true_positive_rate_gap'][0]:.3f} to {b['true_positive_rate_gap'][1]:.3f}",
            f"{g['false_positive_rate_gap']:.3f} {b['false_positive_rate_gap'][0]:.3f} to {b['false_positive_rate_gap'][1]:.3f}"])
    table(['Attribute','Flagged-rate gap [95%]','True-positive gap [95%]','False-positive gap [95%]'],rows,[1.4,1.9,1.9,1.9])
    p('Groups are descriptive. Some subgroups are small, so intervals are wide. The audit does not establish fairness: excluded attributes can still be proxied by other inputs.')
else:
    p('Subgroup audit: not run.')
h('Human clarity study: materials ready, no ratings',2)
status=json.loads((HS/'status.json').read_text())['status'] if (HS/'status.json').exists() else 'not performed'
p(f"Status: {status}. The packet (ten participant cards, counterbalanced A/B assignment, blinded key and analysis script) is built from the rubric. No ratings were collected or created. Interpretability on this page is measured by K80 and stability proxies only, until the study is run.")
h('Temporal validation',2)
p('Not performed. The dataset has no observation or departure dates. scripts/temporal_validation.py is ready for a dated extract with a stated horizon, but no such extract is in this project.')

page('Conclusion limitations and references')
p(CON['answer'])
h('What this project establishes',2)
p('The system meets the original objective by producing an attrition-label probability, a class, and an interpretable individual explanation, while testing domain engineering in controlled model comparisons. Findings are model-specific and measure-specific. A mixed result is a valid research conclusion, not a reason to replace measured results with the hoped-for answer.')
h('Limits and next steps',2)
p('The data are fictional and cross-sectional; feature timing relative to attrition is unknown. There is no prospective turnover horizon or external test; the locked holdout is an internal split of the same fictional data. No human comprehension study has been run. The subgroup audit is descriptive and reports gaps that were not removed. The dataset was already examined before this upgrade. Outer folds overlap; corrected t inference is approximate and interpretation intervals use only five repeat means. Tuning and calibration share development data; choosing a winner across outer comparisons adds selection optimism.')
p('Correlated or redundant attributes can share explanation credit, and intervention-style hybrids may be implausible. Stability conditions on selected parameters, not all model-selection uncertainty. TreeSHAP diagnostics explain the base forest, not its calibrated probability. A real forecasting study needs dated longitudinal data, temporal validation, a specified horizon, external testing, calibration monitoring and carefully designed human evaluation. This prototype is for coursework.')
h('References',2)
refs=[
 '[1] IBM HR Analytics Employee Attrition and Performance. Kaggle. https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset',
 '[2] scikit-learn. Nested versus non-nested cross-validation. https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html',
 '[3] scikit-learn. Probability calibration. https://scikit-learn.org/stable/modules/calibration.html',
 '[4] scikit-learn. Statistical comparison of models using grid search. https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html',
 '[5] Nadeau, C. and Bengio, Y. Inference for the Generalization Error. NeurIPS, 1999. https://papers.nips.cc/paper/1999/file/7d12b66d3df6af8d429c1a357d8b9e1a-Paper.pdf',
 '[6] SHAP. TreeExplainer documentation. https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html',
 '[7] scikit-learn. Permutation feature importance. https://scikit-learn.org/stable/modules/permutation_importance.html']
for ref in refs:
 q=D.add_paragraph(ref)
 for run in q.runs:run.font.size=Pt(8.7)
 md.append(ref+'\n')
D.core_properties.title='Employee Attrition Prediction — Upgraded Nested-CV Study'
D.core_properties.subject='Probability prediction and interpretable domain feature comparison'
D.core_properties.author=''
D.save(OUT/'PROJECT_REPORT.docx')
(OUT/'PROJECT_REPORT.md').write_text('\n'.join(md),encoding='utf-8')
print('Created upgraded PROJECT_REPORT.docx and PROJECT_REPORT.md')
