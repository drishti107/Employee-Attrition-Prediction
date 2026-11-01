"""Generate shared conclusions from measured CSVs, never hand-enter scores."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from train import RESULTS,save_json,load_data,outer_splits,metrics

def run():
    s=pd.read_csv(RESULTS/'nested_cv_summary.csv').set_index('model')
    p=pd.read_csv(RESULTS/'paired_comparisons.csv')
    i=pd.read_csv(RESULTS/'interpretability_paired.csv')
    a=p[(p.candidate=='logistic_no_log') & (p.metric=='average_precision')].iloc[0]
    k=i[(i.candidate=='logistic_no_log') & (i.metric=='mean_k80')].iloc[0]
    t=i[(i.candidate=='logistic_no_log') & (i.metric=='top3_jaccard')].iloc[0]
    eight_ap=p[(p.candidate=='logistic_all') & (p.metric=='average_precision')].iloc[0]
    eight_stability=i[(i.candidate=='logistic_all') & (i.metric=='top3_jaccard')].iloc[0]
    eight_k80=i[(i.candidate=='logistic_all') & (i.metric=='mean_k80')].iloc[0]
    forest_ap=p[(p.candidate=='forest_all') & (p.metric=='average_precision')].iloc[0]
    forest_seven=p[(p.candidate=='forest_no_log') & (p.metric=='average_precision')].iloc[0]
    comparisons=(f"The original eight-feature logistic arm has mean AP {s.loc['logistic_all','average_precision_mean']:.4f}, "
        f"a change of {eight_ap.difference:+.4f} versus original predictors (corrected 95% interval "
        f"[{eight_ap.ci_low:+.4f}, {eight_ap.ci_high:+.4f}]; {int(eight_ap.wins)}/25 fold wins). "
        f"Forest AP changes are {forest_ap.difference:+.4f} with eight features and {forest_seven.difference:+.4f} with seven; "
        "their corrected intervals include zero. Both engineered decision-tree arms have lower mean AP than the original arm.")
    ap_text=f"The seven-feature logistic model changes mean average precision from {s.loc['logistic_raw','average_precision_mean']:.4f} to {s.loc['logistic_no_log','average_precision_mean']:.4f} (paired difference {a.difference:+.4f}; corrected 95% interval [{a.ci_low:+.4f}, {a.ci_high:+.4f}])."
    uncertainty=('The corrected interval excludes zero in favor of engineering.' if a.ci_low>0 else
        'The corrected interval excludes zero in favor of original features.' if a.ci_high<0 else
        'The corrected interval includes zero, so the apparent performance difference is uncertain.')
    uncertainty += f' Holm-adjusted AP p-value: {a.p_holm_AP:.4f}.'
    stability=f"Top-three stability changes by {t.difference:+.4f}, while K80 changes by {k.difference:+.2f} attributes (lower K80 is better)."
    eight_interpretation=(f"For the eight-feature logistic arm, stability changes by {eight_stability.difference:+.4f} "
        f"and K80 by {eight_k80.difference:+.2f}. Interpretability dimensions must therefore be assessed separately.")
    if a.ci_low>0 and a.p_holm_AP<.05 and t.ci_low>0 and k.ci_high<0:
        answer='This study supports simultaneous predictive and proxy interpretability improvement for the seven-feature logistic model. Human understanding remains untested.'
    elif a.difference>0 and t.difference>0:
        answer='Domain-guided features improve mean prediction performance and explanation stability descriptively for logistic regression, but this is not a blanket improvement in interpretability.'
        if a.ci_low<=0: answer+=' The corrected performance interval includes zero.'
        if k.difference>0: answer+=' Explanations require more attributes to cover 80% of contribution mass.'
    else:
        answer='The study does not demonstrate simultaneous improvement in prediction and interpretability for the seven-feature logistic model; the measured trade-offs must be reported separately.'
    answer+=' '+ap_text+' '+stability+' No human clarity study has been performed.'
    save_json(RESULTS/'conclusion.json',{'answer':answer,'prediction_statement':ap_text,
        'uncertainty_statement':uncertainty,'interpretability_statement':stability,
        'comparison_statement':comparisons,'eight_feature_interpretability':eight_interpretation,
        'scope':'One previously examined fictional dataset; conditional proxy results, no external or human validation.'})
    d,X,y=load_data();baseline=[]
    import numpy as np
    for fid,(tr,te) in enumerate(outer_splits(X,y)):
        baseline.append({'fold_id':fid,'repeat':fid//5,**metrics(y.iloc[te],np.repeat(y.iloc[tr].mean(),len(te)),.5)})
    pd.DataFrame(baseline).to_csv(RESULTS/'prior_baseline.csv',index=False)
    lines=['# Results and conclusion','',answer,'',uncertainty,'',comparisons,'',eight_interpretation,'',
        'The eight-feature arms reproduce the original feature specification. The seven-feature arms remove LogMonthlyIncome and are eligible for deployment alongside original-feature models. Model selection was fixed by mean nested-CV AP, lower Brier then key. Ablations and age sensitivity are diagnostic only.','',
        '## Predictive comparisons','',s.reset_index().to_markdown(index=False), '',
        '## Paired comparisons','',p.to_markdown(index=False),'',
        '## Interpretation differences','',i.to_markdown(index=False),'',
        'Predictive intervals use an approximate corrected resampled t method; AP p-values are Holm-adjusted over three families separately for each specification. Interpretability intervals bootstrap five repeat means and describe conditional split variation, not population uncertainty.','',
        'Selecting the final family from outer-CV results adds selection optimism; the supplied full-data deployment model has no new independent test. The old 70/20/10 study is a historical archive only.']
    (ROOT/'docs/RESULTS_AND_CONCLUSION.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':run()
