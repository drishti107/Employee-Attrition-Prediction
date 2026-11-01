"""Build shared charts and a held-out illustrative comparison from saved fits."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import json, joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from train import load_data, RESULTS, CACHE, save_json
from explanations import make_report, compare_reports

def run():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    table=pd.read_csv(RESULTS/'nested_cv_summary.csv')
    fig,ax=plt.subplots(figsize=(9,4.4))
    families=['logistic','tree','forest']; y=np.arange(3)
    for offset,mode,label,color in [(-.25,'raw','Original','#536978'),(0,'all','+ 8 features','#CA793F'),(.25,'no_log','+ 7 features','#177B7A')]:
        sub=table.set_index('model').loc[[f+'_'+mode for f in families]]
        ax.barh(y+offset,sub.average_precision_mean,height=.22,color=color,label=label)
    ax.set_yticks(y,['Logistic regression','Decision tree','Random forest']);ax.invert_yaxis()
    ax.set_xlabel('Mean outer-fold average precision (higher is better)');ax.set_xlim(0,.8)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=3,frameon=False)
    fig.tight_layout();fig.savefig(RESULTS/'feature_comparison.png',dpi=180);plt.close(fig)
    cal=json.loads((RESULTS/'calibration_summary.json').read_text())
    pooled=pd.read_csv(RESULTS/'reliability_pooled.csv'); repeats=pd.read_csv(RESULTS/'reliability_by_repeat.csv')
    fig,ax=plt.subplots(figsize=(6.5,4.6));ax.plot([0,1],[0,1],'--',c='#86949D',label='Perfect calibration')
    for _,r in repeats.groupby('repeat'):ax.plot(r.mean_probability,r.observed_rate,c='#A5CFCD',alpha=.6,lw=1)
    ax.plot(pooled.mean_probability,pooled.observed_rate,'o-',c='#177B7A',lw=2,label='Pooled held-out estimates')
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed attrition proportion')
    ax.set_title(f"{cal['model']} · mean repeat ECE {cal['mean_repeat_ECE']:.3f}")
    ax.legend(loc='upper left',frameon=False,fontsize=8);fig.tight_layout()
    fig.savefig(RESULTS/'calibration_curve.png',dpi=180);plt.close(fig)
    d,X,y=load_data(); index=int(d.index[d.EmployeeNumber==478][0])
    manifest=json.loads((RESULTS/'split_manifest.json').read_text())
    fold=next(f['fold_id'] for f in manifest[:5] if index in f['test_indices'])
    models={k:joblib.load(CACHE/f'{fold:02d}_{k}.joblib')['model'] for k in ['logistic_raw','logistic_no_log','logistic_all']}
    row=X.iloc[[index]]; comparison=compare_reports(models['logistic_raw'],models['logistic_no_log'],row)
    comparison.update({'employee_id':478,'row_index':index,'fold_id':fold,'repeat':0,
        'sampling_note':'Pre-existing high-risk illustration reused; not a representative or randomly selected employee.',
        'evaluation_note':'Held out of both example models; separate from the final all-data deployment model.'})
    save_json(RESULTS/'example_comparison.json',comparison)
    joblib.dump({'models':models,'row':row,'fold_id':fold,'comparison':comparison},ROOT/'models/example_fold.joblib',compress=3)
    bundle=joblib.load(ROOT/'models/bundle.joblib'); final=bundle['models'][bundle['selected']]
    save_json(RESULTS/'example_deployment.json',make_report(final,row,bundle['selected']))
    factors=pd.DataFrame(comparison['engineered']['factors']).sort_values('contribution_probability_points')
    fig,ax=plt.subplots(figsize=(8.5,3.9));v=factors.contribution_probability_points
    ax.barh(factors.factor_group,v,color=['#CA793F' if t>0 else '#177B7A' for t in v]);ax.axvline(0,c='#536978',lw=.8)
    ax.set_xlabel('Contribution to calibrated probability (percentage points)')
    ax.set_title('Employee 478 · held-out seven-feature logistic model')
    fig.tight_layout();fig.savefig(RESULTS/'example_factors.png',dpi=180);plt.close(fig)
    if (RESULTS/'interpretability_summary.csv').exists():
        frame=pd.read_csv(RESULTS/'interpretability_summary.csv')
        fig,axes=plt.subplots(1,2,figsize=(9.5,4.2))
        keys=['logistic_raw','logistic_all','logistic_no_log','forest_raw','forest_all','forest_no_log']
        labels=['Logistic original','Logistic +8','Logistic +7','Forest original','Forest +8','Forest +7']
        for ax,metric,title in zip(axes,['mean_k80','top3_jaccard'],['Compactness · K80 ↓','Top-three stability · Jaccard ↑']):
            r=frame[frame.metric==metric].set_index('model').loc[keys]
            ax.errorbar(r.estimate,np.arange(6),xerr=np.array([r.estimate-r.ci_low,r.ci_high-r.estimate]),fmt='o',c='#177B7A',capsize=3)
            ax.set_yticks(np.arange(6),labels);ax.invert_yaxis();ax.set_title(title)
        fig.tight_layout();fig.savefig(RESULTS/'interpretability_comparison.png',dpi=180);plt.close(fig)

if __name__=='__main__':run()
