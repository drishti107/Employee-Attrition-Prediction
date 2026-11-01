"""Held-out explanation diagnostics; python interpretability.py --jobs 6.

All 1,470 employees are evaluated once in each of five repeats. Each outer
training fold is bootstrapped 50 times, with selected hyperparameters fixed.
TreeSHAP here explains BASE forest probability, before sigmoid calibration.
"""
import argparse, hashlib, json, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from attrition import group_encoded_contributions, linear_contributions
from train import ROOT, CACHE, RESULTS, load_data, save_json, fingerprint

ICACHE=ROOT/'.cache/interpretability'
KEYS=[f'{family}_{mode}' for family in ['logistic','forest'] for mode in ['raw','all','no_log']]

def transformed(pipeline,X):
    return pipeline.named_steps['prep'].transform(pipeline.named_steps['domain'].transform(X))

def attributions(pipeline,X,reference):
    prep=pipeline.named_steps['prep']; clf=pipeline.named_steps['model']
    Z=transformed(pipeline,X)
    if isinstance(clf,LogisticRegression):
        mean=transformed(pipeline,reference).mean(axis=0)
        C=(Z-mean)*clf.coef_[0]
        base=float(clf.intercept_[0]+mean@clf.coef_[0])
        error=float(np.max(abs(base+C.sum(axis=1)-pipeline.decision_function(X))))
    else:
        import shap
        explainer=shap.TreeExplainer(clf,feature_perturbation='tree_path_dependent',model_output='raw')
        C=explainer.shap_values(Z,check_additivity=True)[:,:,1]
        base=float(explainer.expected_value[1])
        error=float(np.max(abs(base+C.sum(axis=1)-pipeline.predict_proba(X)[:,1])))
    assert error<1e-7, error
    return group_encoded_contributions(prep,C,index=X.index),error

def k80_top3(C):
    arr=np.abs(C.to_numpy()); ordered=np.sort(arr,axis=1)[:,::-1]
    total=ordered.sum(axis=1)
    k80=(np.cumsum(ordered,axis=1)<.8*total[:,None]).sum(axis=1)+1
    k80[total==0]=0
    top=[set(C.columns[np.argsort(-r,kind='stable')[:3]]) if r.sum() else set() for r in arr]
    return k80,top

def transformed_permutation(pipeline,X,y,repeats=5,seed=42):
    """Grouped one-hot permutation; derived columns are independently perturbed.
    This tests representation reliance, NOT realistic interventions on employees.
    """
    prep=pipeline.named_steps['prep']; clf=pipeline.named_steps['model']; Z=transformed(pipeline,X)
    numeric=list(prep.transformers_[0][2]); categorical=list(prep.transformers_[1][2])
    categories=prep.named_transformers_['cat'].named_steps['encode'].categories_
    groups={name:[j] for j,name in enumerate(numeric)}; offset=len(numeric)
    for name,levels in zip(categorical,categories):
        groups[name]=list(range(offset,offset+len(levels))); offset+=len(levels)
    baseline=average_precision_score(y,clf.predict_proba(Z)[:,1]); rng=np.random.default_rng(seed)
    permutations=[rng.permutation(len(Z)) for _ in range(repeats)]
    values={}
    for name,indices in groups.items():
        drops=[]
        for order in permutations:
            z=Z.copy(); z[:,indices]=Z[order][:,indices]
            drops.append(baseline-average_precision_score(y,clf.predict_proba(z)[:,1]))
        values[name]=float(np.mean(drops))
    return pd.Series(values)

def analyze_fold(key,fold_id,stamp):
    path=ICACHE/f'{fold_id:02d}_{key}.joblib'
    if path.exists():
        out=joblib.load(path)
        if out['fingerprint'] != stamp: raise RuntimeError('Interpretability checkpoint/code mismatch')
        return path
    start=time.monotonic(); d,X,y=load_data()
    rec=joblib.load(CACHE/f'{fold_id:02d}_{key}.joblib')
    tr,te=rec['train_indices'],rec['test_indices']; model=rec['model']
    Xtr,ytr=X.iloc[tr],y.iloc[tr]; probe=X.iloc[te]
    with threadpool_limits(limits=1):
        C,error=attributions(model.pipeline,probe,Xtr)
        if key.startswith('logistic'):
            # Positive scalar calibration preserves K80 and absolute ranking.
            C*=float(model.calibrator.coef_[0,0])
            _,exact=linear_contributions(model,probe)
            np.testing.assert_allclose(C,exact,atol=1e-8)
        k80,top=k80_top3(C)
        mass=C.abs().mean(); pi=transformed_permutation(model.pipeline,probe,y.iloc[te],seed=20000+fold_id)
        rho=float(spearmanr(mass.reindex(pi.index),pi).statistic)
        jaccard=np.zeros((50,len(te)))
        for b in range(50):
            rng=np.random.default_rng(30000+fold_id*100+b)
            sampled=np.concatenate([rng.choice(np.flatnonzero(ytr.to_numpy()==v),
                size=int((ytr==v).sum()),replace=True) for v in [0,1]])
            boot=clone(model.pipeline).fit(Xtr.iloc[sampled],ytr.iloc[sampled])
            B,_=attributions(boot,probe,Xtr)
            _,bt=k80_top3(B)
            jaccard[b]=[len(a&c)/len(a|c) if a|c else 1.0 for a,c in zip(top,bt)]
    out={'fingerprint':stamp,'model':key,'fold_id':fold_id,'repeat':fold_id//5,
        'indices':te,'employee_ids':d.EmployeeNumber.iloc[te].to_numpy(),'k80':k80,
        'top3':['|'.join(sorted(a)) for a in top], 'jaccard':jaccard,
        'global_attribution':mass,'permutation_AP_drop':pi,'spearman':rho,
        'max_additivity_error':error,'n_features':C.shape[1],'seconds':time.monotonic()-start}
    joblib.dump(out,path,compress=3)
    print(f'Explained {fold_id+1:02d}/25 {key}: 294 held-out rows, 50 refits, {out["seconds"]:.1f}s',flush=True)
    return path

def repeat_bootstrap(values):
    values=np.asarray(values,dtype=float)
    assert len(values)==5
    draws=np.random.default_rng(42).integers(0,5,size=(10000,5))
    distribution=values[draws].mean(axis=1)
    return float(values.mean()),*map(float,np.quantile(distribution,[.025,.975]))

def summarize():
    rows=[]; folds=[]; global_rows=[]; draws=[]
    for path in sorted(ICACHE.glob('*.joblib')):
        r=joblib.load(path); key=r['model']; fid=r['fold_id']
        for i,index in enumerate(r['indices']):
            rows.append({'model':key,'fold_id':fid,'repeat':fid//5,'row_index':int(index),
                'employee_id':int(r['employee_ids'][i]),'k80':int(r['k80'][i]),
                'top3':r['top3'][i],'mean_top3_jaccard':float(r['jaccard'][:,i].mean()),
                'bootstrap_refits':50})
        folds.append({'model':key,'fold_id':fid,'repeat':fid//5,
            'mean_k80':float(np.mean(r['k80'])),'median_k80':float(np.median(r['k80'])),
            'top3_jaccard':float(r['jaccard'].mean()),'spearman':r['spearman'],
            'max_additivity_error':r['max_additivity_error'],'n_features':r['n_features']})
        for name,value in r['global_attribution'].items():
            global_rows.append({'model':key,'fold_id':fid,'repeat':fid//5,'feature':name,
                'mean_absolute_contribution':value,'permutation_AP_drop':r['permutation_AP_drop'][name]})
        for b,values in enumerate(r['jaccard']):
            draws.append({'model':key,'fold_id':fid,'bootstrap_refit':b,'mean_jaccard':float(values.mean())})
    rows=pd.DataFrame(rows); folds=pd.DataFrame(folds); global_rows=pd.DataFrame(global_rows)
    rows.to_csv(RESULTS/'interpretability_employees.csv.gz',index=False)
    folds.to_csv(RESULTS/'interpretability_folds.csv',index=False)
    pd.DataFrame(draws).to_csv(RESULTS/'bootstrap_refit_summary.csv.gz',index=False)
    global_rows.to_csv(RESULTS/'global_attributions_by_fold.csv.gz',index=False)
    repeat=folds.groupby(['model','repeat'])[['mean_k80','top3_jaccard','spearman']].mean().reset_index()
    repeat.to_csv(RESULTS/'interpretability_repeats.csv',index=False)
    summary=[]; paired=[]
    for key,frame in repeat.groupby('model'):
        for metric in ['mean_k80','top3_jaccard','spearman']:
            estimate,lo,hi=repeat_bootstrap(frame[metric])
            summary.append({'model':key,'metric':metric,'estimate':estimate,'ci_low':lo,'ci_high':hi,
                'unique_employees':1470,'held_out_observations':7350,'refits_per_fold':50,
                'interval_method':'bootstrap 5 repeat means; conditional descriptive interval'})
    for family in ['logistic','forest']:
        ref=repeat[repeat.model==family+'_raw'].set_index('repeat')
        for mode in ['all','no_log']:
            eng=repeat[repeat.model==family+'_'+mode].set_index('repeat')
            for metric in ['mean_k80','top3_jaccard','spearman']:
                estimate,lo,hi=repeat_bootstrap(eng[metric]-ref[metric])
                paired.append({'reference':family+'_raw','candidate':family+'_'+mode,
                    'metric':metric,'difference':estimate,'ci_low':lo,'ci_high':hi})
    pd.DataFrame(summary).to_csv(RESULTS/'interpretability_summary.csv',index=False)
    pd.DataFrame(paired).to_csv(RESULTS/'interpretability_paired.csv',index=False)
    global_summary=global_rows.groupby(['model','feature'])[['mean_absolute_contribution','permutation_AP_drop']].mean().reset_index()
    global_summary.to_csv(RESULTS/'global_attributions.csv',index=False)
    save_json(RESULTS/'interpretability_audit.json',{'models':KEYS,'outer_folds_per_model':25,
        'unique_employees_per_model':1470,'held_out_observations_per_model':7350,
        'bootstrap_refits_per_fold':50,'total_bootstrap_refits':7500,
        'max_base_additivity_error':float(folds.max_additivity_error.max()),
        'interval_limit':'Five repeat means share a dataset and training records. Intervals describe split variation conditional on data, not population uncertainty.',
        'forest_units':'TreeSHAP for uncalibrated forest class-1 probability; not calibrated probability contributions.',
        'ranking_limit':'Spearman is agreement between different notions of importance, not causal validity or proof of explanation fidelity.',
        'stability_limit':'Conditional on outer-selected hyperparameters; no retuning or recalibration in bootstrap refits. Linear scalar calibration cannot change K80 or top3.'})

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--summarize-only',action='store_true'); args=parser.parse_args()
    if not args.summarize_only:
        ICACHE.mkdir(parents=True,exist_ok=True)
        stamp=hashlib.sha256((Path(__file__).read_bytes()+fingerprint().encode())).hexdigest()
        with joblib.parallel_config(backend='multiprocessing'):
            joblib.Parallel(n_jobs=args.jobs)(joblib.delayed(analyze_fold)(key,fold,stamp)
                for fold in range(25) for key in KEYS)
    summarize()
