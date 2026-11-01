"""Repeated nested CV: python train.py --jobs 6. See protocol.json.

Outer test predictions follow training-only tuning, calibration and threshold
selection. Checkpoints are resumable and fingerprinted against code/protocol.
"""
import argparse, hashlib, json, platform, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, log_loss)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from threadpoolctl import threadpool_limits
from attrition import SEED, BLOCKS, EXCLUDED, make_pipeline, ProbabilityModel, calibration_bins

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv'
RESULTS = ROOT/'results'
MODELS = ROOT/'models'
CACHE = ROOT/'.cache/nested'

def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False,
        default=lambda o: o.item() if hasattr(o, 'item') else str(o))+'\n', encoding='utf-8')

def load_data():
    d = pd.read_csv(DATA)
    protocol = json.loads((ROOT/'protocol.json').read_text())
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == protocol['dataset_sha256']
    assert d.shape == (1470,35) and not d.isna().any().any()
    assert not d.EmployeeNumber.duplicated().any() and set(d.Attrition) == {'Yes','No'}
    return d, d.drop(columns='Attrition'), d.Attrition.map({'No':0,'Yes':1})

def candidates():
    return {
      'logistic': (LogisticRegression(max_iter=3000,solver='lbfgs',random_state=SEED),
          {'model__C':[.05,.2,1,5], 'model__class_weight':[None,'balanced']}),
      'tree': (DecisionTreeClassifier(random_state=SEED),
          {'model__max_depth':[3,5], 'model__min_samples_leaf':[10,25],
           'model__class_weight':[None,'balanced']}),
      'forest': (RandomForestClassifier(n_estimators=150,random_state=SEED,n_jobs=1),
          {'model__max_depth':[5,None], 'model__min_samples_leaf':[3,10],
           'model__class_weight':[None,'balanced']}),
    }

def specifications():
    return ([(family,mode) for family in candidates() for mode in ['raw','all','no_log']]
            + [('logistic','without_'+b) for b in BLOCKS]
            + [('logistic','no_log_age')])

def outer_splits(X,y):
    return list(RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=SEED).split(X,y))

def metrics(y,p,threshold):
    pred = np.asarray(p) >= threshold
    tn,fp,fn,tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
    _,ece = calibration_bins(y,p)
    return {'average_precision':average_precision_score(y,p), 'roc_auc':roc_auc_score(y,p),
        'brier':brier_score_loss(y,p),'log_loss':log_loss(y,p,labels=[0,1]), 'ece':ece,
        'accuracy':accuracy_score(y,pred),'precision':precision_score(y,pred,zero_division=0),
        'recall':recall_score(y,pred,zero_division=0),'f1':f1_score(y,pred,zero_division=0),
        'threshold':float(threshold),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}

def choose_threshold(y,p):
    grid = np.linspace(.05,.80,151)
    return float(max(grid,key=lambda t:(f1_score(y,np.asarray(p)>=t,zero_division=0),t)))

def corrected_interval(differences):
    """Approximate Nadeau-Bengio correction; 25 folds are NOT independent."""
    d = np.asarray(differences,dtype=float)
    se = np.sqrt((1/len(d)+1/4)*np.var(d,ddof=1))
    mean = float(d.mean()); margin = stats.t.ppf(.975,len(d)-1)*se
    p = float(2*stats.t.sf(abs(mean/se),len(d)-1)) if se else (1.0 if mean==0 else 0.0)
    return {'difference':mean,'ci_low':mean-margin,'ci_high':mean+margin,
            'corrected_se':float(se),'p_two_sided':p}

def fit_model(family,mode,X,y,seed):
    estimator,grid = candidates()[family]
    search = GridSearchCV(make_pipeline(estimator,mode), grid, scoring='average_precision',
        cv=StratifiedKFold(5,shuffle=True,random_state=seed), n_jobs=1,
        refit=True,error_score='raise')
    search.fit(X,y)
    model = ProbabilityModel(search.best_estimator_,X,y,cv_seed=seed+1)
    model.threshold = choose_threshold(y,model.training_oof_probabilities)
    assert model.calibrator.coef_[0,0] > 0, 'Unexpected reversed calibration ranking'
    return model,search

def fingerprint():
    h=hashlib.sha256()
    for name in ['protocol.json','attrition.py','train.py']: h.update((ROOT/name).read_bytes())
    return h.hexdigest()

def fit_fold(family,mode,fold_id,tr,te,X,y,stamp):
    key=family+'_'+mode; path=CACHE/f'{fold_id:02d}_{key}.joblib'
    if path.exists():
        record=joblib.load(path)
        if record['fingerprint'] != stamp: raise RuntimeError('Checkpoint/code mismatch: '+str(path))
        return record['metrics']
    start=time.monotonic()
    with threadpool_limits(limits=1):
        model,search=fit_model(family,mode,X.iloc[tr],y.iloc[tr],1000+fold_id*10)
        p=model.predict_proba(X.iloc[te])[:,1]
    row={'model':key,'family':family,'feature_mode':mode,'fold_id':fold_id,
         'repeat':fold_id//5,'fold':fold_id%5,'train_rows':len(tr),'test_rows':len(te),
         'inner_best_AP':float(search.best_score_), **metrics(y.iloc[te],p,model.threshold)}
    record={'fingerprint':stamp,'metrics':row,'train_indices':tr,'test_indices':te,
            'model':model,'probabilities':p,'best_params':search.best_params_,
            'inner_results':pd.DataFrame(search.cv_results_),'fit_seconds':time.monotonic()-start}
    joblib.dump(record,path,compress=3)
    print(f'Completed {fold_id+1:02d}/25 {key} in {record["fit_seconds"]:.1f}s',flush=True)
    return row

def run_cv(jobs):
    RESULTS.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True); CACHE.mkdir(parents=True,exist_ok=True)
    d,X,y=load_data(); splits=outer_splits(X,y); stamp=fingerprint()
    save_json(RESULTS/'protocol_lock.json', {'protocol':json.loads((ROOT/'protocol.json').read_text()),
        'fit_fingerprint':stamp,'families':{k:v[1] for k,v in candidates().items()}})
    manifest=[{'fold_id':i,'repeat':i//5,'fold':i%5,'train_indices':tr.tolist(),
               'test_indices':te.tolist(),'inner_seed':1000+i*10,'calibration_seed':1001+i*10}
              for i,(tr,te) in enumerate(splits)]
    save_json(RESULTS/'split_manifest.json',manifest)
    with joblib.parallel_config(backend='multiprocessing'):
        rows=joblib.Parallel(n_jobs=jobs)(joblib.delayed(fit_fold)(family,mode,i,tr,te,X,y,stamp)
            for i,(tr,te) in enumerate(splits) for family,mode in specifications())
    pd.DataFrame(rows).to_csv(RESULTS/'outer_fold_metrics.csv',index=False)
    prediction_frames=[]; tuning=[]; inner=[]
    for path in sorted(CACHE.glob('*.joblib')):
        rec=joblib.load(path); m=rec['metrics']; te=rec['test_indices']
        prediction_frames.append(pd.DataFrame({'model':m['model'],'fold_id':m['fold_id'],
            'repeat':m['repeat'],'row_index':te,'employee_id':d.EmployeeNumber.iloc[te].to_numpy(),
            'y_true':y.iloc[te].to_numpy(),'probability':rec['probabilities'],'threshold':m['threshold']}))
        tuning.append({'model':m['model'],'fold_id':m['fold_id'],'best_params':rec['best_params'],
            'threshold':m['threshold'],'sigmoid_slope':float(rec['model'].calibrator.coef_[0,0]),
            'sigmoid_intercept':float(rec['model'].calibrator.intercept_[0]),
            'income_fallback':rec['model'].pipeline.named_steps['domain'].income_fallback_,
            'role_medians':rec['model'].pipeline.named_steps['domain'].role_medians_})
        r=rec['inner_results'][['params','mean_test_score','std_test_score','rank_test_score']].copy()
        r.insert(0,'fold_id',m['fold_id']); r.insert(0,'model',m['model']); inner.append(r)
    pd.concat(prediction_frames).to_csv(RESULTS/'outer_predictions.csv.gz',index=False)
    pd.concat(inner).to_csv(RESULTS/'inner_search_results.csv.gz',index=False)
    save_json(RESULTS/'outer_fit_audit.json',tuning)
    save_json(RESULTS/'data_audit.json', {'rows':len(d),'columns':len(d.columns),'positive':int(y.sum()),
        'negative':int((y==0).sum()),'missing_cells':int(d.isna().sum().sum()),
        'excluded_predictors':EXCLUDED,'source_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),
        'unique_employees':1470,'outer_test_observations_per_model':7350,'outer_folds':25})
    for col in ['OverTime','JobRole','JobSatisfaction','WorkLifeBalance']:
        d.assign(attrition=y).groupby(col).agg(n=('attrition','size'),attrition_rate=('attrition','mean')).to_csv(RESULTS/f'eda_{col}.csv')
    summarize()

def summarize():
    frame=pd.read_csv(RESULTS/'outer_fold_metrics.csv')
    cols=['average_precision','roc_auc','brier','ece','accuracy','precision','recall','f1']
    summary=frame.groupby(['model','family','feature_mode'])[cols].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(RESULTS/'nested_cv_summary.csv',index=False)
    paired=[]; pairs=[]
    for mode in ['all','no_log']: pairs += [(mode,f'{f}_raw',f'{f}_{mode}') for f in candidates()]
    pairs += [('ablation','logistic_no_log','logistic_without_'+b) for b in BLOCKS]
    pairs += [('age','logistic_no_log','logistic_no_log_age')]
    for comparison,left,right in pairs:
        a=frame.query('model == @left').set_index('fold_id'); b=frame.query('model == @right').set_index('fold_id')
        for metric in ['average_precision','roc_auc','brier']:
            diff=(b[metric]-a[metric]).sort_index()
            paired.append({'comparison':comparison,'reference':left,'candidate':right,'metric':metric,
                **corrected_interval(diff),'wins':int((diff<0).sum() if metric=='brier' else (diff>0).sum()),
                'ties':int((diff==0).sum()),'folds':len(diff)})
    paired=pd.DataFrame(paired); paired['p_holm_AP']=np.nan
    for comparison in ['all','no_log']:
        ix=paired.index[(paired.comparison==comparison)&(paired.metric=='average_precision')]
        ordered=paired.loc[ix].sort_values('p_two_sided').index
        adjusted=np.maximum.accumulate(paired.loc[ordered,'p_two_sided'].to_numpy()*np.arange(len(ix),0,-1))
        paired.loc[ordered,'p_holm_AP']=np.minimum(adjusted,1)
    paired.to_csv(RESULTS/'paired_comparisons.csv',index=False)

def final_fit():
    d,X,y=load_data(); summary=pd.read_csv(RESULTS/'nested_cv_summary.csv')
    eligible=summary[summary.feature_mode.isin(['raw','no_log'])].sort_values(
        ['average_precision_mean','brier_mean','model'],ascending=[False,True,True])
    selected=eligible.iloc[0].model
    keys=list(dict.fromkeys([selected,'logistic_raw','logistic_all','logistic_no_log']))
    models={}; fit_info={}
    for key in keys:
        family,mode=key.split('_',1); print('Final refit:',key,flush=True)
        with threadpool_limits(limits=1): model,search=fit_model(family,mode,X,y,5000)
        models[key]=model
        fit_info[key]={'best_params':search.best_params_,'threshold':model.threshold,'inner_AP':float(search.best_score_)}
    bundle={'version':'2.0','selected':selected,'models':models,'protocol':json.loads((ROOT/'protocol.json').read_text()),
            'fit_info':fit_info,'predictors':[c for c in X.columns if c not in EXCLUDED]}
    joblib.dump(bundle,MODELS/'bundle.joblib',compress=3)
    save_json(RESULTS/'selection.json',{'selected':selected,'eligible':eligible.model.tolist(),
        'selection_rule':bundle['protocol']['selection'],'fit_info':fit_info,
        'caveat':'Outer CV compares fixed candidate procedures; selecting the winner adds selection optimism. No new external test.'})
    X.to_csv(ROOT/'data/training_input_template.csv',index=False)
    X[bundle['predictors']].iloc[:5].to_csv(ROOT/'data/sample_employees.csv',index=False)
    preds=pd.read_csv(RESULTS/'outer_predictions.csv.gz'); selected_predictions=preds[preds.model==selected]
    reliability=[]; ec=[]
    for repeat,p in selected_predictions.groupby('repeat'):
        bins,ece=calibration_bins(p.y_true,p.probability); bins.insert(0,'repeat',repeat)
        reliability.append(bins); ec.append(ece)
    pd.concat(reliability).to_csv(RESULTS/'reliability_by_repeat.csv',index=False)
    pooled,pool_ece=calibration_bins(selected_predictions.y_true,selected_predictions.probability)
    pooled.to_csv(RESULTS/'reliability_pooled.csv',index=False)
    save_json(RESULTS/'calibration_summary.json',{'model':selected,'bins':10,
        'repeat_ECE':ec,'mean_repeat_ECE':float(np.mean(ec)),'sd_repeat_ECE':float(np.std(ec,ddof=1)),
        'pooled_ECE':pool_ece,'observations':7350,'unique_employees':1470,
        'meaning':'Repeated held-out estimates of the selected training procedure, not 7350 independent employees; not a calibration guarantee for the full-data refit.'})
    import importlib.metadata
    save_json(RESULTS/'environment.json',{'python':platform.python_version(),
        'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','joblib','matplotlib']}})

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--stage',choices=['cv','final','all'],default='all'); args=parser.parse_args()
    if args.stage in ['cv','all']: run_cv(args.jobs)
    if args.stage in ['final','all']: final_fit()
