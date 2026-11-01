"""Verify data, matched folds, saved measurements and deployment explanations.
Run after extracting the complete project: python verify.py
--with-cache also audits every saved outer fitted model (available after rerun).
"""
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from train import ROOT,RESULTS,CACHE,load_data,outer_splits,metrics,corrected_interval,choose_threshold,specifications,fingerprint,save_json
from attrition import calibration_bins
from explanations import make_report

def verify(with_cache=False):
    d,X,y=load_data(); checks=[]
    manifest=json.loads((RESULTS/'split_manifest.json').read_text())
    assert len(manifest)==25
    for expected,actual in zip(outer_splits(X,y),manifest):
        tr,te=expected
        np.testing.assert_array_equal(tr,actual['train_indices']);np.testing.assert_array_equal(te,actual['test_indices'])
        assert len(tr)==1176 and len(te)==294 and not set(tr)&set(te)
    checks.append('25 reproducible matched outer folds; disjoint train/test per fold')
    lock=json.loads((RESULTS/'protocol_lock.json').read_text())
    assert lock['fit_fingerprint']==fingerprint()
    assert lock['protocol']==json.loads((ROOT/'protocol.json').read_text())
    fold=pd.read_csv(RESULTS/'outer_fold_metrics.csv')
    # Preserve near-tied tree scores exactly; the default CSV parser can merge
    # adjacent floating-point values and subtly change AP/AUC tie handling.
    pred=pd.read_csv(RESULTS/'outer_predictions.csv.gz',float_precision='round_trip')
    assert len(fold)==350 and len(pred)==102900
    expected_keys={a+'_'+b for a,b in specifications()}
    assert set(fold.model)==set(pred.model)==expected_keys
    assert pred.groupby(['model','repeat']).size().eq(1470).all()
    assert pred.groupby(['model','repeat']).row_index.nunique().eq(1470).all()
    assert pred.groupby(['model','row_index']).size().eq(5).all()
    np.testing.assert_array_equal(pred.y_true,y.iloc[pred.row_index].to_numpy())
    assert pred.probability.between(0,1).all()
    indexed=fold.set_index(['model','fold_id'])
    for (key,fid),p in pred.groupby(['model','fold_id']):
        assert set(p.row_index)==set(manifest[fid]['test_indices'])
        assert p.threshold.nunique()==1
        m=metrics(p.y_true,p.probability,p.threshold.iloc[0])
        for metric,value in m.items():np.testing.assert_allclose(indexed.loc[(key,fid),metric],value,atol=1e-12)
    summary=pd.read_csv(RESULTS/'nested_cv_summary.csv').set_index('model')
    for key,f in fold.groupby('model'):
        for metric in ['average_precision','roc_auc','brier','ece','accuracy','precision','recall','f1']:
            np.testing.assert_allclose(summary.loc[key,metric+'_mean'],f[metric].mean(),atol=1e-12)
            np.testing.assert_allclose(summary.loc[key,metric+'_std'],f[metric].std(),atol=1e-12)
    checks.append('All 102900 saved held-out probabilities reproduce 350 metric rows and summary means/SDs')
    pairs=pd.read_csv(RESULTS/'paired_comparisons.csv')
    for r in pairs.itertuples():
        a=fold[fold.model==r.reference].set_index('fold_id')[r.metric]
        b=fold[fold.model==r.candidate].set_index('fold_id')[r.metric]
        ci=corrected_interval((b-a).sort_index())
        for metric,value in ci.items():np.testing.assert_allclose(getattr(r,metric),value,atol=1e-12)
        assert r.wins==int(((b-a)<0).sum() if r.metric=='brier' else ((b-a)>0).sum())
    checks.append('Paired differences, corrected intervals and win counts match matched fold results')
    audit=json.loads((RESULTS/'outer_fit_audit.json').read_text());assert len(audit)==350
    for rec in audit:
        Xtr=X.iloc[manifest[rec['fold_id']]['train_indices']]
        assert rec['role_medians']==Xtr.groupby('JobRole').MonthlyIncome.median().to_dict()
        assert rec['income_fallback']==Xtr.MonthlyIncome.median()
        if with_cache:
            saved=joblib.load(CACHE/f"{rec['fold_id']:02d}_{rec['model']}.joblib")
            model=saved['model'];tr=saved['train_indices'];te=saved['test_indices']
            assert set(model.reference_X.index)==set(tr)
            assert set(model.background.index)<=set(tr)
            assert model.threshold==choose_threshold(y.iloc[tr],model.training_oof_probabilities)
            np.testing.assert_allclose(model.predict_proba(X.iloc[te])[:,1],saved['probabilities'],atol=1e-12)
    checks.append('350 training-fold salary-reference audits passed'+('; every outer fit, reference and training-CV threshold recomputed' if with_cache else ''))
    bundle=joblib.load(ROOT/'models/bundle.joblib');selection=json.loads((RESULTS/'selection.json').read_text())
    eligible=summary[summary.feature_mode.isin(['raw','no_log'])].reset_index().sort_values(['average_precision_mean','brier_mean','model'],ascending=[False,True,True])
    assert bundle['selected']==selection['selected']==eligible.iloc[0].model
    chosen=pred[pred.model==bundle['selected']]
    cal=json.loads((RESULTS/'calibration_summary.json').read_text())
    for repeat,p in chosen.groupby('repeat'):
        _,ece=calibration_bins(p.y_true,p.probability)
        np.testing.assert_allclose(cal['repeat_ECE'][repeat],ece,atol=1e-12)
    checks.append('Prespecified final selection and held-out calibration table agree')
    interp=pd.read_csv(RESULTS/'interpretability_employees.csv.gz')
    assert len(interp)==44100 and interp.groupby('model').row_index.nunique().eq(1470).all()
    assert interp.groupby(['model','repeat']).size().eq(1470).all()
    assert interp.bootstrap_refits.eq(50).all() and interp.mean_top3_jaccard.between(0,1).all()
    draws=pd.read_csv(RESULTS/'bootstrap_refit_summary.csv.gz')
    assert len(draws)==7500 and draws.groupby(['model','fold_id']).size().eq(50).all()
    from interpretability import repeat_bootstrap,KEYS,ICACHE
    assert set(interp.model)==set(KEYS)
    assert not interp.duplicated(['model','fold_id','row_index']).any()
    for (key,fid),rows in interp.groupby(['model','fold_id']):
        assert set(rows.row_index)==set(manifest[fid]['test_indices'])
        if with_cache:
            saved=joblib.load(ICACHE/f'{fid:02d}_{key}.joblib')
            assert saved['jaccard'].shape==(50,294)
            assert saved['max_additivity_error']<1e-7
            observed=rows.set_index('row_index').loc[saved['indices']]
            np.testing.assert_array_equal(observed.k80,saved['k80'])
            np.testing.assert_allclose(observed.mean_top3_jaccard,saved['jaccard'].mean(axis=0),atol=1e-12)
    folds_i=pd.read_csv(RESULTS/'interpretability_folds.csv')
    repeat_i=folds_i.groupby(['model','repeat'])[['mean_k80','top3_jaccard','spearman']].mean()
    for r in pd.read_csv(RESULTS/'interpretability_summary.csv').itertuples():
        np.testing.assert_allclose(repeat_bootstrap(repeat_i.loc[r.model,r.metric]),[r.estimate,r.ci_low,r.ci_high],atol=1e-12)
    for r in pd.read_csv(RESULTS/'interpretability_paired.csv').itertuples():
        difference=repeat_i.loc[r.candidate,r.metric]-repeat_i.loc[r.reference,r.metric]
        np.testing.assert_allclose(repeat_bootstrap(difference),[r.difference,r.ci_low,r.ci_high],atol=1e-12)
    checks.append('Interpretability covers 44100 held-out observations, 7500 bootstrap refits, and repeat-level intervals')
    example=joblib.load(ROOT/'models/example_fold.joblib')
    row=example['row'];fid=example['fold_id']
    assert row.index[0] in manifest[fid]['test_indices']
    for key,model in example['models'].items():
        assert row.index[0] not in model.reference_X.index
        report=make_report(model,row,key)
        assert report['group_additivity_error']<1e-9
    sys_path=str(ROOT/'tests')
    import sys
    sys.path.insert(0,sys_path)
    from test_core import run
    run();checks.append('Held-out worked example and focused core tests passed')
    manifest_path=ROOT/'FILE_CHECKSUMS.json'
    if manifest_path.exists():
        for name,digest in json.loads(manifest_path.read_text()).items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,'File changed: '+name
        checks.append('Delivered-file SHA-256 manifest passed')
    # Stable output so rerunning verification does not invalidate its own checksum.
    result={'status':'PASS','checks':checks,'with_outer_cache':with_cache}
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--with-cache',action='store_true');p.add_argument('--save',action='store_true');a=p.parse_args()
    result=verify(a.with_cache)
    if a.save:save_json(RESULTS/'verification_summary.json',result)
