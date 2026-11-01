"""Cross-check the optional accelerator against reference SHAP for 882 rows."""
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,joblib,shap
from train import load_data,CACHE,RESULTS,save_json
from interpretability import transformed,group_encoded_contributions,k80_top3
from fast_tree_backend import activate
reference_explainer=shap.TreeExplainer
_,X,y=load_data();checks=[]
for key in ['forest_raw','forest_all','forest_no_log']:
    rec=joblib.load(CACHE/f'00_{key}.joblib');p=rec['model'].pipeline;Z=transformed(p,X.iloc[rec['test_indices']])
    expected=reference_explainer(p.named_steps['model'],feature_perturbation='tree_path_dependent',model_output='raw').shap_values(Z,check_additivity=True)
    activate();e=shap.TreeExplainer(p.named_steps['model'],feature_perturbation='tree_path_dependent',model_output='raw')
    actual=e.shap_values(Z,check_additivity=True)
    error=float(np.max(abs(actual-expected)));assert error<1e-10
    A=group_encoded_contributions(p.named_steps['prep'],actual[:,:,1]);B=group_encoded_contributions(p.named_steps['prep'],expected[:,:,1])
    ka,ta=k80_top3(A);kb,tb=k80_top3(B)
    np.testing.assert_array_equal(ka,kb);assert ta==tb
    checks.append({'model':key,'employees':len(Z),'max_absolute_difference':error,'k80_identical':True,'top3_identical':True})
    print(key,error,flush=True)
save_json(RESULTS/'explanation_backend.json',{'status':'PASS','reference':'shap 0.50.0 TreeExplainer tree_path_dependent','accelerator':'fasttreeshap 0.1.6 exact v1, one thread per worker','checks':checks,'note':'The supplied bootstrap measurements use either reference or numerically matched fast TreeSHAP. Both compute exact path-dependent base-forest SHAP, not an approximation. Standard reproduction needs only shap; the optional accelerator needs a compiler and compatibility wrapper.'})
