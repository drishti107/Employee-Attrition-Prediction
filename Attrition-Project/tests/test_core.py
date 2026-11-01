"""Focused leakage, input and explanation checks for upgraded project."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import joblib,numpy as np,pandas as pd
from sklearn.base import clone
from attrition import DomainFeatures, validate_input, calibration_bins, linear_contributions
from explanations import make_report,compare_reports
from train import load_data,choose_threshold

def run():
    _,X,y=load_data(); bundle=joblib.load(ROOT/'models/bundle.joblib')
    raw=bundle['models']['logistic_raw'];eng=bundle['models']['logistic_no_log']
    d=DomainFeatures('no_log').fit(X.iloc[:100])
    expected=X.iloc[:100].groupby('JobRole').MonthlyIncome.median().to_dict()
    assert d.role_medians_==expected
    assert d.income_fallback_==X.iloc[:100].MonthlyIncome.median()
    row=X.iloc[[100]].copy();row['YearsAtCompany']=0
    for name in ['YearsSinceLastPromotion','YearsWithCurrManager','YearsInCurrentRole']:row[name]=0
    row['JobRole']='Unseen role'
    z=d.transform(row)
    assert np.isfinite(z.select_dtypes('number')).all().all()
    assert np.isclose(z.IncomeToRoleMedian.iloc[0],row.MonthlyIncome.iloc[0]/d.income_fallback_)
    assert 'LogMonthlyIncome' not in z and len(z.columns)==34
    for mode,expected_count in [('raw',27),('all',35),('no_log',34),('without_workload',32),
        ('without_career',31),('without_reward',33),('without_sentiment',33),('no_log_age',35)]:
        transformer=DomainFeatures(mode).fit(X)
        assert transformer.transform(X.iloc[:2]).shape[1]==expected_count
    for model in bundle['models'].values():
        altered=X.iloc[:3].copy();altered['Age']=99;altered['Gender']='Excluded';altered['MaritalStatus']='Excluded'
        np.testing.assert_allclose(model.predict_proba(altered),model.predict_proba(X.iloc[:3]),atol=0)
        assert model.threshold==choose_threshold(y,model.training_oof_probabilities)
    for index in [0,50,300]:
        comparison=compare_reports(raw,eng,X.iloc[[index]])
        for report in [comparison['original'],comparison['engineered']]:
            assert report['group_additivity_error']<1e-9
            assert abs(report['feature_explanation']['reconstructed_probability']-report['probability'])<1e-9
        domain=[f for f in comparison['engineered']['feature_explanation']['factors'] if f['is_engineered']]
        assert len(domain)==7 and all(f['definition'] for f in domain)
    for column,value in [('WorkLifeBalance',9),('MonthlyIncome',float('inf')),('YearsAtCompany',1000),('TrainingTimesLastYear',1.5)]:
        bad=X.iloc[[0]].copy();bad[column]=value
        try:validate_input(bad,eng.reference_X)
        except ValueError:pass
        else:raise AssertionError('Invalid input accepted: '+column)
    try:validate_input(X.iloc[[0]].drop(columns='JobRole'),eng.reference_X)
    except ValueError:pass
    else:raise AssertionError('Missing column accepted')
    bins,ece=calibration_bins([0,1],[0,1])
    assert ece==0 and bins.n.sum()==2
    print('PASS: fold-fitted medians, feature variants, excluded-field invariance, CV thresholds, local additivity and input validation.')

if __name__=='__main__':run()
