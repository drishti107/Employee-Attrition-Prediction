"""Core models and auditable local explanations. See docs/METHODOLOGY.md."""
from math import factorial
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
EXCLUDED = ['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours',
            'Age', 'Gender', 'MaritalStatus']
GROUPS = {
    'Workload and commute': ['BusinessTravel', 'DistanceFromHome', 'OverTime', 'WorkLifeBalance'],
    'Satisfaction and involvement': ['EnvironmentSatisfaction', 'JobSatisfaction', 'RelationshipSatisfaction', 'JobInvolvement'],
    'Career and tenure': ['TotalWorkingYears', 'NumCompaniesWorked', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'],
    'Pay and benefits': ['MonthlyIncome', 'DailyRate', 'HourlyRate', 'MonthlyRate', 'PercentSalaryHike', 'StockOptionLevel'],
    'Education and performance': ['Education', 'EducationField', 'PerformanceRating'],
    'Role and department': ['Department', 'JobRole', 'JobLevel'],
}
BLOCKS = {
    'workload': ['OvertimeBalancePressure', 'OvertimeCommute'],
    'career': ['PromotionDelayShare', 'RoleTenureShare', 'ManagerTenureShare'],
    'reward': ['IncomeToRoleMedian', 'LogMonthlyIncome'],
    'sentiment': ['MeanDissatisfaction'],
}

class DomainFeatures(TransformerMixin, BaseEstimator):
    """All learned statistics are fitted inside each training fold."""
    def __init__(self, mode='raw'):
        self.mode = mode

    def fit(self, X, y=None):
        X = X.copy()
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        excluded = [c for c in EXCLUDED if not (c == 'Age' and self.mode == 'no_log_age')]
        self.columns_ = [c for c in X.columns if c not in excluded + ['Attrition']]
        income = pd.to_numeric(X['MonthlyIncome'], errors='coerce')
        self.income_fallback_ = float(income.median())
        self.role_medians_ = X.assign(MonthlyIncome=income).groupby('JobRole')['MonthlyIncome'].median().to_dict()
        if self.mode not in ['raw','all','no_log','no_log_age'] + ['without_' + b for b in BLOCKS]:
            raise ValueError('Unknown feature mode: ' + self.mode)
        return self

    def transform(self, X):
        Z = X[self.columns_].copy()
        if self.mode == 'raw':
            return Z
        overtime = X['OverTime'].map({'Yes': 1.0, 'No': 0.0})
        denom = X['YearsAtCompany'] + 1.0
        peer = X['JobRole'].map(self.role_medians_).fillna(self.income_fallback_).clip(lower=1)
        values = {
            'OvertimeBalancePressure': overtime * (5.0 - X['WorkLifeBalance']),
            'OvertimeCommute': overtime * X['DistanceFromHome'],
            'PromotionDelayShare': X['YearsSinceLastPromotion'] / denom,
            'RoleTenureShare': X['YearsInCurrentRole'] / denom,
            'ManagerTenureShare': X['YearsWithCurrManager'] / denom,
            'IncomeToRoleMedian': X['MonthlyIncome'] / peer,
            'LogMonthlyIncome': np.log1p(X['MonthlyIncome'].clip(lower=0)),
            'MeanDissatisfaction': 5.0 - X[['JobSatisfaction','EnvironmentSatisfaction','RelationshipSatisfaction']].mean(axis=1),
        }
        for block, names in BLOCKS.items():
            if self.mode != 'without_' + block:
                for name in names:
                    if name == 'LogMonthlyIncome' and self.mode != 'all':
                        continue
                    Z[name] = values[name]
        return Z


def make_pipeline(model, mode):
    num = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    cat = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                    ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    prep = ColumnTransformer([('num',num,make_column_selector(dtype_include=np.number)),
                              ('cat',cat,make_column_selector(dtype_exclude=np.number))],
                             verbose_feature_names_out=False)
    return Pipeline([('domain',DomainFeatures(mode)),('prep',prep),('model',model)])


def raw_score(estimator, X):
    if hasattr(estimator, 'decision_function'):
        return estimator.decision_function(X)
    return logit(np.clip(estimator.predict_proba(X)[:,1],1e-6,1-1e-6))

class ProbabilityModel:
    """Training-only OOF sigmoid calibration; one full-training estimator."""
    def __init__(self, fitted_pipeline, X_train, y_train, cv_seed=SEED + 1):
        self.pipeline = fitted_pipeline
        method = 'decision_function' if hasattr(fitted_pipeline,'decision_function') else 'predict_proba'
        cv = StratifiedKFold(5, shuffle=True, random_state=cv_seed)
        oof = cross_val_predict(clone(fitted_pipeline), X_train, y_train, cv=cv, method=method, n_jobs=1)
        score = oof if method == 'decision_function' else logit(np.clip(oof[:,1],1e-6,1-1e-6))
        self.calibrator = LogisticRegression(C=1e6, solver='lbfgs',max_iter=2000).fit(score.reshape(-1,1), y_train)
        self.training_oof_scores = np.asarray(score)
        self.training_oof_probabilities = self.calibrator.predict_proba(score.reshape(-1,1))[:,1]
        self.calibration_cv_seed = cv_seed
        self.reference_X = X_train.copy()
        self.background = X_train.sample(min(32,len(X_train)),random_state=SEED).copy()
        self.threshold = 0.5

    def predict_proba(self, X):
        return self.calibrator.predict_proba(raw_score(self.pipeline,X).reshape(-1,1))

    def predict(self, X):
        return (self.predict_proba(X)[:,1] >= self.threshold).astype(int)


def group_encoded_contributions(prep, values, index=None):
    """Sum one-hot levels into their original variable; preserve additivity."""
    names = prep.get_feature_names_out()
    frame = pd.DataFrame(values, columns=names, index=index)
    numeric = list(prep.transformers_[0][2])
    categorical = list(prep.transformers_[1][2])
    categories = prep.named_transformers_['cat'].named_steps['encode'].categories_
    result = frame[numeric].copy()
    offset = len(numeric)
    for name, levels in zip(categorical, categories):
        result[name] = frame.iloc[:, offset:offset + len(levels)].sum(axis=1)
        offset += len(levels)
    return result


def linear_contributions(model, X):
    """Exact additive decomposition of calibrated log-odds around training mean.
    Not SHAP, not probability points, and not a causal effect.
    """
    p = model.pipeline
    if not isinstance(p.named_steps['model'], LogisticRegression):
        raise TypeError('Linear contributions require logistic regression.')
    transform = lambda data: p.named_steps['prep'].transform(p.named_steps['domain'].transform(data))
    Z, R = transform(X), transform(model.reference_X)
    mean = R.mean(axis=0)
    classifier = p.named_steps['model']
    a = float(model.calibrator.coef_[0,0])
    b = float(model.calibrator.intercept_[0])
    coef = classifier.coef_[0]
    base = float(a * (classifier.intercept_[0] + mean @ coef) + b)
    contributions = a * (Z - mean) * coef
    np.testing.assert_allclose(expit(base + contributions.sum(axis=1)),model.predict_proba(X)[:,1],atol=1e-9)
    return base, group_encoded_contributions(p.named_steps['prep'],contributions,index=X.index)


def grouped_shapley(model, row, background=None):
    """Exact enumeration of six-group empirical interventional Shapley values.
    Background expectation uses 32 training records, not a population estimate.
    Recompute derived features for every coalition. Correlated-group hybrids
    can be unrealistic; attributions depend on groups and background.
    """
    if len(row) != 1:
        raise ValueError('Explain exactly one employee at a time.')
    bg = model.background.copy() if background is None else background.copy()
    groups = list(GROUPS.items())
    batches = []
    for mask in range(1 << len(groups)):
        hybrid = bg.copy()
        for j, (_, columns) in enumerate(groups):
            if mask & (1 << j):
                for col in columns:
                    hybrid[col] = row.iloc[0][col]
        batches.append(hybrid)
    values = model.predict_proba(pd.concat(batches,ignore_index=True))[:,1].reshape(1 << len(groups),len(bg)).mean(axis=1)
    phis = []
    M = len(groups)
    for j, (name, _) in enumerate(groups):
        phi = 0.0
        for mask in range(1 << M):
            if not mask & (1 << j):
                size = mask.bit_count()
                weight = factorial(size)*factorial(M-size-1)/factorial(M)
                phi += weight*(values[mask | (1 << j)]-values[mask])
        phis.append({'factor_group':name,'contribution_probability_points':100*float(phi)})
    prediction = float(model.predict_proba(row)[0,1])
    np.testing.assert_allclose(values[0]+sum(p['contribution_probability_points'] for p in phis)/100,prediction,atol=1e-9)
    return {'probability':prediction,'reference_probability':float(values[0]),
            'factors':sorted(phis,key=lambda x:abs(x['contribution_probability_points']),reverse=True)}


def validate_input(frame, reference):
    """Check required predictors and documented survey/range/tenure constraints."""
    if frame.empty:
        raise ValueError('Input contains no employee rows.')
    if frame.columns.duplicated().any():
        raise ValueError('Input contains duplicate column names.')
    required = [c for c in reference.columns if c not in EXCLUDED+['Attrition']]
    missing = sorted(set(required)-set(frame.columns))
    if missing:
        raise ValueError('Missing predictor columns: ' + ', '.join(missing))
    X = frame.copy()
    if X[required].isna().any().any():
        raise ValueError('Demo input contains missing values. Complete all predictor fields.')
    for c in required:
        if pd.api.types.is_numeric_dtype(reference[c]):
            X[c] = pd.to_numeric(X[c],errors='raise')
            if not np.isfinite(X[c]).all() or (X[c] < 0).any():
                raise ValueError(c + ' must be finite and nonnegative.')
        elif not X[c].isin(reference[c].dropna().unique()).all():
            raise ValueError(c + ' contains an unknown category.')
    ranges = {'Education':(1,5),'JobLevel':(1,5),'StockOptionLevel':(0,3),'PerformanceRating':(1,4),
              **{c:(1,4) for c in ['EnvironmentSatisfaction','JobInvolvement','JobSatisfaction','RelationshipSatisfaction','WorkLifeBalance']}}
    for c,(low,high) in ranges.items():
        if not X[c].between(low,high).all() or (X[c] % 1 != 0).any():
            raise ValueError(f'{c} must be an integer from {low} to {high}.')
    for c in ['NumCompaniesWorked', 'TrainingTimesLastYear']:
        if (X[c] % 1 != 0).any():
            raise ValueError(c + ' must be a whole-number count.')
    if not X['PercentSalaryHike'].between(0, 100).all():
        raise ValueError('PercentSalaryHike must be between 0 and 100.')
    if (X['YearsAtCompany'] > X['TotalWorkingYears']).any():
        raise ValueError('YearsAtCompany cannot exceed TotalWorkingYears.')
    for c in ['YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']:
        if (X[c] > X['YearsAtCompany']).any():
            raise ValueError(c + ' cannot exceed YearsAtCompany.')
    return X


def calibration_bins(y, probabilities, n_bins=10):
    """Equal-width reliability bins and empirical ECE; empty bins are omitted.

    ECE is bin-dependent and has finite-sample bias. It is not a guarantee that
    an individual estimate is accurate. Call on held-out probabilities.
    """
    y, p = np.asarray(y), np.asarray(probabilities)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            rows.append({'bin': b, 'lower': b/n_bins, 'upper': (b+1)/n_bins,
                         'n': int(mask.sum()), 'mean_probability': float(p[mask].mean()),
                         'observed_rate': float(y[mask].mean())})
    frame = pd.DataFrame(rows)
    ece = float((frame.n * abs(frame.mean_probability-frame.observed_rate)).sum()/len(y))
    return frame, ece
