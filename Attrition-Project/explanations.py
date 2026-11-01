"""One reporting path for the app, CLI and exported individual explanations.

No rules invent reasons for leaving: every direction comes from the fitted model.
"""
import re
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from attrition import BLOCKS, GROUPS, grouped_shapley, linear_contributions, validate_input

FEATURE_GUIDE = [
    ('Workload', 'OvertimeBalancePressure', 'I(OverTime = Yes) × (5 − WorkLifeBalance)',
     'Overtime together with poorer reported work-life balance; a proxy, not a burnout diagnosis.'),
    ('Workload', 'OvertimeCommute', 'I(OverTime = Yes) × DistanceFromHome',
     'Commute distance in the context of overtime. The source does not establish distance units.'),
    ('Career', 'PromotionDelayShare', 'YearsSinceLastPromotion / (YearsAtCompany + 1)',
     'Promotion delay relative to company tenure. The added year prevents division by zero.'),
    ('Career', 'RoleTenureShare', 'YearsInCurrentRole / (YearsAtCompany + 1)',
     'Relative time in the current role; it need not imply stagnation.'),
    ('Career', 'ManagerTenureShare', 'YearsWithCurrManager / (YearsAtCompany + 1)',
     'Manager continuity relative to tenure; it does not measure manager quality.'),
    ('Reward', 'IncomeToRoleMedian', 'MonthlyIncome / training role median income',
     'Income relative to the same broad job role. The median is fitted inside each training fold.'),
    ('Reward', 'LogMonthlyIncome', 'log(1 + MonthlyIncome)',
     'A diminishing-scale representation of income, not new information.'),
    ('Sentiment', 'MeanDissatisfaction', '5 − mean(job, environment, relationship satisfaction)',
     'Equal-weight summary of three ratings; not a validated psychological scale.'),
]


def display_name(name):
    special = {'OverTime': 'Overtime', 'YearsWithCurrManager': 'Years with current manager',
               'NumCompaniesWorked': 'Number of companies worked',
               'IncomeToRoleMedian': 'Income relative to role median'}
    return special.get(name, re.sub(r'(?<!^)(?=[A-Z])', ' ', name).lower().capitalize())


def scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def make_report(model, row, model_name):
    """Return JSON-safe probability, label, local factors and reconstruction checks."""
    row = validate_input(row, model.reference_X)
    if len(row) != 1:
        raise ValueError('Select exactly one employee for an individual explanation.')
    report = grouped_shapley(model, row)
    report.update({
        'model': model_name,
        'threshold': float(model.threshold),
        'predicted_class': int(report['probability'] >= model.threshold),
        'predicted_label': 'Attrition' if report['probability'] >= model.threshold else 'Stay',
        'probability_meaning': 'Estimated probability of the dataset Attrition=Yes label; no future time horizon is available.',
        'reference_meaning': 'Mean model prediction over 32 fixed training background employees.',
        'explanation_method': 'Exact six-group Shapley enumeration for the fixed empirical interventional background.',
        'interpretation': 'Model associations relative to a reference, not causes or guaranteed effects of changing a factor.',
        'threshold_source': 'F1-maximizing threshold on training-only cross-validated calibrated scores.',
        'inputs_used': {c: scalar(row.iloc[0][c]) for cols in GROUPS.values() for c in cols},
    })
    factors = report['factors']
    report['principal_increasing_groups'] = [f for f in factors if f['contribution_probability_points'] > 0][:3]
    report['principal_decreasing_groups'] = [f for f in factors if f['contribution_probability_points'] < 0][:3]
    report['group_reconstructed_probability'] = report['reference_probability'] + sum(
        f['contribution_probability_points'] for f in factors) / 100
    report['group_additivity_error'] = abs(report['group_reconstructed_probability'] - report['probability'])
    if isinstance(model.pipeline.named_steps['model'], LogisticRegression):
        base, contributions = linear_contributions(model, row)
        domain = model.pipeline.named_steps['domain'].transform(row).iloc[0]
        engineered = set(sum(BLOCKS.values(), []))
        details = []
        definitions = {name: (formula, rationale) for block, name, formula, rationale in FEATURE_GUIDE}
        for name, amount in contributions.iloc[0].sort_values(key=abs, ascending=False).items():
            details.append({'feature': name, 'display_name': display_name(name),
                            'value': scalar(domain[name]), 'is_engineered': name in engineered,
                            'contribution_calibrated_log_odds': float(amount),
                            'direction': 'raises' if amount > 0 else ('lowers' if amount < 0 else 'neutral'),
                            'definition': definitions.get(name, ('Original input attribute.', ''))[0],
                            'interpretation_note': definitions.get(name, ('', 'Model association; not a causal effect.'))[1]})
        report['feature_explanation'] = {'baseline_calibrated_log_odds': float(base),
            'baseline_probability': float(expit(base)), 'factors': details,
            'reconstructed_probability': float(expit(base + contributions.iloc[0].sum())),
            'units': 'calibrated log-odds; these are not probability percentage points',
            'reference': 'Mean transformed training feature vector; different from the Shapley background mean.'}
    return report


def compare_reports(raw_model, engineered_model, row):
    """Same submitted employee, identical reporting methods in both arms."""
    raw = make_report(raw_model, row, 'logistic_raw')
    engineered = make_report(engineered_model, row, 'logistic_no_log')
    return {'original': raw, 'engineered': engineered,
            'note': 'Model comparison, not a treatment effect. Each fitted model has its own training reference.'}
