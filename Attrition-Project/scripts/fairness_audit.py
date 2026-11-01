"""Subgroup audit of the saved held-out predictions (descriptive, not a legal test).

Gap addressed: Gender, MaritalStatus and Age were excluded from the model, but
the report never measured whether predictions or errors differ by these groups.
This script joins the saved outer-CV predictions (every employee, 5 held-out
predictions each) to the source demographics and reports, per group:
flagged rate (prediction >= that fold's threshold), true-positive rate,
false-positive rate, mean predicted probability and observed attrition rate.

Gaps between groups are bootstrap-resampled over EMPLOYEES (not rows), so the
five repeats of one person stay together.

Limits: excluded attributes are not used as model inputs, but other inputs may
proxy them. Small subgroups give wide intervals. Results describe this fictional
data only and do not establish fairness.

Run: python scripts/fairness_audit.py
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import DATA  # noqa: E402

RESULTS = ROOT / 'results'
OUT = RESULTS / 'fairness'
MODELS = ['logistic_no_log', 'logistic_raw', 'forest_no_log', 'forest_raw', 'tree_no_log', 'tree_raw']
ATTRIBUTES = ['Gender', 'MaritalStatus', 'AgeBand']
BOOTSTRAP = 2000


def age_band(age):
    return pd.cut(age, bins=[0, 29, 39, 200], labels=['under 30', '30 to 39', '40 and over'], right=True).astype(str)


def demographics():
    raw = pd.read_csv(DATA)
    out = raw[['EmployeeNumber', 'Gender', 'MaritalStatus', 'Age']].copy()
    out['AgeBand'] = age_band(out.Age)
    return out.set_index('EmployeeNumber')


def group_metrics(frame):
    rows = []
    for (attribute, group), g in frame.groupby(['attribute', 'group']):
        pos = g.y_true == 1
        flagged = g.flag == 1
        rows.append({
            'attribute': attribute, 'group': group,
            'employees': g.employee_id.nunique(), 'predictions': len(g),
            'observed_rate': g.y_true.mean(),
            'mean_predicted': g.probability.mean(),
            'flagged_rate': flagged.mean(),
            'true_positive_rate': flagged[pos].mean() if pos.any() else np.nan,
            'false_positive_rate': flagged[~pos].mean() if (~pos).any() else np.nan,
            'average_precision': average_precision_score(g.y_true, g.probability) if pos.any() else np.nan,
        })
    return pd.DataFrame(rows)


def gaps(table):
    out = {}
    for attribute, t in table.groupby('attribute'):
        out[attribute] = {
            'flagged_rate_gap': float(t.flagged_rate.max() - t.flagged_rate.min()),
            'true_positive_rate_gap': float(t.true_positive_rate.max() - t.true_positive_rate.min()),
            'false_positive_rate_gap': float(t.false_positive_rate.max() - t.false_positive_rate.min()),
            'calibration_in_large_gap': float((t.mean_predicted - t.observed_rate).abs().max()),
        }
    return out


def bootstrap_gaps(frame, rng):
    """Resample employees with replacement; keep each employee's 5 predictions together."""
    employees = frame.employee_id.unique()
    by_employee = {e: g for e, g in frame.groupby('employee_id')}
    draws = []
    for _ in range(BOOTSTRAP):
        sample = rng.choice(employees, size=len(employees), replace=True)
        boot = pd.concat([by_employee[e] for e in sample], ignore_index=True)
        draws.append(gaps(group_metrics(boot)))
    result = {}
    for attribute in ATTRIBUTES:
        result[attribute] = {}
        for metric in ['flagged_rate_gap', 'true_positive_rate_gap', 'false_positive_rate_gap', 'calibration_in_large_gap']:
            values = np.array([d[attribute][metric] for d in draws if attribute in d])
            result[attribute][metric] = [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    preds = pd.read_csv(RESULTS / 'outer_predictions.csv.gz', float_precision='round_trip')
    preds = preds[preds.model.isin(MODELS)].copy()
    demo = demographics()
    preds = preds.join(demo, on='employee_id')
    assert preds[ATTRIBUTES].notna().all().all()
    preds['flag'] = (preds.probability >= preds.threshold).astype(int)

    tables, summaries = [], {}
    rng = np.random.default_rng(42)
    for model in MODELS:
        m = preds[preds.model == model]
        long = pd.concat([m.assign(attribute=a, group=m[a]) for a in ATTRIBUTES], ignore_index=True)
        table = group_metrics(long)
        table.insert(0, 'model', model)
        tables.append(table)
        point = gaps(table)
        if model == 'logistic_no_log':
            summaries[model] = {'point_gaps': point, 'bootstrap_95_percentile': bootstrap_gaps(long, rng)}
        else:
            summaries[model] = {'point_gaps': point}
    result = pd.concat(tables, ignore_index=True)
    result.to_csv(OUT / 'fairness_group_metrics.csv', index=False)
    (OUT / 'fairness_gap_summary.json').write_text(json.dumps({
        'models': summaries,
        'bootstrap_draws': BOOTSTRAP,
        'bootstrap_unit': 'employees (each keeps its 5 held-out predictions)',
        'groups': {'Gender': 'as recorded', 'MaritalStatus': 'as recorded',
                   'AgeBand': 'under 30 / 30 to 39 / 40 and over'},
        'caveat': 'Descriptive subgroup audit on one fictional dataset. Excluded attributes may still proxy through other inputs. Not a fairness certification.',
    }, indent=2, default=float) + '\n', encoding='utf-8')
    with pd.option_context('display.width', 200):
        print(result[result.model == 'logistic_no_log'].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
