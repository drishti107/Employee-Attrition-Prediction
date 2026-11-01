"""Forward-in-time validation, ready for DATED attrition data.

Gap addressed: the supplied dataset has no observation dates or departure dates, so
a temporal test cannot be run on it. This script is the test to run once a
dated extract exists. It is NOT evidence on the IBM file.

Input CSV must contain:
  EmployeeNumber, observation_date (YYYY-MM-DD), the 35 original HR columns
  (Attrition kept as in the source), and the label describing the same observation.
Each row is one employee snapshot. The script:
  1. sorts by observation_date;
  2. trains on the earliest --train-fraction of rows and tests on the latest rows;
  3. checks no employee appears on both sides of the cut;
  4. applies the same candidate models, tuning, calibration and threshold rules as train.py.

State the outcome horizon (e.g. "left within 12 months of snapshot") in --horizon.
That horizon is recorded in the output; without it the probability is not a
time-specific turnover forecast.

Run: python scripts/temporal_validation.py --csv path/to/dated.csv --horizon "12 months" --train-fraction 0.7
"""
import argparse, sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import candidates, fit_model, metrics, save_json  # noqa: E402

OUT = ROOT / 'results' / 'temporal'
MODES = ['raw', 'no_log']


def run(csv, horizon, train_fraction, out=OUT):
    d = pd.read_csv(csv)
    for col in ['EmployeeNumber', 'observation_date', 'Attrition']:
        if col not in d.columns:
            raise ValueError(f'missing required column: {col}')
    d['observation_date'] = pd.to_datetime(d.observation_date, errors='raise')
    d = d.sort_values(['observation_date', 'EmployeeNumber']).reset_index(drop=True)
    cut = int(len(d) * train_fraction)
    train, test = d.iloc[:cut], d.iloc[cut:]
    overlap = set(train.EmployeeNumber) & set(test.EmployeeNumber)
    y_train = train.Attrition.map({'No': 0, 'Yes': 1})
    y_test = test.Attrition.map({'No': 0, 'Yes': 1})
    X_train = train.drop(columns=['Attrition', 'observation_date'])
    X_test = test.drop(columns=['Attrition', 'observation_date'])
    rows = []
    for family in candidates():
        for mode in MODES:
            model, _ = fit_model(family, mode, X_train, y_train, 42)
            p = model.predict_proba(X_test)[:, 1]
            rows.append({'model': f'{family}_{mode}', 'n_train': len(train), 'n_test': len(test),
                         'test_positive_rate': float(y_test.mean()), 'employees_on_both_sides': len(overlap),
                         'horizon': horizon, **metrics(y_test, p, model.threshold)})
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / 'temporal_metrics.csv', index=False)
    save_json(out / 'temporal_run.json', {
        'csv': str(csv), 'horizon': horizon, 'train_fraction': train_fraction,
        'train_date_range': [str(train.observation_date.min().date()), str(train.observation_date.max().date())],
        'test_date_range': [str(test.observation_date.min().date()), str(test.observation_date.max().date())],
        'employees_on_both_sides': len(overlap),
        'note': 'Single forward split. Thresholds and calibration are fitted on the training side only.'})
    return table


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--horizon', required=True)
    p.add_argument('--train-fraction', type=float, default=0.7)
    a = p.parse_args()
    print(run(a.csv, a.horizon, a.train_fraction).round(4).to_string(index=False))
