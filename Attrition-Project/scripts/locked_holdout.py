"""Locked holdout test: choose the model on development rows only, then score a
held-out block ONCE.

Gap addressed: the original protocol selected the deployed family among outer-CV
results that also produced its headline score, so that score is optimistic.
Here 294 employees (20%) are split off with a fixed, pre-stated seed before any
model is compared. The remaining 1,176 development rows run the same nested
selection (5x5 repeated CV, same tuning/calibration/threshold rules, same
winner rule as protocol.json, restricted to raw and no_log). The winner is
refitted on all development rows and scored on the holdout exactly once.

Limitation: this is an untouched split of the SAME fictional dataset, not an
external or temporal test. It removes selection optimism for the holdout
number only.

Run: python scripts/locked_holdout.py --jobs 6
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from attrition import SEED  # noqa: E402
from train import candidates, fit_model, load_data, metrics, save_json  # noqa: E402

OUT = ROOT / 'results' / 'holdout'
HOLDOUT_SEED = 20260101          # fixed before running; recorded in holdout_split.json
HOLDOUT_SIZE = 294               # 20% of 1,470, same size as one outer test fold
SPECS = [(family, mode) for family in candidates() for mode in ['raw', 'no_log']]
KEYS = [f'{family}_{mode}' for family, mode in SPECS]


def make_split():
    _, X, y = load_data()
    dev, hold = train_test_split(np.arange(len(y)), test_size=HOLDOUT_SIZE,
                                 stratify=y, random_state=HOLDOUT_SEED)
    dev, hold = np.sort(dev), np.sort(hold)
    assert len(dev) == 1176 and len(hold) == 294 and not set(dev) & set(hold)
    return X, y, dev, hold


def dev_fold_unit(key, family, mode, fold_id, tr, te, X, y):
    with threadpool_limits(limits=1):
        model, search = fit_model(family, mode, X.iloc[tr], y.iloc[tr], 1000 + fold_id * 10)
        p = model.predict_proba(X.iloc[te])[:, 1]
    row = {'model': key, 'fold_id': fold_id, 'repeat': fold_id // 5, 'fold': fold_id % 5,
           **metrics(y.iloc[te], p, model.threshold)}
    return row


def fit_and_score_holdout(key, family, mode, X_dev, y_dev, X_hold, y_hold):
    with threadpool_limits(limits=1):
        model, search = fit_model(family, mode, X_dev, y_dev, 5000)
        p = model.predict_proba(X_hold)[:, 1]
    return {'model': key, 'threshold': model.threshold, 'best_params': search.best_params_,
            'probabilities': p, **metrics(y_hold, p, model.threshold)}


def main(jobs):
    OUT.mkdir(parents=True, exist_ok=True)
    X, y, dev, hold = make_split()
    save_json(OUT / 'holdout_split.json', {
        'seed': HOLDOUT_SEED, 'holdout_size': HOLDOUT_SIZE, 'development_rows': len(dev),
        'holdout_rows': len(hold), 'holdout_employee_ids': X.EmployeeNumber.iloc[hold].tolist()
        if 'EmployeeNumber' in X else None, 'holdout_row_indices': hold.tolist(),
        'rule': 'stratified train_test_split on Attrition, fixed seed, performed before any model comparison',
    })
    Xd, yd = X.iloc[dev].reset_index(drop=True), y.iloc[dev].reset_index(drop=True)
    Xh, yh = X.iloc[hold].reset_index(drop=True), y.iloc[hold].reset_index(drop=True)

    # 1) nested CV on development rows only (same design as protocol.json, smaller n)
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(Xd, yd))
    units = [(family, mode, f'{family}_{mode}', i, tr, te)
             for i, (tr, te) in enumerate(splits) for family, mode in SPECS]
    with parallel_config(backend='multiprocessing'):
        rows = Parallel(n_jobs=jobs)(delayed(dev_fold_unit)(key, fam, mode, i, tr, te, Xd, yd)
                                     for fam, mode, key, i, tr, te in units)
    dev_cv = pd.DataFrame(rows)
    dev_cv.to_csv(OUT / 'holdout_dev_cv_folds.csv', index=False)
    summary = dev_cv.groupby('model')[['average_precision', 'roc_auc', 'brier', 'f1']].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(OUT / 'holdout_dev_cv_summary.csv', index=False)

    # 2) same winner rule as protocol.json, applied to development CV only
    eligible = summary.reset_index().sort_values(
        ['average_precision_mean', 'brier_mean', 'model'], ascending=[False, True, True])
    selected = eligible.iloc[0].model
    save_json(OUT / 'holdout_selection.json', {
        'selected': selected, 'rule': 'highest mean dev-CV AP; lower Brier, then key, breaks ties',
        'dev_cv_ranking': eligible.model.tolist()})

    # 3) refit every candidate on ALL development rows; score holdout once each
    finals = Parallel(n_jobs=min(jobs, len(KEYS)))(
        delayed(fit_and_score_holdout)(key, fam, mode, Xd, yd, Xh, yh) for fam, mode, key in
        [(f, m, f'{f}_{m}') for f, m in SPECS])
    probs = {f['model']: f.pop('probabilities') for f in finals}
    table = pd.DataFrame(finals)
    table['selected'] = table.model == selected
    table.to_csv(OUT / 'holdout_test_metrics.csv', index=False)

    # 4) paired bootstrap of holdout AP: selected minus the original-feature logistic
    rng = np.random.default_rng(SEED)
    reference = 'logistic_raw'
    diffs = []
    for _ in range(2000):
        idx = rng.integers(0, len(yh), len(yh))
        diffs.append(average_precision_score(yh.iloc[idx], probs[selected][idx])
                     - average_precision_score(yh.iloc[idx], probs[reference][idx]))
    diffs = np.asarray(diffs)
    observed = (average_precision_score(yh, probs[selected]) - average_precision_score(yh, probs[reference]))
    save_json(OUT / 'holdout_paired_ap.json', {
        'selected': selected, 'reference': reference, 'difference_observed': float(observed),
        'bootstrap_95_percentile_interval': [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        'bootstrap_draws': 2000, 'unit': 'holdout employees (294), not independent of the same data source',
        'caveat': 'Single 294-row holdout from one fictional dataset. Interval reflects sampling of this block only.'})
    print(table[['model', 'average_precision', 'roc_auc', 'brier', 'f1', 'selected']].to_string(index=False))
    print('selected:', selected, '| paired AP diff vs', reference, round(float(observed), 4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type=int, default=4)
    main(parser.parse_args().jobs)
