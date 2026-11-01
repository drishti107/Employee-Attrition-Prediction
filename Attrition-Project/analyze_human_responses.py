"""Analyse the human clarity responses, descriptively, after unblinding.

Input: human_study/responses.csv (researcher fills this from paper or forms; one row per
explanation rated). Required columns:
  participant_code, profile_row_index, display_label,
  clarity_1_to_5, usefulness_1_to_5, confidence_1_to_5,
  largest_factor_correct (0/1), decrease_factor_correct (0/1), causal_claim_correct (0/1)

The representation (original vs engineered) is joined from KEY_do_not_distribute.csv.

If responses.csv is missing or empty the script writes status 'not performed' and
reports nothing else. It never fabricates ratings.

Run: python scripts/analyze_human_responses.py
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'human_study'
RESULTS = ROOT / 'results' / 'human_study'
RATINGS = ['clarity_1_to_5', 'usefulness_1_to_5', 'confidence_1_to_5']
ITEMS = ['largest_factor_correct', 'decrease_factor_correct', 'causal_claim_correct']
REQUIRED = ['participant_code', 'profile_row_index', 'display_label'] + RATINGS + ITEMS
BOOTSTRAP = 2000


def write_status(status, **extra):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / 'status.json').write_text(json.dumps({'status': status, **extra}, indent=2) + '\n', encoding='utf-8')


def analyse(responses, key):
    merged = responses.merge(key, on=['participant_code', 'profile_row_index', 'display_label'], how='left', validate='many_to_one')
    if merged.representation.isna().any():
        raise ValueError('Responses do not match the key for some rows; check labels.')
    for col in RATINGS:
        if not merged[col].dropna().between(1, 5).all():
            raise ValueError(f'{col} outside 1..5')
    per_person = merged.groupby(['participant_code', 'representation'])[RATINGS + ITEMS].mean().unstack('representation')
    people = per_person.index.tolist()
    result = {'status': 'completed', 'participants': len(people), 'ratings': int(len(merged))}
    rng = np.random.default_rng(42)
    summary = {}
    for col in RATINGS + ITEMS:
        eng = per_person[(col, 'engineered')]
        orig = per_person[(col, 'original')]
        diff = (eng - orig).dropna()
        if len(diff) == 0:
            continue
        boots = [diff.iloc[rng.integers(0, len(diff), len(diff))].mean() for _ in range(BOOTSTRAP)]
        summary[col] = {
            'mean_original': float(orig.mean()), 'mean_engineered': float(eng.mean()),
            'median_difference': float(diff.median()), 'mean_difference': float(diff.mean()),
            'iqr_difference': [float(diff.quantile(.25)), float(diff.quantile(.75))],
            'bootstrap_95_by_participant': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            'participant_differences': diff.round(3).tolist(),
        }
    result['summary'] = summary
    result['caveat'] = ('Descriptive only. With few participants, intervals are unstable. Bootstrap resamples '
                        'participants as clusters. Ratings do not establish population-level understanding.')
    return result


def main():
    path = STUDY / 'responses.csv'
    key_path = STUDY / 'KEY_do_not_distribute.csv'
    if not path.exists() or path.stat().st_size == 0:
        write_status('not performed', note='No responses.csv. No ratings exist. Nothing was analysed.')
        print('Human study: not performed (no responses).')
        return
    responses = pd.read_csv(path)
    if responses.empty:
        write_status('not performed', note='responses.csv has a header only.')
        print('Human study: not performed (empty responses).')
        return
    missing = [c for c in REQUIRED if c not in responses.columns]
    if missing:
        raise ValueError(f'responses.csv missing columns: {missing}')
    result = analyse(responses, pd.read_csv(key_path))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / 'analysis.json').write_text(json.dumps(result, indent=2, default=float) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2, default=float)[:2000])


if __name__ == '__main__':
    main()
