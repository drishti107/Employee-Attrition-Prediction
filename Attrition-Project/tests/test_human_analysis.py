"""Tests for the human-study analysis code. Uses in-memory synthetic frames only;
nothing here is written to results or treated as study data."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from analyze_human_responses import analyse  # noqa: E402


def make_key():
    rows = []
    for pid in ['P01', 'P02']:
        for prof in [0, 100]:
            rows.append({'participant_code': pid, 'profile_row_index': prof, 'display_label': 'A',
                         'representation': 'original'})
            rows.append({'participant_code': pid, 'profile_row_index': prof, 'display_label': 'B',
                         'representation': 'engineered'})
    return pd.DataFrame(rows)


def make_responses(eng_clarity=4, orig_clarity=3):
    rows = []
    for pid in ['P01', 'P02']:
        for prof in [0, 100]:
            for label, clarity in [('A', orig_clarity), ('B', eng_clarity)]:
                rows.append({'participant_code': pid, 'profile_row_index': prof, 'display_label': label,
                             'clarity_1_to_5': clarity, 'usefulness_1_to_5': clarity,
                             'confidence_1_to_5': clarity, 'largest_factor_correct': 1,
                             'decrease_factor_correct': 1, 'causal_claim_correct': 1})
    return pd.DataFrame(rows)


def test_difference_sign_and_size():
    result = analyse(make_responses(eng_clarity=4, orig_clarity=3), make_key())
    assert result['participants'] == 2 and result['ratings'] == 8
    c = result['summary']['clarity_1_to_5']
    assert c['mean_engineered'] == 4 and c['mean_original'] == 3
    assert c['median_difference'] == 1


def test_rejects_out_of_range_ratings():
    responses = make_responses()
    responses.loc[0, 'clarity_1_to_5'] = 9
    try:
        analyse(responses, make_key())
    except ValueError:
        return
    raise AssertionError('out-of-range rating was accepted')


def test_rejects_unmatched_labels():
    responses = make_responses()
    responses.loc[0, 'display_label'] = 'C'
    try:
        analyse(responses, make_key())
    except ValueError:
        return
    raise AssertionError('unmatched label was accepted')


if __name__ == '__main__':
    test_difference_sign_and_size()
    test_rejects_out_of_range_ratings()
    test_rejects_unmatched_labels()
    print('human analysis tests passed')
