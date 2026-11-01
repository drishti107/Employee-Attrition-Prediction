"""Build the blinded human clarity study packet from docs/HUMAN_EVALUATION_RUBRIC.md.

Gap addressed: the human comprehension study had a protocol but no runnable
materials. This script creates them. It does NOT create any ratings. No human
responses exist in this project and none are generated here.

Outputs (under human_study/):
  cards/P01.md ... P10.md   participant-facing sheets: 5 profiles x A/B, no model names
  participant_assignments.csv  presentation order per participant (original-first or engineered-first)
  KEY_do_not_distribute.csv   maps each display label back to the model; researcher only
  profiles.json               chosen profile rows and their explanations (for audit)

Profiles: fixed source row indices 0, 100, 200, 300, 400 from the rubric, chosen before
any rating. Profiles are training employees; explanations come from the final
full-data models in models/bundle.joblib, so they show how the app explains a
record, not held-out accuracy.

Run: python scripts/human_study_kit.py
"""
import json, sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from explanations import compare_reports  # noqa: E402

OUT = ROOT / 'human_study'
PROFILE_ROWS = [0, 100, 200, 300, 400]           # fixed in docs/HUMAN_EVALUATION_RUBRIC.md
PARTICIPANTS = [f'P{i:02d}' for i in range(1, 11)]
SHOWN_INPUTS = ['OverTime', 'WorkLifeBalance', 'JobSatisfaction', 'EnvironmentSatisfaction',
                'RelationshipSatisfaction', 'MonthlyIncome', 'YearsAtCompany', 'JobRole', 'DistanceFromHome']
TOP_N = 5


def load_profiles():
    X = pd.read_csv(ROOT / 'data' / 'training_input_template.csv')
    bundle = joblib.load(ROOT / 'models' / 'bundle.joblib')
    raw, eng = bundle['models']['logistic_raw'], bundle['models']['logistic_no_log']
    profiles = []
    for r in PROFILE_ROWS:
        row = X.iloc[[r]].reset_index(drop=True)
        rep = compare_reports(raw, eng, row)
        profiles.append({'row_index': r, 'inputs': {c: row.iloc[0][c].item() if hasattr(row.iloc[0][c], 'item') else row.iloc[0][c] for c in SHOWN_INPUTS},
                         'original': rep['original'], 'engineered': rep['engineered']})
    return profiles


def factor_lines(report):
    factors = report['feature_explanation']['factors'][:TOP_N]
    lines = []
    for f in factors:
        arrow = 'raises' if f['direction'] == 'raises' else 'lowers'
        value = f['value'] if isinstance(f['value'], str) else f"{float(f['value']):.3g}"
        lines.append(f"- **{f['display_name']}** (value {value}) {arrow} the score by {abs(f['contribution_calibrated_log_odds']):.3f} log-odds")
    return '\n'.join(lines)


def card(participant, presentation_order, profiles):
    lines = [f'# Explanation task, participant {participant}', '',
             'Read each explanation. It describes how a model links inputs to a prediction for one fictional employee. '
             'It does not prove that changing a factor would change attrition. Rate each explanation on the response sheet.', '']
    for p in profiles:
        first, second = (('original', 'engineered') if presentation_order == 'original_first' else ('engineered', 'original'))
        lines += [f"## Employee profile {p['row_index']}", '',
                  'Inputs: ' + '; '.join(f'{k} = {v}' for k, v in p['inputs'].items()), '']
        for label, rep_key in zip(['A', 'B'], [first, second]):
            rep = p[rep_key]
            lines += [f"### Explanation {label}", '',
                      f"Estimated probability of the dataset Attrition = Yes label: {rep['probability']*100:.1f}%.",
                      '', f'Largest contributions (units: calibrated log-odds, not percentage points):', '',
                      factor_lines(rep), '',
                      'Note: contributions are relative to a reference profile and describe model associations only.', '']
    return '\n'.join(lines)


def main():
    (OUT / 'cards').mkdir(parents=True, exist_ok=True)
    profiles = load_profiles()
    assignments, key = [], []
    for i, pid in enumerate(PARTICIPANTS):
        order = 'original_first' if i % 2 == 0 else 'engineered_first'   # 5 and 5, deterministic
        for p in profiles:
            first, second = (('original', 'engineered') if order == 'original_first' else ('engineered', 'original'))
            for label, rep in zip(['A', 'B'], [first, second]):
                assignments.append({'participant_code': pid, 'presentation_order': order,
                                    'profile_row_index': p['row_index'], 'display_label': label})
                key.append({'participant_code': pid, 'profile_row_index': p['row_index'],
                            'display_label': label, 'representation': rep})
        (OUT / 'cards' / f'{pid}.md').write_text(card(pid, order, profiles), encoding='utf-8')
    pd.DataFrame(assignments).to_csv(OUT / 'participant_assignments.csv', index=False)
    pd.DataFrame(key).to_csv(OUT / 'KEY_do_not_distribute.csv', index=False)
    audit = [{k: v for k, v in p.items()} for p in profiles]
    (OUT / 'profiles.json').write_text(json.dumps(audit, indent=2, default=float) + '\n', encoding='utf-8')
    print('Wrote', len(PARTICIPANTS), 'participant cards. Ratings: none collected.')


if __name__ == '__main__':
    main()
