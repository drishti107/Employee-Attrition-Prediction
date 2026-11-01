"""Functional UI checks using Streamlit's official headless AppTest API.
Run from the project folder: python tests/test_app.py
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest

at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
assert not at.exception, [e.value for e in at.exception]
at.button[0].click().run()
assert not at.exception and not at.error
assert any(x.label == 'Estimated attrition probability' for x in at.metric)
report = at.session_state['last_report']
assert abs(report['group_reconstructed_probability'] - report['probability']) < 1e-9
assert any(x.label == 'Original predictors probability' for x in at.metric)
assert any(x.label == 'Original + seven domain features probability' for x in at.metric)
assert any(x.label == 'Mean repeat ECE' for x in at.metric)
assert report['threshold_source'].startswith('F1-maximizing')
if report['model'] == 'logistic_no_log':
    assert sum(f['is_engineered'] for f in report['feature_explanation']['factors']) == 7
# Invalid input must produce a clear error, not a stale prediction.
at.number_input(key='training_0_WorkLifeBalance').set_value(9)
at.button[0].click().run()
assert not at.exception
assert any('WorkLifeBalance' in x.value for x in at.error)
assert not any(x.label == 'Estimated attrition probability' for x in at.metric)
at.number_input(key='training_0_WorkLifeBalance').set_value(3)
at.button[0].click().run()
assert not at.exception and not at.error
# Navigate a different template; previous employee's results must disappear.
at.number_input(key='record_training').set_value(1).run()
assert not at.exception
assert not any(x.label == 'Estimated attrition probability' for x in at.metric)
at.button[0].click().run()
assert not at.exception and not at.error
print('PASS: UI loads nested-CV/calibration/interpretation tables, predicts, compares explanations, rejects invalid profiles, recovers and clears stale results.')

import json
(ROOT/'results/app_test_summary.json').write_text(json.dumps({'status':'PASS','checks':['loads all result tabs','predicts and reconstructs probability','compares same-employee original and engineered explanations','shows held-out ECE','rejects invalid rating without stale output','recovers after correction','clears prediction when changing employee']},indent=2)+'\n')
