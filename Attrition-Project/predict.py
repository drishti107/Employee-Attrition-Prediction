"""CLI prediction with the same explanations as the Streamlit demo."""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from explanations import make_report


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True, help='CSV with the original predictor columns')
    parser.add_argument('--row', type=int, default=0, help='Zero-based row index')
    parser.add_argument('--output', help='Optional path for the explanation JSON')
    args = parser.parse_args()
    bundle = joblib.load(root / 'models/bundle.joblib')
    model = bundle['models'][bundle['selected']]
    data = pd.read_csv(args.csv)
    if not 0 <= args.row < len(data):
        parser.error('Row index outside CSV.')
    try:
        report = make_report(model, data.iloc[[args.row]], bundle['selected'])
    except ValueError as exc:
        parser.error(str(exc))
    result = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result, encoding='utf-8')
    print(result)


if __name__ == '__main__':
    main()
