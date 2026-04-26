"""
Drift detection using Evidently AI.

Compares the feature distribution for a given period against the training
baseline (2017-02).  Writes an HTML report and a JSON sidecar with the
aggregate drift score.

CLI usage:
    python -m src.monitoring.drift --period 2017-04 --threshold 0.15

Exit code 1 if drift_score > threshold (used by Jenkinsfile.monitoring).
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.pipeline.db import get_engine
from src.models.retrain_submit_14 import FEATURE_COLS_V5


BASELINE_PERIOD = "2017-02"


def load_features(period: str) -> pd.DataFrame:
    engine = get_engine()
    query = f"""
        SELECT {", ".join(FEATURE_COLS_V5)}
        FROM features_monthly
        WHERE period = '{period}'
    """
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


def run_drift(
    period: str,
    threshold: float = 0.15,
    output: str | None = None,
) -> float:
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except ImportError:
        print("evidently not installed — skipping drift check", file=sys.stderr)
        return 0.0

    baseline = load_features(BASELINE_PERIOD)
    current  = load_features(period)

    if current.empty:
        print(f"No features found for period {period}", file=sys.stderr)
        return 0.0

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=baseline, current_data=current)

    out_path = Path(output) if output else Path(f"reports/monitoring/drift_{period}.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out_path))

    result = report.as_dict()
    drift_score = result["metrics"][0]["result"]["dataset_drift_share"]

    sidecar = out_path.with_suffix(".json")
    sidecar.write_text(json.dumps({"period": period, "drift_score": drift_score}, indent=2))

    print(f"Drift score for {period}: {drift_score:.4f} (threshold: {threshold})")
    return drift_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--period",    required=True)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--output",    default=None)
    args = parser.parse_args()

    score = run_drift(args.period, args.threshold, args.output)
    sys.exit(1 if score > args.threshold else 0)


if __name__ == "__main__":
    main()
