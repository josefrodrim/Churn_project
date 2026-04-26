"""
Model performance tracking using ground truth labels.

Joins predictions table with ground_truth table for a given period,
computes AUC + LogLoss, and writes a JSON report.

CLI usage:
    python -m src.monitoring.performance --period 2017-03 --min-auc 0.85

Exit code 1 if roc_auc < min_auc (retrain trigger for Jenkinsfile.monitoring).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from src.pipeline.db import get_engine


def load_eval_data(period: str) -> pd.DataFrame:
    engine = get_engine()
    query = f"""
        SELECT p.msno, p.churn_prob, g.is_churn
        FROM predictions p
        JOIN ground_truth g
          ON p.msno = g.msno AND p.period = g.period
        WHERE p.period = '{period}'
          AND p.source = 'batch'
        ORDER BY p.predicted_at DESC
    """
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


def compute_metrics(period: str, min_auc: float = 0.85, output: str | None = None) -> dict:
    df = load_eval_data(period)

    if df.empty:
        print(f"No ground truth available for period {period} — skipping", file=sys.stderr)
        return {"period": period, "roc_auc": None, "log_loss": None, "n_samples": 0}

    y_true  = df["is_churn"].values
    y_prob  = df["churn_prob"].values

    auc = roc_auc_score(y_true, y_prob)
    ll  = log_loss(y_true, y_prob)

    churn_rate = float(np.mean(y_true))
    result = {
        "period":     period,
        "roc_auc":    round(auc, 4),
        "log_loss":   round(ll, 4),
        "churn_rate": round(churn_rate, 4),
        "n_samples":  len(df),
        "alert":      auc < min_auc,
    }

    out_path = Path(output) if output else Path(f"reports/monitoring/performance_{period}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    print(f"Period {period}: AUC={auc:.4f}  LogLoss={ll:.4f}  n={len(df)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--period",  required=True)
    parser.add_argument("--min-auc", type=float, default=0.85)
    parser.add_argument("--output",  default=None)
    args = parser.parse_args()

    result = compute_metrics(args.period, args.min_auc, args.output)

    auc = result.get("roc_auc")
    if auc is not None and auc < args.min_auc:
        print(f"ALERT: AUC {auc:.4f} < threshold {args.min_auc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
