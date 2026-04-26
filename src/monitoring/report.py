"""
Weekly monitoring HTML report.

Aggregates drift + performance JSON files for the given period and renders
a standalone HTML summary.

CLI usage:
    python -m src.monitoring.report --period 2017-04
"""

import argparse
import json
from datetime import datetime
from pathlib import Path


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Churn Model — Monitoring Report {period}</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; color: #333; }}
    h1 {{ color: #2c3e50; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 16px 0; }}
    .ok {{ color: #27ae60; font-weight: bold; }}
    .alert {{ color: #e74c3c; font-weight: bold; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
    th {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>Churn Model — Monitoring Report</h1>
  <p><strong>Period:</strong> {period} &nbsp;|&nbsp;
     <strong>Generated:</strong> {generated_at}</p>

  <div class="card">
    <h2>Data Drift</h2>
    {drift_section}
  </div>

  <div class="card">
    <h2>Model Performance</h2>
    {perf_section}
  </div>
</body>
</html>"""


def _drift_section(period: str) -> str:
    path = Path(f"reports/monitoring/drift_{period}.json")
    if not path.exists():
        return "<p>No drift data available for this period.</p>"
    data  = json.loads(path.read_text())
    score = data.get("drift_score", "N/A")
    cls   = "alert" if isinstance(score, float) and score > 0.15 else "ok"
    return f"""
    <table>
      <tr><th>Metric</th><th>Value</th><th>Threshold</th><th>Status</th></tr>
      <tr>
        <td>Dataset drift share</td>
        <td>{score if score == 'N/A' else f'{score:.4f}'}</td>
        <td>0.15</td>
        <td class="{cls}">{'ALERT' if cls == 'alert' else 'OK'}</td>
      </tr>
    </table>
    <p><a href="drift_{period}.html">Full Evidently drift report →</a></p>
    """


def _perf_section(period: str) -> str:
    path = Path(f"reports/monitoring/performance_{period}.json")
    if not path.exists():
        return "<p>No performance data available (ground truth not yet received).</p>"
    d   = json.loads(path.read_text())
    auc = d.get("roc_auc", "N/A")
    ll  = d.get("log_loss", "N/A")
    n   = d.get("n_samples", 0)
    cls = "alert" if isinstance(auc, float) and auc < 0.85 else "ok"
    return f"""
    <table>
      <tr><th>Metric</th><th>Value</th><th>Threshold</th><th>Status</th></tr>
      <tr>
        <td>ROC-AUC</td>
        <td>{auc if auc == 'N/A' else f'{auc:.4f}'}</td>
        <td>&ge; 0.85</td>
        <td class="{cls}">{'ALERT' if cls == 'alert' else 'OK'}</td>
      </tr>
      <tr>
        <td>Log Loss</td>
        <td>{ll if ll == 'N/A' else f'{ll:.4f}'}</td>
        <td>&le; 0.240</td>
        <td>—</td>
      </tr>
      <tr><td>Samples evaluated</td><td>{n}</td><td>—</td><td>—</td></tr>
    </table>
    """


def generate_report(period: str, output: str | None = None) -> Path:
    out_path = Path(output) if output else Path(f"reports/monitoring/weekly_{period}.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    html = TEMPLATE.format(
        period       = period,
        generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        drift_section = _drift_section(period),
        perf_section  = _perf_section(period),
    )
    out_path.write_text(html)
    print(f"Report written to {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    generate_report(args.period, args.output)


if __name__ == "__main__":
    main()
