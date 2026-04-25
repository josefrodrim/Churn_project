# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Customer churn prediction project using tabular ML. The stack includes scikit-learn, XGBoost, LightGBM, imbalanced-learn (for class imbalance), and SHAP (for explainability). Work happens in Jupyter notebooks (`notebooks/`) and modular Python source code (`src/`).

## Setup

```bash
pip install -r requirements.txt
jupyter lab
```

## Project Structure

```
data/
  raw/          # Original, immutable data — never edit these files
  processed/    # Cleaned/transformed data ready for modeling
  external/     # Third-party or supplementary data
models/         # Serialized trained models
notebooks/      # Exploratory analysis and experiments
reports/
  figures/      # Generated plots for reports
src/
  features/     # Feature engineering code
  models/       # Training, evaluation, and prediction logic
  utils/        # Shared helpers (I/O, logging, etc.)
  visualization/# Plotting utilities
```

## Conventions

- Raw data under `data/raw/` is read-only; all transformations produce outputs in `data/processed/`.
- Trained model artifacts go in `models/`.
- Notebook outputs that become figures for reports go in `reports/figures/`.
- Source modules in `src/` should be importable from notebooks via `sys.path` or a package install.
