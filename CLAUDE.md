# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

University coursework (MTH00051, deadline 2026-05-30) implementing linear regression from scratch in Python — without NumPy in the core math layer. Two parts:

- **Part 1**: OLS theory — solvers, metrics, regularization, diagnostics, cross-validation, Gauss-Markov Monte Carlo demo.
- **Part 2**: Real-world application — data pipeline, multi-model comparison (OLS, Ridge, Lasso), advanced methods on `AirQualityUCI.csv`.

## Setup

```bash
pip install -r requirements.txt
```

No build system. Run modules directly with `python` or execute Jupyter notebooks.

## Architecture

### Layers

```
config.py          — Constants (RANDOM_STATE=42, EPSILON=1e-12) and numerical helpers
utils.py           — Foundation: linear algebra (no NumPy), data generators, assert helpers
test_logger.py     — Colored terminal output for manual test suites
part1/             — Theory implementations (F1–F10 + notebook)
part2/             — Application pipeline (T1–T6 + notebook)
```

### `utils.py` — no NumPy rule

All functions in `utils.py` implement math from scratch using only pure Python lists. **Do not import NumPy or any scientific library here.** The constraint is intentional for educational purposes. The module provides:

- Vector/matrix primitives: `dot`, `matmul`, `matvec`, `transpose`, `inverse`, `norm`, `normalize`
- Data generators: `make_linear_data`, `make_multifeature_data`, `make_collinear_data`
- Assert helpers: `assert_close`, `assert_equal`, `assert_shape`, `assert_in_range`, `assert_raises`

### `config.py` — shared constants

`config.py` imports `matvec`, `vector_sub`, `norm` from `utils.py`. Use `config.RANDOM_STATE` as seed for all random operations and `config.EPSILON` for numerical zero comparisons. Prefer `config.is_zero(x)` and `config.zero_rectify(value)` over ad-hoc comparisons.

### Implementation files

Each file maps to specific feature IDs in `docs/TASKS.md`:
- `part1/ols_implementation.py` → F1–F5
- `part1/ridge_lasso.py` → F6–F7
- `part1/residual_analysis.py` → F8
- `part1/cross_validation.py` → F9
- `part1/gauss_markov_demo.py` → F10
- `part2/data_pipeline.py` → T3
- `part2/model_comparison.py` → T4
- `part2/advanced_methods.py` → T6

`scikit-learn` and `scipy` are allowed **only for verification**, not as the primary implementation. `matplotlib`/`seaborn` are used for plots.

### Testing pattern

No pytest setup. Tests are standalone Python scripts using assert helpers from `utils.py` and `test_logger.TestLogger`. The main test suite is `part1/test_case.py`. Run directly:

```bash
python part1/test_case.py
```

## Implementation Status

### Complete

| File | Features | Notes |
|------|----------|-------|
| `utils.py` | Linear algebra, data gen, asserts | Pure Python, no NumPy |
| `config.py` | Constants + helpers | Imports matvec/vector_sub/norm from utils |
| `test_logger.py` | TestLogger class | Terminal output only |
| `part1/ols_implementation.py` | F1–F5 | ols_fit, hat_matrix, model_metrics, coef_inference, vif |
| `part1/ridge_lasso.py` | F6–F7 | ridge_fit (closed-form), lasso_fit (coordinate descent) |
| `part1/residual_analysis.py` | F8 | residual_diagnostics (4-plot: resid vs fitted, Q-Q, scale-loc, Cook's D) |
| `part1/cross_validation.py` | F9 | kfold_cv, predict, select_lambda_cv |
| `part1/gauss_markov_demo.py` | F10 | run_gauss_markov_simulation (N_SIM=1000, Monte Carlo BLUE demo) |
| `part1/test_case.py` | All F1–F10 | Comprehensive unit tests |
| `part2/data_pipeline.py` | T3 | EDAToolkit + DataPipeline (impute, winsorize, encode, scale, poly) |
| `part2/model_comparison.py` | T4 | ModelComparator: OLS full, OLS selected (p-value/VIF), Ridge, Lasso |

### Stub / Not started

| File | Features | Notes |
|------|----------|-------|
| `part2/advanced_methods.py` | T6 | Empty — Kernel Ridge or Bayesian LR (bonus) |
| `part1/part1_notebook.ipynb` | F11 | Notebook demos for F1–F10 |
| `part2/part2_notebook.ipynb` | T5 | Part 2 analysis notebook |

## Data

- **Dataset:** `part2/data/AirQualityUCI.csv` — Air Quality UCI (hourly sensor readings)
- **Target column:** CO sensor response (continuous regression target)
- **Cached outputs:** `part2/configs/feature_selection.json`, `part2/outputs/ols_selected.json`

## Git Workflow

Current branch: `feat/part1-ols-metrics`

```
main
├── feat/part1-ols-metrics     ← F1, F2, F3
├── feat/part1-inference-gm    ← F4, F5, F10
├── feat/part1-ridge-cv        ← F6, F7, F8, F9
├── feat/part1-notebook        ← F11
├── feat/part2-pipeline        ← T3, compare
├── feat/part2-models          ← T4
├── feat/part2-advanced        ← T6
└── feat/part2-notebook        ← T5
```

Commit format: `[P1/F3] Implement model_metrics with R2, adj-R2, F-test`

## Data Conventions

| Symbol | Type | Shape | Description |
|--------|------|-------|-------------|
| `X` | `list[list[float]]` | `(n, p)` | Features, **no** bias column |
| `X_bias` | `list[list[float]]` | `(n, p+1)` | Design matrix with leading 1s column |
| `y` | `list[float]` | `(n,)` | Target vector |
| `beta_hat` | `list[float]` | `(p+1,)` | Coefficients including intercept |
| `y_hat` | `list[float]` | `(n,)` | Predicted values |
| `lam` | `float` | scalar | Regularization lambda |
| `k` | `int` | scalar | Number of CV folds |

All `part1/` functions accept `X` **without** bias column and add it internally.
