# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a university coursework project (MTH00051, deadline 2026-05-30) implementing linear regression from scratch in Python — without NumPy in the core math layer. It has two parts:

- **Part 1**: OLS theory — implement solvers, metrics, regularization, diagnostics, cross-validation, and a Gauss-Markov Monte Carlo demo.
- **Part 2**: Real-world application — data pipeline, multi-model comparison (OLS, Ridge, Lasso), and analysis on an air quality dataset.

All `part1/` and `part2/` implementation files are currently empty stubs. The full specification lives in [docs/TASKS.md](docs/TASKS.md).

## Setup

```bash
pip install -r requirements.txt
```

No build system. Run modules directly with `python` or execute Jupyter notebooks.

## Architecture

### Layers

```
config.py          — Constants (RANDOM_STATE=42, EPSILON=1e-12) and numerical helpers
utils.py           — Foundation: linear algebra, data generators, assert helpers
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

Use `config.RANDOM_STATE` as the seed for all random operations and `config.EPSILON` for numerical zero comparisons. Prefer `config.is_zero(x)` and `config.zero_rectify(value)` over ad-hoc comparisons.

### Implementation files (part1/, part2/)

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

There is no pytest setup. Tests are written as standalone Python scripts that use the assert helpers from `utils.py` and print results via `test_logger.TestLogger`. Run them directly:

```bash
python <test_script>.py
```
