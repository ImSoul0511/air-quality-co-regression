import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import RANDOM_STATE


def _validate_data(X: list[list[float]], y: list[float]) -> None:
    if len(X) == 0:
        raise ValueError("X must not be empty")

    if len(y) == 0:
        raise ValueError("y must not be empty")

    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows")

    if len(X[0]) == 0:
        raise ValueError("X must contain at least one feature")

    p = len(X[0])
    for row in X:
        if len(row) != p:
            raise ValueError("All rows in X must have the same number of columns")


def _take_rows(X: list[list[float]], indices) -> list[list[float]]:
    return [X[int(i)][:] for i in indices]


def _take_values(y: list[float], indices) -> list[float]:
    return [y[int(i)] for i in indices]


def _standardize_row(
    row: list[float],
    mean_X: list[float],
    std_X: list[float],
) -> list[float]:
    return [(row[j] - mean_X[j]) / std_X[j] for j in range(len(row))]


def predict(X: list[list[float]], fit_result: dict) -> list[float]:
    """
    Predict y for X using a model fit result.

    Supported conventions:
        - beta_hat on original feature scale: [intercept, beta_1, ...]
        - beta_scaled with mean_X/std_X: [intercept, beta_1, ...] in scaled space
    """
    if len(X) == 0:
        return []

    if "beta_scaled" in fit_result:
        beta = fit_result["beta_scaled"]
        mean_X = fit_result["mean_X"]
        std_X = fit_result["std_X"]

        y_pred = []
        for row in X:
            row_scaled = _standardize_row(row, mean_X, std_X)
            prediction = beta[0]
            for j in range(len(row_scaled)):
                prediction += beta[j + 1] * row_scaled[j]
            y_pred.append(prediction)
        return y_pred

    if "beta_hat" not in fit_result:
        raise ValueError("fit_result must contain beta_hat or beta_scaled")

    beta = fit_result["beta_hat"]
    y_pred = []
    for row in X:
        prediction = beta[0]
        for j in range(len(row)):
            prediction += beta[j + 1] * row[j]
        y_pred.append(prediction)
    return y_pred


def _mse(y_true: list[float], y_pred: list[float]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    total = 0.0
    for i in range(len(y_true)):
        diff = y_true[i] - y_pred[i]
        total += diff * diff
    return total / len(y_true)


def kfold_cv(
    X: list[list[float]],
    y: list[float],
    k: int,
    model_fn,
    **model_kwargs,
) -> dict:
    """
    Run k-Fold Cross-Validation from scratch.

    Args:
        X: feature matrix, shape (n, p)
        y: target vector, shape (n,)
        k: number of folds
        model_fn: callable model_fn(X_train, y_train, **kwargs) -> dict
        **model_kwargs: parameters passed to model_fn, e.g. lam=0.1

    Returns:
        dict with cv_scores, mean_cv_score, std_cv_score.
    """
    import numpy as np

    _validate_data(X, y)

    if not callable(model_fn):
        raise ValueError("model_fn must be callable")

    n = len(X)
    if k < 2:
        raise ValueError("k must be >= 2")

    if k > n:
        raise ValueError("k must be <= number of samples")

    indices = np.arange(n)
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(indices)
    folds = np.array_split(indices, k)

    cv_scores = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        X_train = _take_rows(X, train_idx)
        y_train = _take_values(y, train_idx)
        X_val = _take_rows(X, val_idx)
        y_val = _take_values(y, val_idx)

        result = model_fn(X_train, y_train, **model_kwargs)
        y_pred_val = predict(X_val, result)
        cv_scores.append(_mse(y_val, y_pred_val))

    return {
        "cv_scores": [float(score) for score in cv_scores],
        "mean_cv_score": float(np.mean(cv_scores)),
        "std_cv_score": float(np.std(cv_scores)),
    }


def lambda_grid_logspace(
    start_exp: float = -3,
    stop_exp: float = 3,
    num: int = 50,
) -> list[float]:
    """Return logspace(start_exp, stop_exp, num) as a list."""
    if num <= 0:
        raise ValueError("num must be > 0")

    if num == 1:
        return [10.0**start_exp]

    step = (stop_exp - start_exp) / (num - 1)
    return [10.0 ** (start_exp + i * step) for i in range(num)]


def select_lambda_cv(
    X: list[list[float]],
    y: list[float],
    k: int,
    model_fn,
    lambda_grid: list[float] | None = None,
    show: bool = True,
    **model_kwargs,
) -> dict:
    """
    Select the best lambda by k-Fold CV and plot lambda vs mean CV MSE.
    """
    import numpy as np

    if not show:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    if lambda_grid is None:
        lambda_grid = lambda_grid_logspace(-3, 3, 50)

    if len(lambda_grid) == 0:
        raise ValueError("lambda_grid must not be empty")

    cv_means = []
    cv_stds = []
    cv_results = []

    for lam in lambda_grid:
        result = kfold_cv(X, y, k=k, model_fn=model_fn, lam=lam, **model_kwargs)
        cv_results.append(result)
        cv_means.append(result["mean_cv_score"])
        cv_stds.append(result["std_cv_score"])

    best_idx = int(np.argmin(cv_means))
    lambda_opt = lambda_grid[best_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lambda_grid, cv_means, marker="o", label="Mean CV MSE")
    ax.axvline(
        lambda_opt,
        color="red",
        linestyle="--",
        label=f"Best lambda = {lambda_opt:.6g}",
    )
    ax.set_xscale("log")
    ax.set_title("Lambda Selection by k-Fold Cross-Validation", fontsize=13)
    ax.set_xlabel("lambda (log scale)")
    ax.set_ylabel("Mean validation MSE")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    if show:
        plt.show()

    return {
        "lambda_grid": lambda_grid,
        "cv_means": [float(value) for value in cv_means],
        "cv_stds": [float(value) for value in cv_stds],
        "cv_results": cv_results,
        "lambda_opt": float(lambda_opt),
        "best_cv_score": float(cv_means[best_idx]),
        "fig": fig,
    }


def run_cv_tests() -> tuple[int, int]:
    from part1.test_case import run_cv_test_cases

    return run_cv_test_cases()
