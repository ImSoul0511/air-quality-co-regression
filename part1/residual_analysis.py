import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import is_zero
from utils import inverse, matmul, transpose


def _validate_vector(name: str, values: list[float]) -> None:
    if len(values) == 0:
        raise ValueError(f"{name} must not be empty")


def _validate_inputs(
    y: list[float],
    y_hat: list[float],
    X: list[list[float]] | None,
) -> None:
    _validate_vector("y", y)
    _validate_vector("y_hat", y_hat)

    if len(y) != len(y_hat):
        raise ValueError("y and y_hat must have the same length")

    if X is None:
        return

    if len(X) != len(y):
        raise ValueError("X must have the same number of rows as y")

    if len(X) == 0 or len(X[0]) == 0:
        raise ValueError("X must contain at least one feature")

    p = len(X[0])
    for row in X:
        if len(row) != p:
            raise ValueError("All rows in X must have the same number of columns")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    mu = _mean(values)
    variance = sum((value - mu) ** 2 for value in values) / len(values)
    return variance**0.5


def _add_bias_column(X: list[list[float]]) -> list[list[float]]:
    return [[1.0] + row[:] for row in X]


def hat_matrix(X: list[list[float]]) -> dict:
    """
    Compute the OLS hat matrix H = X(X^T X)^(-1)X^T.

    X should not include the intercept column. This function adds it.
    """
    if len(X) == 0 or len(X[0]) == 0:
        raise ValueError("X must contain at least one feature")

    p = len(X[0])
    for row in X:
        if len(row) != p:
            raise ValueError("All rows in X must have the same number of columns")

    X_bias = _add_bias_column(X)
    Xt = transpose(X_bias)
    XtX = matmul(Xt, X_bias)
    XtX_inv = inverse(XtX)
    H = matmul(matmul(X_bias, XtX_inv), Xt)

    return {
        "H": H,
        "leverage": [H[i][i] for i in range(len(H))],
        "X_bias": X_bias,
    }


def _normal_quantiles(n: int) -> list[float]:
    from scipy.stats import norm

    return [float(norm.ppf((i - 0.5) / n)) for i in range(1, n + 1)]


def _compute_cooks_distance(
    residuals: list[float],
    X: list[list[float]] | None,
) -> dict:
    n = len(residuals)

    if X is None:
        return {
            "sigma2": None,
            "leverage": None,
            "cooks_d": None,
            "cooks_threshold": None,
            "influential_indices": [],
        }

    p = len(X[0])
    denominator = n - p - 1
    if denominator <= 0:
        raise ValueError("Cook's Distance requires n > p + 1")

    rss = sum(residual * residual for residual in residuals)
    sigma2 = rss / denominator
    leverage = hat_matrix(X)["leverage"]

    if is_zero(sigma2):
        cooks_d = [0.0 for _ in range(n)]
    else:
        cooks_d = []
        for i in range(n):
            h_ii = leverage[i]
            if is_zero(1.0 - h_ii):
                cooks_d.append(float("inf"))
            else:
                cooks_d.append(
                    (residuals[i] ** 2 * h_ii)
                    / (sigma2 * (p + 1) * (1.0 - h_ii) ** 2)
                )

    threshold = 4.0 / n
    influential = [i for i, value in enumerate(cooks_d) if value > threshold]

    return {
        "sigma2": sigma2,
        "leverage": leverage,
        "cooks_d": cooks_d,
        "cooks_threshold": threshold,
        "influential_indices": influential,
    }


def residual_diagnostics(
    y: list[float],
    y_hat: list[float],
    X: list[list[float]] | None = None,
) -> tuple:
    """
    Draw four residual diagnostic plots and return (fig, metrics).

    Plots:
        1. Residuals vs Fitted
        2. Q-Q Plot
        3. Scale-Location
        4. Cook's Distance
    """
    _validate_inputs(y, y_hat, X)

    import matplotlib.pyplot as plt

    n = len(y)
    residuals = [y[i] - y_hat[i] for i in range(n)]
    rss = sum(residual * residual for residual in residuals)

    residual_mean = _mean(residuals)
    residual_std = _std(residuals)
    if is_zero(residual_std):
        residuals_std = [0.0 for _ in residuals]
    else:
        residuals_std = [
            (residual - residual_mean) / residual_std for residual in residuals
        ]

    theoretical_quantiles = _normal_quantiles(n)
    empirical_quantiles = sorted(residuals_std)
    sqrt_abs_std_residuals = [abs(value) ** 0.5 for value in residuals_std]
    cooks_metrics = _compute_cooks_distance(residuals, X)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax_residuals = axes[0][0]
    ax_qq = axes[0][1]
    ax_scale = axes[1][0]
    ax_cooks = axes[1][1]

    ax_residuals.scatter(y_hat, residuals, label="Residuals", alpha=0.8)
    ax_residuals.axhline(0.0, color="red", linestyle="--", label="Zero residual")
    ax_residuals.set_title("Residuals vs Fitted", fontsize=13)
    ax_residuals.set_xlabel("Fitted values (y_hat)")
    ax_residuals.set_ylabel("Residuals (y - y_hat)")
    ax_residuals.legend()

    ax_qq.scatter(
        theoretical_quantiles,
        empirical_quantiles,
        label="Standardized residuals",
        alpha=0.8,
    )
    ax_qq.plot([-3, 3], [-3, 3], "r--", label="Ideal normal line")
    ax_qq.set_title("Q-Q Plot", fontsize=13)
    ax_qq.set_xlabel("Theoretical Normal Quantiles")
    ax_qq.set_ylabel("Empirical Residual Quantiles")
    ax_qq.legend()

    ax_scale.scatter(y_hat, sqrt_abs_std_residuals, label="Scale-Location", alpha=0.8)
    ax_scale.set_title("Scale-Location", fontsize=13)
    ax_scale.set_xlabel("Fitted values (y_hat)")
    ax_scale.set_ylabel("sqrt(|standardized residuals|)")
    ax_scale.legend()

    cooks_d = cooks_metrics["cooks_d"]
    threshold = cooks_metrics["cooks_threshold"]
    if cooks_d is None:
        ax_cooks.plot([], [], label="Cook's Distance unavailable")
        ax_cooks.text(
            0.5,
            0.5,
            "X is required for Cook's Distance",
            ha="center",
            va="center",
            transform=ax_cooks.transAxes,
        )
        ax_cooks.set_ylim(0.0, 1.0)
    else:
        indices = list(range(n))
        ax_cooks.stem(indices, cooks_d, basefmt=" ", label="Cook's D")
        ax_cooks.axhline(
            threshold,
            color="red",
            linestyle="--",
            label=f"Threshold 4/n = {threshold:.4f}",
        )

    ax_cooks.set_title("Cook's Distance", fontsize=13)
    ax_cooks.set_xlabel("Observation index")
    ax_cooks.set_ylabel("Cook's D")
    ax_cooks.legend()

    plt.tight_layout()
    plt.show()

    metrics = {
        "residuals": residuals,
        "residuals_standardized": residuals_std,
        "rss": rss,
        "residual_mean": residual_mean,
        "residual_std": residual_std,
        "theoretical_quantiles": theoretical_quantiles,
        "empirical_quantiles": empirical_quantiles,
        "sqrt_abs_standardized_residuals": sqrt_abs_std_residuals,
        "sigma2": cooks_metrics["sigma2"],
        "leverage": cooks_metrics["leverage"],
        "cooks_d": cooks_metrics["cooks_d"],
        "cooks_threshold": cooks_metrics["cooks_threshold"],
        "influential_indices": cooks_metrics["influential_indices"],
    }

    return fig, metrics
