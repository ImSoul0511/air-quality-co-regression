import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import is_zero
from utils import inverse, matmul, transpose, add_bias_column


def _validate_vector(name: str, values: list[float]) -> None:
    if len(values) == 0:
        raise ValueError(f"{name} không được rỗng")


def _validate_inputs(
    y: list[float],
    y_hat: list[float],
    X: list[list[float]] | None,
) -> None:
    _validate_vector("y", y)
    _validate_vector("y_hat", y_hat)

    if len(y) != len(y_hat):
        raise ValueError("y và y_hat phải có cùng kích thước")

    if X is None:
        return

    if len(X) != len(y):
        raise ValueError("X phải có cùng số dòng với y")

    if len(X) == 0 or len(X[0]) == 0:
        raise ValueError("X phải chứa ít nhất một đặc trưng")

    p = len(X[0])
    for row in X:
        if len(row) != p:
            raise ValueError("Tất cả các dòng của X phải có cùng số lượng cột")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    mu = _mean(values)
    variance = sum((value - mu) ** 2 for value in values) / len(values)
    return variance**0.5


def hat_matrix(X: list[list[float]]) -> dict:
    """Tính toán ma trận hình chiếc mũ H = X(X^T X)^{-1} X^T của OLS.

    Tham số
    -------
    X : list[list[float]] -- ma trận đặc trưng (không bao gồm intercept).

    Trả về
    ------
    dict -- ma trận H, leverage của từng quan sát và ma trận X_bias.
    """
    if len(X) == 0 or len(X[0]) == 0:
        raise ValueError("X phải chứa ít nhất một đặc trưng")

    p = len(X[0])
    for row in X:
        if len(row) != p:
            raise ValueError("Tất cả các dòng của X phải có cùng số lượng cột")

    X_bias = add_bias_column(X)
    Xt = transpose(X_bias)
    XtX = matmul(Xt, X_bias)
    XtX_inv = inverse(XtX)
    H = matmul(matmul(X_bias, XtX_inv), Xt)

    return {
        "H": H,
        "leverage": [H[i][i] for i in range(len(H))],
        "X_bias": X_bias,
    }


def _standard_normal_ppf(p: float) -> float:
    """Nghịch đảo CDF chuẩn N(0,1) qua xấp xỉ hữu tỷ Acklam (sai số < 3.65e-9).

    Tham số
    -------
    p : float -- xác suất, 0 < p < 1.

    Trả về
    ------
    float -- z sao cho Phi(z) = p.
    """
    import math
    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]
    lo, hi = 0.02425, 1.0 - 0.02425
    if p < lo:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    elif p <= hi:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)


def _normal_quantiles(n: int) -> list[float]:
    return [_standard_normal_ppf((i - 0.5) / n) for i in range(1, n + 1)]


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
        raise ValueError("Khoảng cách Cook yêu cầu n > p + 1")

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
    """Vẽ 4 biểu đồ chẩn đoán phần dư hồi quy và trả về hình ảnh cùng các chỉ số chẩn đoán.

    Các biểu đồ bao gồm:
        1. Residuals vs Fitted
        2. Q-Q Plot
        3. Scale-Location
        4. Cook's Distance

    Tham số
    -------
    y     : list[float]              -- vector thực tế.
    y_hat : list[float]              -- vector dự đoán.
    X     : list[list[float]] | None -- ma trận đặc trưng.

    Trả về
    ------
    tuple -- (fig, metrics) trong đó fig là đối tượng matplotlib figure và metrics là từ điển các chỉ số chẩn đoán.
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
