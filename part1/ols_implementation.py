"""
part1/ols_implementation.py
Triết - F1, F2, F3
Branch: feat/part1-ols-metrics
Deadline: 15/5/2026
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import (
    transpose, matmul, matvec, inverse,
    dot, norm,
    make_linear_data, make_multifeature_data, make_collinear_data,
    assert_close, assert_equal, assert_true, assert_shape, assert_in_range,
)
from config import RANDOM_STATE, EPSILON, is_zero
from test_logger import TestLogger

import math


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_bias(X: list[list[float]]) -> list[list[float]]:
    """Thêm cột 1 (intercept) vào đầu X → X_bias shape (n, p+1)."""
    return [[1.0] + row for row in X]


def _diag(M: list[list[float]]) -> list[float]:
    """Trả về đường chéo chính của ma trận vuông M."""
    return [M[i][i] for i in range(len(M))]


def vector_sub(a: list[float], b: list[float]) -> list[float]:
    return [a[i] - b[i] for i in range(len(a))]


def _eigenvalues_symmetric(M: list[list[float]]) -> list[float]:
    """Power iteration with deflation for symmetric matrix eigenvalues."""
    n = len(M)
    A = [row[:] for row in M]
    eigenvalues = []

    for _ in range(n):
        v = [1.0 / math.sqrt(n)] * n
        for _ in range(200):
            Av = matvec(A, v)
            nrm = norm(Av)
            if nrm < 1e-14:
                break
            v = [x / nrm for x in Av]
        Av = matvec(A, v)
        lam = dot(v, Av)
        eigenvalues.append(lam)
        for i in range(n):
            for j in range(n):
                A[i][j] -= lam * v[i] * v[j]

    return eigenvalues


# ---------------------------------------------------------------------------
# F1. ols_fit
# ---------------------------------------------------------------------------

def ols_fit(X: list[list[float]], y: list[float]) -> dict:
    """
    Giải Normal Equations: beta_hat = (XᵀX)⁻¹ Xᵀy.

    Tham số
    -------
    X : list[list[float]], shape (n, p)  — features, CHƯA có cột bias.
    y : list[float],       shape (n,)    — target liên tục.

    Trả về
    ------
    dict với các key:
        'beta_hat'   : list[float] (p+1,)  — [intercept, β1, ..., βp]
        'sigma2_hat' : float                — ước lượng phương sai nhiễu
        'y_hat'      : list[float] (n,)    — fitted values
        'residuals'  : list[float] (n,)    — y - y_hat
    """
    n = len(y)
    p = len(X[0])
    X_bias = _add_bias(X)

    X_bias_T = transpose(X_bias) # (p+1, n)
    A = matmul(X_bias_T, X_bias) # Ma trận X_bias^T * X_bias (p+1, p+1)
    b = matvec(X_bias_T, y) # Vector X_bias^T * y (p+1,)

    A_inv = inverse(A) # Ma trận A nghịch đảo (p+1, p+1)
    beta_hat = matvec(A_inv, b) # Ước lượng hệ số beta (p+1,)

    y_hat = matvec(X_bias, beta_hat) # (n, p+1) @ (p+1,) = (n,)
    residuals = vector_sub(y, y_hat)
    RSS = 0.0
    for val in residuals:
        RSS += val * val

    df = n - p - 1
    if df <= 0:
        raise ValueError("Số mẫu phải lớn hơn số hệ số")

    sigma2_hat = RSS / df

    return {
        "beta_hat":   beta_hat,
        "sigma2_hat": sigma2_hat,
        "y_hat":      y_hat,
        "residuals":  residuals,
    }


# ---------------------------------------------------------------------------
# F2. hat_matrix
# ---------------------------------------------------------------------------

def hat_matrix(X: list[list[float]]) -> dict:
    """
    Tính Hat Matrix H = X_bias (XᵀX)⁻¹ X_biasᵀ.

    Tham số
    -------
    X : list[list[float]], shape (n, p)  — features, CHƯA có cột bias.

    Trả về
    ------
    dict với các key:
        'H'             : list[list[float]] (n, n)
        'is_idempotent' : bool   — H² ≈ H (sai số 1e-8)
        'is_symmetric'  : bool   — Hᵀ ≈ H
        'rank'          : int    — rank(H) = p+1
        'eigenvalues'   : list[float]  — chỉ gần 0 hoặc 1
    """
    n = len(X)

    X_bias = _add_bias(X)
    X_bias_T = transpose(X_bias)
    A = matmul(X_bias_T, X_bias)
    A_inv = inverse(A)

    # TODO F2-4: H = X_bias @ A_inv @ X_biasᵀ
    #            Gợi ý: H = matmul(matmul(X_bias, A_inv), transpose(X_bias))
    H = matmul(matmul(X_bias, A_inv), transpose(X_bias))

    # TODO F2-5: Kiểm tra idempotent: H_sq = H @ H, so sánh từng phần tử với H
    #            Dùng sai số EPSILON = 1e-8
    H_sq = matmul(H, H)
    is_idempotent = True
    for i in range(n):
        for j in range(n):
            if abs(H_sq[i][j] - H[i][j]) > EPSILON:
                is_idempotent = False
                break

    # TODO F2-6: Kiểm tra symmetric: so sánh H[i][j] với H[j][i]
    is_symmetric = True
    for i in range(n):
        for j in range(i+1, n):
            if abs(H[i][j] - H[j][i]) > EPSILON:
                is_symmetric = False
                break

    eigenvalues = _eigenvalues_symmetric(H)
    rank = round(sum(_diag(H)))  # trace(H) = rank(H) for projection matrix

    return {
        "H":             H,
        "is_idempotent": is_idempotent,
        "is_symmetric":  is_symmetric,
        "rank":          rank,
        "eigenvalues":   eigenvalues,
    }


# ---------------------------------------------------------------------------
# F3. model_metrics
# ---------------------------------------------------------------------------

def model_metrics(y: list[float], y_hat: list[float], p: int) -> dict:
    """
    Tính các chỉ số đánh giá mô hình hồi quy.

    Tham số
    -------
    y     : list[float] (n,)  — ground truth.
    y_hat : list[float] (n,)  — fitted values.
    p     : int               — số features (KHÔNG tính intercept).

    Trả về
    ------
    dict:
        'RSS'      : float  — Residual Sum of Squares
        'TSS'      : float  — Total Sum of Squares
        'MSS'      : float  — Model Sum of Squares  (= TSS - RSS)
        'R2'       : float  — Hệ số xác định
        'R2_adj'   : float  — R² hiệu chỉnh
        'F_stat'   : float  — F-statistic
        'F_pvalue' : float  — p-value của F-test (dùng scipy.stats.f.sf)
        'MAE'      : float  — Mean Absolute Error
        'RMSE'     : float  — Root Mean Squared Error
    """
    from scipy.stats import f as f_dist

    n = len(y)

    RSS = sum((y[i] - y_hat[i]) ** 2 for i in range(n))

    y_mean = sum(y) / n
    TSS = sum((yi - y_mean) ** 2 for yi in y)

    MSS = TSS - RSS

    R2 = 1.0 if abs(TSS) < 1e-12 else 1.0 - RSS / TSS

    df = n - p - 1
    if df <= 0:
        raise ValueError("n - p - 1 <= 0: số mẫu phải lớn hơn số tham số")
    R2_adj = 1.0 - (n - 1) / df * (1.0 - R2)

    F_stat = (MSS / p) / (RSS / df) if RSS > 1e-14 else float("inf")

    F_pvalue = float(f_dist.sf(F_stat, dfn=p, dfd=df))

    MAE = sum(abs(y[i] - y_hat[i]) for i in range(n)) / n

    RMSE = math.sqrt(sum((y[i] - y_hat[i]) ** 2 for i in range(n)) / n)

    return {
        "RSS":      RSS,
        "TSS":      TSS,
        "MSS":      MSS,
        "R2":       R2,
        "R2_adj":   R2_adj,
        "F_stat":   F_stat,
        "F_pvalue": F_pvalue,
        "MAE":      MAE,
        "RMSE":     RMSE,
    }


# ---------------------------------------------------------------------------
# Visualizations (dùng trong notebook hoặc khi chạy standalone)
# ---------------------------------------------------------------------------

def plot_hat_matrix(H: list[list[float]], eigenvalues: list[float], save_path: str = None):
    """
    Vẽ 2 biểu đồ bắt buộc cho F2:
      1. Heatmap của H (chỉ khi n <= 20)
      2. Histogram của eigenvalues (chứng minh chỉ có giá trị 0 hoặc 1)

    Tham số
    -------
    H            : Hat matrix.
    eigenvalues  : Danh sách eigenvalues của H.
    save_path    : Nếu không None, lưu figure vào đường dẫn này.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    n = len(H)
    n_plots = 2 if n <= 20 else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    # Subplot 1 — Histogram eigenvalues
    ax = axes[0]
    ax.hist(eigenvalues, bins=20, edgecolor="black")
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="0")
    ax.axvline(x=1, color="red", linestyle="--", linewidth=1.5, label="1")
    ax.set_title("Eigenvalues of Hat Matrix H")
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Count")
    ax.legend()

    # Subplot 2 — Heatmap H (chỉ khi n <= 20)
    if n <= 20:
        ax2 = axes[1]
        import numpy as np
        sns.heatmap(
            np.array(H),
            annot=(n <= 10),
            fmt=".2f",
            cmap="coolwarm",
            ax=ax2,
        )
        ax2.set_title("Hat Matrix H")
        ax2.set_xlabel("Observation index")
        ax2.set_ylabel("Observation index")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


# ---------------------------------------------------------------------------
# Unit Tests — F1: ols_fit  (≥ 4 tests)
# ---------------------------------------------------------------------------

def test_ols_perfect_fit():
    """Khi sigma=0, beta_hat phải trùng khớp với TRUE_BETA."""
    feat_beta = [1.0, 2.0, -3.0]   # weights cho 3 features (không có intercept)
    X, y, _ = make_linear_data(n=50, beta=feat_beta, sigma=0.0, seed=RANDOM_STATE)
    result = ols_fit(X, y)
    true_beta_full = [0.0] + feat_beta  # intercept=0, rồi feature weights
    assert_close(result['beta_hat'], true_beta_full, label="beta_hat vs TRUE_BETA")
    assert_close(sum(r**2 for r in result['residuals']), 0.0, label="RSS ≈ 0")


def test_ols_output_shapes():
    """Kiểm tra shape của tất cả output."""
    n, p = 30, 3
    X, y, _ = make_multifeature_data(n=n, p=p, sigma=1.0, seed=RANDOM_STATE)
    result = ols_fit(X, y)
    assert_shape(result['beta_hat'], (p + 1,), label="beta_hat shape")
    assert_shape(result['y_hat'], (n,), label="y_hat shape")
    assert_shape(result['residuals'], (n,), label="residuals shape")
    assert_true(isinstance(result['sigma2_hat'], float), label="sigma2_hat is float")


def test_ols_sigma2_positive():
    """sigma2_hat phải luôn dương."""
    X, y, _ = make_multifeature_data(n=40, p=2, sigma=2.0, seed=RANDOM_STATE)
    result = ols_fit(X, y)
    assert_true(result['sigma2_hat'] > 0, label="sigma2_hat > 0")


def test_ols_residuals_sum_zero():
    """Với OLS chuẩn, tổng phần dư Σeᵢ = 0 (intercept included)."""
    X, y, _ = make_multifeature_data(n=50, p=3, sigma=1.0, seed=RANDOM_STATE)
    result = ols_fit(X, y)
    assert_close(sum(result['residuals']), 0.0, atol=1e-6, label="sum(residuals) ≈ 0")


def test_ols_verify_with_sklearn():
    """So sánh beta_hat với sklearn.LinearRegression."""
    from sklearn.linear_model import LinearRegression
    import numpy as np
    X, y, _ = make_multifeature_data(n=60, p=4, sigma=1.5, seed=RANDOM_STATE)
    result = ols_fit(X, y)
    lr = LinearRegression().fit(np.array(X), np.array(y))
    sklearn_beta = [lr.intercept_] + list(lr.coef_)
    assert_close(result['beta_hat'], sklearn_beta, rtol=1e-4, label="beta_hat vs sklearn")


# ---------------------------------------------------------------------------
# Unit Tests — F2: hat_matrix  (≥ 4 tests)
# ---------------------------------------------------------------------------

def test_hat_matrix_idempotent():
    """H @ H phải xấp xỉ H (H² = H)."""
    X, *_ = make_linear_data(n=20, beta=[1.0, 2.0], sigma=0.0, seed=RANDOM_STATE)
    result = hat_matrix(X)
    assert_true(result['is_idempotent'], label="H is idempotent")


def test_hat_matrix_symmetric():
    """H phải symmetric: H[i][j] == H[j][i]."""
    X, *_ = make_multifeature_data(n=15, p=3, sigma=0.0, seed=RANDOM_STATE)
    result = hat_matrix(X)
    assert_true(result['is_symmetric'], label="H is symmetric")


def test_hat_matrix_rank():
    """rank(H) phải bằng p+1."""
    p = 3
    X, *_ = make_multifeature_data(n=20, p=p, sigma=0.0, seed=RANDOM_STATE)
    result = hat_matrix(X)
    assert_equal(result['rank'], p + 1, label="rank(H) = p+1")


def test_hat_matrix_eigenvalues_binary():
    """Tất cả eigenvalue phải xấp xỉ 0 hoặc 1."""
    X, *_ = make_multifeature_data(n=15, p=2, sigma=0.0, seed=RANDOM_STATE)
    result = hat_matrix(X)
    for ev in result['eigenvalues']:
        near_zero = abs(ev) < 1e-6
        near_one  = abs(ev - 1.0) < 1e-6
        assert_true(near_zero or near_one, label=f"eigenvalue {ev:.4f} near 0 or 1")


def test_hat_matrix_output_shape():
    """H phải có shape (n, n)."""
    n = 12
    X, *_ = make_multifeature_data(n=n, p=2, sigma=0.0, seed=RANDOM_STATE)
    result = hat_matrix(X)
    assert_shape(result['H'], (n, n), label="H shape (n, n)")


# ---------------------------------------------------------------------------
# Unit Tests — F3: model_metrics  (≥ 4 tests)
# ---------------------------------------------------------------------------

def test_metrics_perfect_prediction():
    """Khi y_hat = y thì RSS=0, R2=1, MAE=0, RMSE=0."""
    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_hat = y[:]
    metrics = model_metrics(y, y_hat, p=1)
    assert_close(metrics['RSS'],  0.0, label="RSS = 0")
    assert_close(metrics['R2'],   1.0, label="R2 = 1")
    assert_close(metrics['MAE'],  0.0, label="MAE = 0")
    assert_close(metrics['RMSE'], 0.0, label="RMSE = 0")


def test_metrics_r2_range():
    """R² phải nằm trong [-∞, 1] (thực tế test [-1, 1] với data tốt)."""
    X, y, _ = make_multifeature_data(n=50, p=3, sigma=1.0, seed=RANDOM_STATE)
    result = ols_fit(X, y)
    metrics = model_metrics(y, result['y_hat'], p=3)
    assert_in_range(metrics['R2'], -1.0, 1.0, label="R2 in [-1, 1]")
    assert_in_range(metrics['R2_adj'], -2.0, 1.0, label="R2_adj reasonable")


def test_metrics_mss_identity():
    """MSS = TSS - RSS phải đúng."""
    y = [1.0, 3.0, 2.0, 5.0, 4.0]
    y_hat = [1.5, 2.5, 2.5, 4.5, 3.5]
    metrics = model_metrics(y, y_hat, p=1)
    assert_close(metrics['MSS'], metrics['TSS'] - metrics['RSS'], label="MSS = TSS - RSS")


def test_metrics_f_pvalue():
    """p-value của F-test phải nằm trong [0, 1]."""
    X, y, _ = make_multifeature_data(n=50, p=3, sigma=1.0, seed=RANDOM_STATE)
    result = ols_fit(X, y)
    metrics = model_metrics(y, result['y_hat'], p=3)
    assert_in_range(metrics['F_pvalue'], 0.0, 1.0, label="F_pvalue in [0, 1]")


def test_metrics_verify_r2_with_sklearn():
    """So sánh R² với sklearn.metrics.r2_score."""
    from sklearn.metrics import r2_score
    import numpy as np
    X, y, _ = make_multifeature_data(n=50, p=3, sigma=1.0, seed=RANDOM_STATE)
    result = ols_fit(X, y)
    metrics = model_metrics(y, result['y_hat'], p=3)
    sklearn_r2 = r2_score(np.array(y), np.array(result['y_hat']))
    assert_close(metrics['R2'], sklearn_r2, rtol=1e-5, label="R2 vs sklearn")


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_all_tests():
    logger = TestLogger()
    passed = 0
    total  = 0

    suites = [
        ("F1 - ols_fit", [
            test_ols_perfect_fit,
            test_ols_output_shapes,
            test_ols_sigma2_positive,
            test_ols_residuals_sum_zero,
            test_ols_verify_with_sklearn,
        ]),
        ("F2 - hat_matrix", [
            test_hat_matrix_idempotent,
            test_hat_matrix_symmetric,
            test_hat_matrix_rank,
            test_hat_matrix_eigenvalues_binary,
            test_hat_matrix_output_shape,
        ]),
        ("F3 - model_metrics", [
            test_metrics_perfect_prediction,
            test_metrics_r2_range,
            test_metrics_mss_identity,
            test_metrics_f_pvalue,
            test_metrics_verify_r2_with_sklearn,
        ]),
    ]

    for suite_name, tests in suites:
        logger.print_suite_header(suite_name)
        for test_fn in tests:
            total += 1
            try:
                test_fn()
                logger.print_result(test_fn.__name__, True)
                passed += 1
            except NotImplementedError as e:
                logger.print_result(test_fn.__name__, False, f"NotImplemented: {e}")
            except Exception as e:
                logger.print_result(test_fn.__name__, False, str(e))

    logger.print_summary(passed, total)

def coef_inference(X, y, beta_hat, sigma2):
    """
    F4: Tính toán các chỉ số thống kê cho các hệ số.
    Quy tắc: X chưa có bias, hàm tự thêm bên trong.
    """
    n = len(X)
    p = len(X[0]) # số lượng features
    
    # 1. Thêm cột bias (cột 1 đầu tiên)
    X_bias = [[1.0] + row for row in X]
    
    # 2. Tính Ma trận hiệp phương sai của beta: Cov = sigma2 * (X^T @ X)^-1
    # Dùng hoàn toàn hàm từ utils để tính toán đại số
    XT = transpose(X_bias)
    XTX = matmul(XT, X_bias)
    XTX_inv = inverse(XTX)
    
    # 3. Tính Standard Errors (Căn bậc hai các phần tử trên đường chéo chính)
    std_errs = []
    for i in range(len(XTX_inv)):
        # Công thức: sqrt(sigma^2 * C_jj)
        se = math.sqrt(sigma2 * XTX_inv[i][i])
        std_errs.append(se)
        
    # 4. Tính t-statistics: t = beta / std_err
    t_stats = [b / se if se != 0 else 0 for b, se in zip(beta_hat, std_errs)]
    
    # 5. Tra bảng thống kê (Đây là nơi duy nhất dùng Scipy)
    # Bậc tự do: dof = n - (p + 1) vì có p features + 1 intercept
    dof = n - p - 1
    
    # p-value = 2 * (1 - CDF(|t|))
    p_values = [2 * stats.t.sf(abs(t), dof) for t in t_stats]
    
    # t_critical cho khoảng tin cậy 95% (alpha = 0.05)
    t_crit = stats.t.ppf(0.975, dof)
    
    ci_lower = [b - t_crit * se for b, se in zip(beta_hat, std_errs)]
    ci_upper = [b + t_crit * se for b, se in zip(beta_hat, std_errs)]
    
    # 6. Tạo DataFrame kết quả theo chuẩn
    data = {
        'coef': beta_hat,
        'std_err': std_errs,
        't_stat': t_stats,
        'p_value': p_values,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }
    
    index = ['intercept'] + [f'x{i+1}' for i in range(p)]
    return pd.DataFrame(data, index=index)

def vif(X):
    """
    F5: Tính Variance Inflation Factor.
    Input: X (n, p) chưa có bias.
    """
    n = len(X)
    p = len(X[0])
    vif_dict = {}
    
    for j in range(p):
        # 1. Tách biến thứ j làm biến mục tiêu (y_j)
        y_j = [row[j] for row in X]
        
        # 2. Các biến còn lại đóng vai trò là features (X_others)
        X_others = [[row[i] for i in range(p) if i != j] for row in X]
        
        # 3. Hồi quy y_j theo X_others 
        # Sử dụng hàm ols_fit bạn đã viết (nó sẽ tự thêm bias cho X_others)
        from .ols_implementation import ols_fit, model_metrics
        
        result_j = ols_fit(X_others, y_j)
        
        # 4. Tính R-squared của mô hình phụ này
        # model_metrics(y_true, y_pred, p_features)
        metrics = model_metrics(y_j, result_j['y_hat'], p - 1)
        r2_j = metrics['R2']
        
        # 5. Tính VIF = 1 / (1 - R^2)
        # Xử lý trường hợp R2 xấp xỉ 1 để tránh chia cho 0
        if r2_j >= 1.0 - 1e-10:
            vif_val = float('inf')
        else:
            vif_val = 1 / (1 - r2_j)
            
        vif_dict[f'x{j+1}'] = vif_val
        
    return vif_dict

if __name__ == "__main__":
    run_all_tests()