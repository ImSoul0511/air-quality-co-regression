import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import (
    transpose, matmul, matvec, inverse,
    dot, norm, vector_sub, add_bias_column
)
from config import EPSILON
import math

def _diag(M: list[list[float]]) -> list[float]:
    """Trả về đường chéo chính của ma trận vuông M."""
    return [M[i][i] for i in range(len(M))]


def _eigenvalues_symmetric(M: list[list[float]]) -> list[float]:
    """Tính các trị riêng của ma trận đối xứng bằng phương pháp lặp lũy thừa và giảm cấp ma trận.

    Tham số
    -------
    M : list[list[float]], shape (n, n) -- ma trận đối xứng.

    Trả về
    ------
    list[float] -- danh sách trị riêng của ma trận.
    """
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

def ols_fit(X: list[list[float]], y: list[float]) -> dict:
    """Giải hệ phương trình chuẩn: beta_hat = (X^T X)^{-1} X^T y.

    Tham số
    -------
    X : list[list[float]], shape (n, p) -- ma trận đặc trưng, chưa có cột bias.
    y : list[float],       shape (n,)   -- vector mục tiêu.

    Trả về
    ------
    dict -- từ điển chứa các kết quả:
        'beta_hat'   : list[float] (p+1,) -- hệ số hồi quy [intercept, beta_1, ..., beta_p]
        'sigma2_hat' : float              -- ước lượng phương sai nhiễu
        'y_hat'      : list[float] (n,)   -- giá trị dự đoán
        'residuals'  : list[float] (n,)   -- phần dư: y - y_hat
    """
    n = len(y)
    p = len(X[0])
    X_bias = add_bias_column(X)

    X_bias_T = transpose(X_bias) # Ma trận X_bias^T 
    A = matmul(X_bias_T, X_bias) # Ma trận X_bias^T * X_bias 
    b = matvec(X_bias_T, y) # Vector X_bias^T * y 

    A_inv = inverse(A) # Ma trận A nghịch đảo 
    beta_hat = matvec(A_inv, b) # Ước lượng hệ số beta 

    y_hat = matvec(X_bias, beta_hat) # Giá trị dự đoán
    residuals = vector_sub(y, y_hat) # Phần dư: y - y_hat
    RSS = 0.0
    for val in residuals:
        RSS += val * val

    df = n - p - 1
    if df <= 0:
        raise ValueError("Số mẫu phải lớn hơn số hệ số")

    sigma2_hat = RSS / df

    return {
        "beta_hat":   beta_hat,     # Vector trọng số
        "sigma2_hat": sigma2_hat,   # Phương sai nhiễu
        "y_hat":      y_hat,        # Fitted values
        "residuals":  residuals,    # Phần dư: y - y_hat
    }

def hat_matrix(X: list[list[float]]) -> dict:
    """Tính Hat Matrix H = X_bias (X^T X)^{-1} X_bias^T.

    Tham số
    -------
    X : list[list[float]], shape (n, p) -- ma trận đặc trưng, chưa có cột bias.

    Trả về
    ------
    dict -- từ điển chứa các kết quả:
        'H'             : list[list[float]] (n, n) -- ma trận hình chiếc mũ
        'is_idempotent' : bool -- True nếu H^2 == H (trong sai số cho phép)
        'is_symmetric'  : bool -- True nếu H^T == H
        'rank'          : int  -- hạng của ma trận H
        'eigenvalues'   : list[float] -- các trị riêng của H (gần bằng 0 hoặc 1)
    """
    n = len(X)

    X_bias = add_bias_column(X)
    X_bias_T = transpose(X_bias)
    A = matmul(X_bias_T, X_bias)
    A_inv = inverse(A)

    # H = X_bias @ A_inv @ X_bias^T
    # H = matmul(matmul(X_bias, A_inv), transpose(X_bias))
    H = matmul(matmul(X_bias, A_inv), transpose(X_bias))

    H_sq = matmul(H, H)
    is_idempotent = True
    for i in range(n):
        for j in range(n):
            if abs(H_sq[i][j] - H[i][j]) > EPSILON:
                is_idempotent = False
                break

    is_symmetric = True
    for i in range(n):
        for j in range(i+1, n):
            if abs(H[i][j] - H[j][i]) > EPSILON:
                is_symmetric = False
                break

    eigenvalues = _eigenvalues_symmetric(H)
    rank = round(sum(_diag(H)))  # trace(H) = rank(H) cho projection matrix

    return {
        "H":             H,
        "is_idempotent": is_idempotent,
        "is_symmetric":  is_symmetric,
        "rank":          rank,
        "eigenvalues":   eigenvalues,
    }

def model_metrics(y: list[float], y_hat: list[float], p: int) -> dict:
    """Tính các chỉ số đánh giá mô hình hồi quy.

    Tham số
    -------
    y     : list[float] (n,) -- vector thực tế.
    y_hat : list[float] (n,) -- vector dự đoán.
    p     : int              -- số đặc trưng (không tính intercept).

    Trả về
    ------
    dict -- từ điển chứa các chỉ số:
        'RSS'      : float -- tổng bình phương phần dư (Residual Sum of Squares)
        'TSS'      : float -- tổng bình phương toàn bộ (Total Sum of Squares)
        'MSS'      : float -- tổng bình phương mô hình (Model Sum of Squares)
        'R2'       : float -- hệ số xác định (R-squared)
        'R2_adj'   : float -- hệ số xác định hiệu chỉnh (Adjusted R-squared)
        'F_stat'   : float -- giá trị thống kê F
        'F_pvalue' : float -- p-value của F-test
        'MAE'      : float -- sai số tuyệt đối trung bình (Mean Absolute Error)
        'RMSE'     : float -- căn sai số bình phương trung bình (Root Mean Squared Error)
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


def plot_hat_matrix(H: list[list[float]], eigenvalues: list[float], save_path: str = None):
    """
    Vẽ 2 biểu đồ:
        1. Heatmap của H (chỉ khi n <= 20)
        2. Biểu đồ của eigenvalues (chứng minh chỉ có giá trị 0 hoặc 1)

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

    # Subplot 1 -- Histogram eigenvalues
    ax = axes[0]
    ax.hist(eigenvalues, bins=20, edgecolor="black")
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="0")
    ax.axvline(x=1, color="red", linestyle="--", linewidth=1.5, label="1")
    ax.set_title("Eigenvalues of Hat Matrix H")
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Count")
    ax.legend()

    # Subplot 2 -- Heatmap H (chỉ khi n <= 20)
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

def coef_inference(X, y, beta_hat, sigma2):
    """Tính toán các chỉ số thống kê suy diễn cho các hệ số hồi quy.

    Tham số
    -------
    X        : list[list[float]], shape (n, p) -- ma trận đặc trưng (chưa có bias).
    y        : list[float],       shape (n,)   -- vector mục tiêu.
    beta_hat : list[float],       shape (p+1,) -- hệ số hồi quy ước lượng.
    sigma2   : float                           -- ước lượng phương sai nhiễu.

    Trả về
    ------
    pd.DataFrame -- bảng chứa các cột: coef, std_err, t_stat, p_value, ci_lower, ci_upper.
    """
    from scipy import stats as scipy_stats
    import pandas as pd

    n = len(X)
    p = len(X[0]) # số lượng features
    
    # 1. Thêm cột bias (cột 1 đầu tiên)
    # Kiểm tra số lượng cột của X so với số lượng hệ số beta_hat
    if len(X[0]) == len(beta_hat):
        # Trường hợp 1: X đã có sẵn cột bias
        X_bias = X
        p = len(X[0]) - 1  # p là số features thực tế (không tính intercept)
    elif len(X[0]) == len(beta_hat) - 1:
        # Trường hợp 2: X chưa có cột bias, hàm phải tự thêm
        X_bias = add_bias_column(X)
        p = len(X[0])      # p là số lượng features hiện tại
    else:
        raise ValueError(f"Bất thường: Số cột của X ({len(X[0])}) không khớp với beta_hat ({len(beta_hat)})")
    
    # 2. Tính Ma trận hiệp phương sai của beta: Cov = sigma2 * (X^T * X)^{-1}
    XT = transpose(X_bias)
    XTX = matmul(XT, X_bias)
    XTX_inv = inverse(XTX)
    
    # 3. Tính Standard Errors 
    std_errs = []
    for i in range(len(XTX_inv)):
        se = math.sqrt(sigma2 * XTX_inv[i][i])
        std_errs.append(se)
        
    # 4. Tính t-statistics: t = beta / std_err
    t_stats = [b / se if se != 0 else 0 for b, se in zip(beta_hat, std_errs)]
    
    # 5. Tra bảng thống kê
    # Bậc tự do: dof = n - (p + 1) vì có p features + 1 intercept
    dof = n - p - 1
    
    # p-value = 2 * (1 - CDF(|t|))
    p_values = [2 * scipy_stats.t.sf(abs(t), dof) for t in t_stats]
    
    # t_critical cho khoảng tin cậy 95% (alpha = 0.05)
    t_crit = scipy_stats.t.ppf(0.975, dof)
    
    ci_lower = [b - t_crit * se for b, se in zip(beta_hat, std_errs)]
    ci_upper = [b + t_crit * se for b, se in zip(beta_hat, std_errs)]
    
    # 6. Tạo DataFrame kết quả
    data = {
        'coef': beta_hat,
        'std_err': std_errs,
        't_stat': t_stats,
        'p_value': p_values,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }
    index = ['intercept'] + [f'x{i+1}' for i in range(len(beta_hat) - 1)]
    return pd.DataFrame(data, index=index)

def vif(X):
    """Tính hệ số phóng đại phương sai (Variance Inflation Factor - VIF).

    Tham số
    -------
    X : list[list[float]], shape (n, p) -- ma trận đặc trưng, chưa có bias.

    Trả về
    ------
    dict -- từ điển chứa VIF của từng đặc trưng (x1, x2, ...).
    """
    n = len(X)
    p = len(X[0])

    if p == 1:
        return {"x1": 1.0}

    vif_dict = {}
    
    for j in range(p):
        # 1. Tách biến thứ j làm biến mục tiêu (y_j)
        y_j = [row[j] for row in X]
        
        # 2. Các biến còn lại đóng vai trò là features (X_others)
        X_others = [[row[i] for i in range(p) if i != j] for row in X]
        
        # 3. Hồi quy y_j theo X_others 
        result_j = ols_fit(X_others, y_j)
        
        # 4. Tính R-squared của mô hình phụ này
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

def run_ols_tests() -> tuple[int, int]:
    """Chạy unit tests cho ols_fit từ part1/test_case.py."""
    from part1.test_case import (
        _log,
        run_test_cases,
        test_ols_output_shapes,
        test_ols_perfect_fit,
        test_ols_residuals_sum_zero,
        test_ols_sigma2_positive,
        test_ols_verify_with_sklearn,
    )

    _log.print_suite_header("OLS FIT - UNIT TESTS")
    passed_count, total_count = run_test_cases(
        [
            test_ols_perfect_fit,
            test_ols_output_shapes,
            test_ols_sigma2_positive,
            test_ols_residuals_sum_zero,
            test_ols_verify_with_sklearn,
        ]
    )
    _log.print_summary(passed_count, total_count)
    return passed_count, total_count


def run_hat_matrix_tests() -> tuple[int, int]:
    """Chạy unit tests cho hat_matrix từ part1/test_case.py."""
    from part1.test_case import (
        _log,
        run_test_cases,
        test_hat_matrix_eigenvalues_binary,
        test_hat_matrix_idempotent,
        test_hat_matrix_output_shape,
        test_hat_matrix_rank,
        test_hat_matrix_symmetric,
    )

    _log.print_suite_header("HAT MATRIX - UNIT TESTS")
    passed_count, total_count = run_test_cases(
        [
            test_hat_matrix_idempotent,
            test_hat_matrix_symmetric,
            test_hat_matrix_rank,
            test_hat_matrix_eigenvalues_binary,
            test_hat_matrix_output_shape,
        ]
    )
    _log.print_summary(passed_count, total_count)
    return passed_count, total_count


def run_metrics_tests() -> tuple[int, int]:
    """Chạy unit tests cho model_metrics từ part1/test_case.py."""
    from part1.test_case import (
        _log,
        run_test_cases,
        test_metrics_f_pvalue,
        test_metrics_mss_identity,
        test_metrics_perfect_prediction,
        test_metrics_r2_range,
        test_metrics_verify_r2_with_sklearn,
    )

    _log.print_suite_header("MODEL METRICS - UNIT TESTS")
    passed_count, total_count = run_test_cases(
        [
            test_metrics_perfect_prediction,
            test_metrics_r2_range,
            test_metrics_mss_identity,
            test_metrics_f_pvalue,
            test_metrics_verify_r2_with_sklearn,
        ]
    )
    _log.print_summary(passed_count, total_count)
    return passed_count, total_count


def run_coef_inference_tests() -> tuple[int, int]:
    """Chạy unit tests cho coef_inference từ part1/test_case.py."""
    from part1.test_case import (
        _log,
        run_test_cases,
        test_coef_inference_coefficients_match_input,
        test_coef_inference_output_structure,
        test_coef_inference_p_values_and_ci_valid,
        test_coef_inference_sigma2_scales_standard_errors,
        test_coef_inference_standard_errors_positive,
        test_coef_inference_t_stat_formula,
    )

    _log.print_suite_header("COEFFICIENT INFERENCE - UNIT TESTS")
    passed_count, total_count = run_test_cases(
        [
            test_coef_inference_output_structure,
            test_coef_inference_coefficients_match_input,
            test_coef_inference_standard_errors_positive,
            test_coef_inference_t_stat_formula,
            test_coef_inference_p_values_and_ci_valid,
            test_coef_inference_sigma2_scales_standard_errors,
        ]
    )
    _log.print_summary(passed_count, total_count)
    return passed_count, total_count


def run_vif_tests() -> tuple[int, int]:
    """Chạy unit tests cho vif từ part1/test_case.py."""
    from part1.test_case import (
        _log,
        run_test_cases,
        test_vif_near_collinearity_is_large,
        test_vif_orthogonal_features_are_one,
        test_vif_output_keys,
        test_vif_perfect_collinearity_is_infinite,
        test_vif_single_feature_is_one,
        test_vif_values_are_at_least_one,
    )

    _log.print_suite_header("VIF - UNIT TESTS")
    passed_count, total_count = run_test_cases(
        [
            test_vif_output_keys,
            test_vif_orthogonal_features_are_one,
            test_vif_perfect_collinearity_is_infinite,
            test_vif_near_collinearity_is_large,
            test_vif_single_feature_is_one,
            test_vif_values_are_at_least_one,
        ]
    )
    _log.print_summary(passed_count, total_count)
    return passed_count, total_count


def run_all_tests():
    """Chạy tất cả các tests."""
    run_ols_tests()
    run_hat_matrix_tests()
    run_metrics_tests()
    run_coef_inference_tests()
    run_vif_tests()

if __name__ == "__main__":
    run_all_tests()
