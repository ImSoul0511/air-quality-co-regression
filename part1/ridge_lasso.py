import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import is_zero
from utils import (
    identity_matrix, matmul, matvec, transpose,
    add_bias_column, solve_system, make_lambda_grid
)


def mean_columns(X: list[list[float]]) -> list[float]:
    """Tính giá trị trung bình của từng cột trong ma trận X.

    Tham số
    -------
    X : list[list[float]] -- ma trận đầu vào.

    Trả về
    ------
    list[float] -- danh sách giá trị trung bình của từng cột.
    """
    n = len(X)
    p = len(X[0])

    means = []
    for j in range(p):
        s = 0.0
        for i in range(n):
            s += X[i][j]
        means.append(s / n)
    return means


def std_columns(X: list[list[float]], means: list[float]) -> list[float]:
    """Tính độ lệch chuẩn của từng cột trong ma trận X.

    Tham số
    -------
    X     : list[list[float]] -- ma trận đầu vào.
    means : list[float]        -- danh sách trung bình tương ứng của mỗi cột.

    Trả về
    ------
    list[float] -- danh sách độ lệch chuẩn của từng cột.
    """
    n = len(X)
    p = len(X[0])

    stds = []
    for j in range(p):
        s = 0.0
        for i in range(n):
            diff = X[i][j] - means[j]
            s += diff * diff

        std = (s / n) ** 0.5
        if is_zero(std):
            std = 1.0
        stds.append(std)
    return stds


def standardize(
    X: list[list[float]],
    means: list[float],
    stds: list[float],
) -> list[list[float]]:
    """Chuẩn hóa Z-score cho ma trận X dựa trên trung bình và độ lệch chuẩn đã tính trước.

    Tham số
    -------
    X     : list[list[float]] -- ma trận đặc trưng ban đầu.
    means : list[float]        -- trung bình của mỗi cột.
    stds  : list[float]        -- độ lệch chuẩn của mỗi cột.

    Trả về
    ------
    list[list[float]] -- ma trận đặc trưng đã chuẩn hóa.
    """
    X_scaled = []
    for row in X:
        scaled_row = []
        for j in range(len(row)):
            scaled_row.append((row[j] - means[j]) / stds[j])
        X_scaled.append(scaled_row)
    return X_scaled


def median(values: list[float]) -> float:
    """Tính trung vị của danh sách khác rỗng.

    Tham số
    -------
    values : list[float] -- danh sách các số thực.

    Trả về
    ------
    float -- giá trị trung vị.
    """
    if len(values) == 0:
        raise ValueError("danh sách giá trị không được rỗng")

    sorted_values = sorted(values)
    n = len(sorted_values)
    mid = n // 2

    if n % 2 == 1:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def validate_regression_input(X: list[list[float]], y: list[float], lam: float) -> None:
    """Kiểm tra tính hợp lệ của dữ liệu đầu vào hồi quy Ridge/Lasso.

    Tham số
    -------
    X   : list[list[float]] -- ma trận đặc trưng.
    y   : list[float]       -- vector mục tiêu.
    lam : float             -- hệ số điều chuẩn lambda.
    """
    if lam < 0:
        raise ValueError("lam phải lớn hơn hoặc bằng 0")

    if len(X) == 0:
        raise ValueError("X không được rỗng")

    if len(X) != len(y):
        raise ValueError("Số dòng của X phải khớp với số phần tử của y")

    if len(X[0]) == 0:
        raise ValueError("X phải chứa ít nhất một đặc trưng")

    p = len(X[0])
    for row in X:
        if len(row) != p:
            raise ValueError("Tất cả các dòng của X phải có cùng số lượng cột")

def ridge_fit(X: list[list[float]], y: list[float], lam: float) -> dict:
    """Mô hình hồi quy Ridge với nghiệm dạng đóng.

    Tối thiểu hóa: ||y - X * beta||^2 + lam * ||beta||^2.
    Trả về beta_hat ở thang đo gốc của các đặc trưng.

    Tham số
    -------
    X   : list[list[float]] -- ma trận đặc trưng.
    y   : list[float]       -- vector mục tiêu.
    lam : float             -- hệ số điều chuẩn lambda.

    Trả về
    ------
    dict -- beta_hat, y_hat, mean_X, std_X.
    """
    validate_regression_input(X, y, lam)

    mean_X = mean_columns(X)
    std_X = std_columns(X, mean_X)
    X_scaled = standardize(X, mean_X, std_X)
    X_bias = add_bias_column(X_scaled)

    p1 = len(X_bias[0])
    I = identity_matrix(p1)
    I[0][0] = 0.0

    Xt = transpose(X_bias)
    XtX = matmul(Xt, X_bias)
    b = matvec(Xt, y)

    A = []
    for i in range(p1):
        row = []
        for j in range(p1):
            row.append(XtX[i][j] + lam * I[i][j])
        A.append(row)

    beta_scaled = solve_system(A, b)
    beta_original = coefficients_to_original_scale(
        beta_scaled[0],
        beta_scaled[1:],
        mean_X,
        std_X,
    )

    X_orig_bias = add_bias_column(X)
    y_hat = matvec(X_orig_bias, beta_original)

    return {
        "beta_hat": beta_original,
        "y_hat": y_hat,
        "mean_X": mean_X,
        "std_X": std_X,
    }


def soft_threshold(rho: float, lam: float) -> float:
    """Toán tử ngưỡng mềm (soft-thresholding) cho cập nhật tọa độ của Lasso.

    Tham số
    -------
    rho : float -- giá trị tích vô hướng của đặc trưng với phần dư.
    lam : float -- hệ số điều chuẩn lambda.

    Trả về
    ------
    float -- giá trị sau khi áp dụng ngưỡng mềm.
    """
    if rho > lam:
        return rho - lam
    if rho < -lam:
        return rho + lam
    return 0.0


def coefficients_to_original_scale(
    intercept_scaled: float,
    beta_scaled: list[float],
    mean_X: list[float],
    std_X: list[float],
) -> list[float]:
    """Chuyển đổi các hệ số từ không gian chuẩn hóa về thang đo gốc của X.

    Tham số
    -------
    intercept_scaled : float       -- hệ số chặn trong không gian chuẩn hóa.
    beta_scaled      : list[float] -- các hệ số độ dốc trong không gian chuẩn hóa.
    mean_X           : list[float] -- trung bình của X.
    std_X            : list[float] -- độ lệch chuẩn của X.

    Trả về
    ------
    list[float] -- danh sách hệ số ở thang đo gốc.
    """
    p = len(beta_scaled)
    beta_original = [0.0] * (p + 1)

    for j in range(p):
        beta_original[j + 1] = beta_scaled[j] / std_X[j]

    intercept = intercept_scaled
    for j in range(p):
        intercept -= beta_original[j + 1] * mean_X[j]

    beta_original[0] = intercept
    return beta_original


def lasso_fit(
    X: list[list[float]],
    y: list[float],
    lam: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> dict:
    """Mô hình hồi quy Lasso với phương pháp hạ tọa độ.

    Tối thiểu hóa: ||y - X * beta||^2 + lam * ||beta||_1.
    Cập nhật tọa độ được thực hiện trên ma trận X đã chuẩn hóa,
    sau đó chuyển đổi beta_hat về thang đo đặc trưng gốc.

    Tham số
    -------
    X        : list[list[float]] -- ma trận đặc trưng.
    y        : list[float]       -- vector mục tiêu.
    lam      : float             -- hệ số điều chuẩn lambda.
    max_iter : int               -- số lượng vòng lặp tối đa.
    tol      : float             -- ngưỡng hội tụ.

    Trả về
    ------
    dict -- beta_hat, y_hat, mean_X, std_X, n_iter.
    """
    validate_regression_input(X, y, lam)

    if max_iter <= 0:
        raise ValueError("max_iter phải lớn hơn 0")

    if tol <= 0:
        raise ValueError("tol phải lớn hơn 0")

    n = len(X)
    p = len(X[0])

    mean_X = mean_columns(X)
    std_X = std_columns(X, mean_X)
    X_scaled = standardize(X, mean_X, std_X)

    intercept_scaled = median(y)
    y_centered = [value - intercept_scaled for value in y]
    beta_scaled = [0.0 for _ in range(p)]
    n_iter = 0

    # Chuyển vị trước ma trận để truy cập cột với độ phức tạp O(1) thay vì trích xuất O(n) mỗi lần
    X_T = [[X_scaled[i][j] for i in range(n)] for j in range(p)]

    # Tính toán trước bình phương chuẩn của cột -- hằng số qua các vòng lặp
    z = [sum(X_T[j][i] * X_T[j][i] for i in range(n)) for j in range(p)]

    residual = y_centered[:]

    for iteration in range(1, max_iter + 1):
        beta_old = beta_scaled[:]

        for j in range(p):
            X_j = X_T[j]
            beta_j_old = beta_scaled[j]
            z_j = z[j]

            if is_zero(z_j):
                beta_scaled[j] = 0.0
                continue

            # rho_j = X_j * (residual + X_j * beta_j_old)
            rho_j = 0.0
            for i in range(n):
                rho_j += X_j[i] * (residual[i] + X_j[i] * beta_j_old)

            beta_scaled[j] = soft_threshold(rho_j, lam) / z_j

            # Cập nhật residual tăng dần
            delta = beta_scaled[j] - beta_j_old
            if delta != 0.0:
                for i in range(n):
                    residual[i] -= X_j[i] * delta

        n_iter = iteration
        max_change = max(abs(beta_scaled[j] - beta_old[j]) for j in range(p))
        if max_change < tol:
            break

    beta_original = coefficients_to_original_scale(
        intercept_scaled,
        beta_scaled,
        mean_X,
        std_X,
    )

    X_orig_bias = add_bias_column(X)
    y_hat = matvec(X_orig_bias, beta_original)

    return {
        "beta_hat": beta_original,
        "y_hat": y_hat,
        "mean_X": mean_X,
        "std_X": std_X,
        "n_iter": n_iter,
    }


def ridge_trace(
    X: list[list[float]],
    y: list[float],
    lambda_grid: list[float] | None = None,
) -> dict:
    """Vẽ đường biểu diễn của các hệ số Ridge theo hệ số lambda trên thang log.

    Mặc định lambda_grid là logspace(-3, 3, 50).

    Tham số
    -------
    X           : list[list[float]]   -- ma trận đặc trưng.
    y           : list[float]         -- vector mục tiêu.
    lambda_grid : list[float] | None  -- lưới các giá trị lambda.

    Trả về
    ------
    dict -- lambda_grid và danh sách coefficients tương ứng.
    """
    import matplotlib.pyplot as plt

    if lambda_grid is None:
        lambda_grid = make_lambda_grid(-3, 3, 50)

    if len(lambda_grid) == 0:
        raise ValueError("lambda_grid không được rỗng")

    coefficients = []
    for lam in lambda_grid:
        coefficients.append(ridge_fit(X, y, lam)["beta_hat"])

    fig, ax = plt.subplots(figsize=(8, 5))
    for j in range(len(coefficients[0])):
        beta_path = [coefficients[i][j] for i in range(len(lambda_grid))]
        label = "intercept" if j == 0 else f"beta_{j}"
        ax.plot(lambda_grid, beta_path, label=label)

    ax.set_xscale("log")
    ax.set_title("Ridge Trace - coefficients by lambda")
    ax.set_xlabel("lambda (log scale)")
    ax.set_ylabel("coefficient value")
    ax.axhline(y=0.0, linestyle="--", linewidth=1, color="gray")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    return {
        "lambda_grid": lambda_grid,
        "coefficients": coefficients,
    }


def lasso_trace(
    X: list[list[float]],
    y: list[float],
    lambda_grid: list[float] | None = None,
    zero_tol: float = 1e-10,
) -> None:
    """Vẽ đường biểu diễn của các hệ số Lasso và báo cáo giá trị lambda đầu tiên mà mỗi hệ số triệt tiêu.

    Mặc định lambda_grid là logspace(-3, 3, 50).

    Tham số
    -------
    X           : list[list[float]]   -- ma trận đặc trưng.
    y           : list[float]         -- vector mục tiêu.
    lambda_grid : list[float] | None  -- lưới các giá trị lambda.
    zero_tol    : float               -- sai số cho phép để coi hệ số bằng 0.
    """
    import matplotlib.pyplot as plt

    if lambda_grid is None:
        lambda_grid = make_lambda_grid(-3, 3, 50)

    if len(lambda_grid) == 0:
        raise ValueError("lambda_grid không được rỗng")

    coefficients = []
    for lam in lambda_grid:
        coefficients.append(lasso_fit(X, y, lam)["beta_hat"])

    zero_events = {}
    p1 = len(coefficients[0])
    for j in range(1, p1):
        for i, lam in enumerate(lambda_grid):
            if abs(coefficients[i][j]) <= zero_tol:
                zero_events[f"beta_{j}"] = lam
                break

    fig, ax = plt.subplots(figsize=(8, 5))
    for j in range(p1):
        beta_path = [coefficients[i][j] for i in range(len(lambda_grid))]
        label = "intercept" if j == 0 else f"beta_{j}"
        ax.plot(lambda_grid, beta_path, label=label)

    for label, lam in zero_events.items():
        ax.axvline(x=lam, linestyle=":", linewidth=1, color="gray")
        ax.text(lam, 0.0, f"{label}=0", rotation=90, va="bottom", fontsize=8)

    ax.set_xscale("log")
    ax.set_title("Lasso Path - sparse coefficients by lambda")
    ax.set_xlabel("lambda (log scale)")
    ax.set_ylabel("coefficient value")
    ax.axhline(y=0.0, linestyle="--", linewidth=1, color="gray")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    plt.show()

    if zero_events:
        print("Giá trị lambda đầu tiên khiến hệ số triệt tiêu hoàn toàn:")
        for label, lam in zero_events.items():
            print(f"  {label}: {lam:.6g}")
    else:
        print("Không có hệ số độ dốc nào triệt tiêu hoàn toàn trên lưới này.")


def run_ridge_tests() -> tuple[int, int]:
    """Chạy unit tests cho ridge_fit từ part1/test_case.py."""
    from part1.test_case import (
        _log,
        run_test_cases,
        test_ridge_empty_X_raises,
        test_ridge_lambda_zero_matches_ols,
        test_ridge_large_lambda_shrinks_coefficients,
        test_ridge_negative_lambda_raises,
        test_ridge_output_shape,
        test_ridge_prediction_accuracy,
    )

    _log.print_suite_header("RIDGE FIT - UNIT TESTS")
    passed_count, total_count = run_test_cases(
        [
            test_ridge_lambda_zero_matches_ols,
            test_ridge_large_lambda_shrinks_coefficients,
            test_ridge_output_shape,
            test_ridge_negative_lambda_raises,
            test_ridge_empty_X_raises,
            test_ridge_prediction_accuracy,
        ]
    )
    _log.print_summary(passed_count, total_count)
    return passed_count, total_count


def run_lasso_tests() -> tuple[int, int]:
    """Chạy unit tests cho lasso_fit từ part1/test_case.py."""
    from part1.test_case import (
        _log,
        run_test_cases,
        test_lasso_handles_constant_feature,
        test_lasso_large_lambda_sets_slopes_to_zero,
        test_lasso_low_lambda_recovers_linear_signal,
        test_lasso_negative_lambda_raises,
        test_lasso_output_shape_and_iterations,
    )

    _log.print_suite_header("LASSO FIT - UNIT TESTS")
    passed_count, total_count = run_test_cases(
        [
            test_lasso_output_shape_and_iterations,
            test_lasso_large_lambda_sets_slopes_to_zero,
            test_lasso_low_lambda_recovers_linear_signal,
            test_lasso_negative_lambda_raises,
            test_lasso_handles_constant_feature,
        ]
    )
    _log.print_summary(passed_count, total_count)
    return passed_count, total_count


if __name__ == "__main__":
    run_ridge_tests()
    run_lasso_tests()
