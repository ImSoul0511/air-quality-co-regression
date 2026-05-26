import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import is_zero, zero_rectify
from utils import identity_matrix, matmul, matvec, transpose


def mean_columns(X: list[list[float]]) -> list[float]:
    """Return the mean of each column in X."""
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
    """Return population standard deviations for each column in X."""
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
    """Standardize X with precomputed column means and standard deviations."""
    X_scaled = []
    for row in X:
        scaled_row = []
        for j in range(len(row)):
            scaled_row.append((row[j] - means[j]) / stds[j])
        X_scaled.append(scaled_row)
    return X_scaled


def add_bias_column(X: list[list[float]]) -> list[list[float]]:
    """Prepend a column of ones to X."""
    return [[1.0] + row[:] for row in X]


def median(values: list[float]) -> float:
    """Return the median of a non-empty list."""
    if len(values) == 0:
        raise ValueError("values must not be empty")

    sorted_values = sorted(values)
    n = len(sorted_values)
    mid = n // 2

    if n % 2 == 1:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def validate_regression_input(X: list[list[float]], y: list[float], lam: float) -> None:
    """Validate shared Ridge/Lasso inputs."""
    if lam < 0:
        raise ValueError("lam must be >= 0")

    if len(X) == 0:
        raise ValueError("X must not be empty")

    if len(X) != len(y):
        raise ValueError("Number of rows in X must match length of y")

    if len(X[0]) == 0:
        raise ValueError("X must contain at least one feature")

    p = len(X[0])
    for row in X:
        if len(row) != p:
            raise ValueError("All rows in X must have the same number of columns")


def solve_system(A: list[list[float]], b: list[float]) -> list[float]:
    """
    Solve Ax = b using Gauss-Jordan elimination with partial pivoting.
    """
    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]

    for col in range(n):
        pivot = col
        for r in range(col + 1, n):
            if abs(M[r][col]) > abs(M[pivot][col]):
                pivot = r

        if is_zero(abs(M[pivot][col])):
            raise ValueError("Linear system has no unique solution")

        M[col], M[pivot] = M[pivot], M[col]

        pivot_val = M[col][col]
        for j in range(col, n + 1):
            M[col][j] /= pivot_val

        for r in range(n):
            if r != col:
                factor = M[r][col]
                for j in range(col, n + 1):
                    M[r][j] -= factor * M[col][j]

    return [zero_rectify(M[i][n]) for i in range(n)]


def ridge_fit(X: list[list[float]], y: list[float], lam: float) -> dict:
    """
    Fit Ridge Regression using the closed-form solution.

    Objective:
        min_beta ||y - X beta||^2 + lam ||beta||^2

    Returns beta_hat on the original feature scale.
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
    """
    Soft-thresholding operator for one-coordinate Lasso updates.
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
    """Convert standardized-space coefficients back to the original X scale."""
    p = len(beta_scaled)
    beta_original = [0.0] * (p + 1)

    for j in range(p):
        beta_original[j + 1] = beta_scaled[j] / std_X[j]

    intercept = intercept_scaled
    for j in range(p):
        intercept -= beta_original[j + 1] * mean_X[j]

    beta_original[0] = intercept
    return beta_original


def lasso_fituti(
    X: list[list[float]],
    y: list[float],
    lam: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> dict:
    """
    Fit Lasso Regression using Coordinate Descent.

    Objective:
        min_beta ||y - X beta||^2 + lam ||beta||_1

    The coordinate updates are done on standardized X. The returned beta_hat is
    converted back to the original feature scale, matching ridge_fit.
    """
    validate_regression_input(X, y, lam)

    if max_iter <= 0:
        raise ValueError("max_iter must be > 0")

    if tol <= 0:
        raise ValueError("tol must be > 0")

    n = len(X)
    p = len(X[0])

    mean_X = mean_columns(X)
    std_X = std_columns(X, mean_X)
    X_scaled = standardize(X, mean_X, std_X)

    intercept_scaled = median(y)
    y_centered = [value - intercept_scaled for value in y]
    beta_scaled = [0.0 for _ in range(p)]
    n_iter = 0

    for iteration in range(1, max_iter + 1):
        beta_old = beta_scaled[:]

        for j in range(p):
            X_j = [X_scaled[i][j] for i in range(n)]

            y_pred_scaled = matvec(X_scaled, beta_scaled)
            r_j = []
            for i in range(n):
                r_j.append(y_centered[i] - y_pred_scaled[i] + X_j[i] * beta_scaled[j])

            rho_j = 0.0
            z_j = 0.0
            for i in range(n):
                rho_j += X_j[i] * r_j[i]
                z_j += X_j[i] * X_j[i]

            if is_zero(z_j):
                beta_scaled[j] = 0.0
            else:
                beta_scaled[j] = soft_threshold(rho_j, lam) / z_j

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


def make_lambda_grid(
    start_exp: float = -3,
    stop_exp: float = 3,
    num: int = 50,
) -> list[float]:
    """Return a logspace-style lambda grid without requiring NumPy."""
    if num <= 0:
        raise ValueError("num must be > 0")

    if num == 1:
        return [10.0**start_exp]

    step = (stop_exp - start_exp) / (num - 1)
    return [10.0 ** (start_exp + i * step) for i in range(num)]


def ridge_trace(
    X: list[list[float]],
    y: list[float],
    lambda_grid: list[float] | None = None,
) -> dict:
    """
    Plot Ridge trace with lambda on a log scale.

    Default lambda_grid is logspace(-3, 3, 50).
    """
    import matplotlib.pyplot as plt

    if lambda_grid is None:
        lambda_grid = make_lambda_grid(-3, 3, 50)

    if len(lambda_grid) == 0:
        raise ValueError("lambda_grid must not be empty")

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
    """
    Plot Lasso path and report the first lambda where each coefficient is zero.

    Default lambda_grid is logspace(-3, 3, 50).
    """
    import matplotlib.pyplot as plt

    if lambda_grid is None:
        lambda_grid = make_lambda_grid(-3, 3, 50)

    if len(lambda_grid) == 0:
        raise ValueError("lambda_grid must not be empty")

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
        print("First lambda where coefficients become exactly zero:")
        for label, lam in zero_events.items():
            print(f"  {label}: {lam:.6g}")
    else:
        print("No non-intercept coefficient became exactly zero on this grid.")


def run_ridge_tests() -> tuple[int, int]:
    """Run ridge_fit unit tests from part1/test_case.py."""
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
    """Run lasso_fit unit tests from part1/test_case.py."""
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
