import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import RANDOM_STATE
from utils import make_lambda_grid


def _validate_data(X: list[list[float]], y: list[float]) -> None:
    if len(X) == 0:
        raise ValueError("X không được rỗng")

    if len(y) == 0:
        raise ValueError("y không được rỗng")

    if len(X) != len(y):
        raise ValueError("Số dòng của X phải khớp với số phần tử của y")

    if len(X[0]) == 0:
        raise ValueError("X phải chứa ít nhất một đặc trưng")

    p = len(X[0])
    for row in X:
        if len(row) != p:
            raise ValueError("Tất cả các dòng của X phải có cùng số lượng cột")


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
    """Dự đoán giá trị mục tiêu y cho ma trận đặc trưng X dựa trên kết quả khớp mô hình.

    Hỗ trợ hai dạng hệ số:
        - beta_hat trên thang đo gốc: [intercept, beta_1,...]
        - beta_scaled trong không gian chuẩn hóa cùng với mean_X/std_X

    Tham số
    -------
    X          : list[list[float]] -- ma trận đặc trưng cần dự đoán.
    fit_result : dict              -- từ điển chứa kết quả khớp mô hình.

    Trả về
    ------
    list[float] -- danh sách các giá trị dự đoán.
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
        raise ValueError("fit_result phải chứa beta_hat hoặc beta_scaled")

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
        raise ValueError("y_true và y_pred phải có cùng kích thước")

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
    """Thực hiện kiểm định chéo k-Fold.

    Tham số
    -------
    X            : list[list[float]] -- ma trận đặc trưng, shape (n, p).
    y            : list[float]       -- vector mục tiêu, shape (n,).
    k            : int               -- số lượng fold.
    model_fn     : callable          -- hàm khớp mô hình: model_fn(X_train, y_train, **kwargs) -> dict.
    model_kwargs : dict              -- các tham số truyền thêm cho model_fn, ví dụ lam=0.1.

    Trả về
    ------
    dict -- các kết quả đánh giá (cv_scores, mean_cv_score, std_cv_score).
    """
    import random
    import math

    _validate_data(X, y)

    if not callable(model_fn):
        raise ValueError("model_fn phải là hàm có thể gọi được")

    n = len(X)
    if k < 2:
        raise ValueError("k phải lớn hơn hoặc bằng 2")

    if k > n:
        raise ValueError("k phải nhỏ hơn hoặc bằng số lượng mẫu")

    indices = list(range(n))
    random.Random(RANDOM_STATE).shuffle(indices)

    fold_sizes = [n // k] * k
    for i in range(n % k):
        fold_sizes[i] += 1

    folds = []
    start = 0
    for size in fold_sizes:
        folds.append(indices[start : start + size])
        start += size

    cv_scores = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = []
        for j in range(k):
            if j != i:
                train_idx.extend(folds[j])

        X_train = _take_rows(X, train_idx)
        y_train = _take_values(y, train_idx)
        X_val = _take_rows(X, val_idx)
        y_val = _take_values(y, val_idx)

        result = model_fn(X_train, y_train, **model_kwargs)
        y_pred_val = predict(X_val, result)
        cv_scores.append(_mse(y_val, y_pred_val))

    mean_score = sum(cv_scores) / len(cv_scores)
    # Tính standard deviation (population std)
    std_score = math.sqrt(sum((x - mean_score)**2 for x in cv_scores) / len(cv_scores))

    return {
        "cv_scores": cv_scores,
        "mean_cv_score": mean_score,
        "std_cv_score": std_score,
    }


def select_lambda_cv(
    X: list[list[float]],
    y: list[float],
    k: int,
    model_fn,
    lambda_grid: list[float] | None = None,
    **model_kwargs,
) -> dict:
    """Lựa chọn hệ số lambda tối ưu bằng phương pháp kiểm định chéo k-Fold.

    Tham số
    -------
    X            : list[list[float]]  -- ma trận đặc trưng.
    y            : list[float]        -- vector mục tiêu.
    k            : int                -- số lượng fold.
    model_fn     : callable           -- hàm khớp mô hình.
    lambda_grid  : list[float] | None -- lưới giá trị lambda lựa chọn.
    model_kwargs : dict               -- các tham số truyền cho model_fn.

    Trả về
    ------
    dict -- kết quả lựa chọn bao gồm lambda tối ưu và thông tin MSE tương ứng.
    """
    import matplotlib.pyplot as plt

    if lambda_grid is None:
        lambda_grid = make_lambda_grid(-3, 3, 50)

    if len(lambda_grid) == 0:
        raise ValueError("lambda_grid không được rỗng")

    cv_means = []
    cv_stds = []
    cv_results = []

    for lam in lambda_grid:
        result = kfold_cv(X, y, k=k, model_fn=model_fn, lam=lam, **model_kwargs)
        cv_results.append(result)
        cv_means.append(result["mean_cv_score"])
        cv_stds.append(result["std_cv_score"])

    best_idx = 0
    for i in range(1, len(cv_means)):
        if cv_means[i] < cv_means[best_idx]:
            best_idx = i
            
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
    plt.close(fig)

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
    """Chạy unit tests cho kfold_cv từ part1/test_case.py."""
    from part1.test_case import (
        _log,
        run_test_cases,
        test_cv_invalid_model_fn,
        test_cv_k_out_of_bounds,
        test_cv_mismatched_lengths_raises,
        test_cv_output_shape_and_keys,
        test_cv_perfect_fit,
    )

    _log.print_suite_header("CROSS VALIDATION - UNIT TESTS")
    passed_count, total_count = run_test_cases(
        [
            test_cv_output_shape_and_keys,
            test_cv_k_out_of_bounds,
            test_cv_perfect_fit,
            test_cv_mismatched_lengths_raises,
            test_cv_invalid_model_fn,
        ]
    )
    _log.print_summary(passed_count, total_count)
    return passed_count, total_count


if __name__ == "__main__":
    run_cv_tests()
