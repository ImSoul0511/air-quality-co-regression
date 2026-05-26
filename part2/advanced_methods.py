import math
import random
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import RANDOM_STATE
from part1.ridge_lasso import solve_system
from utils import identity_matrix, matvec


# =========================================================
# VALIDATION HELPERS
# =========================================================

def validate_X(X: list[list[float]], name: str = "X") -> None:
    """Kiểm tra X không rỗng và mọi row có cùng số feature."""
    if len(X) == 0:
        raise ValueError(f"{name} must not be empty")
    p = len(X[0])
    if p == 0:
        raise ValueError(f"{name} must have at least one feature")
    for i, row in enumerate(X):
        if len(row) != p:
            raise ValueError(
                f"{name} row {i} has {len(row)} features, expected {p}"
            )


def validate_xy(X: list[list[float]], y: list[float]) -> None:
    """Kiểm tra X, y hợp lệ và số dòng khớp nhau."""
    validate_X(X)
    if len(y) == 0:
        raise ValueError("y must not be empty")
    if len(X) != len(y):
        raise ValueError(
            f"X has {len(X)} rows but y has {len(y)} elements"
        )


def validate_non_negative(value: float, name: str) -> None:
    """Kiểm tra giá trị >= 0 (dùng cho lam)."""
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


def validate_positive(value: float, name: str) -> None:
    """Kiểm tra giá trị > 0 (dùng cho length_scale)."""
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


# =========================================================
# TRAIN / TEST SPLIT HELPER
# =========================================================

def train_test_split(df, test_size: float = 0.2, seed: int = RANDOM_STATE):
    """Split một pandas DataFrame thành train và test set.

    Args:
        df: pandas DataFrame đầu vào.
        test_size: Tỷ lệ test set (0 < test_size < 1).
        seed: Random seed để tái lập kết quả.

    Returns:
        tuple: (df_train, df_test)
    """
    n = len(df)
    n_test = max(1, int(round(n * test_size)))
    test_idx = df.sample(n=n_test, random_state=seed).index
    df_test = df.loc[test_idx].copy()
    df_train = df.drop(index=test_idx).copy()
    return df_train, df_test


# =========================================================
# RBF KERNEL (không NumPy)
# =========================================================

def squared_distance(x1: list[float], x2: list[float]) -> float:
    """Tính bình phương khoảng cách Euclidean giữa hai vector.

    ||x1 - x2||^2 = sum((x1[k] - x2[k])^2)
    """
    if len(x1) != len(x2):
        raise ValueError(
            f"Vectors must have same length: {len(x1)} vs {len(x2)}"
        )
    return sum((x1[k] - x2[k]) ** 2 for k in range(len(x1)))


def rbf_kernel_value(
    x1: list[float],
    x2: list[float],
    length_scale: float = 1.0,
) -> float:
    """Tính giá trị RBF kernel giữa hai điểm.

    k(x1, x2) = exp(-||x1 - x2||^2 / (2 * length_scale^2))
    """
    validate_positive(length_scale, "length_scale")
    dist2 = squared_distance(x1, x2)
    return math.exp(-dist2 / (2.0 * length_scale ** 2))


def rbf_kernel(
    X1: list[list[float]],
    X2: list[list[float]],
    length_scale: float = 1.0,
) -> list[list[float]]:
    """Tính Gram matrix K[i][j] = k(X1[i], X2[j]).

    Returns:
        K: ma trận kích thước len(X1) x len(X2)
    """
    validate_positive(length_scale, "length_scale")
    validate_X(X1, "X1")
    validate_X(X2, "X2")

    denom = 2.0 * length_scale ** 2
    K = []
    for i in range(len(X1)):
        row = []
        for j in range(len(X2)):
            dist2 = squared_distance(X1[i], X2[j])
            row.append(math.exp(-dist2 / denom))
        K.append(row)
    return K


# =========================================================
# HELPER: THÊM LAMBDA + JITTER VÀO ĐƯỜNG CHÉO GRAM MATRIX
# =========================================================

def _add_diagonal(K: list[list[float]], value: float) -> list[list[float]]:
    """Trả về bản sao của K với K[i][i] += value."""
    n = len(K)
    A = [K[i][:] for i in range(n)]
    for i in range(n):
        A[i][i] += value
    return A


# =========================================================
# KERNEL RIDGE FIT
# =========================================================

def kernel_ridge_fit(
    X_train: list[list[float]],
    y_train: list[float],
    lam: float,
    length_scale: float,
    jitter: float = 1e-10,
) -> dict:
    """Fit Kernel Ridge Regression với RBF kernel.

    Giải hệ: (K + (lam + jitter) * I) * alpha = y_train
    Trong đó K[i][j] = k_RBF(x_i, x_j)

    Args:
        X_train: Dữ liệu train, list[list[float]], shape (n, p).
        y_train: Target train, list[float], shape (n,).
        lam: Hệ số regularization (>= 0).
        length_scale: Length scale của RBF kernel (> 0).
        jitter: Giá trị nhỏ thêm vào đường chéo để ổn định số.

    Returns:
        dict chứa: model_type, alpha, X_train (copy), length_scale, lam, jitter, y_hat.
    """
    validate_xy(X_train, y_train)
    validate_non_negative(lam, "lam")
    validate_positive(length_scale, "length_scale")

    n = len(X_train)
    if n > 2000:
        raise ValueError(
            f"Kernel Ridge is O(n^3); n_train={n} > 2000. "
            "Pass a smaller subset via sample_rows()."
        )

    K = rbf_kernel(X_train, X_train, length_scale)
    A = _add_diagonal(K, lam + jitter)
    alpha = solve_system(A, y_train)

    y_hat = matvec(K, alpha)

    return {
        "model_type": "kernel_ridge_rbf",
        "alpha": alpha,
        "X_train": [row[:] for row in X_train],
        "length_scale": length_scale,
        "lam": lam,
        "jitter": jitter,
        "y_hat": y_hat,
    }


# =========================================================
# KERNEL RIDGE PREDICT (streaming, tiết kiệm bộ nhớ)
# =========================================================

def kernel_ridge_predict(
    model: dict,
    X_test: list[list[float]],
) -> list[float]:
    """Dự đoán từ mô hình KRR đã fit.

    Dùng streaming prediction để tránh tạo K_test đầy đủ:
        y_pred[i] = sum_j k(x_test_i, x_train_j) * alpha[j]

    Args:
        model: dict trả về từ kernel_ridge_fit.
        X_test: Dữ liệu test, list[list[float]], shape (m, p).

    Returns:
        list[float] chứa m giá trị dự đoán.
    """
    validate_X(X_test, "X_test")

    alpha = model["alpha"]
    X_tr = model["X_train"]
    length_scale = model["length_scale"]
    n_train = len(X_tr)

    y_pred = []
    for x in X_test:
        pred = 0.0
        for j in range(n_train):
            pred += rbf_kernel_value(x, X_tr[j], length_scale) * alpha[j]
        y_pred.append(pred)
    return y_pred


# =========================================================
# REGRESSION METRICS (thuần Python)
# =========================================================

def regression_metrics(
    y_true: list[float],
    y_pred: list[float],
) -> dict:
    """Tính MAE, RMSE, R2, RSS, TSS từ thuần Python.

    Args:
        y_true: Giá trị thực, list[float].
        y_pred: Giá trị dự đoán, list[float].

    Returns:
        dict: {"MAE", "RMSE", "R2", "RSS", "TSS"}
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have same length: "
            f"{len(y_true)} vs {len(y_pred)}"
        )
    if len(y_true) == 0:
        raise ValueError("y_true must not be empty")

    n = len(y_true)
    y_mean = sum(y_true) / n

    rss = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n))
    tss = sum((y_true[i] - y_mean) ** 2 for i in range(n))
    mae = sum(abs(y_true[i] - y_pred[i]) for i in range(n)) / n
    rmse = math.sqrt(rss / n)
    r2 = 1.0 - rss / tss if tss > 0.0 else 0.0

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "RSS": rss,
        "TSS": tss,
    }


# =========================================================
# SAMPLE ROWS HELPER
# =========================================================

def sample_rows(
    X: list[list[float]],
    y: list[float],
    max_rows: int,
    seed: int = RANDOM_STATE,
) -> tuple[list[list[float]], list[float]]:
    """Lấy ngẫu nhiên tối đa max_rows hàng từ X, y theo seed.

    Nếu len(X) <= max_rows, trả nguyên bản sao.
    Gọi hai lần cùng seed sẽ trả cùng subset.

    Args:
        X: list[list[float]], shape (n, p).
        y: list[float], shape (n,).
        max_rows: Số hàng tối đa cần lấy.
        seed: Random seed để tái lập.

    Returns:
        tuple (X_sub, y_sub)
    """
    validate_xy(X, y)
    if max_rows <= 0:
        raise ValueError("max_rows must be > 0")

    n = len(X)
    if n <= max_rows:
        return [row[:] for row in X], y[:]

    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    chosen = sorted(indices[:max_rows])

    X_sub = [X[i][:] for i in chosen]
    y_sub = [y[i] for i in chosen]
    return X_sub, y_sub


# =========================================================
# K-FOLD INDICES (thuần Python)
# =========================================================

def kfold_indices(
    n: int,
    k: int,
    seed: int = RANDOM_STATE,
) -> list[list[int]]:
    """Tạo danh sách k fold, mỗi fold là list các index trong validation set.

    Các fold có kích thước cân bằng: n % k fold đầu có thêm 1 phần tử.

    Args:
        n: Tổng số mẫu.
        k: Số fold.
        seed: Random seed.

    Returns:
        list[list[int]]: Danh sách k fold, mỗi fold là list index.
    """
    if k < 2:
        raise ValueError("k must be >= 2")
    if n < k:
        raise ValueError(f"n={n} must be >= k={k}")

    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    fold_sizes = [n // k] * k
    for i in range(n % k):
        fold_sizes[i] += 1

    folds = []
    start = 0
    for size in fold_sizes:
        folds.append(indices[start : start + size])
        start += size
    return folds


def _std_sample(values: list[float]) -> float:
    """Sample standard deviation (chia n-1)."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


# =========================================================
# KERNEL RIDGE CV (grid search 2D: lambda × length_scale)
# =========================================================

def kernel_ridge_cv(
    X: list[list[float]],
    y: list[float],
    lambda_grid: list[float],
    length_scale_grid: list[float],
    k: int = 5,
    seed: int = RANDOM_STATE,
) -> dict:
    """Cross-validation 2D grid search cho KRR.

    Với mỗi cặp (lam, length_scale), chạy k-fold CV và ghi lại MSE trung bình.
    Chọn cặp tham số có mean validation MSE nhỏ nhất.

    Args:
        X: list[list[float]], shape (n, p).
        y: list[float], shape (n,).
        lambda_grid: Danh sách các giá trị lambda cần thử.
        length_scale_grid: Danh sách các giá trị length_scale cần thử.
        k: Số fold (>= 2).
        seed: Random seed.

    Returns:
        dict: {
            "best_lam", "best_length_scale", "best_cv_score",
            "cv_results": list of {lam, length_scale, fold_scores, mean_score, std_score}
        }
    """
    validate_xy(X, y)
    if len(lambda_grid) == 0:
        raise ValueError("lambda_grid must not be empty")
    if len(length_scale_grid) == 0:
        raise ValueError("length_scale_grid must not be empty")

    n = len(X)
    folds = kfold_indices(n, k, seed)

    cv_results = []
    best_score = float("inf")
    best_lam = lambda_grid[0]
    best_ls = length_scale_grid[0]

    for lam in lambda_grid:
        for ls in length_scale_grid:
            fold_scores = []

            for fold_idx in range(k):
                val_idx = set(folds[fold_idx])
                train_idx = [i for i in range(n) if i not in val_idx]
                val_idx_list = folds[fold_idx]

                X_tr = [X[i] for i in train_idx]
                y_tr = [y[i] for i in train_idx]
                X_val = [X[i] for i in val_idx_list]
                y_val = [y[i] for i in val_idx_list]

                try:
                    model = kernel_ridge_fit(X_tr, y_tr, lam=lam, length_scale=ls)
                    y_val_pred = kernel_ridge_predict(model, X_val)
                    n_val = len(y_val)
                    mse = sum(
                        (y_val[i] - y_val_pred[i]) ** 2 for i in range(n_val)
                    ) / n_val
                    fold_scores.append(mse)
                except (ValueError, ZeroDivisionError):
                    fold_scores.append(float("inf"))

            mean_score = sum(fold_scores) / len(fold_scores)
            std_score = _std_sample(fold_scores)

            cv_results.append({
                "lam": lam,
                "length_scale": ls,
                "fold_scores": fold_scores,
                "mean_score": mean_score,
                "std_score": std_score,
            })

            if mean_score < best_score:
                best_score = mean_score
                best_lam = lam
                best_ls = ls

    return {
        "best_lam": best_lam,
        "best_length_scale": best_ls,
        "best_cv_score": best_score,
        "cv_results": cv_results,
    }


# =========================================================
# UNIT TESTS
# =========================================================

def _run_tests() -> tuple[int, int]:
    """Chạy unit tests cơ bản cho advanced_methods.py."""
    passed = 0
    total = 0

    def check(label: str, condition: bool):
        nonlocal passed, total
        total += 1
        status = "PASS" if condition else "FAIL"
        if condition:
            passed += 1
        print(f"  [{status}] {label}")

    def check_raises(label: str, fn, *args, **kwargs):
        nonlocal passed, total
        total += 1
        try:
            fn(*args, **kwargs)
            print(f"  [FAIL] {label} — expected ValueError, got none")
        except ValueError:
            passed += 1
            print(f"  [PASS] {label}")
        except Exception as e:
            print(f"  [FAIL] {label} — unexpected exception: {e}")

    print("\n=== Test 1: RBF kernel cơ bản ===")
    X2 = [[0.0], [1.0]]
    K = rbf_kernel(X2, X2, length_scale=1.0)
    check("K[0][0] == 1.0", abs(K[0][0] - 1.0) < 1e-12)
    check("K[1][1] == 1.0", abs(K[1][1] - 1.0) < 1e-12)
    check("K[0][1] == K[1][0] (symmetry)", abs(K[0][1] - K[1][0]) < 1e-12)
    check(
        "K[0][1] ≈ exp(-0.5)",
        abs(K[0][1] - math.exp(-0.5)) < 1e-9,
    )

    print("\n=== Test 2: Validate length_scale <= 0 ===")
    check_raises(
        "rbf_kernel với length_scale=0.0 phải raise ValueError",
        rbf_kernel, [[1.0]], [[1.0]], length_scale=0.0,
    )
    check_raises(
        "rbf_kernel_value với length_scale=-1.0 phải raise ValueError",
        rbf_kernel_value, [1.0], [1.0], -1.0,
    )

    print("\n=== Test 3: Fit/predict shape và MSE ===")
    X3 = [[0.0], [1.0], [2.0]]
    y3 = [0.0, 1.0, 4.0]
    model3 = kernel_ridge_fit(X3, y3, lam=0.01, length_scale=1.0)
    y_hat3 = kernel_ridge_predict(model3, X3)
    check("len(alpha) == len(y)", len(model3["alpha"]) == len(y3))
    check("len(y_hat) == len(y)", len(y_hat3) == len(y3))
    mse3 = sum((y3[i] - y_hat3[i]) ** 2 for i in range(len(y3))) / len(y3)
    check(f"MSE on train < 0.1 (MSE={mse3:.4f})", mse3 < 0.1)
    check("model_type correct", model3["model_type"] == "kernel_ridge_rbf")

    print("\n=== Test 4: sample_rows tái lập ===")
    X4 = [[float(i)] for i in range(50)]
    y4 = [float(i) for i in range(50)]
    X4a, y4a = sample_rows(X4, y4, max_rows=20, seed=7)
    X4b, y4b = sample_rows(X4, y4, max_rows=20, seed=7)
    check("sample_rows cùng seed cho cùng X_sub", X4a == X4b)
    check("sample_rows cùng seed cho cùng y_sub", y4a == y4b)
    check("sample_rows trả đúng kích thước", len(X4a) == 20)
    Xfull, yfull = sample_rows(X4, y4, max_rows=100, seed=0)
    check("sample_rows khi max_rows >= n trả toàn bộ", len(Xfull) == 50)

    print("\n=== Test 5: CV trả về tham số từ grid ===")
    X5 = [[float(i)] for i in range(30)]
    y5 = [float(i) * 0.5 for i in range(30)]
    lam_grid = [0.01, 0.1, 1.0]
    ls_grid = [1.0, 2.0]
    cv_result = kernel_ridge_cv(X5, y5, lam_grid, ls_grid, k=3, seed=42)
    check("best_lam in lambda_grid", cv_result["best_lam"] in lam_grid)
    check("best_length_scale in length_scale_grid", cv_result["best_length_scale"] in ls_grid)
    expected_n = len(lam_grid) * len(ls_grid)
    check(
        f"len(cv_results) == {expected_n}",
        len(cv_result["cv_results"]) == expected_n,
    )

    print(f"\n{'='*40}")
    print(f"Kết quả: {passed}/{total} tests PASSED")
    print(f"{'='*40}\n")
    return passed, total


if __name__ == "__main__":
    _run_tests()
