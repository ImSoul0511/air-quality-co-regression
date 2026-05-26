import math
import random
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import RANDOM_STATE
from part1.ridge_lasso import solve_system
from utils import identity_matrix, matvec, vector_sub, vector_add, dot, transpose, matmul, inverse, scalar_multiply
from part1.ols_implementation import model_metrics


# =========================================================
# SAMPLE ROWS HELPER (module-level, dùng chung cho mọi model)
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
# KERNEL RIDGE REGRESSION CLASS
# =========================================================

class KernelRidgeRegression:
    """Kernel Ridge Regression với RBF kernel (thuần Python).

    Gom toàn bộ logic: RBF kernel, fit, predict, cross-validation
    vào một class duy nhất để tách biệt với các phương pháp khác
    (ví dụ: Bayesian).

    Usage:
        krr = KernelRidgeRegression()
        model = krr.fit(X_train, y_train, lam=1.0, length_scale=5.0)
        y_pred = krr.predict(model, X_test)
        cv = krr.cross_validate(X, y, lam_grid, ls_grid, k=5)
    """

    # ----- RBF Kernel -----

    @staticmethod
    def rbf_kernel_value(
        x1: list[float],
        x2: list[float],
        length_scale: float = 1.0,
    ) -> float:
        """Tính giá trị RBF kernel giữa hai điểm.

        k(x1, x2) = exp(-||x1 - x2||^2 / (2 * length_scale^2))
        """
        v_diff = vector_sub(x1, x2)
        dist2 = dot(v_diff, v_diff)
        return math.exp(-dist2 / (2.0 * length_scale ** 2))

    @staticmethod
    def rbf_kernel(
        X1: list[list[float]],
        X2: list[list[float]],
        length_scale: float = 1.0,
    ) -> list[list[float]]:
        """Tính Gram matrix K[i][j] = k(X1[i], X2[j]).

        Returns:
            K: ma trận kích thước len(X1) x len(X2)
        """
        denom = 2.0 * length_scale ** 2
        K = []
        for i in range(len(X1)):
            row = []
            for j in range(len(X2)):
                v_diff = vector_sub(X1[i], X2[j])
                dist2 = dot(v_diff, v_diff)
                row.append(math.exp(-dist2 / denom))
            K.append(row)
        return K

    # ----- Helper -----

    @staticmethod
    def _add_diagonal(K: list[list[float]], value: float) -> list[list[float]]:
        """Trả về bản sao của K với K[i][i] += value."""
        n = len(K)
        A = [K[i][:] for i in range(n)]
        for i in range(n):
            A[i][i] += value
        return A

    @staticmethod
    def _std_sample(values: list[float]) -> float:
        """Sample standard deviation (chia n-1)."""
        n = len(values)
        if n < 2:
            return 0.0
        mean = sum(values) / n
        return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))

    @staticmethod
    def _kfold_indices(
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

    # ----- Fit -----

    @staticmethod
    def fit(
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
        n = len(X_train)
        if n > 2000:
            raise ValueError(
                f"Kernel Ridge is O(n^3); n_train={n} > 2000. "
                "Pass a smaller subset via sample_rows()."
            )

        K = KernelRidgeRegression.rbf_kernel(X_train, X_train, length_scale)
        A = KernelRidgeRegression._add_diagonal(K, lam + jitter)
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

    # ----- Predict -----

    @staticmethod
    def predict(
        model: dict,
        X_test: list[list[float]],
    ) -> list[float]:
        """Dự đoán từ mô hình KRR đã fit.

        Dùng streaming prediction để tránh tạo K_test đầy đủ:
            y_pred[i] = sum_j k(x_test_i, x_train_j) * alpha[j]

        Args:
            model: dict trả về từ fit().
            X_test: Dữ liệu test, list[list[float]], shape (m, p).

        Returns:
            list[float] chứa m giá trị dự đoán.
        """
        alpha = model["alpha"]
        X_tr = model["X_train"]
        length_scale = model["length_scale"]
        n_train = len(X_tr)

        y_pred = []
        for x in X_test:
            pred = 0.0
            for j in range(n_train):
                pred += KernelRidgeRegression.rbf_kernel_value(x, X_tr[j], length_scale) * alpha[j]
            y_pred.append(pred)
        return y_pred

    # ----- Cross-Validation -----

    @staticmethod
    def cross_validate(
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
        if len(lambda_grid) == 0:
            raise ValueError("lambda_grid must not be empty")
        if len(length_scale_grid) == 0:
            raise ValueError("length_scale_grid must not be empty")

        n = len(X)
        folds = KernelRidgeRegression._kfold_indices(n, k, seed)

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
                        model = KernelRidgeRegression.fit(X_tr, y_tr, lam=lam, length_scale=ls)
                        y_val_pred = KernelRidgeRegression.predict(model, X_val)
                        n_val = len(y_val)
                        mse = sum(
                            (y_val[i] - y_val_pred[i]) ** 2 for i in range(n_val)
                        ) / n_val
                        fold_scores.append(mse)
                    except (ValueError, ZeroDivisionError):
                        fold_scores.append(float("inf"))

                mean_score = sum(fold_scores) / len(fold_scores)
                std_score = KernelRidgeRegression._std_sample(fold_scores)

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
# BAYESIAN LINEAR REGRESSION CLASS
# =========================================================

class BayesianLinearRegression:
    """Bayesian Linear Regression (thuần Python).

    Prior:      β ~ N(m0, S0)
    Likelihood: y | X, β ~ N(X_bias @ β, σ² I)
    Posterior:  β | X, y ~ N(m_n, S_n)

    Công thức posterior:
        S_n_inv = S0_inv + (1/σ²) X_bias^T X_bias
        S_n     = S_n_inv^{-1}
        m_n     = S_n (S0_inv m0 + (1/σ²) X_bias^T y)

    Dự đoán:
        ŷ     = X_bias_new @ m_n
        σ²_pred = σ² + x^T S_n x   (predictive variance cho mỗi điểm)

    Usage:
        blr = BayesianLinearRegression()
        sigma2 = blr.estimate_sigma2(X_train, y_train)
        model  = blr.fit(X_train, y_train, sigma2, alpha=1.0)
        y_pred, y_lower, y_upper = blr.predict(model, X_test, sigma2)
        cv     = blr.cross_validate(X, y, alpha_grid, k=5)
    """

    # ----- Helpers -----

    @staticmethod
    def _add_bias(X: list[list[float]]) -> list[list[float]]:
        """Thêm cột 1 (intercept) vào đầu X → shape (n, p+1)."""
        return [[1.0] + row for row in X]

    @staticmethod
    def _matrix_add(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
        """Cộng hai ma trận cùng kích thước."""
        n = len(A)
        m = len(A[0])
        return [[A[i][j] + B[i][j] for j in range(m)] for i in range(n)]

    @staticmethod
    def _scalar_matrix(k: float, A: list[list[float]]) -> list[list[float]]:
        """Nhân ma trận với scalar."""
        n = len(A)
        m = len(A[0])
        return [[k * A[i][j] for j in range(m)] for i in range(n)]

    # ----- Estimate σ² từ OLS -----

    @staticmethod
    def estimate_sigma2(
        X_train: list[list[float]],
        y_train: list[float],
    ) -> float:
        """Ước lượng σ² (noise variance) từ OLS residuals.

        σ² = RSS / (n - p - 1)

        Args:
            X_train: list[list[float]], shape (n, p).
            y_train: list[float], shape (n,).

        Returns:
            float: Ước lượng σ².
        """
        from part1.ols_implementation import ols_fit
        result = ols_fit(X_train, y_train)
        return result['sigma2_hat']

    # ----- Fit -----

    @staticmethod
    def fit(
        X_train: list[list[float]],
        y_train: list[float],
        sigma2: float,
        alpha: float = 1.0,
        m0: list[float] = None,
        S0: list[list[float]] = None,
    ) -> dict:
        """Fit Bayesian Linear Regression.

        Tính posterior parameters (m_n, S_n) từ prior và likelihood.

        Args:
            X_train: list[list[float]], shape (n, p). CHƯA có bias.
            y_train: list[float], shape (n,).
            sigma2: Noise variance (σ²). Dùng estimate_sigma2() để ước lượng.
            alpha: Prior precision. S0 = (1/alpha) * I nếu S0 không được truyền.
                   alpha nhỏ → prior yếu (ít ràng buộc).
                   alpha lớn → prior mạnh (hệ số bị kéo về m0).
            m0: Prior mean, shape (p+1,). Mặc định = vector 0.
            S0: Prior covariance, shape (p+1, p+1). Mặc định = (1/alpha)*I.

        Returns:
            dict: model_type, m_n, S_n, sigma2, alpha.
        """
        n = len(X_train)
        p = len(X_train[0])
        p1 = p + 1  # bao gồm intercept

        X_bias = BayesianLinearRegression._add_bias(X_train)
        X_bias_T = transpose(X_bias)

        # Prior defaults
        if m0 is None:
            m0 = [0.0] * p1
        if S0 is None:
            S0 = BayesianLinearRegression._scalar_matrix(1.0 / alpha, identity_matrix(p1))

        # S0_inv = alpha * I (nếu dùng default)
        S0_inv = inverse(S0)

        # XTX = X_bias^T @ X_bias  (p+1 x p+1)
        XTX = matmul(X_bias_T, X_bias)

        # S_n_inv = S0_inv + (1/σ²) * XTX
        XTX_scaled = BayesianLinearRegression._scalar_matrix(1.0 / sigma2, XTX)
        S_n_inv = BayesianLinearRegression._matrix_add(S0_inv, XTX_scaled)

        # S_n = S_n_inv^{-1}
        S_n = inverse(S_n_inv)

        # m_n = S_n @ (S0_inv @ m0 + (1/σ²) * X_bias^T @ y)
        S0_inv_m0 = matvec(S0_inv, m0)
        XTy = matvec(X_bias_T, y_train)
        XTy_scaled = scalar_multiply(XTy, 1.0 / sigma2)
        rhs = vector_add(S0_inv_m0, XTy_scaled)
        m_n = matvec(S_n, rhs)

        # Fitted values
        y_hat = matvec(X_bias, m_n)

        return {
            "model_type": "bayesian_lr",
            "m_n": m_n,
            "S_n": S_n,
            "sigma2": sigma2,
            "alpha": alpha,
            "y_hat": y_hat,
            "beta_hat": m_n,  # Tương thích với model_metrics
        }

    # ----- Predict -----

    @staticmethod
    def predict(
        model: dict,
        X_test: list[list[float]],
        sigma2: float = None,
        credible_interval: float = 0.95,
    ) -> tuple[list[float], list[float], list[float]]:
        """Dự đoán từ mô hình BLR đã fit.

        Trả về mean prediction và credible interval.

        Args:
            model: dict trả về từ fit().
            X_test: list[list[float]], shape (m, p). CHƯA có bias.
            sigma2: Noise variance. Nếu None, dùng sigma2 từ model.
            credible_interval: Mức credible interval (mặc định 95%).

        Returns:
            tuple: (y_mean, y_lower, y_upper)
                y_mean: list[float] - giá trị dự đoán trung bình.
                y_lower: list[float] - cận dưới credible interval.
                y_upper: list[float] - cận trên credible interval.
        """
        m_n = model["m_n"]
        S_n = model["S_n"]
        if sigma2 is None:
            sigma2 = model["sigma2"]

        # z-score cho credible interval (xấp xỉ Normal)
        # Dùng bảng tra cứng cho các mức phổ biến thay vì scipy
        z_table = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}
        z = z_table.get(credible_interval, 1.9600)

        X_bias = BayesianLinearRegression._add_bias(X_test)

        y_mean = []
        y_lower = []
        y_upper = []

        for x in X_bias:
            # ŷ = x^T @ m_n
            pred = dot(x, m_n)

            # Predictive variance: σ²_pred = σ² + x^T S_n x
            Sx = matvec(S_n, x)
            var_pred = sigma2 + dot(x, Sx)
            std_pred = math.sqrt(max(var_pred, 0.0))

            y_mean.append(pred)
            y_lower.append(pred - z * std_pred)
            y_upper.append(pred + z * std_pred)

        return y_mean, y_lower, y_upper

    # ----- Cross-Validation -----

    @staticmethod
    def cross_validate(
        X: list[list[float]],
        y: list[float],
        alpha_grid: list[float],
        k: int = 5,
        seed: int = RANDOM_STATE,
    ) -> dict:
        """Cross-validation 1D grid search cho BLR (tìm alpha tối ưu).

        Với mỗi alpha, chạy k-fold CV và ghi lại MSE trung bình.
        σ² được ước lượng tự động trong mỗi fold từ OLS trên tập train fold.

        Args:
            X: list[list[float]], shape (n, p).
            y: list[float], shape (n,).
            alpha_grid: Danh sách các giá trị alpha (prior precision) cần thử.
            k: Số fold (>= 2).
            seed: Random seed.

        Returns:
            dict: {
                "best_alpha", "best_cv_score",
                "cv_results": list of {alpha, fold_scores, mean_score, std_score}
            }
        """
        if len(alpha_grid) == 0:
            raise ValueError("alpha_grid must not be empty")

        n = len(X)
        folds = KernelRidgeRegression._kfold_indices(n, k, seed)

        cv_results = []
        best_score = float("inf")
        best_alpha = alpha_grid[0]

        for alpha in alpha_grid:
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
                    sigma2 = BayesianLinearRegression.estimate_sigma2(X_tr, y_tr)
                    model = BayesianLinearRegression.fit(X_tr, y_tr, sigma2=sigma2, alpha=alpha)
                    y_val_pred, _, _ = BayesianLinearRegression.predict(model, X_val, sigma2)
                    n_val = len(y_val)
                    mse = sum(
                        (y_val[i] - y_val_pred[i]) ** 2 for i in range(n_val)
                    ) / n_val
                    fold_scores.append(mse)
                except (ValueError, ZeroDivisionError):
                    fold_scores.append(float("inf"))

            mean_score = sum(fold_scores) / len(fold_scores)
            std_score = KernelRidgeRegression._std_sample(fold_scores)

            cv_results.append({
                "alpha": alpha,
                "fold_scores": fold_scores,
                "mean_score": mean_score,
                "std_score": std_score,
            })

            if mean_score < best_score:
                best_score = mean_score
                best_alpha = alpha

        return {
            "best_alpha": best_alpha,
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

    krr = KernelRidgeRegression()

    def check(label: str, condition: bool):
        nonlocal passed, total
        total += 1
        status = "PASS" if condition else "FAIL"
        if condition:
            passed += 1
        print(f"  [{status}] {label}")

    print("\n=== Test 1: RBF kernel cơ bản ===")
    X2 = [[0.0], [1.0]]
    K = krr.rbf_kernel(X2, X2, length_scale=1.0)
    check("K[0][0] == 1.0", abs(K[0][0] - 1.0) < 1e-12)
    check("K[1][1] == 1.0", abs(K[1][1] - 1.0) < 1e-12)
    check("K[0][1] == K[1][0] (symmetry)", abs(K[0][1] - K[1][0]) < 1e-12)
    check(
        "K[0][1] ≈ exp(-0.5)",
        abs(K[0][1] - math.exp(-0.5)) < 1e-9,
    )

    print("\n=== Test 3: Fit/predict shape và MSE ===")
    X3 = [[0.0], [1.0], [2.0]]
    y3 = [0.0, 1.0, 4.0]
    model3 = krr.fit(X3, y3, lam=0.01, length_scale=1.0)
    y_hat3 = krr.predict(model3, X3)
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
    cv_result = krr.cross_validate(X5, y5, lam_grid, ls_grid, k=3, seed=42)
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
