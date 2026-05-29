import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import random
from config import RANDOM_STATE
from utils import (
    identity_matrix, matvec, vector_sub, vector_add, dot, transpose,
    matmul, inverse, scalar_multiply, solve_system
)

def sample_rows(
    X: list[list[float]],
    y: list[float],
    max_rows: int,
    seed: int = RANDOM_STATE,
) -> tuple[list[list[float]], list[float]]:
    """Lấy ngẫu nhiên tối đa max_rows hàng từ X, y theo seed.

    Nếu len(X) <= max_rows, trả nguyên bản sao.
    Gọi hai lần cùng seed sẽ trả cùng subset.

    Tham số
    -------
    X        : list[list[float]], shape (n, p) -- tập đặc trưng đầu vào.
    y        : list[float], shape (n,) -- tập mục tiêu đầu vào.
    max_rows : int -- số hàng tối đa cần lấy.
    seed     : int -- hạt giống ngẫu nhiên để tái lập kết quả.

    Trả về
    ------
    tuple[list[list[float]], list[float]] -- bộ (X_sub, y_sub) đã lấy mẫu.
    """
    if max_rows <= 0:
        raise ValueError("max_rows phải lớn hơn 0")

    n = len(X)
    if n <= max_rows:
        return [row[:] for row in X], y[:]

    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    chosen = sorted(indices[:max_rows])

    X_sub = [X[i][:] for i in chosen]
    y_sub = [y[i] for i in chosen]
    return X_sub, y_sub

class KernelRidgeRegression:
    """
    Kernel Ridge Regression với RBF kernel.
    """

    @staticmethod
    def rbf_kernel_value(
        x1: list[float],
        x2: list[float],
        length_scale: float = 1.0,
    ) -> float:
        """Tính giá trị RBF kernel giữa hai điểm.

        Tham số
        -------
        x1           : list[float] -- điểm dữ liệu thứ nhất.
        x2           : list[float] -- điểm dữ liệu thứ hai.
        length_scale : float -- tham số tỷ lệ độ dài (mặc định 1.0).

        Trả về
        ------
        float -- giá trị k(x1, x2) = exp(-||x1 - x2||^2 / (2 * length_scale^2)).
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
        """Tính ma trận Gram K[i][j] = k(X1[i], X2[j]) với RBF kernel.

        Tham số
        -------
        X1           : list[list[float]] -- tập điểm dữ liệu thứ nhất.
        X2           : list[list[float]] -- tập điểm dữ liệu thứ hai.
        length_scale : float -- tham số tỷ lệ độ dài (mặc định 1.0).

        Trả về
        ------
        list[list[float]] -- ma trận Gram K kích thước len(X1) x len(X2).
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


    @staticmethod
    def _add_diagonal(K: list[list[float]], value: float) -> list[list[float]]:
        """Cộng thêm giá trị vào đường chéo chính của ma trận K.

        Tham số
        -------
        K     : list[list[float]] -- ma trận vuông đầu vào.
        value : float -- giá trị cần cộng thêm vào đường chéo.

        Trả về
        ------
        list[list[float]] -- bản sao của K với K[i][i] += value.
        """
        n = len(K)
        A = [K[i][:] for i in range(n)]
        for i in range(n):
            A[i][i] += value
        return A

    @staticmethod
    def _std_sample(values: list[float]) -> float:
        """Tính độ lệch chuẩn mẫu hiệu chỉnh (chia cho n-1).

        Tham số
        -------
        values : list[float] -- danh sách các giá trị mẫu.

        Trả về
        ------
        float -- độ lệch chuẩn mẫu hiệu chỉnh.
        """
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
        """Tạo danh sách k fold phân chia các chỉ số mẫu cho kiểm định chéo.

        Các fold có kích thước cân bằng: n % k fold đầu có thêm 1 phần tử.

        Tham số
        -------
        n    : int -- tổng số mẫu dữ liệu.
        k    : int -- số lượng fold cần chia.
        seed : int -- hạt giống ngẫu nhiên.

        Trả về
        ------
        list[list[int]] -- danh sách k fold, mỗi fold chứa các chỉ số tập kiểm thử.
        """
        if k < 2:
            raise ValueError("k phải lớn hơn hoặc bằng 2")
        if n < k:
            raise ValueError(f"n={n} phải lớn hơn hoặc bằng k={k}")

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

    @staticmethod
    def fit(
        X_train: list[list[float]],
        y_train: list[float],
        lam: float,
        length_scale: float,
        jitter: float = 1e-10,
    ) -> dict:
        """Huấn luyện Kernel Ridge Regression với RBF kernel.

        Giải hệ: (K + (lam + jitter) * I) * alpha = y_train

        Tham số
        -------
        X_train      : list[list[float]], shape (n, p) -- dữ liệu đặc trưng huấn luyện.
        y_train      : list[float], shape (n,) -- dữ liệu mục tiêu huấn luyện.
        lam          : float -- hệ số điều hòa (>= 0).
        length_scale : float -- tham số tỷ lệ độ dài của RBF kernel (> 0).
        jitter       : float -- giá trị ổn định số học cộng vào đường chéo.

        Trả về
        ------
        dict -- từ điển chứa thông tin mô hình đã huấn luyện.
        """
        n = len(X_train)
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

    @staticmethod
    def predict(
        model: dict,
        X_test: list[list[float]],
    ) -> list[float]:
        """Dự đoán từ mô hình KRR đã huấn luyện.

        Tham số
        -------
        model  : dict -- từ điển mô hình trả về từ fit().
        X_test : list[list[float]], shape (m, p) -- dữ liệu cần dự đoán.

        Trả về
        ------
        list[float] -- danh sách m giá trị dự đoán.
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

    @staticmethod
    def cross_validate(
        X: list[list[float]],
        y: list[float],
        lambda_grid: list[float],
        length_scale_grid: list[float],
        k: int = 5,
        seed: int = RANDOM_STATE,
    ) -> dict:
        """Kiểm định chéo với tìm kiếm lưới 2 chiều cho KRR.

        Với mỗi cặp (lam, length_scale), chạy k-fold CV và ghi lại MSE trung bình.

        Tham số
        -------
        X                 : list[list[float]], shape (n, p) -- dữ liệu đặc trưng.
        y                 : list[float], shape (n,) -- dữ liệu mục tiêu.
        lambda_grid       : list[float] -- danh sách các giá trị lambda cần thử.
        length_scale_grid : list[float] -- danh sách các giá trị length_scale cần thử.
        k                 : int -- số fold (>= 2).
        seed              : int -- hạt giống ngẫu nhiên.

        Trả về
        ------
        dict -- từ điển chứa best_lam, best_length_scale, best_cv_score và cv_results.
        """
        if len(lambda_grid) == 0:
            raise ValueError("lambda_grid không được rỗng")
        if len(length_scale_grid) == 0:
            raise ValueError("length_scale_grid không được rỗng")

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

class BayesianLinearRegression:
    """
    Bayesian Linear Regression.
    """

    @staticmethod
    def _add_bias(X: list[list[float]]) -> list[list[float]]:
        """Thêm cột bias (intercept = 1.0) vào đầu mỗi hàng của X.

        Tham số
        -------
        X : list[list[float]] -- ma trận đặc trưng kích thước (n, p).

        Trả về
        ------
        list[list[float]] -- ma trận với cột bias, kích thước (n, p+1).
        """
        return [[1.0] + row for row in X]

    @staticmethod
    def _matrix_add(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
        """Cộng hai ma trận cùng kích thước.

        Tham số
        -------
        A : list[list[float]] -- ma trận thứ nhất.
        B : list[list[float]] -- ma trận thứ hai cùng kích thước với A.

        Trả về
        ------
        list[list[float]] -- ma trận tổng A + B.
        """
        n = len(A)
        m = len(A[0])
        return [[A[i][j] + B[i][j] for j in range(m)] for i in range(n)]

    @staticmethod
    def _scalar_matrix(k: float, A: list[list[float]]) -> list[list[float]]:
        """Nhân ma trận với một hằng số vô hướng.

        Tham số
        -------
        k : float -- hằng số vô hướng.
        A : list[list[float]] -- ma trận cần nhân.

        Trả về
        ------
        list[list[float]] -- ma trận k * A.
        """
        n = len(A)
        m = len(A[0])
        return [[k * A[i][j] for j in range(m)] for i in range(n)]

    @staticmethod
    def estimate_sigma2(
        X_train: list[list[float]],
        y_train: list[float],
    ) -> float:
        """Ước lượng sigma2 (phương sai nhiễu) từ phần dư OLS.

        Tham số
        -------
        X_train : list[list[float]], shape (n, p) -- dữ liệu đặc trưng huấn luyện.
        y_train : list[float], shape (n,) -- dữ liệu mục tiêu huấn luyện.

        Trả về
        ------
        float -- ước lượng sigma2 = RSS / (n - p - 1).
        """
        from part1.ols_implementation import ols_fit
        result = ols_fit(X_train, y_train)
        return result['sigma2_hat']

    @staticmethod
    def fit(
        X_train: list[list[float]],
        y_train: list[float],
        sigma2: float,
        alpha: float = 1.0,
        m0: list[float] = None,
        S0: list[list[float]] = None,
    ) -> dict:
        """Huấn luyện mô hình hồi quy tuyến tính Bayes.

        Tính tham số hậu nghiệm (m_n, S_n) từ tiên nghiệm và hàm hợp lý.

        Tham số
        -------
        X_train : list[list[float]], shape (n, p) -- dữ liệu đặc trưng, CHƯA có bias.
        y_train : list[float], shape (n,) -- dữ liệu mục tiêu.
        sigma2  : float -- phương sai nhiễu. Dùng estimate_sigma2() để ước lượng.
        alpha   : float -- độ chính xác tiên nghiệm.
        m0      : list[float], shape (p+1,) -- trung bình tiên nghiệm. Mặc định vector 0.
        S0      : list[list[float]], shape (p+1, p+1) -- hiệp phương sai tiên nghiệm.

        Trả về
        ------
        dict -- từ điển chứa model_type, m_n, S_n, sigma2, alpha, y_hat, beta_hat.
        """
        n = len(X_train)
        p = len(X_train[0])
        p1 = p + 1  # bao gồm intercept

        X_bias = BayesianLinearRegression._add_bias(X_train)
        X_bias_T = transpose(X_bias)

        # Gia tri mac dinh cua phan phoi tien nghiem (Prior defaults)
        if m0 is None:
            m0 = [0.0] * p1
        if S0 is None:
            S0 = BayesianLinearRegression._scalar_matrix(1.0 / alpha, identity_matrix(p1))

        # S0_inv = alpha * I (nếu dùng default)
        S0_inv = inverse(S0)

        # XTX = X_bias^T @ X_bias  (p+1 x p+1)
        XTX = matmul(X_bias_T, X_bias)

        # S_n_inv = S0_inv + (1/sigma2) * XTX
        XTX_scaled = BayesianLinearRegression._scalar_matrix(1.0 / sigma2, XTX)
        S_n_inv = BayesianLinearRegression._matrix_add(S0_inv, XTX_scaled)

        # S_n = S_n_inv^{-1}
        S_n = inverse(S_n_inv)

        # m_n = S_n @ (S0_inv @ m0 + (1/sigma2) * X_bias^T @ y)
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
            "beta_hat": m_n,  
        }

    @staticmethod
    def predict(
        model: dict,
        X_test: list[list[float]],
        sigma2: float = None,
        credible_interval: float = 0.95,
    ) -> tuple[list[float], list[float], list[float]]:
        """Dự đoán từ mô hình BLR đã huấn luyện.

        Tham số
        -------
        model             : dict -- từ điển mô hình trả về từ fit().
        X_test            : list[list[float]], shape (m, p) -- dữ liệu cần dự đoán, CHƯA có bias.
        sigma2            : float -- phương sai nhiễu. Nếu None, dùng sigma2 từ model.
        credible_interval : float -- mức khoảng tin cậy (mặc định 0.95).

        Trả về
        ------
        tuple -- bộ (y_mean, y_lower, y_upper):
            y_mean  : list[float] -- giá trị dự đoán trung bình.
            y_lower : list[float] -- cận dưới khoảng tin cậy.
            y_upper : list[float] -- cận trên khoảng tin cậy.
        """
        m_n = model["m_n"]
        S_n = model["S_n"]
        if sigma2 is None:
            sigma2 = model["sigma2"]

        # z-score cho credible interval (xấp xỉ Normal)
        z_table = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}
        z = z_table.get(credible_interval, 1.9600)

        X_bias = BayesianLinearRegression._add_bias(X_test)

        y_mean = []
        y_lower = []
        y_upper = []

        for x in X_bias:
            # y_hat = x^T @ m_n
            pred = dot(x, m_n)

            # Predictive variance: sigma2_pred = sigma2 + x^T S_n x
            Sx = matvec(S_n, x)
            var_pred = sigma2 + dot(x, Sx)
            std_pred = math.sqrt(max(var_pred, 0.0))

            y_mean.append(pred)
            y_lower.append(pred - z * std_pred)
            y_upper.append(pred + z * std_pred)

        return y_mean, y_lower, y_upper

    @staticmethod
    def cross_validate(
        X: list[list[float]],
        y: list[float],
        alpha_grid: list[float],
        k: int = 5,
        seed: int = RANDOM_STATE,
    ) -> dict:
        """Kiểm định chéo với tìm kiếm lưới 1 chiều cho BLR (tìm alpha tối ưu).

        Với mỗi alpha, chạy k-fold CV và ghi lại MSE trung bình.
        sigma2 được ước lượng tự động trong mỗi fold từ OLS.

        Tham số
        -------
        X          : list[list[float]], shape (n, p) -- dữ liệu đặc trưng.
        y          : list[float], shape (n,) -- dữ liệu mục tiêu.
        alpha_grid : list[float] -- danh sách các giá trị alpha cần thử.
        k          : int -- số fold (>= 2).
        seed       : int -- hạt giống ngẫu nhiên.

        Trả về
        ------
        dict -- từ điển chứa best_alpha, best_cv_score và cv_results.
        """
        if len(alpha_grid) == 0:
            raise ValueError("alpha_grid không được rỗng")

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

def compare_models_summary(results: dict) -> list[dict]:
    """Tạo bảng so sánh MAE/RMSE/R2 từ kết quả các mô hình.

    Tham số
    -------
    results : dict[str, dict] -- key là tên mô hình,
              value chứa key 'metrics' với MAE, RMSE, R2.

    Trả về
    ------
    list[dict] -- mỗi phần tử là {'Model', 'MAE', 'RMSE', 'R2'},
                 sắp xếp theo RMSE tăng dần.
    """
    rows = []
    for name, data in results.items():
        m = data.get('metrics', {})
        rows.append({
            'Model': name,
            'MAE': m.get('MAE', float('nan')),
            'RMSE': m.get('RMSE', float('nan')),
            'R2': m.get('R2', float('nan')),
        })
    rows.sort(key=lambda r: r['RMSE'])
    return rows



def _run_tests() -> tuple[int, int]:
    """Chạy unit tests cho advanced_methods.py từ part2/test_case.py."""
    from part2.test_case import (
        _log,
        run_test_cases,
        test_krr_rbf_kernel,
        test_krr_fit_predict_shape_mse,
        test_sample_rows_reproducible,
        test_krr_cv_returns_grid_values,
        test_blr_fit_predict_shape,
        test_blr_credible_interval_order,
        test_blr_cv_returns_grid_values,
    )

    _log.print_suite_header("KRR & BLR METHODS - UNIT TESTS")
    passed_count, total_count = run_test_cases(
        [
            test_krr_rbf_kernel,
            test_krr_fit_predict_shape_mse,
            test_sample_rows_reproducible,
            test_krr_cv_returns_grid_values,
            test_blr_fit_predict_shape,
            test_blr_credible_interval_order,
            test_blr_cv_returns_grid_values,
        ]
    )
    _log.print_summary(passed_count, total_count)
    return passed_count, total_count


if __name__ == "__main__":
    _run_tests()

