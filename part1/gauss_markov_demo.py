import os
import sys
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import transpose, matmul, inverse, matvec, vector_add
from part1.ols_implementation import ols_fit
from part1.ridge_lasso import add_bias_column


def run_gauss_markov_simulation(n_sim=1000, n_obs=100, true_beta=None, true_sigma=1.0):
    """
    F9: Mô phỏng Monte Carlo chứng minh OLS là BLUE.

    Tham số
    -------
    n_sim      : int   — số lần mô phỏng
    n_obs      : int   — số quan sát mỗi lần
    true_beta  : list  — hệ số thực [intercept, β1, β2, ...]
    true_sigma : float — độ lệch chuẩn nhiễu

    Trả về
    ------
    (beta_ols_list, beta_alt_list, true_beta)
    """
    if true_beta is None:
        true_beta = [2.0, -1.5, 0.8]

    random.seed(42)

    p_features = len(true_beta) - 1  # số features (không tính intercept)

    # 1. Sinh X cố định
    X_fixed = [[random.gauss(0, 1) for _ in range(p_features)] for _ in range(n_obs)]
    X_bias = add_bias_column(X_fixed)

    # Tính toán trước ma trận cho Estimator thay thế (Ridge alpha=0.5)
    XT = transpose(X_bias)
    XTX = matmul(XT, X_bias)
    for i in range(len(XTX)):
        XTX[i][i] += 0.5  # Ridge alpha=0.5
    XTX_inv_alt = inverse(XTX)
    H_alt = matmul(XTX_inv_alt, XT)  # Ma trận trọng số của Alt estimator

    beta_ols_list = []
    beta_alt_list = []

    # Tính phần cố định của y: X * beta
    y_pure = matvec(X_bias, true_beta)

    for sim in range(n_sim):
        # 2. Sinh nhiễu
        epsilon = [random.gauss(0, true_sigma) for _ in range(n_obs)]

        # 3. y = X_beta + epsilon
        y_sim = vector_add(y_pure, epsilon)

        # 4. Ước lượng OLS
        res_ols = ols_fit(X_fixed, y_sim)
        beta_ols_list.append(res_ols['beta_hat'])

        # 5. Ước lượng thay thế (Ridge — biased estimator)
        beta_alt = matvec(H_alt, y_sim)
        beta_alt_list.append(beta_alt)

    return beta_ols_list, beta_alt_list, true_beta


def run_gauss_markov_tests() -> tuple[int, int]:
    """Run gauss_markov unit tests from part1/test_case.py."""
    from part1.test_case import (
        _log,
        run_test_cases,
        test_gauss_markov_beta_vector_dimensions,
        test_gauss_markov_custom_true_beta,
        test_gauss_markov_deterministic_seed,
        test_gauss_markov_output_lengths,
        test_gauss_markov_zero_noise_ols_exact,
    )

    _log.print_suite_header("GAUSS-MARKOV SIMULATION - UNIT TESTS")
    passed_count, total_count = run_test_cases(
        [
            test_gauss_markov_output_lengths,
            test_gauss_markov_beta_vector_dimensions,
            test_gauss_markov_custom_true_beta,
            test_gauss_markov_deterministic_seed,
            test_gauss_markov_zero_noise_ols_exact,
        ]
    )
    _log.print_summary(passed_count, total_count)
    return passed_count, total_count


if __name__ == "__main__":
    run_gauss_markov_tests()
