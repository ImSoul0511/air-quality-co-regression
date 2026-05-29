import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
from test_logger import TestLogger

from part1.ridge_lasso import lasso_fit, ridge_fit
from part1.cross_validation import kfold_cv
from utils import make_linear_data, make_multifeature_data
from config import RANDOM_STATE


_log = TestLogger()


def _print_case(
    name: str,
    description: str,
    expected_output,
    expected_error,
) -> None:
    _log.print_group(name)
    _log.print_info(f"Test case       : {name}")
    _log.print_info(f"Noi dung        : {description}")
    _log.print_info(f"Expected output : {expected_output}")
    _log.print_info(f"Expected error  : {expected_error}")


def _finish_case(name: str, passed: bool, details: str = "") -> bool:
    _log.print_result(name, passed, details=details)
    assert passed, details or f"{name} failed"
    return True


def _close(actual: float, expected: float, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    return abs(actual - expected) <= atol + rtol * abs(expected)


def _list_close(
    actual: list[float],
    expected: list[float],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    if len(actual) != len(expected):
        return False
    return all(_close(actual[i], expected[i], rtol=rtol, atol=atol) for i in range(len(actual)))


def _nested_close(
    actual: list[list[float]],
    expected: list[list[float]],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    if len(actual) != len(expected):
        return False
    return all(_list_close(actual[i], expected[i], rtol=rtol, atol=atol) for i in range(len(actual)))


def _coef_inference_fixture():
    from part1.ols_implementation import ols_fit

    X = [
        [-3.0, -1.0],
        [-2.0, 0.0],
        [-1.0, 1.0],
        [0.0, -1.0],
        [1.0, 0.0],
        [2.0, 1.0],
        [3.0, -1.0],
        [4.0, 0.0],
        [5.0, 1.0],
        [6.0, -1.0],
        [7.0, 0.0],
        [8.0, 1.0],
    ]
    noise = [0.20, -0.10, 0.05, -0.15, 0.10, -0.05, 0.12, -0.08, 0.03, -0.02, 0.04, -0.06]
    y = [1.0 + 2.0 * row[0] - 0.5 * row[1] + noise[i] for i, row in enumerate(X)]
    result = ols_fit(X, y)
    return X, y, result

def test_ols_perfect_fit():
    name = "test_ols_perfect_fit"
    _print_case(
        name,
        "Noiseless linear data should recover the exact coefficients.",
        "beta_hat approximately [0.0, 1.0, 2.0, -3.0] and RSS approximately 0",
        None,
    )

    from part1.ols_implementation import ols_fit

    feat_beta = [1.0, 2.0, -3.0]
    X, y, _ = make_linear_data(n=50, beta=feat_beta, sigma=0.0, seed=RANDOM_STATE)
    result = ols_fit(X, y)
    true_beta_full = [0.0] + feat_beta
    rss = sum(r * r for r in result["residuals"])

    passed = _list_close(result["beta_hat"], true_beta_full, atol=1e-7) and _close(rss, 0.0, atol=1e-8)
    _log.print_value("beta_hat", [round(value, 6) for value in result["beta_hat"]])
    _log.print_value("RSS", round(rss, 12))
    return _finish_case(name, passed)


def test_ols_output_shapes():
    name = "test_ols_output_shapes"
    _print_case(
        name,
        "OLS result should expose beta_hat, y_hat, residuals, and sigma2_hat with expected shapes.",
        "len(beta_hat)=p+1, len(y_hat)=n, len(residuals)=n, sigma2_hat is float",
        None,
    )

    from part1.ols_implementation import ols_fit

    n, p = 30, 3
    X, y, _ = make_multifeature_data(n=n, p=p, sigma=1.0, seed=RANDOM_STATE)
    result = ols_fit(X, y)

    passed = (
        len(result["beta_hat"]) == p + 1
        and len(result["y_hat"]) == n
        and len(result["residuals"]) == n
        and isinstance(result["sigma2_hat"], float)
    )
    _log.print_value("len(beta_hat)", len(result["beta_hat"]), expected=p + 1)
    _log.print_value("len(y_hat)", len(result["y_hat"]), expected=n)
    _log.print_value("len(residuals)", len(result["residuals"]), expected=n)
    return _finish_case(name, passed)


def test_ols_sigma2_positive():
    name = "test_ols_sigma2_positive"
    _print_case(
        name,
        "Noisy data should produce a positive residual variance estimate.",
        "sigma2_hat > 0",
        None,
    )

    from part1.ols_implementation import ols_fit

    X, y, _ = make_multifeature_data(n=40, p=2, sigma=2.0, seed=RANDOM_STATE)
    result = ols_fit(X, y)

    passed = result["sigma2_hat"] > 0
    _log.print_value("sigma2_hat", round(result["sigma2_hat"], 8))
    return _finish_case(name, passed)


def test_ols_residuals_sum_zero():
    name = "test_ols_residuals_sum_zero"
    _print_case(
        name,
        "OLS with an intercept should have residuals that sum to zero.",
        "abs(sum(residuals)) < 1e-6",
        None,
    )

    from part1.ols_implementation import ols_fit

    X, y, _ = make_multifeature_data(n=50, p=3, sigma=1.0, seed=RANDOM_STATE)
    result = ols_fit(X, y)
    residual_sum = sum(result["residuals"])

    passed = abs(residual_sum) < 1e-6
    _log.print_value("sum(residuals)", round(residual_sum, 12))
    return _finish_case(name, passed)


def test_ols_verify_with_sklearn():
    name = "test_ols_verify_with_sklearn"
    _print_case(
        name,
        "OLS coefficients should match sklearn LinearRegression.",
        "beta_hat approximately equals [intercept] + coef_",
        None,
    )

    import numpy as np
    from sklearn.linear_model import LinearRegression
    from part1.ols_implementation import ols_fit

    X, y, _ = make_multifeature_data(n=60, p=4, sigma=1.5, seed=RANDOM_STATE)
    result = ols_fit(X, y)
    lr = LinearRegression().fit(np.array(X), np.array(y))
    sklearn_beta = [float(lr.intercept_)] + [float(value) for value in lr.coef_]

    passed = _list_close(result["beta_hat"], sklearn_beta, rtol=1e-4, atol=1e-6)
    _log.print_value("beta_hat", [round(value, 6) for value in result["beta_hat"]])
    _log.print_value("sklearn", [round(value, 6) for value in sklearn_beta])
    return _finish_case(name, passed)

def test_hat_matrix_idempotent():
    name = "test_hat_matrix_idempotent"
    _print_case(
        name,
        "Hat matrix should satisfy H @ H approximately equals H.",
        "is_idempotent is True",
        None,
    )

    from part1.ols_implementation import hat_matrix

    X, *_ = make_linear_data(n=20, beta=[1.0, 2.0], sigma=0.0, seed=RANDOM_STATE)
    result = hat_matrix(X)

    _log.print_value("is_idempotent", result["is_idempotent"], expected=True)
    return _finish_case(name, result["is_idempotent"])


def test_hat_matrix_symmetric():
    name = "test_hat_matrix_symmetric"
    _print_case(
        name,
        "Hat matrix should be symmetric.",
        "is_symmetric is True",
        None,
    )

    from part1.ols_implementation import hat_matrix

    X, *_ = make_multifeature_data(n=15, p=3, sigma=0.0, seed=RANDOM_STATE)
    result = hat_matrix(X)

    _log.print_value("is_symmetric", result["is_symmetric"], expected=True)
    return _finish_case(name, result["is_symmetric"])


def test_hat_matrix_rank():
    name = "test_hat_matrix_rank"
    _print_case(
        name,
        "Projection matrix rank should equal p + 1 when X has full column rank.",
        "rank(H) = p + 1",
        None,
    )

    from part1.ols_implementation import hat_matrix

    p = 3
    X, *_ = make_multifeature_data(n=20, p=p, sigma=0.0, seed=RANDOM_STATE)
    result = hat_matrix(X)

    passed = result["rank"] == p + 1
    _log.print_value("rank", result["rank"], expected=p + 1)
    return _finish_case(name, passed)


def test_hat_matrix_eigenvalues_binary():
    name = "test_hat_matrix_eigenvalues_binary"
    _print_case(
        name,
        "Hat matrix eigenvalues should be near 0 or 1.",
        "Every eigenvalue is approximately binary",
        None,
    )

    from part1.ols_implementation import hat_matrix

    X, *_ = make_multifeature_data(n=15, p=2, sigma=0.0, seed=RANDOM_STATE)
    result = hat_matrix(X)
    eigenvalues = result["eigenvalues"]

    passed = all(abs(ev) < 1e-5 or abs(ev - 1.0) < 1e-5 for ev in eigenvalues)
    _log.print_value("eigenvalues", [round(value, 8) for value in eigenvalues])
    return _finish_case(name, passed)


def test_hat_matrix_output_shape():
    name = "test_hat_matrix_output_shape"
    _print_case(
        name,
        "For n observations, H should have shape n x n.",
        "len(H)=n and all rows have length n",
        None,
    )

    from part1.ols_implementation import hat_matrix

    n = 12
    X, *_ = make_multifeature_data(n=n, p=2, sigma=0.0, seed=RANDOM_STATE)
    result = hat_matrix(X)
    H = result["H"]

    passed = len(H) == n and all(len(row) == n for row in H)
    _log.print_value("shape", (len(H), len(H[0]) if H else 0), expected=(n, n))
    return _finish_case(name, passed)

def test_metrics_perfect_prediction():
    name = "test_metrics_perfect_prediction"
    _print_case(
        name,
        "Perfect predictions should have zero error and R2=1.",
        "RSS=0, R2=1, MAE=0, RMSE=0",
        None,
    )

    from part1.ols_implementation import model_metrics

    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    metrics = model_metrics(y, y[:], p=1)

    passed = (
        _close(metrics["RSS"], 0.0)
        and _close(metrics["R2"], 1.0)
        and _close(metrics["MAE"], 0.0)
        and _close(metrics["RMSE"], 0.0)
    )
    _log.print_value("RSS", metrics["RSS"], expected=0.0)
    _log.print_value("R2", metrics["R2"], expected=1.0)
    return _finish_case(name, passed)


def test_metrics_r2_range():
    name = "test_metrics_r2_range"
    _print_case(
        name,
        "OLS fitted values on generated data should produce a reasonable R2 and adjusted R2.",
        "-1 <= R2 <= 1 and -2 <= R2_adj <= 1",
        None,
    )

    from part1.ols_implementation import model_metrics, ols_fit

    X, y, _ = make_multifeature_data(n=50, p=3, sigma=1.0, seed=RANDOM_STATE)
    result = ols_fit(X, y)
    metrics = model_metrics(y, result["y_hat"], p=3)

    passed = -1.0 <= metrics["R2"] <= 1.0 and -2.0 <= metrics["R2_adj"] <= 1.0
    _log.print_value("R2", round(metrics["R2"], 8))
    _log.print_value("R2_adj", round(metrics["R2_adj"], 8))
    return _finish_case(name, passed)


def test_metrics_mss_identity():
    name = "test_metrics_mss_identity"
    _print_case(
        name,
        "Model sum of squares should satisfy MSS = TSS - RSS.",
        "MSS approximately equals TSS - RSS",
        None,
    )

    from part1.ols_implementation import model_metrics

    y = [1.0, 3.0, 2.0, 5.0, 4.0]
    y_hat = [1.5, 2.5, 2.5, 4.5, 3.5]
    metrics = model_metrics(y, y_hat, p=1)

    passed = _close(metrics["MSS"], metrics["TSS"] - metrics["RSS"])
    _log.print_value("MSS", round(metrics["MSS"], 8))
    _log.print_value("TSS - RSS", round(metrics["TSS"] - metrics["RSS"], 8))
    return _finish_case(name, passed)


def test_metrics_f_pvalue():
    name = "test_metrics_f_pvalue"
    _print_case(
        name,
        "F-test p-value should be a probability.",
        "0 <= F_pvalue <= 1",
        None,
    )

    from part1.ols_implementation import model_metrics, ols_fit

    X, y, _ = make_multifeature_data(n=50, p=3, sigma=1.0, seed=RANDOM_STATE)
    result = ols_fit(X, y)
    metrics = model_metrics(y, result["y_hat"], p=3)

    passed = 0.0 <= metrics["F_pvalue"] <= 1.0
    _log.print_value("F_pvalue", metrics["F_pvalue"])
    return _finish_case(name, passed)


def test_metrics_verify_r2_with_sklearn():
    name = "test_metrics_verify_r2_with_sklearn"
    _print_case(
        name,
        "Custom R2 should match sklearn.metrics.r2_score.",
        "R2 approximately equals sklearn r2_score",
        None,
    )

    import numpy as np
    from sklearn.metrics import r2_score
    from part1.ols_implementation import model_metrics, ols_fit

    X, y, _ = make_multifeature_data(n=50, p=3, sigma=1.0, seed=RANDOM_STATE)
    result = ols_fit(X, y)
    metrics = model_metrics(y, result["y_hat"], p=3)
    sklearn_r2 = float(r2_score(np.array(y), np.array(result["y_hat"])))

    passed = _close(metrics["R2"], sklearn_r2, rtol=1e-5)
    _log.print_value("R2", metrics["R2"])
    _log.print_value("sklearn R2", sklearn_r2)
    return _finish_case(name, passed)

def test_coef_inference_output_structure():
    name = "test_coef_inference_output_structure"
    _print_case(
        name,
        "coef_inference should return a DataFrame with expected columns and coefficient labels.",
        "shape=(p+1, 6), index=[intercept, x1, x2], expected columns present",
        None,
    )

    from part1.ols_implementation import coef_inference

    X, y, result = _coef_inference_fixture()
    table = coef_inference(X, y, result["beta_hat"], result["sigma2_hat"])
    expected_columns = ["coef", "std_err", "t_stat", "p_value", "ci_lower", "ci_upper"]
    expected_index = ["intercept", "x1", "x2"]

    passed = list(table.columns) == expected_columns and list(table.index) == expected_index and table.shape == (3, 6)
    _log.print_value("columns", list(table.columns))
    _log.print_value("index", list(table.index))
    return _finish_case(name, passed)


def test_coef_inference_coefficients_match_input():
    name = "test_coef_inference_coefficients_match_input"
    _print_case(
        name,
        "The coef column should copy beta_hat exactly.",
        "table['coef'] approximately equals beta_hat",
        None,
    )

    from part1.ols_implementation import coef_inference

    X, y, result = _coef_inference_fixture()
    table = coef_inference(X, y, result["beta_hat"], result["sigma2_hat"])
    coefficients = [float(value) for value in table["coef"]]

    passed = _list_close(coefficients, result["beta_hat"], atol=1e-12)
    _log.print_value("coef column", [round(value, 8) for value in coefficients])
    return _finish_case(name, passed)


def test_coef_inference_standard_errors_positive():
    name = "test_coef_inference_standard_errors_positive"
    _print_case(
        name,
        "Standard errors should be positive finite values for full-rank noisy data.",
        "all std_err > 0 and finite",
        None,
    )

    from part1.ols_implementation import coef_inference

    X, y, result = _coef_inference_fixture()
    table = coef_inference(X, y, result["beta_hat"], result["sigma2_hat"])
    std_errs = [float(value) for value in table["std_err"]]

    passed = all(value > 0 and math.isfinite(value) for value in std_errs)
    _log.print_value("std_err", [round(value, 8) for value in std_errs])
    return _finish_case(name, passed)


def test_coef_inference_t_stat_formula():
    name = "test_coef_inference_t_stat_formula"
    _print_case(
        name,
        "Each t_stat should equal coef / std_err.",
        "t_stat approximately equals coef / std_err",
        None,
    )

    from part1.ols_implementation import coef_inference

    X, y, result = _coef_inference_fixture()
    table = coef_inference(X, y, result["beta_hat"], result["sigma2_hat"])
    expected = [float(row.coef) / float(row.std_err) for row in table.itertuples()]
    actual = [float(value) for value in table["t_stat"]]

    passed = _list_close(actual, expected, rtol=1e-8, atol=1e-10)
    _log.print_value("t_stat", [round(value, 8) for value in actual])
    return _finish_case(name, passed)


def test_coef_inference_p_values_and_ci_valid():
    name = "test_coef_inference_p_values_and_ci_valid"
    _print_case(
        name,
        "p-values should lie in [0, 1] and each confidence interval should contain its coefficient.",
        "0 <= p_value <= 1 and ci_lower <= coef <= ci_upper",
        None,
    )

    from part1.ols_implementation import coef_inference

    X, y, result = _coef_inference_fixture()
    table = coef_inference(X, y, result["beta_hat"], result["sigma2_hat"])

    p_values_ok = all(0.0 <= float(value) <= 1.0 for value in table["p_value"])
    ci_ok = all(
        float(row.ci_lower) <= float(row.coef) <= float(row.ci_upper)
        for row in table.itertuples()
    )
    passed = p_values_ok and ci_ok
    _log.print_value("p_value", [round(float(value), 8) for value in table["p_value"]])
    return _finish_case(name, passed)


def test_coef_inference_sigma2_scales_standard_errors():
    name = "test_coef_inference_sigma2_scales_standard_errors"
    _print_case(
        name,
        "Multiplying sigma2 by 4 should double all standard errors.",
        "std_err(sigma2*4) / std_err(sigma2) approximately 2",
        None,
    )

    from part1.ols_implementation import coef_inference

    X, y, result = _coef_inference_fixture()
    table = coef_inference(X, y, result["beta_hat"], result["sigma2_hat"])
    wider = coef_inference(X, y, result["beta_hat"], result["sigma2_hat"] * 4.0)
    ratios = [float(wider["std_err"].iloc[i]) / float(table["std_err"].iloc[i]) for i in range(len(table))]

    passed = all(_close(ratio, 2.0, rtol=1e-8, atol=1e-10) for ratio in ratios)
    _log.print_value("ratios", [round(value, 8) for value in ratios])
    return _finish_case(name, passed)

def test_vif_output_keys():
    name = "test_vif_output_keys"
    _print_case(
        name,
        "vif should return one entry per input feature.",
        "keys are ['x1', 'x2', 'x3']",
        None,
    )

    from part1.ols_implementation import vif

    X = [
        [0.0, 0.0, 1.0],
        [1.0, 2.0, 0.0],
        [2.0, 1.0, 1.0],
        [3.0, 3.0, 0.0],
        [4.0, 2.0, 1.0],
        [5.0, 5.0, 0.0],
        [6.0, 4.0, 1.0],
        [7.0, 6.0, 0.0],
    ]
    result = vif(X)

    passed = list(result.keys()) == ["x1", "x2", "x3"]
    _log.print_value("keys", list(result.keys()))
    return _finish_case(name, passed)


def test_vif_orthogonal_features_are_one():
    name = "test_vif_orthogonal_features_are_one"
    _print_case(
        name,
        "Centered orthogonal features should have VIF close to 1.",
        "VIF(x1) and VIF(x2) approximately 1",
        None,
    )

    from part1.ols_implementation import vif

    X = [[-2.0, 2.0], [-1.0, -1.0], [0.0, -2.0], [1.0, -1.0], [2.0, 2.0]]
    result = vif(X)

    passed = _close(result["x1"], 1.0, atol=1e-8) and _close(result["x2"], 1.0, atol=1e-8)
    _log.print_value("vif", {key: round(value, 8) for key, value in result.items()})
    return _finish_case(name, passed)


def test_vif_perfect_collinearity_is_infinite():
    name = "test_vif_perfect_collinearity_is_infinite"
    _print_case(
        name,
        "Perfectly collinear features should produce infinite VIF.",
        "VIF values are inf",
        None,
    )

    from part1.ols_implementation import vif

    X = [[float(i), float(2 * i)] for i in range(1, 8)]
    result = vif(X)

    passed = math.isinf(result["x1"]) and math.isinf(result["x2"])
    _log.print_value("vif", result)
    return _finish_case(name, passed)


def test_vif_near_collinearity_is_large():
    name = "test_vif_near_collinearity_is_large"
    _print_case(
        name,
        "Nearly collinear features should produce large VIF values.",
        "max VIF > 100",
        None,
    )

    from part1.ols_implementation import vif

    X = [[float(i), 2.0 * i + (0.01 if i % 2 == 0 else -0.01)] for i in range(1, 30)]
    result = vif(X)
    max_vif = max(result.values())

    passed = max_vif > 100.0
    _log.print_value("max_vif", round(max_vif, 4))
    return _finish_case(name, passed)


def test_vif_single_feature_is_one():
    name = "test_vif_single_feature_is_one"
    _print_case(
        name,
        "A single feature has no auxiliary predictors, so its VIF should be 1.",
        "{'x1': 1.0}",
        None,
    )

    from part1.ols_implementation import vif

    X = [[float(i)] for i in range(1, 8)]
    result = vif(X)

    passed = list(result.keys()) == ["x1"] and _close(result["x1"], 1.0)
    _log.print_value("vif", result)
    return _finish_case(name, passed)


def test_vif_values_are_at_least_one():
    name = "test_vif_values_are_at_least_one"
    _print_case(
        name,
        "VIF should be at least 1 for finite non-collinear examples.",
        "all finite VIF values >= 1",
        None,
    )

    from part1.ols_implementation import vif

    X = [
        [-3.0, 1.0, 0.0],
        [-2.0, 0.0, 1.0],
        [-1.0, 2.0, 0.0],
        [0.0, -1.0, 1.0],
        [1.0, 1.0, 0.0],
        [2.0, 3.0, 1.0],
        [3.0, 2.0, 0.0],
        [4.0, 5.0, 1.0],
    ]
    result = vif(X)

    passed = all(value >= 1.0 and math.isfinite(value) for value in result.values())
    _log.print_value("vif", {key: round(value, 8) for key, value in result.items()})
    return _finish_case(name, passed)

def test_gauss_markov_output_lengths():
    name = "test_gauss_markov_output_lengths"
    _print_case(
        name,
        "Simulation should return one OLS and one alternative beta vector per simulation.",
        "len(beta_ols_list)=len(beta_alt_list)=n_sim",
        None,
    )

    from part1.gauss_markov_demo import run_gauss_markov_simulation

    beta_ols, beta_alt, true_beta = run_gauss_markov_simulation(n_sim=7, n_obs=25)

    passed = len(beta_ols) == 7 and len(beta_alt) == 7 and true_beta == [2.0, -1.5, 0.8]
    _log.print_value("len(beta_ols)", len(beta_ols), expected=7)
    _log.print_value("len(beta_alt)", len(beta_alt), expected=7)
    return _finish_case(name, passed)


def test_gauss_markov_beta_vector_dimensions():
    name = "test_gauss_markov_beta_vector_dimensions"
    _print_case(
        name,
        "Every estimated beta vector should have the same length as true_beta.",
        "all len(beta)=len(true_beta)",
        None,
    )

    from part1.gauss_markov_demo import run_gauss_markov_simulation

    beta_ols, beta_alt, true_beta = run_gauss_markov_simulation(n_sim=5, n_obs=30)
    expected_len = len(true_beta)

    passed = all(len(row) == expected_len for row in beta_ols) and all(len(row) == expected_len for row in beta_alt)
    _log.print_value("expected length", expected_len)
    return _finish_case(name, passed)


def test_gauss_markov_custom_true_beta():
    name = "test_gauss_markov_custom_true_beta"
    _print_case(
        name,
        "Simulation should respect a custom true_beta vector.",
        "returned true_beta equals input and beta vectors have matching length",
        None,
    )

    from part1.gauss_markov_demo import run_gauss_markov_simulation

    expected_beta = [1.0, 0.5, -0.25, 2.0]
    beta_ols, beta_alt, true_beta = run_gauss_markov_simulation(
        n_sim=4,
        n_obs=35,
        true_beta=expected_beta,
        true_sigma=0.5,
    )

    passed = (
        true_beta == expected_beta
        and all(len(row) == len(expected_beta) for row in beta_ols)
        and all(len(row) == len(expected_beta) for row in beta_alt)
    )
    _log.print_value("true_beta", true_beta)
    return _finish_case(name, passed)


def test_gauss_markov_deterministic_seed():
    name = "test_gauss_markov_deterministic_seed"
    _print_case(
        name,
        "The simulation sets a fixed random seed, so repeated calls should match.",
        "two calls with same arguments return identical beta lists",
        None,
    )

    from part1.gauss_markov_demo import run_gauss_markov_simulation

    beta_ols_1, beta_alt_1, true_beta_1 = run_gauss_markov_simulation(n_sim=3, n_obs=20, true_sigma=0.75)
    beta_ols_2, beta_alt_2, true_beta_2 = run_gauss_markov_simulation(n_sim=3, n_obs=20, true_sigma=0.75)

    passed = (
        true_beta_1 == true_beta_2
        and _nested_close(beta_ols_1, beta_ols_2, atol=1e-12)
        and _nested_close(beta_alt_1, beta_alt_2, atol=1e-12)
    )
    _log.print_value("first OLS beta", [round(value, 8) for value in beta_ols_1[0]])
    return _finish_case(name, passed)


def test_gauss_markov_zero_noise_ols_exact():
    name = "test_gauss_markov_zero_noise_ols_exact"
    _print_case(
        name,
        "With zero noise, OLS should recover true_beta exactly while ridge-style alternative is biased.",
        "OLS betas approximately true_beta and at least one alt beta differs",
        None,
    )

    from part1.gauss_markov_demo import run_gauss_markov_simulation

    beta_ols, beta_alt, true_beta = run_gauss_markov_simulation(n_sim=4, n_obs=25, true_sigma=0.0)
    ols_ok = all(_list_close(row, true_beta, atol=1e-8) for row in beta_ols)
    alt_differs = any(not _list_close(row, true_beta, atol=1e-4) for row in beta_alt)

    passed = ols_ok and alt_differs
    _log.print_value("true_beta", [round(value, 8) for value in true_beta])
    _log.print_value("first alt beta", [round(value, 8) for value in beta_alt[0]])
    return _finish_case(name, passed)


def test_ridge_lambda_zero_matches_ols():
    name = "test_ridge_lambda_zero_matches_ols"
    _print_case(
        name,
        "lam=0 should match OLS on noiseless data y = 2*x + 3.",
        "beta_hat approximately [3.0, 2.0]",
        None,
    )

    X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
    y = [5.0, 7.0, 9.0, 11.0, 13.0]
    beta = ridge_fit(X, y, lam=0.0)["beta_hat"]

    passed = abs(beta[0] - 3.0) < 0.1 and abs(beta[1] - 2.0) < 0.1
    _log.print_value("beta_hat", [round(b, 4) for b in beta])
    return _finish_case(name, passed)


def test_ridge_large_lambda_shrinks_coefficients():
    name = "test_ridge_large_lambda_shrinks_coefficients"
    _print_case(
        name,
        "A very large lambda should shrink non-intercept coefficients.",
        "|beta_hat[j]| < 0.1 for j >= 1",
        None,
    )

    X = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0],
    ]
    y = [10.0, 20.0, 30.0, 40.0, 50.0]
    beta = ridge_fit(X, y, lam=1e6)["beta_hat"]

    passed = all(abs(beta[j]) < 0.1 for j in range(1, len(beta)))
    _log.print_value("beta_hat", [round(b, 6) for b in beta])
    return _finish_case(name, passed)


def test_ridge_output_shape():
    name = "test_ridge_output_shape"
    _print_case(
        name,
        "X has shape 5x3, so beta_hat has length 4 and y_hat has length 5.",
        "len(beta_hat)=4, len(y_hat)=5",
        None,
    )

    X = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [2.0, 3.0, 1.0],
        [5.0, 1.0, 4.0],
    ]
    y = [10.0, 20.0, 30.0, 15.0, 25.0]
    result = ridge_fit(X, y, lam=1.0)

    passed = len(result["beta_hat"]) == 4 and len(result["y_hat"]) == 5
    _log.print_value("len(beta_hat)", len(result["beta_hat"]), expected=4)
    _log.print_value("len(y_hat)", len(result["y_hat"]), expected=5)
    return _finish_case(name, passed)


def test_ridge_negative_lambda_raises():
    name = "test_ridge_negative_lambda_raises"
    _print_case(
        name,
        "Passing lam=-1 should raise ValueError.",
        None,
        "ValueError",
    )

    try:
        ridge_fit([[1.0], [2.0], [3.0]], [1.0, 2.0, 3.0], lam=-1.0)
    except ValueError:
        return _finish_case(name, True)

    return _finish_case(name, False, "ValueError was not raised")


def test_ridge_empty_X_raises():
    name = "test_ridge_empty_X_raises"
    _print_case(
        name,
        "Passing X=[] should raise ValueError.",
        None,
        "ValueError",
    )

    try:
        ridge_fit([], [1.0], lam=1.0)
    except ValueError:
        return _finish_case(name, True)

    return _finish_case(name, False, "ValueError was not raised")


def test_ridge_prediction_accuracy():
    name = "test_ridge_prediction_accuracy"
    _print_case(
        name,
        "Small lambda on noiseless linear data should predict y accurately.",
        "max |y_hat - y| < 0.5",
        None,
    )

    X = [
        [1.0, 1.0],
        [2.0, 3.0],
        [3.0, 2.0],
        [4.0, 5.0],
        [5.0, 4.0],
        [6.0, 1.0],
    ]
    y = [8.0, 13.0, 12.0, 19.0, 18.0, 13.0]
    y_hat = ridge_fit(X, y, lam=0.001)["y_hat"]

    max_err = max(abs(y_hat[i] - y[i]) for i in range(len(y)))
    passed = max_err < 0.5
    _log.print_value("max |y_hat - y|", round(max_err, 6))
    return _finish_case(name, passed)


def test_lasso_output_shape_and_iterations():
    name = "test_lasso_output_shape_and_iterations"
    _print_case(
        name,
        "Lasso output should include beta_hat, y_hat, mean_X, std_X, and n_iter.",
        "len(beta_hat)=3, len(y_hat)=6, 1 <= n_iter <= 1000",
        None,
    )

    X = [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0], [5.0, 0.0]]
    y = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    result = lasso_fit(X, y, lam=0.1)

    passed = (
        len(result["beta_hat"]) == 3
        and len(result["y_hat"]) == 6
        and 1 <= result["n_iter"] <= 1000
    )
    _log.print_value("beta_hat", [round(b, 4) for b in result["beta_hat"]])
    _log.print_value("n_iter", result["n_iter"])
    return _finish_case(name, passed)


def test_lasso_large_lambda_sets_slopes_to_zero():
    name = "test_lasso_large_lambda_sets_slopes_to_zero"
    _print_case(
        name,
        "A very large lambda should produce an exactly sparse slope vector.",
        "all non-intercept beta_hat values equal 0",
        None,
    )

    X = [[1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0], [5.0, 0.0], [6.0, 1.0]]
    y = [4.0, 7.0, 8.0, 11.0, 12.0, 15.0]
    beta = lasso_fit(X, y, lam=1e6)["beta_hat"]

    passed = all(abs(beta[j]) <= 1e-12 for j in range(1, len(beta)))
    _log.print_value("beta_hat", [round(b, 12) for b in beta])
    return _finish_case(name, passed)


def test_lasso_low_lambda_recovers_linear_signal():
    name = "test_lasso_low_lambda_recovers_linear_signal"
    _print_case(
        name,
        "With lam=0 on noiseless full-rank data, Lasso CD should fit y closely.",
        "max |y_hat - y| < 1e-3",
        None,
    )

    X = [
        [-1.0, -1.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [1.0, 1.0],
        [-2.0, 0.0],
        [2.0, 0.0],
        [0.0, -2.0],
        [0.0, 2.0],
    ]
    y = [1.0 + 2.0 * row[0] - 3.0 * row[1] for row in X]
    y_hat = lasso_fit(X, y, lam=0.0, max_iter=1000, tol=1e-9)["y_hat"]

    max_err = max(abs(y_hat[i] - y[i]) for i in range(len(y)))
    passed = max_err < 1e-3
    _log.print_value("max |y_hat - y|", round(max_err, 10))
    return _finish_case(name, passed)


def test_lasso_negative_lambda_raises():
    name = "test_lasso_negative_lambda_raises"
    _print_case(
        name,
        "Passing lam=-0.5 should raise ValueError.",
        None,
        "ValueError",
    )

    try:
        lasso_fit([[1.0], [2.0], [3.0]], [1.0, 2.0, 3.0], lam=-0.5)
    except ValueError:
        return _finish_case(name, True)

    return _finish_case(name, False, "ValueError was not raised")


def test_lasso_handles_constant_feature():
    name = "test_lasso_handles_constant_feature"
    _print_case(
        name,
        "A constant feature becomes a zero column after standardization and should not crash.",
        "constant feature coefficient remains 0",
        None,
    )

    X = [[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0], [5.0, 5.0]]
    y = [3.0, 5.0, 7.0, 9.0, 11.0]
    beta = lasso_fit(X, y, lam=0.1)["beta_hat"]

    passed = abs(beta[2]) <= 1e-12
    _log.print_value("beta_hat", [round(b, 8) for b in beta])
    return _finish_case(name, passed)


def run_test_cases(test_functions: list) -> tuple[int, int]:
    passed_count = 0
    total_count = len(test_functions)

    for test_fn in test_functions:
        try:
            test_fn()
            passed_count += 1
        except AssertionError:
            pass
        except Exception as exc:
            _log.print_result(test_fn.__name__, False, details=str(exc))

    return passed_count, total_count


def test_cv_output_shape_and_keys():
    name = "test_cv_output_shape_and_keys"
    _print_case(
        name,
        "kfold_cv should return a dict with cv_scores, mean_cv_score, std_cv_score.",
        "Keys present and len(cv_scores) == k",
        None,
    )

    X = [[float(i)] for i in range(10)]
    y = [float(i) * 2 for i in range(10)]
    
    result = kfold_cv(X, y, k=5, model_fn=ridge_fit, lam=0.1)

    passed = (
        "cv_scores" in result
        and "mean_cv_score" in result
        and "std_cv_score" in result
        and len(result["cv_scores"]) == 5
    )
    _log.print_value("cv_scores length", len(result.get("cv_scores", [])))
    return _finish_case(name, passed)


def test_cv_k_out_of_bounds():
    name = "test_cv_k_out_of_bounds"
    _print_case(
        name,
        "Passing k=1 or k > n should raise ValueError.",
        None,
        "ValueError",
    )

    X = [[1.0], [2.0], [3.0]]
    y = [2.0, 4.0, 6.0]
    
    passed_1 = False
    try:
        kfold_cv(X, y, k=1, model_fn=ridge_fit)
    except ValueError:
        passed_1 = True

    passed_2 = False
    try:
        kfold_cv(X, y, k=5, model_fn=ridge_fit)
    except ValueError:
        passed_2 = True

    passed = passed_1 and passed_2
    return _finish_case(name, passed)


def test_cv_perfect_fit():
    name = "test_cv_perfect_fit"
    _print_case(
        name,
        "CV with lam=0 on strictly linear noiseless data should yield MSE approximately 0.",
        "mean_cv_score < 1e-5",
        None,
    )

    X = [[float(i), float(i * i)] for i in range(20)]
    y = [1.0 + 2.0 * row[0] - 1.0 * row[1] for row in X]
    
    result = kfold_cv(X, y, k=4, model_fn=ridge_fit, lam=0.0)
    
    passed = result["mean_cv_score"] < 1e-5
    _log.print_value("mean_cv_score", round(result["mean_cv_score"], 8))
    return _finish_case(name, passed)


def test_cv_mismatched_lengths_raises():
    name = "test_cv_mismatched_lengths_raises"
    _print_case(
        name,
        "Passing X and y with different numbers of rows should raise ValueError.",
        None,
        "ValueError",
    )

    X = [[1.0], [2.0], [3.0]]
    y = [2.0, 4.0]

    try:
        kfold_cv(X, y, k=2, model_fn=ridge_fit)
    except ValueError:
        return _finish_case(name, True)

    return _finish_case(name, False, "ValueError was not raised")


def test_cv_invalid_model_fn():
    name = "test_cv_invalid_model_fn"
    _print_case(
        name,
        "Passing a non-callable model_fn should raise ValueError.",
        None,
        "ValueError",
    )

    X = [[1.0], [2.0], [3.0]]
    y = [2.0, 4.0, 6.0]
    
    try:
        kfold_cv(X, y, k=2, model_fn="not_a_function")
    except ValueError:
        return _finish_case(name, True)

    return _finish_case(name, False)


