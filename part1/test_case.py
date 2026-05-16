import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from test_logger import TestLogger

from part1.ridge_lasso import lasso_fit, lasso_trace, make_lambda_grid, ridge_fit
from part1.cross_validation import kfold_cv


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


def run_ridge_test_cases() -> tuple[int, int]:
    """Run all ridge_fit unit tests."""
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


def run_lasso_test_cases() -> tuple[int, int]:
    """Run all lasso_fit unit tests."""
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


def run_cv_test_cases() -> tuple[int, int]:
    """Run all kfold_cv unit tests."""
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
