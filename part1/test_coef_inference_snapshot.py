"""Snapshot / golden-value regression test for coef_inference.

Captures the exact numerical output of the current (scipy-based)
implementation so that after refactoring to pure Python we can
verify the results are identical.

Golden values were generated with:
    scipy 1.x  |  Student-t CDF/PPF
    Fixture: _coef_inference_fixture() from test_case.py
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
from test_logger import TestLogger

_log = TestLogger()


# ---------------------------------------------------------------------------
# Fixture (same as _coef_inference_fixture in test_case.py)
# ---------------------------------------------------------------------------
def _fixture():
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


# ---------------------------------------------------------------------------
# Golden values (captured from scipy-based implementation)
# ---------------------------------------------------------------------------
GOLDEN = {
    "index": ["intercept", "x1", "x2"],
    "coef": [
        1.017777777777776560e+00,
        1.995555555555555527e+00,
        -5.180555555555554914e-01,
    ],
    "std_err": [
        3.935827834235210154e-02,
        9.408428057785097817e-03,
        3.977773898197969199e-02,
    ],
    "t_stat": [
        2.585930636814925521e+01,
        2.121029722817844174e+02,
        -1.302375571900272178e+01,
    ],
    "p_value": [
        9.323187366510818596e-10,
        5.855913414869361407e-18,
        3.822583638904783001e-07,
    ],
    "ci_lower": [
        9.287431665102192646e-01,
        1.974272212633965351e+00,
        -6.080390527135581813e-01,
    ],
    "ci_upper": [
        1.106812389045333855e+00,
        2.016838898477145481e+00,
        -4.280720583975527460e-01,
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _print_case(name, description, expected_output, expected_error):
    _log.print_group(name)
    _log.print_info(f"Test case       : {name}")
    _log.print_info(f"Noi dung        : {description}")
    _log.print_info(f"Expected output : {expected_output}")
    _log.print_info(f"Expected error  : {expected_error}")


def _finish_case(name, passed, details=""):
    _log.print_result(name, passed, details=details)
    assert passed, details or f"{name} failed"
    return True


def _close(actual, expected, rtol=1e-8, atol=1e-12):
    return abs(actual - expected) <= atol + rtol * abs(expected)


def _get_table():
    """Call coef_inference and return the result (works with both old DataFrame and new dict)."""
    from part1.ols_implementation import coef_inference

    X, y, result = _fixture()
    return coef_inference(X, y, result["beta_hat"], result["sigma2_hat"])


def _extract_column(table, col_name):
    """Extract a column as list[float], supporting both pd.DataFrame and dict."""
    if isinstance(table, dict):
        return [float(v) for v in table[col_name]]
    return [float(v) for v in table[col_name]]


def _extract_index(table):
    """Extract the index/labels, supporting both pd.DataFrame and dict."""
    if isinstance(table, dict):
        return list(table["index"])
    return list(table.index)


def _extract_keys(table):
    """Extract column names, supporting both pd.DataFrame and dict."""
    if isinstance(table, dict):
        return sorted([k for k in table.keys() if k != "index"])
    return sorted(list(table.columns))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_snapshot_index():
    name = "test_snapshot_index"
    _print_case(
        name,
        "Index labels should match golden values exactly.",
        f"index = {GOLDEN['index']}",
        None,
    )

    table = _get_table()
    actual_index = _extract_index(table)

    passed = actual_index == GOLDEN["index"]
    _log.print_value("actual index", actual_index)
    _log.print_value("expected index", GOLDEN["index"])
    return _finish_case(name, passed, f"got {actual_index}")


def test_snapshot_keys():
    name = "test_snapshot_keys"
    _print_case(
        name,
        "Output should contain all expected column keys.",
        "coef, std_err, t_stat, p_value, ci_lower, ci_upper",
        None,
    )

    table = _get_table()
    actual_keys = _extract_keys(table)
    expected_keys = sorted(["coef", "std_err", "t_stat", "p_value", "ci_lower", "ci_upper"])

    passed = actual_keys == expected_keys
    _log.print_value("actual keys", actual_keys)
    _log.print_value("expected keys", expected_keys)
    return _finish_case(name, passed, f"got {actual_keys}")


def test_snapshot_coef():
    name = "test_snapshot_coef"
    _print_case(
        name,
        "coef values should match golden values to 1e-12.",
        "all coef values within tolerance",
        None,
    )

    table = _get_table()
    actual = _extract_column(table, "coef")
    expected = GOLDEN["coef"]

    passed = all(_close(a, e) for a, e in zip(actual, expected))
    _log.print_value("actual", [f"{v:.15e}" for v in actual])
    _log.print_value("expected", [f"{v:.15e}" for v in expected])
    return _finish_case(name, passed)


def test_snapshot_std_err():
    name = "test_snapshot_std_err"
    _print_case(
        name,
        "std_err values should match golden values to 1e-12.",
        "all std_err values within tolerance",
        None,
    )

    table = _get_table()
    actual = _extract_column(table, "std_err")
    expected = GOLDEN["std_err"]

    passed = all(_close(a, e) for a, e in zip(actual, expected))
    _log.print_value("actual", [f"{v:.15e}" for v in actual])
    _log.print_value("expected", [f"{v:.15e}" for v in expected])
    return _finish_case(name, passed)


def test_snapshot_t_stat():
    name = "test_snapshot_t_stat"
    _print_case(
        name,
        "t_stat values should match golden values to 1e-12.",
        "all t_stat values within tolerance",
        None,
    )

    table = _get_table()
    actual = _extract_column(table, "t_stat")
    expected = GOLDEN["t_stat"]

    passed = all(_close(a, e) for a, e in zip(actual, expected))
    _log.print_value("actual", [f"{v:.15e}" for v in actual])
    _log.print_value("expected", [f"{v:.15e}" for v in expected])
    return _finish_case(name, passed)


def test_snapshot_p_value():
    name = "test_snapshot_p_value"
    _print_case(
        name,
        "p_value values should match golden values (relaxed tolerance for pure-Python CDF).",
        "all p_value values within rtol=1e-6",
        None,
    )

    table = _get_table()
    actual = _extract_column(table, "p_value")
    expected = GOLDEN["p_value"]

    # p-values can be very small; use relative tolerance
    passed = all(_close(a, e, rtol=1e-6, atol=1e-15) for a, e in zip(actual, expected))
    _log.print_value("actual", [f"{v:.15e}" for v in actual])
    _log.print_value("expected", [f"{v:.15e}" for v in expected])
    if not passed:
        diffs = [abs(a - e) / max(abs(e), 1e-30) for a, e in zip(actual, expected)]
        _log.print_value("relative diffs", [f"{d:.6e}" for d in diffs])
    return _finish_case(name, passed)


def test_snapshot_ci_lower():
    name = "test_snapshot_ci_lower"
    _print_case(
        name,
        "ci_lower values should match golden values (relaxed tolerance for pure-Python PPF).",
        "all ci_lower values within rtol=1e-6",
        None,
    )

    table = _get_table()
    actual = _extract_column(table, "ci_lower")
    expected = GOLDEN["ci_lower"]

    passed = all(_close(a, e, rtol=1e-6, atol=1e-12) for a, e in zip(actual, expected))
    _log.print_value("actual", [f"{v:.15e}" for v in actual])
    _log.print_value("expected", [f"{v:.15e}" for v in expected])
    return _finish_case(name, passed)


def test_snapshot_ci_upper():
    name = "test_snapshot_ci_upper"
    _print_case(
        name,
        "ci_upper values should match golden values (relaxed tolerance for pure-Python PPF).",
        "all ci_upper values within rtol=1e-6",
        None,
    )

    table = _get_table()
    actual = _extract_column(table, "ci_upper")
    expected = GOLDEN["ci_upper"]

    passed = all(_close(a, e, rtol=1e-6, atol=1e-12) for a, e in zip(actual, expected))
    _log.print_value("actual", [f"{v:.15e}" for v in actual])
    _log.print_value("expected", [f"{v:.15e}" for v in expected])
    return _finish_case(name, passed)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_all_snapshot_tests():
    from part1.test_case import run_test_cases

    _log.print_suite_header("COEF INFERENCE — SNAPSHOT REGRESSION TESTS")
    tests = [
        test_snapshot_index,
        test_snapshot_keys,
        test_snapshot_coef,
        test_snapshot_std_err,
        test_snapshot_t_stat,
        test_snapshot_p_value,
        test_snapshot_ci_lower,
        test_snapshot_ci_upper,
    ]
    passed, total = run_test_cases(tests)
    _log.print_summary(passed, total)
    return passed, total


if __name__ == "__main__":
    passed, total = run_all_snapshot_tests()
    sys.exit(0 if passed == total else 1)
