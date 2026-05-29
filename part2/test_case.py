import os
import sys
import math
import copy
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from test_logger import TestLogger
from part2.advanced_methods import KernelRidgeRegression, BayesianLinearRegression, sample_rows, compare_models_summary
from part2.data_pipeline import DataPipeline

_log = TestLogger()


def _print_case(
    name: str,
    description: str,
    expected_output,
    expected_error,
) -> None:
    """In ra thông tin của một ca kiểm thử."""
    _log.print_group(name)
    _log.print_info(f"Ca kiem thu     : {name}")
    _log.print_info(f"Noi dung        : {description}")
    _log.print_info(f"Dau ra ky vong  : {expected_output}")
    _log.print_info(f"Loi ky vong     : {expected_error}")


def _finish_case(name: str, passed: bool, details: str = "") -> bool:
    """Hoàn tất một ca kiểm thử và in kết quả."""
    _log.print_result(name, passed, details=details)
    assert passed, details or f"{name} that bai"
    return True


def run_test_cases(test_functions: list) -> tuple[int, int]:
    """Chạy một danh sách các hàm kiểm thử và trả về số ca thành công."""
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


def test_krr_rbf_kernel():
    name = "test_krr_rbf_kernel"
    _print_case(
        name,
        "KRR RBF kernel phai tinh toan dung cac gia tri va bao dam tinh doi xung.",
        "K[0][0] == 1.0, K[0][1] == K[1][0], K[0][1] == exp(-0.5)",
        None,
    )
    krr = KernelRidgeRegression()
    X = [[0.0], [1.0]]
    K = krr.rbf_kernel(X, X, length_scale=1.0)

    passed = (
        abs(K[0][0] - 1.0) < 1e-12
        and abs(K[1][1] - 1.0) < 1e-12
        and abs(K[0][1] - K[1][0]) < 1e-12
        and abs(K[0][1] - math.exp(-0.5)) < 1e-9
    )

    _log.print_value("K[0][0]", round(K[0][0], 8))
    _log.print_value("K[0][1]", round(K[0][1], 8))
    _log.print_value("K[1][0]", round(K[1][0], 8))
    return _finish_case(name, passed)


def test_krr_fit_predict_shape_mse():
    name = "test_krr_fit_predict_shape_mse"
    _print_case(
        name,
        "KRR fit va predict phai tra ve dung kich thuoc va dat sai so MSE thap tren tap train.",
        "len(alpha) == len(y), len(y_hat) == len(y), MSE < 0.1",
        None,
    )
    krr = KernelRidgeRegression()
    X = [[0.0], [1.0], [2.0]]
    y = [0.0, 1.0, 4.0]
    model = krr.fit(X, y, lam=0.01, length_scale=1.0)
    y_hat = krr.predict(model, X)

    mse = sum((y[i] - y_hat[i]) ** 2 for i in range(len(y))) / len(y)

    passed = (
        len(model["alpha"]) == len(y)
        and len(y_hat) == len(y)
        and mse < 0.1
        and model["model_type"] == "kernel_ridge_rbf"
    )

    _log.print_value("MSE", round(mse, 6))
    _log.print_value("model_type", model["model_type"])
    return _finish_case(name, passed)


def test_sample_rows_reproducible():
    name = "test_sample_rows_reproducible"
    _print_case(
        name,
        "sample_rows phai bao dam tinh tai lap khi trung seed va tra ve dung so luong dong.",
        "X_a == X_b, y_a == y_b, kich thuoc chinh xac",
        None,
    )
    X = [[float(i)] for i in range(50)]
    y = [float(i) for i in range(50)]
    Xa, ya = sample_rows(X, y, max_rows=20, seed=7)
    Xb, yb = sample_rows(X, y, max_rows=20, seed=7)
    X_full, y_full = sample_rows(X, y, max_rows=100, seed=0)

    passed = (
        Xa == Xb
        and ya == yb
        and len(Xa) == 20
        and len(X_full) == 50
    )

    _log.print_value("Kich thuoc du lieu con", len(Xa))
    _log.print_value("Trung khop du lieu con", Xa == Xb)
    _log.print_value("Kich thuoc du lieu day du", len(X_full))
    return _finish_case(name, passed)


def test_krr_cv_returns_grid_values():
    name = "test_krr_cv_returns_grid_values"
    _print_case(
        name,
        "KRR kiem dinh cheo phai tra ve cac sieu tham so toi uu nam trong luoi.",
        "best_lam in lambda_grid, best_length_scale in ls_grid, len(cv_results) == 6",
        None,
    )
    krr = KernelRidgeRegression()
    X = [[float(i)] for i in range(30)]
    y = [float(i) * 0.5 for i in range(30)]
    lam_grid = [0.01, 0.1, 1.0]
    ls_grid = [1.0, 2.0]
    cv_result = krr.cross_validate(X, y, lam_grid, ls_grid, k=3, seed=42)

    passed = (
        cv_result["best_lam"] in lam_grid
        and cv_result["best_length_scale"] in ls_grid
        and len(cv_result["cv_results"]) == len(lam_grid) * len(ls_grid)
    )

    _log.print_value("best_lam", cv_result["best_lam"])
    _log.print_value("best_length_scale", cv_result["best_length_scale"])
    _log.print_value("cv_results length", len(cv_result["cv_results"]))
    return _finish_case(name, passed)

def test_blr_fit_predict_shape():
    name = "test_blr_fit_predict_shape"
    _print_case(
        name,
        "BLR fit va predict phai dung kich thuoc cua tham so va gia tri du doan.",
        "sigma2 > 0, len(m_n) == 3, len(y_mean) == 5, MSE < 200",
        None,
    )
    X = [[1.0, 10.0], [3.0, 7.0], [5.0, 3.0], [7.0, 12.0], [9.0, 1.0]]
    y = [2.5, 6.5, 10.5, 14.5, 18.5]
    sigma2 = BayesianLinearRegression.estimate_sigma2(X, y)
    model = BayesianLinearRegression.fit(X, y, sigma2=sigma2, alpha=1.0)
    y_mean, y_lower, y_upper = BayesianLinearRegression.predict(model, X, sigma2)

    mse = sum((y[i] - y_mean[i]) ** 2 for i in range(len(y))) / len(y)

    passed = (
        sigma2 > 0
        and model["model_type"] == "bayesian_lr"
        and len(model["m_n"]) == 3
        and len(model["S_n"]) == 3
        and len(model["S_n"][0]) == 3
        and len(y_mean) == 5
        and mse < 200.0
    )

    _log.print_value("sigma2", round(sigma2, 6))
    _log.print_value("beta_hat length", len(model["m_n"]))
    _log.print_value("MSE", round(mse, 6))
    return _finish_case(name, passed)


def test_blr_credible_interval_order():
    name = "test_blr_credible_interval_order"
    _print_case(
        name,
        "BLR phai bao dam khoang tin cay sap xep dung thu tu y_lower <= y_mean <= y_upper.",
        "True cho tat ca phan tu",
        None,
    )
    X = [[1.0, 10.0], [3.0, 7.0], [5.0, 3.0], [7.0, 12.0], [9.0, 1.0]]
    y = [2.5, 6.5, 10.5, 14.5, 18.5]
    sigma2 = BayesianLinearRegression.estimate_sigma2(X, y)
    model = BayesianLinearRegression.fit(X, y, sigma2=sigma2, alpha=1.0)
    y_mean, y_lower, y_upper = BayesianLinearRegression.predict(model, X, sigma2)

    all_ordered = all(y_lower[i] <= y_mean[i] <= y_upper[i] for i in range(5))

    passed = all_ordered
    _log.print_value("Khoang tin cay co dung thu tu", all_ordered)
    return _finish_case(name, passed)


def test_blr_cv_returns_grid_values():
    name = "test_blr_cv_returns_grid_values"
    _print_case(
        name,
        "BLR kiem dinh cheo phai tra ve alpha toi uu nam trong luoi.",
        "best_alpha in alpha_grid, best_cv_score >= 0, len(cv_results) == 4",
        None,
    )
    X = [[1.0, 10.0], [3.0, 7.0], [5.0, 3.0], [7.0, 12.0], [9.0, 1.0]]
    y = [2.5, 6.5, 10.5, 14.5, 18.5]
    alpha_grid = [0.01, 0.1, 1.0, 10.0]
    cv_result = BayesianLinearRegression.cross_validate(X, y, alpha_grid, k=3)

    passed = (
        cv_result["best_alpha"] in alpha_grid
        and cv_result["best_cv_score"] >= 0
        and len(cv_result["cv_results"]) == len(alpha_grid)
    )

    _log.print_value("best_alpha", cv_result["best_alpha"])
    _log.print_value("best_cv_score", round(cv_result["best_cv_score"], 6))
    _log.print_value("cv_results length", len(cv_result["cv_results"]))
    return _finish_case(name, passed)

def _make_train_test_dfs():
    """Tạo hai DataFrame giả lập train/test cho unit test."""
    train_data = {
        'feat_A': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        'feat_B': [10.0, 20.0, None, 40.0, 50.0, 60.0, None, 80.0, 90.0, 100.0],
        'feat_C': [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        'CO(GT)': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    }
    test_data_1 = {
        'feat_A': [100.0, 200.0, 300.0],
        'feat_B': [None, 500.0, 600.0],
        'feat_C': [5.0, 5.0, 5.0],
        'CO(GT)': [10.0, 20.0, 30.0],
    }
    test_data_2 = {
        'feat_A': [-50.0, -100.0],
        'feat_B': [999.0, None],
        'feat_C': [5.0, 5.0],
        'CO(GT)': [0.1, 0.2],
    }
    df_train = pd.DataFrame(train_data)
    df_test_1 = pd.DataFrame(test_data_1)
    df_test_2 = pd.DataFrame(test_data_2)
    return df_train, df_test_1, df_test_2


def test_pipeline_scale_stats_consistent():
    name = "test_pipeline_scale_stats_consistent"
    _print_case(
        name,
        "Thong ke chuan hoa (scale means, scale stds) cua pipeline phai dong nhat giua cac lan fit tren cung train data.",
        "means1 == means2, stds1 == stds2",
        None,
    )
    df_train, _, _ = _make_train_test_dfs()

    pipeline1 = DataPipeline()
    pipeline1.fit(df_train.copy())
    means1 = dict(pipeline1.models_['scale_means'])
    stds1 = dict(pipeline1.models_['scale_stds'])

    pipeline2 = DataPipeline()
    pipeline2.fit(df_train.copy())
    means2 = dict(pipeline2.models_['scale_means'])
    stds2 = dict(pipeline2.models_['scale_stds'])

    passed = (means1 == means2) and (stds1 == stds2)
    _log.print_value("Scale means trung khop", means1 == means2)
    _log.print_value("Scale stds trung khop", stds1 == stds2)
    return _finish_case(name, passed)


def test_pipeline_transform_no_side_effects():
    name = "test_pipeline_transform_no_side_effects"
    _print_case(
        name,
        "Goi transform() khong duoc phep thay doi trang thai (state) da fit cua pipeline.",
        "scale_means, scale_stds, winsorize bounds va medians khong thay doi",
        None,
    )
    df_train, df_test_1, df_test_2 = _make_train_test_dfs()
    pipeline = DataPipeline()
    pipeline.fit(df_train.copy())

    state_before = copy.deepcopy(pipeline.models_)
    medians_before = copy.deepcopy(pipeline.medians_)

    pipeline.transform(df_test_1.copy())
    pipeline.transform(df_test_2.copy())

    state_after = pipeline.models_
    medians_after = pipeline.medians_

    passed = (
        dict(state_before.get('scale_means', {})) == dict(state_after.get('scale_means', {}))
        and dict(state_before.get('scale_stds', {})) == dict(state_after.get('scale_stds', {}))
        and state_before.get('winsorize', {}) == state_after.get('winsorize', {})
        and medians_before == medians_after
    )

    _log.print_value("Scale means khong doi", dict(state_before.get('scale_means', {})) == dict(state_after.get('scale_means', {})))
    _log.print_value("Scale stds khong doi", dict(state_before.get('scale_stds', {})) == dict(state_after.get('scale_stds', {})))
    _log.print_value("Winsorize bounds khong doi", state_before.get('winsorize', {}) == state_after.get('winsorize', {}))
    return _finish_case(name, passed)


def test_pipeline_winsorize_train_bounds():
    name = "test_pipeline_winsorize_train_bounds"
    _print_case(
        name,
        "Phep Winsorize tren test set phai su dung nguong (bounds) hoc tu tap train va clip dung cach.",
        "feat_A q01 va q99 hop le, cac gia tri test cuc doan phai bang nhau sau clip",
        None,
    )
    df_train, df_test_1, _ = _make_train_test_dfs()
    pipeline_w = DataPipeline()
    pipeline_w.fit(df_train.copy())

    train_bounds = pipeline_w.models_['winsorize']
    X_test_w, _ = pipeline_w.transform(df_test_1.copy())

    passed = False
    if 'feat_A' in train_bounds:
        q01, q99 = train_bounds['feat_A']
        valid_bounds = (q01 >= 1.0 and q99 <= 10.0)
        
        # Tat ca gia tri test cua feat_A deu vuot qua q99, nen sau standardize chung phai bang nhau
        feat_a_idx = list(pipeline_w.models_['encoded_cols']).index('feat_A')
        test_vals = [X_test_w[i][feat_a_idx] for i in range(len(X_test_w))]
        vals_equal = (abs(test_vals[0] - test_vals[1]) < 1e-9 and abs(test_vals[1] - test_vals[2]) < 1e-9)
        passed = valid_bounds and vals_equal

        _log.print_value("q01 feat_A", round(q01, 4))
        _log.print_value("q99 feat_A", round(q99, 4))
        _log.print_value("Gia tri test bang nhau sau clip", vals_equal)
    else:
        _log.print_warning("feat_A khong co trong winsorize bounds")

    return _finish_case(name, passed)


def test_pipeline_no_leakage_fit():
    name = "test_pipeline_no_leakage_fit"
    _print_case(
        name,
        "fit() tren train chi duoc su dung train data, khong ro ri test data.",
        "scale_means khac nhau giua fit(train) va fit(train + test)",
        None,
    )
    df_train, df_test_1, _ = _make_train_test_dfs()

    pipeline_leak = DataPipeline()
    pipeline_leak.fit(df_train.copy())
    means_train_only = dict(pipeline_leak.models_['scale_means'])

    df_leaked = pd.concat([df_train, df_test_1], ignore_index=True)
    pipeline_leaked = DataPipeline()
    pipeline_leaked.fit(df_leaked.copy())
    means_leaked = dict(pipeline_leaked.models_['scale_means'])

    any_diff = any(
        abs(means_train_only.get(k, 0) - means_leaked.get(k, 0)) > 1e-6
        for k in means_train_only
    )

    passed = any_diff
    _log.print_value("Co su khac biet khi ro ri du lieu", any_diff)
    return _finish_case(name, passed)

def run_advanced_methods_tests() -> tuple[int, int]:
    """Chạy toàn bộ unit tests cho các phương pháp nâng cao (KRR, BLR)."""
    _log.print_suite_header("KRR & BLR METHODS - UNIT TESTS")
    passed, total = run_test_cases([
        test_krr_rbf_kernel,
        test_krr_fit_predict_shape_mse,
        test_sample_rows_reproducible,
        test_krr_cv_returns_grid_values,
        test_blr_fit_predict_shape,
        test_blr_credible_interval_order,
        test_blr_cv_returns_grid_values,
    ])
    _log.print_summary(passed, total)
    return passed, total


def run_pipeline_tests() -> tuple[int, int]:
    """Chạy toàn bộ unit tests cho DataPipeline chống rò rỉ dữ liệu (No-Leakage)."""
    _log.print_suite_header("DATA PIPELINE & NO-LEAKAGE - UNIT TESTS")
    passed, total = run_test_cases([
        test_pipeline_scale_stats_consistent,
        test_pipeline_transform_no_side_effects,
        test_pipeline_winsorize_train_bounds,
        test_pipeline_no_leakage_fit,
    ])
    _log.print_summary(passed, total)
    return passed, total


def run_all_tests():
    """Chạy tất cả các unit tests của phần 2."""
    run_advanced_methods_tests()
    run_pipeline_tests()


if __name__ == "__main__":
    run_all_tests()
