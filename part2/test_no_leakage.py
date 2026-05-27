"""No-leakage unit tests cho DataPipeline.

Kiểm tra pipeline fit trên train không bị ảnh hưởng bởi test data.
Đảm bảo transform() không sửa state đã học từ fit().
"""
import os
import sys
import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from part2.data_pipeline import DataPipeline
import pandas as pd


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


def _run_tests() -> tuple[int, int]:
    passed = 0
    total = 0

    def check(label: str, condition: bool):
        nonlocal passed, total
        total += 1
        status = "PASS" if condition else "FAIL"
        if condition:
            passed += 1
        print(f"  [{status}] {label}")

    df_train, df_test_1, df_test_2 = _make_train_test_dfs()

    # =================================================================
    print("\n=== Test 1: Scale stats không đổi khi fit cùng train ===")
    # =================================================================
    pipeline1 = DataPipeline()
    pipeline1.fit(df_train.copy())
    means1 = dict(pipeline1.models_['scale_means'])
    stds1 = dict(pipeline1.models_['scale_stds'])

    pipeline2 = DataPipeline()
    pipeline2.fit(df_train.copy())
    means2 = dict(pipeline2.models_['scale_means'])
    stds2 = dict(pipeline2.models_['scale_stds'])

    check(
        "scale_means giống nhau khi fit 2 lần cùng train",
        means1 == means2,
    )
    check(
        "scale_stds giống nhau khi fit 2 lần cùng train",
        stds1 == stds2,
    )

    # =================================================================
    print("\n=== Test 2: transform() không thay đổi pipeline state ===")
    # =================================================================
    pipeline = DataPipeline()
    pipeline.fit(df_train.copy())

    state_before = copy.deepcopy(pipeline.models_)
    medians_before = copy.deepcopy(pipeline.medians_)

    pipeline.transform(df_test_1.copy())
    pipeline.transform(df_test_2.copy())

    state_after = pipeline.models_
    medians_after = pipeline.medians_

    # So sánh scale_means
    check(
        "scale_means không đổi sau transform",
        dict(state_before.get('scale_means', {})) == dict(state_after.get('scale_means', {})),
    )
    # So sánh scale_stds
    check(
        "scale_stds không đổi sau transform",
        dict(state_before.get('scale_stds', {})) == dict(state_after.get('scale_stds', {})),
    )
    # So sánh winsorize bounds
    check(
        "winsorize bounds không đổi sau transform",
        state_before.get('winsorize', {}) == state_after.get('winsorize', {}),
    )
    # So sánh medians (dùng cho fallback imputation)
    check(
        "medians không đổi sau transform",
        medians_before == medians_after,
    )

    # =================================================================
    print("\n=== Test 3: Winsorize dùng bounds từ train ===")
    # =================================================================
    pipeline_w = DataPipeline()
    pipeline_w.fit(df_train.copy())

    train_bounds = pipeline_w.models_['winsorize']
    X_test_w, _ = pipeline_w.transform(df_test_1.copy())

    # df_test_1 có feat_A = [100, 200, 300], rất xa range train [1, 10]
    # Sau transform, giá trị phải bị clip về bounds từ train
    # Kiểm tra: giá trị scale sẽ khác nếu dùng test bounds vs train bounds
    if 'feat_A' in train_bounds:
        q01, q99 = train_bounds['feat_A']
        check(
            f"Winsorize feat_A bounds từ train: q01={q01:.2f}, q99={q99:.2f}",
            q01 >= 1.0 and q99 <= 10.0,
        )
        # Sau clip + standardize, giá trị test phải bằng nhau (vì đã bị clip cùng giá trị q99)
        # Test: nếu 3 giá trị test đều > q99, sau clip đều = q99, 
        # nên sau standardize đều bằng nhau
        feat_a_idx = list(pipeline_w.models_['encoded_cols']).index('feat_A')
        test_vals = [X_test_w[i][feat_a_idx] for i in range(len(X_test_w))]
        check(
            "3 giá trị test feat_A bằng nhau sau clip (đều vượt q99)",
            abs(test_vals[0] - test_vals[1]) < 1e-9 and abs(test_vals[1] - test_vals[2]) < 1e-9,
        )
    else:
        check("feat_A có trong winsorize bounds", False)

    # =================================================================
    print("\n=== Test 4: fit() không nhận test data ===")
    # =================================================================
    # fit() chỉ nhận 1 DataFrame (train), không có tham số test
    pipeline_leak = DataPipeline()
    pipeline_leak.fit(df_train.copy())
    means_train_only = dict(pipeline_leak.models_['scale_means'])

    # Nếu bị leakage, fit trên concat(train, test) sẽ cho means khác
    df_leaked = pd.concat([df_train, df_test_1], ignore_index=True)
    pipeline_leaked = DataPipeline()
    pipeline_leaked.fit(df_leaked.copy())
    means_leaked = dict(pipeline_leaked.models_['scale_means'])

    # Means phải khác nhau → chứng minh fit() chỉ dùng data truyền vào
    any_diff = any(
        abs(means_train_only.get(k, 0) - means_leaked.get(k, 0)) > 1e-6
        for k in means_train_only
    )
    check(
        "fit(train) cho means khác fit(train+test) → no leakage",
        any_diff,
    )

    print(f"\n{'='*50}")
    print(f"No-leakage tests: {passed}/{total} PASSED")
    print(f"{'='*50}\n")
    return passed, total


if __name__ == "__main__":
    _run_tests()
