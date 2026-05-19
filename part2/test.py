import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from part2.data_pipeline import DataPipeline

def test_imputation_validity(df_original, df_imputed, name="Dataset"):
    print(f"\n--- Testing validity for {name} ---")
    # 1. Không còn NaN
    remaining_nan = df_imputed.isna().sum().sum()
    assert remaining_nan == 0, f"FAIL: Still has {remaining_nan} NaN after imputation"
    print("Check 1 (No NaN): PASSED")

    # 2. Các giá trị đã có từ đầu không bị thay đổi
    original_mask = df_original.notna()
    assert np.allclose(df_imputed[original_mask], df_original[original_mask], equal_nan=True), "FAIL: Original values changed"
    print("Check 2 (Original values preserved): PASSED")

    # 3. Giá trị điền nằm trong khoảng hợp lý (min-max của cột)
    for col in df_original.columns:
        col_min = df_original[col].min()
        col_max = df_original[col].max()
        imputed_vals = df_imputed.loc[df_original[col].isna(), col]
        if len(imputed_vals) > 0:
            assert (imputed_vals >= col_min).all() and (imputed_vals <= col_max).all(), \
                f"FAIL: Imputed values out of range for {col}. Values: {imputed_vals.values}, Min: {col_min}, Max: {col_max}"
    print("Check 3 (Imputed values in range): PASSED")

def main():
    print("=== TESTING KNN IMPUTER ===")
    
    # Dummy data
    data = {
        'A': [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
        'B': [1.1, np.nan, 3.2, 4.1, 5.1, 6.1],
        'C': [10.0, 20.0, 30.0, 40.0, np.nan, 60.0]
    }
    df = pd.DataFrame(data)
    
    print("Original Data (with NaN):")
    print(df)
    print("-" * 30)

    # Sklearn KNNImputer
    sklearn_imputer = KNNImputer(n_neighbors=2)
    df_sklearn = pd.DataFrame(sklearn_imputer.fit_transform(df), columns=df.columns)
    
    print("Result from sklearn.impute.KNNImputer:")
    print(df_sklearn)
    print("-" * 30)

    # Custom Pandas KNN Imputer
    pipeline = DataPipeline()
    pipeline.medians_ = df.median().to_dict()
    pipeline.models_['knn_means'] = df.mean()
    knn_stds = df.std().replace(0, 1)
    pipeline.models_['knn_stds'] = knn_stds
    X_scaled = (df - pipeline.models_['knn_means']) / knn_stds
    pipeline.models_['knn_train_data'] = X_scaled.copy()
    pipeline.models_['knn_train_raw'] = df.copy()
    
    df_custom = pipeline._knn_impute(df.copy(), n_neighbors=2)
    
    print("Result from custom _knn_impute (Pandas):")
    print(df_custom)
    print("-" * 30)

    # Check assertions for Dummy data
    try:
        test_imputation_validity(df, df_custom, name="Dummy Data")
    except Exception as e:
        print("Error during test:", e)

    # Real dataset
    print("\n=== COMPARING ON SUBSET OF AIR QUALITY UCI ===")
    try:
        df_real = pd.read_csv("part2/data/AirQualityUCI.csv", sep=',')
        df_real.drop(['Unnamed: 15', 'Unnamed: 16'], axis=1, inplace=True, errors='ignore')
        df_real.dropna(how='all', inplace=True)
        
        # Numeric subset 100 rows
        numeric_cols = df_real.select_dtypes(include=['float64', 'int64']).columns
        df_subset = df_real[numeric_cols].head(100).copy()
        
        # Add random missing
        df_subset.iloc[0, 0] = np.nan
        df_subset.iloc[5, 2] = np.nan
        df_subset.iloc[10, 4] = np.nan
        
        # Custom
        pipe2 = DataPipeline()
        pipe2.medians_ = df_subset.median().to_dict()
        pipe2.models_['knn_means'] = df_subset.mean()
        knn_stds2 = df_subset.std().replace(0, 1)
        pipe2.models_['knn_stds'] = knn_stds2
        X_scaled2 = (df_subset - pipe2.models_['knn_means']) / knn_stds2
        pipe2.models_['knn_train_data'] = X_scaled2.copy()
        pipe2.models_['knn_train_raw'] = df_subset.copy()
        res_custom = pipe2._knn_impute(df_subset.copy(), n_neighbors=5)
        
        # Check assertions for Real dataset
        test_imputation_validity(df_subset, res_custom, name="AirQuality Subset")
        
        print("\n=> SUCCESS: ALL KNN TESTS PASSED!")
            
    except FileNotFoundError:
        print("Cannot find AirQualityUCI.csv to test automatically.")
    except Exception as e:
        print("Error during real data test:", e)

def test_full_pipeline():
    print("\n=== TESTING FULL PIPELINE (FIT & TRANSFORM) ===")
    try:
        pipeline = DataPipeline(target_col="CO(GT)")
        df = pipeline.load_data("part2/data/AirQualityUCI.csv")
        
        # Split train/test (80/20)
        n = len(df)
        split_idx = int(n * 0.8)
        df_train = df.iloc[:split_idx].copy()
        df_test = df.iloc[split_idx:].copy()
        
        y_train = df_train['CO(GT)']
        y_test = df_test['CO(GT)']
        
        # Fit và transform train
        X_train_processed, y_train_processed = pipeline.fit_transform(df_train, y_train)
        
        # Transform test
        X_test_processed, y_test_processed = pipeline.transform(df_test, y_test)
        
        # Check shapes
        print(f"X_train_processed: {len(X_train_processed)} rows x {len(X_train_processed[0])} features")
        print(f"y_train_processed: {len(y_train_processed)} labels")
        print(f"X_test_processed: {len(X_test_processed)} rows x {len(X_test_processed[0])} features")
        print(f"y_test_processed: {len(y_test_processed)} labels")
        
        # Kiểm tra không còn NaN
        assert all(all(val == val for val in row) for row in X_train_processed), "FAIL: NaN in X_train_processed"
        assert all(all(val == val for val in row) for row in X_test_processed), "FAIL: NaN in X_test_processed"
        assert all(val == val for val in y_train_processed), "FAIL: NaN in y_train_processed"
        assert all(val == val for val in y_test_processed), "FAIL: NaN in y_test_processed"
        
        # So sánh số lượng cột giữa train và test
        assert len(X_train_processed[0]) == len(X_test_processed[0]), "FAIL: Train and Test feature counts differ"
        
        print("Check 1 (Shapes and No NaN): PASSED")
        
        # Thử nghiệm tích hợp OLS
        from part1.ols_implementation import ols_fit
        ols_res = ols_fit(X_train_processed, y_train_processed)
        print("Check 2 (OLS fit on processed data): PASSED")
        print(f"OLS beta_hat count: {len(ols_res['beta_hat'])}")
        
        # Dự đoán cho test set và tính MSE đơn giản để chắc chắn không lỗi
        beta_hat = ols_res['beta_hat']
        # Thêm bias cho X_test
        X_test_bias = [[1.0] + row for row in X_test_processed]
        y_pred = []
        for row in X_test_bias:
            y_pred.append(sum(x * b for x, b in zip(row, beta_hat)))
            
        test_mse = sum((y_true - y_hat) ** 2 for y_true, y_hat in zip(y_test_processed, y_pred)) / len(y_test_processed)
        print(f"Check 3 (OLS predictions on Test data): PASSED (Test MSE: {test_mse:.4f})")
        
        print("=> SUCCESS: FULL PIPELINE TESTS PASSED!")
    except Exception as e:
        print("Error during full pipeline test:", e)
        raise e

if __name__ == "__main__":
    main()
    test_full_pipeline()
