import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import RANDOM_STATE
from part1.ols_implementation import (
    ols_fit, model_metrics, coef_inference, vif,
    _chi2_sf, _standard_normal_ppf_mc,
)
from part1.ridge_lasso import ridge_fit, lasso_fit
from part1.cross_validation import predict, select_lambda_cv
from part2.data_pipeline import DataPipeline
from part2.advanced_methods import KernelRidgeRegression, BayesianLinearRegression, sample_rows
import pandas as pd
import math


def _sw_norm_cdf(z: float) -> float:
    """
    Tính CDF phan phoi chuan N(0,1) tại điểm z.

    Sử dụng hàm erf từ stdlib math.
    Tham số:
        z : float -- điểm cần tính xác suất.

    Trả về:
        float -- P(Z <= z) với Z ~ N(0,1).
    """
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


class ModelComparator:
    def __init__(self, data_filepath: str, test_size: float = 0.2):
        self.data_filepath = data_filepath
        self.test_size = test_size
        self.pipeline = DataPipeline()

        self.df_train = None
        self.df_test = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None

        # Lưu kết quả từng mô hình
        self.results = {}

    def prepare_data(self):
        """Chuẩn bị dữ liệu train/test.

        Quy trình chuẩn:
            1. Load raw DataFrame từ filepath.
            2. Loại bỏ các hàng có target CO(GT) bị NaN trên DataFrame thô
               Trước khi split và trước khi pipeline
               - KNN Imputer không chụp các dòng lỗi vào bộ nhớ nền.
               - X và y luôn đồng bộ số dòng xuyên suốt pipeline.
               3. Shuffle + split thành df_train (80%) / df_test (20%).
            4. pipeline.fit(df_train) -- chỉ học thống kê từ train.
            5. pipeline.transform() cho cả train và test.
        """
        # Load raw DataFrame
        df = self.pipeline.load_data(self.data_filepath)

        # Loại bỏ các hàng có target bị NaN ngay trên DataFrame thô
        n_before = len(df)
        df = df.dropna(subset=[self.pipeline.target_col])
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            print(f"[prepare_data] Loai {n_dropped}/{n_before} mau co target NaN")

        # Shuffle rồi split trên tập đã sạch nhãn
        df_shuffled = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        n_test = int(len(df_shuffled) * self.test_size)
        self.df_test  = df_shuffled.iloc[:n_test].copy()
        self.df_train = df_shuffled.iloc[n_test:].copy()

        # Fit pipeline trên train
        self.pipeline.fit(self.df_train)

        # Transform cả train lẫn test bằng thống kê đã học từ train
        self.X_train, self.y_train = self.pipeline.transform(self.df_train)
        self.X_test,  self.y_test  = self.pipeline.transform(self.df_test)

        # Lấy tên features (trước polynomial expansion)
        encoded_cols = self.pipeline.models_.get('encoded_cols', None)
        if encoded_cols:
            self.feature_names = list(encoded_cols)
        else:
            p_base = len(self.X_train[0]) if self.X_train else 0
            self.feature_names = [f'x{i+1}' for i in range(p_base)]

        print(
            f"[prepare_data] Train: {len(self.X_train)} mau | "
            f"Test: {len(self.X_test)} mau | "
            f"Features (sau poly): {len(self.X_train[0])}"
        )

    def _compute_test_metrics(self, y_pred: list[float], p: int) -> dict:
        """Tính các metrics đánh giá mô hình trên test set.

        Tham số
        -------
        y_pred : list[float] -- giá trị dự đoán trên X_test.
        p      : int         -- số features (không tính intercept).

        Trả về
        ------
        dict với các key: 'MAE', 'RMSE', 'R2', 'R2_adj', 'RSS', 'TSS',
                          'MSS', 'F_stat', 'F_pvalue'
        """
        return model_metrics(self.y_test, y_pred, p)

    # Helpers: JSON I/O

    @staticmethod
    def _save_result_json(model_name: str, data: dict, out_dir: str = "part2/outputs"):
        """Lưu metrics + diagnostics của một model ra file JSON.

        Chỉ serialize các giá trị JSON-safe (bỏ qua 'result', 'cv_result', ...).
        """
        import json
        os.makedirs(out_dir, exist_ok=True)
        safe_keys = {'metrics', 'selected_features', 'removed_features',
                     'lambda_opt', 'shapiro', 'bp_test'}
        payload = {k: v for k, v in data.items() if k in safe_keys}
        filename = os.path.join(out_dir, f"{model_name.replace(' ', '_').lower()}.json")
        with open(filename, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"[save] Ket qua '{model_name}' -> {filename}")

    def _load_or_run_lambda_cv(
        self,
        model_fn,
        model_name: str,
        k: int = 5,
        config_path: str = None,
        **cv_model_kwargs,
    ) -> float:
        """Đọc lambda_opt từ cache hoặc chạy k-Fold CV nếu chưa có.

        config_path : str | None -- đường dẫn tới file JSON lưu lambda_opt.
        **cv_model_kwargs -- kwargs truyền vào model_fn khi chạy CV (vd: max_iter, tol).
        """
        import json
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            lam = cfg['lambda_opt']
            print(f"[{model_name}] Doc lambda_opt tu cache: {lam:.6g} ({config_path})")
            return lam

        cv_result = select_lambda_cv(
            self.X_train, self.y_train, k=k, model_fn=model_fn, **cv_model_kwargs
        )
        lam = cv_result['lambda_opt']
        print(f"[{model_name}] lambda_opt = {lam:.6g} (CV MSE={cv_result['best_cv_score']:.4f})")

        if config_path:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump({'lambda_opt': lam, 'best_cv_score': cv_result['best_cv_score']}, f, indent=2)
            print(f"[{model_name}] Da luu lambda config: {config_path}")
        return lam

    def train_ols_full(self):
        """Train OLS với toàn bộ features sau pipeline.

        Lưu vào self.results['OLS Full']:
            result    : dict trả về bởi ols_fit (beta_hat, sigma2_hat, y_hat, residuals)
            y_pred    : list[float] -- dự đoán trên test set
            metrics   : dict từ _compute_test_metrics
            residuals : list[float] -- phần dư trên test set (y_test - y_pred)
        """
        # Fit trên train
        result = ols_fit(self.X_train, self.y_train)

        # Predict trên test
        y_pred = predict(self.X_test, result)

        # Số features (sau poly expansion, không tính intercept)
        p = len(self.X_train[0])

        # Tính metrics trên test set
        metrics = self._compute_test_metrics(y_pred, p)

        # Tính residuals trên test set
        residuals = [self.y_test[i] - y_pred[i] for i in range(len(self.y_test))]

        # Lưu kết quả
        self.results['OLS Full'] = {
            'result':    result,
            'y_pred':    y_pred,
            'metrics':   metrics,
            'residuals': residuals,
        }

        print(
            f"[OLS Full] R2={metrics['R2']:.4f} | "
            f"RMSE={metrics['RMSE']:.4f} | "
            f"MAE={metrics['MAE']:.4f}"
        )

    def _select_features(
        self,
        p_threshold: float = 0.05,
        vif_threshold: float = 10.0,
        config_path: str = None,
    ) -> tuple[list[int], list[str], list[str]]:
        """Chọn biến theo Cách 2: p-value trước, VIF có điều kiện trên survivors.

        Bước 1: Lọc p-value -- loại biến có p_value > p_threshold.
        Bước 2: Tính VIF trên tập biến đã qua lọc p-value.
                (Nếu poly_degree > 1, VIF không có ý nghĩa -> bỏ qua).
        Bước 3: Loại thêm biến có VIF > vif_threshold trong tập survivors.

        config_path : str | None
            Nếu truyền vào, đọc kết quả đã tính từ file JSON (bỏ qua tính toán).
            Nếu chưa có file, sau khi tính xong sẽ lưu vào file đó.

        Returns: (selected_idx, selected_names, removed_names)
        """
        import json

        # Đọc từ cache nếu có
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            selected_idx   = cfg['selected_idx']
            selected_names = cfg['selected_names']
            removed_names  = cfg['removed_names']
            print(f"[_select_features] Doc tu cache: {config_path}")
            print(f"[_select_features] Giu lai: {len(selected_idx)} bien | Loai: {len(removed_names)} bien")
            return selected_idx, selected_names, removed_names

        p_total = len(self.X_train[0])
        names_all = [
            self.feature_names[j] if j < len(self.feature_names) else f'x{j+1}'
            for j in range(p_total)
        ]

        # Lọc bằng p-value
        result = ols_fit(self.X_train, self.y_train)
        inference_result = coef_inference(
            self.X_train, self.y_train,
            result['beta_hat'], result['sigma2_hat']
        )
        p_values = inference_result['p_value'][1:]

        pval_pass_idx   = [j for j in range(p_total) if p_values[j] <= p_threshold]
        pval_reject_idx = [j for j in range(p_total) if p_values[j] >  p_threshold]
        print(f"[_select_features] Buoc 1 (p-value<={p_threshold}): "
              f"con lai {len(pval_pass_idx)}/{p_total} bien")

        # VIF chỉ trên survivors
        poly_degree = self.pipeline.models_.get('poly_degree', 1)
        vif_reject_idx = []

        if poly_degree > 1:
            print(f"[_select_features] Buoc 2: Bo qua VIF (poly_degree={poly_degree} > 1)")
        elif len(pval_pass_idx) > 1:
            X_survivors = [[row[j] for j in pval_pass_idx] for row in self.X_train]
            vif_dict = vif(X_survivors)
            # vif_dict key là 'x1','x2',... tương ứng với pval_pass_idx
            for local_j, global_j in enumerate(pval_pass_idx):
                vif_val = vif_dict.get(f'x{local_j+1}', float('inf'))
                if vif_val > vif_threshold:
                    vif_reject_idx.append(global_j)
            print(f"[_select_features] Buoc 2 (VIF<={vif_threshold}): "
                  f"them loai {len(vif_reject_idx)} bien")
        else:
            print("[_select_features] Buoc 2: Bo qua VIF (< 2 survivors)")

        # Tổng hợp kết quả
        reject_set = set(pval_reject_idx) | set(vif_reject_idx)
        selected_idx   = [j for j in range(p_total) if j not in reject_set]
        selected_names = [names_all[j] for j in selected_idx]
        removed_names  = [names_all[j] for j in range(p_total) if j in reject_set]

        print(f"[_select_features] Ket qua cuoi: "
              f"Giu lai {len(selected_idx)}/{p_total} bien | Loai {len(removed_names)} bien")

        # Lưu cache
        if config_path:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump({
                    'selected_idx':   selected_idx,
                    'selected_names': selected_names,
                    'removed_names':  removed_names,
                    'p_threshold':    p_threshold,
                    'vif_threshold':  vif_threshold,
                    'poly_degree':    poly_degree,
                }, f, indent=2)
            print(f"[_select_features] Da luu config: {config_path}")

        return selected_idx, selected_names, removed_names

    def train_ols_selected(
        self,
        p_threshold: float = 0.05,
        vif_threshold: float = 10.0,
        config_path: str = None,
    ):
        """Train OLS chỉ với các biến được chọn qua p-value/VIF (Cach 2).

        config_path : str | None
            Đường dẫn file JSON cache feature selection.
            Nếu có sẵn, bỏ qua bước tính toán (nhanh hơn nhiều).
        """
        # Chọn biến (có cache)
        selected_idx, selected_names, removed_names = self._select_features(
            p_threshold=p_threshold,
            vif_threshold=vif_threshold,
            config_path=config_path,
        )

        if len(selected_idx) == 0:
            raise ValueError(
                "Không có biến nào vượt qua bộ lọc p-value/VIF. "
                "Hãy tăng ngưỡng p_threshold hoặc vif_threshold."
            )

        # Tạo X_train_sel và X_test_sel chỉ gồm các cột đã chọn
        X_train_sel = [[row[j] for j in selected_idx] for row in self.X_train]
        X_test_sel  = [[row[j] for j in selected_idx] for row in self.X_test]

        # Fit OLS trên tập feature đã chọn
        result = ols_fit(X_train_sel, self.y_train)

        # Predict trên test
        y_pred = predict(X_test_sel, result)

        # p = số features đã chọn
        p = len(selected_idx)

        # Tính metrics và residuals trên test set
        metrics   = model_metrics(self.y_test, y_pred, p)
        residuals = [self.y_test[i] - y_pred[i] for i in range(len(self.y_test))]

        # Lưu kết quả
        self.results['OLS Selected'] = {
            'result':            result,
            'y_pred':            y_pred,
            'metrics':           metrics,
            'residuals':         residuals,
            'selected_features': selected_names,
            'removed_features':  removed_names,
        }

        print(
            f"[OLS Selected] R2={metrics['R2']:.4f} | "
            f"RMSE={metrics['RMSE']:.4f} | "
            f"MAE={metrics['MAE']:.4f} | "
            f"Features: {p}/{len(self.X_train[0])}"
        )

    def train_ridge_optimal(self, k: int = 5, lambda_config: str = None):
        """Train Ridge regression với lambda tối ưu qua k-Fold CV.

        lambda_config : str | None
            Đường dẫn file JSON luu lambda_opt đã tính sẵn.
            Nếu chưa có, chạy CV rồi lưu vào file đó.
        """
        # Tìm lambda (từ cache hoặc chạy CV)
        lambda_opt = self._load_or_run_lambda_cv(
            ridge_fit, 'Ridge', k=k, config_path=lambda_config
        )

        # Fit Ridge với lambda tối ưu trên toàn bộ tập train
        result = ridge_fit(self.X_train, self.y_train, lam=lambda_opt)

        # Predict trên test
        y_pred = predict(self.X_test, result)

        # Tính metrics + residuals
        p = len(self.X_train[0])
        metrics = self._compute_test_metrics(y_pred, p)
        residuals = [self.y_test[i] - y_pred[i] for i in range(len(self.y_test))]

        # Lưu kết quả
        self.results['Ridge'] = {
            'result':     result,
            'y_pred':     y_pred,
            'metrics':    metrics,
            'residuals':  residuals,
            'lambda_opt': lambda_opt,
        }

        print(
            f"[Ridge] R2={metrics['R2']:.4f} | "
            f"RMSE={metrics['RMSE']:.4f} | "
            f"MAE={metrics['MAE']:.4f}"
        )

    def train_lasso_optimal(self, k: int = 5, lambda_config: str = None):
        """Train Lasso regression với lambda tối ưu qua k-Fold CV.

        lambda_config : str | None
            Đường dẫn file JSON lưu lambda_opt đã tính sẵn.
            Nếu chưa có, chạy CV rồi lưu vào file đó.
        """
        # Tìm lambda (từ cache hoặc chạy CV)
        # max_iter=100, tol=1e-4: loosened cho CV speed; final fit dùng default.
        lambda_opt = self._load_or_run_lambda_cv(
            lasso_fit, 'Lasso', k=k, config_path=lambda_config,
            max_iter=100, tol=1e-4,
        )

        # Fit Lasso với lambda tối ưu (default max_iter=1000, tol=1e-6)
        result = lasso_fit(self.X_train, self.y_train, lam=lambda_opt)

        # Predict trên test
        y_pred = predict(self.X_test, result)

        # Tính metrics + residuals
        p = len(self.X_train[0])
        metrics = self._compute_test_metrics(y_pred, p)
        residuals = [self.y_test[i] - y_pred[i] for i in range(len(self.y_test))]

        # Lưu kết quả
        self.results['Lasso'] = {
            'result':     result,
            'y_pred':     y_pred,
            'metrics':    metrics,
            'residuals':  residuals,
            'lambda_opt': lambda_opt,
        }

        print(
            f"[Lasso] R2={metrics['R2']:.4f} | "
            f"RMSE={metrics['RMSE']:.4f} | "
            f"MAE={metrics['MAE']:.4f}"
        )

    def train_kernel_ridge_optimal(self, k: int = 5, config_path: str = None):
        """Train Kernel Ridge Regression với lambda và length_scale tối ưu qua k-Fold CV.
        
        Sử dụng subset dữ liệu nếu số lượng dòng quá lớn để tránh treo máy.
        """
        import json
        model_name = 'Kernel Ridge'

        # KRR không cần polynomial features vì RBF kernel tự ánh xạ phi tuyến
        print(f"[{model_name}] Trích xuất đặc trưng thuần (poly=False, Features = {len(self.feature_names)})...")
        X_krr_train, _ = self.pipeline.transform(self.df_train, poly=False)
        X_krr_test,  _ = self.pipeline.transform(self.df_test, poly=False)

        # Tìm siêu tham số
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            best_lam = cfg['best_lam']
            best_ls = cfg['best_length_scale']
            print(f"[{model_name}] Đọc config từ cache: lam={best_lam:.6g}, ls={best_ls:.6g}")
        else:
            X_cv, y_cv = X_krr_train, self.y_train
            if len(X_krr_train) > 800:
                print(f"[{model_name}] Dữ liệu train lớn ({len(X_krr_train)}). Lấy mẫu 800 dòng chạy CV...")
                X_cv, y_cv = sample_rows(X_krr_train, self.y_train, max_rows=800)
                
            lam_grid = [0.01, 0.1, 1.0, 10.0]
            ls_grid = [0.1, 1.0, 5.0, 10.0]
            
            cv_res = KernelRidgeRegression.cross_validate(X_cv, y_cv, lam_grid, ls_grid, k=k)
            best_lam = cv_res['best_lam']
            best_ls = cv_res['best_length_scale']
            
            if config_path:
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                with open(config_path, 'w') as f:
                    json.dump({
                        'best_lam': best_lam, 
                        'best_length_scale': best_ls, 
                        'best_cv_score': cv_res['best_cv_score']
                    }, f, indent=2)

        # Fit mô hình
        X_fit, y_fit = X_krr_train, self.y_train
        result = KernelRidgeRegression.fit(X_fit, y_fit, lam=best_lam, length_scale=best_ls)

        # Dự đoán trên tập test
        y_pred = KernelRidgeRegression.predict(result, X_krr_test)

        # Tính metrics + residuals
        p = len(X_krr_test[0]) if X_krr_test else 0
        metrics = self._compute_test_metrics(y_pred, p=p)
        residuals = [self.y_test[i] - y_pred[i] for i in range(len(self.y_test))]

        # Lưu kết quả
        self.results[model_name] = {
            'result':       result,
            'y_pred':       y_pred,
            'metrics':      metrics,
            'residuals':    residuals,
            'lambda_opt':   best_lam,
            'length_scale': best_ls,
        }

        print(
            f"[{model_name}] R2={metrics['R2']:.4f} | "
            f"RMSE={metrics['RMSE']:.4f} | "
            f"MAE={metrics['MAE']:.4f}"
        )

    def train_bayesian_optimal(self, k: int = 5, config_path: str = None):
        """Train Bayesian Linear Regression với alpha tối ưu qua k-Fold CV.

        BLR sử dụng polynomial features (giống OLS/Ridge/Lasso) vì nó là
        mô hình tuyến tính trong không gian đặc trưng.
        """
        import json
        model_name = 'Bayesian LR'

        # Tìm siêu tham số (alpha = prior precision)
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            best_alpha = cfg['best_alpha']
            print(f"[{model_name}] Đọc config từ cache: alpha={best_alpha:.6g}")
        else:
            alpha_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

            print(f"[{model_name}] Chạy CV {k}-Fold tìm alpha tối ưu trên {len(self.X_train)} mẫu...")
            cv_res = BayesianLinearRegression.cross_validate(
                self.X_train, self.y_train, alpha_grid, k=k
            )
            best_alpha = cv_res['best_alpha']

            if config_path:
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                with open(config_path, 'w') as f:
                    json.dump({
                        'best_alpha': best_alpha,
                        'best_cv_score': cv_res['best_cv_score']
                    }, f, indent=2)

        # Ước lượng sigma2 từ OLS trên tập train
        sigma2 = BayesianLinearRegression.estimate_sigma2(self.X_train, self.y_train)
        print(f"[{model_name}] sigma2={sigma2:.4f}, alpha={best_alpha:.6g}")

        # Fit mô hình
        result = BayesianLinearRegression.fit(
            self.X_train, self.y_train, sigma2=sigma2, alpha=best_alpha
        )

        # Dự đoán trên tập test
        y_pred, y_lower, y_upper = BayesianLinearRegression.predict(
            result, self.X_test, sigma2
        )

        # Tính metrics + residuals
        p = len(self.X_test[0])
        metrics = self._compute_test_metrics(y_pred, p=p)
        residuals = [self.y_test[i] - y_pred[i] for i in range(len(self.y_test))]

        # Lưu kết quả
        self.results[model_name] = {
            'result':       result,
            'y_pred':       y_pred,
            'y_lower':      y_lower,
            'y_upper':      y_upper,
            'metrics':      metrics,
            'residuals':    residuals,
            'alpha_opt':    best_alpha,
            'sigma2':       sigma2,
        }

        print(
            f"[{model_name}] R2={metrics['R2']:.4f} | "
            f"RMSE={metrics['RMSE']:.4f} | "
            f"MAE={metrics['MAE']:.4f}"
        )

    def compare_models(self) -> pd.DataFrame:
        """Tạo bảng so sánh metrics của tất cả mô hình đã train.

        Returns:
            pd.DataFrame -- index=Model, columns=['MAE', 'RMSE', 'R2_test'],
                           sắp xếp theo RMSE tăng dần.
        """
        rows = []
        for name, data in self.results.items():
            metrics = data['metrics']
            rows.append({
                'Model':   name,
                'MAE':     metrics['MAE'],
                'RMSE':    metrics['RMSE'],
                'R2_test': metrics['R2'],
            })
        df = pd.DataFrame(rows).set_index('Model')
        return df.sort_values('RMSE', ascending=True)

    def plot_model_comparison(self, save_path: str = None):
        """Vẽ biểu đồ cột so sánh RMSE và R2 giữa các mô hình."""
        import matplotlib.pyplot as plt
        df = self.compare_models()
        if df.empty:
            print("[Warning] Chưa có mô hình nào được huấn luyện để vẽ biểu đồ.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=100)
        
        # Vẽ biểu đồ cột cho RMSE (sắp xếp ngược để model tốt nhất ở trên cùng)
        df_rmse = df.sort_values('RMSE', ascending=False)
        colors_rmse = ['lightcoral' if x == df['RMSE'].min() else 'skyblue' for x in df_rmse['RMSE']]
        bars_rmse = axes[0].barh(df_rmse.index, df_rmse['RMSE'], color=colors_rmse, edgecolor='black', height=0.6)
        axes[0].set_title('So sánh RMSE giữa các mô hình (Càng nhỏ càng tốt)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('RMSE')
        axes[0].grid(axis='x', linestyle='--', alpha=0.7)
        for bar in bars_rmse:
            width = bar.get_width()
            axes[0].text(width + 0.005, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                         va='center', ha='left', fontsize=10, fontweight='bold')

        # Vẽ biểu đồ cột cho R2
        df_r2 = df.sort_values('R2_test', ascending=True)
        colors_r2 = ['lightgreen' if x == df['R2_test'].max() else 'lightgray' for x in df_r2['R2_test']]
        bars_r2 = axes[1].barh(df_r2.index, df_r2['R2_test'], color=colors_r2, edgecolor='black', height=0.6)
        axes[1].set_title('So sánh R2 giữa các mô hình (Càng lớn càng tốt)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('R2 Score')
        axes[1].grid(axis='x', linestyle='--', alpha=0.7)
        for bar in bars_r2:
            width = bar.get_width()
            axes[1].text(width + 0.005, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                         va='center', ha='left', fontsize=10, fontweight='bold')

        plt.suptitle('Báo cáo So sánh Hiệu năng giữa các Mô hình Hồi quy', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"[Plot] Đã lưu biểu đồ so sánh mô hình tại: {save_path}")
        
        plt.show()

    @staticmethod
    def plot_comparison_from_json(outputs_dir: str = "part2/outputs", save_path: str = None):
        """Đọc kết quả của tất cả các mô hình từ các file JSON trong outputs_dir và vẽ biểu đồ so sánh."""
        import json
        import glob
        import matplotlib.pyplot as plt
        
        json_files = glob.glob(os.path.join(outputs_dir, "*.json"))
        
        model_names_map = {
            "ols_full": "OLS Full",
            "ols_selected": "OLS Selected",
            "ridge": "Ridge",
            "lasso": "Lasso",
            "kernel_ridge": "Kernel Ridge",
            "bayesian_lr": "Bayesian LR"
        }
        
        rows = []
        for file_path in json_files:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            display_name = model_names_map.get(base_name, base_name.replace('_', ' ').title())
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if 'metrics' in data:
                    metrics = data['metrics']
                    rows.append({
                        'Model':   display_name,
                        'MAE':     metrics.get('MAE', None),
                        'RMSE':    metrics.get('RMSE', None),
                        'R2_test': metrics.get('R2', None),
                    })
            except Exception as e:
                print(f"[Warning] Không thể đọc file {file_path}: {e}")
                
        if not rows:
            print(f"[Warning] Không tìm thấy dữ liệu hợp lệ trong thư mục: {outputs_dir}")
            return None
            
        df = pd.DataFrame(rows).set_index('Model')
        df = df.sort_values('RMSE', ascending=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=100)
        
        # RMSE
        df_rmse = df.sort_values('RMSE', ascending=False)
        colors_rmse = ['lightcoral' if x == df['RMSE'].min() else 'skyblue' for x in df_rmse['RMSE']]
        bars_rmse = axes[0].barh(df_rmse.index, df_rmse['RMSE'], color=colors_rmse, edgecolor='black', height=0.6)
        axes[0].set_title('So sánh RMSE giữa các mô hình (Càng nhỏ càng tốt)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('RMSE')
        axes[0].grid(axis='x', linestyle='--', alpha=0.7)
        for bar in bars_rmse:
            width = bar.get_width()
            axes[0].text(width + 0.005, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                         va='center', ha='left', fontsize=10, fontweight='bold')

        # R2
        df_r2 = df.sort_values('R2_test', ascending=True)
        colors_r2 = ['lightgreen' if x == df['R2_test'].max() else 'lightgray' for x in df_r2['R2_test']]
        bars_r2 = axes[1].barh(df_r2.index, df_r2['R2_test'], color=colors_r2, edgecolor='black', height=0.6)
        axes[1].set_title('So sánh R2 giữa các mô hình (Càng lớn càng tốt)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('R2 Score')
        axes[1].grid(axis='x', linestyle='--', alpha=0.7)
        for bar in bars_r2:
            width = bar.get_width()
            axes[1].text(width + 0.005, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                         va='center', ha='left', fontsize=10, fontweight='bold')

        plt.suptitle('Báo cáo So sánh Hiệu năng (Đọc từ file JSON)', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"[Plot] Đã lưu biểu đồ so sánh mô hình tại: {save_path}")
        
        plt.show()
        return df

    @staticmethod
    def shapiro_wilk_test(residuals: list[float]) -> dict:
        """Kiểm tra phần dư có phân phối chuẩn không (Shapiro-Wilk).

        Thực hiện kiểm định Shapiro-Wilk theo thuật toán Royston (1992) AS R94.
        Nếu n > 5000, lấy ngẫu nhiên 5000 mẫu.

        Returns:
            dict: {'statistic', 'p_value', 'is_normal'}
                  is_normal = True nếu p_value > 0.05 (không bác bỏ H0)
        """
        import math
        import random
        from part1.ols_implementation import _standard_normal_ppf_mc, _chi2_sf

        sample = list(residuals)
        if len(sample) > 5000:
            random.seed(RANDOM_STATE)
            sample = random.sample(sample, 5000)

        n = len(sample)
        if n < 3:
            raise ValueError("Shapiro-Wilk yêu cầu ít nhất 3 mẫu")

        x = sorted(sample)
        x_bar = sum(x) / n

        # Sum of squared deviations
        ss = sum((xi - x_bar) ** 2 for xi in x)
        if ss < 1e-300:
            return {'statistic': 1.0, 'p_value': 1.0, 'is_normal': True}

        # Expected values of standard normal order statistics (Blom approximation)
        m = [_standard_normal_ppf_mc((i + 1 - 0.375) / (n + 0.25))
             for i in range(n)]

        # Coefficient computation per Royston (1992)
        m_sq_sum = sum(mi ** 2 for mi in m)
        m_sq_norm = math.sqrt(m_sq_sum)

        n2 = n // 2
        # Polynomial for a_n (last weight)
        c1 = [0.0, 0.221157, 0.147981, -2.071190, 4.434685, -2.706056]
        c2 = [0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633]

        u = 1.0 / math.sqrt(n)

        an = m[-1] / m_sq_norm
        # Correct a_n via polynomial in u
        an += ((((c1[5]*u + c1[4])*u + c1[3])*u + c1[2])*u + c1[1])*u

        a = [0.0] * n
        a[n - 1] = an
        a[0] = -an

        if n > 5:
            an1 = m[-2] / m_sq_norm
            an1 += ((((c2[5]*u + c2[4])*u + c2[3])*u + c2[2])*u + c2[1])*u
            a[n - 2] = an1
            a[1] = -an1

            # Remaining weights from normalisation
            phi_sq = (m_sq_sum - 2.0 * m[-1] ** 2 - 2.0 * m[-2] ** 2) / \
                     (1.0 - 2.0 * an ** 2 - 2.0 * an1 ** 2)
            for i in range(2, n2):
                a[n - 1 - i] = m[n - 1 - i] / math.sqrt(phi_sq)
                a[i] = -a[n - 1 - i]
        elif n == 5:
            an1 = m[-2] / m_sq_norm
            an1 += ((((c2[5]*u + c2[4])*u + c2[3])*u + c2[2])*u + c2[1])*u
            a[n - 2] = an1
            a[1] = -an1
            phi_sq = (m_sq_sum - 2.0 * m[-1] ** 2 - 2.0 * m[-2] ** 2) / \
                     (1.0 - 2.0 * an ** 2 - 2.0 * an1 ** 2)
            a[2] = 0.0
        elif n == 4:
            a[1] = 0.0
            a[2] = 0.0
        # n == 3: only a[0]=-a[2]=an set above, a[1]=0

        # W statistic
        W_num = sum(a[i] * x[i] for i in range(n)) ** 2
        W = W_num / ss
        W = max(0.0, min(1.0, W))

        # Tinh p-value qua xap xi da thuc Royston (1992)
        # Bien doi: y = log(1 - W) xap xi phan phoi chuan voi tham so phu thuoc n
        def _poly(coefs, x_val):
            return sum(c * (x_val ** i) for i, c in enumerate(coefs))

        if n == 3:
            pi = math.acos(-1.0)
            p_value = max(0.0, min(1.0,
                6.0 / pi * (math.asin(math.sqrt(W)) - math.asin(math.sqrt(0.75)))))
        elif n <= 11:
            # He so da thuc trong W cho truong hop n nho (Royston AS R94)
            c3 = [0.544236431, -0.3955066, 0.17898, 0.0, 0.0, 0.0]
            c4 = [1.3822, -5.244, 7.3278, -3.4662, 0.0, 0.0]
            c5 = [0.60461, -1.40, 0.8, 0.0, 0.0, 0.0]
            gamma = _poly(c3[:n - 2], W)
            mu    = _poly(c4[:n - 2], W)
            sigma = math.exp(_poly(c5[:n - 2], W))
            z = (math.log(1.0 - W) - gamma - mu) / sigma
            p_value = 1.0 - _sw_norm_cdf(z)
        else:
            # Xap xi cho n >= 12: log(1-W) ~ N(mu_w, sigma_w)
            # He so da thuc bac 2 trong u = log(n) -- hieu chinh theo thuc nghiem
            # mu_w:    c_mu[0]*u^2 + c_mu[1]*u + c_mu[2]
            # log(sigma_w): c_ls[0]*u^2 + c_ls[1]*u + c_ls[2]
            c_mu = [-0.05277822, -0.34406583, -1.69550237]
            c_ls = [ 0.01733788, -0.23424872, -0.09760698]

            u = math.log(n)
            mu_w    = c_mu[0]*u*u + c_mu[1]*u + c_mu[2]
            log_sig = c_ls[0]*u*u + c_ls[1]*u + c_ls[2]
            sigma_w = math.exp(log_sig)

            y = math.log(1.0 - W)
            z = (y - mu_w) / sigma_w
            # P(Z > z) = 0.5 * erfc(z / sqrt(2))
            p_value = 0.5 * math.erfc(z / math.sqrt(2.0))

        return {
            'statistic': float(W),
            'p_value':   p_value,
            'is_normal': bool(p_value > 0.05),
        }

    @staticmethod
    def breusch_pagan_test(X: list[list[float]], residuals: list[float]) -> dict:
        """Kiểm tra tính đồng nhất phương sai (Breusch-Pagan).

        Quy trình:
            1. e2 = residuals^2
            2. Fit OLS phụ: e2 ~ X
            3. R2 của mô hình phụ
            4. LM = n * R2
            5. p_value = chi2_sf(LM, df=p) 
        Returns:
            dict: {'LM_stat', 'p_value', 'is_homoscedastic'}
                  is_homoscedastic = True nếu p_value > 0.05
        """
        from part1.ols_implementation import _chi2_sf

        n = len(residuals)
        p = len(X[0])

        # 1. Bình phương phần dư
        e_squared = [r ** 2 for r in residuals]

        # 2. Fit OLS phụ: e2 ~ X (ols_fit tự thêm bias)
        aux_result = ols_fit(X, e_squared)

        # 3. Tính R2 của mô hình phụ
        aux_metrics = model_metrics(e_squared, aux_result['y_hat'], p)
        R2_aux = aux_metrics['R2']

        # 4. Thống kê LM
        LM = n * R2_aux

        # 5. p-value từ phân phối chi-squared 
        p_value = float(_chi2_sf(LM, p))

        return {
            'LM_stat':          LM,
            'p_value':          p_value,
            'is_homoscedastic': bool(p_value > 0.05),
        }


    def run_diagnostics(self):
        """Chạy Shapiro-Wilk và Breusch-Pagan cho tất cả mô hình đã train."""
        for model_name, data in self.results.items():
            residuals = data['residuals']

            # Shapiro-Wilk
            sw = self.shapiro_wilk_test(residuals)
            data['shapiro'] = sw

            # Breusch-Pagan
            bp = self.breusch_pagan_test(self.X_test, residuals)
            data['bp_test'] = bp

            print(
                f"[{model_name}] "
                f"Shapiro p={sw['p_value']:.4f} ({'Normal' if sw['is_normal'] else 'Non-normal'}) | "
                f"BP p={bp['p_value']:.4f} ({'Homoscedastic' if bp['is_homoscedastic'] else 'Heteroscedastic'})"
            )

    def run_all(self):
        """Orchestrate toàn bộ pipeline: prepare -> train -> diagnostics -> compare.

        Returns:
            pd.DataFrame -- bảng so sánh 4 mô hình, sắp xếp theo RMSE tăng dần.
        """
        # 1. Chuẩn bị dữ liệu
        self.prepare_data()

        # 2. Train các mô hình
        self.train_ols_full()
        self.train_ols_selected()
        self.train_ridge_optimal()
        self.train_lasso_optimal()
        self.train_kernel_ridge_optimal()

        # 3. Kiểm định thống kê
        self.run_diagnostics()

        # 4. Trả về bảng so sánh
        return self.compare_models()

if __name__ == '__main__':
    import argparse

    VALID_MODELS = ['ols_full', 'ols_selected', 'ridge', 'lasso', 'krr', 'bayesian', 'all']

    parser = argparse.ArgumentParser(
        description='ModelComparator -- chay tung model va luu ket qua JSON'
    )
    parser.add_argument(
        '--model', choices=VALID_MODELS, default='all',
        help='Model can chay (mac dinh: all). Vi du: --model ridge'
    )
    parser.add_argument(
        '--data', default='part2/data/AirQualityUCI.csv',
        help='Duong dan file CSV (mac dinh: part2/data/AirQualityUCI.csv)'
    )
    parser.add_argument(
        '--out-dir', default='part2/outputs',
        help='Thu muc luu ket qua JSON (mac dinh: part2/outputs)'
    )
    parser.add_argument(
        '--config-dir', default='part2/configs',
        help='Thu muc luu/doc config (lambda, feature selection) (mac dinh: part2/configs)'
    )
    parser.add_argument(
        '--diagnostics', action='store_true',
        help='Chay Shapiro-Wilk + Breusch-Pagan sau khi train xong'
    )
    args = parser.parse_args()

    # Đường dẫn config (cache)
    cfg_dir        = args.config_dir
    feature_cfg    = os.path.join(cfg_dir, 'feature_selection.json')
    ridge_lam_cfg  = os.path.join(cfg_dir, 'ridge_lambda.json')
    lasso_lam_cfg  = os.path.join(cfg_dir, 'lasso_lambda.json')
    krr_cfg        = os.path.join(cfg_dir, 'krr_config.json')
    blr_cfg        = os.path.join(cfg_dir, 'blr_config.json')

    # Khởi tạo và load data
    comparator = ModelComparator(args.data)
    comparator.prepare_data()

    target_models = VALID_MODELS[:-1] if args.model == 'all' else [args.model]

    for m in target_models:
        print(f"\n{'='*55}\n  Chay model: {m.upper()}\n{'='*55}")

        if m == 'ols_full':
            comparator.train_ols_full()
            key = 'OLS Full'

        elif m == 'ols_selected':
            comparator.train_ols_selected(config_path=feature_cfg)
            key = 'OLS Selected'

        elif m == 'ridge':
            comparator.train_ridge_optimal(lambda_config=ridge_lam_cfg)
            key = 'Ridge'

        elif m == 'lasso':
            comparator.train_lasso_optimal(lambda_config=lasso_lam_cfg)
            key = 'Lasso'

        elif m == 'krr':
            comparator.train_kernel_ridge_optimal(k=5, config_path=krr_cfg)
            key = 'Kernel Ridge'

        elif m == 'bayesian':
            comparator.train_bayesian_optimal(k=5, config_path=blr_cfg)
            key = 'Bayesian LR'

        # Chạy diagnostics nếu yêu cầu
        if args.diagnostics and key in comparator.results:
            data = comparator.results[key]
            data['shapiro'] = ModelComparator.shapiro_wilk_test(data['residuals'])
            data['bp_test'] = ModelComparator.breusch_pagan_test(
                comparator.X_test, data['residuals']
            )
            sw = data['shapiro']
            bp = data['bp_test']
            print(f"  Shapiro p={sw['p_value']:.4f} | BP p={bp['p_value']:.4f}")

        # Lưu JSON ngay sau khi chạy xong mỗi model
        if key in comparator.results:
            ModelComparator._save_result_json(key, comparator.results[key], args.out_dir)

    # In bảng so sánh nếu có ít nhất 1 model đã chạy
    if comparator.results:
        print("\n" + "=" * 55)
        print("BANG SO SANH (sap xep theo RMSE tang dan)")
        print("=" * 55)
        print(comparator.compare_models().to_string())