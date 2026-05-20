import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import RANDOM_STATE
import pandas as pd

MIN_COMMON_FEATURES = 3

class EDAToolkit:
    @staticmethod
    def get_raw_data_summary(file_path: str, target_col: str = 'CO(GT)') -> dict:
        """Lấy thông tin tổng quan về dữ liệu
        
        Args:
            file_path (str): Đường dẫn đến file CSV
            target_col (str): Tên cột target
        Returns:
            dict: Dictionary chứa thông tin tổng quan về dữ liệu
        """

        # Đọc file CSV
        df = pd.read_csv(file_path, sep=',')
        
        # Xóa cột có toàn bộ giá trị bị thiếu
        df.dropna(how='all', axis=1, inplace=True)
        # Xóa hàng có toàn bộ giá trị bị thiếu
        df.dropna(how='all', axis=0, inplace=True)
        
        # Chuyển đổi sang kiểu số
        for col in df.columns:
            if col not in ['Date', 'Time']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.replace(-200, np.nan, inplace=True)
        
        return {
            'n_samples': df.shape[0],
            'p_features': df.shape[1],
            'target': target_col,
            'total_missing': int(df.isnull().sum().sum()),
            'df': df
        }
    @staticmethod
    def descriptive_statistics_report(df: pd.DataFrame) -> pd.DataFrame:
        """Báo cáo thống kê mô tả

        Args:
            df (pd.DataFrame): DataFrame chứa dữ liệu
        Returns:
            pd.DataFrame: DataFrame chứa báo cáo thống kê mô tả
        """

        # Chọn cột số
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Tính thống kê mô tả
        desc_stats = df[numeric_cols].describe().T
        
        desc_stats = desc_stats.rename(columns={
            'count': 'Count', 'mean': 'Mean', 'std': 'Std Dev',
            'min': 'Min', '25%': 'Q1 (25%)', '50%': 'Median (Q2)',
            '75%': 'Q3 (75%)', 'max': 'Max'
        })
        return desc_stats.round(2)

    @staticmethod
    def missing_report(data: pd.DataFrame) -> pd.DataFrame:
        """Báo cáo giá trị bị thiếu

        Args:
            data (pd.DataFrame): DataFrame chứa dữ liệu
        Returns:
            pd.DataFrame: DataFrame chứa báo cáo giá trị bị thiếu
        """

        # Đếm giá trị thiếu
        missing_count = data.isnull().sum()
        # Tính tỷ lệ giá trị thiếu
        missing_pct = (missing_count / len(data)) * 100
        # Tạo DataFrame báo cáo và sắp xếp theo tỷ lệ thiếu giảm dần

        report = pd.DataFrame({
            'Đặc trưng': missing_count.index,
            'Số lượng thiếu': missing_count.values,
            'Tỷ lệ thiếu (%)': missing_pct.values
        }).sort_values(by='Tỷ lệ thiếu (%)', ascending=False)
        
        report = report.reset_index(drop=True)
        report.index += 1
        return report.round(2)

    @staticmethod
    def check_duplicates(df: pd.DataFrame):
        """Kiểm tra dữ liệu trùng lặp
        Args:
            df (pd.DataFrame): DataFrame chứa dữ liệu
            
        Returns:
            int: Số lượng dòng dữ liệu trùng lặp
        """

        # Kiểm tra dữ liệu trùng lặp
        dup_count = df.duplicated().sum()

        print(f"Số dòng dữ liệu trùng lặp phát hiện: {dup_count}")
        return dup_count

    @staticmethod
    def plot_histograms(df: pd.DataFrame, figsize=(18, 22)):
        """
        Vẽ biểu đồ Histogram phân phối cho các đặc trưng số

        Args:
            df (pd.DataFrame): DataFrame chứa dữ liệu
            figsize (tuple): Kích thước biểu đồ
        """

        # Chọn cột số
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Vẽ histogram
        df[numeric_cols].hist(bins=30, figsize=figsize, color='skyblue', edgecolor='black', layout=(5, 3))
        plt.suptitle("Histogram khảo sát phân phối các đặc trưng", fontsize=20, y=1.02)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_boxplots(df: pd.DataFrame):
        """Vẽ biểu đồ Boxplot phân tán và Outliers cho các đặc trưng số
        Args:
            df (pd.DataFrame): DataFrame chứa dữ liệu
        """

        # Chọn cột số
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Tạo grid plot
        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4), dpi=100)
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            sns.boxplot(data=df, y=col, ax=axes[i], color='lightcoral')
            axes[i].set_title(f'Boxplot: {col}', fontsize=14, fontweight='bold'); axes[i].set_ylabel('')
            
        for j in range(i + 1, len(axes)): axes[j].axis('off')
        plt.suptitle("Biểu đồ Boxplot khảo sát phân tán và Outliers", fontsize=20, y=1.00)
        plt.tight_layout(); plt.show()

    @staticmethod
    def plot_correlation_heatmap(df: pd.DataFrame):
        """Vẽ ma trận hệ số tương quan Pearson giữa các đặc trưng

        Args:
            df (pd.DataFrame): DataFrame chứa dữ liệu
        """

        # Chọn cột số
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Tính ma trận tương quan Pearson
        corr_matrix = df[numeric_cols].corr()
        
        # Vẽ heatmap tương quan
        plt.figure(figsize=(12, 10), dpi=100)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, cbar=True)
        plt.title("Ma trận hệ số tương quan Pearson giữa các đặc trưng", fontsize=14, fontweight='bold')
        plt.tight_layout(); plt.show()

    @staticmethod
    def plot_top_correlations(df: pd.DataFrame, target_col='CO(GT)'):
        """
        Vẽ scatterplot giữa target và 5 đặc trưng có tương quan cao nhất

        Args:
            df (pd.DataFrame): DataFrame chứa dữ liệu
            target_col (str): Tên cột target
        """

        # Chọn cột số
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Tính ma trận tương quan Pearson
        corr_matrix = df[numeric_cols].corr()

        # Lấy 5 đặc trưng có tương quan cao nhất với target
        top_5_features = (
            corr_matrix[target_col]
            .abs()
            .sort_values(ascending=False)
            .index[1:6]
        )

        n_cols = 2
        n_rows = (len(top_5_features) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 15), dpi=120)

        axes = axes.flatten()

        for i, feature in enumerate(top_5_features):
            sns.scatterplot(data=df, x=feature, y=target_col, ax=axes[i], color='teal', alpha=0.4, s=20)
            axes[i].set_title(f'{target_col} vs {feature}\n(r = {corr_matrix[target_col][feature]:.2f})', fontsize=14, fontweight='bold')
            axes[i].set_xlabel(feature, fontsize=11)
            axes[i].set_ylabel(target_col, fontsize=11)
            axes[i].tick_params(axis='both', labelsize=10)

        for j in range(len(top_5_features), len(axes)):
            axes[j].axis('off')

        plt.suptitle(
            f'Top 5 đặc trưng tương quan mạnh với {target_col}',
            fontsize=20,
            fontweight='bold',
            y=1.02
        )

        plt.tight_layout()
        plt.show()

    @staticmethod
    def outlier_report_iqr(df: pd.DataFrame) -> pd.DataFrame:
        """Báo cáo giá trị ngoại lai sử dụng phương pháp IQR

        Args:
            df (pd.DataFrame): DataFrame chứa dữ liệu
        Returns:
            pd.DataFrame: DataFrame chứa báo cáo giá trị ngoại lai
        """

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        iqr_outliers = []
        for col in numeric_cols:
            # Loại bỏ giá trị thiếu (NaN) vì quantile không nên tính trên missing values
            clean_series = df[col].dropna()
            if len(clean_series) == 0: continue

            # Tính Q1, Q3 và IQR
            Q1, Q3 = clean_series.quantile(0.25), clean_series.quantile(0.75)
            IQR = Q3 - Q1
            # Tìm index của các outlier
            idx = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)].index
            # Lưu thống kê outlier của đặc trưng hiện tại
            iqr_outliers.append({'Đặc trưng': col, 'Số lượng Outliers': len(idx), 'Tỷ lệ (%)': (len(idx) / len(df)) * 100})

        # Tạo DataFrame báo cáo và sắp xếp theo tỷ lệ outlier giảm dần
        df_iqr = pd.DataFrame(iqr_outliers).sort_values(by='Tỷ lệ (%)', ascending=False)

        report = df_iqr[df_iqr['Số lượng Outliers'] > 0].round(2).reset_index(drop=True)
        report.index += 1
        
        return report

    @staticmethod
    def outlier_report_zscore(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """Báo cáo giá trị ngoại lai sử dụng phương pháp Z-score
        Args:
            df (pd.DataFrame): DataFrame chứa dữ liệu
            threshold (float): Ngưỡng Z-score để xác định giá trị ngoại lai
        Returns:
            pd.DataFrame: DataFrame chứa báo cáo giá trị ngoại lai
        """

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        z_outliers = []
        for col in numeric_cols:
            # Tính trung bình và độ lệch chuẩn của cột
            mean_val, std_val = df[col].mean(), df[col].std()

            # Tránh chia cho 0 nếu cột có giá trị không thay đổi
            if std_val > 0:
                # Tính Z-score: Z = (x - mean) / std
                z_scores = np.abs((df[col] - mean_val) / std_val)  # Tính Z-score: Z = (x - mean) / std
                idx = df[z_scores > threshold].index   # Lấy index của các dòng có |Z-score| vượt ngưỡng

                # Lưu thống kê outlier của đặc trưng hiện tại
                z_outliers.append({'Đặc trưng': col, 'Số lượng Outliers': len(idx), 'Tỷ lệ (%)': (len(idx) / len(df)) * 100})
        
        # Tạo DataFrame báo cáo và sắp xếp theo tỷ lệ outlier giảm dần
        df_z = pd.DataFrame(z_outliers).sort_values(by='Tỷ lệ (%)', ascending=False)

        report = df_z[df_z['Số lượng Outliers'] > 0].round(2).reset_index(drop=True)
        report.index += 1
        
        return report

    @staticmethod
    def plot_feature_importance(model, feature_names, model_name="OLS"):
        """Vẽ biểu đồ feature importance"""

        if isinstance(model, dict) and "beta_hat" in model:
            coeffs = np.array(model["beta_hat"])
        elif hasattr(model, 'coef_'):
            coeffs = model.coef_.flatten()
        else:
            raise ValueError(f"Không thể trích xuất trọng số từ model: {model_name}. Hãy đảm bảo truyền đúng kết quả trả về của hàm fit.")

        # Xử lý hệ số chặn (Intercept)
        # Các hàm Part 1 trả về [intercept, beta_1, ..., beta_p] nên mảng dài hơn feature_names 1 đơn vị
        if len(coeffs) == len(feature_names) + 1:
            coeffs = coeffs[1:]  # Loại bỏ phần tử đầu tiên (intercept)
        elif len(coeffs) != len(feature_names):
            raise ValueError(f"Số lượng trọng số ({len(coeffs)}) không khớp với số đặc trưng ({len(feature_names)}).")

        # Vẽ biểu đồ
        importance_df = pd.DataFrame({
            'Feature': feature_names, 
            'Weight': coeffs, 
            'Abs_Weight': np.abs(coeffs)
        }).sort_values(by='Abs_Weight', ascending=False).head(15)
        
        plt.figure(figsize=(10, 8), dpi=150)
        sns.barplot(x='Weight', y='Feature', data=importance_df, hue='Feature', palette='vlag', legend=False)
        plt.title(f"Top 15 Standardized Regression Coefficients - {model_name}", fontsize=14, fontweight='bold')
        plt.xlabel("Hệ số hồi quy chuẩn hóa (Standardized Coefficients)")
        plt.ylabel("Các đặc trưng")
        plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        plt.tight_layout()
        plt.show()


class DataPipeline:
    """Pipeline tiền xử lý dữ liệu AirQualityUCI."""

    def __init__(self, target_col="CO(GT)", test_size=0.2, random_state=RANDOM_STATE):
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.means_ = {}
        self.stds_ = {}
        self.medians_ = {}
        self.models_ = {}

    def _knn_impute(self, X: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
        """Thực hiện KNN Imputation dùng hoàn toàn bằng Pandas."""
        if 'knn_train_data' not in self.models_:
            return X
            
        X_train_scaled = self.models_['knn_train_data']
        X_train_raw = self.models_['knn_train_raw']
        knn_means = self.models_['knn_means']
        knn_stds = self.models_['knn_stds']
        
        X_out = X.copy()
        
        X_input_scaled = (X - knn_means) / knn_stds
        
        missing_rows = X_out[X_out.isna().any(axis=1)]
        
        for idx, row_scaled in X_input_scaled.loc[missing_rows.index].iterrows():
            valid_features = row_scaled.dropna().index
            if len(valid_features) < MIN_COMMON_FEATURES:
                for col in X_out.loc[idx][X_out.loc[idx].isna()].index:
                    if col in self.medians_:
                        X_out.loc[idx, col] = self.medians_[col]
                continue
                
            diff = X_train_scaled[valid_features] - row_scaled[valid_features]
            sq_dist = (diff ** 2).sum(axis=1)
            
            valid_counts = diff.notna().sum(axis=1)
            valid_counts = valid_counts.replace(0, 1)
            weighted_dist = sq_dist / valid_counts
            
            row_raw = X_out.loc[idx]
            for col in row_raw[row_raw.isna()].index:
                if col not in X_train_raw.columns:
                    continue
                train_valid_for_col = X_train_raw[X_train_raw[col].notna()].index
                if len(train_valid_for_col) == 0:
                    continue
                    
                dist_for_col = weighted_dist.loc[train_valid_for_col]
                if idx in dist_for_col.index:
                    dist_for_col = dist_for_col.drop(idx)
                    
                k_nearest = dist_for_col.nsmallest(n_neighbors).index
                if len(k_nearest) > 0:
                    X_out.loc[idx, col] = X_train_raw.loc[k_nearest, col].mean()
        
        for col in X_out.columns:
            if X_out[col].isna().any() and col in self.medians_:
                X_out[col] = X_out[col].fillna(self.medians_[col])
                    
        return X_out

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Đọc CSV, xử lý cột rỗng, parse Date/Time."""
        # Đọc file CSV
        df = pd.read_csv(filepath, sep=',')
        
        # Xóa hai cột rỗng ở cuối file do dính dấu , cuối dòng
        df.drop(['Unnamed: 15', 'Unnamed: 16'], axis=1, inplace=True, errors='ignore')
        
        # Loại bỏ các dòng hoàn toàn trống ở cuối file
        df.dropna(how='all', inplace=True)
        
        # Định dạng lại cột Date_Time
        if 'Date' in df.columns and 'Time' in df.columns:
            df = df.dropna(subset=['Date', 'Time'])
            date_time_str = df['Date'].astype(str) + ' ' + df['Time'].astype(str)
            df['Date_Time'] = pd.to_datetime(date_time_str, format='%d/%m/%Y %H.%M.%S', errors='coerce')
            df.drop(['Date', 'Time'], axis=1, inplace=True)
        
        # Chuẩn hóa tên cột
        df.columns = df.columns.str.strip()
        
        return df

    def eda(self, df: pd.DataFrame) -> dict:
        """EDA: thống kê mô tả, missing %, correlation."""
        eda_results = {}
        
        # Thống kê mô tả
        eda_results['describe'] = df.describe()
        
        # Phân tích missing values
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df)) * 100
        eda_results['missing_pct'] = missing_pct

        # Ma trận tương quan (Correlation Matrix)
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        eda_results['correlation'] = numeric_df.corr()

        return eda_results

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series = None):
        """Tính toán thống kê từ train set (mean, std, median...)."""
        
        if y_train is not None:
            valid_mask = y_train.notna()
            X_train = X_train.loc[valid_mask].copy()
            y_train = y_train.loc[valid_mask].copy()
            
        # Tách X và y nếu X_train truyền vào thực chất là df_train (chứa cột target)
        if self.target_col in X_train.columns:
            X_tmp = X_train.drop(columns=[self.target_col])
        else:
            X_tmp = X_train.copy()
            
        # Loại bỏ cột datetime (không phải feature cho regression)
        datetime_cols = X_tmp.select_dtypes(include=['datetime64']).columns.tolist()

        # Xác định và loại bỏ các cột có tỷ lệ missing > 70%
        missing_pct = X_tmp.isnull().mean()
        high_missing_cols = missing_pct[missing_pct > 0.7].index.tolist()

        cols_to_drop = datetime_cols + high_missing_cols
        self.models_['drop_cols'] = cols_to_drop
        X_tmp = X_tmp.drop(columns=cols_to_drop)
            
        numeric_cols = X_tmp.select_dtypes(include=['float64', 'int64']).columns
        cat_cols = X_tmp.select_dtypes(include=['object', 'category']).columns

        # Tính thống kê imputation từ X_train
        for col in numeric_cols:
            self.medians_[col] = X_tmp[col].median()
            self.means_[col] = X_tmp[col].mean()
            
        if len(numeric_cols) > 0:
            self.models_['knn_means'] = X_tmp[numeric_cols].mean()
            knn_stds = X_tmp[numeric_cols].std()
            knn_stds = knn_stds.replace(0, 1)
            self.models_['knn_stds'] = knn_stds
            
            X_scaled = (X_tmp[numeric_cols] - self.models_['knn_means']) / self.models_['knn_stds']
            self.models_['knn_train_data'] = X_scaled.copy()
            
            self.models_['knn_train_raw'] = X_tmp[numeric_cols].copy()
            
            X_tmp[numeric_cols] = self._knn_impute(X_tmp[numeric_cols], n_neighbors=5)
            
        # Tính ngưỡng Winsorize từ train
        self.models_['winsorize'] = {}
        for col in numeric_cols:
            q01 = X_tmp[col].quantile(0.01)
            q99 = X_tmp[col].quantile(0.99)
            self.models_['winsorize'][col] = (q01, q99)
            X_tmp[col] = X_tmp[col].clip(lower=q01, upper=q99)
            
        # One-hot encoding cho categorical cols
        if len(cat_cols) > 0:
            X_tmp = pd.get_dummies(X_tmp, columns=cat_cols, drop_first=False, dtype=float)
        self.models_['encoded_cols'] = X_tmp.columns.tolist()
        
        # Tính mean_X và std_X trên train (sau khi impute + encode)
        self.models_['scale_means'] = {}
        self.models_['scale_stds'] = {}
        for col in X_tmp.columns:
            self.models_['scale_means'][col] = X_tmp[col].mean()
            std_val = X_tmp[col].std(ddof=1)
            self.models_['scale_stds'][col] = std_val if pd.notnull(std_val) and std_val != 0 else 1.0
            
        # Lưu degree cho PolynomialFeatures
        self.models_['poly_degree'] = 2
            
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> tuple:
        """Áp dụng pipeline đã fit lên data mới.
        
        Returns:
            tuple: (X_processed: list[list[float]], y_list: list[float] hoặc None)
        """
        if y is not None:
            valid_mask = y.notna()
            X_clean = X.loc[valid_mask].copy()
            y_clean = y.loc[valid_mask].copy()
        else:
            X_clean = X.copy()
            y_clean = None

        # Tách target nếu còn trong X
        if self.target_col in X_clean.columns:
            if y_clean is None:
                y_clean = X_clean[self.target_col].copy()
            X_out = X_clean.drop(columns=[self.target_col]).copy()
        else:
            X_out = X_clean.copy()

        # Bước 1: Drop các cột đã đánh dấu từ fit
        if 'drop_cols' in self.models_:
            X_out = X_out.drop(columns=self.models_['drop_cols'], errors='ignore')

        # Bước 2: KNN Impute (dùng thống kê từ train)
        numeric_cols = X_out.select_dtypes(include=['float64', 'int64']).columns
        cat_cols = X_out.select_dtypes(include=['object', 'category']).columns

        if len(numeric_cols) > 0 and 'knn_train_data' in self.models_:
            X_out[numeric_cols] = self._knn_impute(X_out[numeric_cols], n_neighbors=5)

        # Bước 3: Winsorize dùng ngưỡng từ fit
        if 'winsorize' in self.models_:
            for col, (lower, upper) in self.models_['winsorize'].items():
                if col in X_out.columns:
                    X_out[col] = X_out[col].clip(lower=lower, upper=upper)

        # Bước 4: One-hot encode + align với train columns
        if len(cat_cols) > 0:
            X_out = pd.get_dummies(X_out, columns=cat_cols, drop_first=False, dtype=float)

        if 'encoded_cols' in self.models_:
            X_out = X_out.reindex(columns=self.models_['encoded_cols'], fill_value=0.0)

        # Bước 5: Standardize = (X - mean) / std
        if 'scale_means' in self.models_ and 'scale_stds' in self.models_:
            for col in X_out.columns:
                mean_val = self.models_['scale_means'].get(col, 0.0)
                std_val = self.models_['scale_stds'].get(col, 1.0)
                X_out[col] = (X_out[col] - mean_val) / std_val

        # Bước 6: Check NaN
        remaining_nan = X_out.isna().sum().sum()
        if remaining_nan > 0:
            print(f"WARNING: {remaining_nan} NaN values remain after pipeline!")
            X_out = X_out.fillna(0.0)

        # Chuyển sang list[list[float]]
        X_result = X_out.values.tolist()

        # Bước 7: Polynomial features (nếu degree > 1)
        poly_degree = self.models_.get('poly_degree', 1)
        if poly_degree > 1:
            X_result = self.add_polynomial_features(X_result, degree=poly_degree)

        # Xử lý y
        y_result = None
        if y_clean is not None:
            y_result = y_clean.tolist()

        return X_result, y_result

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series = None) -> tuple:
        """Fit rồi transform."""
        self.fit(X_train, y_train)
        return self.transform(X_train, y_train)

    def add_polynomial_features(self, X: list[list[float]], degree: int = 2) -> list[list[float]]:
        """Thêm polynomial features (tương tác giữa các biến).
        
        Với degree=2: thêm x_i^2 và x_i*x_j cho mọi cặp (i, j).
        """
        result = []
        for row in X:
            new_row = list(row)
            n = len(row)
            for i in range(n):
                # x_i^2
                new_row.append(row[i] ** 2)
                # x_i * x_j (interaction terms)
                if degree >= 2:
                    for j in range(i + 1, n):
                        new_row.append(row[i] * row[j])
            result.append(new_row)
        return result

if __name__ == "__main__":
    """Ví dụ sử dụng DataPipeline."""
    pipeline = DataPipeline()
    df = pipeline.load_data("part2/data/AirQualityUCI.csv")
    eda_results = pipeline.eda(df)
