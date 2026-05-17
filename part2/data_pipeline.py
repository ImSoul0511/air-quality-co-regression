import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import RANDOM_STATE
import pandas as pd

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

    def eda(self, df: pd.DataFrame, show_plots: bool = True) -> dict:
        """EDA: thống kê mô tả, missing %, correlation, outliers."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os

        # Đảm bảo thư mục tồn tại để lưu ảnh
        os.makedirs("part2/data/plots", exist_ok=True)

        eda_results = {}
        figures = {}
        
        # Thống kê mô tả
        eda_results['describe'] = df.describe()
        
        # Phân tích missing values
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df)) * 100
        eda_results['missing_pct'] = missing_pct

        fig_missing = plt.figure(figsize=(12, 6))
        missing_pct[missing_pct > 0].sort_values(ascending=False).plot(kind='bar', color='salmon')
        plt.title('Tỉ lệ dữ liệu bị thiếu (Missing Values Percentage)')
        plt.ylabel('Phần trăm (%)')
        plt.xlabel('Tên biến')
        plt.tight_layout()
        plt.savefig("part2/data/plots/missing_values.png")
        if show_plots:
            plt.show()
        else:
            plt.close(fig_missing)
        figures['missing_values'] = fig_missing

        # Ma trận tương quan (Correlation Matrix)
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        eda_results['correlation'] = numeric_df.corr()

        fig_corr = plt.figure(figsize=(12, 10))
        sns.heatmap(eda_results['correlation'], annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
        plt.title('Ma trận tương quan giữa các biến')
        plt.tight_layout()
        plt.savefig("part2/data/plots/correlation_matrix.png")
        if show_plots:
            plt.show()
        else:
            plt.close(fig_corr)
        figures['correlation_matrix'] = fig_corr

        # Vẽ histogram cho các biến số
        fig_hist = plt.figure(figsize=(15, 12))
        ax = fig_hist.gca()
        numeric_df.hist(bins=30, ax=ax, color='skyblue', edgecolor='black')
        plt.suptitle('Phân phối của các biến số', y=1.02)
        plt.tight_layout()
        plt.savefig("part2/data/plots/histograms.png")
        if show_plots:
            plt.show()
        else:
            plt.close(fig_hist)
        figures['histograms'] = fig_hist

        eda_results['figures'] = figures
        return eda_results

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Tính toán thống kê từ train set (mean, std, median...)."""
        pass

    def transform(self, X: pd.DataFrame) -> list[list[float]]:
        """Áp dụng pipeline đã fit lên data mới."""
        pass

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> list[list[float]]:
        """Fit rồi transform."""
        self.fit(X_train, y_train)
        return self.transform(X_train)
