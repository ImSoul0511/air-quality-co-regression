import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from part2.model_comparison import ModelComparator

def main():
    # Xác định thư mục gốc của project (nằm ngoài thư mục part2)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    # Xây dựng các đường dẫn tuyệt đối
    data_path = os.path.join(project_root, "part2", "data", "AirQualityUCI.csv")
    feature_cfg = os.path.join(project_root, "part2", "configs", "feature_selection.json")
    ridge_cfg = os.path.join(project_root, "part2", "configs", "ridge_lambda.json")
    lasso_cfg = os.path.join(project_root, "part2", "configs", "lasso_lambda.json")
    krr_cfg = os.path.join(project_root, "part2", "configs", "krr_config.json")
    blr_cfg = os.path.join(project_root, "part2", "configs", "blr_config.json")

    # 1. Initialize and prepare data
    comparator = ModelComparator(data_filepath=data_path)
    comparator.prepare_data()

    # 2. Train/load all models using cache
    print("Training/Loading all models...")
    comparator.train_ols_full()
    comparator.train_ols_selected(config_path=feature_cfg)
    comparator.train_ridge_optimal(lambda_config=ridge_cfg)
    comparator.train_lasso_optimal(lambda_config=lasso_cfg)
    comparator.train_kernel_ridge_optimal(k=5, config_path=krr_cfg)
    comparator.train_bayesian_optimal(k=5, config_path=blr_cfg)

    # 3. Generate comparison bar chart
    print("Generating comparison bar chart...")
    comparator.plot_model_comparison()

    # 4. Generate Actual vs Predicted scatter plots (3x2 grid)
    print("Generating Actual vs Predicted scatter plots...")
    models_to_plot = [
        ('OLS Full', 'royalblue'),
        ('OLS Selected', 'darkorange'),
        ('Ridge', 'forestgreen'),
        ('Lasso', 'crimson'),
        ('Kernel Ridge', 'darkorchid'),
        ('Bayesian LR', 'darkgoldenrod')
    ]

    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True, sharey=True, dpi=100)
    axes = axes.ravel()

    y_true = comparator.y_test

    for idx, (model_name, color) in enumerate(models_to_plot):
        ax = axes[idx]
        if model_name in comparator.results:
            y_pred = comparator.results[model_name]['y_pred']
            
            # Scatter plot: y_true vs y_pred
            ax.scatter(y_true, y_pred, color=color, alpha=0.3, s=10, edgecolors='none', label='Mẫu dự đoán')
            
            # Identity line y = x (Perfect prediction)
            min_val = min(min(y_true), min(y_pred))
            max_val = max(max(y_true), max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, label='Khớp hoàn hảo (y = x)')
            
            # Calculate R2 and RMSE for the title
            r2 = comparator.results[model_name]['metrics']['R2']
            rmse = comparator.results[model_name]['metrics']['RMSE']
            
            ax.set_title(f"{model_name} (R² = {r2:.4f}, RMSE = {rmse:.4f})", fontsize=12, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(loc='upper left', fontsize=9)
            
            if idx >= 4:
                ax.set_xlabel("Giá trị thực tế (Actual CO)", fontsize=10)
            if idx % 2 == 0:
                ax.set_ylabel("Giá trị dự đoán (Predicted CO)", fontsize=10)
        else:
            ax.text(0.5, 0.5, f"Mô hình {model_name}\nchưa được huấn luyện", 
                    ha='center', va='center', fontsize=12, color='gray')

    plt.suptitle("Đồ thị So sánh Thực tế vs Dự đoán (Actual vs Predicted Scatter Plot)", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    plt.show()

if __name__ == "__main__":
    main()
