import math
import pandas as pd
from scipy import stats # Chỉ dùng để tra bảng phân phối t
from ..utils import transpose, matmul, inverse

def coef_inference(X, y, beta_hat, sigma2):
    """
    F4: Tính toán các chỉ số thống kê cho các hệ số.
    Quy tắc: X chưa có bias, hàm tự thêm bên trong.
    """
    n = len(X)
    p = len(X[0]) # số lượng features
    
    # 1. Thêm cột bias (cột 1 đầu tiên)
    X_bias = [[1.0] + row for row in X]
    
    # 2. Tính Ma trận hiệp phương sai của beta: Cov = sigma2 * (X^T @ X)^-1
    # Dùng hoàn toàn hàm từ utils để tính toán đại số
    XT = transpose(X_bias)
    XTX = matmul(XT, X_bias)
    XTX_inv = inverse(XTX)
    
    # 3. Tính Standard Errors (Căn bậc hai các phần tử trên đường chéo chính)
    std_errs = []
    for i in range(len(XTX_inv)):
        # Công thức: sqrt(sigma^2 * C_jj)
        se = math.sqrt(sigma2 * XTX_inv[i][i])
        std_errs.append(se)
        
    # 4. Tính t-statistics: t = beta / std_err
    t_stats = [b / se if se != 0 else 0 for b, se in zip(beta_hat, std_errs)]
    
    # 5. Tra bảng thống kê (Đây là nơi duy nhất dùng Scipy)
    # Bậc tự do: dof = n - (p + 1) vì có p features + 1 intercept
    dof = n - p - 1
    
    # p-value = 2 * (1 - CDF(|t|))
    p_values = [2 * stats.t.sf(abs(t), dof) for t in t_stats]
    
    # t_critical cho khoảng tin cậy 95% (alpha = 0.05)
    t_crit = stats.t.ppf(0.975, dof)
    
    ci_lower = [b - t_crit * se for b, se in zip(beta_hat, std_errs)]
    ci_upper = [b + t_crit * se for b, se in zip(beta_hat, std_errs)]
    
    # 6. Tạo DataFrame kết quả theo chuẩn
    data = {
        'coef': beta_hat,
        'std_err': std_errs,
        't_stat': t_stats,
        'p_value': p_values,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }
    
    index = ['intercept'] + [f'x{i+1}' for i in range(p)]
    return pd.DataFrame(data, index=index)

def vif(X):
    """
    F5: Tính Variance Inflation Factor.
    Input: X (n, p) chưa có bias.
    """
    n = len(X)
    p = len(X[0])
    vif_dict = {}
    
    for j in range(p):
        # 1. Tách biến thứ j làm biến mục tiêu (y_j)
        y_j = [row[j] for row in X]
        
        # 2. Các biến còn lại đóng vai trò là features (X_others)
        X_others = [[row[i] for i in range(p) if i != j] for row in X]
        
        # 3. Hồi quy y_j theo X_others 
        # Sử dụng hàm ols_fit bạn đã viết (nó sẽ tự thêm bias cho X_others)
        from .ols_implementation import ols_fit, model_metrics
        
        result_j = ols_fit(X_others, y_j)
        
        # 4. Tính R-squared của mô hình phụ này
        # model_metrics(y_true, y_pred, p_features)
        metrics = model_metrics(y_j, result_j['y_hat'], p - 1)
        r2_j = metrics['R2']
        
        # 5. Tính VIF = 1 / (1 - R^2)
        # Xử lý trường hợp R2 xấp xỉ 1 để tránh chia cho 0
        if r2_j >= 1.0 - 1e-10:
            vif_val = float('inf')
        else:
            vif_val = 1 / (1 - r2_j)
            
        vif_dict[f'x{j+1}'] = vif_val
        
    return vif_dict