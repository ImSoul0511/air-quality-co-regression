import random
from ..utils import transpose, matmul, inverse, matvec, add_bias_column, vector_add
from .ols_implementation import ols_fit

def run_gauss_markov_simulation(n_sim=1000, n_obs=100, true_beta=[2.0, -1.5, 0.8], true_sigma=1.0):
    """
    F10: Mô phỏng Monte Carlo chứng minh OLS là BLUE.
    """
    random.seed(42)
    # 1. Sinh X cố định
    X_fixed = [[random.gauss(0, 1) for _ in range(2)] for _ in range(n_obs)]
    X_bias = add_bias_column(X_fixed)
    
    # Tính toán trước ma trận cho Estimator thay thế để tăng tốc
    XT = transpose(X_bias)
    XTX = matmul(XT, X_bias)
    for i in range(len(XTX)): XTX[i][i] += 0.5 # Ridge alpha=0.5
    XTX_inv_alt = inverse(XTX)
    H_alt = matmul(XTX_inv_alt, XT) # Ma trận trọng số của Alt estimator

    beta_ols_list = []
    beta_alt_list = []
    
    # Tính phần cố định của y: X * beta
    y_pure = matvec(X_bias, true_beta)

    for sim in range(n_sim):
        # 2. Sinh nhiễu
        epsilon = [random.gauss(0, true_sigma) for _ in range(n_obs)]
        
        # 3. y = X_beta + epsilon
        y_sim = vector_add(y_pure, epsilon)
        
        # 4. Ước lượng OLS
        res_ols = ols_fit(X_fixed, y_sim)
        beta_ols_list.append(res_ols['beta_hat'])
        
        # 5. Ước lượng thay thế
        beta_alt = matvec(H_alt, y_sim)
        beta_alt_list.append(beta_alt)

    return beta_ols_list, beta_alt_list, true_beta