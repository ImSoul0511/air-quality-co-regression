import sys
import os

# Thêm thư mục gốc vào sys.path để có thể import từ các module khác
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import transpose, matmul, matvec, identity_matrix
from config import zero_rectify, is_zero

def mean_columns(X: list[list[float]]) -> list[float]:
    """
    Calculate column-wise mean of a data matrix.
    Used for feature standardization before Ridge.

        μ_j = (1/n) * sum X[i][j]

    Args:
        X: data matrix (n x p)

    Returns:
        list: mean of each column [μ_0, ..., μ_{p-1}]
    """
    n = len(X)
    p = len(X[0])

    means = []
    for j in range(p):
        s = 0.0
        for i in range(n):
            s += X[i][j]
        means.append(s / n)

    return means


def std_columns(X: list[list[float]], means: list[float]) -> list[float]:
    """
    Calculate column-wise standard deviation of a data matrix.
    Used to normalize features before Ridge.

        sigma_j = sqrt( (1/n) * sigma (X[i][j] - μ_j)^2 )

    If std is near zero, it is replaced by 1.0 to avoid division by zero.

    Args:
        X: data matrix (n x p)
        means: pre-computed column means

    Returns:
        list: standard deviation of each column [sigma_0, ..., sigma_{p-1}]
    """
    n = len(X)
    p = len(X[0])

    stds = []
    for j in range(p):
        s = 0.0
        for i in range(n):
            diff = X[i][j] - means[j]
            s += diff * diff

        std = (s / n) ** 0.5

        if is_zero(std):
            std = 1.0

        stds.append(std)

    return stds


def standardize(X: list[list[float]], means: list[float], stds: list[float]) -> list[list[float]]:
    """
    Standardize each feature using z-score normalization:

        X_scaled[i][j] = (X[i][j] - μ_j) / sigma_j

    Ridge regression is sensitive to feature scale,
    so normalization is required before fitting.

    Args:
        X: data matrix (n x p)
        means: column means
        stds: column standard deviations

    Returns:
        list: standardized matrix (n x p)
    """
    n = len(X)
    p = len(X[0])

    X_scaled = []
    for i in range(n):
        row = []
        for j in range(p):
            row.append((X[i][j] - means[j]) / stds[j])
        X_scaled.append(row)

    return X_scaled


def add_bias_column(X: list[list[float]]) -> list[list[float]]:
    """
    Prepend a column of ones to the design matrix
    for learning the intercept (bias) term:

        X_bias = [1 | X]

    Args:
        X: data matrix (n x p)

    Returns:
        list: augmented matrix (n x (p+1))
    """
    X_bias = []
    for row in X:
        X_bias.append([1.0] + row[:])
    return X_bias


def solve_system(A: list[list[float]], b: list[float]) -> list[float]:
    """
    Solve a linear system Ax = b using Gauss-Jordan elimination
    with partial pivoting for numerical stability.

    No external libraries used.

    Args:
        A: coefficient matrix (n x n)
        b: right-hand side vector (n,)

    Returns:
        list: solution vector x (n,)

    Raises:
        ValueError: if the system has no unique solution
    """
    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]

    for col in range(n):
        pivot = col

        for r in range(col + 1, n):
            if abs(M[r][col]) > abs(M[pivot][col]):
                pivot = r

        if is_zero(abs(M[pivot][col])):
            raise ValueError("Hệ phương trình không có nghiệm duy nhất")

        M[col], M[pivot] = M[pivot], M[col]

        pivot_val = M[col][col]
        for j in range(col, n + 1):
            M[col][j] /= pivot_val

        for r in range(n):
            if r != col:
                factor = M[r][col]
                for j in range(col, n + 1):
                    M[r][j] -= factor * M[col][j]

    return [zero_rectify(M[i][n]) for i in range(n)]


def ridge_fit(X: list[list[float]], y: list[float], lam: float) -> dict:
    """
    Fit Ridge Regression using closed-form solution
    with L2 regularization to reduce overfitting and multicollinearity.

    Objective:

        min  ||y - Xβ||² + λ||β||²

    Closed-form solution:

        β = (X^T X + λI)^(-1) X^T y

    Features are standardized (z-score) before fitting.
    The intercept term is not regularized (I[0][0] = 0).

    Args:
        X: feature matrix (n x p), raw (un-standardized)
        y: target vector (n,)
        lam: regularization parameter λ (must be >= 0)

    Returns:
        dict with keys:
            - beta_hat: estimated coefficients (p+1,) including intercept
            - y_hat: predicted values (n,)
            - mean_X: column means used for standardization
            - std_X: column stds used for standardization
    """
    if lam < 0:
        raise ValueError("lam phải >= 0")

    if len(X) == 0:
        raise ValueError("X không được rỗng")

    if len(X) != len(y):
        raise ValueError("Số dòng X phải bằng độ dài y")

    p = len(X[0])

    for row in X:
        if len(row) != p:
            raise ValueError("Các dòng của X phải cùng số cột")

    mean_X = mean_columns(X)
    std_X = std_columns(X, mean_X)
    X_scaled = standardize(X, mean_X, std_X)

    X_bias = add_bias_column(X_scaled)

    n = len(X_bias)
    p1 = len(X_bias[0])

    I = identity_matrix(p1)
    I[0][0] = 0.0

    Xt = transpose(X_bias)
    XtX = matmul(Xt, X_bias)
    b = matvec(Xt, y)

    A = []
    for i in range(p1):
        row = []
        for j in range(p1):
            row.append(XtX[i][j] + lam * I[i][j])
        A.append(row)

    beta_hat = solve_system(A, b)
    y_hat = matvec(X_bias, beta_hat)

    return {
        "beta_hat": beta_hat,
        "y_hat": y_hat,
        "mean_X": mean_X,
        "std_X": std_X,
    }

def ridge_trace(
    X: list[list[float]],
    y: list[float],
    lambdas: list[float],
):
    """
    Plot Ridge trace:

        lambda (log scale) vs beta_hat[j]

    Observe how coefficients shrink toward zero
    as lambda increases.

    Args:
        X:
            feature matrix
        y:
            target vector
        lambdas:
            list of lambda values

    Returns:
        dict:
            {
                "lambdas": lambdas,
                "coefficients": coefficients
            }

    Notes:
        - Reuses ridge_fit()
        - Uses matplotlib only for visualization
        - X-axis is logarithmic scale
    """

    # pyrefly: ignore [missing-import]
    import matplotlib.pyplot as plt

    if len(lambdas) == 0:
        raise ValueError("lambdas không được rỗng")

    coefficients = []

    for lam in lambdas:

        result = ridge_fit(X, y, lam)

        beta_hat = result["beta_hat"]

        coefficients.append(beta_hat)

    p1 = len(coefficients[0])

    fig, ax = plt.subplots(figsize=(8, 5))

    for j in range(p1):

        beta_path = []

        for i in range(len(lambdas)):
            beta_path.append(coefficients[i][j])

        if j == 0:
            label = "intercept"
        else:
            label = f"beta_{j}"

        ax.plot(
            lambdas,
            beta_path,
            label=label,
        )

    ax.set_xscale("log")

    ax.set_title("Ridge Trace — Hệ số theo λ", fontsize=13)
    ax.set_xlabel("λ (log scale)")
    ax.set_ylabel("Giá trị hệ số (β)")

    ax.axhline(
        y=0,
        linestyle="--",
        linewidth=1,
        color="gray",
    )

    ax.legend()
    ax.grid(True)

    import os
    os.makedirs("output", exist_ok=True)

    plt.tight_layout()
    plt.savefig("output/ridge_trace.png", dpi=150, bbox_inches='tight')
    plt.show()

    return {
        "lambdas": lambdas,
        "coefficients": coefficients,
    }