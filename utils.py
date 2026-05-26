from config import RANDOM_STATE
import random
import math 

def dot(x: list, y: list) -> float: 
    """
    Calculate dot product of two vectors:   

        x * y = Σ x[i] * y[i]

    Args:
        x: first vector
        y: second vector
    Returns:
        float: dot product of two vectors
    """

    if len(x) != len(y):
        raise ValueError("Vector must have same length")
    
    s = 0.0
    for i in range(len(x)):
        s += x[i] * y[i]
    return s

def vector_add(x: list[float], y: list[float]) -> list[float]:
    """
    Calculate the sum of two vectors:

        x + y = [x[0] + y[0], ..., x[n] + y[n]]
    
    Args:
        x: first vector
        y: second vector
    Returns:
        list: sum of two vectors
    """

    if len(x) != len(y):
        raise ValueError("Vector must have same length")
    
    result = []
    for i in range(len(x)):
        result.append(x[i] + y[i])
    return result

def vector_sub(x: list[float], y: list[float]) -> list[float]: 
    """
    Subtract two vectors: 

        x - y = [x[0] - y[0], ..., x[n] - y[n]]
    
    Args:
        x: first vector
        y: second vector
    Returns:
        list: difference of two vectors
    """
    if len(x) != len(y):
        raise ValueError("Vector must have same length")
    
    result = []
    for i in range(len(x)):
        result.append(x[i] - y[i])
    return result

def scalar_multiply(x: list[float], k: float) -> list[float]:
    """
    Multiply a vector by a scalar:

        k * x = [k*x[0], ..., k*x[n]]

    Args:
        x: input vector
        k: scalar value

    Returns:
        list: scaled vector
    """

    result = []

    for i in range(len(x)):
        result.append(k * x[i])

    return result

def norm(v: list[float]) -> float: 
    """
    Calculate Euclidean norm of a vector:

        ||v|| = sqrt(v[0]^2 + ... + v[n]^2)
    
    Args:
        v: input vector
    Returns:
        float: Euclidean norm of the vector
    """
    
    s = 0.0
    for val in v:
        s += val * val
    
    return s**0.5

def normalize(v: list) -> list[float]:
    """
    Normalize a vector:

        v / ||v||
    
    Args:
        v: input vector
    Returns:
        list: normalized vector
    """
    return scalar_multiply(1/norm(v), v)

def transpose(A: list[list[float]]) -> list[list[float]]:
    """
    Calculate transposed matrix:

        A^T = [a_{ji}]
    
    Args:
        A: input matrix
    Returns:
        list: transpose matrix
    """
    m = len(A)
    n = len(A[0])

    AT = [[0.0 for _ in range(m)] for _ in range(n)]

    for i in range(m):
        for j in range(n): 
            AT[j][i] = A[i][j]
            
    return AT

def inverse(A: list[list[float]]) -> list[list[float]]:
    from config import zero_rectify, is_zero
    n = len(A)

    if n == 0 or len(A[0]) != n:
        raise ValueError("A phải là ma trận vuông")

    I = identity_matrix(n)
    M = [A[i][:] + I[i][:] for i in range(n)]

    for col in range(n):
        pivot = col

        for r in range(col + 1, n):
            if abs(M[r][col]) > abs(M[pivot][col]):
                pivot = r

        if is_zero(abs(M[pivot][col])):
            raise ValueError("Ma trận không khả nghịch")

        M[col], M[pivot] = M[pivot], M[col]

        pivot_val = M[col][col]
        for j in range(2 * n):
            M[col][j] /= pivot_val

        for r in range(n):
            if r != col:
                factor = M[r][col]
                for j in range(2 * n):
                    M[r][j] -= factor * M[col][j]

    A_inv = []
    for i in range(n):
        row = []
        for j in range(n, 2 * n):
            row.append(zero_rectify(M[i][j]))
        A_inv.append(row)

    return A_inv

def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """
    Calculate matrix multiplication: 
        C = A * B
    """
    m = len(A)
    n = len(A[0])
    p = len(B[0])

    C = [[0.0 for _ in range(p)] for _ in range(m)]

    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

def matvec(A: list[list[float]], v: list[float]) -> list[float]:
    """
    Calculate matrix-vector multiplication: 
        Ax = [A[0]*x, ..., A[m]*x]
    
    Args:
        A: input matrix
        v: input vector
    Returns:
        list: matrix-vector product
    """
    m = len(A)
    n = len(A[0])

    if n != len(v):
        raise ValueError("Matrix và vector phải có kích thước tương thích")

    result = []
    for i in range(m):
        s = 0.0
        for j in range(n):
            s += A[i][j] * v[j]
        result.append(s)
    
    return result

def identity_matrix(n: int) -> list[list[float]]:
    """
    Create identity matrix:
        I_n = [δ_{ij}]
    
    Args:
        n: matrix size
    Returns:
        list: identity matrix with size n x n
    """

    I = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        I[i][i] = 1.0
    return I



# =========================================================
# DATASET GENERATORS
# =========================================================

# Sinh dataset tuyến tính y = X·beta + noise
def make_linear_data(
    n: int = 50,
    beta: list = None,
    sigma: float = 0.0,
    seed: int = RANDOM_STATE,
) -> tuple:
    """
    Generate linear regression dataset:

        y = X * beta + noise

    Args:
        n: number of samples
        beta: true coefficients
        sigma: standard deviation of Gaussian noise
        seed: random seed

    Returns:
        tuple: (X, y, beta)
    """

    random.seed(seed)

    if beta is None:
        beta = [1.0, 2.0]

    p = len(beta)

    X = []
    y = []

    for _ in range(n):
        row = []

        for _ in range(p):
            row.append(random.uniform(-10, 10))

        noise = random.gauss(0, sigma)

        target = 0.0
        for j in range(p):
            target += row[j] * beta[j]

        target += noise

        X.append(row)
        y.append(target)

    return X, y, beta


# Sinh dataset hồi quy nhiều biến ngẫu nhiên
def make_multifeature_data(
    n: int = 100,
    p: int = 4,
    sigma: float = 1.0,
    seed: int = RANDOM_STATE,
) -> tuple:
    """
    Generate multifeature regression dataset.

    Args:
        n: number of samples
        p: number of features
        sigma: noise standard deviation
        seed: random seed

    Returns:
        tuple: (X, y, beta)
    """

    random.seed(seed)

    beta = []

    for _ in range(p):
        beta.append(random.uniform(-5, 5))

    return make_linear_data(
        n=n,
        beta=beta,
        sigma=sigma,
        seed=seed,
    )


# Sinh dataset có đa cộng tuyến
# x3 ≈ x1 + x2
def make_collinear_data(
    n: int = 100,
    seed: int = RANDOM_STATE,
) -> tuple:
    """
    Generate dataset with multicollinearity.

    x3 ≈ x1 + x2

    Returns:
        tuple: (X, y)
    """

    random.seed(seed)

    X = []
    y = []

    beta = [2.0, -1.0, 0.5]

    for _ in range(n):
        x1 = random.uniform(-10, 10)
        x2 = random.uniform(-10, 10)

        # gần phụ thuộc tuyến tính
        x3 = x1 + x2 + random.gauss(0, 0.01)

        row = [x1, x2, x3]

        target = (
            beta[0] * x1 +
            beta[1] * x2 +
            beta[2] * x3
        )

        target += random.gauss(0, 1)

        X.append(row)
        y.append(target)

    return X, y


# =========================================================
# ASSERT HELPERS
# =========================================================

# So sánh hai giá trị/array với sai số cho phép
def assert_close(
    actual,
    expected,
    label: str = "",
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """
    Assert approximate equality.
    """

    def is_close(a, b):
        return abs(a - b) <= atol + rtol * abs(b)

    if isinstance(actual, list) and isinstance(expected, list):

        if len(actual) != len(expected):
            raise AssertionError(
                f"{label}: length mismatch "
                f"{len(actual)} != {len(expected)}"
            )

        for i in range(len(actual)):
            if not is_close(actual[i], expected[i]):
                raise AssertionError(
                    f"{label}: mismatch at index {i}: "
                    f"{actual[i]} != {expected[i]}"
                )

    else:
        if not is_close(actual, expected):
            raise AssertionError(
                f"{label}: {actual} != {expected}"
            )

    return True


# So sánh bằng nhau
def assert_equal(
    actual,
    expected,
    label: str = ""
) -> bool:
    """
    Assert exact equality.
    """

    if actual != expected:
        raise AssertionError(
            f"{label}: {actual} != {expected}"
        )

    return True


# Kiểm tra điều kiện Boolean
def assert_true(
    condition: bool,
    label: str = "",
    details: str = ""
) -> bool:
    """
    Assert condition is True.
    """

    if not condition:
        raise AssertionError(
            f"{label}: condition is False. {details}"
        )

    return True


# Kiểm tra shape của array/list
def assert_shape(
    arr,
    expected_shape: tuple,
    label: str = ""
) -> bool:
    """
    Assert shape of list/matrix.
    """

    if isinstance(arr[0], list):
        actual_shape = (len(arr), len(arr[0]))
    else:
        actual_shape = (len(arr),)

    if actual_shape != expected_shape:
        raise AssertionError(
            f"{label}: shape {actual_shape} "
            f"!= {expected_shape}"
        )

    return True


# Kiểm tra lo ≤ val ≤ hi
def assert_in_range(
    val: float,
    lo: float,
    hi: float,
    label: str = ""
) -> bool:
    """
    Assert value is within range.
    """

    if not (lo <= val <= hi):
        raise AssertionError(
            f"{label}: {val} not in range [{lo}, {hi}]"
        )

    return True


# Kiểm tra fn(*args) ném đúng exception
def assert_raises(
    exc_type: type,
    fn,
    *args,
    label: str = "",
    **kwargs
) -> bool:
    """
    Assert function raises expected exception.
    """

    try:
        fn(*args, **kwargs)

    except exc_type:
        return True

    except Exception as e:
        raise AssertionError(
            f"{label}: raised wrong exception {type(e)}"
        )

    raise AssertionError(
        f"{label}: expected exception {exc_type.__name__}"
    )

def solve_system(A: list[list[float]], b: list[float]) -> list[float]:
    """
    Solve Ax = b using Gauss-Jordan elimination with partial pivoting.
    """
    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]

    for col in range(n):
        pivot = col
        for r in range(col + 1, n):
            if abs(M[r][col]) > abs(M[pivot][col]):
                pivot = r

        if is_zero(abs(M[pivot][col])):
            raise ValueError("Linear system has no unique solution")

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