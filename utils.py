import random
from config import RANDOM_STATE, is_zero, zero_rectify

def dot(x: list, y: list) -> float: 
    """Tính tích vô hướng của hai vector: sum(x[i] * y[i]).

    Tham số
    -------
    x : list[float] -- vector thứ nhất.
    y : list[float] -- vector thứ hai.

    Trả về
    ------
    float -- tích vô hướng của x và y.
    """
    if len(x) != len(y):
        raise ValueError("Các vector phải có cùng độ dài")
    
    s = 0.0
    for i in range(len(x)):
        s += x[i] * y[i]
    return s

def vector_add(x: list[float], y: list[float]) -> list[float]:
    """Cộng hai vector theo từng phần tử: x + y.

    Tham số
    -------
    x : list[float] -- vector thứ nhất.
    y : list[float] -- vector thứ hai.

    Trả về
    ------
    list[float] -- vector tổng [x[0]+y[0], ..., x[n]+y[n]].
    """
    if len(x) != len(y):
        raise ValueError("Các vector phải có cùng độ dài")
    
    result = []
    for i in range(len(x)):
        result.append(x[i] + y[i])
    return result

def vector_sub(x: list[float], y: list[float]) -> list[float]: 
    """Trừ hai vector theo từng phần tử: x - y.

    Tham số
    -------
    x : list[float] -- vector thứ nhất.
    y : list[float] -- vector thứ hai.

    Trả về
    ------
    list[float] -- vector hiệu [x[0]-y[0], ..., x[n]-y[n]].
    """
    if len(x) != len(y):
        raise ValueError("Các vector phải có cùng độ dài")
    
    result = []
    for i in range(len(x)):
        result.append(x[i] - y[i])
    return result

def scalar_multiply(x: list[float], k: float) -> list[float]:
    """Nhân vector với một số vô hướng: k * x.

    Tham số
    -------
    x : list[float] -- vector đầu vào.
    k : float       -- hệ số nhân.

    Trả về
    ------
    list[float] -- vector [k*x[0], ..., k*x[n]].
    """
    result = []
    for i in range(len(x)):
        result.append(k * x[i])
    return result

def norm(v: list[float]) -> float: 
    """Tính chuẩn Euclid (L2-norm) của vector: sqrt(sum(v[i]^2)).

    Tham số
    -------
    v : list[float] -- vector đầu vào.

    Trả về
    ------
    float -- chuẩn Euclid của vector.
    """
    s = 0.0
    for val in v:
        s += val * val
    return s**0.5

def normalize(v: list) -> list[float]:
    """Chuẩn hóa vector về độ dài 1: v / ||v||.

    Tham số
    -------
    v : list[float] -- vector đầu vào.

    Trả về
    ------
    list[float] -- vector đã chuẩn hóa.
    """
    return scalar_multiply(1/norm(v), v)

def transpose(A: list[list[float]]) -> list[list[float]]:
    """Tính chuyển vị ma trận: A^T[j][i] = A[i][j].

    Tham số
    -------
    A : list[list[float]], shape (m, n) -- ma trận đầu vào.

    Trả về
    ------
    list[list[float]], shape (n, m) -- ma trận chuyển vị.
    """
    m = len(A)
    n = len(A[0])
    AT = [[0.0 for _ in range(m)] for _ in range(n)]
    for i in range(m):
        for j in range(n): 
            AT[j][i] = A[i][j]
    return AT

def inverse(A: list[list[float]]) -> list[list[float]]:
    """Tính ma trận nghịch đảo bằng khử Gauss-Jordan với chọn trục.

    Tham số
    -------
    A : list[list[float]], shape (n, n) -- ma trận vuông.

    Trả về
    ------
    list[list[float]], shape (n, n) -- ma trận nghịch đảo A^{-1}.
    """
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
    """Nhân hai ma trận: C = A * B.

    Tham số
    -------
    A : list[list[float]], shape (m, n) -- ma trận thứ nhất.
    B : list[list[float]], shape (n, p) -- ma trận thứ hai.

    Trả về
    ------
    list[list[float]], shape (m, p) -- ma trận tích.
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
    """Nhân ma trận với vector: result[i] = hàng i của A nhân v.

    Tham số
    -------
    A : list[list[float]], shape (m, n) -- ma trận.
    v : list[float],       shape (n,)   -- vector.

    Trả về
    ------
    list[float], shape (m,) -- kết quả phép nhân A*v.
    """
    m = len(A)
    n = len(A[0])

    if n != len(v):
        raise ValueError("Ma trận và vector phải có kích thước tương thích")

    result = []
    for i in range(m):
        s = 0.0
        for j in range(n):
            s += A[i][j] * v[j]
        result.append(s)
    
    return result

def identity_matrix(n: int) -> list[list[float]]:
    """Tạo ma trận đơn vị kích thước n x n.

    Tham số
    -------
    n : int -- kích thước ma trận.

    Trả về
    ------
    list[list[float]], shape (n, n) -- ma trận đơn vị I_n.
    """
    I = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        I[i][i] = 1.0
    return I

def add_bias_column(X: list[list[float]]) -> list[list[float]]:
    """Thêm cột 1 (intercept) vào đầu ma trận X.

    Tham số
    -------
    X : list[list[float]], shape (n, p) -- ma trận đặc trưng.

    Trả về
    ------
    list[list[float]], shape (n, p+1) -- ma trận với cột 1 ở đầu.
    """
    return [[1.0] + row[:] for row in X]

def make_lambda_grid(
    start_exp: float = -3,
    stop_exp: float = 3,
    num: int = 50,
) -> list[float]:
    """Tạo lưới lambda theo thang log (logspace) mà không cần NumPy.

    Tham số
    -------
    start_exp : float -- số mũ bắt đầu (mặc định -3, tương ứng 10^{-3}).
    stop_exp  : float -- số mũ kết thúc (mặc định 3, tương ứng 10^3).
    num       : int   -- số lượng giá trị trong lưới.

    Trả về
    ------
    list[float] -- danh sách num giá trị từ 10^start_exp đến 10^stop_exp.
    """
    if num <= 0:
        raise ValueError("num phải lớn hơn 0")

    if num == 1:
        return [10.0**start_exp]

    step = (stop_exp - start_exp) / (num - 1)
    return [10.0 ** (start_exp + i * step) for i in range(num)]

def make_linear_data(
    n: int = 50,
    beta: list = None,
    sigma: float = 0.0,
    seed: int = RANDOM_STATE,
) -> tuple:
    """Sinh dữ liệu hồi quy tuyến tính: y = X * beta + noise.

    Tham số
    -------
    n     : int         -- số lượng mẫu.
    beta  : list[float] -- hệ số thực (mặc định [1.0, 2.0]).
    sigma : float       -- độ lệch chuẩn của nhiễu Gauss.
    seed  : int         -- hạt ngẫu nhiên.

    Trả về
    ------
    tuple -- (X, y, beta).
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

def make_multifeature_data(
    n: int = 100,
    p: int = 4,
    sigma: float = 1.0,
    seed: int = RANDOM_STATE,
) -> tuple:
    """Sinh dữ liệu hồi quy nhiều biến với hệ số ngẫu nhiên.

    Tham số
    -------
    n     : int   -- số lượng mẫu.
    p     : int   -- số lượng đặc trưng.
    sigma : float -- độ lệch chuẩn nhiễu.
    seed  : int   -- hạt ngẫu nhiên.

    Trả về
    ------
    tuple -- (X, y, beta).
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

def make_collinear_data(
    n: int = 100,
    seed: int = RANDOM_STATE,
) -> tuple:
    """Sinh dữ liệu có đa cộng tuyến: x3 gần bằng x1 + x2.

    Tham số
    -------
    n    : int -- số lượng mẫu.
    seed : int -- hạt ngẫu nhiên.

    Trả về
    ------
    tuple -- (X, y).
    """
    random.seed(seed)
    X = []
    y = []
    beta = [2.0, -1.0, 0.5]

    for _ in range(n):
        x1 = random.uniform(-10, 10)
        x2 = random.uniform(-10, 10)
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

def assert_close(
    actual,
    expected,
    label: str = "",
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """Kiểm tra hai giá trị gần bằng nhau (trong sai số cho phép).

    Tham số
    -------
    actual   : float/list -- giá trị thực tế.
    expected : float/list -- giá trị kỳ vọng.
    label    : str        -- nhãn mô tả.
    rtol     : float      -- sai số tương đối.
    atol     : float      -- sai số tuyệt đối.

    Trả về
    ------
    bool -- True nếu thỏa điều kiện gần bằng.
    """
    def is_close(a, b):
        return abs(a - b) <= atol + rtol * abs(b)

    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(
                f"{label}: độ dài không khớp "
                f"{len(actual)} != {len(expected)}"
            )

        for i in range(len(actual)):
            if not is_close(actual[i], expected[i]):
                raise AssertionError(
                    f"{label}: không khớp tại chỉ số {i}: "
                    f"{actual[i]} != {expected[i]}"
                )
    else:
        if not is_close(actual, expected):
            raise AssertionError(
                f"{label}: {actual} != {expected}"
            )
    return True

def assert_equal(
    actual,
    expected,
    label: str = ""
) -> bool:
    """Kiểm tra hai giá trị bằng nhau chính xác.

    Tham số
    -------
    actual   : any -- giá trị thực tế.
    expected : any -- giá trị kỳ vọng.
    label    : str -- nhãn mô tả.

    Trả về
    ------
    bool -- True nếu actual == expected.
    """
    if actual != expected:
        raise AssertionError(
            f"{label}: {actual} != {expected}"
        )
    return True

def assert_true(
    condition: bool,
    label: str = "",
    details: str = ""
) -> bool:
    """Kiểm tra điều kiện là True.

    Tham số
    -------
    condition : bool -- điều kiện cần kiểm tra.
    label     : str  -- nhãn mô tả.
    details   : str  -- thông tin chi tiết.

    Trả về
    ------
    bool -- True nếu điều kiện đúng.
    """
    if not condition:
        raise AssertionError(
            f"{label}: điều kiện là False. {details}"
        )
    return True

def assert_shape(
    arr,
    expected_shape: tuple,
    label: str = ""
) -> bool:
    """Kiểm tra shape của list hoặc ma trận.

    Tham số
    -------
    arr            : list  -- mảng cần kiểm tra.
    expected_shape : tuple -- shape kỳ vọng, ví dụ (n,) hoặc (n, p).
    label          : str   -- nhãn mô tả.

    Trả về
    ------
    bool -- True nếu shape khớp.
    """
    if isinstance(arr[0], list):
        actual_shape = (len(arr), len(arr[0]))
    else:
        actual_shape = (len(arr),)

    if actual_shape != expected_shape:
        raise AssertionError(
            f"{label}: kích thước {actual_shape} "
            f"!= {expected_shape}"
        )
    return True

def assert_in_range(
    val: float,
    lo: float,
    hi: float,
    label: str = ""
) -> bool:
    """Kiểm tra giá trị nằm trong khoảng [lo, hi].

    Tham số
    -------
    val   : float -- giá trị cần kiểm tra.
    lo    : float -- cận dưới.
    hi    : float -- cận trên.
    label : str   -- nhãn mô tả.

    Trả về
    ------
    bool -- True nếu lo <= val <= hi.
    """
    if not (lo <= val <= hi):
        raise AssertionError(
            f"{label}: {val} không nằm trong khoảng [{lo}, {hi}]"
        )
    return True

def assert_raises(
    exc_type: type,
    fn,
    *args,
    label: str = "",
    **kwargs
) -> bool:
    """Kiểm tra hàm fn(*args) ném đúng loại exception kỳ vọng.

    Tham số
    -------
    exc_type : type     -- loại exception kỳ vọng.
    fn       : callable -- hàm cần kiểm tra.
    *args    :          -- tham số truyền vào fn.
    label    : str      -- nhãn mô tả.
    **kwargs :          -- tham số từ khóa truyền vào fn.

    Trả về
    ------
    bool -- True nếu fn ném đúng exc_type.
    """
    try:
        fn(*args, **kwargs)
    except exc_type:
        return True
    except Exception as e:
        raise AssertionError(
            f"{label}: đã ném sai ngoại lệ {type(e)}"
        )
    raise AssertionError(
        f"{label}: kỳ vọng ngoại lệ {exc_type.__name__}"
    )

def solve_system(A: list[list[float]], b: list[float]) -> list[float]:
    """Giải hệ phương trình tuyến tính Ax = b bằng khử Gauss-Jordan.

    Sử dụng partial pivoting để tăng độ ổn định số học.

    Tham số
    -------
    A : list[list[float]], shape (n, n) -- ma trận hệ số.
    b : list[float],       shape (n,)   -- vector vế phải.

    Trả về
    ------
    list[float], shape (n,) -- nghiệm x của hệ Ax = b.
    """
    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]

    for col in range(n):
        pivot = col
        for r in range(col + 1, n):
            if abs(M[r][col]) > abs(M[pivot][col]):
                pivot = r

        if is_zero(abs(M[pivot][col])):
            raise ValueError("Hệ phương trình tuyến tính không có nghiệm duy nhất")

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