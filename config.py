RANDOM_STATE: int = 42 
EPSILON: float = 1e-12

def is_zero(x: float) -> bool:
    return abs(x) < EPSILON

def zero_rectify(value: float) -> float: 
    """Trả về 0.0 nếu giá trị quá nhỏ gần bằng 0.

    Tham số
    -------
    value : float -- giá trị cần kiểm tra.

    Trả về
    ------
    float -- 0.0 nếu giá trị nhỏ hơn sai số cho phép, ngược lại giữ nguyên.
    """
    return 0.0 if is_zero(value) else value

def calculate_relative_error(A: list, x_hat: list, b: list) -> float:
    """Tính sai số tương đối (relative error) của nghiệm gần đúng x_hat đối với hệ phương trình tuyến tính A * x = b.

    Công thức:
        ||A * x_hat - b|| / ||b||

    Tham số
    -------
    A     : list -- ma trận hệ số.
    x_hat : list -- vector nghiệm gần đúng.
    b     : list -- vector kết quả thực tế.

    Trả về
    ------
    float -- sai số tương đối.
    """
    from utils import matvec, vector_sub, norm

    Ax = matvec(A, x_hat)
    residual = vector_sub(Ax, b)

    residual_norm = norm(residual)
    b_norm = norm(b)

    if is_zero(b_norm):
        if is_zero(residual_norm):
            return 0.0
        raise ValueError("Sai so tuong doi khong xac dinh khi chuan cua b bang khong")

    return residual_norm / b_norm