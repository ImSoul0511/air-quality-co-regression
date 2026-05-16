RANDOM_STATE: int = 42 
EPSILON: float = 1e-12

def is_zero(x: float) -> bool:
    return abs(x) < EPSILON

def zero_rectify(value: float) -> float: 
    """ 
    Return zero if value is too small
    """

    return 0.0 if is_zero(value) else value

def calculate_relative_error(A: list, x_hat: list, b: list) -> float:
    from utils import matvec, vector_sub, norm
    from utils import matvec, vector_sub, norm
    """
    Tính relative error của nghiệm gần đúng x_hat
    đối với hệ phương trình tuyến tính:

        A * x = b

    Công thức:

        ||A*x_hat - b|| / ||b||

    Trong đó:
    - A: ma trận hệ số
    - x_hat: nghiệm gần đúng
    - b: vector kết quả thực tế

    Ý nghĩa:
    - Đo mức độ sai lệch của nghiệm x_hat
    - Giá trị càng nhỏ thì nghiệm càng chính xác
    - Nếu kết quả bằng 0 nghĩa là nghiệm khớp hoàn toàn

    Hàm không sử dụng thư viện ngoài.
    """

    Ax = matvec(A, x_hat)
    residual = vector_sub(Ax, b)

    residual_norm = norm(residual)
    b_norm = norm(b)

    if is_zero(b_norm):
        if is_zero(residual_norm):
            return 0.0
        raise ValueError("Relative error is undefined when norm of b is zero")

    return residual_norm / b_norm