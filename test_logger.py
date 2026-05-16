import math
import random
import pandas as pd

# 1. Import các hàm tính toán lõi 
from part1.ols_implementation import ols_fit, coef_inference, vif
# 2. Import các hàm phục vụ phân hệ mô phỏng F10
from part1.gauss_markov_demo import run_gauss_markov_simulation, calculate_stats

class TestLogger:

    # ANSI color codes
    _GREEN  = "\033[92m"
    _RED    = "\033[91m"
    _YELLOW = "\033[93m"
    _CYAN   = "\033[96m"
    _GRAY   = "\033[90m"
    _BOLD   = "\033[1m"
    _RESET  = "\033[0m"

    # In tiêu đề lớn - gọi một lần đầu mỗi hàm/nhóm test.
    def print_suite_header(cls, suite_name: str) -> None:
        line = "=" * 60
        print(f"\n{cls._BOLD}{cls._CYAN}{line}")
        print(f"  {suite_name}")
        print(f"{line}{cls._RESET}\n")

    # In tiêu đề nhóm nhỏ, gọi trước một nhóm test liên quan.
    def print_group(cls, group_name: str) -> None:
        print(f"\n{cls._BOLD}{cls._YELLOW}--- {group_name} ---{cls._RESET}")

    # In một dòng kết quả PASSED / FAILED.
    def print_result(cls, test_name: str, passed: bool, details: str = "") -> None:
        if passed:
            status = f"{cls._GREEN}PASSED{cls._RESET}"
        else:
            status = f"{cls._RED}FAILED{cls._RESET}"

        msg = f"  [{status}] {test_name}"

        if details:
            msg += f"  {cls._GRAY}({details}){cls._RESET}"

        print(msg)

    # In giá trị actual (và expected nếu có) - tiện debug.
    def print_value(cls, label: str, actual, expected=None) -> None:
        print(f"  {cls._GRAY}{label}:{cls._RESET}  {actual}", end="")

        if expected is not None:
            print(f"  {cls._GRAY}(expected: {expected}){cls._RESET}", end="")

        print()

    # In cảnh báo
    def print_warning(cls, message: str, detail: str = "") -> None:
        msg = f"  {cls._YELLOW}WARNING: {message}{cls._RESET}"

        if detail:
            msg += f"  {cls._GRAY}{detail}{cls._RESET}"

        print(msg)

    # In thông tin phụ - màu xám nhạt
    def print_info(cls, message: str) -> None:
        print(f"  {cls._GRAY}{message}{cls._RESET}")

    # In tổng kết cuối suite - gọi sau khi chạy hết test.
    def print_summary(cls, passed_count: int, total_count: int) -> None:
        failed = total_count - passed_count
        line = "=" * 60

        print(f"\n{cls._BOLD}{line}")

        if failed == 0:
            color = cls._GREEN
        else:
            color = cls._RED

        print(
            f"  {color}{passed_count}/{total_count} passed"
            f"  |  {failed} failed{cls._RESET}"
        )

        print(f"{cls._BOLD}{line}{cls._RESET}\n")

# ---------------------------------------------------------------------------
# Unit Tests — F4: coef_inference 
# ---------------------------------------------------------------------------

def test_coef_inference_output_structure():
    """F4 - Test 1: Đảm bảo cấu trúc DataFrame trả về đúng số dòng và đủ các cột hệ số."""
    X = [[1.0, 2.0], [2.0, 4.5], [3.5, 6.0], [4.0, 8.0], [5.0, 9.5]]
    y = [2.1, 3.9, 5.2, 6.1, 7.4]
    beta_hat = [0.5, 1.2, -0.1] 
    sigma2 = 0.04
    
    df_res = coef_inference(X, y, beta_hat, sigma2)
    
    assert len(df_res) == 3, f"Expected 3 rows, got {len(df_res)}"
    expected_cols = ['coef', 'std_err', 't_stat', 'p_value', 'ci_lower', 'ci_upper']
    for col in expected_cols:
        assert col in df_res.columns, f"Missing required column: {col}"


def test_coef_inference_p_value_bounds():
    """F4 - Test 2: Giá trị p-value thống kê bắt buộc phải nằm trong đoạn [0, 1]."""
    X = [[0.1, -0.5], [1.2, 0.3], [-0.8, 1.1], [2.0, -1.2], [0.5, 0.5]]
    y = [1.0, 2.5, -0.5, 3.0, 1.2]
    beta_hat = [0.8, 1.1, 0.4]
    sigma2 = 0.1
    
    df_res = coef_inference(X, y, beta_hat, sigma2)
    
    for p_val in df_res['p_value']:
        assert 0.0 <= p_val <= 1.0, f"Violation: p_value {p_val} must be between 0 and 1"


def test_coef_inference_ci_logic():
    """F4 - Test 3: Logic khoảng tin cậy phải nhất quán (ci_lower <= coef <= ci_upper)."""
    X = [[10.0], [20.0], [30.0], [40.0]]
    y = [5.0, 9.0, 16.0, 21.0]
    beta_hat = [0.1, 0.5]
    sigma2 = 0.5
    
    df_res = coef_inference(X, y, beta_hat, sigma2)
    
    for i in range(len(df_res)):
        low = df_res['ci_lower'].iloc[i]
        coef = df_res['coef'].iloc[i]
        high = df_res['ci_upper'].iloc[i]
        assert low <= coef <= high, f"Inconsistent CI bounds at row {i}: {low} <= {coef} <= {high}"


def test_coef_inference_zero_variance_handling():
    """F4 - Test 4: Biên dữ liệu đặc biệt - Khi sigma2 tiến gần 0, std_err phải cực nhỏ."""
    X = [[1.0], [2.0], [3.0]]
    y = [2.0, 4.0, 6.0]
    beta_hat = [0.0, 2.0]
    sigma2 = 1e-9
    
    df_res = coef_inference(X, y, beta_hat, sigma2)
    assert df_res['std_err'].iloc[1] < 1e-3, "std_err should be extremely small when noise variance is near zero"


# ---------------------------------------------------------------------------
# Unit Tests — F5: VIF (Đảm bảo 4 tests)
# ---------------------------------------------------------------------------

def test_vif_output_format():
    """F5 - Test 1: Kết quả trả về phải là một dictionary với số phần tử bằng số biến độc lập."""
    X = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    vif_dict = vif(X)
    
    assert isinstance(vif_dict, dict), "VIF target output must be a Python dictionary"
    assert len(vif_dict) == 3, f"Expected 3 feature keys, got {len(vif_dict)}"


def test_vif_lower_bound():
    """F5 - Test 2: VIF của mọi biến độc lập về mặt toán học luôn phải >= 1.0."""
    X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    vif_dict = vif(X)
    
    for var_name, vif_val in vif_dict.items():
        assert vif_val >= 1.0, f"VIF of {var_name} is {vif_val}, cannot be < 1.0"


def test_vif_orthogonal_features():
    """F5 - Test 3: Khi các biến độc lập hoàn toàn trực giao, VIF phải xấp xỉ 1.0."""
    X_orthogonal = [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    vif_dict = vif(X_orthogonal)
    
    for val in vif_dict.values():
        assert abs(val - 1.0) < 1e-4, f"Expected VIF near 1.0, got {val}"


def test_vif_high_collinearity():
    """F5 - Test 4: Khi có 2 biến phụ thuộc tuyến tính mạnh, hệ số VIF của chúng phải > 10."""
    # SỬ DỤNG SEED TẠM THỜI ĐỂ TRÁNH TRÙNG VỚI CÁC BIẾN TOÀN CỤC KHÁC
    random.seed(42)
    X_collinear = []
    for _ in range(30):
        x1 = random.gauss(0, 1)
        x2 = 3.0 * x1 + random.gauss(0, 0.001) 
        X_collinear.append([x1, x2])
        
    vif_dict = vif(X_collinear)
    assert vif_dict['x1'] > 10.0, f"Expected VIF > 10, got {vif_dict['x1']}"
    assert vif_dict['x2'] > 10.0, f"Expected VIF > 10, got {vif_dict['x2']}"


# ---------------------------------------------------------------------------
# Unit Tests — F10: Gauss-Markov Simulation (Đảm bảo 4 tests)
# ---------------------------------------------------------------------------

def test_gauss_markov_output_shapes():
    """F10 - Test 1: Đầu ra của hàm mô phỏng phải chứa đủ số lượng mẫu (1000 vòng lặp)."""
    # ĐÃ BỎ LỆNH IMPORT TRONG HÀM (INLINE IMPORT) VÌ ĐÃ CÓ Ở ĐẦU FILE
    beta_ols, beta_alt, true_beta = run_gauss_markov_simulation()
    
    assert len(beta_ols) == 1000, f"Expected 1000, got {len(beta_ols)}"
    assert len(beta_alt) == 1000, f"Expected 1000, got {len(beta_alt)}"
    assert len(true_beta) == 3, f"True beta size error: {len(true_beta)}"


def test_gauss_markov_unbiasedness():
    """F10 - Test 2: Chứng minh tính không chệch: Kỳ vọng E[β̂] hội tụ về β thực."""
    beta_ols, _, true_beta = run_gauss_markov_simulation()
    mean_ols, _ = calculate_stats(beta_ols)
    
    for i in range(len(true_beta)):
        assert abs(mean_ols[i] - true_beta[i]) < 0.1, f"OLS biased at index {i}"


def test_gauss_markov_blue_variance():
    """F10 - Test 3: Chứng minh tính chất BLUE: Var(β̂_OLS) <= Var(β̂_alt) trên từng chiều."""
    beta_ols, beta_alt, _ = run_gauss_markov_simulation()
    _, var_ols = calculate_stats(beta_ols)
    _, var_alt = calculate_stats(beta_alt)
    
    for i in range(len(var_ols)):
        assert var_ols[i] <= var_alt[i], f"BLUE Violation at coordinate {i}"


def test_gauss_markov_reproducibility():
    """F10 - Test 4: Đảm bảo tính lặp lại (Reproducibility) nhờ seed cố định."""
    beta_ols_run1, _, _ = run_gauss_markov_simulation()
    beta_ols_run2, _, _ = run_gauss_markov_simulation()
    
    for row1, row2 in zip(beta_ols_run1[:5], beta_ols_run2[:5]):
        for b1, b2 in zip(row1, row2):
            assert b1 == b2, f"Simulation is not deterministic: {b1} != {b2}"