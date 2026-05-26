# hân công công việc - TUDTK

> MTH00051 | Deadline: **30/05/2026 - 23:59**
> 

---

## MỤC LỤC

---

## QUY CHUẨN CHUNG

### 1. Python / Môi Trường

```
Python  : 3.10+
matplotlib : 3.8+
seaborn    : 0.13+

pandas  : 2.2+ # Đọc, xử lý và thao tác dữ liệu.
scikit-learn : 1.4+   # scikit-learn chỉ để verify

# scipy, numpy Chỉ dùng để test
scipy      : 1.12+
numpy     : 1.26+
```

- `requirements.txt`
- Chạy `pip install -r requirements.txt`

### 2. Random Seed

```python
RANDOM_STATE = 42
# train_test_split(..., random_state=RANDOM_STATE)
# KFold(..., shuffle=True, random_state=RANDOM_STATE)
```

### 3. Dữ Liệu

| Ký hiệu | **Kiểu dữ liệu** | Mô tả |
| --- | --- | --- |
| `X` | `list[list[float]]` | Ma trận features, **chưa** có cột bias |
| `X_bias` | `list[list[float]]` | Ma trận design đã thêm cột 1 đầu |
| `y` | `list[float]` | Vector target (1D) |
| `beta_hat` | `list[float]` | Vector hệ số (bao gồm intercept) |
| `y_hat` | `list[float]` | Giá trị dự đoán |
| `lam` | `float` | Lambda (hyperparameter regularization) |
| `k` | `int` | Số fold trong cross-validation |

> **Quy tắc:** Tất cả hàm trong `part1/` nhận `X` **chưa có** bias - hàm tự thêm cột 1 bên trong nếu cần. Ghi rõ trong docstring.
> 

### 4. Visualization Standards

```python
# Mọi plot phải có đủ 4 thứ:
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_title("Tên biểu đồ rõ ràng", fontsize=13)
ax.set_xlabel("Tên trục X (đơn vị)")
ax.set_ylabel("Tên trục Y (đơn vị)")
ax.legend()
plt.tight_layout()
plt.savefig("output/ten_bieu_do.png", dpi=150, bbox_inches='tight')
plt.show()
```

### 5. Unit Test

- Mỗi hàm chính có **ít nhất 4 unit test** `test_*()`trong file tương ứng.
- Test phải **không phụ thuộc** vào file data bên ngoài - tự tạo data trong test.

### 6. Git Workflow

```
main
├── feat/part1-ols-metrics     ← F1, F2, F3        (1)
├── feat/part1-inference-gm    ← F4, F5, F10       (2)
├── feat/part1-ridge-cv        ← F6, F7, F8, F9    (3)
├── feat/part1-notebook        ← F11               (4 + 5)
├── feat/part2-pipeline        ← T3, compare       (6)
├── feat/part2-models          ← T4                (7)
├── feat/part2-advanced        ← T6                (8)
└── feat/part2-notebook        ← T5                (9 + 10)
```

- Commit message: `[P1/F3] Implement model_metrics with R2, adj-R2, F-test`

### 7. Tái Sử Dụng Project 1

Các hàm từ Project 1. trong `Utils.py`

- `transpose(A)`
- `matrix_multiply(A, B)` (Nhân ma trận với ma trận, hoặc ma trận với vector cột)
- `inverse(A)` (Nghịch đảo ma trận bằng Gauss-Jordan)
- `gaussian_eliminate(A, b) + back_substitution(U, c)` (Giải hệ phương trình tuyến tính bằng phép khử Gauss)

Các hàm cơ bản khác (tính mean, chuẩn hóa, tính norm...).

### 8. TestLogger / Config / Utils

- Config
    
    ```python
    RANDOM_STATE: int   = 42
    EPSILON:      float = 1e-12
    
    def is_zero(x: float) -> bool:
    def zero_rectify(value: float) -> float:
    def calculate_relative_error(A: list, x_hat: list, b: list) -> float:
    ```
    
- **Các utils chỉ dùng để test**
    
    ```python
    # Sinh dataset tuyến tính  y = X·beta + noise.
    def make_linear_data(
        n:     int   = 50,
        beta:  list  = None,
        sigma: float = 0.0,
        seed:  int   = RANDOM_STATE,
    ) -> tuple
    
    # Sinh dataset hồi quy nhiều biến ngẫu nhiên.
    def make_multifeature_data(
        n:     int   = 100,
        p:     int   = 4,
        sigma: float = 1.0,
        seed:  int   = RANDOM_STATE,
    ) -> tuple:
    
    # Sinh dataset có đa cộng tuyến: x3 ≈ x1 + x2.
    def make_collinear_data(
        n:    int = 100,
        seed: int = RANDOM_STATE,
    ) -> tuple:
    
    # So sánh hai giá trị/array với sai số cho phép.
    def assert_close(
        actual,
        expected,
        label:   str   = "",
        rtol:    float = 1e-5,
        atol:    float = 1e-8,
    ) -> bool:
    
    # So sánh bằng nhau
    def assert_equal(
    		actual,
    		expected,
    		label: str = ""
    ) -> bool:
    
    # Kiểm tra điều kiện Boolean.
    def assert_true(
    		condition: bool,
    		label: str = "",
    		details: str = ""
    ) -> bool:
    
    # Kiểm tra shape của array/list.
    def assert_shape(arr, expected_shape: tuple, label: str = "") -> bool:
    
    # Kiểm tra lo ≤ val ≤ hi.
    def assert_in_range(val: float, lo: float, hi: float, label: str = "") -> bool:
    
    # Kiểm tra fn(*args) ném đúng loại exception.
    def assert_raises(exc_type: type, fn, *args, label: str = "", **kwargs) -> bool:
    ```
    
- TestLogger
    
    ```python
    class TestLogger:
    	# In tiêu đề lớn - gọi một lần đầu mỗi hàm/nhóm test.
    	def print_suite_header(cls, suite_name: str) -> None:
    
    	# In tiêu đề nhóm nhỏ, gọi trước một nhóm test liên quan.
      def print_group(cls, group_name: str) -> None:
    
      # In một dòng kết quả PASSED / FAILED.
      def print_result(cls, test_name: str, passed: bool, details: str = "") -> None:
    
      # In giá trị actual (và expected nếu có) - tiện debug.
      def print_value(cls, label: str, actual, expected=None) -> None:
    
      # In cảnh báo
      def print_warning(cls, message: str, detail: str = "") -> None:
    
      # In thông tin phụ - màu xám nhạt
      def print_info(cls, message: str) -> None:
    
      # In tổng kết cuối suite - gọi sau khi chạy hết test.
      def print_summary(cls, passed_count: int, total_count: int) -> None:
    ```
    

---

## CẤU TRÚC THƯ MỤC

```
Group_<ID>/
├── README.md
├── requirements.txt
├── report/
│   ├── report.pdf
│   └── report.tex
├── part1/
│   ├── ols_implementation.py   ← F1, F2, F3, F4, F5
│   ├── ridge_lasso.py          ← F6, F7
│   ├── residual_analysis.py    ← F8
│   ├── cross_validation.py     ← F9
│   ├── gauss_markov_demo.py    ← F10
│   └── part1_notebook.ipynb
└── part2/
    ├── data/
    │   └── <dataset>.csv
    ├── data_pipeline.py        ← T3: DataPipeline class
    ├── model_comparison.py     ← T4, T5
    ├── advanced_methods.py     ← T7
    └── part2_notebook.ipynb
```

---

## PHẦN 1 - Lý Thuyết & Cài Đặt OLS Từ Đầu

> **File:** `part1/ols_implementation.py` (F1–F5), `ridge_lasso.py` (F6–F7), `residual_analysis.py` (F8), `cross_validation.py` (F9), `gauss_markov_demo.py` (F10)
> 

---

### F1. `ols_fit(X, y)`

**Mục đích:** Giải Normal Equations để tính nghiệm OLS từ đầu (không dùng `np.linalg.lstsq` hay sklearn).

**Công thức:**

```
β̂ = (XᵀX)⁻¹Xᵀy
σ̂² = RSS / (n - p - 1)
```

**Input:**

| Tham số | Kiểu | Shape | Mô tả |
| --- | --- | --- | --- |
| `X` | `list[list[float]]` | `(n, p)` | Features, chưa có bias column |
| `y` | `list[float]` | `(n,)` | Target liên tục |

**Output:** `dict`

| Key | Kiểu | Shape/Type | Mô tả |
| --- | --- | --- | --- |
| `beta_hat` | `list[float]` | `(p+1,)` | `[intercept, β1, ..., βp]` |
| `sigma2_hat` | `float` | scalar | Ước lượng phương sai nhiễu |
| `y_hat` | `list[float]` | `(n,)` | Fitted values |
| `residuals` | `list[float]` | `(n,)` | `y - y_hat` |

**Thuật toán từng bước:**

```
1. Thêm cột 1 vào X → X_bias shape (n, p+1)
2. Tính A = X_biasᵀ @ X_bias          # (p+1, p+1)
3. Tính b = X_biasᵀ @ y               # (p+1,)
4. Giải hệ A @ beta = b:
5. Tính y_hat = X_bias @ beta_hat
6. Tính residuals = y - y_hat
7. Tính RSS
8. Tính sigma2_hat = RSS / (n - p - 1)
```

**Notebook demo (`part1_notebook.ipynb`):**

- Sinh dữ liệu giả lập bằng `make_linear_data`, gọi `ols_fit`, in bảng so sánh `beta_hat` vs `TRUE_BETA`
- Kiểm chứng lại bằng `np.linalg.lstsq` hoặc `sklearn.LinearRegression`, in sai số tương đối

---

### F2. `hat_matrix(X)`

**Mục đích:** Tính Hat Matrix H = X(XᵀX)⁻¹Xᵀ và kiểm tra các tính chất.

**Input:** `X` - `list[float]`, chưa có bias.

**Output:** `dict`

| Key | Kiểu | Mô tả |
| --- | --- | --- |
| `H` | `list[float]` | Hat matrix |
| `is_idempotent` | `bool` | H² ≈ H |
| `is_symmetric` | `bool` | Hᵀ ≈ H |
| `rank` | `int` | rank(H) = p+1 |
| `eigenvalues` | `list[float` | Giá trị riêng (chỉ 0 hoặc 1) |

**Thuật toán:**

```
1. Thêm cột bias → X_bias (n, p+1)
2. Tính A = X_biasᵀ @ X_bias
3. A_inv = inverse(A)
4. H = X_bias @ A_inv @ X_bias.T
5. Kiểm tra idempotent: H_squared = matmul(H, H). So sánh từng phần tử với H (dùng sai số 1e-8).
6. Kiểm tra symmetric: H_T = transpose(H). So sánh từng phần tử.
7. Tính rank
8. Tính eigenvalues
```

**Visualization bắt buộc:**

- Histogram của eigenvalues của H → chứng minh chỉ có giá trị 0 hoặc 1.
- Heatmap của H (với dữ liệu nhỏ n ≤ 20).

**Notebook demo (`part1_notebook.ipynb`):**

- Hiển thị heatmap H và histogram eigenvalues, nhận xét xác nhận tính idempotent và symmetric
- In bảng kết quả `is_idempotent`, `is_symmetric`, `rank`

---

### F3. `model_metrics(y, y_hat, p)`

**Mục đích:** Tính đầy đủ các chỉ số đánh giá mô hình.

**Input:**

| Tham số | Kiểu | Mô tả |
| --- | --- | --- |
| `y` | `list[float]` | Ground truth |
| `y_hat` | `list[float]` | Dự đoán |
| `p` | `int` | Số features (không tính intercept) |

**Output:** `dict`

| Key | Công thức | Mô tả |
| --- | --- | --- |
| `RSS` | `Σ(yᵢ - ŷᵢ)²` | Residual Sum of Squares |
| `TSS` | `Σ(yᵢ - ȳ)²` | Total Sum of Squares |
| `MSS` | `TSS - RSS` | Model Sum of Squares |
| `R2` | `1 - RSS/TSS` | Hệ số xác định |
| `R2_adj` | `1 - (n-1)/(n-p-1) * (1-R²)` | R² hiệu chỉnh |
| `F_stat` | `(MSS/p) / (RSS/(n-p-1))` | F-statistic |
| `F_pvalue` |  | p-value của F-test |
| `MAE` | `mean( | y - ŷ |
| `RMSE` | `sqrt(mean((y-ŷ)²))` | Root Mean Squared Error |

**Notebook demo (`part1_notebook.ipynb`):**

- Gọi `model_metrics` trên kết quả của `ols_fit`, in bảng đầy đủ RSS/TSS/R²/Adj-R²/F-stat/p-value/MAE/RMSE
- Kiểm chứng R² bằng `sklearn.metrics.r2_score`

---

### F4. `coef_inference(X, y, beta_hat, sigma2)`

**Mục đích:** Tính standard errors, t-statistics, p-values, và confidence intervals cho từng hệ số.

**Input:**

| Tham số | Kiểu | Mô tả |
| --- | --- | --- |
| `X` | `list[float] (n, p)` | Features, chưa có bias |
| `y` | `list[float] (n,)` | Target |
| `beta_hat` | `list[float] (p+1,)` | Hệ số đã fit |
| `sigma2` | `float` | σ̂² từ `ols_fit` |

**Output:** `DataFrame` với index là tên cột `['intercept', 'x1', ..., 'xp']`

| Column | Công thức | Mô tả |
| --- | --- | --- |
| `coef` | `beta_hat[j]` | Hệ số |
| `std_err` | `sqrt(σ̂² * [(XᵀX)⁻¹]ⱼⱼ)` | Standard error |
| `t_stat` | `coef / std_err` | t-statistic |
| `p_value` | `2 * t.sf( | t_stat |
| `ci_lower` | `coef - t_{0.025} * std_err` | Cận dưới CI 95% |
| `ci_upper` | `coef + t_{0.025} * std_err` | Cận trên CI 95% |

**Thuật toán:**

```
1. X_bias = thêm cột 1
2. A = X_biasᵀ @ X_bias
3. A_inv = inverse(A)
4. Cov_beta = sigma2 * A_inv        # Ma trận hiệp phương sai của β̂
5. std_errs = sqrt(np.diag(Cov_beta))
6. t_stats = beta_hat / std_errs
7. df = n - p - 1
8. p_values
9. t_crit
10. ci_lower = beta_hat - t_crit * std_errs
11. ci_upper = beta_hat + t_crit * std_errs
```

**Notebook demo (`part1_notebook.ipynb`):**

- In DataFrame kết quả `coef_inference`, highlight các hệ số có `p_value < 0.05`
- Kiểm chứng bằng `statsmodels.OLS().fit().summary()`

---

### F5. `vif(X)`

**Mục đích:** Phát hiện đa cộng tuyến bằng Variance Inflation Factor.

**Input:** `X` - `list[float] (n, p)`, chưa có bias (intercept không tính VIF).

**Output:** `dict` với key là `['x1', ..., 'xp']` (hoặc tên cột nếu truyền vào), value là `float` VIF.

**Thuật toán:**

```
For j = 0, 1, ..., p-1:
    1. X_j = X[:, j]                         # Biến cần tính VIF
    2. X_others = X[:, [i for i ≠ j]]        # Các biến còn lại
    3. result_j = ols_fit(X_others, X_j)     # Hồi quy X_j theo các biến còn lại
    4. R2_j = model_metrics(X_j, result_j['y_hat'], p-1)['R2']
    5. VIF_j = 1 / (1 - R2_j)
```

**Notebook demo (`part1_notebook.ipynb`):**

- Chạy `vif` trên `make_collinear_data`, in bảng VIF, chỉ ra biến nào có VIF > 10
- So sánh với `statsmodels.stats.outliers_influence.variance_inflation_factor`

---

### F6. `ridge_fit(X, y, lam)`

**Mục đích:** Cài đặt Ridge Regression với closed-form solution.

**File:** `part1/ridge_lasso.pydic`

**Công thức:** `β̂_ridge = (XᵀX + λI)⁻¹Xᵀy`

**Input:**

| Tham số | Kiểu | Mô tả |
| --- | --- | --- |
| `X` | `list[float] (n, p)` | Features, chưa có bias |
| `y` | `list[float] (n,)` | Target |
| `lam` | `float ≥ 0` | Hệ số regularization λ |

**Output:** `dict`

| Key | Mô tả |
| --- | --- |
| `beta_hat` | `list[float] (p+1,)` - hệ số ridge (intercept + p features) |
| `y_hat` | Fitted values |
| `residuals` | Phần dư |

**Thuật toán:**

```
1. Chuẩn hóa X truo·ươớc khi fit Ridge:
   - Tính mean_X, std_X trên train set
   - X_scaled = (X - mean_X) / std_X
2. X_bias = thêm cột 1 vào X_scaled
3. n, p1 = X_bias.shape   (p1 = p + 1)
4. Tạo I = ma trận đơn vị p+1
   I[0, 0] = 0
5. A = X_biasᵀ @ X_bias + lam * I
6. b = X_biasᵀ @ y
7. beta_hat = solve_system(A, b)
8. y_hat = X_bias @ beta_hat
```

**Visualization bắt buộc:**

- Ridge trace: vẽ λ (log scale, trục x) vs từng `beta_hat[j]` (trục y) - quan sát hệ số co về 0 khi λ tăng

**Notebook demo (`part1_notebook.ipynb`):**

- Vẽ ridge trace với `lambda_grid = logspace(-3, 3, 50)`
- So sánh `beta_hat` của Ridge (λ tối ưu) vs OLS trên cùng dataset, in bảng

---

### F7. `lasso_fit(X, y, lam)`

**Mục đích:** Cài đặt Lasso bằng Coordinate Descent (không có closed-form).

**File:** `part1/ridge_lasso.py`

**Công thức tối ưu:** `min_β { ||y - Xβ||² + λ||β||₁ }`

**Input:** Tương tự `ridge_fit`.

**Output:** Tương tự `ridge_fit` + `n_iter` (số vòng lặp đến hội tụ).

**Thuật toán - Coordinate Descent:**

```
1. Chuẩn hóa X → X_scaled
2. Khởi tạo beta
3. Intercept = trung vi·ị y
4. y_centered = y - intercept

Repeat (tối đa max_iter = 1000 lần):
    beta_old = beta.copy()
    For j = 0, 1, ..., p-1:
        # Tính partial residual (loại trừ feature j)
        r_j = y_centered - X_scaled @ beta + X_scaled[:, j] * beta[j]

        # Tính rho_j = <X_j, r_j>
        rho_j = X_scaled[:, j] @ r_j

        # Soft-thresholding (nghiệm của bài toán coordinate đơn)
        # Với X đã chuẩn hóa: ||X_j||² = n
        z_j = sum(X_scaled[:, j] ** 2)   # = n nếu đã chuẩn hóa
        beta[j] = soft_threshold(rho_j, lam) / z_j

    # Kiểm tra hội tụ
    if max(abs(beta - beta_old)) < tol (= 1e-6):
        break

5. beta_hat = [intercept] + list(beta)

Hàm soft_threshold(rho, lam):
    if rho > lam:  return rho - lam
    elif rho < -lam: return rho + lam
    else: return 0.0
```

**Visualization bắt buộc:**

- Lasso path: vẽ λ (log scale, trục x) vs từng `beta_hat[j]` - quan sát hệ số (về đúng 0) khi λ tăng

**Notebook demo (`part1_notebook.ipynb`):**

- Vẽ lasso path, chỉ ra tại λ nào một số hệ số bị đưa về đúng 0 (sparse solution)
- So sánh `n_iter` hội tụ ở các mức λ khác nhau

---

### F8. `residual_plots(y, y_hat, X=None)`

**Mục đích:** Vẽ 4 biểu đồ chuẩn đoán phần dư.

**File:** `part1/residual_analysis.py`

**Input:**

| Tham số | Kiểu | Mô tả |
| --- | --- | --- |
| `y` | `list[float] (n,)` | Ground truth |
| `y_hat` | `list[float] (n,)` | Fitted values |
| `X` | `list[float] (n,p)` hoặc `None` | Cần cho Cook's Distance |

**Output:** `matplotlib.figure.Figure` (4 subplots), + `dict` chứa giá trị số liệu.

**4 Biểu đồ:**

| # | Tên | Trục X | Trục Y | Cần kiểm tra |
| --- | --- | --- | --- | --- |
| 1 | Residuals vs Fitted | `y_hat` | `residuals` | Không có pattern → tuyến tính OK |
| 2 | Q-Q Plot | Quantile lý thuyết Normal | Quantile thực nghiệm residuals | Điểm nằm trên đường → Chuẩn OK |
| 3 | Scale-Location | `y_hat` | `sqrt( | residuals |
| 4 | Cook's Distance | Index quan sát | Cook's D | Điểm > 4/n là influential |

**Thuật toán Cook's Distance:**

```
residuals = y - y_hat
sigma2 = RSS / (n - p - 1)
H = hat_matrix(X)['H']
h_ii = diag(H)          # Leverage values

# Công thức Cook's Distance:
cooks_d = (residuals**2 * h_ii) / (sigma2 * (p+1) * (1 - h_ii)**2)
threshold = 4 / n           # Ngưỡng thông dụng
influential = where(cooks_d > threshold)[0]
```

**Thuật toán Q-Q Plot:**

```
residuals_std = (residuals - mean(residuals)) / std(residuals)
n = len(residuals_std)
theoretical_quantiles = scipy.stats.norm.ppf((arange(1, n+1) - 0.5) / n)
empirical_quantiles = sort(residuals_std)
plt.scatter(theoretical_quantiles, empirical_quantiles)
plt.plot([-3, 3], [-3, 3], 'r--')  # Đường lý tưởng
```

---

### F9. `kfold_cv(X, y, k, model_fn, **model_kwargs)`

**Mục đích:** Cài đặt k-Fold Cross-Validation từ đầu để chọn hyperparameter.

**File:** `part1/cross_validation.py`

**Input:**

| Tham số | Kiểu | Mô tả |
| --- | --- | --- |
| `X` | `list[float] (n, p)` | Features |
| `y` | `list[float] (n,)` | Target |
| `k` | `int` | Số fold (khuyến nghị 5 hoặc 10) |
| `model_fn` | `callable` | Hàm fit: `model_fn(X_train, y_train, **kwargs) → dict` |
| `**model_kwargs` | `dict` | Tham số truyền vào `model_fn` (ví dụ: `lam=0.1`) |

**Output:** `dict`

| Key | Mô tả |
| --- | --- |
| `cv_scores` | `list[float] (k,)` - MSE của mỗi fold |
| `mean_cv_score` | `float` - CV score trung bình |
| `std_cv_score` | `float` - Độ lệch chuẩn |

**Thuật toán:**

```
1. Tạo indices = arange(n)
2. Shuffle: rng.shuffle(indices)        # Dùng rng với RANDOM_STATE
3. Chia thành k fold bằng nhau (dùng array_split)

For i = 0, 1, ..., k-1:
    val_idx   = fold[i]
    train_idx = concat(fold[j] for j ≠ i)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]

    result = model_fn(X_train, y_train, **model_kwargs)

    # Predict trên val - cần dùng thông số từ fit (mean/std của train)
    y_pred_val = predict(X_val, result)

    cv_scores[i] = mean((y_val - y_pred_val)**2)   # MSE

return mean(cv_scores), std(cv_scores), cv_scores
```

**Ứng dụng chọn λ tối ưu:**

```python
lambda_grid = logspace(-3, 3, 50)
cv_means = []
for lam in lambda_grid:
    score = kfold_cv(X, y, k=5, model_fn=ridge_fit, lam=lam)['mean_cv_score']
    cv_means.append(score)

lambda_opt = lambda_grid[argmin(cv_means)]
# Plot: lambda_grid vs cv_means (log scale trục x)
```

---

### F10. `Monte Carlo - Minh Họa Gauss–Markov`

**Mục đích:** Mô phỏng để chứng minh `E[β̂] = β` và OLS có phương sai nhỏ nhất (BLUE).

**File:** `part1/gauss_markov_demo.py`

**Thông số mô phỏng (phải ghi rõ trong notebook):**

```python
N_SIM     = 1000      # Số lần mô phỏng
N_OBS     = 100       # n quan sát mỗi lần
TRUE_BETA = ([2.0, -1.5, 0.8])   # β thực (gồm intercept)
TRUE_SIGMA = 1.0      # Độ lệch chuẩn nhiễu
```

**Thuật toán:**

```
X_fixed = rng.normal(0, 1, (N_OBS, 2))  # X cố định cho mọi sim

beta_ols_list = []
For sim in range(N_SIM):
    epsilon = rng.normal(0, TRUE_SIGMA, N_OBS)
    y_sim = X_bias_fixed @ TRUE_BETA + epsilon
    result = ols_fit(X_fixed, y_sim)
    beta_ols_list.append(result['beta_hat'])

beta_ols_arr = np.array(beta_ols_list)   # shape (1000, 3)

# Kiểm tra unbiasedness:
E_beta_hat = beta_ols_arr.mean(axis=0) # → Phải gần bằng nhau

# Kiểm tra variance nhỏ nhất (so với ước lượng tuyến tính khác):
# Định nghĩa 1 ước lượng khác: β_alt = (XᵀX + 0.1I)⁻¹Xᵀy (Ridge nhỏ)
# → Var(β̂_OLS) ≤ Var(β̂_alt) theo từng chiều
```

**Visualization bắt buộc:**

- Histogram của `beta_ols_arr[:, j]` cho từng j, vẽ đường thẳng đứng tại `TRUE_BETA[j]`.
- Bảng so sánh `Mean(β̂)`, `Var(β̂)` giữa OLS và ít nhất 1 estimator khác

---

### F11. `Notebook`

- **Import thư viện và code của nhóm:** Import các hàm từ `ols_implementation.py`, `ridge_lasso.py`, v.v.
- **Minh họa nghiệm OLS (F1 - F4):**
    - Tạo ma trận X và vector Y
    - Gọi hàm `ols_fit` của nhóm in ra vector $\hat{\boldsymbol{\beta}}$.
    - Gọi hàm `LinearRegression` của `sklearn` in ra vector $\hat{\boldsymbol{\beta}}_{sklearn}$.
    - Dùng Markdown và lệnh kiểm tra để kết luận hai vector này giống nhau.
- **Kiểm chứng các tính chất Toán học:**
    - Tính ma trận Hat $\mathbf{H}$ Gọi code kiểm chứng $\mathbf{H}^2 = \mathbf{H}$ (tính chất lũy đẳng - idempotent) và in kết quả True/False.
    - Vẽ Histogram của các giá trị riêng (eigenvalues) của $\mathbf{H}$ để chứng minh nó chỉ chứa 0 và 1.
- **Minh họa định lý Gauss-Markov:**
    - Gọi hàm mô phỏng Monte Carlo.
    - Vẽ biểu đồ Histogram phân phối của $\hat{\boldsymbol{\beta}}$ sau 1000 lần chạy, chứng minh giá trị trung bình $E[\hat{\boldsymbol{\beta}}]$ nằm ngay chính giữa trục giá trị thực $\boldsymbol{\beta}$ .

## PHẦN 2 - Ứng Dụng Dữ Liệu Thực Tế

---

### T1. `Chọn & Load Dataset`

**File:** `README.md` + `part2/part2_notebook.ipynb`

**Tiêu chí dataset (bắt buộc thỏa đồng thời):**

- Dữ liệu thực (không dùng Iris, Boston Housing từ sklearn)
- Có ít nhất 1 cột missing ≥ 5% dữ liệu
- Target là biến liên tục (regression)
- n ≥ 200, p ≥ 3

**Deliverable T1:**

```python
# Trong notebook, cell đầu tiên phải có:
df = pd.read_csv("data/<dataset>.csv")
print(f"Shape: {df.shape}")
print(f"Target: <tên cột target>")
print(f"Missing: {df.isnull().sum().sum()} giá trị")
```

---

### T2. `EDA - Exploratory Data Analysis`

**File:** Notebook `part2_notebook.ipynb` (section EDA)

**Danh sách việc cần làm (theo thứ tự):**

| # | Việc | Hàm/method | Output |
| --- | --- | --- | --- |
| 2.1 | Thống kê mô tả | `df.describe(include='all')` | Bảng mean/std/min/max/quartile |
| 2.2 | Missing value report | Tự viết hàm `missing_report(df)` | DataFrame: cột, số missing, tỉ lệ % |
| 2.3 | Duplicate check | `df.duplicated().sum()` | In số dòng trùng |
| 2.4 | Histogram mỗi biến số | `df.hist(bins=30, figsize=...)` | Grid histogram |
| 2.5 | Boxplot mỗi biến số | Seaborn boxplot | Phát hiện outlier thô |
| 2.6 | Correlation heatmap | `sns.heatmap(df.corr())` | Heatmap |
| 2.7 | Scatter target vs features | Top 5 features tương quan cao | 5 scatter plots |
| 2.8 | Phát hiện outlier IQR | Tự viết `detect_outliers_iqr(df, col)` | Dict: {cột: danh sách index} |
| 2.9 | Phát hiện outlier z-score | `(x - mean) / std > 3` | Tương tự |

**Hàm `missing_report(df)`:**

```
Input:  df - pd.DataFrame
Output: pd.DataFrame, columns=['column', 'n_missing', 'pct_missing'],
        sắp xếp giảm dần theo pct_missing,
        chỉ giữ các cột có missing > 0
```

**Hàm `detect_outliers_iqr(series)`:**

```
Input:  series - pd.Series (1 cột số)
Output: pd.Index - index của các hàng là outlier
Công thức:
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = series[(series < lower) | (series > upper)].index
```

---

### T3. `DataPipeline class`

**File:** `part2/data_pipeline.py`

**Mục đích:** Pipeline tiền xử lý hoàn chỉnh, fit trên train, transform trên test (không leakage).

**Interface bắt buộc:**

```python
class DataPipeline:
    def __init__(
        self,
        numeric_cols: list[str],
        categorical_cols: list[str],
        target_col: str,
        missing_strategy: str = 'median',    # 'mean' | 'median' | 'mode' | 'knn'
        outlier_strategy: str = 'winsorize', # 'winsorize' | 'remove' | 'none'
        scale: bool = True,
        poly_degree: int = 1,                # 1 = không tạo polynomial features
        random_state: int = 42
    ): ...

    def fit(self, df_train: pd.DataFrame) -> 'DataPipeline':
        """Học tham số từ train set. Return self để chain."""

    def transform(self, df: pd.DataFrame) -> tuple[np.list[float], np.ndarray]:
        """
        Áp dụng transformation đã fit.
        Return: (X, y) - np.ndarray
        """

    def fit_transform(self, df_train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Shortcut: fit rồi transform trên cùng 1 df."""

    def get_feature_names(self) -> list[str]:
        """Trả về danh sách tên features sau transform (dùng cho visualization)."""
```

**Thứ tự xử lý bên trong `fit`:**

```
1. Tách X và y từ df_train
2. [Missing] Tính thống kê imputation từ X_train:
   - 'median': lưu median của từng numeric col
   - 'mean':   lưu mean
   - 'knn':    fit KNNImputer (k=5) từ sklearn (phép dùng)
3. [Outlier] Tính ngưỡng Winsorize từ train (1st và 99th percentile mỗi col)
4. [Encoding] One-hot encoding cho categorical cols:
   - Lưu danh sách categories từ train để đảm bảo test có cùng cột
   - Dùng pd.get_dummies hoặc sklearn OneHotEncoder
5. [Scale] Tính mean_X và std_X trên train (sau khi impute + encode)
6. [Polynomial] Lưu degree (transform sẽ dùng PolynomialFeatures từ sklearn)
```

**Thứ tự xử lý bên trong `transform`:**

```
1. Impute missing dùng thống kê đã lưu từ fit
2. Winsorize dùng ngưỡng từ fit
3. One-hot encode dùng categories từ fit (unseen categories → 0)
4. Standardize: X_scaled = (X - mean_X) / std_X
5. Polynomial features (nếu degree > 1)
6. Trả về (X_scaled_np, y_np)
```

**Unit tests `DataPipeline`:**

```python
def test_no_leakage():
    """mean/std được tính từ train, không từ test."""
    # Tạo train với mean=0, test với mean=100
    # Sau transform, X_test phải có giá trị >> 0 (không normalize bằng mean của test)

def test_fit_transform_consistent():
    """fit_transform(train) == fit(train).transform(train)."""
```

---

### T4. `Xây Dựng và So Sánh Mô Hình`

**File:** `part2/model_comparison.py`

**Bắt buộc xây dựng ít nhất 3 mô hình:**

| Mô hình | Hàm dùng | Ghi chú |
| --- | --- | --- |
| OLS đầy đủ | `ols_fit` (từ part1) | Tất cả features sau pipeline |
| OLS chọn biến | `ols_fit` + loại biến | Loại biến có p-value > 0.05 hoặc VIF > 10 |
| Ridge (λ tối ưu) | `ridge_fit` (từ part1) | λ chọn qua `kfold_cv` k=5 |
| Lasso (λ tối ưu) | `lasso_fit` (từ part1) | λ chọn qua `kfold_cv` k=5 |

**Quy trình chuẩn cho mỗi mô hình:**

```
1. Split: train/test = 80/20, shuffle=True, random_state=RANDOM_STATE
2. Fit pipeline chỉ trên train
3. Transform cả train và test
4. Fit model trên (X_train, y_train)
5. Predict trên X_test → y_pred
6. Tính metrics: MAE, RMSE, R² trên test set
7. Lưu vào dict để so sánh
```

**Hàm `compare_models(results_dict)` → `pd.DataFrame`:**

```
Input: dict {'Model Name': {'MAE': ..., 'RMSE': ..., 'R2': ..., ...}}
Output: DataFrame index=model_name, columns=['MAE', 'RMSE', 'R2_test']
        Sắp xếp theo RMSE tăng dần
```

---

### T5. `Notebook`

**File:** `part2/part2_notebook.ipynb`

---

### T6. `Kỹ Thuật Nâng Cao (Bonus)`

**File:** `part2/advanced_methods.py`

**Chọn 1 trong 2 (hoặc cả 2):**

### Option A - Kernel Ridge Regression

**Công thức:** `ŷ(x) = k(x)ᵀ (K + λI)⁻¹ y`

```python
def rbf_kernel(X1, X2, length_scale=1.0):
    """Tính Gram matrix K[i,j] = exp(-||X1[i] - X2[j]||² / (2*l²))."""
    # Efficient implementation dùng broadcasting:
    diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]   # (n1, n2, p)
    sq_dist = np.sum(diff**2, axis=-1)                     # (n1, n2)
    return np.exp(-sq_dist / (2 * length_scale**2))

def kernel_ridge_fit(X_train, y_train, lam, length_scale):
    K = rbf_kernel(X_train, X_train, length_scale)
    alpha = np.linalg.solve(K + lam * np.eye(len(y_train)), y_train)
    return {'alpha': alpha, 'X_train': X_train, 'length_scale': length_scale}

def kernel_ridge_predict(model, X_test):
    K_test = rbf_kernel(X_test, model['X_train'], model['length_scale'])
    return K_test @ model['alpha']
```

### Option B - Bayesian Linear Regression

```python
def bayesian_lr_fit(X, y, sigma2, m0, S0):
    """
    Prior: β ~ N(m0, S0)
    Likelihood: y|X,β ~ N(Xβ, σ²I)
    Returns posterior parameters (m_n, S_n).
    """
    X_bias = np.column_stack([np.ones(len(X)), X])
    S0_inv = np.linalg.inv(S0)
    S_n_inv = S0_inv + (1/sigma2) * X_bias.T @ X_bias
    S_n = np.linalg.inv(S_n_inv)
    m_n = S_n @ (S0_inv @ m0 + (1/sigma2) * X_bias.T @ y)
    return {'m_n': m_n, 'S_n': S_n}

def bayesian_lr_predict(model, X_new, credible_interval=0.95):
    """Trả về mean và credible interval của prediction."""
    X_bias = np.column_stack([np.ones(len(X_new)), X_new])
    y_mean = X_bias @ model['m_n']
    y_var = np.array([x @ model['S_n'] @ x for x in X_bias])
    z = scipy.stats.norm.ppf((1 + credible_interval) / 2)
    return y_mean, y_mean - z*np.sqrt(y_var), y_mean + z*np.sqrt(y_var)
```

---

## BÁO CÁO

### **B1. `Report`**

**File:** `report/report.pdf` (LaTeX hoặc Markdown → PDF)

**Checklist cấu trúc báo cáo:**

| # | Section | Nội dung tối thiểu |
| --- | --- | --- |
| 1 | Trang bìa | Họ tên, MSSV, nhóm, GV, ngày |
| 2 | Mục lục | Auto-generated |
| 3 | Phần 1 | Lý thuyết + công thức + chứng minh + code snippet + kết quả minh họa |
| 4 | Phần 2 | Mô tả data + EDA summary + pipeline + bảng so sánh mô hình + phân tích phần dư |
| 5 | Kết luận | Tóm tắt kết quả, bài học, hướng mở rộng |
| 6 | Tài liệu tham khảo | ≥ 5 tài liệu |
| 7 | Phụ lục | Bảng số liệu bổ sung (nếu có) |

**Mọi biểu đồ trong báo cáo phải có:**

- Tiêu đề (title)
- Nhãn trục X và Y (có đơn vị nếu có)
- Legend (nếu nhiều đường)
- Caption bên dưới hình giải thích ý nghĩa

---

## PHÂN CÔNG

### 1. Phần 1

| TT | Thành viên | Nhiệm vụ | File | Chi tiết | Deadline | Deadline |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Triết | F1 `ols_fit`, F2 `hat_matrix`, F3 `model_metrics` | `ols_implementation.py` | Normal equations, hat matrix idempotent/symmetric, RSS/TSS/R²/Adj-R²/F-stat + unit test + vẽ histogram eigenvalues của H, heatmap của H (n ≤ 20). | 15/5/2026 | 20h 7/5/2026 |  |
| 2 | Lợi | F4 `coef_inference`, F5 `vif`, F10 Monte Carlo | `ols_implementation.py`, `gauss_markov_demo.py` | t-stat/p-value/CI 95%, VIF đa cộng tuyến, Gauss–Markov N_SIM=1000 + unit test + vẽ histogram β̂ mỗi chiều với đường thẳng đứng tại TRUE_BETA. | 15/5/2026 | 20h 7/5/2026 |  |
| 3 | Nguyên | F6 `ridge_fit`, F7 `lasso_fit`, F8 `residual_plots`, F9 `kfold_cv` | `ridge_lasso.py`, `residual_analysis.py`, `cross_validation.py` | Ridge closed-form, Lasso coordinate descent, 4 biểu đồ chẩn đoán, k-Fold CV + unit test + vẽ ridge trace (λ vs coef), vẽ λ vs CV score (log scale). | 15/5/2026 | 20h 7/5/2026 |  |
| 4 | Khải + Minh Phan | F11 - Notebook demo của các phần F1–10 | `part1_notebook.ipynb, README.md` | Với mỗi hàm: (1) cell markdown trình bày công thức + ý nghĩa lý thuyết, (2) cell code gọi hàm trên synthetic data, (3) cell hiển thị kết quả/plot. Cụ thể: `ols_fit` → in beta_hat, sigma2, so sánh với numpy; `hat_matrix` → hiển thị heatmap H + histogram eigenvalues; `model_metrics` → in bảng RSS/TSS/R²/F-stat; `coef_inference` → in DataFrame t-stat/p-value/CI; `vif` → in bảng VIF, nhận xét đa cộng tuyến; `ridge_fit` → hiển thị ridge trace (λ vs coef); `lasso_fit` → so sánh hệ số Ridge vs Lasso; `residual_plots` → hiển thị đủ 4 biểu đồ + nhận xét từng biểu đồ; `kfold_cv` → hiển thị λ vs CV score, in λ tối ưu; F10 Monte Carlo → hiển thị histogram β̂ + bảng so sánh Mean/Var OLS vs estimator khác, kết luận Gauss–Markov. | 27/5/2026 | 20h 11/5/2026 |  |

### 2. Phần 2

| TT | Thành viên | Nhiệm vụ | File | Chi tiết | Deadline |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | Nguyên | T3 `DataPipeline` + `compare_models` | `data_pipeline.py`, `model_comparison.py` | Class fit/transform/fit_transform (impute, winsorize, encode, scale, poly); hàm `compare_models` → DataFrame MAE/RMSE/R²; no-leakage unit test. | 18/5/2026 |  |  |
| 7 | Triết | T4 Train & đánh giá 4 mô hình | `model_comparison.py` | Dùng Pipeline của Số 3: train OLS đầy đủ, OLS chọn biến (p-value/VIF), Ridge & Lasso (λ qua CV); predict test set; Shapiro-Wilk, Breusch-Pagan. | 25/5/2026 |  |  |
| 8 | Lợi | T7 Nâng cao (bonus) + QA | `advanced_methods.py` | Chọn Kernel Ridge (RBF) hoặc Bayesian LR; so sánh với mô hình gốc; rà soát toàn bộ unit test, đảm bảo `RANDOM_STATE = 42` nhất quán. | 25/5/2026 |  |  |
| 9 | Khải | T1 Dataset, T2 EDA, T6 Feature Importance, `part2_notebook.ipynb` | `part2_notebook.ipynb`, `data/` | Chọn & load dataset (≥200 obs, ≥3 features, ≥5% missing); viết `missing_report`, `detect_outliers_iqr`; vẽ histogram/boxplot/heatmap/scatter top-5; bar chart feature importance (top 15); tổng hợp toàn bộ notebook Part 2. | 17/5/2026 |  |  |
| 10 | Minh Phan + Nguyên | T5 Phân tích kết quả + Hoàn thiện báo cáo | `report/report.tex`, `part2_notebook.ipynb` | Nhận kết quả từ Số 4: viết bảng so sánh MAE/RMSE/R², phân tích 4 biểu đồ phần dư, nhận xét Gauss–Markov; điền kết quả vào LaTeX, viết Kết luận, caption ảnh, format toàn bộ PDF. BibTeX ≥ 5 tài liệu. | 29/5/2026 |  |  |

### Git Branch

```
main
├── feat/part1-ols-metrics     ← F1, F2, F3        (1)
├── feat/part1-inference-gm    ← F4, F5, F10       (2)
├── feat/part1-ridge-cv        ← F6, F7, F8, F9    (3)
├── feat/part1-notebook        ← F11               (4 + 5)
├── feat/part2-pipeline        ← T3, compare       (6)
├── feat/part2-models          ← T4                (7)
├── feat/part2-advanced        ← T6                (8)
└── feat/part2-notebook        ← T5                (9 + 10)
```