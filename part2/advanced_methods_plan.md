# Kế hoạch triển khai `part2/advanced_methods.py`

## 1. Mục tiêu

Triển khai phần kỹ thuật nâng cao cho Phần 2 của đồ án, tập trung vào **Kernel Ridge Regression với RBF kernel** theo công thức trong tài liệu:

```text
y_hat(x) = k(x)^T (K + lambda I)^(-1) y
K_ij = k(x_i, x_j)
k_RBF(x, x') = exp(-||x - x'||^2 / (2 * length_scale^2))
```

Yêu cầu bắt buộc cho file `advanced_methods.py`:

- Không dùng `numpy`, không dùng broadcasting, không dùng `np.linalg.solve`.
- Không dùng `sklearn` để cài đặt thuật toán chính.
- Có thể dùng lại các hàm tự cài trong `utils.py` và Part 1.
- Input chính là dữ liệu đã được `DataPipeline` xử lý: `list[list[float]]` cho `X`, `list[float]` cho `y`.
- Output phải dễ đưa vào `model_comparison.py` hoặc notebook để so sánh MAE, RMSE, R2 với OLS/Ridge/Lasso.

## 2. Căn cứ từ PDF và code hiện có

### Từ PDF

PDF yêu cầu Phần 2 có pipeline tiền xử lý, so sánh ít nhất 3 mô hình, đánh giá trên test set bằng MAE, RMSE, R2, và phần nâng cao bonus có thể chọn **Kernel Regression / Kernel Ridge Regression**.

Thuật toán cần bám:

1. Tạo Gram matrix `K` từ train set.
2. Giải hệ tuyến tính `(K + lambda I) alpha = y_train`.
3. Dự đoán test bằng `K_test @ alpha`, trong đó `K_test[i][j] = k(x_test_i, x_train_j)`.

### Từ `part2/data_pipeline.py`

`DataPipeline` đã xử lý khá nhiều việc trước khi đưa dữ liệu vào model:

- Đọc AirQualityUCI, bỏ cột rỗng `Unnamed: 15`, `Unnamed: 16`.
- Parse `Date` + `Time` thành `Date_Time`, sau đó loại cột datetime khỏi feature.
- Loại cột missing trên 70%.
- KNN imputation bằng Pandas cho numeric features.
- Winsorize theo ngưỡng 1% và 99%.
- One-hot encoding categorical columns.
- Standardize bằng mean/std của train set.
- Thêm polynomial features degree 2.
- Trả về `X_result = list[list[float]]`, `y_result = list[float]`.

Vì vậy `advanced_methods.py` không nên lặp lại tiền xử lý dữ liệu. File này chỉ nên nhận `X_train`, `y_train`, `X_test` đã sạch và đã scale.

### Từ Part 1 có thể dùng lại

- `utils.dot`, `utils.matvec`, `utils.identity_matrix` nếu cần nhân vector/matrix.
- `part1.ridge_lasso.solve_system` để giải hệ tuyến tính thay cho `np.linalg.solve`.
- `part1.ridge_lasso.make_lambda_grid` để tạo grid lambda logspace không NumPy.
- `part1.ols_implementation.model_metrics` có thể dùng cho mô hình tuyến tính, nhưng với kernel nên tự viết MAE/RMSE/R2 đơn giản trong `advanced_methods.py` hoặc để `model_comparison.py` xử lý chung.

Lưu ý: `part1.cross_validation.kfold_cv` chỉ hỗ trợ grid search 1 chiều (lambda). KRR cần grid search 2 chiều (lambda × length_scale), nên ta sẽ tự viết k-fold thuần Python trong `advanced_methods.py` để phù hợp.

## 3. Thiết kế API đề xuất

### 3.1. Hàm validation

```python
def validate_X(X: list[list[float]], name: str = "X") -> None
def validate_xy(X: list[list[float]], y: list[float]) -> None
def validate_positive(value: float, name: str) -> None
```

Mục đích:

- Bắt lỗi `X=[]`, `y=[]`.
- Kiểm tra số dòng `X` khớp `y`.
- Kiểm tra mọi row có cùng số feature.
- Kiểm tra `lam >= 0`.
- Kiểm tra `length_scale > 0`.

### 3.2. RBF kernel không NumPy

```python
def squared_distance(x1: list[float], x2: list[float]) -> float:
    ...

def rbf_kernel_value(
    x1: list[float],
    x2: list[float],
    length_scale: float = 1.0,
) -> float:
    ...

def rbf_kernel(
    X1: list[list[float]],
    X2: list[list[float]],
    length_scale: float = 1.0,
) -> list[list[float]]:
    ...
```

Thuật toán thay cho broadcasting NumPy:

```text
for i in range(len(X1)):
    row = []
    for j in range(len(X2)):
        dist2 = sum((X1[i][k] - X2[j][k]) ** 2 for k in range(p))
        row.append(exp(-dist2 / (2 * length_scale ** 2)))
    K.append(row)
```

Dùng `math.exp`. Không import `numpy`.

### 3.3. Kernel Ridge fit

```python
def kernel_ridge_fit(
    X_train: list[list[float]],
    y_train: list[float],
    lam: float,
    length_scale: float,
    jitter: float = 1e-10,
) -> dict:
    ...
```

Các bước:

1. Validate input.
2. Tính `K = rbf_kernel(X_train, X_train, length_scale)`.
3. Tạo `A = K + (lam + jitter) * I`.
4. Giải `A alpha = y_train` bằng `solve_system(A, y_train)` từ `part1.ridge_lasso`.
5. Tính `y_hat_train = matvec(K, alpha)` hoặc tự nhân list.
6. Trả về dict:

```python
{
    "model_type": "kernel_ridge_rbf",
    "alpha": alpha,
    "X_train": copied_X_train,
    "length_scale": length_scale,
    "lam": lam,
    "jitter": jitter,
    "y_hat": y_hat_train,
}
```

Ghi chú:

- `jitter` giúp hệ tuyến tính ổn định hơn khi `lam=0` hoặc Gram matrix gần suy biến.
- Nên copy `X_train` bằng `[row[:] for row in X_train]` để tránh người dùng sửa dữ liệu gốc làm model đổi theo.

### 3.4. Kernel Ridge predict

```python
def kernel_ridge_predict(
    model: dict,
    X_test: list[list[float]],
) -> list[float]:
    ...
```

Cách làm ưu tiên: dự đoán streaming từng dòng để tránh tạo `K_test` quá lớn.

```text
for x in X_test:
    pred = 0.0
    for j in range(len(model["X_train"])):
        pred += rbf_kernel_value(x, model["X_train"][j], length_scale) * alpha[j]
    y_pred.append(pred)
```

Ưu điểm: vẫn đúng với công thức `K_test @ alpha`, nhưng tiết kiệm bộ nhớ hơn.

### 3.5. Metrics thuần Python

```python
def regression_metrics(
    y_true: list[float],
    y_pred: list[float],
) -> dict:
    ...
```

Trả về:

```python
{
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
    "RSS": rss,
    "TSS": tss,
}
```

Công thức:

```text
MAE = mean(abs(y_i - y_hat_i))
RMSE = sqrt(mean((y_i - y_hat_i)^2))
R2 = 1 - RSS / TSS
```

Không dùng `numpy.mean`, `numpy.sqrt`; dùng `sum`, `len`, `math.sqrt`.

### 3.6. K-fold CV thuần Python cho KRR

```python
def kfold_indices(
    n: int,
    k: int,
    seed: int = RANDOM_STATE,
) -> list[list[int]]:
    ...

def kernel_ridge_cv(
    X: list[list[float]],
    y: list[float],
    lambda_grid: list[float],
    length_scale_grid: list[float],
    k: int = 5,
    seed: int = RANDOM_STATE,
) -> dict:
    ...
```

Thuật toán:

1. Tạo `indices = list(range(n))`.
2. Shuffle bằng `random.Random(seed).shuffle(indices)`.
3. Chia fold bằng slicing thuần Python:

```text
fold_sizes = [n // k] * k
for i in range(n % k):
    fold_sizes[i] += 1
```

4. Với mỗi cặp `(lam, length_scale)`:
   - Lặp qua từng fold.
   - Train trên `k-1` fold.
   - Predict trên validation fold.
   - Tính RMSE hoặc MSE.
5. Chọn cặp có mean validation MSE nhỏ nhất.
6. Trả về:

```python
{
    "best_lam": best_lam,
    "best_length_scale": best_length_scale,
    "best_cv_score": best_score,
    "cv_results": [
        {
            "lam": lam,
            "length_scale": length_scale,
            "fold_scores": [...],
            "mean_score": ...,
            "std_score": ...,
        },
        ...
    ],
}
```

`std_score` cũng tự tính bằng công thức phương sai mẫu (sample std, chia n-1):

```text
std = sqrt(sum((score - mean)^2) / (len(scores) - 1))
```

## 4. Vấn đề hiệu năng cần xử lý

AirQualityUCI có khoảng 9k dòng raw. Kernel Ridge dùng ma trận `K` kích thước `n_train x n_train`, nên nếu train toàn bộ:

- Bộ nhớ tăng theo `O(n^2)`.
- Giải hệ tuyến tính tăng theo `O(n^3)`.
- Với list thuần Python, `9000 x 9000` là không thực tế.

Kế hoạch xử lý:

1. Thêm guard trong `kernel_ridge_fit`:

```python
if len(X_train) > 2000:
    raise ValueError("Kernel Ridge is O(n^3); pass a smaller training subset.")
```

2. Tạo helper lấy subset tái lập được:

```python
def sample_rows(
    X: list[list[float]],
    y: list[float],
    max_rows: int,
    seed: int = RANDOM_STATE,
) -> tuple[list[list[float]], list[float]]:
    ...
```

3. Trong notebook hoặc `model_comparison.py`, dùng Kernel Ridge trên subset train, ví dụ `max_rows=800` hoặc `1000`, rồi đánh giá trên test set đầy đủ hoặc test subset.

4. Ghi rõ trong báo cáo: KRR là mô hình nâng cao bonus, giới hạn bởi độ phức tạp Gram matrix; dùng subset là quyết định có căn cứ về tài nguyên.

## 5. Tích hợp với `DataPipeline`

Luồng đề xuất trong notebook:

```python
from part2.data_pipeline import DataPipeline
from part2.advanced_methods import (
    train_test_split,
    sample_rows,
    kernel_ridge_cv,
    kernel_ridge_fit,
    kernel_ridge_predict,
    regression_metrics,
)

pipeline = DataPipeline(target_col="CO(GT)")
df = pipeline.load_data("part2/data/AirQualityUCI.csv")

# Split train/test trước khi fit pipeline để tránh leakage.
df_train, df_test = train_test_split(df, test_size=0.2, seed=42)

X_train_processed, y_train = pipeline.fit_transform(
    df_train.drop(columns=["CO(GT)"]),
    df_train["CO(GT)"],
)
X_test_processed, y_test = pipeline.transform(
    df_test.drop(columns=["CO(GT)"]),
    df_test["CO(GT)"],
)

X_krr, y_krr = sample_rows(X_train_processed, y_train, max_rows=800)

cv = kernel_ridge_cv(
    X_krr,
    y_krr,
    lambda_grid=[0.001, 0.01, 0.1, 1.0, 10.0],
    length_scale_grid=[0.5, 1.0, 2.0, 5.0],
    k=5,
)

model = kernel_ridge_fit(
    X_krr,
    y_krr,
    lam=cv["best_lam"],
    length_scale=cv["best_length_scale"],
)

y_pred = kernel_ridge_predict(model, X_test_processed)
metrics = regression_metrics(y_test, y_pred)
```

Lưu ý quan trọng:

- `DataPipeline` đã standardize features, nên RBF distance có ý nghĩa hơn.
- `DataPipeline` đang thêm polynomial degree 2, số feature có thể tăng mạnh. KRR vẫn chạy được nhưng tính distance sẽ đắt hơn; cần giữ subset vừa phải.
- Nếu muốn so sánh công bằng với OLS/Ridge/Lasso, dùng cùng train/test split và cùng pipeline đã fit trên train.

## 6. Grid siêu tham số đề xuất

Vì dữ liệu đã standardize:

- `lambda_grid`: `[0.001, 0.01, 0.1, 1.0, 10.0]`
- `length_scale_grid`: `[0.5, 1.0, 2.0, 5.0, 10.0]`

Nếu CV quá chậm:

- Giảm `max_rows` xuống `500`.
- Giảm `k` từ `5` xuống `3`.
- Giảm grid còn 3 x 3:

```python
lambda_grid = [0.01, 0.1, 1.0]
length_scale_grid = [1.0, 2.0, 5.0]
```

## 7. Unit tests cần viết

Nên bổ sung test cho `advanced_methods.py`, có thể đặt trong `part2/test_advanced_methods.py` hoặc thêm runner trong chính file.

### Test 1: RBF kernel cơ bản

Input:

```python
X = [[0.0], [1.0]]
K = rbf_kernel(X, X, length_scale=1.0)
```

Kỳ vọng:

- `K[0][0] == 1.0`
- `K[1][1] == 1.0`
- `K[0][1] == K[1][0]`
- `K[0][1]` xấp xỉ `exp(-0.5)`

### Test 2: Validate length_scale

`rbf_kernel([[1.0]], [[1.0]], length_scale=0.0)` phải raise `ValueError`.

### Test 3: Fit/predict shape

Input:

```python
X = [[0.0], [1.0], [2.0]]
y = [0.0, 1.0, 4.0]
model = kernel_ridge_fit(X, y, lam=0.01, length_scale=1.0)
y_hat = kernel_ridge_predict(model, X)
```

Kỳ vọng:

- `len(model["alpha"]) == len(y)`
- `len(y_hat) == len(y)`
- MSE nhỏ hơn một ngưỡng hợp lý, ví dụ `< 0.1`.

### Test 4: `sample_rows` tái lập

Gọi hai lần cùng seed phải trả cùng subset.

### Test 5: CV trả về cặp tham số từ grid

`kernel_ridge_cv` phải trả `best_lam in lambda_grid`, `best_length_scale in length_scale_grid`, và số record `cv_results == len(lambda_grid) * len(length_scale_grid)`.

## 8. Checklist triển khai

- [ ] Tạo import tối thiểu: `math`, `random`, `RANDOM_STATE`, `solve_system`.
- [ ] Viết `train_test_split` helper dùng Pandas.
- [ ] Viết validation helpers.
- [ ] Viết `squared_distance`, `rbf_kernel_value`, `rbf_kernel`.
- [ ] Viết helper cộng lambda vào đường chéo Gram matrix.
- [ ] Viết `kernel_ridge_fit`.
- [ ] Viết `kernel_ridge_predict` dạng streaming.
- [ ] Viết `regression_metrics`.
- [ ] Viết `sample_rows`.
- [ ] Viết k-fold thuần Python và `kernel_ridge_cv`.
- [ ] Viết test tối thiểu cho kernel, fit/predict, CV.
- [ ] Chạy test liên quan.
- [ ] Tích hợp vào notebook/model comparison để so sánh với OLS/Ridge/Lasso.

## 9. Các lỗi dễ gặp và cách tránh

- `length_scale` quá nhỏ: kernel gần identity, dễ overfit. Dùng CV để chọn.
- `length_scale` quá lớn: kernel gần toàn 1, mô hình quá mượt/underfit.
- `lambda=0`: có thể làm hệ suy biến. Dùng `jitter=1e-10`.
- Train set quá lớn: `solve_system` Gauss-Jordan trên Gram matrix rất chậm. Bắt lỗi khi `n_train > 2000` và dùng `sample_rows`.
- Data leakage: luôn split train/test trước, sau đó `pipeline.fit` trên train và `pipeline.transform` trên test.
- Dùng nhầm NumPy gián tiếp: không gọi `part1.cross_validation.kfold_cv` trong advanced methods vì hàm đó import NumPy.

## 10. Tiêu chí hoàn thành

`advanced_methods.py` được xem là hoàn thành khi:

- Không có `import numpy`.
- Chạy được KRR RBF bằng list thuần Python.
- Có fit, predict, CV chọn `lambda` và `length_scale`.
- Có metrics MAE/RMSE/R2 để đưa vào bảng so sánh.
- Có test cho kernel và fit/predict.
- Notebook hoặc script Part 2 có thể dùng model này trên output của `DataPipeline`.
- Báo cáo giải thích được vì sao KRR chỉ chạy trên subset do độ phức tạp `O(n^2)` bộ nhớ và `O(n^3)` thời gian.
