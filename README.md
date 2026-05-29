# Đồ án 2: Data Fitting và Phương pháp OLS 

- **Môn học:** Toán ứng dụng và thống kê (Applied Mathematics and Statistics)
- **GVHD:** ThS. Lê Nhựt Nam, ThS. Võ Nam Thục Đoan
- **Nhóm thực hiện:** Nhóm 14 (24CTT2)

## Mục lục
- [Tổng quan dự án](#tổng-quan-dự-án)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Chi tiết các phần thực hiện](#chi-tiết-các-phần-thực-hiện)
  - [Phần 1: Cài đặt thuật toán cốt lõi từ đầu (Lý thuyết)](#phần-1-cài-đặt-thuật-toán-cốt-lõi-từ-đầu-lý-thuyết)
  - [Phần 2: Data Fitting trên Dữ liệu thực tế](#phần-2-data-fitting-trên-dữ-liệu-thực-tế)
- [Thư viện đã dùng](#thư-viện-đã-dùng)
- [Chi tiết các file tiện ích](#chi-tiết-các-file-tiện-ích)
- [Hướng dẫn cài đặt môi trường](#hướng-dẫn-cài-đặt-môi-trường)
- [Cách chạy thử nghiệm](#cách-chạy-thử-nghiệm)
- [Tác giả & Đóng góp](#tác-giả--đóng-góp)

## Tổng quan dự án
Đây là báo cáo mã nguồn cho Đồ án 2 thuộc chuyên đề Data Fitting và phương pháp Bình phương tối thiểu (OLS). 

Chất lượng không khí tại các khu vực đô thị ảnh hưởng nghiêm trọng đến sức khỏe con người, trong đó `CO` (Carbon Monoxide) là một trong những loại khí độc nguy hiểm nhất. Việc đo đạc trực tiếp khí `CO` thường rất tốn kém, do đó một hướng tiếp cận tối ưu là sử dụng các mạng lưới cảm biến đa tinh thể oxit kim loại (chi phí thấp) kết hợp với máy học để ước lượng nồng độ `CO`.

Mục tiêu của dự án này bao gồm hai khía cạnh chính:
1. **Khía cạnh Toán học & Nền tảng:** Tự xây dựng (from scratch) các thuật toán cốt lõi của đại số tuyến tính phục vụ cho OLS (như giải hệ phương trình tuyến tính, tính toán ma trận Hat, phân tích phần dư, và điều chuẩn Ridge/Lasso) để hiểu sâu sắc bản chất và các giả định của mô hình (Định lý Gauss-Markov).
2. **Khía cạnh Ứng dụng Thực tế:** Xây dựng một luồng xử lý dữ liệu (Data Pipeline) hoàn chỉnh để giải quyết các vấn đề gai góc của dữ liệu cảm biến thực tế (Dataset AirQualityUCI) bao gồm: tỷ lệ dữ liệu khuyết cực lớn, hiện tượng đa cộng tuyến (multicollinearity) và sự tồn tại của ngoại lai. Từ đó nhóm tiến hành huấn luyện, chẩn đoán phần dư và so sánh hiệu năng của hàng loạt các mô hình dự đoán (từ OLS thuần túy, Lasso/Ridge cho đến các mô hình phi tuyến nâng cao như Kernel Ridge Regression).

## Cấu trúc thư mục
Dự án được tổ chức theo cấu trúc cây (tree) nhằm phân tách rõ ràng giữa lý thuyết và ứng dụng:

```text
air-quality-co-regression/
├── part1/
│   ├── cross_validation.py        # Cài đặt thuật toán K-Fold Cross-Validation tự code
│   ├── gauss_markov_demo.py       # Kiểm định và minh họa định lý Gauss-Markov
│   ├── ols_implementation.py      # Module giải OLS và tính toán Hat Matrix
│   ├── residual_analysis.py       # Cài đặt các phép kiểm định chẩn đoán phần dư
│   ├── ridge_lasso.py             # Cài đặt hàm mất mát và điều chuẩn L1/L2
│   ├── test_case.py               # Chứa các Unit tests cho thuật toán Phần 1
│   └── part1_notebook.ipynb       # Jupyter Notebook trình bày quá trình chạy Phần 1
├── part2/
│   ├── advanced_methods.py        # Triển khai Kernel Ridge và Bayesian Regression
│   ├── data_pipeline.py           # Pipeline tiền xử lý: Impute, Winsorize, Scaling, Poly
│   ├── generate_plots.py          # Tập hợp các hàm vẽ biểu đồ trực quan hóa dữ liệu
│   ├── model_comparison.py        # Hệ thống huấn luyện, đánh giá và cache mô hình
│   ├── test_no_leakage.py         # Script kiểm tra đảm bảo không có rò rỉ dữ liệu (Leakage)
│   ├── part2_notebook.ipynb       # Jupyter Notebook trình bày quá trình chạy Phần 2
│   ├── configs/                   # Thư mục cấu hình siêu tham số cho các thuật toán
│   ├── data/
│   │   └── AirQualityUCI.csv      # Bộ dữ liệu gốc thu thập từ cảm biến
│   └── outputs/                   # (Thư mục tự sinh) Chứa cache JSON và đồ thị lưu lại
├── report/
│   ├── report.pdf                 # Bản báo cáo hoàn chỉnh (PDF)
│   ├── report.tex                 # Mã nguồn báo cáo (Latex)
│   └── assets/                    # Hình ảnh và tài nguyên của báo cáo
├── config.py                      # Cấu hình hằng số, sai số EPSILON và random seed
├── test_logger.py                 # Module logger định dạng kết quả Unit Test đẹp mắt
├── utils.py                       # Các hàm Đại số tuyến tính gốc (vector, ma trận)
└── requirements.txt               # Danh sách các thư viện cần cài đặt để chạy dự án
```

## Chi tiết các phần thực hiện

### Phần 1: Cài đặt thuật toán cốt lõi từ đầu (Lý thuyết)
Phần này tập trung vào tính toán toán học phía sau các mô hình, bao gồm nhiều mảng lý thuyết quan trọng:

- **Hàm mất mát và nghiệm OLS**
  - Mục tiêu của OLS là tối thiểu hóa tổng bình phương sai số giữa quan sát thực tế và giá trị dự đoán: $L(\beta)=\sum_{i=1}^{n}(y_i - x_i^T \beta)^2$.
  - Việc lấy đạo hàm theo hệ số và giải phương trình đạo hàm bằng 0 dẫn tới phương trình chuẩn: $\hat{\beta} = (X^T X)^{-1} X^T y$.
  - Mô hình giả định ma trận thiết kế $X$ có đa thức đầy đủ (full column rank), để $X^T X$ khả nghịch.

- **Ma Trận Chiếu và Hat Matrix**
  - Hat Matrix $H = X (X^T X)^{-1} X^T$ chiếu vector quan sát $y$ lên không gian cột của $X$.
  - $H$ là ma trận đối xứng và idempotent, nghĩa là $H^T = H$ và $H^2 = H$.
  - Các hệ số đường chéo $h_{ii}$ được gọi là leverage, biểu hiện mức độ ảnh hưởng của mỗi điểm dữ liệu lên giá trị dự đoán $\hat{y}_i$.

- **Định Lý Gauss–Markov**
  - Với giả định lỗi ngẫu nhiên có trung bình 0, phương sai đồng nhất và không tương quan, nghiệm OLS là *BLUE* (Best Linear Unbiased Estimator).
  - Nói cách khác, OLS mang lại ước lượng tuyến tính không chệch có phương sai nhỏ nhất trong không gian các ước lượng tuyến tính.
  - Đây là nền tảng để hiểu tại sao OLS vẫn được dùng rộng rãi dù dữ liệu không nhất thiết tuân theo phân phối chuẩn.

- **Ước Lượng Phương Sai Nhiễu**
  - Phương sai sai số $\sigma^2$ được ước lượng bằng sai số bình phương trung bình dư: $\hat{\sigma}^2 = \frac{1}{n - p - 1} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2$.
  - Ước lượng này đóng vai trò quan trọng trong xây dựng khoảng tin cậy hệ số và kiểm định giả thuyết.
  - Khi giả định phương sai không đổi bị vi phạm, các ước lượng này có thể không chính xác và cần phương pháp robust hoặc weighted regression.

- **Đánh Giá Mô Hình**
  - **Hệ số xác định $R^2$** đo phần phương sai của $y$ được giải thích bởi mô hình: $R^2 = 1 - \frac{SSR}{SST}$.
  - **$R^2$ hiệu chỉnh** điều chỉnh theo số biến giải thích và số quan sát, giúp so sánh mô hình khác kích thước: $\bar{R}^2 = 1 - \frac{n-1}{n-p-1}(1-R^2)$.
  - **Kiểm Định Giả Thuyết**: áp dụng kiểm định t cho từng hệ số và kiểm định F cho toàn bộ mô hình để xác định biến có ý nghĩa thống kê.

- **Đa cộng tuyến (Multicollinearity)**
  - Đa cộng tuyến xảy ra khi các biến giải thích có tương quan cao với nhau, làm cho ma trận $X^T X$ gần suy biến và ước lượng hệ số có phương sai lớn.
  - Hệ số phóng đại phương sai VIF (Variance Inflation Factor) được dùng để đánh giá: $\mathrm{VIF}_j = \frac{1}{1 - R_j^2}$, trong đó $R_j^2$ là hệ số xác định của hồi quy biến $j$ trên các biến khác.
  - VIF cao (thường > 5 hoặc > 10) cảnh báo rằng hệ số tương ứng không ổn định và mô hình dễ bị sai số lớn.

- **Hồi Quy Ridge và Lasso (Regularization)**
  - **Ridge Regression** thêm điều chuẩn L2 vào hàm mất mát: $L(\beta)=\|y - X\beta\|^2 + \lambda \|\beta\|_2^2$.
  - **Lasso Regression** thêm điều chuẩn L1: $L(\beta)=\|y - X\beta\|^2 + \lambda \|\beta\|_1$, giúp đồng thời lựa chọn biến và làm co hệ số.
  - Điều chuẩn giúp giảm thiểu phương sai ước lượng, đặc biệt khi multicollinearity mạnh hoặc khi số biến lớn hơn số quan sát.

- **Phân Tích Phần Dư (Residual Analysis)**
  - Phần dư $e_i = y_i - \hat{y}_i$ được kiểm tra để đánh giá giả định tuyến tính, độc lập, phương sai đồng nhất và phân phối chuẩn.
  - Phân tích bao gồm đồ thị phần dư so với giá trị dự đoán, Q-Q plot, kiểm định Breusch-Pagan và chỉ số ảnh hưởng Cook’s distance.
  - Mục tiêu là phát hiện ngoại lai, điểm có leverage cao và vi phạm giả định dẫn tới mô hình không tin cậy.

- **Cross-Validation và Lựa Chọn Mô Hình**
  - K-fold cross-validation đánh giá mô hình trên nhiều phân tách khác nhau để giảm sai số do lựa chọn bộ huấn luyện cụ thể.
  - Sử dụng cross-validation để chọn siêu tham số điều chuẩn $\lambda$ cho Ridge/Lasso và so sánh hiệu năng dựa trên RMSE trung bình.
  - Việc lựa chọn mô hình dựa trên đánh giá ngoài (out-of-sample) giúp hạn chế overfitting và tăng tính khái quát hóa.

### Phần 2: Data Fitting trên Dữ liệu thực tế
- **Dataset:** Tập dữ liệu **Air Quality UCI** thu thập từ hệ thống cảm biến môi trường tại Ý.
- **Link dataset:** `https://archive.ics.uci.edu/dataset/360/air+quality`
- **Bài toán đặt ra:** Dự đoán nồng độ `CO(GT)` (hợp chất carbon monoxide thật) sử dụng các đặc trưng cảm biến và điều kiện khí hậu, trên cơ sở đảm bảo mô hình không bị rò rỉ dữ liệu và có khả năng khái quát tốt.
- **Quy trình thực hiện:**
  1. **EDA** — phân tích dữ liệu ban đầu và kiểm tra phân phối, độ khuyết, và tương quan giữa biến.
  2. **Tiền xử lý** — xử lý giá trị khuyết, loại bỏ ngoại lai, mã hóa biến, chuẩn hóa và tạo đặc trưng mới khi cần.
  3. **Train/Test Split** — chia dữ liệu thành tập huấn luyện và tập kiểm tra để đánh giá ngoài.
  4. **Xây dựng mô hình** — huấn luyện OLS, Ridge, Lasso và các biến thể khác.
  5. **Đánh giá** — sử dụng RMSE, R², phần dư và kiểm định thống kê để so sánh mô hình.
  6. **Tinh chỉnh** — điều chỉnh tham số với cross-validation và lựa chọn mô hình tốt nhất.
  7. **Báo cáo kết quả** — tổng hợp hiệu năng, phân tích phần dư và đưa ra khuyến nghị.
- **Nhiệm vụ:** Xây dựng mô hình hồi quy để dự đoán nồng độ `CO(GT)` thông qua các cảm biến oxit kim loại và các thông số khí hậu. Quá trình bao gồm kiểm soát nghiêm ngặt rò rỉ dữ liệu (Data Leakage), điền khuyết bằng KNN Imputer, loại bỏ ngoại lai thô bằng Winsorize, tạo đặc trưng đa thức (Polynomial Features), điều chuẩn hóa (Lasso, Ridge) và so sánh toàn diện hiệu năng các mô hình bằng RMSE và R².

## Thư viện đã dùng
Dự án giới hạn sử dụng các thư viện tính toán cơ bản ở Phần 1 để tập trung thể hiện bản chất lý thuyết, và tận dụng sức mạnh của hệ sinh thái Python ở Phần 2 cho dữ liệu thực tế.

Các gói được sử dụng chính trong dự án:
- `numpy` — xử lý đóng gói mảng số học và các phép toán tuyến tính trên dữ liệu lớn.
- `pandas` — xử lý dữ liệu bảng, đọc/ghi CSV và tiền xử lý dữ liệu.
- `scipy` — dùng cho các phép toán thống kê và đại số tuyến tính bổ sung.
- `scikit-learn` — xây dựng pipeline, chuẩn hóa dữ liệu, chọn mô hình, K-fold CV và các thuật toán hồi quy Ridge/Lasso.
- `matplotlib` và `seaborn` — trực quan hóa dữ liệu, đồ thị phần dư, phân phối và kết quả mô hình.
- `statsmodels` — (nếu dùng trong một số phân tích thống kê phụ trợ hoặc kiểm định giả thuyết).

Danh sách đầy đủ package được quản lý trong `requirements.txt`.

## Chi tiết các file tiện ích
- **`config.py`**: Chứa các thiết lập cấu hình chung của hệ thống như hằng số `RANDOM_STATE` để cố định cấu hình ngẫu nhiên (đảm bảo tính tái lập), dung sai `EPSILON` để xử lý sai số dấu phẩy động (float precision) và các hàm xử lý tính toán cơ bản.
- **`test_logger.py`**: Chứa class `TestLogger` hoạt động như một công cụ theo dõi (logger) in ra console. Nó giúp định dạng và trình bày các log kết quả chạy Unit Test (Passed/Failed), cảnh báo (Warning), và thống kê độ chính xác tổng thể một cách rõ ràng và chuyên nghiệp.
- **`utils.py`**: Chứa các hàm toán học hỗ trợ toàn cục phục vụ cho Đại số tuyến tính (như nhân ma trận với vector `matvec`, trừ vector `vector_sub`, tính chuẩn `norm`, v.v.) được sử dụng xuyên suốt cho cả quá trình cài đặt thuật toán ở Phần 1 và kiểm định ở Phần 2.

## Hướng dẫn cài đặt môi trường
Để đảm bảo các thư viện được cài đặt biệt lập và không xung đột với hệ thống, vui lòng thực hiện các bước sau:

1. **Tạo môi trường ảo (Virtual Environment)**  
Chạy lệnh sau tại thư mục gốc của dự án để tạo thư mục `venv`:
```bash
python -m venv venv
```

2. **Kích hoạt môi trường ảo**  
Việc kích hoạt giúp máy tính nhận diện và sử dụng các thư viện trong môi trường ảo.
- **Windows:**
```powershell
.\venv\Scripts\activate
```
- **Linux/macOS:**
```bash
source venv/bin/activate
```

3. **Cài đặt các thư viện cần thiết**  
Tiến hành cài đặt danh sách các thư viện từ file `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Cách chạy thử nghiệm
Để kiểm chứng hoạt động của toàn bộ đồ án, sau khi kích hoạt môi trường ảo, bạn có thể thực hiện theo các bước sau:
- Chạy các file kiểm thử (Unit Tests) tự động thông qua lệnh: `python -m unittest discover`

- **Phần 1 (Kiểm chứng Lý thuyết):**
  - `python part1/ols_implementation.py` — tính nghiệm OLS, ma trận Hat, các thống kê mô hình và đánh giá cơ bản.
  - `python part1/ridge_lasso.py` — so sánh Ridge và Lasso, tính nghiệm điều chuẩn, và kiểm tra ảnh hưởng của tham số $\lambda$.
  - `python part1/residual_analysis.py` — phân tích phần dư, vẽ đồ thị residual vs fitted, Q-Q plot, và kiểm định chẩn đoán.
  - `python part1/cross_validation.py` — chạy K-fold Cross-Validation và chọn tham số điều chuẩn qua đánh giá RMSE.
  - `python part1/gauss_markov_demo.py` — minh họa định lý Gauss–Markov và xác thực tính BLUE của OLS qua mô phỏng.
  - `python part1/part1_notebook.ipynb` — mở bằng Jupyter Notebook hoặc VSCode để xem ví dụ thực thi, đồ thị và nhận xét lý thuyết.

- **Phần 2 (Áp dụng Thực tế):** Mở `part2/part2_notebook.ipynb` để chạy toàn bộ luồng xử lý Data Pipeline và tự động huấn luyện các mô hình Machine Learning. Các đồ thị phân tích phần dư và file kết quả cache sẽ tự động được sinh ra ở thư mục `outputs/`.

## Tác giả & Đóng góp
| MSSV | Họ và Tên | GitHub |
| :--- | :--- | :--- |
| 24120394 | Nguyễn Đặng Khôi Nguyên | [@ImSoul0511](https://github.com/ImSoul0511) |
| 24120331 | Lê Quốc Khải | [@QuocKhai](https://github.com/QuocKhai) |
| 24120384 | Phan Nhật Minh | [@MintFan1607](https://github.com/MintFan1607) |
| 24120370 | Trần Thị Lợi | [@Lowen-here](https://github.com/Lowen-here) |
| 24120474 | Trịnh Vỹ Triết | [@TrinhVyTriet](https://github.com/TrinhVyTriet) |