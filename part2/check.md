| Vấn đề                                           | Vì sao nguy hiểm?                                              | Cần check lại                                                               |
| ------------------------------------------------ | -------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **1. Data leakage**                              | Nếu fit imputer trên toàn bộ dataset, test set bị “nhìn trước” | `knn_train_data` phải lấy từ **X_train**, không phải full data              |
| **2. Dùng target để impute feature**             | Làm mô hình học gián tiếp từ đáp án                            | Nếu dự đoán `CO(GT)`, không dùng `CO(GT)` trong X để tính khoảng cách KNN   |
| **3. Target bị missing**                         | Không nên tự điền `y` rồi train, vì target là đáp án thật      | Drop các dòng có missing ở target trước khi train                           |
| **4. Chưa chuẩn hóa trước KNN**                  | Cột scale lớn sẽ chi phối khoảng cách Euclidean                | Nên standardize numeric features trước khi tính distance                    |
| **5. Categorical chưa xử lý**                    | KNN dùng phép trừ, không chạy đúng với dữ liệu chữ/category    | Chỉ dùng KNN cho numeric hoặc encode categorical trước                      |
| **6. Không có fallback**                         | Nếu không tìm được hàng phù hợp, dữ liệu vẫn còn NaN           | Thêm fallback bằng median/mean từ train                                     |
| **7. Quá ít feature chung khi tính khoảng cách** | Một dòng có thể bị coi là “gần” chỉ vì giống 1 cột             | Đặt `min_common_features`, ví dụ ít nhất 2 hoặc 3                           |
| **8. Cột missing quá nhiều vẫn impute**          | Impute cột thiếu 80–90% dễ tạo dữ liệu giả, thiếu tin cậy      | Drop `NMHC(GT)` vì missing > 80%                                            |
| **9. Không giải thích cơ chế missing**           | Báo cáo có thể bị thiếu phần lý luận                           | Nói rõ: sau khi drop `NMHC(GT)`, các cột còn lại giả định **MAR**           |
| **10. Không giải thích chọn KNN**                | Dễ bị hỏi “sao không dùng mean/median?”                        | Giải thích: KNN phù hợp MAR vì tận dụng quan hệ giữa các biến quan sát được |
| **11. Không tách fit/transform rõ ràng**         | Sai yêu cầu pipeline Part 2                                    | `fit()` lưu thống kê từ train, `transform()` áp dụng cho train/test         |
| **12. Không kiểm tra còn NaN sau pipeline**      | Mô hình OLS/Ridge có thể lỗi hoặc ra kết quả sai               | Sau transform nên check `X_processed.isna().sum()`                          |
