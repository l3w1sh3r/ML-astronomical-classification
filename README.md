# ML-astronomical-classification

1. Giới thiệu

Dự án này nhằm tham gia Mallorn Astronomical Classification Challenge, đây là một bài toán học máy trong lĩnh vực thiên văn học, với mục tiêu phân loại các đối tượng thiên văn dựa trên dữ liệu light curve đa băng tần.

Đây là bài toán phân loại nhị phân, trong đó có một lớp đối tượng hiếm trong tập dữ liệu lớn.

Các thách thức của bài toán:

Dữ liệu kích thước lớn

Light curve là chuỗi thời gian không đều

Quan sát đa băng tần

Mất cân bằng lớp nghiêm trọng

2. quan sát dữ liệu

Dữ liệu được chia thành hai thành phần chính:

2.1. Metadata

Gồm các file:

train_log.csv

test_log.csv

Các cột dữ liệu:

object_id: Định danh đối tượng

Z: Redshift

Z_err: Sai số redshift

EBV: Độ suy giảm ánh sáng do bụi

SpecType: Loại quang phổ

English Translation: Mô tả bằng văn bản

split: Thông tin chia dữ liệu

target (chỉ có trong tập train): Nhãn phân loại

2.2. Dữ liệu Light Curve

Dữ liệu light curve được chia thành 20 folder, mỗi folder gồm:

train_full_lightcurves.csv

test_full_lightcurves.csv

Cấu trúc file:

object_id

time (MJD)

flux

flux_err

filter (g, r, i, z, y)

3. Khảo sát và kiểm tra dữ liệu
3.1. Phân tích giá trị thiếu

Trong dữ liệu:

Cột Z_err trong train_log.csv bị thiếu hoàn toàn (100% NaN)

train_log["Z_err"].isna().mean() == 1.0


Điều này cho thấy:

Cột không chứa thông tin 

Việc điền giá trị (imputation) sẽ tạo nhiễu giả

Quyết định: Loại bỏ hoàn toàn cột Z_err.

3.2. Phát hiện rò rỉ dữ liệu (Data Leakage)

Hai cột sau có nguy cơ rò rỉ nhãn:

SpecType

English Translation

Các cột này liên quan trực tiếp đến phân loại đối tượng và không phù hợp trong tình huống dự đoán thực tế.

Quyết định: Loại bỏ hai cột này.

3.3. Mất cân bằng lớp

Phân bố nhãn:

y.value_counts(normalize=True)


Lớp 0: ~95%

Lớp 1: ~5%

MMất cân bằng nghiêm trọng.

4. Tiền xử lý dữ liệu
4.1. Xử lý Metadata

Các cột bị loại bỏ:

object_id (chỉ là định danh)

Z_err (100% NaN)

SpecType (leakage)

English Translation (leakage)

split (chỉ dùng để chia dữ liệu)

Các đặc trưng được giữ lại:

Z

EBV

4.2. Xử lý Light Curve

Light curve là chuỗi thời gian không đều vì vậyvậy không thể đưa trực tiếp vào mô hình ML truyền thống.

Chiến lược: Trích xuất đặc trưng thống kê để chuyển sang dữ liệu dạng bảng.

5. Xây dựng đặc trưng (Feature Engineering)
5.1. Đặc trưng theo từng băng tần

Với mỗi đối tượng và mỗi bộ lọc (g, r, i, z, y), tính:

Giá trị trung bình (mean)

Độ lệch chuẩn (std)

Giá trị nhỏ nhất (min)

Giá trị lớn nhất (max)

5.2. Đặc trưng toàn cục

Trích xuất thêm:

Thời gian quan sát (max time − min time)

Độ lệch phân bố (skewness)

Độ nhọn phân bố (kurtosis)

Các đặc trưng này giữ lại ý nghĩa vật lý của biến thiên ánh sáng.

6. Lựa chọn mô hình
6.1. Mô hình baseline

Mô hình Random Forest được dùng làm baseline.

Nhận xét:

Accuracy cao

Nhưng gần như không phát hiện được lớp hiếm

F1-score rất thấp (~0.02)

6.2. Mô hình chính: XGBoost

XGBoost được lựa chọn vì:

Hiệu quả với dữ liệu dạng bảng

Mô hình hóa quan hệ phi tuyến

Chịu được giá trị thiếu

Hỗ trợ xử lý mất cân bằng lớp (scale_pos_weight)

7. Xử lý mất cân bằng lớp

Tỷ lệ mẫu:

Lớp 0 : Lớp 1 ≈ 19 : 1

Áp dụng:

scale_pos_weight = n_negative / n_positive

Điều này giúp mô hình chú trọng hơn vào lớp hiếm.

8. Huấn luyện mô hình

Chia dữ liệu thành tập huấn luyện và tập validation

Sử dụng objective binary:logistic

Áp dụng early stopping để tránh overfitting

9. Đánh giá mô hình
9.1. Chỉ số đánh giá

F1-score được chọn vì:

Accuracy không phù hợp với dữ liệu mất cân bằng

F1-score cân bằng precision và recall

Phản ánh tốt khả năng phát hiện lớp hiếm

9.2. Kết quả
Mô hình	  F1-score
Random Forest	~0.02
RF + class weight	~0.10
XGBoost (baseline)	~0.18
XGBoost tối ưu	~0.25 – 0.35

10. Tối ưu ngưỡng dự đoán

Không dùng ngưỡng mặc định 0.5

Tìm ngưỡng tối đa hóa F1-score

Ngưỡng tối ưu ≈ 0.15 – 0.25

11. Dự đoán và nộp kết quả

Dự đoán trên tập test

Áp dụng ngưỡng tối ưu

Xuất file theo sample_submission.csv:
object_id
prediction

12. Tài liệu tham khảo

Kaggle Mallorn Astronomical Classification Challenge

XGBoost Documentation

Tài liệu phân tích chuỗi thời gian thiên văn
