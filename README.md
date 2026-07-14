# 📱 Hướng Dẫn Chạy & Deploy Web App Định Giá Điện Thoại Cũ
# Phân Tích Giá Điện Thoại Trên Web Chợ Tốt

## Giới Thiệu

Đây là ứng dụng web hỗ trợ **phân tích và ước lượng giá điện thoại cũ** dựa trên dữ liệu tin đăng từ Chợ Tốt. Người dùng có thể nhập các thông tin của một chiếc điện thoại như hãng sản xuất, dòng máy, RAM, bộ nhớ trong, tình trạng máy, khu vực đăng bán, thông tin người bán và ngày đăng tin. Từ những thông tin đó, hệ thống sẽ sử dụng mô hình học máy để đưa ra mức giá tham khảo.

Ứng dụng được xây dựng nhằm giải quyết một vấn đề thực tế: người bán thường khó xác định mức giá hợp lý cho điện thoại đã qua sử dụng, còn người mua lại cần một công cụ tham khảo để đánh giá xem mức giá đang được rao có phù hợp với thị trường hay không.

Thông qua website này, người dùng có thể:

- Ước lượng giá bán hợp lý cho một chiếc điện thoại cũ.
- Phân tích các yếu tố ảnh hưởng đến giá như cấu hình, khu vực, tình trạng máy và uy tín người bán.
- Tham khảo kết quả dự đoán từ nhiều thuật toán học máy khác nhau.
- Xem khoảng giá đề xuất thay vì chỉ một con số cố định.

Ứng dụng phù hợp cho người bán muốn định giá sản phẩm trước khi đăng tin, người mua muốn tham khảo giá thị trường, hoặc sinh viên muốn tìm hiểu cách áp dụng Machine Learning vào bài toán phân tích giá trong thương mại điện tử.

---

## Chức Năng Chính Của Website

### 1. Nhập Thông Tin Điện Thoại

Người dùng có thể nhập các thông tin quan trọng của sản phẩm, bao gồm:

- Hệ điều hành: iPhone, Android hoặc điện thoại phổ thông.
- Hãng sản xuất: Iphone, Samsung, Xiaomi, Oppo, Vivo, Sony, Huawei, Nokia và một số hãng khác.
- Dòng máy cụ thể: ví dụ Iphone 15 Pro Max, Samsung S24 Ultra, Xiaomi 15 Ultra.
- RAM và bộ nhớ trong.
- Tình trạng máy: máy mới hoặc đã qua sử dụng.
- Số tháng bảo hành còn lại.
- Số lượng hình ảnh trong tin đăng.

Đối với iPhone, hệ thống có cơ chế tự động xác định RAM theo từng dòng máy để hạn chế sai sót khi nhập dữ liệu.

### 2. Nhập Thông Tin Đăng Bán

Ngoài cấu hình điện thoại, website còn cho phép nhập các yếu tố liên quan đến tin đăng:

- Tỉnh/thành phố và quận/huyện đăng bán.
- Vùng địa lý của tin đăng.
- Ngày đăng tin.
- Tin có rơi vào ngày lễ, cuối tuần hoặc ngày sale hay không.
- Số ngày tin đã tồn tại trên thị trường.

Những yếu tố này giúp mô hình phân tích sát thực tế hơn, vì giá điện thoại cũ có thể thay đổi theo khu vực, thời điểm đăng bán và hành vi mua sắm của người dùng.

### 3. Nhập Thông Tin Người Bán

Một số thông tin về người bán cũng được đưa vào quá trình dự đoán:

- Người bán là cá nhân hay cửa hàng/chuyên trang.
- Có chính sách bảo vệ hoặc thanh toán an toàn hay không.
- Điểm đánh giá của người bán.
- Số tin đã bán thành công.

Đây là nhóm thông tin quan trọng vì uy tín người bán có thể ảnh hưởng trực tiếp đến mức giá kỳ vọng của sản phẩm.

### 4. Dự Đoán Giá Điện Thoại

Sau khi người dùng nhập đầy đủ thông tin và nhấn nút dự đoán, hệ thống sẽ:

1. Chuẩn hóa dữ liệu đầu vào.
2. Mã hóa các biến phân loại như hãng, khu vực, dòng máy.
3. Đưa dữ liệu qua bộ tiền xử lý.
4. Chạy lần lượt các mô hình dự đoán.
5. Tổng hợp kết quả bằng mô hình Hybrid Ensemble.
6. Hiển thị mức giá đề xuất và khoảng giá tham khảo.

Kết quả hiển thị bao gồm:

- Giá dự đoán cuối cùng của mô hình kết hợp.
- Giá dự đoán riêng từ LightGBM.
- Giá dự đoán riêng từ XGBoost.
- Giá dự đoán riêng từ CatBoost.
- Khoảng giá tham khảo dựa trên sai số RMSE.

---

## Các Thuật Toán Đã Áp Dụng

Ứng dụng sử dụng hướng tiếp cận **Hybrid Ensemble**, tức là kết hợp nhiều mô hình học máy để tạo ra kết quả dự đoán ổn định hơn so với việc chỉ dùng một thuật toán duy nhất.

### 1. LightGBM

LightGBM là thuật toán Gradient Boosting được tối ưu cho tốc độ và hiệu năng trên dữ liệu dạng bảng. Trong bài toán này, LightGBM được dùng để học mối quan hệ phi tuyến giữa các đặc trưng của điện thoại và giá bán.

LightGBM phù hợp với bài toán định giá điện thoại vì:

- Xử lý tốt dữ liệu có nhiều đặc trưng.
- Học được các quy luật phức tạp trong thị trường điện thoại cũ.
- Cho kết quả tốt trên tập kiểm tra.
- Tốc độ dự đoán nhanh, phù hợp để tích hợp vào web app.

Trong mô hình Hybrid Ensemble, LightGBM có trọng số **45%**.

### 2. XGBoost

XGBoost là một trong những thuật toán boosting mạnh và phổ biến cho dữ liệu có cấu trúc. XGBoost được sử dụng để bổ sung thêm góc nhìn dự đoán cho LightGBM, giúp hệ thống giảm phụ thuộc vào một mô hình duy nhất.
