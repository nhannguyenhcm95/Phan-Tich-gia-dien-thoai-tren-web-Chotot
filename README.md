# 📱 Hướng Dẫn Chạy & Deploy Web App Định Giá Điện Thoại Cũ

Ứng dụng web định giá điện thoại cũ đã qua sử dụng trên Chợ Tốt sử dụng mô hình học máy kết hợp **Hybrid Ensemble (LightGBM + XGBoost + CatBoost)**. 

Thư mục `webapp/` chứa toàn bộ mã nguồn của ứng dụng.

---

## 🖥️ 1. Hướng Dẫn Chạy Ở Local (Máy Tính Cá Nhân)

Để chạy thử nghiệm ứng dụng trên máy tính của bạn:

### Bước 1: Cài đặt thư viện cần thiết
Mở **Terminal / Command Prompt** tại thư mục `webapp/` và chạy lệnh sau để cài đặt các thư viện:
```bash
pip install -r requirements.txt
```
*(Nếu bạn cài nhiều phiên bản Python, hãy dùng `py -3.13 -m pip install -r requirements.txt`)*

### Bước 2: Sao chép các file mô hình
Nếu trong thư mục `models/` chưa có đủ 6 file mô hình (`preprocessor.pkl`, `brand_encoder.pkl`, `best_lgbm_optuna.pkl`, `hybrid_xgb_model.pkl`, `hybrid_cat_model.pkl`, `webapp_config.json`), hãy chạy script setup sau để tự động sao chép chúng:
```bash
python setup_models.py
```

### Bước 3: Khởi chạy ứng dụng Streamlit
Chạy lệnh sau để bật giao diện web:
```bash
streamlit run app.py
```
*(Hoặc `py -3.13 -m streamlit run app.py`)*

Một trang web sẽ tự động mở ra trên trình duyệt của bạn tại địa chỉ: `http://localhost:8501`.

---

## 🚀 2. Hướng Dẫn Deploy Lên Public Web (Miễn Phí)

Vì Streamlit chạy trên nền Python backend server, **Netlify KHÔNG hỗ trợ** (do Netlify chỉ host web tĩnh HTML/CSS/JS). Dưới đây là 2 cách deploy miễn phí tốt nhất cho ứng dụng Streamlit:

### Cách 1: Sử dụng Streamlit Community Cloud (Khuyến nghị số 1)

Streamlit cung cấp dịch vụ cloud host miễn phí rất mượt mà cho các ứng dụng viết bằng framework này.

#### Bước 1: Đưa code lên GitHub
1. Truy cập [github.com](https://github.com) và tạo một **Repository mới** (ví dụ đặt tên: `dinh-gia-dien-thoai-chotot`).
2. Mở terminal tại thư mục `webapp/` trên máy và chạy các lệnh sau để đẩy code lên GitHub:
```bash
git init
git add .
git commit -m "Initial commit for streamlit app"
git branch -M main
git remote add origin https://github.com/TÊN_GITHUB_CỦA_BẠN/dinh-gia-dien-thoai-chotot.git
git push -u origin main
```

#### Bước 2: Deploy lên Streamlit Cloud
1. Truy cập [share.streamlit.io](https://share.streamlit.io) và đăng nhập bằng tài khoản GitHub của bạn.
2. Nhấn nút **"New app"** (hoặc **"Create app"**).
3. Điền thông tin cấu hình:
   - **Repository**: Chọn repo bạn vừa tạo (`TÊN_GITHUB_CỦA_BẠN/dinh-gia-dien-thoai-chotot`).
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Nhấn **"Deploy!"**.
5. Đợi khoảng 1-2 phút để hệ thống tự động cài đặt các thư viện trong `requirements.txt` và build ứng dụng. Sau đó bạn sẽ nhận được đường link public có dạng: `https://[tên-app].streamlit.app`.

---

### Cách 2: Sử dụng Hugging Face Spaces (Miễn phí & Rất ổn định)

Hugging Face Spaces là một nền tảng tuyệt vời khác để host các ứng dụng AI/ML viết bằng Streamlit hoàn toàn miễn phí.

1. Truy cập [huggingface.co](https://huggingface.co), đăng ký tài khoản và đăng nhập.
2. Click vào ảnh đại diện góc trên bên phải -> chọn **"New Space"**.
3. Điền các thông tin:
   - **Space name**: Tên app của bạn (VD: `dinh-gia-dien-thoai`).
   - **SDK**: Chọn **Streamlit**.
   - **Space hardware**: Chọn **Cpu basic (Free)**.
   - **Privacy**: Chọn **Public**.
4. Sau khi tạo Space, Hugging Face sẽ cung cấp một Git repository. Bạn chỉ cần clone repo đó về máy, copy toàn bộ file trong thư mục `webapp/` vào đó, rồi `git add`, `git commit` và `git push` lên Hugging Face.
5. Ứng dụng sẽ tự động được build và chạy public trên website của Hugging Face.

---

## 🛠️ Lưu Ý Về Kỹ Thuật

- **Vấn đề Pickle & Joblib**: Các file mô hình preprocessor và encoder được tạo ra bằng thư viện `joblib`. Ứng dụng đã được cập nhật sử dụng `joblib.load` thay cho `pickle.load` thông thường để tránh gặp lỗi `STACK_GLOBAL requires str` trên các phiên bản Python mới (như Python 3.13 hoặc 3.14).
- **GitHub LFS (Large File Storage)**: Tổng dung lượng các file mô hình trong thư mục `models/` là khoảng **8MB**. Mức dung lượng này nhỏ hơn giới hạn 100MB của GitHub nên bạn có thể `git push` trực tiếp lên GitHub thông thường mà **không cần cài đặt GitHub LFS**, giúp quá trình deploy trở nên vô cùng đơn giản.
