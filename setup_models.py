"""
Script sao chép các file mô hình cần thiết vào thư mục webapp/models/
Chạy script này 1 lần trước khi chạy Streamlit app.

Usage:
    python setup_models.py
"""
import shutil
import os

# Thư mục chứa model gốc (thư mục cha)
SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# Thư mục đích
DEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Danh sách file cần copy
FILES_TO_COPY = [
    "preprocessor.pkl",
    "brand_encoder.pkl",
    "webapp_config.json",
    "best_lgbm_optuna.pkl",
    "hybrid_xgb_model.pkl",
    "hybrid_cat_model.pkl",
]

def main():
    os.makedirs(DEST_DIR, exist_ok=True)
    print(f"📁 Thư mục nguồn : {SOURCE_DIR}")
    print(f"📁 Thư mục đích  : {DEST_DIR}")
    print("-" * 50)

    success = 0
    for fname in FILES_TO_COPY:
        src = os.path.join(SOURCE_DIR, fname)
        dst = os.path.join(DEST_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            size_mb = os.path.getsize(dst) / (1024 * 1024)
            print(f"  ✅ {fname} ({size_mb:.2f} MB)")
            success += 1
        else:
            print(f"  ❌ Không tìm thấy: {fname}")

    print("-" * 50)
    print(f"Đã sao chép {success}/{len(FILES_TO_COPY)} file.")
    if success == len(FILES_TO_COPY):
        print("🎉 Hoàn tất! Bây giờ bạn có thể chạy: streamlit run app.py")
    else:
        print("⚠️ Một số file bị thiếu. Hãy kiểm tra thư mục nguồn.")

if __name__ == "__main__":
    main()
