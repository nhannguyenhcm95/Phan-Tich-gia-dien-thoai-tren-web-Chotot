import os
import json
import joblib
import pandas as pd
import numpy as np

def test_load():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    
    print("Loading config...")
    with open(os.path.join(models_dir, "webapp_config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    print("Config keys:", config.keys())
    
    print("Loading preprocessor...")
    preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.pkl"))
    print("Preprocessor loaded:", type(preprocessor))
        
    print("Loading brand_encoder...")
    brand_encoder = joblib.load(os.path.join(models_dir, "brand_encoder.pkl"))
    print("Brand encoder loaded:", type(brand_encoder))
        
    print("Loading lgbm...")
    lgbm = joblib.load(os.path.join(models_dir, "best_lgbm_optuna.pkl"))
    print("LGBM loaded:", type(lgbm))
        
    print("Loading xgb...")
    xgb = joblib.load(os.path.join(models_dir, "hybrid_xgb_model.pkl"))
    print("XGB loaded:", type(xgb))
        
    print("Loading cat...")
    cat = joblib.load(os.path.join(models_dir, "hybrid_cat_model.pkl"))
    print("Cat loaded:", type(cat))
    
    # Thử predict thử 1 sample xem có chạy được không
    # Sử dụng các giá trị mặc định giống app.py
    input_dict = {
        "region": "Tp Hồ Chí Minh",
        "area": "Quận 1",
        "exact_model": "Iphone 15 Pro Max",
        "pro": 0,
        "protection": 0,
        "ram_gb": 8,
        "storage_gb": 256,
        "images": 3,
        "seller_rating": 5.0,
        "sold_ads": 718,
        "brand": 3, # Iphone class index or encoded value
        "is_new": 0,
        "warranty_months": 6,
        "days_on_market": 0,
        "is_holiday": 0,
        "is_sale_day": 0,
        "phone_type_Android": 0,
        "phone_type_Basic": 0,
        "phone_type_iPhone": 1,
    }
    
    ZONE_COLUMNS = [
        "region_zone_Đông Nam Bộ", "region_zone_Đồng Bằng Sông Hồng",
        "region_zone_Khác", "region_zone_Tây Bắc Bộ", "region_zone_Tây Nguyên",
        "region_zone_Đông Bắc Bộ", "region_zone_Tây Nam Bộ",
        "region_zone_Nam Trung Bộ", "region_zone_Bắc Trung Bộ"
    ]
    for zc in ZONE_COLUMNS:
        input_dict[zc] = 1 if zc == "region_zone_Đông Nam Bộ" else 0
        
    df = pd.DataFrame([input_dict])
    df = df[config["cols_order"]]
    
    X = preprocessor.transform(df)
    print("Transformed shape:", X.shape)
    
    p_lgbm = lgbm.predict(X)[0]
    p_xgb = xgb.predict(X)[0]
    p_cat = cat.predict(X)[0]
    
    w = config["hybrid_weights"]
    p_hybrid = w["lgbm"] * p_lgbm + w["xgb"] * p_xgb + w["cat"] * p_cat
    
    to_vnd = lambda v: max(0, float(np.expm1(v)))
    
    print("Predictions:")
    print(f"LGBM: {to_vnd(p_lgbm):,.0f} VNĐ")
    print(f"XGB: {to_vnd(p_xgb):,.0f} VNĐ")
    print(f"Cat: {to_vnd(p_cat):,.0f} VNĐ")
    print(f"Hybrid: {to_vnd(p_hybrid):,.0f} VNĐ")
    print("Test passed successfully!")

if __name__ == "__main__":
    test_load()
