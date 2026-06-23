"""
📱 Định Giá Điện Thoại Cũ - Chợ Tốt
Ứng dụng dự đoán giá điện thoại đã qua sử dụng bằng mô hình Hybrid Ensemble
(LightGBM + XGBoost + CatBoost).

Tác giả : Đồ án phân tích dữ liệu
Framework: Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import json
import os

# ════════════════════════════════════════════════════════
# 1. PAGE CONFIG
# ════════════════════════════════════════════════════════
st.set_page_config(
    page_title="📱 Định Giá Điện Thoại Cũ | Chợ Tốt",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════
# 2. CUSTOM CSS  –  Giao diện premium, sát mockup
# ════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* ── Google Fonts ────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800;900&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Inter', sans-serif; 
    }

    /* ── Cấu hình nền và layout chung ────────────── */
    .stApp {
        background-color: #f8fafc !important;
    }

    /* ── Sidebar ────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%) !important;
        border-right: 1px solid #e2e8f0 !important;
        padding-top: 10px !important;
    }
    section[data-testid="stSidebar"] h2 {
        color: #0f172a !important; 
        font-weight: 800 !important;
        font-family: 'Outfit', sans-serif !important;
        border-bottom: 3px solid #ffba00 !important; /* Cam Chợ Tốt */
        padding-bottom: .6rem !important;
        font-size: 1.35rem !important;
        margin-bottom: 20px !important;
    }

    /* ── Hide Streamlit branding ────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Biến các Column thành Bento Cards ────────── */
    div[data-testid="stColumn"] {
        background: #ffffff !important;
        border-radius: 20px !important;
        padding: 30px !important;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.02), 0 8px 10px -6px rgba(15, 23, 42, 0.02) !important;
        border: 1px solid #f1f5f9 !important;
        margin-bottom: 20px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div[data-testid="stColumn"]:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 20px 25px -5px rgba(15, 23, 42, 0.05), 0 10px 10px -6px rgba(15, 23, 42, 0.05) !important;
        border-color: #e2e8f0 !important;
    }

    /* ── Section headers ────────────────────────── */
    .section-hdr {
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.35rem !important; 
        font-weight: 800 !important; 
        color: #0f172a !important;
        margin-top: 0 !important;
        margin-bottom: 20px !important; 
        padding-bottom: 10px !important;
        border-bottom: 2px solid #f1f5f9 !important;
    }

    /* ── Custom Header Premium ────────────────── */
    .header-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 20px;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 30px 40px;
        border-radius: 24px;
        margin-bottom: 32px;
        box-shadow: 0 15px 30px -10px rgba(15, 23, 42, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .header-logo {
        width: 60px;
        height: 60px;
        background: linear-gradient(135deg, #ffba00 0%, #ff5c35 100%); /* Màu Chợ Tốt gradient */
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 8px 20px rgba(255, 186, 0, 0.35);
        flex-shrink: 0;
        animation: pulse 2s infinite alternate;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        100% { transform: scale(1.04); }
    }
    .header-logo svg {
        width: 32px;
        height: 32px;
        fill: #ffffff;
    }
    .header-text {
        display: flex;
        flex-direction: column;
    }
    .header-title {
        font-family: 'Outfit', sans-serif !important;
        font-size: 2.2rem !important;
        font-weight: 900 !important;
        color: #ffffff !important;
        line-height: 1.15 !important;
        margin: 0 !important;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        color: #94a3b8 !important;
        margin: 6px 0 0 0 !important;
        font-weight: 400 !important;
    }

    /* ── Predict button ─────────────────────────── */
    div.stButton > button {
        width: 100% !important;
        background: linear-gradient(135deg, #ffba00 0%, #ff5c35 100%) !important;
        color: #ffffff !important; 
        border: none !important;
        padding: 18px 30px !important;
        font-size: 1.2rem !important; 
        font-weight: 800 !important;
        border-radius: 16px !important; 
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: 1px !important;
        box-shadow: 0 10px 25px -5px rgba(255, 92, 53, 0.4) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #e5a700 0%, #e04a23 100%) !important;
        box-shadow: 0 15px 30px -5px rgba(255, 92, 53, 0.6) !important;
        transform: translateY(-3px) scale(1.005) !important;
    }
    div.stButton > button:active {
        transform: translateY(-1px) !important;
    }

    /* ── Result Card Premium ────────────────────── */
    .result-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
        color: #ffffff !important;
        border-radius: 24px !important;
        padding: 36px !important;
        margin: 28px 0 !important;
        box-shadow: 0 20px 45px -10px rgba(15, 23, 42, 0.3) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        position: relative;
        overflow: hidden;
    }
    .result-card::before {
        content: '';
        position: absolute;
        top: -40%;
        right: -30%;
        width: 250px;
        height: 250px;
        background: radial-gradient(circle, rgba(255, 186, 0, 0.12) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .result-badge {
        background: rgba(255, 186, 0, 0.15) !important;
        color: #ffba00 !important;
        border: 1px solid rgba(255, 186, 0, 0.3) !important;
        padding: 6px 16px !important;
        border-radius: 20px !important;
        font-size: 0.8rem !important;
        font-weight: 800 !important;
        letter-spacing: 1.2px !important;
        display: inline-block;
        margin-bottom: 18px;
    }
    .result-price-label {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0 0 8px 0 !important;
    }
    .result-price-main {
        font-family: 'Outfit', sans-serif !important;
        font-size: 3.4rem !important;
        font-weight: 900 !important;
        color: #ffba00 !important;
        margin: 0 !important;
        line-height: 1.1 !important;
        text-shadow: 0 4px 15px rgba(255, 186, 0, 0.15);
    }
    .result-divider {
        height: 1px;
        background: rgba(255, 255, 255, 0.1);
        margin: 20px 0 !important;
    }
    .result-confidence {
        display: flex;
        align-items: center;
        gap: 12px;
        background: rgba(255, 255, 255, 0.03) !important;
        padding: 12px 20px !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    .result-confidence-icon {
        font-size: 1.2rem;
    }
    .result-confidence-text {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
        color: #cbd5e1 !important;
    }

    /* ── Algorithms Bento grid inner elements ───── */
    .algo-inner {
        display: flex;
        flex-direction: column;
        width: 100%;
    }
    .algo-badge {
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        align-self: flex-start;
        padding: 4px 10px !important;
        border-radius: 8px !important;
        margin-bottom: 10px;
    }
    .algo-lgbm .algo-badge { background: #eff6ff !important; color: #1d4ed8 !important; }
    .algo-xgb .algo-badge { background: #fef2f2 !important; color: #b91c1c !important; }
    .algo-cat .algo-badge { background: #f0fdf4 !important; color: #15803d !important; }
    
    .algo-weight {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.8rem !important;
        color: #64748b !important;
        margin-bottom: 8px;
    }
    .algo-price {
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.45rem !important;
        font-weight: 800 !important;
        color: #0f172a !important;
    }

    /* ── Divider ────────────────────────────────── */
    .sep { border:none; border-top:1px solid #e2e8f0; margin:1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# 3. DATA CONSTANTS  –  Mapping dữ liệu đào tạo
# ════════════════════════════════════════════════════════

# Load Vietnam Provinces & Districts from local JSON
@st.cache_data
def load_vietnam_provinces():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    provinces_path = os.path.join(base_dir, "vietnam_provinces.json")
    if os.path.exists(provinces_path):
        with open(provinces_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

vietnam_data = load_vietnam_provinces()

if vietnam_data:
    ALL_VIETNAM_REGIONS = sorted([p["name"] for p in vietnam_data])
    VIETNAM_REGION_TO_AREAS = {p["name"]: sorted(p["districts"]) for p in vietnam_data}
else:
    ALL_VIETNAM_REGIONS = [
        "Tp Hồ Chí Minh", "Hà Nội", "Đà Nẵng", "Bình Dương", "Đồng Nai",
        "Cần Thơ", "Hải Phòng", "An Giang", "Bà Rịa - Vũng Tàu", "Bến Tre",
        "Bình Định", "Cà Mau", "Đắk Lắk", "Đồng Tháp", "Lâm Đồng",
        "Long An", "Tây Ninh", "Thừa Thiên Huế", "Tiền Giang", "Vĩnh Long", "Other"
    ]
    VIETNAM_REGION_TO_AREAS = {
        "Tp Hồ Chí Minh": ["Quận 1", "Quận 3", "Quận 8", "Quận 10", "Quận 12", "Quận Bình Thạnh", "Quận Bình Tân", "Quận Gò Vấp", "Quận Tân Bình", "Quận Tân Phú", "Thành phố Thủ Đức", "Other"],
        "Hà Nội": ["Quận Cầu Giấy", "Quận Hoàng Mai", "Quận Thanh Xuân", "Quận Đống Đa", "Other"],
        "Đà Nẵng": ["Quận Hải Châu", "Quận Thanh Khê", "Other"],
        "Cần Thơ": ["Quận Ninh Kiều", "Other"],
        "Đồng Nai": ["Thành phố Biên Hòa", "Other"],
        "Đắk Lắk": ["Thành phố Buôn Ma Thuột", "Other"],
    }

def clean_region_name(name):
    """Chuẩn hóa tên tỉnh thành để khớp với nhãn mô hình AI."""
    name_clean = name.replace("Thành phố ", "").replace("Tỉnh ", "")
    if name_clean == "Hồ Chí Minh":
        return "Tp Hồ Chí Minh"
    return name_clean


def get_iphone_ram(model_name):
    """Tự động xác định RAM của iPhone dựa theo model."""
    model_name_lower = model_name.lower()
    if "17 pro" in model_name_lower:
        return 12
    if "17" in model_name_lower:
        return 8
    if "16" in model_name_lower:
        return 8
    if "15 pro" in model_name_lower:
        return 8
    if "15" in model_name_lower:
        return 6
    if "14 pro" in model_name_lower:
        return 6
    if "14" in model_name_lower:
        return 6
    if "13 pro" in model_name_lower:
        return 6
    if "13" in model_name_lower:
        return 4
    if "12 pro" in model_name_lower:
        return 6
    if "12 promax" in model_name_lower:
        return 6
    if "12" in model_name_lower:
        return 4
    if "11 pro" in model_name_lower:
        return 4
    if "11" in model_name_lower:
        return 4
    if "xs" in model_name_lower:
        return 4
    if "xr" in model_name_lower:
        return 3
    if "x" in model_name_lower:
        return 3
    if "8 plus" in model_name_lower:
        return 3
    if "8" in model_name_lower:
        return 2
    if "7 plus" in model_name_lower:
        return 3
    if "7" in model_name_lower:
        return 2
    if "6s plus" in model_name_lower:
        return 2
    if "6s" in model_name_lower:
        return 2
    if "se 2020" in model_name_lower:
        return 3
    if "se" in model_name_lower:
        return 2
    return 6


def get_vietnam_zone(province_name):
    province_name_clean = clean_region_name(province_name)
    if province_name_clean in ["Lào Cai", "Yên Bái", "Điện Biên", "Hoà Bình", "Lai Châu", "Sơn La"]:
        return "Tây Bắc Bộ"
    if province_name_clean in ["Hà Giang", "Cao Bằng", "Bắc Kạn", "Lạng Sơn", "Tuyên Quang", "Thái Nguyên", "Phú Thọ", "Bắc Giang", "Quảng Ninh"]:
        return "Đông Bắc Bộ"
    if province_name_clean in ["Hà Nội", "Hải Phòng", "Bắc Ninh", "Hà Nam", "Hải Dương", "Hưng Yên", "Nam Định", "Ninh Bình", "Thái Bình", "Vĩnh Phúc"]:
        return "Đồng Bằng Sông Hồng"
    if province_name_clean in ["Thanh Hóa", "Nghệ An", "Hà Tĩnh", "Quảng Bình", "Quảng Trị", "Thừa Thiên Huế"]:
        return "Bắc Trung Bộ"
    if province_name_clean in ["Đà Nẵng", "Quảng Nam", "Quảng Ngãi", "Bình Định", "Phú Yên", "Khánh Hòa", "Ninh Thuận", "Bình Thuận"]:
        return "Nam Trung Bộ"
    if province_name_clean in ["Kon Tum", "Gia Lai", "Đắk Lắk", "Đắk Nông", "Lâm Đồng"]:
        return "Tây Nguyên"
    if province_name_clean in ["Bình Phước", "Bình Dương", "Đồng Nai", "Tây Ninh", "Bà Rịa - Vũng Tàu", "Hồ Chí Minh", "Tp Hồ Chí Minh"]:
        return "Đông Nam Bộ"
    if province_name_clean in ["Long An", "Tiền Giang", "Bến Tre", "Trà Vinh", "Vĩnh Long", "Đồng Tháp", "An Giang", "Kiên Giang", "Cần Thơ", "Hậu Giang", "Sóc Trăng", "Bạc Liêu", "Cà Mau"]:
        return "Tây Nam Bộ"
    return "Khác"


def check_special_date(d):
    """Kiểm tra xem ngày chọn có phải là ngày Lễ hay ngày Sale không."""
    holidays = {
        (1, 1): "Tết Dương Lịch",
        (30, 4): "Giải Phóng Miền Nam",
        (1, 5): "Quốc Tế Lao Động",
        (2, 9): "Quốc Khánh",
        (25, 12): "Giáng Sinh"
    }
    lunar_holidays = {
        2025: {
            (1, 28): "Tết Nguyên Đán", (1, 29): "Tết Nguyên Đán", (1, 30): "Tết Nguyên Đán",
            (1, 31): "Tết Nguyên Đán", (2, 1): "Tết Nguyên Đán", (2, 2): "Tết Nguyên Đán", (2, 3): "Tết Nguyên Đán",
            (4, 7): "Giỗ tổ Hùng Vương"
        },
        2026: {
            (2, 16): "Tết Nguyên Đán", (2, 17): "Tết Nguyên Đán", (2, 18): "Tết Nguyên Đán",
            (2, 19): "Tết Nguyên Đán", (2, 20): "Tết Nguyên Đán", (2, 21): "Tết Nguyên Đán", (2, 22): "Tết Nguyên Đán",
            (4, 26): "Giỗ tổ Hùng Vương"
        }
    }
    is_hol = 0
    is_sal = 0
    desc = "Ngày thường"
    
    day_month = (d.day, d.month)
    if day_month in holidays:
        is_hol = 1
        desc = holidays[day_month]
    elif d.year in lunar_holidays and day_month in lunar_holidays[d.year]:
        is_hol = 1
        desc = lunar_holidays[d.year][day_month]
        
    if d.day == d.month:
        is_sal = 1
        desc = f"Ngày Sale (Ngày đôi {d.day}/{d.month})"
    elif d.day == 15:
        is_sal = 1
        desc = "Ngày Sale (Payday giữa tháng)"
    elif d.day == 25:
        is_sal = 1
        desc = "Ngày Sale (Payday cuối tháng)"
    elif d.day in [30, 31] or (d.month == 2 and d.day in [28, 29]):
        is_sal = 1
        desc = "Ngày Sale (Cuối tháng)"
        
    return is_hol, is_sal, desc


import calendar
import datetime

def get_html_calendar(year, month, selected_day):
    """Vẽ bảng lịch tháng với chỉ báo ngày Lễ, ngày Sale và ngày đã chọn."""
    cal = calendar.monthcalendar(year, month)
    month_name = f"Lịch Tháng {month} / {year}"
    
    html = f"""
    <style>
        .cal-table {{ width: 100%; border-collapse: collapse; font-family: 'Inter', sans-serif; font-size: 11px; text-align: center; }}
        .cal-th {{ padding: 4px; background-color: #f8fafc; color: #64748b; font-weight: 600; border: 1px solid #e2e8f0; }}
        .cal-td {{ padding: 6px; border: 1px solid #e2e8f0; position: relative; }}
        .cal-selected {{ background-color: #ffba00 !important; color: white !important; font-weight: bold; border-radius: 6px; }}
        .cal-holiday {{ background-color: #fef2f2; color: #ef4444; font-weight: bold; }}
        .cal-holiday::after {{ content: '🎉'; position: absolute; top: 1px; right: 1px; font-size: 7px; }}
        .cal-sale {{ background-color: #fffbeb; color: #d97706; font-weight: bold; }}
        .cal-sale::after {{ content: '🏷️'; position: absolute; top: 1px; right: 1px; font-size: 7px; }}
        .cal-empty {{ background-color: #f8fafc; }}
    </style>
    <div style="background-color: white; padding: 10px; border-radius: 12px; border: 1px solid #e2e8f0; margin-top: 10px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
        <div style="font-weight: 700; text-align: center; margin-bottom: 6px; color: #0f172a; font-size: 12px;">{month_name}</div>
        <table class="cal-table">
            <tr>
                <th class="cal-th">CN</th>
                <th class="cal-th">T2</th>
                <th class="cal-th">T3</th>
                <th class="cal-th">T4</th>
                <th class="cal-th">T5</th>
                <th class="cal-th">T6</th>
                <th class="cal-th">T7</th>
            </tr>
    """
    for week in cal:
        html += "<tr>"
        for day in week:
            if day == 0:
                html += "<td class='cal-td cal-empty'></td>"
            else:
                classes = []
                is_sel = (day == selected_day)
                
                is_hol, is_sal, _ = check_special_date(datetime.date(year, month, day))
                if is_hol:
                    classes.append("cal-holiday")
                elif is_sal:
                    classes.append("cal-sale")
                
                if is_sel:
                    classes.append("cal-selected")
                    
                class_str = " ".join(classes)
                html += f"<td class='cal-td {class_str}'>{day}</td>"
        html += "</tr>"
    html += "</table>"
    html += "<div style='display: flex; gap: 8px; justify-content: center; margin-top: 6px; font-size: 9px; color: #64748b;'>"
    html += "<span><span style='color: #ef4444;'>●</span> Lễ</span>"
    html += "<span><span style='color: #d97706;'>●</span> Sale</span>"
    html += "<span><span style='color: #ffba00;'>●</span> Chọn</span>"
    html += "</div></div>"
    return html


ZONES = [
    "Đông Nam Bộ", "Đồng Bằng Sông Hồng", "Tây Nam Bộ",
    "Nam Trung Bộ", "Bắc Trung Bộ", "Tây Nguyên",
    "Đông Bắc Bộ", "Tây Bắc Bộ", "Khác"
]

ZONE_COLUMNS = [
    "region_zone_Đông Nam Bộ", "region_zone_Đồng Bằng Sông Hồng",
    "region_zone_Khác", "region_zone_Tây Bắc Bộ", "region_zone_Tây Nguyên",
    "region_zone_Đông Bắc Bộ", "region_zone_Tây Nam Bộ",
    "region_zone_Nam Trung Bộ", "region_zone_Bắc Trung Bộ"
]

BRAND_CLASSES = [
    "Asus", "Google", "Huawei", "Iphone", "Khác", "Nokia",
    "Oppo", "Realme", "Samsung", "Sony", "Tecno", "Vivo", "Xiaomi"
]

# Hệ điều hành → Hãng có thể chọn
PHONE_TYPE_BRANDS = {
    "iPhone (iOS)":     ["Iphone"],
    "Android":          ["Samsung", "Google", "Huawei", "Oppo", "Realme",
                         "Sony", "Vivo", "Xiaomi", "Khác"],
    "Basic (Phổ thông)": ["Nokia", "Asus", "Tecno", "Khác"],
}

# Hãng → Danh sách dòng máy (trích từ dữ liệu huấn luyện)
BRAND_MODELS = {
    "Iphone": [
        "Iphone 17 Pro Max", "Iphone 17 Pro", "Iphone 17",
        "Iphone 16 Pro Max", "Iphone 16 Pro", "Iphone 16 Plus", "Iphone 16",
        "Iphone 15 Pro Max", "Iphone 15 Pro", "Iphone 15 Plus", "Iphone 15",
        "Iphone 14 Pro Max", "Iphone 14 Pro", "Iphone 14 Plus", "Iphone 14",
        "Iphone 13 Pro Max", "Iphone 13 Pro", "Iphone 13 Mini", "Iphone 13",
        "Iphone 12 Pro Max", "Iphone 12 Promax", "Iphone 12 Pro",
        "Iphone 12 Mini", "Iphone 12",
        "Iphone 11 Pro Max", "Iphone 11 Pro", "Iphone 11",
        "Iphone Xs Max", "Iphone Xs", "Iphone Xr", "Iphone X",
        "Iphone 8 Plus", "Iphone 8", "Iphone 7 Plus", "Iphone 7",
        "Iphone 6S Plus", "Iphone 6S",
        "Iphone Se 2020", "Iphone Se",
        "16 Pro Max", "15 Pro Max", "14 Pro Max",
        "13 Pro Max", "12 Pro Max", "11 Pro Max",
        "Iphone", "Khác",
    ],
    "Samsung": [
        "Samsung S26 Ultra",
        "Samsung S25 Ultra", "Samsung S25",
        "Samsung S24 Ultra", "Samsung S24 Plus", "S24 Ultra",
        "Samsung S23 Ultra", "Samsung S23 Plus", "Samsung S23 Fe", "Samsung S23",
        "Samsung S22 Ultra", "Samsung S22",
        "Samsung S21 Ultra", "Samsung S21 Plus", "Samsung S21 Fe", "Samsung S21",
        "Samsung S20 Ultra", "Samsung S20 Fe", "Samsung S10",
        "Samsung Note 20 Ultra", "Samsung Note 20", "Samsung Note 10 Plus",
        "Samsung Z Fold 6", "Samsung Z Fold 4", "Samsung Z Flip 5",
        "Samsung A56", "Samsung A36", "Samsung A12", "Samsung A06",
        "Khác",
    ],
    "Xiaomi": [
        "Xiaomi 15 Ultra", "Xiaomi 15T", "Xiaomi 15",
        "Xiaomi Redmi Note 15", "Xiaomi Redmi Note 14", "Xiaomi Redmi Note 13",
        "Xiaomi Redmi Note 12", "Xiaomi Redmi Note 11", "Xiaomi Redmi Note 10",
        "Xiaomi Redmi Turbo 5", "Xiaomi Redmi Turbo 4",
        "Khác",
    ],
    "Oppo": [
        "Oppo Find X9S", "Oppo Find X9 Pro", "Oppo Find X9",
        "Oppo Find X8 Pro", "Oppo Find X8",
        "Khác",
    ],
    "Vivo": [
        "Vivo X300 Pro", "Vivo X200 Ultra",
        "Vivo X200 Pro Mini", "Vivo X200 Pro",
        "Khác",
    ],
    "Google":  ["Google Pixel 9 Pro", "Khác"],
    "Sony":    ["Sony Xperia 1 Mark", "Khác"],
    "Huawei":  ["Honor Magic 8 Pro", "Honor Magic 7 Pro", "Honor Win Rt", "Khác"],
    "Nokia":   ["Khác"],
    "Asus":    ["Khác"],
    "Realme":  ["Khác"],
    "Tecno":   ["Khác"],
    "Khác":    ["Khác"],
}


# ════════════════════════════════════════════════════════
# 4. LOAD MODELS  (cached)
# ════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="🔄 Đang tải mô hình AI...")
def load_all_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")

    with open(os.path.join(models_dir, "webapp_config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)

    preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.pkl"))
    brand_encoder = joblib.load(os.path.join(models_dir, "brand_encoder.pkl"))
    lgbm = joblib.load(os.path.join(models_dir, "best_lgbm_optuna.pkl"))
    xgb = joblib.load(os.path.join(models_dir, "hybrid_xgb_model.pkl"))
    cat = joblib.load(os.path.join(models_dir, "hybrid_cat_model.pkl"))

    return config, preprocessor, brand_encoder, lgbm, xgb, cat


# ════════════════════════════════════════════════════════
# 5. PREDICTION FUNCTION
# ════════════════════════════════════════════════════════

def predict_price(config, preprocessor, brand_encoder, lgbm, xgb, cat, input_dict):
    """Dự đoán giá điện thoại bằng Hybrid Ensemble."""
    df = pd.DataFrame([input_dict])
    df = df[config["cols_order"]]         # đúng thứ tự cột

    X = preprocessor.transform(df)        # biến đổi đặc trưng

    w = config["hybrid_weights"]          # trọng số ensemble
    p_lgbm = lgbm.predict(X)[0]
    p_xgb  = xgb.predict(X)[0]
    p_cat  = cat.predict(X)[0]

    p_hybrid = w["lgbm"] * p_lgbm + w["xgb"] * p_xgb + w["cat"] * p_cat

    to_vnd = lambda v: max(0, float(np.expm1(v)))
    return {
        "hybrid": to_vnd(p_hybrid),
        "lgbm":   to_vnd(p_lgbm),
        "xgb":    to_vnd(p_xgb),
        "cat":    to_vnd(p_cat),
    }


def encode_brand(brand_encoder, brand_name):
    """Mã hóa tên hãng thành số."""
    try:
        return int(brand_encoder.transform([brand_name])[0])
    except Exception:
        try:
            return BRAND_CLASSES.index(brand_name)
        except ValueError:
            return BRAND_CLASSES.index("Khác")


# ════════════════════════════════════════════════════════
# 6. MAIN APP
# ════════════════════════════════════════════════════════

# ── Load models ──────────────────────────────────────
try:
    config, preprocessor, brand_encoder, lgbm_model, xgb_model, cat_model = load_all_models()
    models_ok = True
except Exception as e:
    models_ok = False
    _err = str(e)

st.markdown("""
<div class="header-container">
    <div class="header-logo">
        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path d="M17 2H7c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h10c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-5 18c-.55 0-1-.45-1-1s.45-1 1-1 1 .45 1 1-.45 1-1 1zM16 16H8V5h8v11z"/>
        </svg>
    </div>
    <div class="header-text">
        <h1 class="header-title">Định Giá Điện Thoại Cũ</h1>
        <p class="header-subtitle">Hệ thống AI/ML dự đoán giá điện thoại đã qua sử dụng trên sàn Chợ Tốt</p>
    </div>
</div>
""", unsafe_allow_html=True)

if not models_ok:
    st.error(
        f"⚠️ Không thể tải mô hình. Hãy chạy `python setup_models.py` trước.\n\n"
        f"Lỗi chi tiết: `{_err}`"
    )
    st.info(
        "📋 **Hướng dẫn:**\n"
        "1. Mở Terminal tại thư mục `webapp/`\n"
        "2. Chạy: `python setup_models.py`\n"
        "3. Khởi động lại: `streamlit run app.py`"
    )
    st.stop()


# ── Khởi tạo 2 Tab ──────────────────────────────────
tab1, tab2 = st.tabs(["🔮 Dự Đoán Giá Điện Thoại", "📊 Thông Tin Mô Hình & Độ Chính Xác"])

with tab1:
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown('<p class="section-hdr">📱 Thông Số Phần Cứng</p>', unsafe_allow_html=True)

        # ── Hệ điều hành ──
        st.markdown("##### 💻 Hệ Điều Hành:")
        phone_type = st.radio(
            "Chọn HĐH:",
            ["iPhone (iOS)", "Android", "Basic (Phổ thông)"],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")

        # ── Hãng ──
        brands = PHONE_TYPE_BRANDS.get(phone_type, ["Khác"])
        st.markdown("##### Hãng Sản Xuất")
        selected_brand = st.selectbox("Hãng", brands, label_visibility="collapsed")

        # ── Dòng máy ──
        models_list = BRAND_MODELS.get(selected_brand, ["Khác"])
        st.markdown("##### Dòng máy (VD: Iphone 15 Pro Max, Samsung S22 Ultra)")
        selected_model = st.selectbox("Dòng máy", models_list, label_visibility="collapsed")

        st.markdown("---")

        # ── RAM ──
        st.markdown("##### RAM (GB)")
        if selected_brand == "Iphone":
            default_ram = get_iphone_ram(selected_model)
            ram_gb = st.selectbox(
                "RAM", [default_ram], index=0, disabled=True,
                label_visibility="collapsed",
                help="iPhone sử dụng cấu hình RAM tối ưu tự động dựa trên dòng máy."
            )
        else:
            ram_gb = st.number_input(
                "RAM", min_value=1, max_value=24, value=8, step=1,
                label_visibility="collapsed",
            )

        # ── Bộ nhớ trong ──
        st.markdown("##### Bộ Nhớ Trong (GB)")
        storage_gb = st.number_input(
            "Storage", min_value=8, max_value=1024, value=256, step=8,
            label_visibility="collapsed",
        )

    with col2:
        st.markdown('<p class="section-hdr">📍 Thông Tin Đăng Bán</p>', unsafe_allow_html=True)

        default_region_name = "Thành phố Hồ Chí Minh"
        region_idx = ALL_VIETNAM_REGIONS.index(default_region_name) if default_region_name in ALL_VIETNAM_REGIONS else 0
        selected_region = st.selectbox("Tỉnh / Thành Phố", ALL_VIETNAM_REGIONS, index=region_idx)

        areas = VIETNAM_REGION_TO_AREAS.get(selected_region, ["Other"])
        selected_area = st.selectbox("Quận / Huyện", areas)

        selected_zone = get_vietnam_zone(selected_region)

        is_new = st.checkbox("📦 Máy Mới (Chưa sử dụng)")

        warranty_months = st.number_input(
            "Số tháng bảo hành (nếu có)",
            min_value=0, max_value=36, value=6, step=1,
        )

        images = st.slider("Số lượng ảnh đăng", min_value=0, max_value=20, value=3)

    with col3:
        st.markdown('<p class="section-hdr">👤 Người Bán & Ngày Đăng</p>', unsafe_allow_html=True)

        is_pro = st.checkbox("Cửa hàng (Chuyên trang / Pro)")
        has_protection = st.checkbox("Có chính sách Bảo Vệ / Thanh toán an toàn")

        seller_rating = st.slider(
            "Đánh giá người bán (Sao)",
            min_value=0.0, max_value=5.0, value=5.0, step=0.01, format="%.2f",
        )

        sold_ads = st.number_input(
            "Số tin đã bán thành công",
            min_value=0, max_value=99999, value=718, step=1,
        )

        st.markdown("##### 📅 Chọn Ngày Đăng Tin")
        import datetime
        today = datetime.date.today()
        selected_date = st.date_input(
            "Ngày đăng tin",
            value=today,
            max_value=today,
            label_visibility="collapsed",
            help="Chọn ngày bạn đăng tin để tự động tính khoảng cách ngày và xác định ngày Lễ/Sale."
        )

        days_on_market = (today - selected_date).days
        is_holiday, is_sale_day, special_desc = check_special_date(selected_date)

        if is_holiday:
            st.markdown(f"""
            <div style="background-color: #fef2f2; border: 1px solid #fca5a5; padding: 12px; border-radius: 12px; margin-top: 10px; margin-bottom: 10px;">
                <span style="color: #dc2626; font-weight: 700; font-size: 13px;">🎉 Đăng vào: {special_desc}</span><br>
                <span style="color: #7f1d1d; font-size: 12px;">Đã đăng: <strong>{days_on_market}</strong> ngày trước</span>
            </div>
            """, unsafe_allow_html=True)
        elif is_sale_day:
            st.markdown(f"""
            <div style="background-color: #fffbeb; border: 1px solid #fcd34d; padding: 12px; border-radius: 12px; margin-top: 10px; margin-bottom: 10px;">
                <span style="color: #d97706; font-weight: 700; font-size: 13px;">🏷️ Đăng vào: {special_desc}</span><br>
                <span style="color: #78350f; font-size: 12px;">Đã đăng: <strong>{days_on_market}</strong> ngày trước</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #f0fdf4; border: 1px solid #86efac; padding: 12px; border-radius: 12px; margin-top: 10px; margin-bottom: 10px;">
                <span style="color: #16a34a; font-weight: 700; font-size: 13px;">📅 Đăng vào: Ngày thường</span><br>
                <span style="color: #14532d; font-size: 12px;">Đã đăng: <strong>{days_on_market}</strong> ngày trước</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(get_html_calendar(selected_date.year, selected_date.month, selected_date.day), unsafe_allow_html=True)

    st.markdown("")
    predict_clicked = st.button("🔮 DỰ ĐOÁN GIÁ", use_container_width=True)

    if predict_clicked:
        # — Mã hoá phone_type —
        pt_iphone  = 1 if phone_type == "iPhone (iOS)" else 0
        pt_android = 1 if phone_type == "Android" else 0
        pt_basic   = 1 if phone_type == "Basic (Phổ thông)" else 0

        # — Mã hoá brand —
        brand_encoded = encode_brand(brand_encoder, selected_brand)

        # — One-hot region zone —
        zone_vals = {}
        for zc in ZONE_COLUMNS:
            zone_name = zc.replace("region_zone_", "")
            zone_vals[zc] = 1 if zone_name == selected_zone else 0

        # — Trích xuất các danh mục được huấn luyện từ config để mapping an toàn —
        FEATURE_NAMES = config.get("feature_names", [])
        ALL_TRAINED_REGIONS = [f.replace("cat__region_", "") for f in FEATURE_NAMES if f.startswith("cat__region_")]
        ALL_TRAINED_AREAS = [f.replace("cat__area_", "") for f in FEATURE_NAMES if f.startswith("cat__area_")]
        ALL_TRAINED_MODELS = [f.replace("cat__exact_model_", "") for f in FEATURE_NAMES if f.startswith("cat__exact_model_")]

        cleaned_region = clean_region_name(selected_region)

        # — Xây dựng dict đầu vào —
        input_dict = {
            "region":           cleaned_region if cleaned_region in ALL_TRAINED_REGIONS else "Other",
            "area":             selected_area if selected_area in ALL_TRAINED_AREAS else "Other",
            "exact_model":      selected_model if selected_model in ALL_TRAINED_MODELS else "Khác",
            "pro":              int(is_pro),
            "protection":       int(has_protection),
            "ram_gb":           ram_gb,
            "storage_gb":       storage_gb,
            "images":           images,
            "seller_rating":    seller_rating,
            "sold_ads":         sold_ads,
            "brand":            brand_encoded,
            "is_new":           int(is_new),
            "warranty_months":  warranty_months,
            "days_on_market":   days_on_market,
            "is_holiday":       int(is_holiday),
            "is_sale_day":      int(is_sale_day),
            "phone_type_Android": pt_android,
            "phone_type_Basic":   pt_basic,
            "phone_type_iPhone":  pt_iphone,
        }
        input_dict.update(zone_vals)

        # — Dự đoán —
        try:
            with st.spinner("⏳ Đang tính toán..."):
                results = predict_price(
                    config, preprocessor, brand_encoder,
                    lgbm_model, xgb_model, cat_model,
                    input_dict,
                )

            # ── Hiển thị kết quả ──
            st.success("✅ Dự đoán thành công!")
            
            # ── Khoảng tin cậy (±1 RMSE) ──
            rmse = config.get("rmse_vnd", 3_430_679)
            lo = max(0, results["hybrid"] - rmse)
            hi = results["hybrid"] + rmse

            # ── Kết quả chính dạng Card Premium ──
            st.markdown(f"""
            <div class="result-card">
                <div class="result-header">
                    <span class="result-badge">✨ KẾT QUẢ ĐỊNH GIÁ AI</span>
                </div>
                <div class="result-body">
                    <p class="result-price-label">Giá đề xuất (Hybrid Ensemble)</p>
                    <h1 class="result-price-main">{results["hybrid"]:,.0f} <span style="font-size: 1.8rem; font-weight: 700; color: #94a3b8;">VNĐ</span></h1>
                    <div class="result-divider"></div>
                    <div class="result-confidence">
                        <span class="result-confidence-icon">🎯</span>
                        <span class="result-confidence-text">
                            Khoảng giá tham khảo: <strong style="color: #ffba00;">{lo:,.0f} VNĐ</strong> – <strong style="color: #ffba00;">{hi:,.0f} VNĐ</strong> (±RMSE)
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Chi tiết từng thuật toán dạng Bento Grid ──
            st.markdown("<h3 style='font-family:\"Outfit\", sans-serif; font-weight:700; color:#0f172a; margin-top:28px; margin-bottom:16px;'>💡 Dự đoán chi tiết từ các mô hình cơ sở</h3>", unsafe_allow_html=True)
            w = config["hybrid_weights"]
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(f"""
                <div class="algo-inner algo-lgbm">
                    <span class="algo-badge">LightGBM</span>
                    <div class="algo-weight">Trọng số {w["lgbm"]*100:.0f}%</div>
                    <div class="algo-price">{results["lgbm"]:,.0f} <span style="font-size:0.85rem; font-weight:600; color:#64748b;">VNĐ</span></div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="algo-inner algo-xgb">
                    <span class="algo-badge">XGBoost</span>
                    <div class="algo-weight">Trọng số {w["xgb"]*100:.0f}%</div>
                    <div class="algo-price">{results["xgb"]:,.0f} <span style="font-size:0.85rem; font-weight:600; color:#64748b;">VNĐ</span></div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="algo-inner algo-cat">
                    <span class="algo-badge">CatBoost</span>
                    <div class="algo-weight">Trọng số {w["cat"]*100:.0f}%</div>
                    <div class="algo-price">{results["cat"]:,.0f} <span style="font-size:0.85rem; font-weight:600; color:#64748b;">VNĐ</span></div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Lỗi khi dự đoán: {str(e)}")
            with st.expander("🔍 Chi tiết lỗi"):
                st.exception(e)

with tab2:
    st.markdown("<h3 style='font-family:\"Outfit\", sans-serif; font-weight:700; color:#0f172a; margin-top:20px; margin-bottom:16px;'>📊 Thông Tin Về Mô Hình & Độ Chính Xác</h3>", unsafe_allow_html=True)
    st.markdown("""
    **Mô hình Hybrid Ensemble** kết hợp 3 thuật toán tốt nhất:
    - **LightGBM** (trọng số 45%) – Gradient Boosting tối ưu bằng Optuna
    - **XGBoost** (trọng số 35%) – eXtreme Gradient Boosting
    - **CatBoost** (trọng số 20%) – Categorical Boosting

    **Dữ liệu huấn luyện:**
    - 30,430 tin đăng bán điện thoại từ Chợ Tốt
    - 28 đặc trưng đầu vào (phần cứng, thông tin đăng bán, người bán)
    """)

    if models_ok:
        st.markdown("**Hiệu suất trên tập kiểm tra:**")
        perf_data = {
            "Chỉ số": ["RMSE (VNĐ)", "MAE (VNĐ)", "RMSE (Log scale)"],
            "Giá trị": [
                f'{config.get("rmse_vnd", 0):,.0f}',
                f'{config.get("mae_vnd", 0):,.0f}',
                f'{config.get("rmse_log_scale", 0):.4f}',
            ],
        }
        st.table(pd.DataFrame(perf_data))

        # Bảng so sánh 14 mô hình
        st.markdown("**Bảng so sánh 14 mô hình cơ sở:**")
        bench = config.get("benchmark_results", [])
        if bench:
            df_bench = pd.DataFrame(bench)
            display_cols = ["Mô hình", "R² Test", "R² CV Mean (k=5)", "MAE (VNĐ)", "RMSE (VNĐ)", "Overfitting?"]
            available_cols = [c for c in display_cols if c in df_bench.columns]
            st.dataframe(df_bench[available_cols], use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════
# 7. FOOTER
# ══════════════════════════════════════════════════════
st.markdown('<hr class="sep">', unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#adb5bd;font-size:.8rem;margin-top:1rem'>"
    "© 2025 Đồ án Phân tích giá điện thoại trên Chợ Tốt | Powered by Streamlit</p>",
    unsafe_allow_html=True,
)
