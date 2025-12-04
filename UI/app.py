# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import plotly.express as px

# # 1. CẤU HÌNH TRANG
# st.set_page_config(page_title="Dự đoán hành vi mua hàng", layout="wide")

# st.title("🛍️ Hệ thống gợi ý sản phẩm tiếp theo (Bayes)")
# st.markdown("Dự đoán **Category** khách hàng sẽ mua dựa trên giao dịch vừa thực hiện.")

# # 2. LOAD MODEL ĐÃ TRAIN
# @st.cache_resource
# def load_model():
#     # Load file .pkl bạn đã tải từ Kaggle về
#     artifacts = joblib.load('bayes_recommendation_model.pkl')
#     return artifacts

# try:
#     artifacts = load_model()
#     model = artifacts['model']
#     enc = artifacts['feature_encoder']
#     le = artifacts['label_encoder']
#     feature_names = artifacts['feature_names']
# except FileNotFoundError:
#     st.error("Không tìm thấy file 'bayes_recommendation_model.pkl'. Hãy copy file model vào cùng thư mục với file app.py")
#     st.stop()

# # 3. TẠO GIAO DIỆN NHẬP LIỆU (SIDEBAR)
# st.sidebar.header("Thông tin giao dịch hiện tại")

# input_data = {}

# # Tự động tạo Selectbox dựa trên dữ liệu đã học từ Encoder
# # enc.categories_ chứa danh sách các giá trị unique của từng cột lúc train
# for i, col_name in enumerate(feature_names):
#     options = list(enc.categories_[i])
#     input_data[col_name] = st.sidebar.selectbox(f"Chọn {col_name}", options)

# # 4. DỰ ĐOÁN
# if st.sidebar.button("Dự đoán hành vi tiếp theo"):
#     # Chuyển input thành DataFrame
#     input_df = pd.DataFrame([input_data])
    
#     # Mã hóa dữ liệu đầu vào (dùng encoder đã load)
#     input_encoded = enc.transform(input_df)
    
#     # Dự đoán class (nhãn)
#     pred_idx = model.predict(input_encoded)
#     pred_label = le.inverse_transform(pred_idx)[0]
    
#     # Dự đoán xác suất (cho biểu đồ)
#     proba = model.predict_proba(input_encoded)[0]
    
#     # Tạo DataFrame kết quả xác suất
#     proba_df = pd.DataFrame({
#         'Category': le.classes_,
#         'Probability': proba
#     }).sort_values(by='Probability', ascending=False)

#     # 5. HIỂN THỊ KẾT QUẢ
#     col1, col2 = st.columns([1, 2])
    
#     with col1:
#         st.success(f"Khách hàng có khả năng cao nhất sẽ mua:")
#         st.metric(label="Next Category", value=pred_label)
#         st.write(f"Độ tin cậy: **{proba_df.iloc[0]['Probability']*100:.1f}%**")
        
#     with col2:
#         st.subheader("Phân phối xác suất các nhóm hàng")
#         # Vẽ biểu đồ cột bằng Plotly
#         fig = px.bar(
#             proba_df, 
#             x='Category', 
#             y='Probability', 
#             color='Probability',
#             color_continuous_scale='Blues',
#             text_auto='.1%'
#         )
#         st.plotly_chart(fig, use_container_width=True)
        
#     # Giải thích thêm (Optional - Giả lập cây Bayes đơn giản)
#     with st.expander("🔍 Chi tiết phân tích Bayes"):
#         st.write("Mô hình Naive Bayes tính toán xác suất dựa trên các yếu tố bạn đã chọn:")
#         st.json(input_data)
#         st.write("Dựa trên lịch sử dữ liệu, đây là tỷ lệ phần trăm khả năng chuyển đổi sang các nhóm hàng khác.")

# else:
#     st.info("👈 Vui lòng chọn thông tin bên trái và bấm nút 'Dự đoán'")
# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import plotly.express as px

# # 1. CẤU HÌNH TRANG
# st.set_page_config(page_title="Dự đoán hành vi mua hàng", layout="wide")

# st.title("🛍️ Hệ thống gợi ý sản phẩm tiếp theo")
# st.markdown("Dự đoán **Category** khách hàng sẽ mua dựa trên giao dịch vừa thực hiện.")

# # 2. LOAD MODEL ĐÃ TRAIN
# @st.cache_resource
# def load_model():
#     try:
#         # Ưu tiên load model CatBoost (nếu có)
#         artifacts = joblib.load('catboost_gpu_model_v2.pkl')
#         return artifacts
#     except FileNotFoundError:
#         try:
#             # Fallback sang model cũ
#             artifacts = joblib.load('bayes_recommendation_model.pkl')
#             return artifacts
#         except FileNotFoundError:
#             return None

# artifacts = load_model()

# if artifacts is None:
#     st.error("⚠️ Không tìm thấy file model (.pkl).")
#     st.warning("Vui lòng tải file model (ví dụ: catboost_gpu_model_v2.pkl) và đặt cùng thư mục với file app.py")
#     st.stop()

# model = artifacts['model']
# # Lấy các thành phần khác (chấp nhận None nếu dùng CatBoost)
# enc = artifacts.get('feature_encoder') 
# le = artifacts['label_encoder']
# feature_names = artifacts.get('feature_names', [
#     'current_category', 'current_subcategory', 'current_articletype', 
#     'customer_gender', 'age_group', 'province'
# ])
# model_type = artifacts.get('model_type', 'naive_bayes')

# # 3. TẠO GIAO DIỆN NHẬP LIỆU (SIDEBAR)
# st.sidebar.header("Thông tin giao dịch hiện tại")

# # Dictionary đổi tên cột sang Tiếng Việt
# column_alias = {
#     'current_category': 'Danh mục chính',
#     'current_subcategory': 'Danh mục phụ',
#     'current_articletype': 'Loại sản phẩm chi tiết',
#     'customer_gender': 'Giới tính',
#     'age_group': 'Nhóm tuổi',
#     'province': 'Tỉnh thành'
# }

# # --- DỮ LIỆU MẪU ĐẦY ĐỦ (Đã làm sạch khoảng trắng) ---
# sample_data = {
#     'current_category': [
#         'accessories', 'apparel', 'personal care', 'footwear', 'free items', 'sporting goods', 'home'
#     ],
#     'current_subcategory': sorted([
#         'belts', 'topwear', 'shoe accessories', 'nails', 'bags', 'fragrance', 'gloves', 'shoes', 'flip flops', 
#         'watches', 'jewellery', 'socks', 'bottomwear', 'innerwear', 'sandal', 'lips', 'headwear', 'saree', 
#         'eyewear', 'ties', 'dress', 'free gifts', 'scarves', 'stoles', 'wallets', 'loungewear and nightwear', 
#         'apparel set', 'cufflinks', 'makeup', 'skin', 'skin care', 'accessories', 'hair', 'wristbands', 'eyes', 
#         'umbrellas', 'perfumes', 'bath and body', 'water bottle', 'sports accessories', 'mufflers', 
#         'sports equipment', 'vouchers', 'beauty accessories', 'home furnishing'
#     ]),
#     'current_articletype': sorted([
#         'belts', 'tshirts', 'shoe accessories', 'kurtas', 'nail polish', 'handbags', 'perfume and body mist', 
#         'gloves', 'casual shoes', 'flip flops', 'backpacks', 'sports shoes', 'watches', 'ring', 'socks', 
#         'salwar', 'necklace and chains', 'briefs', 'sandals', 'shirts', 'mobile pouch', 'formal shoes', 
#         'sports sandals', 'clutches', 'lipstick', 'caps', 'heels', 'lip liner', 'deodorant', 'track pants', 
#         'sarees', 'jackets', 'sweaters', 'tops', 'suspenders', 'sweatshirts', 'sunglasses', 'jeggings', 
#         'lip gloss', 'dresses', 'capris', 'trunk', 'free gifts', 'scarves', 'jeans', 'laptop bag', 'leggings', 
#         'trousers', 'dupatta', 'stoles', 'tunics', 'earrings', 'wallets', 'innerwear vests', 'flats', 'pendant', 
#         'night suits', 'kurta sets', 'bra', 'clothing set', 'cufflinks', 'swimwear', 'shorts', 
#         'highlighter and blush', 'nightdress', 'kurtis', 'bangle', 'eyeshadow', 'messenger bag', 
#         'face moisturisers', 'tablet sleeve', 'face wash and cleanser', 'kajal and eyeliner', 'skirts', 
#         'fragrance gift set', 'patiala', 'accessory gift set', 'hair colour', 'compact', 'boxers', 'tracksuits', 
#         'concealer', 'lounge shorts', 'lounge tshirts', 'wristbands', 'rain jacket', 'rucksacks', 'tights', 
#         'hat', 'duffel bag', 'baby dolls', 'foundation and primer', 'bracelet', 'jewellery set', 'suits', 
#         'travel accessory', 'lounge pants', 'mascara', 'umbrellas', 'eye cream', 'sunscreen', 'waistcoat', 
#         'bath robe', 'nehru jackets', 'booties', 'body lotion', 'mask and peel', 'camisoles', 'lip care', 
#         'stockings', 'toner', 'rompers', 'churidar', 'water bottle', 'face scrub and exfoliator', 'mufflers', 
#         'basketballs', 'footballs', 'salwar and dupatta', 'shapewear', 'nail essentials', 'shrug', 'shoe laces', 
#         'jumpsuit', 'ties and cufflinks', 'hair accessory', 'ipad', 'waist pouch', 'lip plumper', 
#         'body wash and scrub', 'rain trousers', 'beauty accessory', 'makeup remover', 'robe', 'headband', 
#         'mens grooming kit', 'key chain', 'face serum and gel', 'trolley bag', 'blazers', 'lehenga choli', 
#         'cushion covers'
#     ]),
#     'customer_gender': ['M', 'F'],
#     'age_group': ['1', '2', '3', '4', 'u'], # Đã sắp xếp lại
#     'province': sorted([
#         'ACEH', 'BALI', 'BANGKA BELITUNG', 'BANTEN', 'BENGKULU', 'GORONTALO', 'JAKARTA RAYA', 'JAMBI', 
#         'JAWA BARAT', 'JAWA TENGAH', 'JAWA TIMUR', 'KALIMANTAN BARAT', 'KALIMANTAN SELATAN', 'KALIMANTAN TENGAH', 
#         'KALIMANTAN TIMUR', 'KEPULAUAN RIAU', 'LAMPUNG', 'MALUKU', 'MALUKU UTARA', 'NUSA TENGGARA BARAT', 
#         'NUSA TENGGARA TIMUR', 'PAPUA', 'PAPUA BARAT', 'RIAU', 'SULAWESI BARAT', 'SULAWESI SELATAN', 
#         'SULAWESI TENGAH', 'SULAWESI TENGGARA', 'SULAWESI UTARA', 'SUMATERA BARAT', 'SUMATERA SELATAN', 
#         'SUMATERA UTARA', 'YOGYAKARTA'
#     ])
# }

# input_data = {}

# # Hàm làm sạch hiển thị
# def clean_display_text(text):
#     return str(text).strip().title()

# # VÒNG LẶP TẠO INPUT
# for col_name in feature_names:
#     display_name = column_alias.get(col_name, col_name)
    
#     # Ưu tiên lấy options từ Encoder (nếu có - trường hợp Naive Bayes)
#     if enc and hasattr(enc, 'categories_'):
#         # Tìm index của cột trong encoder
#         try:
#             # Lưu ý: feature_names phải khớp thứ tự với encoder
#             idx = feature_names.index(col_name) 
#             options = list(enc.categories_[idx])
#         except:
#             options = sample_data.get(col_name, [])
#     else:
#         # Trường hợp CatBoost (hoặc không có encoder), lấy từ sample_data
#         options = sample_data.get(col_name, [])

#     # Nếu không tìm thấy list option, fallback về text input
#     if not options:
#         input_data[col_name] = st.sidebar.text_input(display_name, "Nhập giá trị...")
#     else:
#         # Selectbox chọn giá trị
#         selected_val = st.sidebar.selectbox(
#             label=display_name,
#             options=options,
#             format_func=clean_display_text
#         )
#         input_data[col_name] = selected_val

# # 4. DỰ ĐOÁN
# if st.sidebar.button("Dự đoán hành vi tiếp theo"):
#     # Tạo DataFrame từ input
#     input_df = pd.DataFrame([input_data])
    
#     # --- QUAN TRỌNG: LÀM SẠCH DATA TRƯỚC KHI GỬI VÀO MODEL ---
#     # Vì dữ liệu mẫu ở trên đã clean, nên ta gửi data clean vào model.
#     # Nếu Model của bạn train bằng data bẩn (có khoảng trắng), nó có thể không hiểu.
#     # Tuy nhiên, CatBoost thường tự xử lý tốt.
    
#     try:
#         # Logic dự đoán
#         if model_type == 'naive_bayes' and enc:
#             input_encoded = enc.transform(input_df)
#             if np.any(input_encoded < 0): input_encoded[input_encoded < 0] = 0
#             pred_idx = model.predict(input_encoded)
#             pred_label = le.inverse_transform(pred_idx.flatten())[0]
#             proba = model.predict_proba(input_encoded)[0]
#         else:
#             # CatBoost / Random Forest
#             pred_idx = model.predict(input_df)
#             pred_label = le.inverse_transform(pred_idx.flatten())[0]
#             proba = model.predict_proba(input_df)[0]

#         # Hiển thị kết quả
#         proba_df = pd.DataFrame({
#             'Category': le.classes_,
#             'Probability': proba
#         }).sort_values(by='Probability', ascending=False)

#         st.divider()
#         col1, col2 = st.columns([1, 2])
        
#         with col1:
#             st.success("🎯 KẾT QUẢ DỰ BÁO")
#             st.metric(label="Khách hàng sẽ mua:", value=clean_display_text(pred_label))
#             st.write(f"Độ tin cậy: **{proba_df.iloc[0]['Probability']*100:.1f}%**")
            
#         with col2:
#             st.subheader("📊 Phân phối xác suất")
#             fig = px.bar(
#                 proba_df, x='Category', y='Probability', 
#                 color='Probability', color_continuous_scale='Teal', text_auto='.1%'
#             )
#             st.plotly_chart(fig, use_container_width=True)

#     except Exception as e:
#         st.error(f"Lỗi khi dự báo: {e}")
#         st.info("Gợi ý: Hãy đảm bảo dữ liệu input (đã clean) khớp với dữ liệu lúc train model.")
# else:
#     st.info("👈 Chọn thông tin bên trái và bấm 'Dự đoán'")
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ================================
# 1. CẤU HÌNH TRANG
# ================================
st.set_page_config(page_title="Dự đoán Category kế tiếp", layout="wide")
st.title("🛍️ Hệ thống gợi ý danh mục sản phẩm tiếp theo (Naive Bayes)")
st.markdown("Dự đoán **Category** mà khách hàng có khả năng mua tiếp theo.")

# ================================
# 2. LOAD MODEL
# ================================
@st.cache_resource
def load_model():
    try:
        artifacts = joblib.load("bayes_recommendation_model.pkl")
        return artifacts
    except:
        return None

artifacts = load_model()

if artifacts is None:
    st.error("❌ Không tìm thấy file bayes_recommendation_model.pkl")
    st.stop()

model = artifacts["model"]
enc = artifacts["feature_encoder"]
le = artifacts["label_encoder"]
feature_names = artifacts["feature_names"]

# ================================
# 3. GIAO DIỆN NHẬP LIỆU
# ================================
st.sidebar.header("Nhập thông tin giao dịch hiện tại")

column_alias = {
    'current_category': 'Danh mục chính',
    'current_subcategory': 'Danh mục phụ',
    'current_articletype': 'Loại sản phẩm chi tiết',
    'customer_gender': 'Giới tính khách hàng',
    'age_group': 'Nhóm tuổi',
    'province': 'Tỉnh thành'
}

input_data = {}

def clean_display(text):
    return str(text).strip().title()

# Sinh input từ encoder (chính xác 100%)
for idx, col in enumerate(feature_names):
    options = list(enc.categories_[idx])  # lấy thẳng từ model
    selected = st.sidebar.selectbox(
        column_alias.get(col, col),
        options=options,
        format_func=clean_display
    )
    input_data[col] = selected

# ================================
# 4. DỰ ĐOÁN
# ================================
if st.sidebar.button("🔮 Dự đoán"):
    try:
        df_input = pd.DataFrame([input_data])

        # Encode
        X_encoded = enc.transform(df_input)

        # Predict
        pred_idx = model.predict(X_encoded)[0]
        pred_label = le.inverse_transform([pred_idx])[0]

        proba = model.predict_proba(X_encoded)[0]

        # Chuẩn bị dataframe hiển thị probability
        proba_df = pd.DataFrame({
            "Category": le.classes_,
            "Probability": proba
        }).sort_values("Probability", ascending=False)

        # ======================
        # HIỂN THỊ KẾT QUẢ
        # ======================
        st.divider()
        col1, col2 = st.columns([1, 2])

        with col1:
            st.success("🎯 Kết quả dự đoán")
            st.metric("Khách có khả năng mua:", clean_display(pred_label))
            st.write(f"Độ tin cậy: **{proba_df.iloc[0]['Probability']*100:.1f}%**")

        with col2:
            st.subheader("📊 Xác suất chi tiết")
            fig = px.bar(
                proba_df,
                x="Category",
                y="Probability",
                text_auto=".2%",
                color="Probability",
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Lỗi dự đoán: {e}")

else:
    st.info("👈 Nhập thông tin bên trái và nhấn **Dự đoán**")
