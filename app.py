import streamlit as st
import pandas as pd
import pickle

# 1. Load Model yang sudah disave
model = pickle.load(open('xgb_churn_model.pkl', 'rb'))

# Dummy blueprint kolom X_train (pastikan urutan sama saat training)
# (Di real-project, blueprint ini juga di-save sebagai .pkl)
kolom_fitur = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 
               'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_No phone service', 
               'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No', 
               'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No internet service', 
               'OnlineBackup_Yes', 'DeviceProtection_No internet service', 'DeviceProtection_Yes', 
               'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No internet service', 
               'StreamingTV_Yes', 'StreamingMovies_No internet service', 'StreamingMovies_Yes', 
               'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes', 
               'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 
               'PaymentMethod_Mailed check']

# 2. Desain UI Aplikasi Web
st.set_page_config(page_title="Prediksi Churn", layout="centered")
st.title("📊 Aplikasi Prediksi Retensi Pelanggan")
st.write("Masukkan data pelanggan di bawah ini untuk melihat probabilitas *Churn*.")

# Input Form
with st.form("form_prediksi"):
    col1, col2 = st.columns(2)
    
    with col1:
        tenure = st.number_input("Lama Berlangganan (Bulan)", min_value=0, max_value=72, value=12)
        MonthlyCharges = st.number_input("Tagihan Bulanan ($)", min_value=0.0, value=50.0)
        TotalCharges = st.number_input("Total Tagihan ($)", min_value=0.0, value=600.0)
        
    with col2:
        Contract = st.selectbox("Tipe Kontrak", ["Month-to-month", "One year", "Two year"])
        Internet = st.selectbox("Layanan Internet", ["DSL", "Fiber optic", "No"])
        
    submit = st.form_submit_button("Prediksi Risiko")

# 3. Logika Prediksi ketika tombol ditekan
if submit:
    # Buat dataframe kosong 1 baris
    df_input = pd.DataFrame(columns=kolom_fitur)
    df_input.loc[0] = 0 # Set default 0
    
    # Isi dengan nilai dari UI
    df_input['tenure'] = tenure
    df_input['MonthlyCharges'] = MonthlyCharges
    df_input['TotalCharges'] = TotalCharges
    
    # Mapping Dropdown ke format One-Hot Encoding
    if Contract == 'One year': df_input['Contract_One year'] = 1
    if Contract == 'Two year': df_input['Contract_Two year'] = 1
    if Internet == 'Fiber optic': df_input['InternetService_Fiber optic'] = 1
    if Internet == 'No': df_input['InternetService_No'] = 1
    
    # Lakukan Prediksi (Ambil probabilitas kelas 1 / Churn)
    prob_churn = model.predict_proba(df_input)[0][1]
    
    # Tampilkan Hasil
    st.markdown("---")
    if prob_churn > 0.5:
        st.error(f"🚨 **RISIKO TINGGI!** Pelanggan diprediksi CHURN dengan probabilitas {prob_churn*100:.2f}%")
        st.write("💡 *Saran: Segera jadwalkan panggilan dari Tim Retention.*")
    else:
        st.success(f"✅ **AMAN.** Probabilitas churn hanya {prob_churn*100:.2f}%")
        st.write("💡 *Saran: Pelanggan loyal, tidak perlu intervensi khusus.*")
