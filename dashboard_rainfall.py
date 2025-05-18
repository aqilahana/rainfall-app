import streamlit as st
import numpy as np
import joblib
import base64
# from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
# pip install scikit-learn==1.2.2

# Fungsi untuk konversi gambar ke base64 (digunakan untuk background)
def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Menambahkan background image dari file lokal
background_image = get_base64("background.jpg")
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{background_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Menambahkan dua logo instansi di kiri atas menggunakan columns
logo_col1, logo_col2 = st.columns([1, 6])
with logo_col1:
    st.image("logo-bmkg.png", width=70)
with logo_col2:
    st.image("logo-unair.png", width=70)

# Load model secara benar
try:
    model = joblib.load("best_xgb.pkl")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# Dashboard Title
st.title("Dashboard Klasifikasi Curah Hujan")
#st.write("Masukkan nilai hanya berupa angka saja")
# Penjelasan tentang dashboard
st.markdown("""
Dashboard ini merupakan alat bantu untuk memprediksi **kategori curah hujan harian** berdasarkan parameter cuaca menggunakan model **XGBoost** yang telah dilatih sebelumnya. 

##### Keterangan:
> 1. Pastikan semua nilai diisi dengan angka, kecuali arah angin terbanyak yang bisa dipilih dari daftar.
> 2. Pastikan tidak ada nilai parameter yang bernilai nol atau negatif
> 3. Masukkan nilai sesuai dengan skala yang telah ditentukan dengan rentang nilai yang wajar
> 4. Setelah semua data diisi, tekan tombol **"Klasifikasi Curah Hujan"**
""")


# Wind direction mapping
arah_angin_mapping = {
    "North (N)": 1,
    "Northeast (NE)": 2,
    "East (E)": 3,
    "Southeast (SE)": 4,
    "South (S)": 5,
    "Southwest (SW)": 6,
    "West (W)": 7,
    "Northwest (NW)": 8
}

# Split layout into two columns
col1, col2 = st.columns(2)

# User input features in two columns
with col1:
    tn = st.number_input("Suhu Minimum (°C)", min_value=0.0, max_value=100.0)
    tx = st.number_input("Suhu Maksimum (°C)", min_value=0.0, max_value=100.0)
    tavg = st.number_input("Suhu Rata-Rata (°C)", min_value=0.0, max_value=100.0)
    rhavg = st.number_input("Kelembaban Rata-Rata (%)", min_value=0, max_value=100, step=1)

with col2:
    ss = st.number_input("Lama Penyinaran Matahari (jam)", min_value=0.0, max_value=24.0)
    ffx = st.number_input("Kecepatan Angin Maksimum (m/s)", min_value=0, max_value=100, step=1)
    ffavg = st.number_input("Kecepatan Angin Rata-Rata (m/s)", min_value=0, max_value=100, step=1)
    dddx0 = st.number_input("Arah Angin pada Kecepatan Maksimum (°)", min_value=0.0, max_value=360.0)

# Convert common wind direction to numeric value
dddcar0 = st.selectbox("Arah Angin Terbanyak", list(arah_angin_mapping.keys()))
dddcar = arah_angin_mapping[dddcar0]

# Convert dddx from degrees to radians
dddx = np.deg2rad(dddx0)

# Load pre-trained scalers instead of fitting on single data point
try:
    scaler_robust = joblib.load("scaler_robust.pkl")
    scaler_minmax = joblib.load("scaler_minmax.pkl")
except FileNotFoundError:
    st.warning("Scaler files not found. Menggunakan default scaling yang mungkin memengaruhi akurasi prediksi.")
    # scaler_robust = RobustScaler()
    # scaler_minmax = MinMaxScaler()

# Convert to 2D array for normalization
features = np.array([[tn, tx, tavg, rhavg, ss, ffx, dddx, ffavg]])

# Apply transformations
try:
    temp_wind_features = features[:, [0, 1, 2, 5, 7]]
    scaled_temp_wind = scaler_robust.transform(temp_wind_features)
    features[:, [0, 1, 2, 5, 7]] = scaled_temp_wind

    other_features = features[:, [3, 4, 6]]
    scaled_other = scaler_minmax.transform(other_features)
    # features[:, [3, 4, 6]] = scaled_other
except:
    st.error(f"Kesalahan normalisasi: {e}")

# Gabungkan dengan dddcar (sudah numerik)
input_data = np.hstack([features, np.array([[dddcar]])])

# Validasi input: jika ada nilai yang masih 0
input_cek = [tn, tx, tavg, rhavg, ss, ffx, ffavg, dddx0]
jika_ada_nol = any(nilai == 0 for nilai in input_cek)

# Tombol prediksi
if st.button("Klasifikasi Curah Hujan"):
    if jika_ada_nol:
        st.warning("⚠️ Silakan isi semua parameter terlebih dahulu sebelum melakukan klasifikasi. Nilai tidak boleh nol.")
    else:
        try:
            prediction = model.predict(input_data)[0]
            klasifikasi_hujan = {
                0: "Hujan Sangat Ringan",
                1: "Hujan Ringan",
                2: "Hujan Sedang",
                3: "Hujan Lebat",
                4: "Hujan Sangat Lebat"
            }
            hasil = klasifikasi_hujan.get(prediction, "Kategori Tidak Diketahui")
            st.subheader(f"**Kategori: {hasil}**")

        except Exception as e:
            st.error(f"Error dalam melakukan prediksi: {e}")
            st.info("Pastikan semua nilai input sudah diisi dengan benar.")


st.markdown("""
**Deskripsi Singkat:**

- 🌦️ **Hujan Sangat Ringan** - Gerimis sebentar, tidak mengganggu aktivitas luar.

- 🌧️ **Hujan Ringan** -  Hujan kecil yang terjadi terus menerus selama beberapa jam.

- 🌧️ **Hujan Sedang** - Hujan mulai deras, bisa menyebabkan genangan kecil.

- ⛈️ **Hujan Lebat** - Hujan deras, bisa menyebabkan banjir kecil di beberapa tempat.

- 🌪️ **Hujan Sangat Lebat** - Badai deras, biasanya disertai petir dan angin kencang.
""")



