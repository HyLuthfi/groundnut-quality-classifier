import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Konfigurasi Halaman (Harus di awal)
st.set_page_config(
    page_title="AI Penyortir Kacang Tanah",
    page_icon="🥜",
    layout="centered"
)

# Custom CSS Premium (Glassmorphism & Dark Mode Aesthetic)
css_kustom = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Latar belakang utama aplikasi */
    .stApp {
        background: linear-gradient(135deg, #12141D 0%, #1A2035 100%);
        color: #E2E8F0;
    }

    /* Container putih/kaca untuk elemen */
    div.css-1r6slb0, div.css-12oz5g7 {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* Kustomisasi Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Tombol Utama */
    .stButton > button {
        background: linear-gradient(90deg, #4F46E5 0%, #3B82F6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -10px rgba(59, 130, 246, 0.5);
        color: white;
        border: none;
    }

    /* Teks Header */
    h1 {
        background: -webkit-linear-gradient(45deg, #60A5FA, #A78BFA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
        text-align: center;
        margin-bottom: 0 !important;
    }

    /* Uploader */
    [data-testid="stFileUploadDropzone"] {
        background-color: rgba(255, 255, 255, 0.02);
        border: 2px dashed rgba(255, 255, 255, 0.2);
        border-radius: 12px;
    }
    
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #3B82F6;
        background-color: rgba(59, 130, 246, 0.05);
    }
</style>
"""
st.markdown(css_kustom, unsafe_allow_html=True)

# Konstanta
UKURAN_INPUT = (224, 224)
KELAS = ['Bersih', 'Kotor (Kontaminasi)']
PATH_MODEL_MOB = 'models/mobilenetv2_best.h5'

# Cache untuk load model
@st.cache_resource
def muat_model():
    if not os.path.exists(PATH_MODEL_MOB):
        return None
    return tf.keras.models.load_model(PATH_MODEL_MOB)

# Fungsi Preprocessing
def praproses_citra(citra):
    citra = citra.convert('RGB')
    citra = citra.resize(UKURAN_INPUT)
    array_citra = tf.keras.preprocessing.image.img_to_array(citra)
    array_citra = np.expand_dims(array_citra, axis=0)
    array_citra = tf.keras.applications.mobilenet_v2.preprocess_input(array_citra)
    return array_citra

# SIDEBAR (Pengaturan)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/751/751508.png", width=80)
    st.title("⚙️ Engine Spesifikasi")
    st.markdown("Aplikasi web ini dideploy secara dinamis menggunakan arsitektur ringan (*Lightweight Architecture*) untuk mendemonstrasikan efisiensi komputasi pada *edge device*.")
    
    st.markdown("---")
    st.info("⚡ **Model Aktif: MobileNetV2**\n- Parameter: ~3.4 Juta\n- Ukuran: ~21 MB\n- Akurasi Testing: 96%")
    st.markdown("*(Catatan: Model VGG16 dengan 138 Juta parameter tidak dimuat pada versi web ini demi optimasi kecepatan server)*")

# MAIN INTERFACE
st.markdown("<h1>Detektor Kualitas Kacang Tanah 🥜</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94A3B8; margin-bottom: 2rem;'>Sistem penyortiran otomatis menggunakan Deep Learning (SNI 01-3921-1995)</p>", unsafe_allow_html=True)

# Muat model
model_aktif = muat_model()

if model_aktif is None:
    st.error(f"⚠️ **File model tidak ditemukan!**\nPastikan file bobot (`mobilenetv2_best.h5`) tersedia di dalam folder `models/`.")
else:
    # Komponen Upload
    col1, col2 = st.columns([1, 1])
    
    with col1:
        berkas_unggah = st.file_uploader("Unggah Foto Kacang (JPG/PNG)", type=['jpg', 'jpeg', 'png'])
        
    with col2:
        if berkas_unggah is not None:
            citra_asli = Image.open(berkas_unggah)
            st.image(citra_asli, caption="Citra Diunggah", use_container_width=True)
        else:
            st.info("☝️ Unggah citra di sebelah kiri untuk melihat *preview*.")

    # Tombol Prediksi
    if berkas_unggah is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍 Analisis Citra"):
            with st.spinner('Mengekstrak fitur visual...'):
                # Proses prediksi
                tensor_input = praproses_citra(citra_asli)
                probabilitas = model_aktif.predict(tensor_input)[0][0]
                
                # Kelas biner: 0 = Bersih, 1 = Kotor
                if probabilitas > 0.5:
                    hasil_kelas = KELAS[1] # Kotor
                    tingkat_keyakinan = probabilitas * 100
                    warna = "#EF4444" # Merah
                    icon = "🚨"
                else:
                    hasil_kelas = KELAS[0] # Bersih
                    tingkat_keyakinan = (1 - probabilitas) * 100
                    warna = "#10B981" # Hijau
                    icon = "✅"
                
                st.markdown("---")
                st.markdown(f"<h3 style='text-align:center;'>Hasil Analisis</h3>", unsafe_allow_html=True)
                
                # Tampilan hasil ala metrik dashboard
                col_hasil1, col_hasil2 = st.columns(2)
                with col_hasil1:
                    st.markdown(
                        f"""
                        <div style='background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 12px; border-left: 5px solid {warna}; text-align: center;'>
                            <p style='margin:0; font-size: 1rem; color: #94A3B8;'>Status Mutu</p>
                            <h2 style='margin:0; color: {warna};'>{icon} {hasil_kelas}</h2>
                        </div>
                        """, unsafe_allow_html=True
                    )
                with col_hasil2:
                    st.markdown(
                        f"""
                        <div style='background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 12px; border-left: 5px solid #3B82F6; text-align: center;'>
                            <p style='margin:0; font-size: 1rem; color: #94A3B8;'>Tingkat Keyakinan (Confidence)</p>
                            <h2 style='margin:0; color: #60A5FA;'>{tingkat_keyakinan:.2f}%</h2>
                        </div>
                        """, unsafe_allow_html=True
                    )
