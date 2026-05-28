import streamlit as st
import numpy as np
from PIL import Image
import os
import tensorflow as tf
import time
from tensorflow.keras.applications import VGG16, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# Konfigurasi Halaman
st.set_page_config(
    page_title="Penyortir Kacang Tanah SNI",
    page_icon="🥜",
    layout="wide"
)

# Custom CSS: Clean, White, Elegant, Minimalist
css_kustom = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #FAFAFA;
        color: #111827;
    }
    
    .stApp {
        background-color: #FAFAFA;
    }

    /* Card styling for modern white look */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 32px 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        text-align: center;
        transition: transform 0.2s ease-in-out;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }

    h1, h2, h3 {
        color: #111827 !important;
        font-weight: 700 !important;
    }

    .stButton > button {
        background-color: #111827;
        color: white;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border: none;
        width: 100%;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #374151;
        color: white;
    }
    
    .status-bersih {
        color: #059669;
        background-color: #D1FAE5;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
    }
    
    .status-kotor {
        color: #DC2626;
        background-color: #FEE2E2;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
    }
    
    hr {
        border-color: #E5E7EB;
    }
</style>
"""
st.markdown(css_kustom, unsafe_allow_html=True)

# Konstanta
UKURAN_INPUT = (224, 224)
KELAS = ['Bersih', 'Kotor (Kontaminasi)']

def get_model_path(nama_model):
    filename = 'vgg16_best.h5' if nama_model == 'VGG16' else 'mobilenetv2_best.h5'
    if os.path.exists(f'models/{filename}'):
        return f'models/{filename}'
    if os.path.exists(filename):
        return filename
    return None

@st.cache_resource
def muat_model(nama_model):
    path = get_model_path(nama_model)
    if path is None:
        return None
    try:
        if nama_model == 'VGG16':
            base_model = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
        else:
            base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
            
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.load_weights(path)
        return model
    except Exception as e:
        return None

def praproses_citra(citra, nama_model):
    citra = citra.convert('RGB').resize(UKURAN_INPUT)
    array_citra = tf.keras.preprocessing.image.img_to_array(citra)
    array_citra = np.expand_dims(array_citra, axis=0)
    if nama_model == 'VGG16':
        return tf.keras.applications.vgg16.preprocess_input(array_citra)
    return tf.keras.applications.mobilenet_v2.preprocess_input(array_citra)

# HEADER
st.markdown("<h1 style='text-align: center; margin-top: 1rem;'>Sistem Inspeksi Mutu Kacang Tanah</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1rem; color: #6B7280; margin-bottom: 3rem;'>Analisis Kualitas Berbasis Kecerdasan Buatan (Standardisasi SNI 01-3921-1995)</p>", unsafe_allow_html=True)

# Muat Kedua Model Sekaligus
model_vgg = muat_model('VGG16')
model_mob = muat_model('MobileNetV2')

if model_vgg is None or model_mob is None:
    st.error("⚠️ File bobot model tidak ditemukan di dalam direktori.")
    st.stop()

# UPLOAD SECTION - Centered and Clean
kolom_kiri, kolom_tengah, kolom_kanan = st.columns([1, 2, 1])

with kolom_tengah:
    berkas_unggah = st.file_uploader("Unggah citra sampel kacang tanah (JPG/PNG)", type=['jpg', 'jpeg', 'png'])
    
    if berkas_unggah is not None:
        citra_asli = Image.open(berkas_unggah)
        st.image(citra_asli, caption="Sampel Inspeksi", use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        tombol_prediksi = st.button("Mulai Inspeksi Visual")

# HASIL PREDIKSI - Symmetrical Head-to-Head
if berkas_unggah is not None and 'tombol_prediksi' in locals() and tombol_prediksi:
    st.markdown("<hr style='margin: 3rem 0;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>Laporan Hasil Inspeksi</h2>", unsafe_allow_html=True)
    
    with st.spinner("Memproses analisis komparatif secara paralel..."):
        # Prediksi VGG16
        start_vgg = time.time()
        tensor_vgg = praproses_citra(citra_asli, 'VGG16')
        prob_vgg = model_vgg.predict(tensor_vgg)[0][0]
        waktu_vgg = time.time() - start_vgg
        
        # Prediksi MobileNetV2
        start_mob = time.time()
        tensor_mob = praproses_citra(citra_asli, 'MobileNetV2')
        prob_mob = model_mob.predict(tensor_mob)[0][0]
        waktu_mob = time.time() - start_mob
        
        def format_hasil(probabilitas):
            is_kotor = probabilitas > 0.5
            kelas = KELAS[1] if is_kotor else KELAS[0]
            confidence = probabilitas if is_kotor else (1 - probabilitas)
            style_class = "status-kotor" if is_kotor else "status-bersih"
            icon = "⚠️" if is_kotor else "✓"
            return kelas, confidence * 100, style_class, icon

        kelas_vgg, conf_vgg, style_vgg, icon_vgg = format_hasil(prob_vgg)
        kelas_mob, conf_mob, style_mob, icon_mob = format_hasil(prob_mob)

        # Symmetrical Layout
        col_vgg, col_mob = st.columns(2)
        
        with col_vgg:
            st.markdown(f"""
            <div class='metric-card'>
                <h3 style='color: #4B5563; font-size: 1.2rem; margin-bottom: 1.5rem;'>Arsitektur VGG-16</h3>
                <div style='margin-bottom: 2rem;'>
                    <span class='{style_vgg}'>{icon_vgg} {kelas_vgg}</span>
                </div>
                <p style='margin: 0; font-size: 0.9rem; color: #6B7280;'>Tingkat Akurasi (Confidence)</p>
                <h2 style='margin: 0 0 1.5rem 0; color: #111827;'>{conf_vgg:.2f}%</h2>
                <div style='background-color: #F3F4F6; padding: 8px; border-radius: 6px; display: inline-block;'>
                    <span style='font-size: 0.85rem; color: #4B5563;'>Waktu Inferensi: <b>{waktu_vgg:.3f} s</b></span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_mob:
            st.markdown(f"""
            <div class='metric-card'>
                <h3 style='color: #4B5563; font-size: 1.2rem; margin-bottom: 1.5rem;'>Arsitektur MobileNetV2</h3>
                <div style='margin-bottom: 2rem;'>
                    <span class='{style_mob}'>{icon_mob} {kelas_mob}</span>
                </div>
                <p style='margin: 0; font-size: 0.9rem; color: #6B7280;'>Tingkat Akurasi (Confidence)</p>
                <h2 style='margin: 0 0 1.5rem 0; color: #111827;'>{conf_mob:.2f}%</h2>
                <div style='background-color: #F3F4F6; padding: 8px; border-radius: 6px; display: inline-block;'>
                    <span style='font-size: 0.85rem; color: #4B5563;'>Waktu Inferensi: <b>{waktu_mob:.3f} s</b></span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br><p style='text-align: center; font-size: 0.9rem; color: #9CA3AF;'>*Hasil di atas merupakan perbandingan langsung (head-to-head) kecepatan dan akurasi antara VGG16 dan MobileNetV2.</p>", unsafe_allow_html=True)
