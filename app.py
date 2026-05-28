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
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
css_kustom = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #F8FAFC;
        color: #1E293B;
    }
    
    .stApp {
        background-color: #F8FAFC;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #64748B;
        font-weight: 600;
        font-size: 1.15rem;
    }
    
    .stTabs [aria-selected="true"] {
        color: #0F172A !important;
        border-bottom: 3px solid #3B82F6 !important;
    }

    /* Card styling */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 32px 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        text-align: center;
        transition: transform 0.2s ease-in-out;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    .info-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #3B82F6;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }

    h1, h2, h3 {
        color: #0F172A !important;
        font-weight: 700 !important;
    }

    .stButton > button {
        background-color: #0F172A;
        color: white;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border: none;
        width: 100%;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #334155;
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
        border-color: #E2E8F0;
    }
    
    /* Markdown Text Styling */
    p, li {
        font-size: 1.05rem;
        line-height: 1.7;
        color: #334155;
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
st.markdown("<h1 style='text-align: center; margin-top: 1rem;'>Sistem Inspeksi Mutu Kacang Tanah SNI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1rem; color: #64748B; margin-bottom: 2rem;'>Sistem Otomatisasi Penilaian Kualitas Fisik Menggunakan Deep Learning</p>", unsafe_allow_html=True)

# TABS NAVIGATION
tab_prediksi, tab_mobilenet, tab_vgg = st.tabs(["🔍 Prediksi Mutu", "⚡ MobileNetV2", "🧠 VGG16"])

# --- TAB 1: PREDIKSI ---
with tab_prediksi:
    st.markdown("<br>", unsafe_allow_html=True)
    
    model_vgg = muat_model('VGG16')
    model_mob = muat_model('MobileNetV2')

    if model_vgg is None or model_mob is None:
        st.warning("⚠️ Bobot model belum tersedia. Menunggu inisialisasi file `.h5` pada server.")
    else:
        kolom_kiri, kolom_tengah, kolom_kanan = st.columns([1, 2, 1])

        with kolom_tengah:
            st.markdown("<div class='info-card'><b>Instruksi:</b> Unggah citra kacang tanah dengan pencahayaan yang cukup dan resolusi yang jelas untuk mendapatkan hasil prediksi yang optimal.</div>", unsafe_allow_html=True)
            berkas_unggah = st.file_uploader("Pilih file citra (JPG/PNG)", type=['jpg', 'jpeg', 'png'])
            
            if berkas_unggah is not None:
                citra_asli = Image.open(berkas_unggah)
                st.image(citra_asli, caption="Sampel Inspeksi", use_container_width=True)
                st.markdown("<br>", unsafe_allow_html=True)
                tombol_prediksi = st.button("Mulai Inspeksi Visual Paralel")

        if berkas_unggah is not None and 'tombol_prediksi' in locals() and tombol_prediksi:
            st.markdown("<hr style='margin: 3rem 0;'>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>Laporan Hasil Inspeksi Head-to-Head</h2>", unsafe_allow_html=True)
            
            with st.spinner("Mengeksekusi jaringan komputasi..."):
                start_vgg = time.time()
                tensor_vgg = praproses_citra(citra_asli, 'VGG16')
                prob_vgg = model_vgg.predict(tensor_vgg)[0][0]
                waktu_vgg = time.time() - start_vgg
                
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

                col_vgg, col_mob = st.columns(2)
                
                with col_vgg:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3 style='color: #475569; font-size: 1.2rem; margin-bottom: 1.5rem;'>Model VGG-16</h3>
                        <div style='margin-bottom: 2rem;'>
                            <span class='{style_vgg}'>{icon_vgg} {kelas_vgg}</span>
                        </div>
                        <p style='margin: 0; font-size: 0.9rem; color: #64748B;'>Tingkat Keyakinan</p>
                        <h2 style='margin: 0 0 1.5rem 0; color: #0F172A;'>{conf_vgg:.2f}%</h2>
                        <div style='background-color: #F1F5F9; padding: 8px; border-radius: 6px; display: inline-block;'>
                            <span style='font-size: 0.85rem; color: #475569;'>Waktu Eksekusi: <b>{waktu_vgg:.3f} detik</b></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_mob:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3 style='color: #475569; font-size: 1.2rem; margin-bottom: 1.5rem;'>Model MobileNetV2</h3>
                        <div style='margin-bottom: 2rem;'>
                            <span class='{style_mob}'>{icon_mob} {kelas_mob}</span>
                        </div>
                        <p style='margin: 0; font-size: 0.9rem; color: #64748B;'>Tingkat Keyakinan</p>
                        <h2 style='margin: 0 0 1.5rem 0; color: #0F172A;'>{conf_mob:.2f}%</h2>
                        <div style='background-color: #F1F5F9; padding: 8px; border-radius: 6px; display: inline-block;'>
                            <span style='font-size: 0.85rem; color: #475569;'>Waktu Eksekusi: <b>{waktu_mob:.3f} detik</b></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# --- TAB 2: MOBILENETV2 ---
with tab_mobilenet:
    st.markdown("<div style='padding: 2rem;'>", unsafe_allow_html=True)
    st.markdown("<h2>Arsitektur MobileNetV2</h2>", unsafe_allow_html=True)
    st.markdown("""
    MobileNetV2 adalah arsitektur *Convolutional Neural Network* (CNN) yang dirancang khusus untuk efisiensi komputasi, menjadikannya standar industri untuk penerapan pada sistem berbasis seluler (*mobile*) dan *edge devices*.
    
    ### ⚙️ Spesifikasi Teknis dalam Penelitian
    - **Total Parameter Beban**: ~3.4 Juta
    - **Fungsi Aktivasi Output**: Sigmoid (Binary Classification)
    - **Teknik Optimasi**: Menggunakan *Inverted Residuals* & *Linear Bottlenecks* untuk mengurangi kelebihan beban komputasi.
    - **Metrik Evaluasi Akhir**:
      - Akurasi (*Accuracy*): Mendekati **96%**
      - Profil Eksekusi: Sangat Cepat, sangat efisien dari sisi penggunaan RAM dan CPU.
      
    ### 🔄 Alur Pemrosesan (Preprocessing)
    1. **Resizing**: Mengubah dimensi citra menjadi standar input $224 \\times 224$ piksel.
    2. **Normalisasi**: Nilai intensitas piksel diskalakan secara otomatis ke dalam rentang $[-1, 1]$ menggunakan teknik bawaan pustaka `tf.keras.applications.mobilenet_v2.preprocess_input`.
    
    ### 🏗️ Modifikasi Head (Transfer Learning)
    Pada penelitian ini, lapisan klasifikasi bawaan ImageNet pada MobileNetV2 dibuang (*include_top=False*) dan digantikan dengan lapisan khusus yang kita rancang:
    - `GlobalAveragePooling2D()`: Berfungsi untuk mereduksi dimensi spasial secara drastis untuk mencegah terjadinya *overfitting*.
    - `Dense(128, ReLU)`: Berfungsi sebagai lapisan pengekstraksi fitur tingkat tinggi yang telah dikompres.
    - `Dropout(0.3)`: Menonaktifkan 30% saraf secara acak untuk meningkatkan daya generalisasi model.
    - `Dense(1, Sigmoid)`: Sebagai lapisan determinasi final untuk membedakan kelas **Bersih** vs **Kotor**.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 3: VGG16 ---
with tab_vgg:
    st.markdown("<div style='padding: 2rem;'>", unsafe_allow_html=True)
    st.markdown("<h2>Arsitektur VGG-16</h2>", unsafe_allow_html=True)
    st.markdown("""
    VGG-16 adalah arsitektur CNN klasik yang sangat dalam (*deep architecture*) buatan Oxford University. Model ini terkenal karena akurasinya yang tajam dalam mengekstrak fitur spasial kompleks pada citra beresolusi tinggi.
    
    ### ⚙️ Spesifikasi Teknis dalam Penelitian
    - **Total Parameter Beban**: ~138 Juta (Sangat Besar/Berat)
    - **Fungsi Aktivasi Output**: Sigmoid (Binary Classification)
    - **Teknik Optimasi**: Tumpukan konvolusi ukuran $3 \\times 3$ yang padat dan konstan pada seluruh layernya.
    - **Metrik Evaluasi Akhir**:
      - Akurasi (*Accuracy*): Mendekati Sempurna (**100%** pada Dataset Validasi)
      - Profil Eksekusi: Memerlukan waktu pemrosesan (*inference time*) yang sedikit lebih lama jika dijalankan pada *CPU* standar.
      
    ### 🔄 Alur Pemrosesan (Preprocessing)
    1. **Resizing**: Mengubah dimensi citra menjadi standar input $224 \\times 224$ piksel.
    2. **Normalisasi**: Tidak seperti model lain, VGG16 tidak menggunakan skala kecil, melainkan memusatkan nilai pada rata-rata citra (*zero-centered by mean pixel*) dan mengonversi format susunan warna asli dari **RGB ke BGR** sesuai arsitektur awal peneliti aslinya menggunakan fungsi `tf.keras.applications.vgg16.preprocess_input`.
    
    ### 🏗️ Modifikasi Head (Transfer Learning)
    Sama halnya dengan MobileNetV2, penyesuaian (*fine-tuning*) pada bagian akhir VGG-16 dibuat identik 100%:
    - `GlobalAveragePooling2D()` 
    - `Dense(128, ReLU)`
    - `Dropout(0.3)`
    - `Dense(1, Sigmoid)`
    
    > **Catatan Peneliti:** Penggunaan layer modifikasi yang identik pada kedua model bertujuan untuk menciptakan ruang komparasi (perbandingan) *head-to-head* yang valid, adil, dan sah secara kaidah akademis.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
