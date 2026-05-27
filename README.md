<div align="center">
  <h1>🥜 AI Penyortir Kacang Tanah (Groundnut Quality Detecor)</h1>
  <p>Sistem Klasifikasi Otomatis Mutu Kacang Tanah berdasarkan SNI 01-3921-1995 menggunakan Arsitektur CNN (VGG16 & MobileNetV2).</p>
  
  [![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)](https://www.tensorflow.org/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-red.svg)](https://streamlit.io/)
  [![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF.svg)](https://www.kaggle.com/)
</div>

<br>

## 📖 Deskripsi Proyek
Proyek Deep Learning ini bertujuan untuk mengotomatisasi inspeksi visual kualitas fisik kacang tanah kupas (Bersih vs Terkontaminasi Kulit/Kotoran). Proyek ini merupakan komparasi riset antara arsitektur beban berat (**VGG16**) dengan arsitektur ringan yang efisien untuk *edge-device* (**MobileNetV2**). 

Pendekatan ini diharapkan dapat menggantikan proses penyortiran manual yang lambat dan rawan *human-error* di industri pertanian.

## ✨ Fitur Aplikasi (Web App)
- **Glassmorphism UI:** Antarmuka pengguna modern dengan estetika *dark mode*.
- **Model Selector:** Fitur dinamis untuk menukar mesin deteksi antara VGG16 (Akurasi Tinggi) dan MobileNetV2 (Kecepatan Tinggi).
- **Confidence Matrix:** Kalkulasi tingkat keyakinan AI (*confidence score*) secara *real-time*.

## 📊 Kinerja Model (Evaluasi Metrik)
Berdasarkan pelatihan dengan rasio pisah *dataset* 80:10:10 dan 20 *epoch* (*Early Stopping*):

| Metrik Evaluasi | VGG16 (~116 MB) | MobileNetV2 (~21 MB) |
|---|---|---|
| **Akurasi** | 100% | 96% |
| **Presisi** | 100% | 100% |
| **Recall** | 100% | 92% |
| **F1-Score**| 100% | 95.8% |

> **Analisis:** VGG16 unggul mutlak dalam akurasi pengenalan fitur, namun MobileNetV2 menawarkan efisiensi komputasi yang luar biasa dengan ukuran model yang lebih kecil hingga 5.5x lipat, menjadikannya kandidat terbaik untuk implementasi *IoT / Edge Computing*.

## 🚀 Cara Menjalankan di Komputer Lokal

1. Kloning repositori ini:
```bash
git clone https://github.com/USERNAME/kacang-tanah-quality-classifier.git
cd kacang-tanah-quality-classifier
```

2. Instalasi dependensi:
```bash
pip install -r requirements.txt
```

3. Jalankan server Streamlit:
```bash
streamlit run app.py
```

## ⚠️ Catatan Penting (Git LFS)
Model **VGG16** berukuran >100 MB. Jika Anda ingin melakukan *fork* atau memodifikasi repositori ini, pastikan Anda telah menginstal [Git Large File Storage (LFS)](https://git-lfs.github.com/) agar *file* bobot `.h5` tidak diblokir oleh GitHub.
