# Sinyal dan Sistem — Deteksi Burung Hama Padi Berbasis Sinyal Akustik

**Topik 1:** Analisis Pengaruh Ukuran dan Jenis Window STFT terhadap Representasi Spektral dan Klasifikasi Suara Burung Hama Padi.

Proyek ini mengevaluasi pengaruh berbagai parameter windowing pada Short-Time Fourier Transform (STFT) untuk ekstraksi fitur akustik dalam klasifikasi suara burung hama padi dan burung lainnya.

---

## 📋 Fitur & Cakupan Eksperimen

- **16 Konfigurasi Window STFT:**
  - **Ukuran Window:** 256, 512, 1024, 2048 sampel
  - **Fungsi Window:** Rectangular (`boxcar`), Hann, Hamming, Blackman
- **Praproses & Filtering:** Pre-emphasis filter, bandpass filtering (1 kHz – 10 kHz), normalisasi audio.
- **Ekstraksi Fitur Spektral:** Spektrogram magnitude, statistical pooling (mean, std, skew, kurtosis), spectral centroid, bandwidth, rolloff, dan flatness.
- **Klasifikasi & Validasi:** SVM (RBF kernel) dengan 5-fold cross validation × 3 random seed.
- **Uji Statistik & Evaluasi:**
  - Friedman test & Nemenyi post-hoc test
  - Kendall's W (tingkat konkordansi)
  - Critical Difference (CD) diagram
  - Analisis efek berpasangan & uji sensitivitas

---

## 📁 Struktur Direktori

```text
├── Topik1_STFT_Lokal.ipynb    # Jupyter Notebook utama (lengkap dengan penjelasan sel & visualisasi)
├── topik1_stft.py             # Script Python CLI untuk menjalankan eksperimen penuh/cepat
├── requirements.txt           # Dependensi pustaka Python
├── hasil/                     # Folder luaran hasil eksperimen (tabel CSV, visualisasi PNG, JSON)
└── README.md                  # Dokumentasi proyek
```

> **Catatan:** Dataset audio (`training/` dan `testing/`) serta file cache fitur (`.npz`) tidak disertakan di repositori karena batas ukuran file.

---

## 🚀 Panduan Menjalankan

### 1. Prasyarat & Instalasi Dependensi
Pastikan Python 3.8+ sudah terpasang. Pasang pustaka yang diperlukan:

```bash
pip install -r requirements.txt
```

### 2. Menjalankan via Jupyter Notebook
Buka dan jalankan `Topik1_STFT_Lokal.ipynb` menggunakan Jupyter Lab atau VS Code:

```bash
jupyter notebook Topik1_STFT_Lokal.ipynb
```

### 3. Menjalankan via CLI Script
Jalankan script `topik1_stft.py` langsung dari terminal:

```bash
# Mode uji cepat (subset 300 klip/kelas training, 100 testing)
python topik1_stft.py

# Mode data penuh (seluruh data untuk hasil laporan akhir)
python topik1_stft.py --full

# Mode 6 kelas (menggabungkan Bulbul & Chubbs_cis menjadi Other_Birds)
python topik1_stft.py --full --merge6
```

---

## 📊 Hasil Eksperimen

Semua hasil tabel CSV dan grafik visualisasi (spektrogram, matriks konfusi, heatmap F1-score, diagram Nemenyi CD, dll.) secara otomatis disimpan di dalam direktori `hasil/`.
