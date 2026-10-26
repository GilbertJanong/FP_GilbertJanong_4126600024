# Final Project Oprec PENSHIP - Gilbert Janong (4126600024)

## Identitas
- Nama: Gilbert Janong
- NRP: 4126600024
- Program studi: D4 Teknik Mekatronika

## Project Roboflow
- Project ID Roboflow: gilbert-janong-s-workspace/penship-final-project-zjfsn, version 3
- Link Roboflow Universe: https://universe.roboflow.com/gilbert-janong-s-workspace/penship-final-project-zjfsn

## Dataset
- Jumlah foto asli: bottle 47, laptop 39, pencil 47
- Metode pengambilan foto: jarak dekat dan jauh, perspektif, background yang bukan merupakan class, dan gabungan dari semua class di satu foto
- Fitur AI labeling Roboflow yang digunakan (apabila ada) dan cara perapiannya: labeling otomatis menggunakan fitur AI Auto Labeling yang kemudian annote dirapikan melalui preview manual

## Training
- Lingkungan training: laptop Lenovo LOQ NVIDIA RTX-3050 6 GB VRAM
- Model: yolov8n dan yolov8s pada eksperimen, hasil akhir menggunakan yolov8n
- Grafik: lihat folder grafik/

## Pengembangan Kode
Daftar perubahan terhadap kode dasar, beserta alasan dan pengaruhnya terhadap hasil:
1. Pengembangan augmentasi data:
- Menerapkan variasi citra berupa perubahan kecerahan, saturasi warna, skala, rotasi, translasi, dan mosaic.
- Bertujuan meningkatkan kemampuan model dalam menghadapi variasi visual.
- Mengurangi risiko model terlalu bergantung pada karakteristik gambar training tertentu.
2. Eksperimen hyperparameter dan arsitektur
- Membandingkan sembilan konfigurasi training.
- Menguji perubahan learning rate, optimizer, batch size, jumlah epoch, augmentasi, dan arsitektur YOLOv8n serta YOLOv8s.
- Bertujuan menemukan konfigurasi yang memberikan performa deteksi terbaik.
3. Evaluasi dan pemilihan model
- Menggunakan precision, recall, mAP50, dan mAP50-95 untuk membandingkan hasil eksperimen.
- Menggunakan validation set untuk memilih kandidat model.
- Menggunakan test set untuk mengevaluasi kemampuan model pada data yang tidak digunakan dalam pemilihan konfigurasi.
4. Pengembangan inferensi video
- Menggabungkan YOLOv8 dengan OpenCV untuk membaca video per frame.
- Menampilkan label kelas, bounding box, dan confidence.
- Menyimpan video hasil deteksi serta mengukur FPS inferensi.

## Hasil Eksperimen
No	Skenario	Epochs	Batch	Optimizer	mAP50	mAP50-95	Keterangan
0	Baseline (kode dasar)	50	16	auto	0.643	0.316	Titik acuan eksperimen
1	Learning Rate Rendah (lr0 rendah)	50	16	AdamW	0.640	0.337	mAP50-95 sedikit meningkat, tetapi mAP50 hampir sama dengan baseline
2	Learning Rate Tinggi (lr0 tinggi)	50	16	AdamW	0.501	0.205	Performa menurun; kemungkinan learning rate terlalu tinggi
3	Optimizer SGD	50	16	SGD	0.652	0.277	mAP50 meningkat sedikit, tetapi mAP50-95 menurun
4	Batch Size 8	50	8	auto	0.699	0.344	Peningkatan mAP dibanding baseline
5	Epochs 100	100	16	auto	0.572	0.296	Training berhenti pada epoch ke-46
6	Augmentasi Ringan (07_augment_light)	50	16	AdamW	0.911	0.486	Hasil validasi terbaik; dipilih sebagai kandidat model
7	Model YOLOv8s	50	16	auto	0.486	0.379	Training baru mencapai 16 epoch sehingga belum sebanding langsung dengan eksperimen 50 epoch
8	Tanpa Mosaic (no_mosaic)	50	16	auto	0.657	0.322	Performa lebih baik dari baseline pada mAP50 dan mAP50-95

## Analisis dan Kendala
- Penyebab skenario terbaik menghasilkan nilai lebih tinggi: Skenario 07_augment_light memperoleh hasil terbaik karena penggunaan augmentasi ringan membantu model mengenali variasi objek dan kondisi visual. Model menghasilkan mAP50 sebesar 0,911 dan mAP50-95 sebesar 0,486 pada validasi.

- Kendala yang ditemui dan cara penyelesaiannya: Kendala meliputi keterbatasan jumlah dataset, perbedaan hasil antar-konfigurasi, file best.pt yang tidak ditemukan, serta error OpenCV saat menampilkan video. Penanganannya dilakukan melalui augmentasi data, perbandingan hasil eksperimen, penyesuaian path model, dan perbaikan instalasi OpenCV.