DATASET
Dataset yang dipakai adalah accident-detection-from-cctv-footage dari Kaggle. Isinya berupa kumpulan frame dari CCTV. Tujuan utamanya: melatih autoencoder supaya bisa memperbaiki gambar yang mobilnya sudah diblur otomatis.

Langkah-langkah awalnya:
- Download dataset-nya pakai kagglehub.
- Semua gambar langsung dipindahin ke folder frames agar mudah diakses.

DETEKSI OBJEK & AUTO-BLUR MOBIL
Kita pakai model pre-trained Faster R-CNN (tipe fasterrcnn_resnet50_fpn) buat mendeteksi mobil di gambar.
- Di COCO dataset, label buat "car" itu angka 3.
- Kalau confidence score-nya > 0.7 dan objeknya mobil, maka area mobil langsung diblur pakai cv2.GaussianBlur. Blur-nya otomatis tiap gambar.

TRANSFORMASI GAMBAR
Semua gambar:
- Diresize jadi 128x128 piksel
- Diubah ke format tensor supaya bisa diproses sama PyTorch

CUSTOM DATASET
Dataset-nya dibaca dari folder frames.
Setiap gambar akan diproses:
1. Ditransformasi (resize & to tensor)
2. Dicek dan diblur bagian mobilnya
3. Output-nya berupa pasangan: (gambar_blur, gambar_asli)

ARSITEKTUR AUTOENCODER
Input: Gambar RGB ukuran (3, 128, 128)

Encoder:
(3, 128, 128) → (64, 64, 64) → (128, 32, 32) → (256, 16, 16)
- Pakai Conv2D dengan kernel 4x4, stride 2, padding 1
- Aktivasi pakai ReLU

Decoder:
(256, 16, 16) → (128, 32, 32) → (64, 64, 64) → (3, 128, 128)
- Pakai ConvTranspose2D buat balikin ke ukuran awal
- Output-nya pakai Sigmoid, biar nilainya antara 0 dan 1

TRAINING
- Loss function: Mean Squared Error (MSE) antara output dan gambar asli
- Optimizer: Adam dengan learning rate 0.001
- Model dilatih selama 5 epoch
- Kita juga hitung PSNR (Peak Signal-to-Noise Ratio) buat ngukur seberapa bagus kualitas hasil rekonstruksi
- ![image](https://github.com/user-attachments/assets/d042c0ae-7ab6-4e6d-848c-7ca6aad54d25)


HASIL TRAINING
Dibikin grafik:
- Perubahan loss tiap epoch
- Nilai PSNR tiap epoch

VISUALISASI OUTPUT
Ditampilkan hasilnya:
1. Gambar input (udah diblur otomatis)
2. Gambar asli (target)
3. Gambar hasil rekonstruksi dari autoencoder
![image](https://github.com/user-attachments/assets/78c2ff33-ca76-461c-8d2a-6124c3f96077)


CATATAN
Tujuan training ini adalah agar autoencoder bisa "memperbaiki" gambar yang mobilnya diblur.
Jadi model akan diajarkan untuk menebak bentuk asli dari bagian gambar yang ilang sebagian.
Konsep ini bisa menjadi awal untuk ngebangun sistem pemulihan citra CCTV atau rekonstruksi gambar dengan privasi yang lebih aman.
