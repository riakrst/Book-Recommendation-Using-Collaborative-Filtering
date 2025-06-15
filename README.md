# Laporan Proyek Machine Learning - Ria Kristi

## Project Overview

Di era digital, pengguna internet semakin dibanjiri dengan informasi dan pilihan, termasuk dalam hal konsumsi buku. Banyaknya buku yang tersedia membuat pengguna kesulitan untuk menemukan bacaan yang sesuai dengan minat mereka. Oleh karena itu, dibutuhkan sistem rekomendasi yang dapat membantu pengguna dalam menemukan buku yang relevan dan menarik.

Proyek ini bertujuan untuk membangun sistem rekomendasi buku menggunakan pendekatan **machine learning**, dengan fokus pada metode **collaborative filtering** berbasis deep learning. Model ini dilatih menggunakan dataset Book-Crossing yang tersedia secara publik di Kaggle, dan akan memprediksi rating buku yang belum dibaca oleh pengguna untuk memberikan rekomendasi yang dipersonalisasi.

Sistem rekomendasi memiliki peran penting dalam meningkatkan pengalaman pengguna, retensi pelanggan, serta mendorong konsumsi konten yang lebih luas dan relevan. Berbagai platform besar seperti Amazon dan Goodreads telah memanfaatkan sistem ini untuk meningkatkan kepuasan pengguna.

---

## Business Understanding

### Problem Statements

1. Bagaimana cara memprediksi preferensi pengguna terhadap buku yang belum mereka baca berdasarkan interaksi historis?
2. Bagaimana sistem dapat memberikan rekomendasi buku yang relevan dan berkualitas tinggi dengan memanfaatkan data interaksi pengguna?

### Goals

1. Membangun model machine learning yang dapat memprediksi rating buku berdasarkan data pengguna dan buku menggunakan metode collaborative filtering.
2. Menghasilkan daftar rekomendasi 10 buku teratas yang dipersonalisasi untuk setiap pengguna.

### Solution Statement
Untuk mencapai tujuan proyek, digunakan pendekatan Collaborative Filtering (CF), yaitu metode rekomendasi yang memanfaatkan pola interaksi antara pengguna dan item (buku). Model mempelajari preferensi pengguna dan karakteristik buku melalui proses matrix factorization berbasis neural network. Dalam pendekatan ini, dibangun embedding untuk user dan buku, lalu diprediksi rating berdasarkan interaksi keduanya.

Model ini dipilih karena tidak memerlukan metadata tambahan seperti genre atau sinopsis buku—cukup dengan data interaksi user–item. Hal ini menjadikannya cocok untuk situasi dengan informasi konten yang terbatas namun memiliki cukup interaksi (rating). Selain itu, pendekatan ini memungkinkan model untuk menangkap pola laten antar pengguna dan buku yang serupa, sehingga tetap dapat merekomendasikan item meskipun belum pernah dilihat oleh pengguna tersebut.

---
## Data Understanding

Dataset yang digunakan dalam proyek ini berasal dari Kaggle, dengan judul [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset/data). Dataset ini sangat populer untuk eksperimen sistem rekomendasi berbasis rating karena memuat informasi lengkap mengenai pengguna, buku, dan interaksi di antara keduanya.

### Struktur Dataset
Dataset terdiri dari **tiga file utama** yang saling terhubung melalui kolom `User-ID` dan `ISBN`. Berikut adalah ringkasan masing-masing file:

**1. Users.csv**

| Kolom    | Tipe Data | Deskripsi                                           | Data Unik |
| -------- | --------- | --------------------------------------------------- | --------- |
| User-ID  | Integer   | ID unik pengguna                                    | 278.858   |
| Location | String    | Lokasi pengguna dalam format `City, State, Country` | 278.858   |
| Age      | Integer   | Usia pengguna (banyak missing value)                | 168.096    |
- Kolom `Age` hanya memiliki ~60% data yang valid.
  
**2. Books.csv**
| Kolom               | Tipe Data | Deskripsi                                           | Data Unik |
| ------------------- | --------- | --------------------------------------------------- | --------- |
| ISBN                | String    | Kode unik buku (International Standard Book Number) | 271.360   |
| Book-Title          | String    | Judul buku                                          | 271.360    |
| Book-Author         | String    | Nama penulis (pertama saja jika lebih dari satu)    | 271.358    |
| Year-Of-Publication | Integer   | Tahun terbit buku                                   | 271.360    |
| Publisher           | String    | Nama penerbit                                       | 271.358   |
| Image-URL-S         | String    | URL gambar sampul ukuran kecil                      | 271.360   |
| Image-URL-M         | String    | URL gambar sampul ukuran sedang                     | 271.360   |
| Image-URL-L         | String    | URL gambar sampul ukuran besar                      | 271.357    |
- Terdapat sedikit missing values pada kolom `Book-Author`, `Publisher`, dan `Image-URL-L`.

**3. Book-Ratings.csv**
| Kolom       | Tipe Data | Deskripsi                                                 | Data Unik |
| ----------- | --------- | --------------------------------------------------------- | --------- |
| User-ID     | Integer   | ID pengguna                                               | 1.149.780 |
| ISBN        | String    | Kode unik buku                                            | 1.149.780 |
| Book-Rating | Integer   | Nilai rating (skala 0–10), 0 = implisit, 1–10 = eksplisit | 1.149.780 |
- 0 menunjukkan interaksi implisit, seperti melihat atau memiliki buku tanpa memberi penilaian eksplisit. Jadi, rating 0 bukan berarti buku tersebut tidak disukai.

### Exploratory Data Analysis (EDA)

Beberapa tahapan eksplorasi data dilakukan untuk memahami karakteristik dataset:

- **Distribusi Rating**:
  
  ![image](https://github.com/user-attachments/assets/b5a65e74-d418-44b3-af75-db76d6affe4c)

  Mayoritas nilai rating berada di angka tinggi, seperti 8–10. Sementara rating `0` mendominasi secara kuantitas, namun tidak digunakan untuk pelatihan karena bukan bentuk evaluasi eksplisit.

- **10 Buku paling banyak dirating**
  ![image](https://github.com/user-attachments/assets/5b526c2c-59f0-43b0-8ea8-3d0ba6ee195c)

### Data Preprocessing

**Missing Value**

Proses awal yang dilakukan adalah mengecek nilai kosong (missing value) pada masing-masing dataset.

Hasil pemeriksaan:

- Dataset `books` memiliki sejumlah kecil nilai kosong:
  - `Book-Author`: 2 missing
  - `Publisher`: 2 missing
  - `Image-URL-L`: 3 missing

- Dataset `users` memiliki missing value yang cukup besar pada kolom:
  - `Age`: 110.762 missing dari total 278.858 baris

Penanganan yang dilakukan:

1. Nilai kosong pada `Book-Author`, `Publisher`, dan `Image-URL-L` diisi dengan teks placeholder `"Unknown"` karena jumlahnya sangat kecil dan tidak signifikan memengaruhi data.
2. Kolom `Age` diabaikan dalam proses pemodelan karena pendekatan yang digunakan adalah **collaborative filtering**, yang tidak memerlukan informasi demografis pengguna secara langsung.

**Filter Rating = 0**

Menurut dokumentasi dataset di Kaggle, rating dengan nilai 0 merepresentasikan interaksi implisit, yaitu ketika pengguna melihat atau memiliki buku tetapi tidak memberikan penilaian eksplisit terhadap buku tersebut.
Artinya, rating 0 tidak berarti buku tersebut tidak disukai atau diberi penilaian buruk, melainkan tidak ada rating eksplisit dari pengguna.

Jika nilai 0 tidak difilter, dapat menyebabkan:
- Model bisa belajar dari data yang tidak akurat, karena rating 0 bukanlah penilaian nyata (eksplisit) terhadap kualitas buku.
- Prediksi menjadi bias atau tidak stabil, karena model berusaha mempelajari "rating" yang sebenarnya tidak diberikan oleh user.
- Bisa menurunkan performa model dalam mempelajari preferensi pengguna dan menghasilkan rekomendasi yang relevan.

Langkah yang diambil: 
- Menghapus semua entri dengan nilai Book-Rating = 0.
- Jumlah data setelah filter rating 0: 433.671 entri rating eksplisit

**Filter Data untuk Mengurangi Sparsity**

Setelah menghapus rating 0, dilakukan filtering lebih lanjut untuk mengurangi sparsity dalam matriks interaksi. Banyak pengguna dan buku hanya memiliki sedikit rating, sehingga model kesulitan belajar dari data yang terlalu jarang.
Kriteria filtering:
- Hanya menyertakan pengguna yang memberikan minimal 10 rating
- Hanya menyertakan buku yang menerima minimal 10 rating

Jumlah data setelah filtering sparsity: 74.907 entri rating

Meskipun cukup banyak data yang dihapus dari total awal, filtering ini penting untuk:
- Mengurangi jumlah data yang jarang terisi dengan menghapus pengguna dan buku yang hanya punya sedikit rating.
- Memastikan bahwa model dilatih pada data dari pengguna dan buku yang memiliki cukup riwayat interaksi eksplisit.
- Data yang tersisa sebanyak 74.907 entri, meskipun jauh lebih sedikit dari awal, tetap mencukupi untuk membangun model *collaborative filtering* yang efektif dan representatif pada populasi pengguna dan buku yang lebih aktif dan populer.

---

## Data Preparation

Pada tahap ini, dilakukan beberapa langkah penting untuk menyiapkan data agar dapat digunakan dalam model *collaborative filtering*. Model tersebut membutuhkan representasi data dalam bentuk numerik (integer) dan normalisasi target, agar proses pelatihan model berjalan efisien dan akurat. Berikut tahapan yang dilakukan secara berurutan:

### Encoding User dan ISBN

Model *collaborative filtering* seperti Neural Collaborative Filtering hanya dapat memproses data input dalam bentuk indeks integer. Oleh karena itu, kolom `User-ID` dan `ISBN` perlu disandikan (di-*encode*) menjadi indeks numerik.

**Langkah-langkah:**
- Mengambil seluruh nilai unik dari `User-ID` dan `ISBN`.
- Membuat mapping dari ID ke integer (encoding), dan sebaliknya (decoding) untuk keperluan interpretasi hasil.
- Menambahkan kolom baru `user` dan `book` pada DataFrame `ratings` yang berisi hasil encoding.

| Kolom Asli | Kolom Encode | Keterangan |
|------------|--------------|------------|
| `User-ID`  | `user`       | Disandikan ke indeks integer mulai dari 0. |
| `ISBN`     | `book`       | Disandikan ke indeks integer mulai dari 0. |

> **Alasan:** Encoding diperlukan agar model bisa membaca ID pengguna dan buku dalam format integer (bukan string), karena embedding layer hanya bekerja dengan input berupa indeks numerik.

### Normalisasi Book-rating

Model *collaborative filtering* yang digunakan pada proyek ini bertujuan untuk memprediksi nilai rating yang diberikan pengguna terhadap buku tertentu. Oleh karena itu, kolom `Book-Rating` digunakan sebagai **target** (label) dalam pelatihan model.

Nilai asli pada `Book-Rating` berada dalam skala **1–10**, namun model neural network lebih optimal jika target berada pada skala **0–1**. Maka dari itu, diperlukan proses **normalisasi**.

**Tujuan Normalisasi**
- Menyesuaikan nilai target agar berada dalam skala yang seragam (`0–1`), sehingga model dapat belajar dengan lebih stabil.
- Mempercepat konvergensi saat pelatihan neural network.

```
y = ratings['Book-Rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

```
- Hasil normalisasi (y) disimpan dalam bentuk array dan digunakan sebagai target pada proses pelatihan model.

###  Split Data Training dan Validasi
Data dibagi menjadi 80% untuk training dan 20% untuk validasi agar performa model dapat diuji secara obyektif. 

| Dataset                         | Jumlah Baris | Persentase |
| ------------------------------- | ------------ | ---------- |
| Training (`x_train`, `y_train`) | 59.925       | 80%        |
| Validasi (`x_val`, `y_val`)     | 14.982       | 20%        |

**Proses Pembagian**

Sebelum melakukan pembagian, data terlebih dahulu **diacak (shuffled)** menggunakan `random_state=42`. Tujuannya agar distribusi data menjadi acak dan tidak terurut berdasarkan urutan pengguna atau buku tertentu.

Langkah-langkah yang dilakukan:

1. Acak dataframe `ratings` agar distribusinya merata.
2. Gabungkan kolom hasil encoding `user` dan `book` sebagai **fitur input**:
   ```python
   x = ratings[['user', 'book']].values
   ```
3. Gunakan hasil normalisasi dari Book-Rating sebagai target output (y).
4. Bagi data menjadi 80% training dan 20% validasi

**Tujuan dari pembagian ini**

- Mengukur generalisasi model: memastikan model tidak hanya belajar dari data pelatihan, tetapi juga mampu memprediksi data baru yang tidak pernah dilihat sebelumnya.
- Mendeteksi overfitting: jika performa pada data validasi jauh lebih buruk dibanding data training, artinya model overfit terhadap data pelatihan.
- Dengan pembagian ini, kita bisa melakukan pelatihan dan evaluasi model dengan lebih adil dan representatif terhadap data sebenarnya.
---

## Modeling and Result

Pada tahap ini, model sistem rekomendasi dibangun menggunakan pendekatan Collaborative Filtering berbasis TensorFlow. Tujuannya adalah mempelajari representasi (embedding) dari user dan buku berdasarkan interaksi sebelumnya, kemudian memprediksi skor kecocokan yang dinormalisasi dalam rentang [0, 1].

Collaborative Filtering (CF) adalah teknik sistem rekomendasi yang memanfaatkan pola interaksi historis antar pengguna dan item (dalam hal ini, buku). CF tidak memerlukan informasi konten seperti deskripsi atau genre buku, melainkan hanya bergantung pada data rating. Sistem ini merekomendasikan item berdasarkan kesamaan perilaku pengguna lain, sehingga cocok digunakan saat informasi konten terbatas tetapi tersedia banyak interaksi user–item.

### Model: Neural Collaborative Filtering

Model dikembangkan dengan pendekatan dot product embedding antara user dan book, yang kemudian dijumlahkan dengan bias dan diproses melalui fungsi aktivasi sigmoid. Arsitektur utama:

1. **Embedding Layer** untuk user dan book.
2. **Dropout** digunakan untuk mencegah overfitting.
3. **Dot Product** antar embedding untuk memodelkan interaksi.
4. **Bias Layer** untuk menangkap kecenderungan masing-masing entitas.
5. **Sigmoid Activation** membatasi output skor ke rentang [0, 1].

Model dikompilasi menggunakan:
```python
embedding_size = 50
loss = MeanSquaredError()
metric = RootMeanSquaredError()
optimizer = Adam(learning_rate=0.001)
```
Model dilatih maksimal selama 50 epoch dengan early stopping yang menghentikan pelatihan pada epoch ke-28, menggunakan batch size 64, dan validasi dilakukan dengan data sebesar 20%.

### Top-N Recommendation Output
Setelah pelatihan model, sistem menghasilkan 10 rekomendasi buku teratas yang dipersonalisasi untuk pengguna tertentu. 
Output mencakup:
- 5 buku yang sebelumnya diberi rating tinggi oleh user (referensi preferensi).
- 10 buku rekomendasi berdasarkan prediksi model.

![image](https://github.com/user-attachments/assets/685ad222-6c7c-4819-a2c2-e7d701275653)

Skor prediksi dikembalikan ke skala 1–10 menggunakan transformasi linier dari hasil sigmoid.

---

## Evaluation
RMSE (Root Mean Squared Error) digunakan sebagai metrik evaluasi utama karena model melakukan prediksi pada nilai rating dalam skala numerik kontinu (setelah dinormalisasi ke [0, 1]). RMSE cocok untuk mengukur seberapa dekat prediksi model terhadap nilai aktual pada masalah regresi seperti sistem rekomendasi berbasis rating.

Formula RMSE:
![image](https://github.com/user-attachments/assets/94d1ebdf-49fe-4647-bd42-abd0cc8998dc)

- N adalah jumlah total data (jumlah pasangan nilai aktual dan prediksi)
- 𝑦𝑖 : rating asli (target)
- ^𝑦𝑖 : rating hasil prediksi

Semakin kecil nilai RMSE, semakin akurat model

**Bagaimana RMSE Bekerja:**
- Mula-mula, RMSE menghitung selisih antara nilai prediksi dan nilai aktual
- Selisih tersebut dikuadratkan agar semua error menjadi positif dan kesalahan besar memiliki dampak lebih besar
- Nilai kuadrat error dijumlahkan dan dirata-ratakan
- Terakhir, diambil akar kuadrat dari nilai rata-rata kuadrat tersebut agar hasilnya kembali ke skala asli dan dapat diinterpretasikan dengan lebih mudah

RMSE bersifat sensitif terhadap outlier, sehingga cocok untuk konteks seperti ini di mana kita ingin menghindari prediksi dengan kesalahan besar.

**Hasil Training & Validasi**

![image](https://github.com/user-attachments/assets/bbf1000b-2d4f-4b76-98a1-65bd1892c784)

Grafik di atas menunjukkan perkembangan Root Mean Squared Error (RMSE) selama proses training dan validasi.
- RMSE Training menunjukkan penurunan konsisten dari awal hingga akhir epoch, mendekati 0.15.
- RMSE Validasi juga menurun di awal hingga mencapai titik stabil sekitar epoch ke-10, dan bertahan di kisaran ±0.18 hingga akhir training.
- Tidak terdapat indikasi overfitting yang signifikan karena gap antara training dan validasi kecil dan stabil.

---

## Kesimpulan

Model sistem rekomendasi yang dikembangkan berhasil menjawab dua problem utama: memprediksi preferensi pengguna dan memberikan rekomendasi buku yang relevan. Dengan pendekatan Collaborative Filtering berbasis neural network, model mampu mempelajari pola interaksi pengguna dan menghasilkan rekomendasi 10 buku teratas yang dipersonalisasi.

Hasil evaluasi menunjukkan nilai RMSE yang rendah dan stabil, menandakan prediksi yang cukup akurat. Tanpa memerlukan data konten tambahan, model ini efektif dalam memberikan rekomendasi yang relevan berdasarkan interaksi historis pengguna.

---

## Referensi
- C3 AI. What is Root Mean Square Error (RMSE)? Retrieved from: [https://c3.ai/glossary/data-science/root-mean-square-error-rmse](https://c3.ai/glossary/data-science/root-mean-square-error-rmse)
- Dicoding Indonesia. Collaborative Filtering. Retrieved from: [https://www.dicoding.com/academies/319/tutorials/17119](https://www.dicoding.com/academies/319/tutorials/17119)
