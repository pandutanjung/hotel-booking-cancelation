# Laporan Proyek Machine Learning - Pandu Persada Tanjung

## Domain Proyek

Industri perhotelan menghadapi tantangan besar terkait pembatalan pemesanan reservasi oleh pelanggan. Dalam beberapa kasus, pelanggan cenderung membatalkan pemesanan mereka tanpa pemberitahuan yang cukup jauh hari, yang menyulitkan pihak hotel untuk mengganti pemesanan yang hilang. Oleh karena itu, penting untuk mengidentifikasi pola-pola dalam pembatalan guna mengantisipasi dan mengelola pembatalan dengan lebih baik ([Hermawan et al., 2025](https://journal.aptii.or.id/index.php/Router/article/download/400/567/2204)).

Menurut [Antonio et al. (2019)](https://journal.aptii.or.id/index.php/Router/article/download/400/567/2204), penerapan algoritma pembelajaran mesin dalam memprediksi pembatalan reservasi telah terbukti mampu meningkatkan ketepatan prediksi dan membantu meminimalkan kerugian dari pembatalan mendadak. Berbagai pendekatan seperti regresi logistik, pohon keputusan, hingga model ensemble seperti Random Forest dan XGBoost dimanfaatkan untuk mengenali faktor-faktor yang memengaruhi pembatalan serta untuk mendeteksi pemesanan yang berpotensi tinggi dibatalkan.

## Business Understanding

### Problem Statements
Bagaimana cara memprediksi apakah suatu reservasi hotel akan dibatalkan atau tidak berdasarkan informasi reservasi dan profil pelanggan?

### Goals
Membangun model klasifikasi yang akurat untuk memprediksi pembatalan reservasi.

### Solution statements
- Mengembangkan model baseline menggunakan Logistic Regression.
- Mengembangkan model alternatif menggunakan Random Forest Classifier.
- Metrik evaluasi: Accuracy, Precision, Recall, F1-score.

## Data Understanding
Sumber Data: [Kaggle Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
Jumlah Data: 119.390 baris dan 32 kolom
Target: is_canceled (0 = tidak dibatalkan, 1 = dibatalkan)

### Variabel-variabel pada dataset adalah sebagai berikut:
| Variabel | Deskripsi |
| --- | --- |
| ADR | Rata-rata Tarif Harian |
| Adults | Jumlah orang dewasa	 |
| Agent | ID agen perjalanan yang melakukan pemesanan |
| ArrivalDateDayOfMonth | Tanggal bulan tanggal kedatangan |
| ArrivalDateMonth | Tanggal bulan kedatangan dengan 12 kategori: “Januari” hingga “Desember” |
| ArrivalDateWeekNumber | Nomor minggu tanggal kedatangan |
| ArrivalDateYear | Tahun tanggal kedatangan |
| AssignedRoomType | Kode untuk tipe kamar yang ditetapkan untuk pemesanan |
| Babies | Jumlah bayi |
| BookingChanges | Jumlah perubahan/amandemen yang dilakukan pada pemesanan sejak pemesanan dimasukkan ke PMS hingga saat check-in atau pembatalan |
| Children | Jumlah anak |
| Company | ID perusahaan/entitas yang melakukan pemesanan atau yang bertanggung jawab untuk membayar pemesanan |
| Country | Negara asal |
| CustomerType | Jenis pemesanan |
| DaysInWaitingList | Jumlah hari pemesanan berada dalam daftar tunggu sebelum dikonfirmasi ke pelanggan |
| DepositType | Indikasi apakah pelanggan telah melakukan deposit untuk menjamin pemesanan |
| DistributionChannel | Saluran distribusi pemesanan |
| IsCanceled | Nilai yang menunjukkan apakah pemesanan dibatalkan (1) atau tidak (0) |
| IsRepeatedGuest | Nilai yang menunjukkan apakah nama pemesanan berasal dari tamu yang berulang (1) atau tidak (0) |
| LeadTime | Jumlah hari yang berlalu antara tanggal pemesanan yang dimasukkan ke dalam PMS dan tanggal kedatangan |
| MarketSegment | Penunjukan segmen pasar |
| Meal | Jenis makanan yang dipesan |
| PreviousBookingsNotCanceled | Jumlah pemesanan sebelumnya yang tidak dibatalkan oleh pelanggan sebelum pemesanan saat ini |
| PreviousCancellations | Jumlah pemesanan sebelumnya yang dibatalkan oleh pelanggan sebelum pemesanan saat ini |
| RequiredCardParkingSpaces | Jumlah tempat parkir mobil yang dibutuhkan oleh pelanggan |
| ReservationStatus | Status terakhir reservasi |
| ReservationStatusDate | Tanggal saat status terakhir ditetapkan |
| ReservedRoomType |Kode tipe kamar yang dipesan |
| StaysInWeekendNights | Jumlah malam akhir pekan (Sabtu atau Minggu) tamu menginap atau memesan untuk menginap di hotel |
| StaysInWeekNights | Jumlah malam dalam seminggu (Senin sampai Jumat) tamu menginap atau memesan untuk menginap di hotel |
| TotalOfSpecialRequests | Jumlah permintaan khusus yang dibuat oleh pelanggan (misalnya tempat tidur kembar atau lantai tinggi) |

![Cancel vs Not Canceled](https://raw.githubusercontent.com/pandutanjung/hotel-booking-cancelation/b0479ebbd6dc905313153ae4d1650c195f1b2751/images/hotel-cancelvsnot.png)

Grafik di atas menunjukkan bahwa data tidak seimbang, dengan mayoritas pemesanan tidak dibatalkan (is_canceled = 0). Ini penting untuk dipertimbangkan dalam proses modeling agar model tidak bias terhadap kelas mayoritas.

![Heatmap Korelasi](https://raw.githubusercontent.com/pandutanjung/hotel-booking-cancelation/b0479ebbd6dc905313153ae4d1650c195f1b2751/images/hotel-heatmap.png)

Heatmap di atas menunjukkan korelasi antar fitur numerik. Beberapa insight penting:
is_canceled memiliki korelasi positif cukup tinggi dengan:
*   lead_time (semakin lama waktu tunggu, makin tinggi kemungkinan pembatalan)
*   previous_cancellations

Korelasi negatif terlihat dengan:
*   total_of_special_requests dan booking_changes, menandakan bahwa tamu yang benar-benar berniat datang cenderung lebih aktif dalam melakukan permintaan atau perubahan pemesanan

#### Menangani Missing Values
![Missing Values](https://raw.githubusercontent.com/pandutanjung/hotel-booking-cancelation/a21611041807d6a3dc0adf3cb1a9214a703c382b/images/missing-values.png)
1. children diisi dengan median
2. country dengan modus
3. agent dengan 0 (tanpa agen)
4. company dihapus karena mayoritas kosong

#### Menangani Duplikasi Data
![Cek Duplicate](https://raw.githubusercontent.com/pandutanjung/hotel-booking-cancelation/a21611041807d6a3dc0adf3cb1a9214a703c382b/images/cek-duplicate.png)

Terdapat sebanyak 31.994 baris data yang teridentifikasi duplikat. Lakukan imputasi terhadap entry data yang duplikat. Hasilnya sebagai berikut:

![Drop Duplicate](https://raw.githubusercontent.com/pandutanjung/hotel-booking-cancelation/a21611041807d6a3dc0adf3cb1a9214a703c382b/images/drop-duplicate.png)

Sudah tidak ada lagi data yang terduplikasi.

#### Menangani Outlier
![Outlier](https://raw.githubusercontent.com/pandutanjung/hotel-booking-cancelation/a21611041807d6a3dc0adf3cb1a9214a703c382b/images/outlier.png)
Dapat dilihat pada boxplot bahwa fitur 'lead_time', 'adr', dan 'days_in_waiting_list' memiliki outlier yang cukup ekstrem, maka perlu dilakukan penghapusan terhadap outlier tersebut. Berikut adalah hasilnya:
![Hasil Outlier](https://raw.githubusercontent.com/pandutanjung/hotel-booking-cancelation/cd276836593ec23c7faf8a125cf696a07dd0f3d2/images/hasil-outlier.png)

## Data Preparation
1. Salin DataFrame
Bertujuan untuk menghindari modifikasi langsung pada DataFrame asli
2. Encoding Fitur Kategorikal
Semua kolom bertipe object (kategorikal) diubah menjadi numerik menggunakan Label Encoding.
3. Split Fitur dan Target
4. Train-Test Split
Memisahkan data latih (80%) dan uji (20%)
5. Undersampling Kelas Mayoritas
Menyeimbangkan kelas target is_canceled agar tidak bias. Not Canceled (0) disampling acak agar jumlahnya sama dengan Canceled (1)

![Distribusi Setelah Undersampling](https://raw.githubusercontent.com/pandutanjung/hotel-booking-cancelation/b0479ebbd6dc905313153ae4d1650c195f1b2751/images/hotel-cancelvsnot-under.png)

7. Standarisasi (Scaling)
Menggunakan StandardScaler untuk menyesuaikan skala semua fitur

## Modeling
### Logistic Regression 
Logistic Regression merupakan algoritma yang sederhana, cepat dilatih, dan sangat cocok digunakan sebagai model baseline dalam masalah klasifikasi. Salah satu keunggulan utamanya adalah kemampuannya dalam memberikan interpretasi yang jelas terhadap pengaruh masing-masing fitur melalui nilai koefisien. Hal ini menjadikan Logistic Regression sangat berguna ketika interpretabilitas menjadi prioritas utama. Namun, model ini memiliki keterbatasan karena hanya mampu menangkap hubungan linear antar fitur, sehingga kurang efektif dalam menangani pola kompleks atau non-linear. Berikut adalah cara kerjanya:
1. Model menghitung kombinasi linier dari fitur
   
![Kombinasi Linier](https://raw.githubusercontent.com/pandutanjung/hotel-booking-cancelation/3d567ff6693fb8c3097f3d75aeb258a80b0805bb/images/kombinasi-linier.png)

Di sini, 𝛽 adalah bobot/koefisien, dan 𝑥 adalah nilai fitur.

2. Hasil linier 𝑧 kemudian diubah menjadi nilai antara 0 dan 1 menggunakan fungsi sigmoid

![Fungsi Sigmoid](https://raw.githubusercontent.com/pandutanjung/hotel-booking-cancelation/3d567ff6693fb8c3097f3d75aeb258a80b0805bb/images/sigmoid-function.png)

Hasil ini mewakili probabilitas prediksi kelas 1 misalnya pembatalan terjadi.

3. Klasifikasi Berdasarkan Threshold
- Jika probabilitas ≥ 0.5 maka kelas 1 (Canceled)
- Jika probabilitas < 0.5 maka kelas 0 (Not Canceled)
  
4. Model mencari bobot 𝛽 terbaik dengan meminimalkan log loss 

Pada proyek ini, Mmeskipun performanya cukup baik namun Logistic Regression menghasilkan 239 kesalahan klasifikasi untuk kasus pembatalan (false negatives) dan 7 prediksi positif yang salah (false positives). Hal ini menunjukkan bahwa Logistic Regression masih kurang optimal dalam mendeteksi kasus pembatalan, yang justru menjadi fokus utama dalam konteks bisnis hotel.

### Random Forest
Random Forest adalah algoritma berbasis ensemble yang sangat kuat dan mampu menangani berbagai jenis data, termasuk yang memiliki hubungan non-linear dan fitur saling berinteraksi. Cara kerja Random Forest:
1. Ambil sampel bootstrap dari data latih
2. Pilih secara acak m fitur dari total p fitur
3. Buat pohon keputusan dari data sampel tadi.
4. Cari split terbaik di setiap node hanya dari m fitur tadi.
5. Terus bagi node sampai mencapai batas minimum.
6. Ulangi sebanyak B kali
7. Gabungkan semua prediksi pohon dengan menggunakan mayoritas voting

Pada proyek ini, random forest dibangun dengan mengatur beberapa parameter:
1. n_estimators=100 yakni model akan membangun 100 pohon keputusan. Nilai 100 cukup umum dan seimbang antara akurasi dan efisiensi waktu.
2. random_state=42 mengatur seed untuk keacakan agar hasil eksperimen konsisten dan bisa reproduceable.

Berikut adalah sample hasil dari Random Forest dalam bentuk Pohon/Tree:
![Tree Random Forest](https://raw.githubusercontent.com/pandutanjung/hotel-booking-cancelation/87848dd0f885e0167d21f3ffec3ef42e999e4be2/images/tree-random-forest.png)

Gambar tersebut menunjukkan salah satu pohon keputusan dalam model Random Forest dengan kedalaman tiga, yang menggambarkan bagaimana model mengambil keputusan berdasarkan fitur-fitur penting seperti lead_time, market_segment, dan total_of_special_requests. Setiap node membagi data berdasarkan kondisi tertentu, dan warna node mencerminkan prediksi dominan (biru untuk pembatalan dan oranye untuk tidak dibatalkan)

Pada proyek ini, Random Forest menunjukkan performa yang sangat tinggi dengan tidak melakukan kesalahan prediksi sama sekali pada data uji (zero false positives dan zero false negatives). Hal ini mencerminkan kemampuannya dalam memetakan pola pembatalan dengan sangat baik. Kekurangannya terletak pada sifatnya yang kompleks, sehingga sulit untuk diinterpretasikan. Selain itu, waktu pelatihan dan komputasi relatif lebih besar dibanding Logistic Regression. Meskipun begitu, keunggulan dari segi akurasi dan kemampuan generalisasi menjadikan Random Forest sebagai pilihan yang lebih unggul.

### Model paling optimal
Berdasarkan evaluasi keseluruhan, Random Forest dipilih sebagai model terbaik karena memberikan hasil prediksi paling akurat dan stabil. Random Forest terbukti mampu mengidentifikasi baik kasus pembatalan maupun non-pembatalan dengan sangat baik. Dalam konteks bisnis hotel, akurasi tinggi dalam memprediksi pembatalan sangat krusial untuk mengantisipasi potensi kerugian dan menyusun strategi operasional yang lebih efektif.

## Evaluation

Dalam proyek prediksi pembatalan hotel ini, digunakan beberapa metrik evaluasi yang umum dalam klasifikasi biner yaitu Accuracy (Akurasi), Precision (Presisi), Recall (Sensitivitas), F1-score.

Setiap metrik tersebut menunjukkan:
1. Accuracy
Mengukur seberapa banyak prediksi yang benar dibandingkan dengan total jumlah prediksi.
![Accuracy](https://raw.githubusercontent.com/pandutanjung/hotel-booking-cancelation/4c2b6019eb34d161735ae477bb86ef00b4a4114e/images/accuracy.png)
2. Precision
Mengukur proporsi prediksi positif yang benar-benar positif. Penting saat false positive harus diminimalisir.
![Precision](https://raw.githubusercontent.com/pandutanjung/hotel-booking-cancelation/4c2b6019eb34d161735ae477bb86ef00b4a4114e/images/precision.png)
3. Recall
Mengukur kemampuan model mendeteksi seluruh kasus positif (pembatalan sebenarnya). Penting saat false negative berisiko tinggi.
![Recall](https://raw.githubusercontent.com/pandutanjung/hotel-booking-cancelation/4c2b6019eb34d161735ae477bb86ef00b4a4114e/images/recall.png)
4. F1-Score
Merupakan rata-rata harmonik dari precision dan recall, digunakan saat ingin mempertimbangkan keduanya secara seimbang.
![F1 Score](https://raw.githubusercontent.com/pandutanjung/hotel-booking-cancelation/4c2b6019eb34d161735ae477bb86ef00b4a4114e/images/f1.png)

Hasil proyek berdasarkan metrik evaluasi

![Confusion Matrix - Logistic Regression](https://raw.githubusercontent.com/pandutanjung/hotel-booking-cancelation/934860c18f82520881b069642ca3bc4dab80dd92/images/confu-lr.png)
- Accuracy: sangat tinggi
- Recall: masih cukup baik (meski ada 220 false negatives)
- Precision: sangat baik (hanya 1 false positives)
- F1-score: tinggi, tapi kalah dari Random Forest
Logistic Regression memberikan performa yang solid, namun tidak mampu menangkap semua kasus pembatalan secara sempurna.

![Confusion Matrix - Random Forest](https://raw.githubusercontent.com/pandutanjung/hotel-booking-cancelation/934860c18f82520881b069642ca3bc4dab80dd92/images/confu-rf.png)
- Accuracy = 100%
- Recall = 100%
- Precision = 100%
- F1-score = 1.0
Random Forest menunjukkan performa sempurna pada data uji: tidak ada satu pun pembatalan atau non-pembatalan yang salah klasifikasi. Berdasarkan metrik evaluasi, model ini unggul mutlak dalam hal akurasi dan keseimbangan antar metrik.

Metrik seperti recall dan F1-score sangat penting dalam konteks prediksi pembatalan karena:
- Recall yang tinggi membantu mendeteksi lebih banyak potensi pembatalan.
- Precision yang tinggi memastikan prediksi pembatalan tidak banyak yang salah.

Dengan semua metrik menunjukkan hasil sempurna, Random Forest dinilai sebagai model terbaik dan paling andal untuk digunakan dalam sistem prediksi pembatalan hotel ini.
