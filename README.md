# Laporan Proyek Machine Learning - Pandu Persada Tanjung

## Domain Proyek

Industri perhotelan menghadapi tantangan besar terkait pembatalan pemesanan reservasi oleh pelanggan. Dalam beberapa kasus, pelanggan cenderung membatalkan pemesanan mereka tanpa pemberitahuan yang cukup jauh hari, yang menyulitkan pihak hotel untuk mengganti pemesanan yang hilang. Oleh karena itu, penting untuk mengidentifikasi pola-pola dalam pembatalan guna mengantisipasi dan mengelola pembatalan dengan lebih baik (Hermawan et al., 2025).

Menurut Antonio et al. (2019), penerapan algoritma pembelajaran mesin dalam memprediksi pembatalan reservasi telah terbukti mampu meningkatkan ketepatan prediksi dan membantu meminimalkan kerugian dari pembatalan mendadak. Berbagai pendekatan seperti regresi logistik, pohon keputusan, hingga model ensemble seperti Random Forest dan XGBoost dimanfaatkan untuk mengenali faktor-faktor yang memengaruhi pembatalan serta untuk mendeteksi pemesanan yang berpotensi tinggi dibatalkan.

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

## Data Preparation
1. Salin DataFrame
Bertujuan untuk menghindari modifikasi langsung pada DataFrame asli
2. Menangani Missing Values
-children diisi dengan median
-country dengan modus
-agent dengan 0 (tanpa agen)
-company dihapus karena mayoritas kosong
3. Encoding Fitur Kategorikal
Semua kolom bertipe object (kategorikal) diubah menjadi numerik menggunakan Label Encoding.
4. Split Fitur dan Target
5. Train-Test Split
Memisahkan data latih (80%) dan uji (20%)
6. Undersampling Kelas Mayoritas
Menyeimbangkan kelas target is_canceled agar tidak bias. Not Canceled (0) disampling acak agar jumlahnya sama dengan Canceled (1)
![Distribusi Setelah Undersampling](https://raw.githubusercontent.com/pandutanjung/hotel-booking-cancelation/b0479ebbd6dc905313153ae4d1650c195f1b2751/images/hotel-cancelvsnot-under.png)

7. Standarisasi (Scaling)
Menggunakan StandardScaler untuk menyesuaikan skala semua fitur

## Modeling
Model 1: Logistic Regression 
Logistic Regression merupakan algoritma yang sederhana, cepat dilatih, dan sangat cocok digunakan sebagai model baseline dalam masalah klasifikasi. Salah satu keunggulan utamanya adalah kemampuannya dalam memberikan interpretasi yang jelas terhadap pengaruh masing-masing fitur melalui nilai koefisien. Hal ini menjadikan Logistic Regression sangat berguna ketika interpretabilitas menjadi prioritas utama. Namun, model ini memiliki keterbatasan karena hanya mampu menangkap hubungan linear antar fitur, sehingga kurang efektif dalam menangani pola kompleks atau non-linear. Meskipun performanya cukup baik dalam eksperimen ini, Logistic Regression menghasilkan 239 kesalahan klasifikasi untuk kasus pembatalan (false negatives) dan 7 prediksi positif yang salah (false positives). Hal ini menunjukkan bahwa Logistic Regression masih kurang optimal dalam mendeteksi kasus pembatalan, yang justru menjadi fokus utama dalam konteks bisnis hotel.

Model 2: Random Forest
Random Forest adalah algoritma berbasis ensemble yang sangat kuat dan mampu menangani berbagai jenis data, termasuk yang memiliki hubungan non-linear dan fitur saling berinteraksi. Dalam eksperimen ini, Random Forest menunjukkan performa yang sangat tinggi dengan tidak melakukan kesalahan prediksi sama sekali pada data uji (zero false positives dan zero false negatives). Hal ini mencerminkan kemampuannya dalam memetakan pola pembatalan dengan sangat baik. Kekurangannya terletak pada sifatnya yang kompleks, sehingga sulit untuk diinterpretasikan. Selain itu, waktu pelatihan dan komputasi relatif lebih besar dibanding Logistic Regression. Meskipun begitu, keunggulan dari segi akurasi dan kemampuan generalisasi menjadikan Random Forest sebagai pilihan yang lebih unggul.

Berdasarkan evaluasi keseluruhan, Random Forest dipilih sebagai model terbaik karena memberikan hasil prediksi paling akurat dan stabil. Random Forest terbukti mampu mengidentifikasi baik kasus pembatalan maupun non-pembatalan dengan sangat baik. Dalam konteks bisnis hotel, akurasi tinggi dalam memprediksi pembatalan sangat krusial untuk mengantisipasi potensi kerugian dan menyusun strategi operasional yang lebih efektif.

## Evaluation

Dalam proyek prediksi pembatalan hotel ini, digunakan beberapa metrik evaluasi yang umum dalam klasifikasi biner yaitu Accuracy (Akurasi), Precision (Presisi), Recall (Sensitivitas), F1-score.

Setiap metrik tersebut menunjukkan:
1. Accuracy
Mengukur seberapa banyak prediksi yang benar dibandingkan dengan total jumlah prediksi.
2. Precision
Mengukur proporsi prediksi positif yang benar-benar positif. Penting saat false positive harus diminimalisir.
3. Recall
Mengukur kemampuan model mendeteksi seluruh kasus positif (pembatalan sebenarnya). Penting saat false negative berisiko tinggi.
4. F1-Score
Merupakan rata-rata harmonik dari precision dan recall, digunakan saat ingin mempertimbangkan keduanya secara seimbang.

Hasil proyek berdasarkan metrik evaluasi
![Confusion Matrix - Logistic Regression](https://raw.githubusercontent.com/pandutanjung/hotel-booking-cancelation/b0479ebbd6dc905313153ae4d1650c195f1b2751/images/hotel-cflog.png)
- Accuracy: sangat tinggi
- Recall: masih cukup baik (meski ada 239 false negatives)
- Precision: sangat baik (hanya 7 false positives)
- F1-score: tinggi, tapi kalah dari Random Forest
Logistic Regression memberikan performa yang solid, namun tidak mampu menangkap semua kasus pembatalan secara sempurna.

![Confusion Matrix - Random Forest](https://raw.githubusercontent.com/pandutanjung/hotel-booking-cancelation/b0479ebbd6dc905313153ae4d1650c195f1b2751/images/hotel-cfrf.png)
- Accuracy = 100%
- Recall = 100%
- Precision = 100%
- F1-score = 1.0
Random Forest menunjukkan performa sempurna pada data uji: tidak ada satu pun pembatalan atau non-pembatalan yang salah klasifikasi. Berdasarkan metrik evaluasi, model ini unggul mutlak dalam hal akurasi dan keseimbangan antar metrik.

Metrik seperti recall dan F1-score sangat penting dalam konteks prediksi pembatalan karena:
- Recall yang tinggi membantu mendeteksi lebih banyak potensi pembatalan.
- Precision yang tinggi memastikan prediksi pembatalan tidak banyak yang salah.

Dengan semua metrik menunjukkan hasil sempurna, Random Forest dinilai sebagai model terbaik dan paling andal untuk digunakan dalam sistem prediksi pembatalan hotel ini.
