# Optimasi Pencarian Lokasi Parkir Terdekat

Tujuan dari algoritma ini adalah merekomendasikan tempat parkir terbaik yang bukan hanya terdekat, tetapi juga mempertimbangkan faktor kualitas (keamanan dan lokasi).

---

## 1. Struktur Data Lokasi Parkir (parkiran2.csv)

Setiap lokasi parkir diwakili oleh enam kolom, memisahkan faktor lokasi dan keamanan untuk perhitungan yang lebih akurat:

| Atribut | Deskripsi | Tipe Data |
| :--- | :--- | :--- |
| **x, y** | Koordinat lokasi parkir. | Numerik |
| **Slot** | Jumlah slot parkir kosong (harus > 0 agar layak dipilih). | Numerik |
| **Safety** | Tingkat Keamanan Kualitas (`Aman`, `Kurang aman`, `Tidak aman`). | Teks |
| **Lokasi** | Tipe Lokasi (1 = Indoor, 0 = Outdoor). | Numerik/Boolean |

---

## 2. Algoritma Pencarian

Digunakan dua metode untuk perbandingan performa:

### A. Brute Force (Pencarian Jarak Saja)

Murni mencari lokasi parkir dengan **Jarak Euclidean terpendek** dari pengguna.

### B. Weighted Nearest (Pencarian Optimal)

Mencari lokasi parkir dengan **Skor Optimasi TERTINGGI** berdasarkan kombinasi Jarak dan Kualitas Parkir.

---

## 3. Rumus Skor Optimasi Tiga Faktor

Skor Optimasi ($S$) adalah hasil perkalian antara **Skor Kualitas Parkir** ($R_{norm}$) dan **Faktor Kedekatan** ($F_{dist}$).



### A. Komponen 1: Skor Kualitas Parkir ($R_{norm}$)

Rating dihitung dari dua sub-komponen terpisah (Lokasi dan Keamanan) lalu dinormalisasi.

#### 1. Konversi Skor Rating ($R_{total}$)

| Kriteria | Status | Skor |
| :--- | :--- | :---: |
| **Lokasi** | Indoor (Nilai 1 di kolom Lokasi) | +1 |
| | Outdoor (Nilai 0 di kolom Lokasi) | 0 |
| **Keamanan** | Aman | +2 |
| | Kurang aman | +1 |
| | Tidak aman | -1 |

**Rumus Total Rating:**
$$
R_{total} = \text{Skor Lokasi} + \text{Skor Keamanan}
$$

*Range total rating: **-1 hingga 3**.*

#### 2. Normalisasi Rating ($R_{norm}$)

Rating total dinormalisasi ke rentang $0.0$ hingga $1.0$. Nilai ini menjadi bobot utama kualitas parkir.

$$
R_{norm} = \frac{R_{total} + 1}{4}
$$

### B. Komponen 2: Faktor Kedekatan ($F_{dist}$)

Jarak Euclidean ($D$) dihitung dan dibalik untuk memastikan bahwa jarak yang lebih kecil menghasilkan skor yang lebih besar.

**Rumus Jarak Euclidean:**
$$
D = \sqrt{(x_{\text{user}} - x_{\text{parkir}})^2 + (y_{\text{user}} - y_{\text{parkir}})^2}
$$

**Rumus Faktor Kedekatan:**
$$
F_{dist} = \frac{10}{D + 1e-5}
$$
*(Angka 10 adalah pengali skala, dan $1e-5$ adalah $\epsilon$ untuk menghindari pembagian dengan nol.)*

### C. Rumus Skor Optimasi Akhir

Skor yang menentukan rekomendasi terbaik.

$$
\text{Skor Optimasi} (S) = R_{norm} \times F_{dist}
$$

**Rekomendasi Terbaik:** Tempat parkir dengan $S$ tertinggi.