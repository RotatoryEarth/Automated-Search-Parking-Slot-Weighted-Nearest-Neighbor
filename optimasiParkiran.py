import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Optional, Tuple, Any

#  1. KONFIGURASI FILE
FILENAME = "parkiran2.csv" 
USER_X, USER_Y = 0, 0 

#  2. LOAD DATASET
def load_dataset(filename: str) -> pd.DataFrame:
    """Membaca file CSV dan memastikan kolom yang diperlukan ada."""
    try:
        df = pd.read_csv(filename)
        df['Safety'] = df['Safety'].astype(str).str.strip()
        df['Lokasi'] = df['Lokasi'].apply(lambda x: True if x == 1 else False)

        valid_safety = ['aman', 'kurang aman', 'tidak aman']
        df = df[df['Safety'].str.lower().isin(valid_safety) | (df['Safety'].str.lower() == 'outdoor') | (df['Safety'].str.lower() == 'indoor')].copy()

        df['Keamanan_Kualitas'] = df['Safety'].apply(
            lambda x: 'Aman' if x.lower() in ['indoor', 'outdoor'] else x
        )

        return df
    except FileNotFoundError:
        print(f"Error: File '{filename}' tidak ditemukan.")
        return pd.DataFrame()
    except KeyError as e:
        print(f"Error: Kolom {e} tidak ditemukan. Pastikan CSV memiliki kolom 'Safety' dan 'Lokasi'.")
        return pd.DataFrame()


#  3. FUNGSI PENDUKUNG
def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Menghitung Jarak Euclidean."""
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

#  4. RUMUS BOBOT 3 FAKTOR (Sesuai ReadMe)
def calculate_normalized_rating(is_indoor: bool, quality_status: str) -> float:
    """
    Menghitung dan Menormalisasi Rating Kualitas Parkir (R_norm)
    menggunakan 3 kriteria: Lokasi (+1/+0), Keamanan (+2/+1/-1).
    """
    
    # 1. Skor Lokasi
    lokasi_score = 1 if is_indoor else 0 # Indoor: +1, Outdoor: 0

    # 2. Skor Keamanan Kualitas
    keamanan_score = 0
    status = quality_status.lower()
    
    if "aman" in status:
        keamanan_score = 2
    elif "kurang aman" in status:
        keamanan_score = 1
    elif "tidak aman" in status:
        keamanan_score = -1
    else:
        keamanan_score = 2 

    # 3. Total Rating 
    r_total = lokasi_score + keamanan_score
    
    # 4. Normalisasi Rating 
    r_norm = (r_total + 1) / 4
    
    return max(0.0, min(1.0, r_norm))


def weighted_nearest(df: pd.DataFrame, user_x: float = USER_X, user_y: float = USER_Y) -> Optional[pd.Series]:
    """Mencari parkiran dengan Skor Optimasi tertinggi."""
    best_score = -float("inf")
    best_spot = None
    
    DISTANCE_MULTIPLIER = 10 
    EPSILON = 1e-5 

    for _, row in df.iterrows():
        if row["Slot"] > 0:
            
            # A. Komponen Kualitas 
            r_norm = calculate_normalized_rating(row['Lokasi'], row['Safety'])
            
            # B. Komponen Jarak
            dist = distance(user_x, user_y, row["x"], row["y"])
            
            # C. Skor Optimasi Akhir
            final_score = r_norm * (DISTANCE_MULTIPLIER / (dist + EPSILON))
            
            if final_score > best_score:
                best_score = final_score
                best_spot = row
    return best_spot

#  5. ALGORITMA BRUTE FORCE
def brute_force_search(df: pd.DataFrame, user_x: float = USER_X, user_y: float = USER_Y) -> Optional[pd.Series]:
    min_dist = float("inf")
    best_spot = None
    for _, row in df.iterrows():
        if row["Slot"] > 0:
            d = distance(user_x, user_y, row["x"], row["y"])
            if d < min_dist:
                min_dist = d
                best_spot = row
    return best_spot

#  6. VISUALISASI PETA
def plot_map(df: pd.DataFrame, user_x: float, user_y: float, best_brute: Optional[pd.Series], best_weighted: Optional[pd.Series]):
    """Menampilkan peta parkir dengan hasil rekomendasi."""
    plt.figure(figsize=(10, 6))
    
    # User
    plt.scatter(user_x, user_y, c='red', marker='X', s=150, label='User', zorder=5)
    
    # Parkiran (Tersedia vs Penuh)
    available = df[df['Slot'] > 0]
    full = df[df['Slot'] == 0]
    plt.scatter(available['x'], available['y'], c='blue', alpha=0.5, label='Tersedia')
    plt.scatter(full['x'], full['y'], c='gray', alpha=0.3, label='Penuh')

    # Hasil
    if best_brute is not None:
        plt.scatter(best_brute['x'], best_brute['y'], s=250, facecolors='none', edgecolors='orange', linewidth=2, label='Brute Force (Terdekat)')
    if best_weighted is not None:
        plt.scatter(best_weighted['x'], best_weighted['y'], c='green', marker='*', s=200, label='Weighted (Rekomendasi)')
        
        lokasi_str = "Indoor" if best_weighted['Lokasi'] else "Outdoor"
        detail_text = f"PILIHAN: {best_weighted['ID']}\n{lokasi_str} ({best_weighted['Safety']})"
        plt.text(best_weighted['x']+1, best_weighted['y']+1, detail_text, color='green', fontweight='bold')

    plt.title(f"Peta Parkir (Total Data: {len(df)})")
    plt.xlabel("Posisi X")
    plt.ylabel("Posisi Y")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

#  7. VISUALISASI PERFORMA
def compare_performance_real(df: pd.DataFrame):
    """Menguji performa dengan incremental slicing data asli."""
    total_rows = len(df)
    print(f"\n--- Memulai Tes Performa pada {total_rows} data ---")
    
    if total_rows < 10:
        print("Data terlalu sedikit (<10) untuk membuat grafik performa yang valid. Harap gunakan dataset yang lebih besar.")
        return

    steps = np.linspace(min(10, total_rows), total_rows, num=10, dtype=int)
    steps = sorted(list(set(steps)))

    times_brute = []
    times_weighted = []

    print(f"Menguji pada titik data: {steps}")

    for n in steps:
        subset_df = df.iloc[:n].reset_index(drop=True)

        # 1. Ukur Brute Force
        start = time.perf_counter()
        brute_force_search(subset_df)
        end = time.perf_counter()
        times_brute.append(end - start)

        # 2. Ukur Weighted
        start = time.perf_counter()
        weighted_nearest(subset_df)
        end = time.perf_counter()
        times_weighted.append(end - start)
        
    # Plot Grafik 
    plt.figure(figsize=(10, 6))
    plt.plot(steps, times_brute, marker='o', linestyle='-', color='orange', label='Brute Force (Jarak Saja)')
    plt.plot(steps, times_weighted, marker='s', linestyle='--', color='green', label='Weighted Nearest (Skor Optimasi)')
    
    plt.title(f"Perbandingan Performa (Berdasarkan {total_rows} Data Asli)")
    plt.xlabel("Jumlah Data yang Diproses")
    plt.ylabel("Waktu Eksekusi (Detik)")
    plt.legend()
    plt.grid(True)
    plt.show()

#  MAIN PROGRAM
if __name__ == "__main__":
    df_parkir = load_dataset(FILENAME)
    
    if not df_parkir.empty:
        # 1. Tampilkan Peta & Hasil Rekomendasi
        print(f"Dataset dimuat: {len(df_parkir)} baris.")
        
        bf = brute_force_search(df_parkir, USER_X, USER_Y)
        wn = weighted_nearest(df_parkir, USER_X, USER_Y)
        
        if wn is not None:
            r_norm = calculate_normalized_rating(wn['Lokasi'], wn['Safety'])
            print(f"Rekomendasi Terbaik: {wn['ID']} ({'Indoor' if wn['Lokasi'] else 'Outdoor'}, {wn['Safety']})")
            print(f"Normalisasi Rating (R_norm): {r_norm:.2f}")
        
        plot_map(df_parkir, USER_X, USER_Y, bf, wn)

        # 2. Tampilkan Grafik Performa
        print("\nApakah Anda ingin melihat grafik perbandingan performa? (y/n)")
        choice = input("> ").lower()
        if choice == 'y':
            compare_performance_real(df_parkir)
    else:
        print("Gagal memuat data. Pastikan file 'parkiran2.csv' ada dan memiliki kolom 'Safety' dan 'Lokasi'.")