import os
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import time

def time_str_to_seconds(t_str):
    """
    Átalakítja az óra:perc:másodperc formátumú időt másodpercekké.
    Ha nem sikerül (pl. hibás adat), NaN-t ad vissza.
    """
    try:
        parts = t_str.split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except:
        return np.nan
    
def prepare_stop_times(path):
    """
    Beolvassa és előkészíti a megállók menetrendi adatait (stop_times.txt).
    Visszaad egy DataFrame-et a szükséges oszlopokkal és a másodpercekben
    kifejezett tervezett érkezési időkkel.
    """
    cols = ['trip_id', 'stop_id', 'stop_sequence', 'arrival_time']
    stops = pd.read_csv(path, usecols=lambda c: c in cols, dtype={'trip_id': str, 'stop_id': str})
    stops['trip_id'] = stops['trip_id'].str.strip()
    stops['stop_id'] = stops['stop_id'].str.strip()
    stops['planned_sec'] = stops['arrival_time'].apply(time_str_to_seconds)
    # Duplikációk kiszűrése, hogy a merge ne növelje meg a sorok számát
    return stops[['trip_id', 'stop_id', 'stop_sequence', 'planned_sec']].drop_duplicates(subset=['trip_id', 'stop_sequence'])

def calculate_delay_on_chunk(chunk, stops_df):
    """
    Kiszámítja a késést a GPS-adatok egy adott darabjára (chunk).
    A helyi (jármű által küldött) időpontot veti össze a menetrendi idővel.
    """
    chunk['trip_id'] = chunk['trip_id'].astype(str).str.strip()
    
    # Elég a trip_id és a megálló sorszáma alapján összekapcsolni
    merged = pd.merge(chunk, stops_df, 
                      left_on=['trip_id', 'current_stop_sequence'],
                      right_on=['trip_id', 'stop_sequence'],
                      how='left')
    
    # A jármű időbélyegének átalakítása datetime objektummá.
    v_ts_local = pd.to_datetime(merged['vehicle_timestamp'], errors='coerce')

    # Az éjféltől eltelt aktuális másodpercek kiszámítása
    actual_sec = v_ts_local.dt.hour * 3600 + v_ts_local.dt.minute * 60 + v_ts_local.dt.second
    
    delay = (actual_sec - merged['planned_sec']).to_numpy()
    
    # Éjféli átnyúlás kezelése (NaN értékek figyelmen kívül hagyásával)
    delay = np.where(delay < -43200, delay + 86400, delay)
    delay = np.where(delay > 43200, delay - 86400, delay)
    
    return delay



def main():
    # 1. Jármű pozícióadatok beolvasása, szigorú típusmegadással, hogy a kezdő nullák ne vesszenek el
    data = pd.read_csv(
        "vehicle_updates.csv", 
        usecols=range(12), 
        on_bad_lines='skip',
        dtype={'trip_id': str, 'route_id': str, 'stop_id': str}
    )  

    # Dátum formátum ellenőrzése és hibás sorok számlálása
    data['timestamp_utc'] = pd.to_datetime(data['timestamp_utc'], errors='coerce')
    invalid_rows = data['timestamp_utc'].isna().sum()
    print(f"Invalid rows: {invalid_rows}")

    df_filtered = data.copy()
    
    # 2. Menetrendi referenciatábla előkészítése
    stops_lookup = prepare_stop_times("stop_times.txt")

    DF_IN_MEMORY = df_filtered
    BATCH_SIZE = 500_000

    # A DataFrame felosztása kisebb darabokra a gyorsabb és memóriabarát feldolgozás érdekében
    chunks = np.array_split(DF_IN_MEMORY, len(DF_IN_MEMORY) // BATCH_SIZE + 1)

    print(f"Feldolgozás indítása {len(chunks)} részletben...")

    results = []
    start_global = time.time()

    # 3. Késések kiszámítása darabonként, folyamatjelzővel
    for i, chunk in enumerate(tqdm(chunks, desc="Számolás")):
        delays = calculate_delay_on_chunk(chunk.copy(), stops_lookup)
        results.append(delays)
    
        if i % 10 == 0:
            elapsed = time.time() - start_global
            print(f"  -> {i+1}/{len(chunks)} csomag kész ({elapsed:.1f}s)")

    print("Eredmények összefűzése...")
    DF_IN_MEMORY['delay_seconds'] = np.concatenate(results)

    print("KÉSZ! A 'delay_seconds' oszlop frissítve.")
    print(DF_IN_MEMORY[['vehicle_timestamp', 'trip_id', 'delay_seconds']].head())

    df_filtered = DF_IN_MEMORY

    # 4. Csak azokat a sorokat tartjuk meg, ahol sikerült kiszámolni a késést (nincs NaN)
    df_filtered = df_filtered[df_filtered['delay_seconds'].notna()]

    # Végleges eredmény elmentése egy új fájlba
    df_filtered.to_csv("vehicle_updates_delays_filtered.csv", index=False)

    print("A késések szűrt adatai elmentve a 'vehicle_updates_delays_filtered.csv' fájlba.")
        
if __name__ == "__main__":
    main()
