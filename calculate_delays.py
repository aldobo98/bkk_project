import glob
import os
import re
import zipfile
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

GTFS_DIR = "gtfs"  # bkk_gtfs_ÉÉÉÉHHNN.zip fájlok, lásd download_gtfs.py


def time_str_to_seconds(series):
    """
    Átalakítja az óra:perc:másodperc formátumú időket másodpercekké (vektorizáltan).
    A GTFS-ben az óra 24 fölötti is lehet (éjfél utáni járatok).
    Hibás adat esetén NaN-t ad vissza.
    """
    parts = series.str.strip().str.split(':', expand=True)
    parts = parts.apply(pd.to_numeric, errors='coerce')
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def list_gtfs_versions(gtfs_dir):
    """
    Visszaadja a rendelkezésre álló GTFS változatokat dátum szerint rendezve:
    [(datum, zip_utvonal), ...]
    """
    versions = []
    for path in glob.glob(os.path.join(gtfs_dir, "bkk_gtfs_*.zip")):
        m = re.search(r"bkk_gtfs_(\d{8})\.zip$", path)
        if m:
            versions.append((pd.Timestamp(m.group(1)).date(), path))
    if not versions:
        raise FileNotFoundError(
            f"Nincs GTFS a '{gtfs_dir}' mappában. Futtasd: python download_gtfs.py <kezdő nap> <utolsó nap>")
    return sorted(versions)


def pick_version(versions, day):
    """
    Az adott napon érvényes GTFS változat: a legutolsó, amelyik nem későbbi a napnál.
    Ha a nap minden változatnál korábbi, a legkorábbit adja vissza.
    """
    valid = [v for v in versions if v[0] <= day]
    return valid[-1] if valid else versions[0]


def prepare_stop_times(zip_path, trip_ids=None):
    """
    Beolvassa a megállók menetrendi adatait (stop_times.txt) egy GTFS zip-ből.
    Ha trip_ids meg van adva, csak ezeket a járatokat tartja meg (memóriatakarékosság).
    Visszaad egy DataFrame-et a másodpercekben kifejezett tervezett érkezési időkkel.
    """
    cols = ['trip_id', 'stop_sequence', 'arrival_time']
    with zipfile.ZipFile(zip_path) as z:
        stops = pd.read_csv(z.open('stop_times.txt'), usecols=cols,
                            dtype={'trip_id': str, 'arrival_time': str})
    stops['trip_id'] = stops['trip_id'].str.strip()
    if trip_ids is not None:
        stops = stops[stops['trip_id'].isin(trip_ids)]
    stops['planned_sec'] = time_str_to_seconds(stops['arrival_time'])
    # Duplikációk kiszűrése, hogy a merge ne növelje meg a sorok számát
    return stops[['trip_id', 'stop_sequence', 'planned_sec']].drop_duplicates(subset=['trip_id', 'stop_sequence'])


def calculate_delay(chunk, stops_df):
    """
    Kiszámítja a késést a GPS-adatok egy adott darabjára (egy nap adatai).
    A helyi (jármű által küldött) időpontot veti össze a menetrendi idővel.
    """
    merged = pd.merge(chunk[['trip_id', 'current_stop_sequence']], stops_df,
                      left_on=['trip_id', 'current_stop_sequence'],
                      right_on=['trip_id', 'stop_sequence'],
                      how='left')

    # Az éjféltől eltelt aktuális másodpercek kiszámítása
    v_ts_local = pd.to_datetime(chunk['vehicle_timestamp'], errors='coerce')
    actual_sec = (v_ts_local.dt.hour * 3600 + v_ts_local.dt.minute * 60 + v_ts_local.dt.second).to_numpy()

    delay = actual_sec - merged['planned_sec'].to_numpy()

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
    data['trip_id'] = data['trip_id'].astype(str).str.strip()

    # Dátum formátum ellenőrzése és hibás sorok számlálása
    data['timestamp_utc'] = pd.to_datetime(data['timestamp_utc'], errors='coerce')
    invalid_rows = data['timestamp_utc'].isna().sum()
    print(f"Invalid rows: {invalid_rows}")

    # 2. Minden sorhoz a mérés napja (megjegyzés: az időbélyegek helyi idő szerintiek),
    #    és ehhez a naphoz tartozó GTFS változat
    versions = list_gtfs_versions(GTFS_DIR)
    print("Elérhető GTFS változatok:", ", ".join(str(d) for d, _ in versions))
    data['_day'] = data['timestamp_utc'].dt.date

    data['delay_seconds'] = np.nan
    data['gtfs_version'] = None

    # 3. Késések kiszámítása naponként, a napon érvényes menetrenddel.
    #    Az éjfél utáni járatok miatt az előző napi változatot is figyelembe vesszük
    #    (az aznapi változat élvez elsőbbséget).
    cache = {}
    for day, idx in tqdm(data.groupby('_day').groups.items(), desc="Napok"):
        day_rows = data.loc[idx]
        trip_ids = set(day_rows['trip_id'].unique())

        cur = pick_version(versions, day)
        prev = pick_version(versions, day - pd.Timedelta(days=1))
        needed = list(dict.fromkeys([cur[1], prev[1]]))
        cache = {p: cache[p] for p in needed if p in cache}  # a már nem kellő változatok eldobása
        frames = []
        for path in needed:
            if path not in cache:
                cache[path] = prepare_stop_times(path)
            frames.append(cache[path][cache[path]['trip_id'].isin(trip_ids)])
        stops_lookup = pd.concat(frames).drop_duplicates(subset=['trip_id', 'stop_sequence'], keep='first')

        delays = calculate_delay(day_rows, stops_lookup)
        data.loc[idx, 'delay_seconds'] = delays
        data.loc[idx, 'gtfs_version'] = str(cur[0])

        matched = np.isfinite(delays).mean()
        tqdm.write(f"  {day}: GTFS {cur[0]}, {len(day_rows):,} sor, {matched:.1%} párosítva")

    data = data.drop(columns=['_day'])
    print("KÉSZ! A 'delay_seconds' oszlop frissítve.")

    # 4. Csak azokat a sorokat tartjuk meg, ahol sikerült kiszámolni a késést (nincs NaN)
    df_filtered = data[data['delay_seconds'].notna()]
    print(f"Megtartott sorok: {len(df_filtered):,} / {len(data):,}, "
          f"járatok (route_id): {df_filtered['route_id'].nunique()} / {data['route_id'].nunique()}")

    # Végleges eredmény elmentése egy új fájlba
    df_filtered.to_csv("vehicle_updates_delays_filtered.csv", index=False)

    print("A késések szűrt adatai elmentve a 'vehicle_updates_delays_filtered.csv' fájlba.")


if __name__ == "__main__":
    main()
