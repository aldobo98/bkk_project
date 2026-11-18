"""
A BKK GTFS menetrend napi archív változatainak letöltése a Mobility Database-ből
(https://mobilitydatabase.org/feeds/gtfs/mdb-990).

A BKK naponta új GTFS-t publikál, és a trip_id-k változatonként eltérhetnek, ezért
a késésszámításhoz a mérés napján érvényes változat kell. A Mobility Database minden
éjjel letölti a feedet, a fájlok neve: mdb-990-ÉÉÉÉHHNNÓÓPP.zip (UTC időbélyeg).
A fájlok listázása nem nyilvános, ezért az éjféli órák perceit próbáljuk végig.

Használat:
    python download_gtfs.py 2026-09-28 2026-10-12
"""
import os
import sys
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

FEED = "mdb-990"
BASE_URL = "https://files.mobilitydatabase.org/{feed}/{feed}-{ts}/{feed}-{ts}.zip"
OUT_DIR = "gtfs"
SEARCH_HOURS = range(0, 6)  # az archiválás jellemzően 00:00 és 03:00 UTC között fut


def url_for(ts):
    return BASE_URL.format(feed=FEED, ts=ts)


def exists(ts):
    """Igaz, ha az adott időbélyegű archív fájl létezik (HEAD kérés)."""
    req = urllib.request.Request(url_for(ts), method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status == 200
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return False


def find_snapshot(day, pool):
    """Megkeresi az adott napi archív változat időbélyegét, vagy None-t ad vissza."""
    candidates = [f"{day:%Y%m%d}{h:02d}{m:02d}" for h in SEARCH_HOURS for m in range(60)]
    for ts, ok in zip(candidates, pool.map(exists, candidates)):
        if ok:
            return ts
    return None


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    start, end = (date.fromisoformat(a) for a in sys.argv[1:3])
    os.makedirs(OUT_DIR, exist_ok=True)

    with ThreadPoolExecutor(max_workers=32) as pool:
        day = start
        while day <= end:
            target = os.path.join(OUT_DIR, f"bkk_gtfs_{day:%Y%m%d}.zip")
            if os.path.exists(target):
                print(f"{day}: már letöltve")
            else:
                ts = find_snapshot(day, pool)
                if ts is None:
                    print(f"{day}: nincs archív változat")
                else:
                    print(f"{day}: letöltés ({ts})...")
                    urllib.request.urlretrieve(url_for(ts), target + ".part")
                    os.replace(target + ".part", target)
            day += timedelta(days=1)


if __name__ == "__main__":
    main()
