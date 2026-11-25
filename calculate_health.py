"""
Adatminőség-elemzés a UI "Adatminőség" tabjához.

A szűretlen vehicle_updates.csv-ből számol (abban minden lekérdezett sor benne van,
így a hézagok valódi elnémulást jelentenek, nem a késésszámítás szűrését), és kis
CSV fájlokat ír a data_health/ mappába, amelyeket a UI olvas be.

Futtatás (a project mappából):
    python calculate_health.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = "vehicle_updates.csv"
OUT_DIR = Path("data_health")
LOCAL_TZ = "Europe/Budapest"
TS_FORMAT = "%Y-%m-%d %H:%M:%S"
POLL_GAP = 150                  # ennyi mp-ig senkitől sem jött adat = a gyűjtő script kiesése
GAP_MIN, GAP_MAX = 180, 900     # egy jármű ennyi ideig néma = kiesés; ennél hosszabb = valószínűleg végállomási állás / nincs forgalomban
MISSED_POLLS = 2                # ennyi lefutott lekérdezésből kell hiányoznia a járműnek


def load():
    """Beolvassa a szűretlen adatot; járművenként időrendben."""
    df = pd.read_csv(
        SOURCE,
        usecols=["timestamp_utc", "vehicle_id", "route_id", "lat", "lon", "vehicle_timestamp"],
        dtype={"vehicle_id": "category", "route_id": "category"},
        on_bad_lines="skip",
    )
    received = pd.to_datetime(df["timestamp_utc"], format=TS_FORMAT, errors="coerce")
    vehicle_ts = pd.to_datetime(df["vehicle_timestamp"], format=TS_FORMAT, errors="coerce")
    df = df.drop(columns=["timestamp_utc", "vehicle_timestamp"])
    # késleltetés = lekérdezés ideje - a jármű saját időbélyege (a kettő ugyanabban az időzónában van)
    df["lag_s"] = (received - vehicle_ts).dt.total_seconds().astype("float32")
    df["time"] = received.dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    df = df.dropna(subset=["time"]).sort_values("time", kind="stable").reset_index(drop=True)
    df["vehicle"] = df["vehicle_id"].cat.rename_categories(
        lambda c: c.replace("VehiclePosition-", ""))
    return df.drop(columns="vehicle_id")


def main():
    df = load()
    print(f"Beolvasott sorok: {len(df):,}, járművek: {df['vehicle'].nunique():,}")
    OUT_DIR.mkdir(exist_ok=True)

    # 1) a gyűjtő script kiesései: egyetlen jármű adata sem érkezett
    polls = np.sort(df["time"].unique())
    poll_gap = np.diff(polls) / np.timedelta64(1, "s")
    out = poll_gap > POLL_GAP
    outages = pd.DataFrame({"from": polls[:-1][out], "to": polls[1:][out],
                            "minutes": (poll_gap[out] / 60).round(1)})

    # 2) járművenkénti jelentési időköz és kiesések
    by_vehicle = df.groupby("vehicle", sort=False, observed=True)
    interval = by_vehicle["time"].diff().dt.total_seconds()
    prev = by_vehicle[["lat", "lon", "route_id"]].shift()
    # csak akkor kiesés, ha a jármű járaton volt, amikor elnémult (a garázsban leállítás nem az)
    is_gap = interval.between(GAP_MIN, GAP_MAX, inclusive="left") & prev["route_id"].notna()
    gaps = pd.DataFrame({
        "vehicle": df.loc[is_gap, "vehicle"].astype(str),
        "route_id": prev.loc[is_gap, "route_id"].astype(str),
        "lat": prev.loc[is_gap, "lat"],      # utolsó ismert pozíció az elnémulás előtt
        "lon": prev.loc[is_gap, "lon"],
        "end": df.loc[is_gap, "time"],
        "minutes": (interval[is_gap] / 60).round(1),
    }).dropna(subset=["lat", "lon"])
    # a gyűjtő kiesése miatti hézagok nem a jármű hibái: csak az számít, ha a jármű
    # legalább MISSED_POLLS lekérdezésből hiányzott, amelyek ténylegesen lefutottak
    start = gaps["end"] - pd.to_timedelta(gaps["minutes"], unit="min")
    missed = (np.searchsorted(polls, gaps["end"].to_numpy(), "left")
              - np.searchsorted(polls, start.to_numpy(), "right"))
    gaps = gaps[missed >= MISSED_POLLS]

    # 3) kiesések járatonként, a járat adatmennyiségéhez viszonyítva
    route_pings = df.groupby("route_id", observed=True).size().rename("pings")
    routes = (
        gaps.groupby("route_id")
        .agg(dropouts=("minutes", "size"), vehicles=("vehicle", "nunique"),
             silent_minutes=("minutes", "sum"))
        .join(route_pings, how="inner")
        .reset_index()
    )

    # 4) késleltetés eloszlása és óránkénti adatmennyiség
    lag = df["lag_s"].dropna()
    lag_hist = (lag.clip(0, 60) // 2 * 2).value_counts().sort_index()
    lag_hist = lag_hist.rename_axis("lag_s").reset_index(name="pings")
    per_hour = df["time"].dt.floor("h").value_counts().sort_index()
    per_hour = per_hour.reindex(pd.date_range(per_hour.index[0], per_hour.index[-1], freq="h"),
                                fill_value=0).rename_axis("hour").reset_index(name="pings")

    summary = pd.DataFrame([{
        "rows": len(df),
        "vehicles": df["vehicle"].nunique(),
        "lag_median": float(lag.median()),
        "lag_p95": float(lag.quantile(0.95)),
        "interval_median": float(interval[interval < GAP_MIN].median()),
        "first": df["time"].iloc[0],
        "last": df["time"].iloc[-1],
        # a használt küszöbök, hogy a UI ugyanazokat írja ki
        "poll_gap": POLL_GAP, "gap_min": GAP_MIN, "gap_max": GAP_MAX, "missed_polls": MISSED_POLLS,
    }])

    for name, table in [("summary", summary), ("outages", outages), ("gaps", gaps),
                        ("route_dropouts", routes), ("lag_hist", lag_hist), ("per_hour", per_hour)]:
        table.to_csv(OUT_DIR / f"{name}.csv", index=False)

    s = summary.iloc[0]
    print(f"Késleltetés medián {s.lag_median:.0f} mp, 95% {s.lag_p95:.0f} mp, "
          f"jelentési időköz medián {s.interval_median:.0f} mp")
    print(f"Gyűjtőkiesések: {len(outages)}, járműkiesések: {len(gaps):,}")
    print(f"Eredmények mentve a '{OUT_DIR}/' mappába.")


if __name__ == "__main__":
    main()
