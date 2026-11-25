"""
Budapest network map - Streamlit UI.

Futtatás (a project mappából):
    streamlit run ui/app.py
"""

import time
import zipfile
from datetime import datetime, timedelta
from datetime import time as dtime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

DATA_FILE = Path(__file__).resolve().parent.parent / "vehicle_updates_delays_filtered.csv"
HEALTH_DIR = Path(__file__).resolve().parent.parent / "data_health"   # calculate_health.py kimenete
GTFS_DIR = Path(__file__).resolve().parent.parent / "gtfs"   # bkk_gtfs_ÉÉÉÉHHNN.zip, lásd download_gtfs.py
ROUTE_TYPES = {"0": "villamos", "1": "metró", "3": "busz", "4": "hajó", "11": "trolibusz", "109": "HÉV"}
REPLACEMENT_PREFIXES = {"VP": "villamospótló busz", "MP": "metrópótló busz", "TP": "trolibusz-pótló busz"}   # GTFS route_id prefixes
LOCAL_TZ = "Europe/Budapest"
WINDOW = timedelta(minutes=5)        # one map frame = 5 minutes of data
MAX_ABS_DELAY = 3600                 # delays beyond +-1 h are treated as bad readings
GRID_DEG = 0.01                      # density grid cell size (~1 km)
DENSITY_CAP = 6                      # vehicle count at which a cell is fully red
DELAY_CAP = 600                      # delay (s) at which a vehicle is fully red
MAP_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
DAYS = ["Hétfő", "Kedd", "Szerda", "Csütörtök", "Péntek", "Szombat", "Vasárnap"]
MIN_CELL_PINGS = 500                 # heatmap cells with fewer pings are hidden (too noisy)
LATE_CUTOFF = 180                    # "late" = more than 3 minutes behind schedule
MIN_ROUTE_PINGS = 1000               # routes with fewer pings are left out of the dropout ranking

st.set_page_config(page_title="Budapesti hálózati térkép", layout="wide")


@st.cache_data(show_spinner="Adatok betöltése...")
def load_data() -> pd.DataFrame:
    """Beolvassa a CSV-t, helyi időre vált és kiszűri a hibás késésértékeket."""
    df = pd.read_csv(
        DATA_FILE,
        usecols=["timestamp_utc", "vehicle_id", "trip_id", "route_id",
                 "lat", "lon", "speed", "delay_seconds"],
        dtype={"route_id": str},
    )
    df = df[df["delay_seconds"].abs() <= MAX_ABS_DELAY]
    df["time"] = pd.to_datetime(df["timestamp_utc"]).dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    df["vehicle"] = df["vehicle_id"].str.replace("VehiclePosition-", "", regex=False)
    df["delay_min"] = (df["delay_seconds"] / 60).round(1)
    return (df.drop(columns=["timestamp_utc", "vehicle_id"])
              .sort_values("time").reset_index(drop=True))


@st.cache_data
def load_routes() -> pd.DataFrame:
    """Járatnevek a GTFS routes.txt-ből (az összes letöltött változatból, a legújabb az irányadó)."""
    frames = []
    for path in sorted(GTFS_DIR.glob("bkk_gtfs_*.zip")):
        with zipfile.ZipFile(path) as z:
            frames.append(pd.read_csv(z.open("routes.txt"), dtype=str))
    if not frames:
        return pd.DataFrame(columns=["name", "kind", "label", "order"])
    r = pd.concat(frames).drop_duplicates("route_id", keep="last").set_index("route_id")
    kind = r["route_type"].map(ROUTE_TYPES).fillna("egyéb")
    # replacement buses reuse the number of the line they replace (e.g. VP01 = bus "1" instead of tram 1)
    for prefix, name in REPLACEMENT_PREFIXES.items():
        kind = kind.mask(r.index.str.startswith(prefix), name)
    return pd.DataFrame({
        "name": r["route_short_name"],
        "kind": kind,
        "label": r["route_short_name"] + " · " + kind + " – " + r["route_desc"].fillna(""),
        "order": pd.to_numeric(r["route_sort_order"], errors="coerce").fillna(np.inf),
    })


def route_names(route_ids: pd.Series) -> pd.Series:
    """route_id -> rövid járatnév (pl. 1335 -> 133E); ismeretlen azonosítónál marad az azonosító."""
    return route_ids.map(ROUTES["name"]).fillna(route_ids)


@st.cache_data(show_spinner="Óra × nap rács készítése...")
def week_grid(routes: tuple[str, ...]) -> pd.DataFrame:
    """Átlagos késés és a késő pingek aránya óránként és a hét napjai szerint."""
    df = load_data()
    if routes:
        df = df[df["route_id"].isin(routes)]
    grid = (
        df.assign(day=df["time"].dt.dayofweek, hour=df["time"].dt.hour,
                  late=df["delay_seconds"] > LATE_CUTOFF)
        .groupby(["day", "hour"])
        .agg(avg_delay=("delay_seconds", "mean"), late_share=("late", "mean"),
             pings=("delay_seconds", "size"))
        .reset_index()
    )
    grid = grid[grid["pings"] >= MIN_CELL_PINGS]
    grid["day_name"] = grid["day"].map(dict(enumerate(DAYS)))
    grid["avg_delay_min"] = (grid["avg_delay"] / 60).round(2)
    grid["late_pct"] = (grid["late_share"] * 100).round(1)
    return grid


@st.cache_data(show_spinner="Adatminőség-eredmények betöltése...")
def health_stats() -> dict | None:
    """A calculate_health.py által előre kiszámolt adatminőség-táblák (None, ha még nem futott)."""
    if not (HEALTH_DIR / "summary.csv").exists():
        return None

    def read(name: str, dates: list[str] = ()) -> pd.DataFrame:
        return pd.read_csv(HEALTH_DIR / f"{name}.csv", parse_dates=list(dates),
                           dtype={"route_id": str, "vehicle": str})

    routes = read("route_dropouts")
    routes = routes[routes["pings"] >= MIN_ROUTE_PINGS].copy()
    routes["per_1000"] = (routes["dropouts"] / routes["pings"] * 1000).round(1)
    routes["silent_minutes"] = routes["silent_minutes"].round(0).astype(int)
    return {
        "summary": read("summary", ["first", "last"]).iloc[0],
        "outages": read("outages", ["from", "to"]),
        "gaps": read("gaps", ["end"]),
        "worst": routes.sort_values("per_1000", ascending=False).head(15).reset_index(drop=True),
        "lag_hist": read("lag_hist"),
        "per_hour": read("per_hour", ["hour"]),
    }


def thousands(n: int) -> str:
    """Ezres tagolás magyarosan: 1234567 -> '1 234 567'."""
    return f"{int(n):,}".replace(",", " ")


def frame(df: pd.DataFrame, start: datetime) -> pd.DataFrame:
    """Az adott 5 perces ablakban minden jármű utolsó ismert pozíciója."""
    lo, hi = df["time"].searchsorted([start, start + WINDOW])
    window = df.iloc[lo:hi]
    return window.groupby("vehicle", as_index=False).last()


def ramp(values: np.ndarray, cap: float) -> list[list[int]]:
    """0 -> zöld, cap/2 -> sárga, cap -> piros."""
    t = np.clip(values / cap, 0, 1)
    r = np.where(t < 0.5, 510 * t, 255)
    g = np.where(t < 0.5, 200, 200 * (1 - (t - 0.5) * 2))
    return np.stack([r, g, np.full_like(t, 40), np.full_like(t, 200)], axis=1).astype(int).tolist()


def delay_layer(vehicles: pd.DataFrame) -> pdk.Layer:
    """Egy kör járművenként, a színe a késéstől függ."""
    vehicles = vehicles.assign(color=ramp(vehicles["delay_seconds"].to_numpy(), DELAY_CAP))
    return pdk.Layer(
        "ScatterplotLayer",
        vehicles,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius=120,
        radius_min_pixels=4,
        pickable=True,
    )


def density_layer(vehicles: pd.DataFrame) -> pdk.Layer:
    """Rácsba gyűjti a járműveket; minél több egy cellában, annál nagyobb és pirosabb a kör."""
    cells = (
        vehicles.assign(
            gx=(vehicles["lon"] / GRID_DEG).round(),
            gy=(vehicles["lat"] / GRID_DEG).round(),
        )
        .groupby(["gx", "gy"])
        .agg(lat=("lat", "mean"), lon=("lon", "mean"),
             vehicles=("vehicle", "count"), delay_min=("delay_min", "mean"))
        .reset_index()
    )
    cells["delay_min"] = cells["delay_min"].round(1)
    cells["color"] = ramp(cells["vehicles"].to_numpy(dtype=float), DENSITY_CAP)
    cells["radius"] = 150 + 100 * np.sqrt(cells["vehicles"])
    return pdk.Layer(
        "ScatterplotLayer",
        cells,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius="radius",
        pickable=True,
    )


df = load_data()
ROUTES = load_routes()
first, last = df["time"].min(), df["time"].max()

# ---- session state: current frame start + play flag ----
if "day" not in st.session_state:
    # default to a weekday morning rush hour if available
    st.session_state.day = min(first.date() + timedelta(days=1), last.date())
    st.session_state.tod = dtime(8, 0)
    st.session_state.playing = False

if st.session_state.playing:
    nxt = datetime.combine(st.session_state.day, st.session_state.tod) + WINDOW
    if nxt > last:
        nxt = first.floor("5min").to_pydatetime()
    st.session_state.day, st.session_state.tod = nxt.date(), nxt.time()

# ---- sidebar controls ----
with st.sidebar:
    st.header("Beállítások")
    mode = st.radio("Körök színezése", ["Átlagos késés", "Járműszám"])
    route_ids = df["route_id"].unique()
    route_ids = sorted(route_ids, key=lambda r: (ROUTES["order"].get(r, np.inf), r))
    routes = st.multiselect(
        "Járatok (üres = mind)", route_ids,
        format_func=lambda r: ROUTES["label"].get(r, r),
        placeholder="Írd be a járat számát, pl. 133E",
    )
    speed = st.slider("Lejátszási sebesség (mp / képkocka)", 0.2, 2.0, 0.7, 0.1)
    st.caption(f"Adatok: {first:%Y.%m.%d %H:%M} – {last:%Y.%m.%d %H:%M} (budapesti idő)")

st.title("Budapesti közösségi közlekedés – hálózati állapot")
tab_map, tab_when, tab_health = st.tabs(["🗺️ Térkép", "🕒 Mikor a legrosszabb?", "📡 Adatminőség"])

# =====================================================================
# Tab 1 – map replay
# =====================================================================
with tab_map:
    c1, c2, c3 = st.columns([2, 3, 1])
    c1.date_input("Nap", min_value=first.date(), max_value=last.date(), key="day")
    c2.slider("Időpont", min_value=dtime(0, 0), max_value=dtime(23, 55),
              step=WINDOW, format="HH:mm", key="tod")
    c3.write("")
    c3.button(
        "⏸ Szünet" if st.session_state.playing else "▶ Lejátszás",
        on_click=lambda: st.session_state.update(playing=not st.session_state.playing),
        width="stretch",
    )

    start = datetime.combine(st.session_state.day, st.session_state.tod)
    data = df[df["route_id"].isin(routes)] if routes else df
    vehicles = frame(data, start)
    vehicles["route_name"] = route_names(vehicles["route_id"])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Időablak", f"{start:%H:%M}–{start + WINDOW:%H:%M}")
    k2.metric("Aktív járművek", len(vehicles))
    if len(vehicles):
        k3.metric("Átlagos késés", f"{vehicles['delay_seconds'].mean() / 60:.1f} perc")
        k4.metric("3 percnél többet késik", f"{(vehicles['delay_seconds'] > 180).mean():.0%}")

    if vehicles.empty:
        st.info("Ebben az 5 perces ablakban egy jármű sem jelentkezett.")
    else:
        if mode == "Átlagos késés":
            layer = delay_layer(vehicles)
            tooltip = {"text": "Jármű: {vehicle}\nJárat: {route_name}\nKésés: {delay_min} perc"}
            legend = f"Zöld = időben · sárga ≈ {DELAY_CAP // 120} perc késés · piros ≥ {DELAY_CAP // 60} perc késés"
        else:
            layer = density_layer(vehicles)
            tooltip = {"text": "{vehicles} jármű\nÁtl. késés: {delay_min} perc"}
            legend = f"Nagyobb / pirosabb = több jármű egy kb. 1 km-es cellában (piros ≥ {DENSITY_CAP})"

        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=pdk.ViewState(latitude=47.497, longitude=19.07, zoom=10.5),
                map_style=MAP_STYLE,
                tooltip=tooltip,
            ),
            height=600,
        )
        st.caption(legend)

# =====================================================================
# Tab 2 – hour of day × day of week
# =====================================================================
with tab_when:
    grid = week_grid(tuple(routes))
    metric = st.radio("Színezés", ["Átlagos késés (perc)", f"Késő pingek (> {LATE_CUTOFF // 60} perc, %)"],
                      horizontal=True)
    field = "avg_delay_min" if metric.startswith("Átlagos") else "late_pct"

    if grid.empty:
        st.info("A kiválasztott járatokhoz nincs elég adat.")
    else:
        heat = (
            alt.Chart(grid)
            .mark_rect(stroke="white", strokeWidth=1)
            .encode(
                x=alt.X("hour:O", title="Óra"),
                y=alt.Y("day_name:O", sort=DAYS, title=None),
                color=alt.Color(f"{field}:Q", title=metric, scale=alt.Scale(scheme="yelloworangered")),
                tooltip=[alt.Tooltip("day_name:N", title="Nap"), alt.Tooltip("hour:O", title="Óra"),
                         alt.Tooltip("avg_delay_min:Q", title="Átl. késés (perc)"),
                         alt.Tooltip("late_pct:Q", title="Késő pingek (%)"),
                         alt.Tooltip("pings:Q", title="Pingek", format=",")],
            )
            .properties(height=320)
        )
        st.altair_chart(heat, width="stretch")
        st.caption(f"Minden cella az adott óra összes pingjének átlaga a teljes időszakra "
                   f"({first:%Y.%m.%d} – {last:%Y.%m.%d}). A {MIN_CELL_PINGS}-nál kevesebb pinget tartalmazó "
                   f"cellák (főleg éjszaka) üresen maradnak. Az oldalsáv járatszűrője itt is érvényes.")

        st.subheader("Hétköznap vs. hétvége")
        by_hour = (
            grid.assign(kind=np.where(grid["day"] >= 5, "Hétvége", "Hétköznap"),
                        weighted=grid[field] * grid["pings"])
            .groupby(["kind", "hour"], as_index=False)[["weighted", "pings"]].sum()
        )
        by_hour[field] = (by_hour["weighted"] / by_hour["pings"]).round(2)
        line = (
            alt.Chart(by_hour)
            .mark_line(point=True)
            .encode(x=alt.X("hour:O", title="Óra"),
                    y=alt.Y(f"{field}:Q", title=metric),
                    color=alt.Color("kind:N", title=None),
                    tooltip=[alt.Tooltip("kind:N", title="Nap típusa"), alt.Tooltip("hour:O", title="Óra"),
                             alt.Tooltip(f"{field}:Q", title=metric)])
            .properties(height=280)
        )
        st.altair_chart(line, width="stretch")

# =====================================================================
# Tab 3 – IoT pipeline health
# =====================================================================
with tab_health:
    h = health_stats()
    if h is None:
        st.info("Az adatminőség-elemzés még nem készült el. Futtasd a project mappából: "
                "`python calculate_health.py` (a szűretlen vehicle_updates.csv-ből számol).")
    else:
        sm = h["summary"]
        st.markdown("Mennyire jól működik maga az adatgyűjtés (**jármű → BKK API → gyűjtő script → CSV**)? "
                    "A gyűjtő script percenként lekérdezi a járművek pozícióját, és minden sorba elmenti "
                    "a jármű saját időbélyegét és a lekérdezés idejét is. Az elemzés a szűretlen adatból "
                    f"készült: {thousands(sm['rows'])} sor, {thousands(sm['vehicles'])} jármű, "
                    f"{sm['first']:%Y.%m.%d %H:%M} – {sm['last']:%Y.%m.%d %H:%M}.")

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Késleltetés (medián)", f"{sm['lag_median']:.0f} mp", help="Lekérdezés ideje − a jármű időbélyege")
        k2.metric("Késleltetés (95. percentilis)", f"{sm['lag_p95']:.0f} mp")
        k3.metric("Jelentési időköz (medián)", f"{sm['interval_median']:.0f} mp")
        k4.metric("Járműkiesések", thousands(len(h['gaps'])),
                  help=f"Járaton lévő jármű {sm['gap_min'] // 60}–{sm['gap_max'] // 60} percig nem jelentkezett, "
                       f"és legalább {sm['missed_polls']} lefutott lekérdezésből hiányzott")
        k5.metric("Gyűjtőkiesések", len(h["outages"]),
                  help=f"{sm['poll_gap']} mp-nél tovább egyetlen ping sem érkezett")

        left, right = st.columns(2)
        with left:
            st.subheader("Késleltetés: jármű jelentése → lekérdezés")
            st.altair_chart(
                alt.Chart(h["lag_hist"]).mark_bar()
                .encode(x=alt.X("lag_s:Q", bin=alt.Bin(step=2, extent=[0, 62]), title="Késleltetés (mp, utolsó oszlop = 60+)"),
                        y=alt.Y("sum(pings):Q", title="Pingek"),
                        tooltip=[alt.Tooltip("lag_s:Q", title="Késleltetés ettől (mp)"),
                                 alt.Tooltip("pings:Q", title="Pingek", format=",")])
                .properties(height=280),
                width="stretch",
            )
        with right:
            st.subheader("Óránként beérkezett pingek")
            st.altair_chart(
                alt.Chart(h["per_hour"]).mark_area(line=True, opacity=0.3)
                .encode(x=alt.X("hour:T", title=None), y=alt.Y("pings:Q", title="Pingek / óra"),
                        tooltip=[alt.Tooltip("hour:T", title="Óra", format="%Y.%m.%d %H:00"),
                                 alt.Tooltip("pings:Q", title="Pingek", format=",")])
                .properties(height=280),
                width="stretch",
            )
            st.caption("A nullára esések a gyűjtő kiesései, a napi hullámzás a menetrendből adódik.")

        st.subheader("Hol némulnak el a járművek?")
        gaps = h["gaps"]
        if gaps.empty:
            st.info("Nem találtunk kiesést.")
        else:
            gaps = gaps.assign(color=ramp(gaps["minutes"].to_numpy(), 30.0), when=gaps["end"].dt.strftime("%m-%d %H:%M"),
                               route_name=route_names(gaps["route_id"]))
            st.pydeck_chart(
                pdk.Deck(
                    layers=[pdk.Layer("ScatterplotLayer", gaps, get_position=["lon", "lat"],
                                      get_fill_color="color", get_radius=80, radius_min_pixels=2,
                                      pickable=True)],
                    initial_view_state=pdk.ViewState(latitude=47.497, longitude=19.07, zoom=10.5),
                    map_style=MAP_STYLE,
                    tooltip={"text": "Jármű: {vehicle} ({route_name} járat)\nNéma volt: {minutes} perc\nÚjra jelentkezett: {when}"},
                ),
                height=500,
            )
            st.caption("Minden pont a jármű utolsó ismert helye, mielőtt elnémult "
                       "(zöld = rövid, piros ≥ 30 perc). A sűrűsödések GPS- vagy mobilhálózati "
                       "holtterekre utalhatnak.")

        left, right = st.columns(2)
        with left:
            st.subheader("Járatok, ahol a járművek a legtöbbször elnémulnak")
            worst = h["worst"]
            table = pd.DataFrame({
                "Járat": route_names(worst["route_id"]),
                "Típus": worst["route_id"].map(ROUTES["kind"]).fillna(""),
                "Kiesés / 1000 ping": worst["per_1000"],
                "Kiesések": worst["dropouts"],
                "Érintett járművek": worst["vehicles"],
                "Néma percek": worst["silent_minutes"],
            })
            st.dataframe(table, hide_index=True, width="stretch")
            st.caption(f"Csak a legalább {thousands(MIN_ROUTE_PINGS)} pinget adó járatok szerepelnek.")
        with right:
            st.subheader("Gyűjtőkiesések")
            if h["outages"].empty:
                st.success("A gyűjtő script egyszer sem állt le.")
            else:
                st.dataframe(h["outages"].sort_values("minutes", ascending=False)
                             .rename(columns={"from": "Kezdete", "to": "Vége", "minutes": "Perc"}),
                             hide_index=True, width="stretch")

# ---- advance animation ----
if st.session_state.playing:
    time.sleep(speed)
    st.rerun()
