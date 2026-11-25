"""
Budapest network map - Streamlit UI.

Futtatás (a project mappából):
    streamlit run ui/app.py
"""

import time
from datetime import datetime, timedelta
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

DATA_FILE = Path(__file__).resolve().parent.parent / "vehicle_updates_delays_filtered.csv"
LOCAL_TZ = "Europe/Budapest"
WINDOW = timedelta(minutes=5)        # one map frame = 5 minutes of data
MAX_ABS_DELAY = 3600                 # delays beyond +-1 h are treated as bad readings
GRID_DEG = 0.01                      # density grid cell size (~1 km)
DENSITY_CAP = 6                      # vehicle count at which a cell is fully red
DELAY_CAP = 600                      # delay (s) at which a vehicle is fully red
MAP_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"

st.set_page_config(page_title="Budapest network map", layout="wide")


@st.cache_data(show_spinner="Loading data...")
def load_data() -> pd.DataFrame:
    """Beolvassa a CSV-t, helyi időre vált és kiszűri a hibás késésértékeket."""
    df = pd.read_csv(
        DATA_FILE,
        usecols=["timestamp_utc", "vehicle_id", "trip_id", "route_id",
                 "lat", "lon", "speed", "delay_seconds"],
        dtype={"route_id": str},
    )
    df = df[df["delay_seconds"].abs() <= MAX_ABS_DELAY]
    df["time"] = (
        pd.to_datetime(df["timestamp_utc"], utc=True)
        .dt.tz_convert(LOCAL_TZ)
        .dt.tz_localize(None)
    )
    df["vehicle"] = df["vehicle_id"].str.replace("VehiclePosition-", "", regex=False)
    df["delay_min"] = (df["delay_seconds"] / 60).round(1)
    return df.drop(columns=["timestamp_utc", "vehicle_id"]).sort_values("time").reset_index(drop=True)


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
first, last = df["time"].min(), df["time"].max()

# ---- session state: current frame start + play flag ----
if "day" not in st.session_state:
    # default to a weekday morning rush hour if available
    st.session_state.day = first.date() + timedelta(days=1)
    st.session_state.tod = dtime(8, 0)
    st.session_state.playing = False

if st.session_state.playing:
    nxt = datetime.combine(st.session_state.day, st.session_state.tod) + WINDOW
    if nxt > last:
        nxt = first.floor("5min").to_pydatetime()
    st.session_state.day, st.session_state.tod = nxt.date(), nxt.time()

# ---- sidebar controls ----
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Colour circles by", ["Average delay", "Vehicle count"])
    routes = st.multiselect("Routes (empty = all)", sorted(df["route_id"].unique()))
    speed = st.slider("Playback speed (s per frame)", 0.2, 2.0, 0.7, 0.1)
    st.caption(f"Data: {first:%Y-%m-%d %H:%M} – {last:%Y-%m-%d %H:%M} (Budapest time)")

st.title("Budapest public transport – network state")

c1, c2, c3 = st.columns([2, 3, 1])
c1.date_input("Day", min_value=first.date(), max_value=last.date(), key="day")
c2.slider("Time of day", min_value=dtime(0, 0), max_value=dtime(23, 55),
          step=WINDOW, format="HH:mm", key="tod")
c3.write("")
c3.button(
    "⏸ Pause" if st.session_state.playing else "▶ Play",
    on_click=lambda: st.session_state.update(playing=not st.session_state.playing),
    use_container_width=True,
)

start = datetime.combine(st.session_state.day, st.session_state.tod)
data = df[df["route_id"].isin(routes)] if routes else df
vehicles = frame(data, start)

# ---- KPIs ----
k1, k2, k3, k4 = st.columns(4)
k1.metric("Window", f"{start:%H:%M}–{start + WINDOW:%H:%M}")
k2.metric("Active vehicles", len(vehicles))
if len(vehicles):
    k3.metric("Average delay", f"{vehicles['delay_seconds'].mean() / 60:.1f} min")
    k4.metric("Delayed > 3 min", f"{(vehicles['delay_seconds'] > 180).mean():.0%}")

# ---- map ----
if vehicles.empty:
    st.info("No vehicles reported in this 5-minute window.")
else:
    if mode == "Average delay":
        layer = delay_layer(vehicles)
        tooltip = {"text": "Vehicle {vehicle}\nRoute {route_id}\nDelay {delay_min} min"}
        legend = f"Green = on time · yellow ≈ {DELAY_CAP // 120} min late · red ≥ {DELAY_CAP // 60} min late"
    else:
        layer = density_layer(vehicles)
        tooltip = {"text": "{vehicles} vehicles\nAvg delay {delay_min} min"}
        legend = f"Bigger / redder = more vehicles in a ~1 km cell (red ≥ {DENSITY_CAP})"

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

# ---- advance animation ----
if st.session_state.playing:
    time.sleep(speed)
    st.rerun()
