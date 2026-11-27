"""
Site Map & Diversion Simulator (Upload CSV Only - Single-Screen Streamlit App)

    pip install streamlit pandas numpy matplotlib pydeck
    streamlit run site_map_diversion.py
"""
import io
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="Site Map & Diversion Simulator", page_icon="🗺️", layout="wide")
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 0.65rem;
            padding-bottom: 0.35rem;
            padding-left: 1.2rem;
            padding-right: 1.2rem;
        }
        h2 {
            font-size: 1.3rem !important;
            margin: 0 0 0.25rem 0 !important;
            padding: 0 !important;
        }
        div[data-testid="stAlert"] {
            padding: 0.25rem 0.6rem !important;
            margin: 0.25rem 0 !important;
            font-size: 0.8rem !important;
        }
        div[data-testid="stMetricValue"] > div {
            font-size: 1.15rem !important;
            line-height: 1.2 !important;
            white-space: nowrap !important;
        }
        div[data-testid="stMetricLabel"] > div {
            font-size: 0.75rem !important;
            white-space: nowrap !important;
        }
        div[data-testid="stMetricDelta"] > div {
            font-size: 0.72rem !important;
        }
        .stSlider > div, .stSelectbox > div, .stRadio > div {
            margin-bottom: 0.15rem !important;
        }
        button[kind="secondary"] {
            padding: 0.2rem 0.6rem !important;
            font-size: 0.78rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

COLORS = [
    [255, 255, 178], [254, 217, 118], [254, 178, 76],
    [253, 141, 60], [240, 59, 32], [189, 0, 38],
]


@st.cache_data
def load_map_csv(b: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(b), low_memory=False)
    if {"source_name", "lat", "lon", "start_time", "emissions_quantity"} <= set(df.columns):
        if "subsector" in df:
            df = df[df["subsector"].astype(str).str.contains("solid-waste", na=False)]
        df = df.rename(columns={"source_name": "name", "start_time": "month", "emissions_quantity": "value"})
    if not {"name", "lat", "lon", "month", "value"} <= set(df.columns):
        raise ValueError("CSV needs columns: name, lat, lon, month, value")

    df = df[["name", "lat", "lon", "month", "value"]].copy()
    df["month"] = pd.to_datetime(df["month"]).dt.to_period("M").dt.to_timestamp()
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)

    valid_coords = (
        df[(df["lat"] != 0) & (df["lon"] != 0)]
        .groupby("name")[["lat", "lon"]]
        .first()
        .to_dict("index")
    )
    for idx, row in df[(df["lat"] == 0) | (df["lon"] == 0)].iterrows():
        if row["name"] in valid_coords:
            df.at[idx, "lat"] = valid_coords[row["name"]]["lat"]
            df.at[idx, "lon"] = valid_coords[row["name"]]["lon"]
    return df


def apply_site_diversion(
    df: pd.DataFrame,
    target_site: str,
    start_ts: pd.Timestamp,
    method: str,
    pct_share: float,
    tonnes_per_yr: float,
    gradual: bool,
    k: float = 0.06,
    cap_pct: float = 37.0,
    doc: float = 0.15,
    docf: float = 0.70,
    mcf: float = 1.0,
    f_ch4: float = 0.50,
    ox: float = 0.10,
) -> pd.DataFrame:
    out = df.copy()
    out["baseline_value"] = out["value"]
    out["avoided_ch4"] = 0.0

    if target_site == "None (no simulation)":
        return out

    mask = (out["name"] == target_site) & (out["month"] >= start_ts)
    if not mask.any():
        return out

    yrs = np.maximum((out.loc[mask, "month"] - start_ts).dt.days.values / 365.25, 0.0)
    ramp_factor = (1.0 - np.exp(-k * yrs)) if gradual else np.ones_like(yrs)

    if method == "Percentage reduction (%)":
        avoided = out.loc[mask, "baseline_value"].values * pct_share * ramp_factor
    else:
        ch4_per_tonne = doc * docf * mcf * f_ch4 * (16.0 / 12.0) * (1.0 - cap_pct / 100.0) * (1.0 - ox)
        monthly_potential_ch4 = (tonnes_per_yr * ch4_per_tonne) / 12.0
        avoided = monthly_potential_ch4 * ramp_factor

    avoided = np.minimum(avoided, out.loc[mask, "baseline_value"].values)
    out.loc[mask, "avoided_ch4"] = avoided
    out.loc[mask, "value"] = out.loc[mask, "baseline_value"] - avoided
    return out


st.title("Site Map & Diversion Simulator")

c_src, c_sim1, c_sim2 = st.columns([1.15, 1.45, 1.4])

with c_src:
    st.caption("**1. Upload CSV**")
    up_map = st.file_uploader(
        "CSV (name, lat, lon, month, value)",
        type=["csv"],
        key="map_uploader",
        label_visibility="collapsed",
    )
    raw_site_df = load_map_csv(up_map.getvalue()) if up_map is not None else None

if raw_site_df is None:
    st.info("Upload a CSV with columns `name, lat, lon, month, value` (or a Climate TRACE file) to begin.")
else:
    all_sites = sorted(raw_site_df["name"].unique())
    all_months = sorted(raw_site_df["month"].unique())
    month_labels = [pd.Timestamp(m).strftime("%Y-%m") for m in all_months]

    with c_sim1:
        st.caption("**2. Diversion Target**")
        default_idx = (
            all_sites.index("East Calgary Landfill") + 1
            if "East Calgary Landfill" in all_sites
            else (1 if len(all_sites) >= 1 else 0)
        )
        target_site = st.selectbox(
            "Target landfill",
            ["None (no simulation)"] + all_sites,
            index=default_idx,
            label_visibility="collapsed",
        )
        default_start_idx = max(0, len(month_labels) - 36)
        sim_start_label = st.select_slider(
            "Start month",
            month_labels,
            value=month_labels[default_start_idx],
        )
        gradual = st.radio(
            "Curve",
            ["Gradual (k=0.06)", "Immediate"],
            horizontal=True,
            label_visibility="collapsed",
        ).startswith("Gradual")

    with c_sim2:
        st.caption("**3. Diversion Input**")
        sim_method = st.radio(
            "Input mode",
            ["% reduction", "Tonnes diverted / yr"],
            horizontal=True,
            label_visibility="collapsed",
        )
        if sim_method == "% reduction":
            pct_share = st.slider("Diverted share (%)", 0, 80, 25) / 100.0
            tonnes_diverted = 0.0
            cap_pct = 37.0
        else:
            pct_share = 0.0
            tonnes_diverted = st.number_input(
                "Organic waste diverted (t / yr)",
                min_value=0, max_value=2_000_000, value=50_000, step=5_000,
            )
            cap_pct = st.slider("Gas capture (%)", 0, 95, 37)

    sim_df = apply_site_diversion(
        df=raw_site_df,
        target_site=target_site,
        start_ts=pd.Timestamp(sim_start_label + "-01"),
        method="Percentage reduction (%)" if sim_method == "% reduction" else "Tonnes",
        pct_share=pct_share,
        tonnes_per_yr=float(tonnes_diverted),
        gradual=gradual,
        cap_pct=float(cap_pct),
    )

    st.warning(
        "Uploaded data: baseline values reflect the uploaded dataset; "
        "green/dashed lines show simulated diversion impact."
    )

    left, right = st.columns([3.1, 1.9])
    with right:
        f_col1, f_col2 = st.columns([1.2, 1.0])
        with f_col1:
            view_choice = st.selectbox("Focus", ["All landfills"] + all_sites, label_visibility="collapsed")
        with f_col2:
            play = st.button("▶ Play", use_container_width=True)

        cur_label = st.select_slider("Month", month_labels, value=month_labels[-1], label_visibility="collapsed")

    vmax = float(sim_df["baseline_value"].max()) * 1.05

    def build_deck(m_label: str):
        d = sim_df[sim_df["month"] == pd.Timestamp(m_label + "-01")].copy()
        if view_choice != "All landfills":
            d = d[d["name"] == view_choice]
        d["label"] = np.where(
            d["name"] == target_site, d["name"] + " [diversion target]", d["name"]
        )
        d["value_txt"] = (
            d["value"].round(1).astype(str)
            + " t CH4/mo (avoided: "
            + d["avoided_ch4"].round(1).astype(str)
            + " t)"
        )
        if view_choice == "All landfills":
            view = pdk.ViewState(latitude=53.5, longitude=-114.0, zoom=4.6)
        else:
            full = sim_df[sim_df["name"] == view_choice]
            view = pdk.ViewState(
                latitude=float(full["lat"].iloc[0]),
                longitude=float(full["lon"].iloc[0]),
                zoom=9,
            )
        heat = pdk.Layer(
            "HeatmapLayer",
            data=d,
            get_position="[lon, lat]",
            get_weight="value",
            radius_pixels=80,
            aggregation="SUM",
            color_domain=[0, vmax],
            threshold=0.03,
            opacity=0.85,
            color_range=COLORS,
        )
        dots = pdk.Layer(
            "ScatterplotLayer",
            data=d,
            get_position="[lon, lat]",
            get_radius=2500,
            radius_min_pixels=4,
            radius_max_pixels=9,
            get_fill_color=[30, 30, 30, 210],
            pickable=True,
        )
        return pdk.Deck(
            layers=[heat, dots],
            initial_view_state=view,
            map_style="light",
            tooltip={"text": "{label}\n{value_txt}"},
        )

    MAP_H = 300
    with left:
        map_ph = st.empty()
        cap_ph = st.empty()
        if play:
            for lab in month_labels[month_labels.index(cur_label):]:
                map_ph.pydeck_chart(build_deck(lab), height=MAP_H)
                cap_ph.caption(f"{lab} · scale: 0 to {vmax:,.0f} t CH4/mo")
                time.sleep(0.25)
        else:
            map_ph.pydeck_chart(build_deck(cur_label), height=MAP_H)
            cap_ph.caption(f"{cur_label} · scale: 0 to {vmax:,.0f} t CH4/mo")

    cur_ts = pd.Timestamp(cur_label + "-01")
    sel = sim_df if view_choice == "All landfills" else sim_df[sim_df["name"] == view_choice]
    agg = sel.groupby("month")[["baseline_value", "value", "avoided_ch4"]].sum()

    with right:
        if cur_ts not in agg.index:
            st.info("No reading for this month.")
        else:
            now_sim = agg.loc[cur_ts, "value"]
            now_base = agg.loc[cur_ts, "baseline_value"]
            now_av = agg.loc[cur_ts, "avoided_ch4"]
            pct_drop = (now_av / now_base * 100) if now_base > 0 else 0.0

            m = st.columns(3)
            m[0].metric("Simulated CH4", f"{now_sim:,.1f} t/mo")
            m[1].metric("Baseline CH4", f"{now_base:,.1f} t/mo")
            m[2].metric("Avoided CH4", f"-{now_av:,.1f} t", f"-{pct_drop:.1f}%")

        fig, ax = plt.subplots(figsize=(5.0, 1.55))
        if view_choice == "All landfills":
            for n_, g in sim_df.groupby("name"):
                if n_ == target_site:
                    ax.plot(g["month"], g["baseline_value"], color="tab:red", ls="--", lw=1.1, label="Baseline")
                    ax.plot(g["month"], g["value"], color="tab:green", lw=1.8, label="Simulated")
                else:
                    ax.plot(g["month"], g["value"], color="lightgrey", lw=0.6)
            if target_site != "None (no simulation)":
                ax.legend(fontsize=6.5, loc="upper left")
        else:
            ax.plot(agg.index, agg["baseline_value"], color="tab:red", ls="--", lw=1.3, label="Baseline")
            ax.plot(agg.index, agg["value"], color="tab:green", lw=1.8, label="Simulated")
            ax.fill_between(agg.index, agg["value"], agg["baseline_value"], color="tab:green", alpha=0.15)
            ax.legend(fontsize=6.5)

        ax.axvline(cur_ts, color="black", ls=":")
        if target_site != "None (no simulation)":
            ax.axvline(pd.Timestamp(sim_start_label + "-01"), color="tab:blue", ls="--", lw=1)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=6.5)
        ax.set_ylabel("t CH4/mo", fontsize=7)
        fig.tight_layout(pad=0.3)
        st.pyplot(fig, use_container_width=True)

        cum_avoided = sel["avoided_ch4"].sum()
        st.caption(
            f"Start: {sim_start_label} · Cum. avoided: **{cum_avoided:,.1f} t CH4** "
            f"(~**{cum_avoided * 28:,.0f} t CO2e**)"
        )