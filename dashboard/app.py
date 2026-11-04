"""Interactive view of the ERCOT load forecasts and Houston hub prices."""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ercot_forecast.config import (  # noqa: E402
    CENTRAL,
    FORECASTS_PATH,
    HOURLY_PATH,
    MODEL_LABELS,
    SUMMARY_PATH,
)

NAVY = "#0E3A5D"
TEAL = "#1B7F79"
AMBER = "#D97706"
SLATE = "#52606D"
BLUE = "#3B6D9A"
RED = "#B42318"
INK = "#111827"

MODEL_COLORS = {
    "seasonal_naive_168": SLATE,
    "seasonal_naive_24": BLUE,
    "sarimax": TEAL,
    "ols_lagged": TEAL,
    "gbm_lagged_weather": NAVY,
    "gbm_realized_weather": AMBER,
    "gbm_realized_load_weather": AMBER,
}

DAY_AHEAD = {
    "load_ercot": ["seasonal_naive_168", "seasonal_naive_24", "sarimax", "gbm_lagged_weather"],
    "dam_hb_houston": ["seasonal_naive_168", "seasonal_naive_24", "ols_lagged", "gbm_lagged_weather"],
}


def show_markdown(text: str) -> None:
    """Streamlit treats `$` as a LaTeX delimiter. Prices are currency, not math."""
    st.markdown(text.replace("$", r"\$"))


def main() -> None:
    st.set_page_config(page_title="ERCOT load and price forecast", layout="wide")
    st.title("ERCOT load and Houston hub prices")
    st.caption("Tanner Allred · Houston · day-ahead study on public ERCOT and Open-Meteo data")

    if not HOURLY_PATH.exists() or not SUMMARY_PATH.exists() or not FORECASTS_PATH.exists():
        st.error(
            "The cached dataset or the scored forecasts are missing. From the repo root, run "
            "`python scripts/download_data.py` and then `python scripts/run_analysis.py`."
        )
        return

    summary = json.loads(SUMMARY_PATH.read_text())
    forecasts = _forecasts()
    hourly = _hourly()

    forecasts_tab, scorecard_tab, prices_tab, about_tab = st.tabs(
        ["Load and price forecasts", "Scorecard", "Prices and spikes", "About"]
    )
    with forecasts_tab:
        _forecast_tab(forecasts, summary)
    with scorecard_tab:
        _scorecard_tab(summary)
    with prices_tab:
        _price_tab(hourly, summary)
    with about_tab:
        _about_tab(summary)


def _forecast_tab(forecasts: pd.DataFrame, summary: dict) -> None:
    target = st.selectbox(
        "Series",
        options=["load_ercot", "dam_hb_houston"],
        format_func=lambda key: "ERCOT load" if key == "load_ercot" else "DAM HB_HOUSTON price",
    )
    folds = list(dict.fromkeys(forecasts["fold"].tolist()))
    fold = st.selectbox("Test window", options=folds, index=min(3, len(folds) - 1))
    default_models = DAY_AHEAD[target]
    available = [model for model in default_models if model in set(forecasts["model"])]
    chosen = st.multiselect(
        "Models",
        options=available,
        default=available,
        format_func=lambda key: MODEL_LABELS.get(key, key),
    )
    block = forecasts.loc[(forecasts["target"] == target) & (forecasts["fold"] == fold)].copy()
    if block.empty or not chosen:
        st.info("No forecast rows for that selection.")
        return
    actual = block.drop_duplicates("interval_start").sort_values("interval_start")
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=_central(actual["interval_start"]),
            y=_display_units(actual["y_true"], target),
            name="Actual",
            line={"color": INK, "width": 2.4},
        )
    )
    for model in chosen:
        pred = block.loc[block["model"] == model].sort_values("interval_start")
        figure.add_trace(
            go.Scatter(
                x=_central(pred["interval_start"]),
                y=_display_units(pred["y_pred"], target),
                name=MODEL_LABELS.get(model, model),
                line={"color": MODEL_COLORS.get(model, NAVY), "width": 1.6},
            )
        )
    unit = "GW" if target == "load_ercot" else "$/MWh"
    figure.update_layout(
        height=480,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
        yaxis_title=unit,
        legend={"orientation": "h", "y": 1.12},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    figure.update_xaxes(showgrid=True, gridcolor="#E6E8EB")
    figure.update_yaxes(showgrid=True, gridcolor="#E6E8EB")
    st.plotly_chart(figure, width="stretch")
    headline = summary["headline"]
    if target == "load_ercot":
        left, right = headline["load_gbm_lagged"], headline["load_naive_24"]
        show_markdown(
            f"Across the five test windows the day-ahead gradient boosting model has MAE "
            f"**{left['mae']:,.0f} MW** (MAPE **{left['mape']:.2f}%**) against "
            f"**{right['mae']:,.0f} MW** for the 24-hour seasonal naive. "
            f"Skill versus that naive forecast is **{left['skill_vs_naive_24']:.1%}**."
        )
    else:
        left, right = headline["price_gbm_lagged"], headline["price_naive_24"]
        show_markdown(
            f"Across the five test windows the day-ahead price model has MAE "
            f"**${left['mae']:.2f}/MWh** against **${right['mae']:.2f}/MWh** for the 24-hour naive. "
            "These windows are ordinary 2024–2025 weeks. They are not Winter Storm Uri, "
            "and a model with only load and weather lags will not price a scarcity event."
        )


def _scorecard_tab(summary: dict) -> None:
    frame = pd.DataFrame(summary["metrics_pooled"])
    frame["model_label"] = frame["model"].map(MODEL_LABELS).fillna(frame["model"])
    view = frame.rename(
        columns={
            "model_label": "Model",
            "target": "Target",
            "info_set": "Information set",
            "mae": "MAE",
            "rmse": "RMSE",
            "mape": "MAPE %",
            "skill_vs_naive_24": "Skill vs 24h naive",
            "n": "Hours",
        }
    )
    columns = ["Target", "Model", "Information set", "MAE", "RMSE", "MAPE %", "Skill vs 24h naive", "Hours"]
    st.dataframe(
        view[columns].style.format(
            {"MAE": "{:,.2f}", "RMSE": "{:,.2f}", "MAPE %": "{:.2f}", "Skill vs 24h naive": "{:.1%}"}
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Skill is 1 minus the model's MAE divided by the 24-hour naive MAE. "
        "MAPE ignores hours with absolute actual value below 1, which matters for prices near zero. "
        "The realized-weather and realized-load rows are diagnostics: they see information a day-ahead desk would not have unless a weather or load forecast supplied it."
    )
    by_fold = pd.DataFrame(summary["metrics_by_fold"])
    load = by_fold.loc[
        (by_fold["target"] == "load_ercot") & (by_fold["model"].isin(DAY_AHEAD["load_ercot"]))
    ].copy()
    if load.empty:
        return
    figure = go.Figure()
    for model in DAY_AHEAD["load_ercot"]:
        sub = load.loc[load["model"] == model]
        figure.add_trace(
            go.Bar(
                name=MODEL_LABELS.get(model, model),
                x=sub["fold"],
                y=sub["mae"],
                marker_color=MODEL_COLORS.get(model, NAVY),
            )
        )
    figure.update_layout(
        barmode="group",
        height=420,
        yaxis_title="Load MAE (MW)",
        title="Day-ahead load error by test window",
        legend={"orientation": "h", "y": 1.14},
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin={"t": 60},
    )
    st.plotly_chart(figure, width="stretch")


def _price_tab(hourly: pd.DataFrame, summary: dict) -> None:
    prices = summary["prices"]
    uri = prices["uri"]
    peak = prices["peak_load_hour"]
    peak_price = prices["peak_rtm_hour"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Uri RTM peak", f"${uri['rtm_hb_houston_max']:,.0f}/MWh")
    c2.metric("Uri load peak", f"{uri['load_max_mw'] / 1000:.1f} GW")
    c3.metric("Sample load peak", f"{peak['load_mw'] / 1000:.1f} GW")
    c4.metric("Hours RTM ≥ $1,000", f"{prices['rtm_hours_ge_1000']:,}")
    show_markdown(
        f"During February 13–19, 2021, Houston real-time prices reached "
        f"**${uri['rtm_hb_houston_max']:,.0f}/MWh** at { _short(uri['rtm_max_at_ct']) }, "
        f"while ERCOT load in that window peaked at **{uri['load_max_mw'] / 1000:.1f} GW**. "
        f"The highest load hour in the sample was **{_short(peak['at_ct'])}** at "
        f"**{peak['load_mw'] / 1000:.1f} GW**, with a Houston day-ahead price of "
        f"**${peak['dam_hb_houston']:,.2f}/MWh** and a real-time hourly max of "
        f"**${peak['rtm_hb_houston_max']:,.2f}/MWh**. "
        f"The single highest real-time print was **${peak_price['rtm_hb_houston_max']:,.2f}/MWh** "
        f"at {_short(peak_price['at_ct'])}, when load was **{peak_price['load_mw'] / 1000:.1f} GW**."
    )
    preset = st.selectbox(
        "Window",
        options=["Winter Storm Uri", "Highest-load week", "Highest real-time price week", "Custom"],
    )
    if preset == "Winter Storm Uri":
        start, end = pd.Timestamp(uri["start"], tz=CENTRAL), pd.Timestamp(uri["end"], tz=CENTRAL)
    elif preset == "Highest-load week":
        start, end = _window_bounds(prices["peak_load_week"])
    elif preset == "Highest real-time price week":
        start, end = _window_bounds(prices["peak_rtm_week"])
    else:
        bounds = hourly["interval_start"].dt.tz_convert(CENTRAL)
        picked = st.date_input(
            "Central-time dates",
            value=(bounds.min().date(), bounds.max().date()),
        )
        if not isinstance(picked, tuple) or len(picked) != 2:
            st.info("Select a start and an end date.")
            return
        start = pd.Timestamp(picked[0], tz=CENTRAL)
        end = pd.Timestamp(picked[1], tz=CENTRAL) + timedelta(days=1)
    local = hourly["interval_start"].dt.tz_convert(CENTRAL)
    view = hourly.loc[(local >= start) & (local < end)].copy()
    if view.empty:
        st.warning("That window has no hours in the cache.")
        return
    clock = view["interval_start"].dt.tz_convert(CENTRAL)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=clock, y=view["load_ercot"] / 1000.0, name="Load (GW)", line={"color": NAVY}, yaxis="y1")
    )
    figure.add_trace(
        go.Scatter(
            x=clock,
            y=view["dam_hb_houston"],
            name="DAM HB_HOUSTON",
            line={"color": TEAL},
            yaxis="y2",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=clock,
            y=view["rtm_hb_houston_mean"],
            name="RTM HB_HOUSTON mean",
            line={"color": AMBER},
            yaxis="y2",
        )
    )
    figure.update_layout(
        height=480,
        yaxis={"title": "Load (GW)", "side": "left"},
        yaxis2={"title": "$/MWh", "overlaying": "y", "side": "right"},
        legend={"orientation": "h", "y": 1.12},
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin={"t": 40, "r": 60},
    )
    st.plotly_chart(figure, width="stretch")
    threshold = st.slider("Real-time spike threshold ($/MWh)", min_value=50, max_value=2000, value=100, step=50)
    spiked = int((view["rtm_hb_houston_max"] >= threshold).sum())
    show_markdown(
        f"**{spiked:,}** of **{len(view):,}** hours in this window have a Houston real-time "
        f"15-minute print at or above **${threshold:,.0f}/MWh**."
    )


def _about_tab(summary: dict) -> None:
    weights = summary["zone_weights"]
    weight_frame = pd.DataFrame(
        {"Zone": list(weights.keys()), "Share of average load": list(weights.values())}
    ).sort_values("Share of average load", ascending=False)
    show_markdown(
        """
**About me.** I am Tanner Allred, a Houston-based data analyst. I have an M.S. in Data Analytics
and a B.S. in Economics, and I have spent the last few years on annuity and pension-risk-transfer
analytics: reconciling large policy datasets, building cash-flow forecasts, and replacing manual
reports with Python. I am looking for energy trading, scheduling, and analytics roles.
Other work is on [github.com/tannerman1227](https://github.com/tannerman1227).

**This study.** The cache is ERCOT native load by weather zone, day-ahead and real-time
settlement point prices for the Houston hub and the hub average, and Open-Meteo temperature
and wind at a representative city in each weather zone. No API key is required. Forecasts are
day-ahead: features are lagged at least 24 hours, except the rows marked as realized weather
or realized load, which are there to show what a perfect weather or load forecast would add.
        """
    )
    show_markdown(
        f"Sample: **{summary['rows']:,}** hours from **{summary['start_ct']}** to **{summary['end_ct']}**."
    )
    figure = go.Figure(
        go.Bar(
            x=weight_frame["Zone"],
            y=weight_frame["Share of average load"],
            marker_color=NAVY,
        )
    )
    figure.update_layout(
        height=360,
        yaxis_tickformat=".0%",
        yaxis_title="Share of average ERCOT load",
        title="Weather-zone weights used for the temperature index",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Weights are each zone's share of average load over the cached sample. "
        "They are constants, not hourly features, so they do not leak the forecast target."
    )


@st.cache_data(show_spinner=False)
def _forecasts() -> pd.DataFrame:
    frame = pd.read_parquet(FORECASTS_PATH)
    frame["interval_start"] = pd.to_datetime(frame["interval_start"], utc=True)
    return frame


@st.cache_data(show_spinner=False)
def _hourly() -> pd.DataFrame:
    frame = pd.read_parquet(HOURLY_PATH)
    if frame["interval_start"].dt.tz is None:
        frame["interval_start"] = frame["interval_start"].dt.tz_localize("UTC")
    return frame


def _central(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).dt.tz_convert(CENTRAL)


def _display_units(series: pd.Series, target: str) -> pd.Series:
    if target == "load_ercot":
        return series / 1000.0
    return series


def _window_bounds(window: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(window["start_ct"])
    end = pd.Timestamp(window["end_ct"]) + timedelta(hours=1)
    return start, end


def _short(stamp: str) -> str:
    parsed = pd.Timestamp(stamp)
    return f"{parsed.strftime('%b')} {parsed.day}, {parsed.year} {parsed.strftime('%H:%M')} CT"


main()
