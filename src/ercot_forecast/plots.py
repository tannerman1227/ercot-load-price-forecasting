"""Charts written to reports/figures for the notebook and the README."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ercot_forecast.config import CENTRAL, FIGURES, MODEL_LABELS
from ercot_forecast.features import date_mask

NAVY = "#0E3A5D"
TEAL = "#1B7F79"
AMBER = "#D97706"
RED = "#B42318"
SLATE = "#52606D"
BLUE = "#3B6D9A"
GOLD = "#C4A35A"

MODEL_COLORS = {
    "seasonal_naive_168": SLATE,
    "seasonal_naive_24": BLUE,
    "sarimax": TEAL,
    "ols_lagged": TEAL,
    "gbm_lagged_weather": NAVY,
    "gbm_realized_weather": AMBER,
    "gbm_realized_load_weather": AMBER,
}


def save_all(frame: pd.DataFrame, forecasts: pd.DataFrame, metrics: pd.DataFrame, directory: Path | None = None) -> list[Path]:
    directory = directory or FIGURES
    directory.mkdir(parents=True, exist_ok=True)
    apply_style()
    written = [
        _hourly_profile(frame, directory / "load_hourly_profile.png"),
        _load_temperature(frame, directory / "load_vs_temperature.png"),
        _forecast_overlay(frame, forecasts, directory / "load_forecast_overlay.png"),
        _metric_bars(metrics, "load_ercot", "Load forecast error", "MAE (MW)", directory / "load_metrics.png"),
        _residual_by_hour(forecasts, directory / "load_residual_by_hour.png"),
        _event_panel(frame, "2021-02-13", "2021-02-20", "Winter Storm Uri, Houston hub", directory / "uri_february_2021.png"),
        _peak_week(frame, directory / "peak_load_week.png"),
        _price_load_scatter(frame, directory / "price_vs_load.png"),
        _metric_bars(metrics, "dam_hb_houston", "Houston hub day-ahead price error", "MAE ($/MWh)", directory / "price_metrics.png"),
    ]
    return written


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D5DBE1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E6E8EB",
            "grid.linewidth": 0.8,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlecolor": NAVY,
            "axes.labelsize": 11,
            "axes.labelcolor": "#243140",
            "font.size": 11,
            "text.color": "#243140",
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
        }
    )


def _hourly_profile(frame: pd.DataFrame, path: Path) -> Path:
    local_month = frame["month"]
    groups = {
        "Winter (Dec-Feb)": frame.loc[local_month.isin([12, 1, 2])],
        "Shoulder": frame.loc[local_month.isin([3, 4, 5, 9, 10, 11])],
        "Summer (Jun-Aug)": frame.loc[local_month.isin([6, 7, 8])],
    }
    colors = {"Winter (Dec-Feb)": BLUE, "Shoulder": GOLD, "Summer (Jun-Aug)": RED}
    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    for label, sub in groups.items():
        profile = sub.groupby("hour")["load_ercot"].mean() / 1000.0
        ax.plot(profile.index, profile.values, label=label, color=colors[label])
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour beginning, Central time")
    ax.set_ylabel("Average load (GW)")
    ax.set_title("ERCOT load is a summer afternoon shape, with a smaller winter morning peak")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _load_temperature(frame: pd.DataFrame, path: Path) -> Path:
    sample = frame.dropna(subset=["temp_f_weighted", "load_ercot"])
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    months = sample["month"].to_numpy()
    colors = np.where(np.isin(months, [6, 7, 8]), RED, np.where(np.isin(months, [12, 1, 2]), BLUE, GOLD))
    ax.scatter(sample["temp_f_weighted"], sample["load_ercot"] / 1000.0, s=6, c=colors, alpha=0.25, linewidths=0)
    ax.set_xlabel("Load-weighted temperature (F)")
    ax.set_ylabel("ERCOT load (GW)")
    ax.set_title("Load bends up on both hot and cold hours")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _forecast_overlay(frame: pd.DataFrame, forecasts: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 7.2), sharey=False)
    panels = (
        ("summer_2025", "Summer 2025 test window, day-ahead"),
        ("winter_2024", "December 2024 test window, day-ahead"),
    )
    models = ("seasonal_naive_24", "sarimax", "gbm_lagged_weather")
    for ax, (fold, title) in zip(axes, panels):
        block = forecasts.loc[(forecasts["fold"] == fold) & (forecasts["target"] == "load_ercot")]
        if block.empty:
            ax.set_title(f"{title} (no forecast rows)")
            continue
        actual = block.drop_duplicates("interval_start").sort_values("interval_start")
        local = pd.to_datetime(actual["interval_start"], utc=True).dt.tz_convert(CENTRAL)
        ax.plot(local, actual["y_true"] / 1000.0, color="#111827", label="Actual", linewidth=2.0)
        for model in models:
            pred = block.loc[block["model"] == model].sort_values("interval_start")
            pred_local = pd.to_datetime(pred["interval_start"], utc=True).dt.tz_convert(CENTRAL)
            ax.plot(pred_local, pred["y_pred"] / 1000.0, color=MODEL_COLORS[model], label=MODEL_LABELS[model], alpha=0.9)
        ax.set_title(title)
        ax.set_ylabel("Load (GW)")
        ax.legend(loc="upper left", ncol=2)
    axes[-1].set_xlabel("Central time")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _metric_bars(metrics: pd.DataFrame, target: str, title: str, ylabel: str, path: Path) -> Path:
    order = [
        "seasonal_naive_168",
        "seasonal_naive_24",
        "sarimax",
        "ols_lagged",
        "gbm_lagged_weather",
        "gbm_realized_weather",
        "gbm_realized_load_weather",
    ]
    sub = metrics.loc[metrics["target"] == target].copy()
    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    if sub.empty:
        ax.set_title(title)
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return path
    sub["model"] = pd.Categorical(sub["model"], [model for model in order if model in set(sub["model"])], ordered=True)
    sub = sub.sort_values("model")
    sub["label"] = sub["model"].astype(str).map(MODEL_LABELS).fillna(sub["model"].astype(str))
    colors = [MODEL_COLORS.get(model, NAVY) for model in sub["model"]]
    positions = np.arange(len(sub))
    ax.bar(positions, sub["mae"], color=colors, width=0.72)
    ax.set_xticks(positions)
    ax.set_xticklabels(sub["label"], rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _residual_by_hour(forecasts: pd.DataFrame, path: Path) -> Path:
    block = forecasts.loc[
        (forecasts["target"] == "load_ercot") & (forecasts["model"] == "gbm_lagged_weather")
    ].copy()
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    if block.empty:
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return path
    local = pd.to_datetime(block["interval_start"], utc=True).dt.tz_convert(CENTRAL)
    block["hour"] = local.dt.hour
    block["residual_mw"] = block["y_true"] - block["y_pred"]
    means = block.groupby("hour")["residual_mw"].mean()
    ax.axhline(0, color=SLATE, linewidth=1)
    ax.bar(means.index, means.values, color=NAVY, width=0.8)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour beginning, Central time")
    ax.set_ylabel("Actual minus forecast (MW)")
    ax.set_title("Gradient-boosting load residual by hour, pooled test folds")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _event_panel(frame: pd.DataFrame, start: str, end: str, title: str, path: Path) -> Path:
    sub = frame.loc[date_mask(frame, start, end)].copy()
    local = sub["interval_start"].dt.tz_convert(CENTRAL)
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 6.6), sharex=True)
    axes[0].plot(local, sub["load_ercot"] / 1000.0, color=NAVY)
    axes[0].set_ylabel("Load (GW)")
    axes[0].set_title(title)
    axes[1].plot(local, sub["dam_hb_houston"], color=TEAL, label="DAM HB_HOUSTON")
    axes[1].plot(local, sub["rtm_hb_houston_mean"], color=AMBER, label="RTM HB_HOUSTON hourly mean")
    axes[1].plot(local, sub["dam_hb_hubavg"], color=SLATE, label="DAM HB_HUBAVG", alpha=0.8)
    axes[1].set_ylabel("$/MWh")
    axes[1].set_xlabel("Central time")
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _peak_week(frame: pd.DataFrame, path: Path) -> Path:
    values = frame["load_ercot"].to_numpy(dtype=float)
    window = 168
    means = pd.Series(values).rolling(window).mean().to_numpy()
    end = int(np.nanargmax(means))
    start = end - window + 1
    block = frame.iloc[start : end + 1].copy()
    local = block["interval_start"].dt.tz_convert(CENTRAL)
    label_start = f"{local.iloc[0].strftime('%b')} {local.iloc[0].day}"
    label_end = f"{local.iloc[-1].strftime('%b')} {local.iloc[-1].day}, {local.iloc[-1].year}"
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 6.6), sharex=True)
    axes[0].plot(local, block["load_ercot"] / 1000.0, color=NAVY)
    axes[0].set_ylabel("Load (GW)")
    axes[0].set_title(f"Highest-load week in the sample ({label_start}–{label_end})")
    axes[1].plot(local, block["dam_hb_houston"], color=TEAL, label="DAM HB_HOUSTON")
    axes[1].plot(local, block["rtm_hb_houston_mean"], color=AMBER, label="RTM HB_HOUSTON hourly mean")
    axes[1].plot(local, block["dam_hb_hubavg"], color=SLATE, label="DAM HB_HUBAVG", alpha=0.8)
    axes[1].set_ylabel("$/MWh")
    axes[1].set_xlabel("Central time")
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _price_load_scatter(frame: pd.DataFrame, path: Path) -> Path:
    sub = frame.dropna(subset=["load_ercot", "dam_hb_houston", "temp_f_weighted"])
    uri = date_mask(sub, "2021-02-13", "2021-02-20")
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    ordinary = sub.loc[~uri]
    ax.scatter(
        ordinary["load_ercot"] / 1000.0,
        ordinary["dam_hb_houston"],
        s=8,
        c=ordinary["temp_f_weighted"],
        cmap="coolwarm",
        alpha=0.35,
        linewidths=0,
    )
    storm = sub.loc[uri]
    ax.scatter(
        storm["load_ercot"] / 1000.0,
        storm["dam_hb_houston"],
        s=18,
        c=RED,
        alpha=0.8,
        linewidths=0,
        label="Feb 13–19, 2021",
    )
    ax.set_xlabel("ERCOT load (GW)")
    ax.set_ylabel("DAM HB_HOUSTON ($/MWh)")
    ax.set_title("Day-ahead Houston price against load. Uri hours are marked.")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
