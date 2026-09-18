"""
AUGUR — feature engineering.

A single day's sensor snapshot is weak; a machine failing announces itself in the *trend* — vibration
creeping up, throughput sagging, jams climbing. So for each machine we add, per reading, short rolling
statistics and a slope over the last week. Everything is computed on PAST+CURRENT rows only (shift-safe
rolling), so no future information leaks into a feature — the only thing that knows the future is the label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dataset import SENSOR_COLUMNS, STATIC_COLUMNS

ROLL = 7  # days in the rolling window
# the sensors whose recent trend matters most for an incipient failure
TREND_SENSORS = ["motor_temp_c", "bearing_vibration_mm_s", "jam_rate_per_k",
                 "motor_current_a", "throughput_kpph"]


def _slope(a: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    if len(a) < 2:
        return 0.0
    x = np.arange(len(a), dtype=float)
    return float(np.polyfit(x, a, 1)[0])


def build_features(df: pd.DataFrame, sensors: list[str] | None = None,
                   static: list[str] | None = None, trend: list[str] | None = None,
                   ) -> tuple[pd.DataFrame, list[str]]:
    """Return (dataframe with feature columns added, list of feature column names).

    Column-agnostic: defaults to the USPS synthetic schema, but pass `sensors`/`static`/`trend` to run on
    ANY dataset (a real CMMS export, or a public benchmark) — the same rolling-trend recipe applies to
    whatever numeric columns you give it. This is what lets real data drop in without renaming everything."""
    sensors = list(sensors if sensors is not None else SENSOR_COLUMNS)
    static = list(static if static is not None else STATIC_COLUMNS)
    trend = list(trend if trend is not None else TREND_SENSORS)
    df = df.sort_values(["machine_id", "date"]).reset_index(drop=True)
    parts = []
    for _, g in df.groupby("machine_id", sort=False):
        g = g.copy()
        for c in trend:
            r = g[c].rolling(ROLL, min_periods=2)
            g[f"{c}_mean7"] = r.mean()
            g[f"{c}_std7"] = r.std().fillna(0.0)
            g[f"{c}_slope7"] = r.apply(_slope, raw=True).fillna(0.0)
        parts.append(g)
    df = pd.concat(parts, ignore_index=True)

    feat_cols = sensors + static
    for c in trend:
        feat_cols += [f"{c}_mean7", f"{c}_std7", f"{c}_slope7"]
    # rolling leaves NaNs on the first row of each machine — fill with the column median (imputer-safe too)
    df[feat_cols] = df[feat_cols].fillna(df[feat_cols].median(numeric_only=True))
    return df, feat_cols
