"""
AUGUR — data layer.

Two ways to get data:
  • generate(...)  — a SYNTHETIC but physically-plausible fleet of USPS mail-processing machines
    (bar-code / flat sorters). Each machine has THREE independently-wearing components — bearing, belt,
    motor — each with its own hidden health that degrades with use and heat. When a component's health
    collapses it FAILS, and the failure is tagged with which component gave out. The sensors are tied to
    the components (vibration↔bearing, tension↔belt, temp/current↔motor), so per-failure-mode models have
    a real signal to learn, not a faked one. Clearly synthetic — swap for real data the moment it exists.
  • load_csv(path)  — the SAME schema, so a real export drops straight in. See REQUIRED_COLUMNS.

Labels built by add_label():
  • fail_within_h   — a failure of ANY kind in the next HORIZON_DAYS (the primary risk target)
  • mode_within_h   — which component fails first in that window ("none" if no failure) → per-mode model
  • days_to_failure — days until the next failure (censored at CENSOR_DAYS) → the learned RUL target
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON_DAYS = 14   # predict a failure inside this many days — the planning window
CENSOR_DAYS = 60    # RUL is measured/emitted up to this cap ("more than CENSOR days out")

# The raw daily sensor reading schema (what a real telemetry/CMMS export must provide).
SENSOR_COLUMNS = [
    "motor_temp_c", "bearing_vibration_mm_s", "belt_tension_pct", "jam_rate_per_k",
    "motor_current_a", "throughput_kpph", "ambient_temp_c",
]
STATIC_COLUMNS = ["machine_age_years", "cycles_k", "days_since_maintenance"]
REQUIRED_COLUMNS = ["machine_id", "date"] + SENSOR_COLUMNS + STATIC_COLUMNS
# `failed` (0/1) is required for TRAINING; `failure_mode` (bearing|belt|motor, blank if not failed)
# enables the per-mode model — if a real export lacks it, AUGUR still trains the risk + RUL models.

MACHINE_TYPES = ["DBCS", "AFSM100", "APBS", "DIOSS"]   # real USPS sorter families (context flavor)
FAILURE_MODES = ["bearing", "belt", "motor"]           # the components AUGUR predicts by name
_COMPONENTS = FAILURE_MODES


def _seasonal(day_idx: float, period: int = 365) -> float:
    return 0.5 + 0.5 * np.sin(2 * np.pi * (day_idx - 172) / period) * 0.4 + 0.8


def generate(n_machines: int = 40, days: int = 400, seed: int = 7) -> pd.DataFrame:
    """Return a tidy daily dataframe for a fleet with `failed`, `failure_mode`, and the derived labels.
    Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    rows = []
    for mid in range(n_machines):
        mtype = MACHINE_TYPES[mid % len(MACHINE_TYPES)]
        age0 = float(rng.uniform(0.5, 12.0))
        duty = float(rng.uniform(0.55, 1.0))
        quality = float(rng.normal(1.0, 0.08))
        # each component wears at its own rate; one component runs "hot" (this machine's weak point)
        base = rng.uniform(0.0015, 0.0040, size=3) / max(quality, 0.6)
        base[rng.integers(0, 3)] *= float(rng.uniform(1.6, 2.4))
        h = rng.uniform(0.80, 0.98, size=3)            # [bearing, belt, motor] latent health
        cycles_k = float(rng.uniform(500, 9000))
        days_since_maint = int(rng.integers(0, 60))
        for d in range(days):
            load = float(np.clip(duty * _seasonal(d) * rng.normal(1.0, 0.05), 0.2, 1.4))
            heat = 1.0 + 0.6 * max(load - 0.8, 0)
            h -= base * load * heat
            if rng.random() < 0.012:                    # random shock to one component
                h[rng.integers(0, 3)] -= float(rng.uniform(0.02, 0.08))

            # per-component failure hazard rises steeply as that component's health falls
            hazard = np.clip((0.26 - h) / 0.26, 0, 1) ** 2 * 0.14
            hit = (h <= 0.12) | (rng.random(3) < hazard)
            failed, mode = 0, ""
            if hit.any():
                comp = int(np.argmin(np.where(hit, h, np.inf)))   # the worst failing component
                failed, mode = 1, _COMPONENTS[comp]

            ub, ubelt, um = 1 - h[0], 1 - h[1], 1 - h[2]
            uov = float(max(ub, ubelt, um))
            temp = 35 + 30 * load + 26 * um + rng.normal(0, 1.6)
            vib = 1.4 + 6.5 * ub ** 1.5 + 0.5 * load + rng.normal(0, 0.25)
            tension = 86 - 22 * ubelt + rng.normal(0, 1.5)
            jam = max(0.0, 0.4 + 9.0 * uov ** 2 + rng.normal(0, 0.4))
            current = 10 + 6 * load + 5.5 * um + rng.normal(0, 0.5)
            thru = max(1.0, 32 * load * (0.6 + 0.4 * (1 - uov)) + rng.normal(0, 1.2))
            amb = 20 + 8 * _seasonal(d) + rng.normal(0, 1.2)

            rows.append({
                "machine_id": f"{mtype}-{mid:02d}", "machine_type": mtype, "date": d,
                "motor_temp_c": temp, "bearing_vibration_mm_s": vib, "belt_tension_pct": tension,
                "jam_rate_per_k": jam, "motor_current_a": current, "throughput_kpph": thru,
                "ambient_temp_c": amb, "machine_age_years": age0 + d / 365.0,
                "cycles_k": cycles_k, "days_since_maintenance": days_since_maint,
                "failed": failed, "failure_mode": mode,
            })

            cycles_k += load * float(rng.uniform(20, 40))
            days_since_maint += 1
            if failed:                                  # repair the failed component (imperfect restore)
                h[comp] = float(np.clip(rng.normal(0.88, 0.04), 0.7, 0.97))
                days_since_maint = 0
            elif days_since_maint >= 90:                # scheduled PM lifts everything a little
                h = np.clip(h + rng.uniform(0.04, 0.10, size=3), 0, 0.98)
                days_since_maint = 0

    df = pd.DataFrame(rows)
    df["date"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(df["date"], unit="D")
    return add_label(df)


def add_label(df: pd.DataFrame, horizon: int = HORIZON_DAYS, censor: int = CENSOR_DAYS) -> pd.DataFrame:
    """Add the three supervised targets from the `failed`/`failure_mode` event columns. Uses only FUTURE
    events relative to each row (the label is the future — features never see it)."""
    df = df.sort_values(["machine_id", "date"]).reset_index(drop=True)
    if "failure_mode" not in df.columns:
        df["failure_mode"] = ""
    out = []
    for _, g in df.groupby("machine_id", sort=False):
        g = g.copy()
        failed = g["failed"].to_numpy()
        modes = g["failure_mode"].fillna("").to_numpy()
        n = len(failed)
        lab = np.zeros(n, dtype=int)
        mlab = np.array(["none"] * n, dtype=object)
        dtf = np.full(n, float(censor))
        # index of each future failure, for a fast "next failure" lookup
        fail_idx = np.where(failed == 1)[0]
        for i in range(n):
            nxt = fail_idx[fail_idx > i]
            if len(nxt):
                j = int(nxt[0])
                dtf[i] = min(censor, j - i)
                if j - i <= horizon:
                    lab[i] = 1
                    mlab[i] = modes[j] or "unknown"
        g["fail_within_h"] = lab
        g["mode_within_h"] = mlab
        g["days_to_failure"] = dtf
        out.append(g)
    return pd.concat(out, ignore_index=True)


def load_csv(path: str) -> pd.DataFrame:
    """Load a real export. Validates required columns, parses dates, and (if `failed` is present) builds
    the labels — so real logs train exactly like the synthetic set."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}\nExpected: {REQUIRED_COLUMNS}")
    df["date"] = pd.to_datetime(df["date"])
    if "machine_type" not in df.columns:
        df["machine_type"] = df["machine_id"].astype(str).str.split("-").str[0]
    if "failed" in df.columns:
        df = add_label(df)
    return df
