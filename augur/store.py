"""
AUGUR — live reading store (SQLite, stdlib).

The daily-CSV workflow is fine for a backfill, but the "live feed" future needs somewhere for readings to
land continuously and for the scorer to read recent history. This is that: a tiny append-only store of
per-machine daily readings. Swap SQLite for Postgres/Timescale later without touching the service — the
functions (`ingest`, `recent`, `machine_history`) are the contract.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager

import pandas as pd

from .dataset import SENSOR_COLUMNS, STATIC_COLUMNS

_HERE = os.path.dirname(__file__)
DB_PATH = os.path.join(_HERE, "..", "data", "augur_live.db")
_COLS = ["machine_id", "machine_type", "date"] + SENSOR_COLUMNS + STATIC_COLUMNS + ["failed", "failure_mode"]


@contextmanager
def _conn(path: str = DB_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    c = sqlite3.connect(path)
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init(path: str = DB_PATH) -> None:
    cols = ", ".join(f'"{c}" REAL' if c in SENSOR_COLUMNS + STATIC_COLUMNS + ["failed"]
                     else f'"{c}" TEXT' for c in _COLS)
    with _conn(path) as c:
        c.execute(f"CREATE TABLE IF NOT EXISTS readings ({cols}, "
                  "PRIMARY KEY (machine_id, date))")
        c.execute("CREATE INDEX IF NOT EXISTS ix_machine ON readings(machine_id)")


def ingest(readings, path: str = DB_PATH) -> int:
    """Insert one reading (dict) or many (list of dicts / DataFrame). Upserts on (machine_id, date).
    Missing optional fields default sensibly. Returns the number of rows written."""
    if isinstance(readings, pd.DataFrame):
        readings = readings.to_dict("records")
    elif isinstance(readings, dict):
        readings = [readings]
    init(path)
    rows = []
    for r in readings:
        r = dict(r)
        r.setdefault("machine_type", str(r.get("machine_id", "")).split("-")[0] or "UNKNOWN")
        r.setdefault("failed", 0)
        r.setdefault("failure_mode", "")
        r["date"] = str(pd.to_datetime(r["date"]).date())
        rows.append([r.get(c) for c in _COLS])
    ph = ",".join("?" * len(_COLS))
    with _conn(path) as c:
        c.executemany(f"INSERT OR REPLACE INTO readings VALUES ({ph})", rows)
    return len(rows)


def recent(days: int = 45, path: str = DB_PATH) -> pd.DataFrame:
    """All readings from the last `days` (enough history for the 7-day rolling features), as a DataFrame."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=_COLS)
    with _conn(path) as c:
        df = pd.read_sql_query("SELECT * FROM readings", c)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    cut = df["date"].max() - pd.Timedelta(days=days)
    return df[df["date"] >= cut].reset_index(drop=True)


def machine_history(machine_id: str, path: str = DB_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=_COLS)
    with _conn(path) as c:
        df = pd.read_sql_query("SELECT * FROM readings WHERE machine_id = ? ORDER BY date",
                               c, params=(machine_id,))
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def seed_from_dataframe(df: pd.DataFrame, path: str = DB_PATH) -> int:
    """Convenience: load a generated/loaded fleet into the live store (for the demo)."""
    with _conn(path) as c:
        c.execute("DROP TABLE IF EXISTS readings")
    return ingest(df, path)
