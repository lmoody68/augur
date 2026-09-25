"""
AUGUR — APPS MODE.

Takes the USPS **APPS** (Automated Package Processing System) parcel sorter and puts it under AUGUR's
predictive lens. The APPS control room already shows an **"APPS Top/Bottom Sites — Thruput Bottom 5"** board
with a **rule-based "Side At Risk %"** — a snapshot of what's struggling RIGHT NOW. AUGUR's leap: turn that
into a **forecast** — *"this side's Singulator drive is trending to fail in ~4 days; service it before it
takes the line down."*

What this module does:
  • generate_apps_fleet() — a synthetic APPS fleet in AUGUR's exact sensor schema, but shaped like the real
    control board: Site · Serial# · Side units, with OP#, Runtime, Pcs Fed, Side Thruput, and failures tagged
    to APPS field-replaceable units (FRUs) — Sorter carrier bearing, Induct/Feed belt, Singulator drive motor.
  • fit_and_score() — reuses AUGUR's 3 heads (risk · FRU-mode · RUL) trained on the unit's HISTORY and scored
    on today's reading (history→today, no scoring a row it trained on).
  • apps_board_html() — a predictive at-risk board that mirrors the APPS control-room layout, worst-first,
    but every risk column is an AI forecast: Predicted At-Risk %, RUL (days), Likely FRU, and an action.

⚠️ Demo uses SYNTHETIC data in the APPS shape (no real USPS site/telemetry data). A real APPS run-stats /
CMMS export drops into the same pipeline unchanged. The honest held-out metrics are AUGUR's grouped-split
`demo`/`benchmark`; this board is the live scoring view.
"""
from __future__ import annotations

import html
import time

import numpy as np
import pandas as pd

from . import dataset
from . import features as F
from .dataset import CENSOR_DAYS, HORIZON_DAYS
from .model import THRESHOLD, _band, _clf, _fmt_rul, _reg

# The generator's three independently-wearing components → the APPS FRU a tech would actually pull.
_MODE_TO_FRU = {"bearing": "Sorter carrier bearing", "belt": "Induct / Feed belt",
                "motor": "Singulator drive motor"}
APPS_FRU = list(_MODE_TO_FRU.values())

# Public USPS facility names for a recognizable demo board (no real telemetry — just labels).
_SITES = [("Ybor City (FL)", "P&DC"), ("Chicago (IL)", "RPDC"), ("Palatine (IL)", "P&DC"),
          ("Jacksonville (FL)", "RPDC"), ("Oklahoma City (OK)", "P&DC"), ("Dulles (VA)", "P&DC"),
          ("Atlanta (GA)", "P&DC"), ("Dallas (TX)", "P&DC"), ("Denver (CO)", "P&DC"),
          ("Phoenix (AZ)", "P&DC"), ("Seattle (WA)", "P&DC"), ("St. Louis (MO)", "P&DC")]
_OP_NUMS = [159, 244, 248, 249]   # APPS operation numbers seen on the real board
_HOME_SITE = "Ybor City (FL)"     # highlighted (yellow) like the operator's own site on the real board


def _remap_mode(m):
    return _MODE_TO_FRU.get(m, m)   # keep 'none'/'unknown' as-is


def generate_apps_fleet(n_units: int = 24, days: int = 380, seed: int = 11) -> pd.DataFrame:
    """Reuse AUGUR's proven 3-component generator, then dress it as an APPS control-room fleet:
    Site · Serial# · Side machine_ids, APPS FRU labels, and the board's display columns (OP#, runtime,
    pcs fed, side thruput)."""
    df = dataset.generate(n_machines=n_units, days=days, seed=seed)
    rng = np.random.default_rng(seed)
    ids = sorted(df["machine_id"].unique())
    meta = {}
    for i, mid in enumerate(ids):
        site, kind = _SITES[i % len(_SITES)]
        serial = f"{int(rng.integers(10, 99)):03d}"
        side = f"{(i % 2) + 1}/2"
        meta[mid] = {"name": f"{site} {kind} - {serial} - {side}", "site": site,
                     "op": int(rng.choice(_OP_NUMS))}
    df["machine_id"] = df["machine_id"].map(lambda m: meta[m]["name"])
    df["_site"] = df["machine_id"].map({v["name"]: v["site"] for v in meta.values()})
    df["op_num"] = df["machine_id"].map({v["name"]: v["op"] for v in meta.values()})
    df["machine_type"] = "APPS"
    df["failure_mode"] = df["failure_mode"].map(_remap_mode)
    if "mode_within_h" in df.columns:
        df["mode_within_h"] = df["mode_within_h"].map(_remap_mode)
    return df


def _display_cols(latest: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    """Board display metrics derived from the live reading — Side Thruput (pcs/hr), Runtime, Pcs Fed."""
    rng = np.random.default_rng(seed + 1)
    d = latest.copy()
    # throughput_kpph (~20-32) → a realistic per-side pieces/hour in the board's range (~900-6700)
    d["side_thruput"] = (d["throughput_kpph"] * 185 + rng.normal(0, 120, len(d))).clip(600).round().astype(int)
    runtime_h = rng.uniform(2.1, 5.5, len(d))
    d["_runtime_h"] = runtime_h
    d["runtime"] = [f"{int(h):02d}:{int((h*60) % 60):02d}:{int((h*3600) % 60):02d}" for h in runtime_h]
    d["pcs_fed"] = (d["side_thruput"] * runtime_h).round().astype(int)
    return d


def fit_and_score(df: pd.DataFrame) -> pd.DataFrame:
    """Train the 3 heads on each unit's HISTORY, score today's reading. Returns the ranked board rows."""
    fdf, feat = F.build_features(df)
    # history → today: hold out each unit's latest row from training, then score it (no self-scoring)
    latest_idx = fdf.groupby("machine_id")["date"].idxmax().to_numpy()
    is_latest = fdf.index.isin(latest_idx)
    tr = fdf[~is_latest]
    Xtr = tr[feat].to_numpy()
    risk = _clf(); risk.fit(Xtr, tr["fail_within_h"].to_numpy())
    mode = _clf(); mode.fit(Xtr, tr["mode_within_h"].to_numpy())
    rul = _reg(); rul.fit(Xtr, tr["days_to_failure"].to_numpy())

    live = _display_cols(fdf[is_latest].copy())
    Xl = live[feat].to_numpy()
    prob = risk.predict_proba(Xl)[:, 1]
    live["risk"] = prob
    live["health"] = np.round(100 * (1 - prob)).astype(int)
    live["band"] = [_band(p) for p in prob]
    live["rul_days"] = [_fmt_rul(x) for x in rul.predict(Xl)]

    classes = list(mode.named_steps["rf"].classes_)
    fail_ix = [i for i, c in enumerate(classes) if c in APPS_FRU]
    if fail_ix:
        mp = mode.predict_proba(Xl)[:, fail_ix]
        live["likely_fru"] = [classes[fail_ix[j]] for j in mp.argmax(axis=1)]
    else:
        live["likely_fru"] = "—"
    return live.sort_values("risk", ascending=False).reset_index(drop=True)


_ACTION = {"CRITICAL": "🔧 Service NOW", "HIGH": "Schedule this week", "WATCH": "Inspect next PM", "OK": "—"}
_ROWCLASS = {"CRITICAL": "crit", "HIGH": "high", "WATCH": "watch", "OK": "ok"}


def apps_board_html(board: pd.DataFrame) -> str:
    """The predictive at-risk board — mirrors the APPS control-room 'Thruput Bottom Sites' layout, but every
    risk column is an AUGUR forecast. Worst (highest predicted risk) first."""
    now = time.strftime("%H:%M:%S")
    n = len(board)
    rows = []
    for i, r in board.iterrows():
        rank = i + 1
        cls = _ROWCLASS.get(r["band"], "ok")
        if str(r["_site"]) == _HOME_SITE:
            cls += " home"
        tag = " (Worst)" if rank == 1 else (" (Best)" if rank == n else "")
        fru = html.escape(str(r["likely_fru"])) if r["band"] != "OK" else "—"
        rows.append(
            f"<tr class='{cls}'>"
            f"<td class='rank'>{rank}{tag}</td>"
            f"<td class='site'>{html.escape(str(r['machine_id']))}</td>"
            f"<td class='num'>{r['op_num']}</td>"
            f"<td class='num risk'>{r['risk']*100:.1f}%</td>"
            f"<td class='num'>{r['runtime']}</td>"
            f"<td class='num rul'>{r['rul_days']}</td>"
            f"<td>{fru}</td>"
            f"<td class='num'>{r['side_thruput']:,}</td>"
            f"<td class='num'>{r['pcs_fed']:,}</td>"
            f"<td class='act'><span class='band {cls}'>{r['band']}</span> {_ACTION.get(r['band'],'')}</td>"
            f"</tr>")
    n_crit = int((board["band"] == "CRITICAL").sum())
    n_high = int((board["band"] == "HIGH").sum())
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>AUGUR — APPS Predictive At-Risk Board</title><style>
:root{{--navy:#12356b;--navy2:#0e2a55;--bg:#eef1f5;--ink:#12233a;--dim:#5b667c;--line:#cfd6e2;
  --crit:#c0392b;--high:#e67e22;--watch:#e0a800;--ok:#2e7d54;--yellow:#fff23a}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:13.5px/1.45 system-ui,Segoe UI,Arial,sans-serif}}
.wrap{{max-width:1160px;margin:0 auto;padding:16px 14px 40px}}
.tabs{{display:flex;gap:6px;margin-bottom:12px}}
.tabs a{{text-decoration:none;color:var(--dim);background:#fff;border:1px solid var(--line);border-bottom:none;
  border-radius:8px 8px 0 0;padding:9px 16px;font-weight:600;font-size:13px}}
.tabs a.on{{color:#fff;background:var(--navy);border-color:var(--navy)}}
.board{{background:#fff;border:1px solid var(--line);border-radius:0 8px 8px 8px;overflow:hidden;box-shadow:0 6px 22px rgba(20,40,80,.12)}}
.hdr{{background:linear-gradient(180deg,var(--navy),var(--navy2));color:#fff;display:flex;align-items:center;
  justify-content:space-between;padding:11px 16px;font-weight:700;letter-spacing:.3px}}
.hdr .t{{font-variant-numeric:tabular-nums;font-size:15px}}.hdr .c{{font-size:16px;text-align:center;flex:1}}
.hdr .r{{font-size:12px;opacity:.9}}
.sub{{background:#1c4a92;color:#dfe8f7;text-align:center;padding:6px;font-size:12.5px;font-weight:600}}
table{{width:100%;border-collapse:collapse}}
th,td{{padding:8px 12px;border-bottom:1px solid #e3e8f0;text-align:left;white-space:nowrap}}
th{{background:#22509a;color:#fff;font-size:11px;letter-spacing:.03em;text-transform:uppercase;position:sticky;top:0}}
td.num{{text-align:right;font-variant-numeric:tabular-nums}}
td.rank{{font-weight:700;color:var(--dim)}}td.site{{font-weight:700}}
td.risk{{font-weight:800}}td.rul{{font-weight:700}}
tr.crit{{background:#fdecea}}tr.crit td.risk{{color:var(--crit)}}
tr.high{{background:#fef1e6}}tr.high td.risk{{color:var(--high)}}
tr.watch{{background:#fff9e0}}tr.watch td.risk{{color:#9a7b00}}
tr.ok td.risk{{color:var(--ok)}}
tr.home{{outline:2px solid var(--yellow);outline-offset:-2px;background:#fffce0}}
.band{{color:#fff;padding:2px 7px;border-radius:999px;font-size:10.5px;font-weight:800;margin-right:6px}}
.band.crit{{background:var(--crit)}}.band.high{{background:var(--high)}}.band.watch{{background:var(--watch);color:#3a2f00}}.band.ok{{background:var(--ok)}}
td.act{{font-size:12px;color:var(--dim)}}
.summary{{display:flex;gap:10px;margin:12px 0;flex-wrap:wrap}}
.card{{background:#fff;border:1px solid var(--line);border-radius:10px;padding:10px 16px;flex:1;min-width:120px}}
.card .n{{display:block;font-size:22px;font-weight:800}}.card .l{{color:var(--dim);font-size:12px}}
.card.crit .n{{color:var(--crit)}}.card.high .n{{color:var(--high)}}.card.ok .n{{color:var(--ok)}}
.foot{{color:var(--dim);font-size:12px;margin-top:14px;line-height:1.6}}
.legend{{margin-top:8px;font-size:12px;color:var(--dim)}}.legend b{{color:var(--ink)}}
</style></head><body><div class="wrap">
<div class="tabs"><a href="watchlist.html">Fleet Watchlist</a><a class="on" href="apps_board.html">APPS Board</a></div>
<div class="summary">
  <div class="card crit"><span class="n">{n_crit}</span><span class="l">Predicted CRITICAL</span></div>
  <div class="card high"><span class="n">{n_high}</span><span class="l">Predicted HIGH</span></div>
  <div class="card"><span class="n">{n}</span><span class="l">APPS sides monitored</span></div>
  <div class="card ok"><span class="n">14d</span><span class="l">Forecast horizon</span></div>
</div>
<div class="board">
  <div class="hdr"><span class="t">{now} ET</span><span class="c">APPS — Predictive At-Risk Board</span><span class="r">AUGUR ▸ 14-day forecast</span></div>
  <div class="sub">APPS Sites — AI-forecast failures, worst first (compare to the control-room "Side At Risk %", but predicted, not reactive)</div>
  <table><thead><tr>
    <th>Rank</th><th>Site · Serial# · Side</th><th>OP#</th><th>Pred. At-Risk</th><th>Runtime</th>
    <th>RUL (days)</th><th>Likely FRU</th><th>Side Thruput</th><th>Pcs Fed</th><th>Action</th>
  </tr></thead><tbody>{''.join(rows)}</tbody></table>
</div>
<div class="legend">🟨 <b>your site</b> (highlighted) · 🔴 CRITICAL · 🟠 HIGH · 🟡 WATCH · 🟢 OK — <b>ranked by predicted risk, not current thruput.</b></div>
<p class="foot">
<b>The leap:</b> the APPS control room shows a rule-based "Side At Risk %" for what's struggling <i>now</i>;
AUGUR forecasts which side will fail in the next {HORIZON_DAYS} days, <b>which FRU</b>, and <b>how many days</b>
of lead time you have — so maintenance is scheduled, not scrambled.<br>
<b>Honesty:</b> this board runs on <b>synthetic data in the APPS shape</b> (no real USPS telemetry) and scores
each unit's live reading with a model trained on its history. A real APPS run-stats / CMMS export drops into
the same pipeline unchanged; the audited held-out accuracy is AUGUR's grouped-split <code>demo</code> /
NASA <code>benchmark</code> report.
</p>
</div></body></html>"""


def run(n_units: int = 24, days: int = 380, out_html: str | None = None) -> dict:
    """Generate → fit → score → write the APPS board HTML. Returns a small summary."""
    import os
    df = generate_apps_fleet(n_units=n_units, days=days)
    board = fit_and_score(df)
    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..", "data", "apps_board.html")
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(apps_board_html(board))
    return {"units": int(len(board)),
            "critical": int((board["band"] == "CRITICAL").sum()),
            "high": int((board["band"] == "HIGH").sum()),
            "html": os.path.abspath(out_html),
            "top": board.head(5)[["machine_id", "band", "risk", "rul_days", "likely_fru"]].to_dict("records")}
