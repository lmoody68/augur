"""
AUGUR — reporting. Turns a scored watchlist into (a) a terminal table a technician can read at a glance
and (b) a standalone HTML dashboard a maintenance manager can share. No chart libraries required.
"""
from __future__ import annotations

import html

import pandas as pd

_BAND_COLOR = {"CRITICAL": "#c0392b", "HIGH": "#e67e22", "WATCH": "#e0a800", "OK": "#2e7d54"}


def text_watchlist(scored: pd.DataFrame, top: int = 15) -> str:
    has_mode = "likely_mode" in scored.columns
    hdr = f"{'MACHINE':<12}{'TYPE':<9}{'RISK':>6}  {'HEALTH':>6}  {'BAND':<9}"
    hdr += f"{'LIKELY':<9}" if has_mode else ""
    hdr += f"{'RUL(d)':>7}"
    lines = ["", "AUGUR — predictive-maintenance watchlist (highest risk first)", "=" * 78, hdr, "-" * 78]
    for _, r in scored.head(top).iterrows():
        line = (f"{r['machine_id']:<12}{r['machine_type']:<9}"
                f"{r['failure_probability']*100:5.1f}% {r['health_score']:>6}  {r['risk_band']:<9}")
        if has_mode:
            mode = r['likely_mode'] if r['risk_band'] not in ("OK",) else "—"
            line += f"{str(mode):<9}"
        line += f"{str(r['rul_days']):>7}"
        lines.append(line)
    n_crit = int((scored["risk_band"] == "CRITICAL").sum())
    n_high = int((scored["risk_band"] == "HIGH").sum())
    lines += ["-" * 78,
              f"{len(scored)} machines · {n_crit} CRITICAL · {n_high} HIGH — schedule these first.", ""]
    return "\n".join(lines)


def html_dashboard(scored: pd.DataFrame, metrics: dict | None = None) -> str:
    has_mode = "likely_mode" in scored.columns
    rows = []
    for _, r in scored.iterrows():
        c = _BAND_COLOR.get(r["risk_band"], "#666")
        mode_cell = ""
        if has_mode:
            m = html.escape(str(r["likely_mode"])) if r["risk_band"] != "OK" else "—"
            mode_cell = f"<td>{m}</td>"
        rows.append(
            f"<tr><td class='m'>{html.escape(str(r['machine_id']))}</td>"
            f"<td>{html.escape(str(r['machine_type']))}</td>"
            f"<td class='num'>{r['failure_probability']*100:.1f}%</td>"
            f"<td class='num'>{r['health_score']}</td>"
            f"<td><span class='band' style='background:{c}'>{r['risk_band']}</span></td>"
            f"{mode_cell}"
            f"<td class='num'>{r['rul_days']}</td></tr>")
    m = metrics or {}
    metric_cards = ""
    if m:
        for label, key, fmt in [("ROC-AUC", "roc_auc", "{:.3f}"), ("PR-AUC", "pr_auc", "{:.3f}"),
                                ("Recall", "recall", "{:.2f}"), ("Precision", "precision", "{:.2f}")]:
            if key in m:
                metric_cards += f"<div class='card'><span class='n'>{fmt.format(m[key])}</span><span class='l'>{label}</span></div>"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>AUGUR Watchlist</title><style>
:root{{color-scheme:light dark;--bg:#f5f6f8;--panel:#fff;--ink:#1a2233;--dim:#5b667c;--line:#e4e8ef;--accent:#2b4a6f}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0e1421;--panel:#161f31;--ink:#e8edf6;--dim:#9aa6be;--line:#26314b}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,Segoe UI,sans-serif}}
.wrap{{max-width:900px;margin:0 auto;padding:28px 18px}}
h1{{font-size:22px;margin:0 0 2px}}.sub{{color:var(--dim);margin:0 0 18px}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 16px;flex:1;min-width:110px}}
.card .n{{display:block;font-size:22px;font-weight:700;color:var(--accent)}}.card .l{{color:var(--dim);font-size:12px}}
table{{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}}
th,td{{padding:9px 12px;text-align:left;border-bottom:1px solid var(--line)}}
th{{background:color-mix(in srgb,var(--panel),var(--ink) 5%);font-size:11px;letter-spacing:.04em;text-transform:uppercase;color:var(--dim)}}
td.num{{text-align:right;font-variant-numeric:tabular-nums}}td.m{{font-weight:600}}
.band{{color:#fff;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700}}
.foot{{color:var(--dim);font-size:12px;margin-top:14px}}
.tabs{{display:flex;gap:6px;margin-bottom:14px}}
.tabs a{{text-decoration:none;color:var(--dim);background:var(--panel);border:1px solid var(--line);
  border-radius:8px;padding:8px 16px;font-weight:600;font-size:13px}}
.tabs a.on{{color:#fff;background:var(--accent);border-color:var(--accent)}}
</style></head><body><div class="wrap">
<div class="tabs"><a class="on" href="watchlist.html">Fleet Watchlist</a><a href="apps_board.html">APPS Board</a></div>
<h1>🔧 AUGUR — Predictive-Maintenance Watchlist</h1>
<p class="sub">Highest failure risk first · horizon = next 14 days · schedule CRITICAL / HIGH before they break.</p>
<div class="cards">{metric_cards}</div>
<table><thead><tr><th>Machine</th><th>Type</th><th>Risk</th><th>Health</th><th>Band</th>{'<th>Likely part</th>' if has_mode else ''}<th>RUL (d)</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p class="foot">Model: RandomForest on sensor trend features, evaluated on machines held out entirely from
training (grouped split). RUL is a heuristic planning aid, not a guarantee. Swap the synthetic fleet for a
real CMMS export with the same columns to retrain on your own equipment.</p>
</div></body></html>"""
