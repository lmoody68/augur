"""
AUGUR — alerting. Turns the watchlist from something you have to READ into something that FINDS the
on-call tech. When a machine crosses the alert band, AUGUR notifies whoever's on call by email and/or
text — no new paid service required (SMS goes through the carrier's email-to-SMS gateway).

SAFETY: dry-run by DEFAULT. It sends for real only when the config says `enabled: true` AND SMTP creds
are present in the environment — otherwise it prints exactly what it WOULD send. A per-machine cooldown
stops it re-paging the same machine every scoring cycle.
"""
from __future__ import annotations

import json
import os
import smtplib
import time
from email.message import EmailMessage

import pandas as pd

_HERE = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(_HERE, "..", "config", "on_call.json")
LOG_PATH = os.path.join(_HERE, "..", "data", "alert_log.json")
_BAND_RANK = {"OK": 0, "WATCH": 1, "HIGH": 2, "CRITICAL": 3}
# carrier email-to-SMS gateways — a text with no SMS provider/API needed
_SMS_GW = {"vzw": "vtext.com", "verizon": "vtext.com", "att": "txt.att.net",
           "tmobile": "tmomail.net", "tmo": "tmomail.net", "sprint": "messaging.sprintpcs.com"}


def load_config(path: str = CONFIG_PATH) -> dict:
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    return {"enabled": False, "min_band": "HIGH", "cooldown_hours": 12,
            "channels": ["email"], "recipients": []}


def _smtp_env() -> dict | None:
    host = os.getenv("AUGUR_SMTP_HOST")
    if not (host and os.getenv("AUGUR_SMTP_USER") and os.getenv("AUGUR_SMTP_PASS")):
        return None
    return {"host": host, "port": int(os.getenv("AUGUR_SMTP_PORT", "465")),
            "user": os.getenv("AUGUR_SMTP_USER"), "pw": os.getenv("AUGUR_SMTP_PASS"),
            "from": os.getenv("AUGUR_SMTP_FROM", os.getenv("AUGUR_SMTP_USER"))}


def _recipients_addrs(rcpt: dict, channels: list[str]) -> list[str]:
    addrs = []
    if "email" in channels and rcpt.get("email"):
        addrs.append(rcpt["email"])
    if "sms" in channels and rcpt.get("sms"):
        gw = _SMS_GW.get(str(rcpt.get("carrier", "")).lower())
        if gw:
            addrs.append(f"{''.join(ch for ch in str(rcpt['sms']) if ch.isdigit())}@{gw}")
    return addrs


def _load_log() -> dict:
    try:
        return json.load(open(LOG_PATH, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _save_log(d: dict) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    json.dump(d, open(LOG_PATH, "w"), indent=2)


def _message(rows: pd.DataFrame) -> tuple[str, str]:
    n = len(rows)
    subject = f"AUGUR: {n} machine{'s' if n != 1 else ''} need attention"
    lines = ["AUGUR predictive-maintenance alert — machines crossing the alert band:\n"]
    for _, r in rows.iterrows():
        part = f" · likely {r['likely_mode']}" if r.get("likely_mode", "—") not in ("—", "") else ""
        lines.append(f"  • {r['machine_id']} ({r['machine_type']}): {r['risk_band']} — "
                     f"{r['failure_probability']*100:.0f}% risk{part} · ~{r['rul_days']} days left")
    lines.append("\nSchedule the CRITICAL/HIGH machines first. — AUGUR")
    return subject, "\n".join(lines)


def dispatch(watchlist: pd.DataFrame, config: dict | None = None,
             dry_run: bool | None = None, force: bool = False) -> dict:
    """Decide who to page for machines at/above the alert band, respecting the cooldown. Sends for real
    only when enabled + SMTP creds exist and dry_run is not forced. Returns a summary dict."""
    cfg = config or load_config()
    min_rank = _BAND_RANK.get(str(cfg.get("min_band", "HIGH")).upper(), 2)
    flagged = watchlist[watchlist["risk_band"].map(lambda b: _BAND_RANK.get(b, 0) >= min_rank)].copy()

    smtp = _smtp_env()
    if dry_run is None:
        dry_run = not (cfg.get("enabled") and smtp)

    # cooldown filter (skip machines paged within cooldown_hours) unless force
    log = _load_log()
    cooldown = float(cfg.get("cooldown_hours", 12)) * 3600
    now = time.time()
    if not force:
        keep = [m for m in flagged["machine_id"]
                if now - log.get(str(m), 0) >= cooldown]
        flagged = flagged[flagged["machine_id"].isin(keep)]

    recipients = cfg.get("recipients", [])
    channels = cfg.get("channels", ["email"])
    summary = {"flagged": int(len(flagged)), "dry_run": bool(dry_run),
               "would_notify": [], "sent": 0, "machines": list(flagged["machine_id"])}
    if flagged.empty or not recipients:
        summary["note"] = ("no machines above alert band" if flagged.empty
                           else "no recipients configured")
        return summary

    subject, body = _message(flagged)
    for rcpt in recipients:
        addrs = _recipients_addrs(rcpt, channels)
        summary["would_notify"].append({"name": rcpt.get("name"), "addrs": addrs})
        if dry_run or not addrs:
            continue
        try:
            msg = EmailMessage()
            msg["Subject"], msg["From"], msg["To"] = subject, smtp["from"], ", ".join(addrs)
            msg.set_content(body)
            with smtplib.SMTP_SSL(smtp["host"], smtp["port"]) as s:
                s.login(smtp["user"], smtp["pw"])
                s.send_message(msg)
            summary["sent"] += 1
        except Exception as e:  # noqa: BLE001
            summary.setdefault("errors", []).append(str(e))

    if not dry_run:                                 # record the page time for cooldown
        for m in flagged["machine_id"]:
            log[str(m)] = now
        _save_log(log)
    return summary
