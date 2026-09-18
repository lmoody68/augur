"""
AUGUR — live-feed service (FastAPI).

The "live feed instead of a daily file" future: telemetry POSTs readings in as they happen, the store
keeps recent history, and the watchlist / alerts are available on demand. Same trained model as the CLI.

Endpoints:
  GET  /health                      service + model status
  POST /ingest    {reading|list}    push one or many readings into the live store
  GET  /watchlist                   score the latest reading per machine → ranked JSON
  GET  /watchlist.html              the same, as the shareable dashboard
  GET  /machine/{id}                that machine's recent history + current score
  POST /alerts/run  ?force=         page on-call for machines above the alert band (dry-run unless armed)
  GET  /meta                        the model's evaluation metrics

Run:  uvicorn augur.service:app --port 8920      (or: python augur_cli.py serve)
"""
from __future__ import annotations

import os

import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from . import store, model, report, alerts

app = FastAPI(title="AUGUR", version="0.1.0")


def _model_ready() -> bool:
    return os.path.exists(model.MODEL_PATH)


def _records(df):
    """DataFrame -> list of JSON-native dicts (pandas handles numpy scalars + ISO dates)."""
    return json.loads(df.to_json(orient="records", date_format="iso"))


@app.get("/health")
def health():
    return {"status": "ok", "service": "augur", "model_trained": _model_ready(),
            "readings": int(len(store.recent(days=3650)))}


@app.post("/ingest")
async def ingest(request: Request):
    try:
        payload = await request.json()
        n = store.ingest(payload)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"bad reading(s): {e}")
    return {"ingested": n}


def _scored():
    df = store.recent(days=45)
    if df.empty:
        raise HTTPException(404, "no readings in the store yet — POST to /ingest first")
    if not _model_ready():
        raise HTTPException(409, "model not trained yet — run `python augur_cli.py demo` or train on a CSV")
    return model.score(df, latest_only=True)


@app.get("/watchlist")
def watchlist():
    return JSONResponse(_records(_scored()))


@app.get("/watchlist.html", response_class=HTMLResponse)
def watchlist_html():
    import json
    meta = json.load(open(model.META_PATH)) if os.path.exists(model.META_PATH) else None
    return report.html_dashboard(_scored(), meta)


@app.get("/machine/{machine_id}")
def machine(machine_id: str):
    hist = store.machine_history(machine_id)
    if hist.empty:
        raise HTTPException(404, f"no readings for {machine_id}")
    row = _scored()
    row = row[row["machine_id"] == machine_id]
    return {"machine_id": machine_id, "n_readings": int(len(hist)),
            "latest_score": (_records(row)[0] if len(row) else None),
            "history_tail": _records(hist.tail(14))}


@app.post("/alerts/run")
def alerts_run(force: bool = False):
    s = alerts.dispatch(_scored(), force=force)
    return json.loads(json.dumps(s, default=str))


@app.get("/meta")
def meta():
    import json
    if not os.path.exists(model.META_PATH):
        raise HTTPException(409, "no model metrics yet")
    return json.load(open(model.META_PATH))
