#!/usr/bin/env python3
"""
AUGUR — Predictive-Maintenance AI · command line.

  python augur_cli.py demo                 # generate a fleet → train 3 heads → watchlist + HTML + alert dry-run
  python augur_cli.py train  --csv logs.csv     # train on a real export (needs a `failed` column)
  python augur_cli.py predict --csv logs.csv    # score a real export with the saved model → watchlist
  python augur_cli.py ingest --csv logs.csv     # push readings into the live store (for the feed)
  python augur_cli.py serve                # start the live-feed API (ingest + watchlist + alerts)
  python augur_cli.py alerts-test          # dry-run the on-call alert for machines above the band
  python augur_cli.py schema               # print the CSV columns a real export must have

Works on synthetic data out of the box; the moment there's a real maintenance/telemetry export with the
same columns, `train`/`predict` use it unchanged.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from augur import dataset, model, report, store, alerts, apps  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_HERE, "data")
HTML_OUT = os.path.join(DATA, "watchlist.html")


def _print_metrics(rep: dict) -> None:
    print("\nAUGUR model — honest evaluation (tested on machines held out of training)")
    print("=" * 74)
    print(f"  task            {rep['task']}")
    print(f"  data            {rep['n_rows']:,} readings · {rep['n_machines']} machines "
          f"· {rep['positive_rate']*100:.1f}% positive (imbalanced)")
    print(f"  ROC-AUC         {rep['roc_auc']}     PR-AUC {rep['pr_auc']}")
    print(f"  @thr={rep['threshold']}      precision {rep['precision']} · recall {rep['recall']} · F1 {rep['f1']}")
    print(f"  confusion       {rep['confusion_matrix']}  {rep['confusion_labels']}")
    lt = rep.get("lead_time", {})
    if lt:
        print(f"  ⏱ WARNING LEAD  caught {lt['warned_pct']}% of {lt['failures']} failures early · "
              f"median {lt['median_days_warning']} days' warning (max {lt['max_days_warning']})")
    md = rep.get("mode")
    if md:
        pc = " · ".join(f"{k} {v['recall']}" for k, v in md.get("per_class", {}).items())
        print(f"  🔩 FAILURE MODE named the right part {md['accuracy_on_failures']*100:.0f}% of the time "
              f"(per-part recall: {pc})")
    ru = rep.get("rul")
    if ru and ru.get("mae_days") is not None:
        print(f"  ⏳ RUL (learned) off by ~{ru['mae_days']} days on average (n={ru['n_eval']} real run-ups)")
    print("  top signals     " + ", ".join(f"{t['feature']}({t['importance']})"
                                            for t in rep["top_features"][:5]))


def _emit(scored, metrics):
    print(report.text_watchlist(scored))
    os.makedirs(DATA, exist_ok=True)
    with open(HTML_OUT, "w", encoding="utf-8") as fh:
        fh.write(report.html_dashboard(scored, metrics))
    print(f"HTML dashboard → {HTML_OUT}\n")


def cmd_demo(_):
    print("Generating a synthetic USPS sorter fleet (deterministic)…")
    df = dataset.generate(n_machines=40, days=400, seed=7)
    csv = os.path.join(DATA, "fleet_synthetic.csv")
    os.makedirs(DATA, exist_ok=True)
    df.to_csv(csv, index=False)
    print(f"  {len(df):,} daily readings across {df['machine_id'].nunique()} machines → {csv}")
    rep = model.train(df)
    _print_metrics(rep)
    scored = model.score(df)
    _emit(scored, rep)
    # seed the live store so the /watchlist service has data, and show what alerting WOULD do
    store.seed_from_dataframe(df)
    print(f"Live store seeded → {store.DB_PATH}  (start the feed: python augur_cli.py serve)")
    a = alerts.dispatch(scored, force=True)
    if a["flagged"]:
        who = ", ".join(x["name"] or "?" for x in a["would_notify"]) or "(no recipients configured)"
        print(f"🔔 ALERTS ({'DRY-RUN' if a['dry_run'] else 'SENT'}): {a['flagged']} machine(s) "
              f"[{', '.join(a['machines'])}] would page → {who}")
    else:
        print("🔔 ALERTS: nothing above the alert band on the latest day (quiet — that's good).")


def cmd_train(a):
    df = dataset.load_csv(a.csv)
    if "failed" not in df.columns:
        sys.exit("train needs a `failed` (0/1) column to build the label. Use `schema` to see the format.")
    _print_metrics(model.train(df))
    print("Saved model → models/augur_rf.joblib\n")


def cmd_predict(a):
    df = dataset.load_csv(a.csv)
    if not os.path.exists(model.MODEL_PATH):
        sys.exit("no trained model yet — run `demo` or `train --csv …` first.")
    import json
    meta = json.load(open(model.META_PATH)) if os.path.exists(model.META_PATH) else None
    _emit(model.score(df), meta)


def cmd_serve(a):
    import uvicorn
    print(f"🔧 AUGUR live feed on http://127.0.0.1:{a.port}  ·  POST /ingest · GET /watchlist(.html) · POST /alerts/run")
    uvicorn.run("augur.service:app", host="127.0.0.1", port=a.port, log_level="warning")


def cmd_ingest(a):
    df = dataset.load_csv(a.csv)
    n = store.ingest(df)
    print(f"ingested {n} readings into the live store → {store.DB_PATH}")


def cmd_alerts_test(_):
    if not os.path.exists(model.MODEL_PATH):
        sys.exit("no trained model — run `demo` or `train --csv …` first.")
    df = store.recent(days=45)
    if df.empty:
        sys.exit("live store is empty — run `demo` or `ingest --csv …` first.")
    res = alerts.dispatch(model.score(df), force=True)
    print("Alert dry-run:" if res["dry_run"] else "Alerts SENT:")
    print(f"  band ≥ {alerts.load_config().get('min_band','HIGH')} · {res['flagged']} machine(s): "
          f"{', '.join(res['machines']) or '(none)'}")
    for w in res["would_notify"]:
        print(f"  → would page {w['name']}: {', '.join(w['addrs']) or '(no address configured)'}")
    if res.get("note"):
        print("  note:", res["note"])


def cmd_benchmark(a):
    from augur import benchmark
    path = a.cmapss or benchmark.DEFAULT_CMAPSS
    if not os.path.exists(path):
        sys.exit(f"C-MAPSS file not found: {path}\nDownload train_FD001.txt (NASA C-MAPSS) into data/.")
    print(f"Validating AUGUR on REAL benchmark data: {os.path.basename(path)} …")
    print(benchmark.report_text(benchmark.run(path)))


def cmd_apps(_):
    print("Generating a synthetic APPS control-room fleet (Site · Serial# · Side)…")
    res = apps.run()
    print(f"  {res['units']} APPS sides scored · {res['critical']} predicted CRITICAL · {res['high']} HIGH")
    print("\nAUGUR — APPS Predicted At-Risk Board (worst first):")
    print("  " + "-" * 74)
    for r in res["top"]:
        print(f"  {r['band']:<9} {r['risk']*100:5.1f}%  RUL {str(r['rul_days']):>4}d  "
              f"{r['likely_fru']:<24} {r['machine_id']}")
    print(f"\nAPPS board HTML → {res['html']}")
    print("  (tabbed with the fleet watchlist — open watchlist.html and click the APPS Board tab)\n")


def cmd_schema(_):
    print("AUGUR CSV schema (one row per machine per day):\n")
    print("  required : " + ", ".join(dataset.REQUIRED_COLUMNS))
    print("  for train: add `failed` (1 on a day the machine had a hard failure, else 0)")
    print("\n  sensors  : " + ", ".join(dataset.SENSOR_COLUMNS))
    print("  static   : " + ", ".join(dataset.STATIC_COLUMNS))
    print(f"\n  label is derived automatically: fail_within_{dataset.HORIZON_DAYS}d "
          "(a failure in the next 14 days).")


def main():
    ap = argparse.ArgumentParser(description="AUGUR — predictive-maintenance AI")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo").set_defaults(fn=cmd_demo)
    for name in ("train", "predict", "ingest"):
        p = sub.add_parser(name)
        p.add_argument("--csv", required=True)
        p.set_defaults(fn={"train": cmd_train, "predict": cmd_predict, "ingest": cmd_ingest}[name])
    ps = sub.add_parser("serve"); ps.add_argument("--port", type=int, default=8920)
    ps.set_defaults(fn=cmd_serve)
    sub.add_parser("apps").set_defaults(fn=cmd_apps)
    sub.add_parser("alerts-test").set_defaults(fn=cmd_alerts_test)
    pb = sub.add_parser("benchmark"); pb.add_argument("--cmapss", default=None)
    pb.set_defaults(fn=cmd_benchmark)
    sub.add_parser("schema").set_defaults(fn=cmd_schema)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
