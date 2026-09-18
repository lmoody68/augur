"""
AUGUR — real-data validation on the NASA C-MAPSS turbofan benchmark.

C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) is NASA's public run-to-failure dataset
and the standard benchmark for remaining-useful-life prediction: 100 engines, each logged every cycle
from healthy until it fails. It is high-fidelity SIMULATION (be honest about that in interviews — it is
not a physical rig), but it's real, published, peer-used data with true failure times — a giant step up
from our own synthetic fleet, and it uses AUGUR's *exact* methodology unchanged:

  • same rolling-trend features (now column-agnostic),
  • same grouped split — trained on some engines, scored on engines it has NEVER seen,
  • same honest metrics + warning-lead-time backtest, plus RUL error in cycles.

This is what proves the AUGUR pipeline generalizes off the synthetic data — and the same adapter pattern
is how a real maintenance export will drop in.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (average_precision_score, mean_absolute_error,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from . import features as F, dataset, model

_HERE = os.path.dirname(__file__)
DEFAULT_CMAPSS = os.path.join(_HERE, "..", "data", "cmaps_train_FD001.txt")

SENSORS = [f"s{i}" for i in range(1, 22)]                    # 21 turbofan sensors
STATIC = ["cycle"]                                          # usage/age proxy
# the informative sensors (the rest are near-constant in FD001) — used for the rolling trend features
TREND = ["s2", "s3", "s4", "s7", "s11", "s12", "s15", "s17", "s20", "s21"]
HORIZON = 30    # predict failure within the next 30 cycles
CENSOR = 130    # RUL measured/emitted up to 130 cycles out


def adapt_cmapss(path: str = DEFAULT_CMAPSS) -> pd.DataFrame:
    """Load C-MAPSS FD001 into AUGUR's tidy schema (one engine = one 'machine', one cycle = one 'day')."""
    cols = ["unit", "cycle", "op1", "op2", "op3"] + SENSORS
    df = pd.read_csv(path, sep=r"\s+", header=None, names=cols)
    df["machine_id"] = "ENGINE-" + df["unit"].astype(int).astype(str).str.zfill(3)
    df["machine_type"] = "CMAPSS"
    df["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(df["cycle"] - 1, unit="D")
    # a failure event on each engine's LAST cycle (that's when it ran to failure)
    last = df.groupby("unit")["cycle"].transform("max")
    df["failed"] = (df["cycle"] == last).astype(int)
    df["failure_mode"] = ""
    return dataset.add_label(df, horizon=HORIZON, censor=CENSOR)


def run(path: str = DEFAULT_CMAPSS) -> dict:
    """Train + honestly evaluate AUGUR's risk + RUL models on C-MAPSS. Returns a metrics report."""
    df = adapt_cmapss(path)
    fdf, feat = F.build_features(df, sensors=SENSORS, static=STATIC, trend=TREND)
    X = fdf[feat].to_numpy()
    y = fdf["fail_within_h"].to_numpy()
    yd = fdf["days_to_failure"].to_numpy()
    groups = fdf["machine_id"].to_numpy()
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(X, y, groups))

    clf = Pipeline([("i", SimpleImputer(strategy="median")),
                    ("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                  class_weight="balanced", n_jobs=-1, random_state=42))])
    clf.fit(X[tr], y[tr])
    prob = clf.predict_proba(X[te])[:, 1]
    pred = (prob >= model.THRESHOLD).astype(int)
    roc = roc_auc_score(y[te], prob)
    pr = average_precision_score(y[te], prob)
    p, r, f1, _ = precision_recall_fscore_support(y[te], pred, average="binary", zero_division=0)
    lead = model._lead_time(fdf.iloc[te].assign(_prob=prob), horizon=HORIZON)

    reg = Pipeline([("i", SimpleImputer(strategy="median")),
                    ("rf", RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                                 n_jobs=-1, random_state=42))])
    reg.fit(X[tr], yd[tr])
    dpred = reg.predict(X[te])
    unc = yd[te] < CENSOR
    mae = mean_absolute_error(yd[te][unc], dpred[unc])

    imp = sorted(zip(feat, clf.named_steps["rf"].feature_importances_), key=lambda t: -t[1])[:6]
    # one held-out engine's risk trajectory into failure (the compelling picture)
    te_engines = pd.unique(groups[te])
    demo = fdf.iloc[te][fdf.iloc[te]["machine_id"] == te_engines[0]].copy()
    demo = demo.assign(_prob=clf.predict_proba(demo[feat].to_numpy())[:, 1])

    return {
        "dataset": "NASA C-MAPSS FD001 (turbofan run-to-failure; simulation benchmark)",
        "engines": int(len(set(groups))), "rows": int(len(y)), "test_engines": int(len(set(groups[te]))),
        "horizon_cycles": HORIZON, "positive_rate": round(float(y.mean()), 3),
        "roc_auc": round(float(roc), 3), "pr_auc": round(float(pr), 3),
        "precision": round(float(p), 3), "recall": round(float(r), 3), "f1": round(float(f1), 3),
        "lead_time": lead, "rul_mae_cycles": round(float(mae), 1), "rul_n_eval": int(unc.sum()),
        "top_features": [f"{n}({round(float(i),3)})" for n, i in imp],
        "demo_engine": str(te_engines[0]),
        "demo_trajectory": [(int(c), round(float(pp), 3)) for c, pp in
                            zip(demo["cycle"].to_numpy()[-12:], demo["_prob"].to_numpy()[-12:])],
    }


def report_text(r: dict) -> str:
    L = ["", "=" * 74, "AUGUR on REAL data — " + r["dataset"], "=" * 74,
         f"  {r['engines']} engines · {r['rows']:,} run-to-failure readings "
         f"· tested on {r['test_engines']} engines never seen in training",
         f"  task            fail within {r['horizon_cycles']} cycles  ({r['positive_rate']*100:.0f}% positive)",
         f"  ROC-AUC         {r['roc_auc']}     PR-AUC {r['pr_auc']}",
         f"  @thr={model.THRESHOLD}      precision {r['precision']} · recall {r['recall']} · F1 {r['f1']}",
         f"  ⏱ WARNING LEAD  caught {r['lead_time']['warned_pct']}% of {r['lead_time']['failures']} engine "
         f"failures early · median {r['lead_time']['median_days_warning']} cycles "
         f"(max {r['lead_time']['max_days_warning']})",
         f"  ⏳ RUL (learned) off by ~{r['rul_mae_cycles']} cycles on average (n={r['rul_n_eval']})",
         f"  top signals     {', '.join(r['top_features'])}", "",
         f"  Risk climbing into failure — held-out engine {r['demo_engine']} (last 12 cycles):"]
    for c, pp in r["demo_trajectory"]:
        bar = "█" * int(pp * 40)
        L.append(f"    cycle {c:>4}  risk {pp*100:5.1f}%  {bar}")
    L += ["", "  Same pipeline, same honest split as the synthetic demo — proven on real benchmark data.", ""]
    return "\n".join(L)
