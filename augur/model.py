"""
AUGUR — the models. Three heads, trained together on ONE honest grouped split (by machine, so every
score is measured on equipment the model never trained on):

  1. RISK       — "will this machine fail within HORIZON_DAYS?" (binary; the watchlist ranking).
  2. MODE       — "which component fails first — bearing, belt, or motor?" (multiclass; per-mode models).
  3. RUL        — "how many days until the next failure?" (regression; the true remaining-life estimate,
                  replacing the old heuristic).

Evaluation is imbalance-aware (ROC-AUC / PR-AUC, not accuracy), plus a warning-lead-time backtest, a
mode-accuracy measured only on machines that actually failed, and RUL error (MAE) on the machines that
actually had a failure ahead.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, confusion_matrix, mean_absolute_error,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
import joblib

from . import features as F
from .dataset import HORIZON_DAYS, CENSOR_DAYS, FAILURE_MODES

_HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(_HERE, "..", "models", "augur_rf.joblib")
META_PATH = os.path.join(_HERE, "..", "models", "augur_meta.json")
THRESHOLD = 0.30   # operating point: flag at 30% risk (recall-leaning — a missed failure costs more)


def _clf():
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                   class_weight="balanced", n_jobs=-1, random_state=42))])


def _reg():
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("rf", RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                                  n_jobs=-1, random_state=42))])


def train(df: pd.DataFrame, save: bool = True) -> dict:
    """Fit all three heads on grouped-held-out machines; return an honest metrics report; optionally persist."""
    fdf, feat_cols = F.build_features(df)
    X = fdf[feat_cols].to_numpy()
    y = fdf["fail_within_h"].to_numpy()
    groups = fdf["machine_id"].to_numpy()
    has_mode = "mode_within_h" in fdf.columns
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(X, y, groups))

    # 1. RISK
    risk = _clf(); risk.fit(X[tr], y[tr])
    prob = risk.predict_proba(X[te])[:, 1]
    pred = (prob >= THRESHOLD).astype(int)
    roc = roc_auc_score(y[te], prob) if len(set(y[te])) > 1 else float("nan")
    pr = average_precision_score(y[te], prob) if len(set(y[te])) > 1 else float("nan")
    p, r, f1, _ = precision_recall_fscore_support(y[te], pred, average="binary", zero_division=0)
    cm = confusion_matrix(y[te], pred).tolist()
    lead = _lead_time(fdf.iloc[te].assign(_prob=prob))
    importances = sorted(zip(feat_cols, risk.named_steps["rf"].feature_importances_),
                         key=lambda t: -t[1])[:10]

    # 2. MODE (multiclass) — trained on all rows; scored on the test machines that truly failed soon
    mode_report = None
    if has_mode:
        ym = fdf["mode_within_h"].to_numpy()
        mode = _clf(); mode.fit(X[tr], ym[tr])
        mpred = mode.predict(X[te])
        real = ym[te] != "none"                    # rows where a real failure is coming
        acc = float((mpred[real] == ym[te][real]).mean()) if real.any() else float("nan")
        classes = list(mode.named_steps["rf"].classes_)
        per_class = {}
        for c in FAILURE_MODES:
            sel = ym[te] == c
            if sel.any():
                per_class[c] = {"n": int(sel.sum()),
                                "recall": round(float((mpred[sel] == c).mean()), 3)}
        mode_report = {"accuracy_on_failures": round(acc, 3), "classes": classes, "per_class": per_class}
    else:
        mode = None

    # 3. RUL (regression) — predict days-to-failure; MAE measured where a failure actually lies ahead
    yd = fdf["days_to_failure"].to_numpy()
    rul = _reg(); rul.fit(X[tr], yd[tr])
    dpred = rul.predict(X[te])
    unc = yd[te] < CENSOR_DAYS                      # "uncensored" — a real failure within the window
    rul_report = {"mae_days": round(float(mean_absolute_error(yd[te][unc], dpred[unc])), 2)
                  if unc.any() else None, "n_eval": int(unc.sum())}

    report = {
        "task": f"fail_within_{HORIZON_DAYS}d", "n_rows": int(len(y)),
        "n_machines": int(len(set(groups))), "positive_rate": round(float(y.mean()), 4),
        "test_machines": int(len(set(groups[te]))), "threshold": THRESHOLD,
        "roc_auc": round(float(roc), 4), "pr_auc": round(float(pr), 4),
        "precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f1), 4),
        "confusion_matrix": cm, "confusion_labels": "[[TN, FP], [FN, TP]]", "lead_time": lead,
        "mode": mode_report, "rul": rul_report,
        "top_features": [{"feature": f, "importance": round(float(i), 4)} for f, i in importances],
    }

    if save:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        risk.fit(X, y)                              # refit on all data for deployment
        if mode is not None:
            mode.fit(X, fdf["mode_within_h"].to_numpy())
        rul.fit(X, yd)
        joblib.dump({"risk": risk, "mode": mode, "rul": rul, "features": feat_cols}, MODEL_PATH)
        json.dump(report, open(META_PATH, "w"), indent=2)
    return report


def load():
    b = joblib.load(MODEL_PATH)
    return b


def _band(prob: float) -> str:
    return "CRITICAL" if prob >= 0.6 else "HIGH" if prob >= THRESHOLD else \
           "WATCH" if prob >= 0.12 else "OK"


def _lead_time(scored: pd.DataFrame, horizon: int = HORIZON_DAYS) -> dict:
    leads, caught, total = [], 0, 0
    for _, g in scored.sort_values("date").groupby("machine_id", sort=False):
        g = g.reset_index(drop=True)
        dates, prob = g["date"].to_numpy(), g["_prob"].to_numpy()
        for i in np.where(g["failed"].to_numpy() == 1)[0]:
            lo = max(0, i - horizon)
            total += 1
            hits = np.where(prob[lo:i] >= THRESHOLD)[0]
            if len(hits):
                caught += 1
                leads.append(int((dates[i] - dates[lo + hits[0]]) / np.timedelta64(1, "D")))
    return {"failures": int(total), "warned_pct": round(100 * caught / total, 1) if total else 0.0,
            "median_days_warning": int(np.median(leads)) if leads else 0,
            "max_days_warning": int(np.max(leads)) if leads else 0}


def _fmt_rul(days: float) -> int | str:
    d = int(round(days))
    return ">60" if d >= CENSOR_DAYS - 2 else max(0, d)


def score(df: pd.DataFrame, latest_only: bool = True) -> pd.DataFrame:
    """Score readings → watchlist. Adds failure_probability, health_score, risk_band, likely_mode,
    mode_confidence, and rul_days (learned regression). With latest_only, one row per machine (latest)."""
    b = load()
    fdf, _ = F.build_features(df)
    if latest_only:
        fdf = fdf.sort_values("date").groupby("machine_id", as_index=False).tail(1)
    Xn = fdf[b["features"]].to_numpy()
    prob = b["risk"].predict_proba(Xn)[:, 1]

    out = fdf[["machine_id", "machine_type", "date"]].copy()
    out["failure_probability"] = np.round(prob, 4)
    out["health_score"] = np.round(100 * (1 - prob)).astype(int)
    out["risk_band"] = [_band(p) for p in prob]

    if b.get("mode") is not None:
        mp = b["mode"].predict_proba(Xn)
        classes = list(b["mode"].named_steps["rf"].classes_)
        fail_ix = [i for i, c in enumerate(classes) if c in FAILURE_MODES]
        sub = mp[:, fail_ix]
        best = sub.argmax(axis=1)
        out["likely_mode"] = [classes[fail_ix[j]] for j in best]
        out["mode_confidence"] = np.round(sub.max(axis=1), 3)
    else:
        out["likely_mode"] = "—"; out["mode_confidence"] = 0.0

    if b.get("rul") is not None:
        out["rul_days"] = [_fmt_rul(x) for x in b["rul"].predict(Xn)]
    else:
        out["rul_days"] = ">60"
    return out.sort_values("failure_probability", ascending=False).reset_index(drop=True)
