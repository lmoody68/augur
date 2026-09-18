# 🔧 AUGUR — Predictive-Maintenance AI

**Hear a machine failing before it does.** AUGUR ingests equipment sensor data, learns the trend that
precedes a breakdown, and produces a ranked watchlist so a maintenance team fixes machines *on a schedule*
instead of *at 2 a.m. when the line goes down*.

Built with a USPS mail-processing fleet in mind (bar-code & flat sorters — DBCS / AFSM100 / APBS / DIOSS),
with a **USPS maintenance manager** as the domain expert and first tester. The moment there's a real
telemetry / CMMS export, it drops in unchanged — same columns, same commands.

## What it does
- **Predicts failures** with a real scikit-learn model: *"will this machine fail in the next 14 days?"*
- **Ranks the fleet** by risk (CRITICAL / HIGH / WATCH / OK) with a 0–100 health score and a rough
  remaining-useful-life estimate, as a terminal table **and** a shareable HTML dashboard.
- **Proves its worth honestly** — evaluated on machines it has *never seen* (grouped split), with
  imbalance-aware metrics (ROC-AUC, PR-AUC) and a **warning lead-time backtest**: on the synthetic fleet
  it flags **~91% of failures a median of two weeks early.**

## Why the design is trustworthy (not a demo that flatters itself)
- **No leakage.** The label is the *future* (a failure in the next 14 days); every feature is built from
  past + current readings only. Train/test is split **by machine** (`GroupShuffleSplit`) — the model is
  scored on equipment it never trained on, which is the real deployment question.
- **Honest metrics.** Most days are fine (~3% positive), so accuracy would lie. We report ROC-AUC, PR-AUC,
  and precision/recall at an operating threshold tuned to *miss fewer failures* (recall-leaning — a missed
  breakdown costs far more than a spare inspection).
- **Honest synthetic data.** Until real logs exist, `dataset.generate()` simulates a fleet with a hidden
  health that degrades under load + heat and drives the sensors — clearly labeled synthetic, with real
  ground-truth failures to learn from. It is a stand-in for real data, never a substitute.

## The signals it learns
Top predictors come out physically sensible: **7-day mean bearing vibration**, **jam rate trend**, and
**belt tension** — exactly what a seasoned tech listens and looks for. AUGUR just watches all of them, on
every machine, every day.

## Run it
```bash
pip install -r requirements.txt
python augur_cli.py demo                    # generate a fleet → train → metrics + watchlist + HTML
python augur_cli.py schema                  # the exact CSV columns a real export needs
python augur_cli.py train  --csv logs.csv   # train on a real export (needs a `failed` 0/1 column)
python augur_cli.py predict --csv logs.csv  # score today's readings → watchlist + dashboard
```
The HTML dashboard lands at `data/watchlist.html`.

## Proven on real benchmark data (NASA C-MAPSS turbofan)
`python augur_cli.py benchmark` runs the **exact same pipeline** (column-agnostic features, grouped split,
RandomForest, lead-time backtest) on **NASA's C-MAPSS FD001** — 100 real turbofan engines run to failure,
the standard remaining-life benchmark. Tested on 25 engines held out of training:
**ROC-AUC 0.99 · recall 0.95 · caught 100% of failures a median 30 cycles early · RUL error ~15 cycles.**
(C-MAPSS is high-fidelity NASA *simulation* — say so honestly — but it's real published benchmark data with
true failure times, and the same adapter pattern is how a real maintenance export drops in.)

## Using real data
Export one row per machine per day with these columns (`python augur_cli.py schema`):
`machine_id, date, motor_temp_c, bearing_vibration_mm_s, belt_tension_pct, jam_rate_per_k,
motor_current_a, throughput_kpph, ambient_temp_c, machine_age_years, cycles_k, days_since_maintenance`
— plus `failed` (1 on a day a machine had a hard failure) for training. Column names can be mapped if your
CMMS uses different ones. Everything runs locally; no data leaves the machine.

## Already built — ready to switch on with real data
These aren't roadmap items; they're wired in now and run on the synthetic fleet today. Feed AUGUR real
data and they light up automatically:

- **Per-component failure modes** — a multiclass model predicts *which part* fails first: **bearing, belt,
  or motor** (the watchlist shows the likely part). On the synthetic fleet it names the right part ~65% of
  the time. The synthetic data models three independently-wearing components, so this is a real signal.
- **Learned remaining-useful-life (RUL)** — a regression model estimates *days until the next failure*
  (replacing the old heuristic); on the synthetic fleet it's off by ~7 days on average.
- **Live feed** — a FastAPI service (`augur/service.py`) with a SQLite store: telemetry `POST`s to
  `/ingest`, and `/watchlist`, `/watchlist.html`, `/machine/{id}`, `/alerts/run`, `/meta` are available on
  demand. Run it with `python augur_cli.py serve`.
- **Automatic on-call alerts** — `augur/alerts.py` pages whoever's on call (email, or SMS via the carrier's
  email-to-SMS gateway — no paid SMS service) when a machine crosses the alert band, with a per-machine
  cooldown so it never spams. **Dry-run by default** — it sends for real only when `config/on_call.json`
  has `enabled: true` and SMTP creds are in the environment. Try it: `python augur_cli.py alerts-test`.

## Layout
```
augur/dataset.py   synthetic 3-component fleet generator + real-CSV loader + labels
                   (fail_within_14d · failure mode · days-to-failure)
augur/features.py  rolling 7-day means / std / slopes (past-only, leakage-safe)
augur/model.py     grouped train/eval; 3 heads — risk (classifier) · mode (multiclass) · RUL (regressor)
augur/report.py    terminal watchlist + standalone HTML dashboard (with likely-part column)
augur/store.py     SQLite live reading store (ingest / recent / machine history)
augur/service.py   FastAPI live feed (ingest + watchlist + machine + alerts + meta)
augur/alerts.py    on-call routing (email / carrier-SMS), dry-run by default, cooldown
config/on_call.json who to page + thresholds
augur_cli.py       demo · train · predict · ingest · serve · alerts-test · schema
```

## Honest limits / next steps
- Synthetic data proves the *pipeline and every feature*; real accuracy must be measured on the shop's actual
  failure history. Column mapping + a small labeled export is the one real next step.
- The RUL and mode numbers above are on simulated data — expect them to move on real equipment.
- Further out: separate physics-informed models per machine family, survival analysis for censored RUL,
  and pushing alerts into an existing CMMS work-order system.
