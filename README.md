# In-Game Market Controller (Fraud & Manipulation Detection)

<!-- Badges -->
![CI](https://github.com/Balghi/online-ingame-market-controller/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://online-ingame-market-controller-demo.streamlit.app)

> 📋 **[View Changelog](CHANGELOG.md)** - See what's new in each release

## 🌐 Live Demo

- Preloaded with demo runs so it loads instantly.
- Tabs: **Single run**, **Compare runs**, **Festival summary**.
- AI section includes plain-English reasons for top features per suspicious account/day.

A lightweight, **QA-driven** framework for detecting **market manipulation** in online games with player-driven economies.  
It simulates a live economy, injects realistic bad behaviors, and detects them via **explainable rules** plus a small **unsupervised AI** layer.  
Outputs are **reproducible CSV artifacts** suitable for dashboards, triage, and regression testing.

---

## ✨ Highlights

- **Fraud behaviors covered**
  - **Wash trading** (small rings, dense clusters, optional cycle motifs)
  - **Mule transfers** (cheap transfers to new/throwaway accounts)
  - **Collusive rapid flips** (buy→relist/resell quickly at markup)
- **Detectors**
  - **Explainable rules**: price anomalies, concentration windows, flip heuristics, mule age/price checks
  - **Unsupervised pass**: per-account/day features + Isolation Forest (low contamination, explainable top features)
- **QA first**
  - Golden run, metamorphic tests, property checks
  - Seed sweeps (festival ON/OFF) with per-type precision/recall
  - Deterministic artifacts for regression (CSV based)
- **Portable**: pure Python, no external services required

---

## 📦 Repo Structure

```
.
├─ sim/                # world, market simulation, fraud injectors
├─ detect/             # rules + unsupervised detectors
├─ features/           # feature builders (account/day)
├─ eval/               # evaluation helpers
├─ scripts/            # CLI runners
├─ notebook.py         # one-click multi-seed sweep (festival ON/OFF)
├─ data/
│  ├─ runs/            # per-run artifacts (flags, labels, metrics…)
│  └─ sweeps/          # sweep summaries (summary.csv)
├─ tests/              # golden, property, metamorphic, sanity tests
├─ docs/               # test plan, procedures, reporting templates
├─ app/                # Streamlit web application
├─ CHANGELOG.md        # version history and release notes
└─ README.md           # this file
```

---

## 🚀 Quickstart

```bash
# 1) Set up environment (Python 3.10+ recommended)
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2) Run a multi-seed sweep with/without festival mode
python notebook.py

# 3) Explore results
open data/sweeps/summary.csv
# …and per-run folders under data/runs/fest_on_seedXXXX / fest_off_seedXXXX
# 4) View visualizations on app
streamlit run app/streamlit_app.py
```

**Key entrypoint:** `notebook.py`  
Runs 10 scenarios (5 seeds with festival ON, 5 with OFF), simulates markets, injects manipulations, runs detectors, and writes evaluation/triage CSVs.

---

## 🧩 How It Works

1. **Economy simulation**  
   - Generates players, items, and hourly market activity (listings/trades)  
   - Optional **“Festival Week”** global price lift to test price-aware detectors  
   - Optional **busy-but-benign clusters** (high concentration, fair prices) to guard against false positives

2. **Fraud injection**  
   - Wash rings, mule transfers, and collusive rapid flips (configurable intensity/timings)

3. **Detection**  
   - **Rules**  
     - **UNDER/OVERPRICED** (robust medians/MAD, item-aware, festival-resilient)  
     - **WASH_CONC** (dense trade concentration among few accounts in a window)  
     - **RAPID_FLIP** (adaptive time windows + two-leg detection)  
     - **MULE_RULE** (piecewise age + price reference using rolling medians & p95)  
   - **Unsupervised**  
     - Isolation Forest on per-account/day features (trade volume, distinct counterparties, concentration, hour-entropy, price-z, flips), **low contamination** and **top-feature explanations** so it’s not a black box

4. **Evaluation & reporting**  
   - Precision/Recall **overall** and **per fraud type**  
   - **Rule contribution** (TPs by rule)  
   - **Ablation (unique TPs)** by rule family vs AI  
   - **Miss audit** (per-run CSV listing missed trades with quick context)  
   - All outputs are written as CSVs for easy downstream visualization

---

## 📊 Outputs (per run)

Each run directory (e.g., `data/runs/fest_on_seed1337/`) contains:

- `labels.csv` – ground truth from injectors  
- `flags.csv` – deduplicated detection results (entity_type, entity_id, code, detail, risk)  
- `metrics_flat.csv` – P/R overall & by fraud_type  
- `rule_contribution.csv` – TPs by rule code  
- `ablation_unique_tp.csv` – unique vs overlapping contributions by rule family (Rules-Price, Rules-Wash, Rules-Flip, Rules-Mule, AI)  
- `missed_by_type.csv` / `miss_audit.csv` – if any FNs, concise drill-down  
- `ai_account_day_explain.csv` – suspicious account-days with **top features** (names + z-scores)  
- (optional) `timings.csv` – stage timings & throughput (events/sec)

A sweep summary is written to `data/sweeps/summary.csv` with rows per (seed × festival_on × fraud_type).

---

## ⚙️ Configuration

Adjust thresholds centrally in `notebook.py` (or expose via CLI):

- **UNDER/OVER**: `UNDER_OVER_K` (robust z threshold)  
- **WASH_CONC**: window, top-k share, min trades, max group size  
- **RAPID_FLIP**: adaptive base window, slow window extra, min markups  
- **MULE_RULE**: price percentage vs rolling median, age thresholds, high-value quantile  
- **AI**: Isolation Forest `contamination` (default **2%**)

Festival settings: start, duration, amplitude.  
Fraud intensity: ring size, flip pairs, delay distribution, etc.

---

## ✅ QA Principles in Practice

- **Structured testing**  
  - **Golden snapshot** (fixed seed) to catch regressions in `flags.csv` IDs  
  - **Metamorphic tests**: time-shift & price-scale invariance  
  - **Property tests**: non-negative values, no orphan trades/listings, conservation where expected

- **Regression safety**  
  - Multi-seed sweeps (festival ON/OFF) with precision/recall tracking per type  
  - Deterministic seeds & CSV artifacts for reproducibility

- **Explainability & triage**  
  - Human-readable rule reasons (`UNDERPRICED 48%`, `WASH_CONC cluster`, `RAPID_FLIP resale 18m`)  
  - AI layer includes **top features** per suspicious account/day

- **Task & workflow management**  
  - Clear procedures to add new behaviors/rules/tests  
  - CSV outputs designed for dashboards (Streamlit/Grafana/Sheets)  
  - Optional perf guardrails (`timings.csv`, events/sec)

---

## 🧪 Tests

This repo ships a small but meaningful pytest suite that exercises rules, AI features, and QA workflows:

| File | What it checks |
|---|---|
| `tests/test_golden_seed1337.py` | **Golden snapshot** of `flags.csv` trade IDs for a known run (regression guard). |
| `tests/test_metamorphic_time_shift.py` | **Metamorphic**: decisions are invariant under a uniform time shift. |
| `tests/test_property_basic.py` | **Property tests** on generated data (e.g., schema, non-negative prices/qty). |
| `tests/test_ablation_sanity.py` | **Ablation sanity**: AI unique TPs within reasonable bounds vs rules. |
| `tests/test_rule_rapid_flip_both_legs.py` | **Rapid flip rule** flags **both legs** of a collusive flip. |
| `tests/test_rule_mule_piecewise_age.py` | **Mule rule**: cheapness + piecewise age (≤14d or ≤30d on high-value items). |
| `tests/test_rule_wash_concentration.py` | **Wash concentration**: flags dense 3-player ring; spares uniform traffic. |
| `tests/test_ai_explainability.py` | **AI layer**: IsolationForest marks something with high contamination; explain columns present. |
| `tests/test_rule_price_baselines.py` | **Price baselines**: UNDER/OVERPRICED thresholds around rolling/global medians. |

Run the whole suite:

```bash
pytest -q
```

> Notes: tests read artifacts from a run directory. By default they look at `data/runs/fest_on_seed1337`. Set `RUN_DIR=/path/to/run` to override.

---

## 🛠️ Extending

- **Add a new injector**: implement in `sim/fraud.py`, add labels, and wire into `scripts/run_with_fraud.py` / `notebook.py`.  
- **Add a new rule**: implement in `detect/rules.py` with signature `df -> flags_df`, include a clear `code` and human-readable `detail`.  
- **Add features to AI**: update `features/accounts.py`; `ai_account_day_explain.csv` will include them in top-feature explanations automatically.

---

## 📈 Example (typical sweep results)

- **Precision**: ~**1.00** (designed to minimize false positives)  
- **Recall** (avg across seeds): **0.98–0.99 overall**, with slight, explainable variance  
- **Per-type**: high recall for **wash** and **mule**, strong but honest recall for **collusive flips** (depends on window & markup thresholds)

> Philosophy: prioritize player trust (precision) while keeping recall high and tunable. All signals are explainable.

---

## 📝 License

MIT — see `LICENSE`.

---

## 🙋‍♀️ Why this is useful to game teams

- Gives **actionable, explainable signals** to catch manipulation in player markets.  
- Ships with a **QA workflow**: tests, sweeps, regression artifacts, and documentation.  
- Easy to integrate: consumers can read the CSVs or call the detection modules directly.  
- Designed to evolve: add new patterns, rules, and features as your economy and threat models grow.
