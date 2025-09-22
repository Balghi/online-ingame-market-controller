# Test Plan — In‑Game Market Controller

## 1. Purpose & Scope
This plan defines the testing strategy for the Market Controller, a framework that simulates game‑economy data, injects manipulation patterns, and detects them via explainable rules and an unsupervised layer. Scope covers functional correctness, regression safety, explainability, and performance at demo scale (≤ ~100k trades).

Out of scope: production data connectors, live deployment, auth/ACL.

## 2. Quality Objectives
- **Precision ≥ 0.99** (minimize false positives to avoid harming fair players)
- **Recall ≥ 0.90** averaged across seeds and modes (Festival ON/OFF)
- **Determinism**: identical outputs for identical seeds/config
- **Explainability**: human‑readable reasons for rule hits; top features for AI hits
- **Performance**: end‑to‑end processing ≤ 10s for a 100k‑trade run on a dev laptop (illustrative target)

## 3. Test Types & Coverage

### 3.1 Unit & Component Tests
- **Rules**: boundary conditions for UNDER/OVER, WASH_CONC windows, RAPID_FLIP two‑leg logic, MULE_RULE thresholds
- **Features**: account/day feature construction (counts, distincts, entropy)
- **Unsupervised**: stable outputs on synthetic feature grids; no crashes on NaN/Inf

### 3.2 Property Tests
- Non‑negative prices and quantities
- No orphan trades (every trade references an existing listing)
- Time consistency: listing.ts ≤ trade.ts ≤ listing.expires_at (if provided)

### 3.3 Metamorphic Tests
- **Time‑shift invariance**: decisions unchanged when all timestamps shift by +24h
- **Price‑scale robustness** (optional): multiplying all prices by a constant should not change classification for z‑based rules

### 3.4 Golden Snapshot (Regression)
- Lock `flags.csv` (trade‑ID set) for `fest_on_seed1337`
- Fail when the set changes unexpectedly; update snapshot via a documented script upon intentional tuning

### 3.5 Seed Sweeps
- Seeds: 5 with Festival ON, 5 OFF
- Metrics captured per run and aggregated:
  - Overall Precision/Recall
  - Per‑type Precision/Recall: `wash_trading`, `mule_transfer`, `collusive_flip`
  - Rule contribution (TPs by rule)
  - Ablation (unique TPs by family)
  - Miss audit (if any)

### 3.6 Performance Checks
- Record timings per stage (generation, injectors, rules, unsupervised)
- Report `events_per_s` throughput
- Set a guardrail threshold and track drift in CI

## 4. Environments
- Python 3.10+ in a virtualenv
- Dependencies pinned in `requirements.txt`
- CI: GitHub Actions (Linux, Python matrix optional)

## 5. Data Sets
- Synthetic data generated on‑the‑fly by simulation
- Fraud injectors (wash/mule/flip) with configurable intensity and timings
- “Festival Week” global price‑lift mode
- “Busy‑but‑benign” clusters to probe wash precision

## 6. Entry/Exit Criteria
**Entry**
- Environment set up
- Dependencies installed
- Baseline seed run available (for golden)

**Exit**
- Unit/property/metamorphic tests pass
- Golden snapshot matches or was intentionally updated with approval
- Seed sweeps meet targets (Precision ≥ 0.99; Recall ≥ 0.90 overall; no unexpected FP bursts)
- Performance within stated goal

## 7. Risks & Mitigations
- **Overfitting to injectors** → Keep injectors configurable (delays, group sizes, price proximity). Track per‑type recall across seeds.
- **Black‑box AI** → Publish top‑feature explanations and keep contamination low (1–3%).
- **False positives on benign clusters** → Maintain specific FP guards, include non‑fraud concentration scenarios.
- **Performance regressions** → Timings CSV and CI thresholds.

## 8. Triage & Reporting
- All runs emit: `metrics_flat.csv`, `rule_contribution.csv`, `ablation_unique_tp.csv`, `miss_audit.csv` (if any), `ai_account_day_explain.csv`, and optionally `timings.csv`.
- Weekly report (template in `report_template.md`) summarises KPIs, deltas, and next actions.

## 9. Responsibilities
- Test ownership: repo maintainers
- Golden updates: PR with reviewer approval
- CI maintenance: repo maintainers

## 10. References
- README for quickstart & structure
- `procedures.md` for adding new behaviors, detectors, and tests
