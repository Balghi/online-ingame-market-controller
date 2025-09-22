# Procedures — Extending & Maintaining the Market Controller

This document describes how to add new behaviors, detectors, tests, and how to run routine workflows.

## 1) Add a New Fraud Injector
**Goal:** simulate a new manipulation pattern and produce labels.
1. Create a function in `sim/fraud.py` returning `(listings, trades, labels)` where `labels` has:
   - `entity_type` = "trade"
   - `entity_id` = trade_id
   - `is_fraud` ∈ {0,1}
   - `fraud_type` ∈ {"wash_trading","mule_transfer","collusive_flip",...}
2. Wire it into `scripts/run_with_fraud.py` or `notebook.py`.
3. Add toggles/parameters (count, window sizes, price offsets, delays).
4. Add a unit test for label counts or a cheap invariant in `tests/`.

## 2) Add a New Rule
**Goal:** detect a pattern with explainable logic.
1. Implement in `detect/rules.py` a function that returns a DataFrame with columns:
   - `entity_id` (usually `trade_id`)
   - `code` (short, e.g., `WASH_CONC`, `UNDERPRICED`)
   - `detail` (short human‑readable reason)
2. Ensure time parsing and duplicates are handled (`drop_duplicates("entity_id")`).
3. Add to the union in the runner, e.g.:
   ```python
   rule_flags = pd.concat([
       to_flags_df(ou, rf),
       mule.assign(entity_type="trade", risk=0.9),
       your_rule.assign(entity_type="trade", risk=0.9),
   ]).drop_duplicates(["entity_type","entity_id"])
   ```
4. Add a unit test around edge cases.

## 3) Tune Thresholds
- Expose parameters in `notebook.py` (central knobs).
- Prefer conservative defaults to protect precision.
- Use the sweep output to compare before/after (commit CSVs if desired).

## 4) Update the Golden Snapshot
- Generate a stable baseline run (default: `data/runs/fest_on_seed1337`).
- Run:
  ```bash
  python -m scripts.update_golden --run_dir data/runs/fest_on_seed1337 --snapshot tests/golden_flags_trade_ids.json
  ```
- Commit the updated JSON alongside code changes.

## 5) Add Features to the AI Layer
- Extend `features/accounts.py` to compute new per‑account/day metrics.
- They will automatically appear in `ai_account_day_explain.csv` (top‑feature list).

## 6) Performance Tracking
- Ensure `timings.csv` is written per run (generation, injectors, rules, unsupervised, total).
- Compare `events_per_s` across branches; set CI guardrails.

## 7) CI Workflow (suggested)
- Step 1: install & lint
- Step 2: run tests
- Step 3: run a reduced sweep (e.g., 2 seeds ON + 2 OFF)
- Step 4: upload `data/sweeps/summary.csv` and one run folder as artifacts

## 8) Miss Audit & Triage
- Inspect `miss_audit.csv` (trade_id, type, price vs median, timestamps).
- Determine whether to:
  - **Adjust thresholds** (e.g., adaptive windows for flips)
  - **Add a micro‑rule** (e.g., K3/K4 cycle motif)
  - **Revisit injector realism**

## 9) Documenting Intentional Changes
- When thresholds or injectors change, add a line to `CHANGELOG.md` with:
  - What changed
  - Why (miss analysis, FP guard, performance)
  - Expected effect on P/R

## 10) Common Pitfalls
- Forgetting to `drop_duplicates` on flags
- Merging on the wrong key (`listing_id` vs `trade_id`)
- Missing `parse_dates=["ts"]` leading to string comparisons
- Treating AI scores as ground truth — they’re *signals*; keep contamination low (1–3%)

