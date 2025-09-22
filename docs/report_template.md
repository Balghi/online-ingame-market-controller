# Weekly QA Report — In‑Game Market Controller

**Week of:** <YYYY‑MM‑DD>  
**Owner:** <name>  
**Commit / Tag:** <hash / tag>

---

## 1) Executive Summary (2–3 bullets)
- e.g., “Precision held at 1.00; overall recall 0.985 across 10 seeds.”
- e.g., “Flip recall increased +3pp with adaptive slow‑window; no precision impact.”
- e.g., “No new FPs on busy‑but‑benign clusters.”

## 2) KPIs (Seed Sweep)
| Mode       | Precision (mean±std) | Recall (mean±std) |
|------------|-----------------------|-------------------|
| Festival ON| <P_mean> ± <P_std>    | <R_mean> ± <R_std>|
| Festival OFF| <P_mean> ± <P_std>   | <R_mean> ± <R_std>|

### Per‑Type Recall
| Fraud Type       | Recall (mean) |
|------------------|---------------|
| wash_trading     | <val>         |
| mule_transfer    | <val>         |
| collusive_flip   | <val>         |

## 3) Rule Contribution & Ablation
- **Rule contribution (TPs):** list top 3 (e.g., WASH_CONC, UNDERPRICED, RAPID_FLIP)
- **Ablation (unique TPs):** Rules vs AI — note any shifts from last week

## 4) Miss Audit (If Any)
- Count by type + top reasons (from `miss_audit.csv`):
  - flips finalizing > base window
  - mule price just above threshold
  - cross‑item ring outside item‑local windows
- Actions taken / proposed

## 5) False Positive Guard
- Busy‑but‑benign cluster FPs: <count>
- Any manual spot checks: <notes>

## 6) Performance
- `events_per_s` and total runtime (min/mean/max across runs)
- Any regressions vs last week? Actions if >20% drift

## 7) Changes This Week
- Thresholds / injectors / rules updated (link to PR/commit)
- Golden snapshot updated? Y/N (why)

## 8) Next Week Focus
- e.g., cycle K4 motif (optional), cross‑item rings, UI polish, CI artifacts to S3/GH‑pages
