# Changelog

All notable changes to this project will be documented in this file.  
This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and Semantic Versioning.

---

## [0.1.0] - 2025-09-23

### Added
- **Fraud detectors**  
  - UNDER/OVER-priced trades  
  - Rapid flips (both legs of collusion)  
  - Mule transfers (piecewise account age logic)  
  - Wash trading (heuristic + concentration rules)  

- **AI anomaly detection**  
  - Isolation Forest on account/day features  
  - Plain-English explanations for flagged anomalies  

- **Simulation & injectors**  
  - Synthetic players, items, trades  
  - Configurable “festival” market events  
  - Fraud injectors for collusion, mules, wash trading  
  - Benign “busy trader” injectors for QA stress tests  

- **Testing suite (9 total)**  
  - Golden snapshot regression  
  - Metamorphic time-shift invariance  
  - Property-based data sanity  
  - Ablation sanity (AI unique detections)  
  - Unit tests for rapid flip, mule transfers, wash concentration, price baselines, AI explainability  

- **Continuous Integration**  
  - GitHub Actions running pytest on push/PR  

- **Streamlit demo app**  
  - Single-run analysis with KPIs and rule contributions  
  - Compare runs across seeds and festival ON/OFF  
  - Summary view with aggregate metrics  

- **Documentation & artifacts**  
  - README with badges, usage, and test overview  
  - One-page PDF project overview (`docs/project_onepager.pdf`)  
  - Report template (`docs/report_template.md`)  
  - Curated demo runs (`fest_on_seed21`, `fest_off_seed909`) for instant exploration  

### Changed
- Tuned thresholds for precision-first detection  
  - Achieved ≈1.00 precision and 0.97–0.99 recall across multiple seeds  
- Expanded README with demo link, CI badge, and test breakdown  

### Fixed
- Mule rule extended-age logic (applies only for high-value trades)  
- Isolation Forest fitting bug (explicit `.fit()` before scoring)  
- Streamlit explainability: added buyer/seller context and clearer feature text  

---

## Unreleased
- Extended fraud patterns: cross-item wash rings, K4+ cycle detection  
- Streamlit enhancements: export audit CSVs and PDF snapshots  
- Dockerfile for portable deployment
