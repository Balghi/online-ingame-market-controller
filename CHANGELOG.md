# Changelog

All notable changes to the In-Game Market Controller will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Features in development

### Changed
- Planned improvements

### Fixed
- Bug fixes in progress

## [1.0.0] - 2024-09-23

### Added
- **Streamlit Web Application**: Interactive demo with single run analysis, comparison between runs, and festival summary
- **Core Fraud Detection Rules**:
  - Under/overpriced transaction detection using robust MAD-based z-scores
  - Rapid flip detection with adaptive time windows
  - Mule transfer detection with piecewise age rules
  - Wash trading detection via concentration heuristics and cycle detection
- **AI-Powered Anomaly Detection**: Isolation Forest with explainable features for account-day patterns
- **Market Simulation Engine**:
  - Configurable world generation (players, items, regions)
  - Dynamic market simulation with festival effects
  - Realistic trading patterns and inventory management
- **Fraud Injection System**:
  - Wash trading rings with configurable group sizes
  - Mule transfer patterns with realistic price manipulation
  - Collusive rapid flip scenarios
  - Busy-but-benign trader patterns for testing false positives
- **Comprehensive Testing Framework**:
  - Golden run regression tests with seed 1337
  - Metamorphic testing for time-shift invariance
  - Property-based testing for rule consistency
  - Ablation testing for detector uniqueness
  - AI explainability validation
- **Data Pipeline & Artifacts**:
  - CSV-based reproducible artifacts
  - Sweep configuration for festival ON/OFF comparison
  - Detailed metrics with precision/recall per fraud type
  - Rule contribution analysis
  - Miss audit capabilities
- **Documentation & Quality Assurance**:
  - Comprehensive test plan and procedures
  - Report templates for analysis
  - CI/CD pipeline with automated testing
  - Code coverage and quality checks

### Features
- **Multi-Rule Detection**: Price anomalies, rapid flips, mule transfers, wash trading
- **Explainable AI**: Top feature identification with plain-English descriptions
- **Festival Economics**: Configurable market events with smooth price impacts
- **Adaptive Detection**: Dynamic time windows based on item characteristics
- **Reproducible Results**: Deterministic simulation with configurable seeds
- **Performance Metrics**: Precision/recall analysis per fraud type and rule
- **Interactive Dashboard**: Real-time analysis and comparison tools

### Security
- Robust fraud detection algorithms resistant to evasion
- Comprehensive validation to minimize false positives
- Audit trail capabilities for compliance requirements

### Technical
- **Python 3.11** compatibility
- **Streamlit** web framework for interactive demos
- **pandas/numpy** for efficient data processing
- **scikit-learn** for machine learning components
- **pytest** framework for comprehensive testing
- **GitHub Actions** for continuous integration

## [0.9.0] - 2024-09-20

### Added
- Initial project structure and core simulation framework
- Basic fraud injection capabilities
- Rule-based detection prototypes
- Test infrastructure setup

### Changed
- Refined detection algorithms based on initial testing
- Improved simulation realism

## [0.1.0] - 2024-09-15

### Added
- Project initialization
- Basic market simulation concepts
- Initial fraud pattern research and design

---

## Release Notes

### Version 1.0.0 Highlights

This major release represents a production-ready fraud detection system for in-game economies. Key improvements include:

- **Production-Ready**: Comprehensive testing, CI/CD, and documentation
- **Interactive Demo**: Live Streamlit application for immediate evaluation
- **Explainable AI**: Clear reasoning for all detection decisions
- **Comprehensive Coverage**: Multiple fraud types with high precision/recall
- **Quality Assurance**: Extensive test suite with multiple validation approaches

For detailed usage instructions, see the [README](README.md) and [documentation](docs/).

### Migration Guide

This is the initial stable release. Future versions will include migration instructions here.

### Known Issues

- None reported for stable release

### Compatibility

- Requires Python 3.11+
- Compatible with modern pandas/numpy versions
- Tested on macOS and Linux environments
