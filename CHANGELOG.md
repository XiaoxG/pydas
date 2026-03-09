# Changelog

## [1.2.0] - 2026-03-09

### Added

- **MPM Reporting**: Integrated Most Probable Maximum (MPM) estimation into the reporting pipeline, enabling automated extreme-value statistics in Excel reports.

### Changed

- **Reporting Refactor**: Major overhaul of `reporting.py` — restructured channel reporting logic, improved MPM calculation accuracy, and reduced code duplication (~475 net lines added/reorganized across multiple commits).
- **Python Version Requirement**: Raised minimum required Python version from `>=3.6` to `>=3.11` in `pyproject.toml`, aligned with the project's actual runtime stack. Updated PyPI classifiers accordingly (removed 3.6–3.10, added 3.11 and 3.12).
- **Plot Module**: Refactored `plot.py` with significantly expanded XY-plot support (~183 net lines added), fixing incorrect axis handling and improving multi-channel scatter plot flexibility.

### Fixed

- **XY Plot**: Resolved functional regression in `plot_xy` that caused incorrect data pairing and axis label rendering.
- **`pydas.py` Adjustments**: Minor interface fixes in the main `PyDAS` class to align with updated reporting and plot APIs.

---

## [1.1.0] - 2026-02-23

### Added

- **Systematic Testing Framework**: Integrated `pytest` with 30+ core test cases covering `unit`, `integration`, and `legacy` scenarios.
- **Automated Documentation**: Established Sphinx-based documentation system with RTD theme and MyST (Markdown) support.
- **New Lifecycle Scripts**: Added `run_tests.ps1` and `build_docs.ps1` for one-click verification and doc building.
- **Robust Initialization**: Enabled `PyDAS(None)` for manual dataset construction with proper metadata handling.
- **Type Hints**: Comprehensive PEP 484 type hinting across `pydas.py`, `process.py`, and `analysis.py`.

### Changed

- **WaveModel Refactor**: Total overhaul of the `waveModel` subpackage into a modular architecture (`core`, `models`, `analysis`, `simulation`, `objects`), replacing the bloated legacy WAFO port.
- **DNV Compliance**: Aligned wave spectral models and directional spreading functions strictly with **DNV-RP-C205** standards.
- **Standardized Comments**: Transitioned the entire core codebase to English comments and docstrings for scientific consistency.
- **Enhanced I/O**: Refactored `.OUT` file writing to use robust `.iloc` indexing, preventing crashes with complex segment metadata.

### Fixed

- **Normalization Correction**: Fixed a 2x factor error in `cos2s_spreading` normalization coefficients.
- **Dynamic Meta Mapping**: Resolved `ValueError` in `add_channel` when working with datasets containing additional scaling metadata.
- **Bug Fixes**: Corrected integer formatting errors in logging and fixed missing attributes in `SpecData1D`.

## [1.0.4] - 2023-05-15
