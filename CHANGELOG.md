# Changelog

## [1.4.2] - 2026-09-22

### Changed

- Teaching docs treat the laboratory **processing pipeline** as the spine.
  `docs/user-guide.md` (Chinese) now starts from that sequence; quality
  detect / preview / apply / qc sits on the chain, not after the API list.
- `examples/lab_workflow.py` plants startup, spike, dropout, and clip
  defects and walks the same spine; audit files are written beside `.out`.
- README ten-minute start points at the user guide and lab workflow instead
  of teaching load → filter → spectrum as the default path.
- Report metric catalogues note a third cause of MPM/EEV NaN: quality grade
  `limited` / `bad`. The 19-column delivery table is unchanged.

### Notes

- `examples/proc.py` moved to `examples/historical/` (do not run).
- Filter-frequency advice, LLM, and Phase 7 grade-honesty code are still
  out of scope.

---

### Added

- Quality grades gate MPM/EEV: `extreme_analysis` and `channel_report` skip
  extremes when a channel-segment is `limited` or `bad`. The 19-column
  delivery table is unchanged (MPM/EEV stay NaN). `repaired` still computes
  MPM and logs that short-gap repairs are included.
- Read-only qc flags: NaN/Inf counts, constant channel, record too short for
  MPM, unsteady startup, Nyquist energy warning, scale hint, filter-not-assessed,
  multi-segment inconsistency.

### Notes

- Pass `respect_quality=False` to keep the pre-1.4.1 extreme behaviour.
- Filter-frequency advice and LLM remain out of scope.

---

## [1.4.0] - 2026-09-21

### Added

- Quality pipeline (`pydas.quality` + `QualityMixin`): `detect_bad_events`, `preview_repair`, `apply_repair`, `qc_report`.
- Events are **runs** (consecutive bad samples), not isolated indices: `spike_burst`, `dropout`, `clip`, with `at_edge`.
- Default repair policy **`short_only`**: short spike/dropout bursts are filled; medium/long gaps, clip, and edge runs are reported and left unchanged.
- Interpolator: linear for ``n<=3``, PCHIP for longer short bursts.
- Multi-channel overlap: short coincident spikes may still be repaired per channel; coincident dropout/clip grades the **segment** `limited` or `bad`.
- Grades `good` / `repaired` / `limited` / `bad`. Audit table `repair_log` (not stored in `.out`).
- Independent `detrend(chName, kind='linear'|'constant')`. Not part of repair.

### Changed

- `data_wash` documented as global mean±kσ and **unsuitable** for irregular-wave crests.
- `add_diff1` / `add_diff2` `filter_cutoff` documented as ``cutoffull`` (full-scale rad/s).
- `updateST` notes that NaN samples are skipped.

---

## [1.3.0] - 2026-09-21

### Added

- MIT `LICENSE` file matching the `pyproject.toml` classifier.
- Package `__version__` on `pydas`.
- `PyDAS.from_dataframe` / `PyDAS.read_csv` for text tables (binary `.out` remains the constructor path).
- Snake-case aliases `update_channel_count` (`updateChN`) and `update_statistics` (`updateST`); historical names stay.
- English README rewritten to the real API; Chinese teaching guide `docs/user-guide.md`; `docs/coding-standards.md`.
- Runnable `examples/basic_usage.py` and `examples/lab_workflow.py`; `examples/README.md`.
- GitHub Actions pytest workflow on Python 3.11 / 3.12.
- Shared kernels from the architecture cleanup: `core/state.py`, `core/channels.py`, `core/io_format.py`, `_apply_butterworth`, `_correlation_lag`.
- Byte-for-byte `.out` pack compatibility test (`tests/unit/test_out_pack_compat.py`).

### Changed

- Version aligned to **1.3.0** (`pyproject.toml` had stayed at 1.1.0 while CHANGELOG already listed 1.2.0).
- Project URLs point at GitHub `XiaoxG/pydas`.
- Mixins are thin proxies; public method names and camelCase parameters are unchanged.
- Docstrings in core / process / analysis / plot unified to NumPy section headers.
- `cutoffull` documented as full-scale rad/s on the mixin and process APIs.

### Fixed

- Phases 1–2 behaviour fixes now in the same release line: `diff1d` NumPy return, empty-object `data`/`segStatis` lists, `segStatis` column `STD`, `to_mat(filename, sseg)`, `cut_series` indexing, `TimeSeries.tospecdata(method=...)`, `print_info` returning a DataFrame, first `add_channel` on an empty object, plot/spectrum pytest regressions.

### Notes

- The binary `.out` pack layout is frozen. Do not change `core/io_format.py` constants.

---

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
