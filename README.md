# PyDAS

Python Data Analysis System for ocean-engineering time series.

PyDAS reads laboratory binary `.out` files, keeps channels and segments on one object, then inspects, cuts, repairs short bad bursts, grades quality, filters, scales, analyses, plots, and writes reports. The public entry point is:

```python
from pydas import PyDAS

data = PyDAS(filename="case.out", lam=36)
data.print_info()
```

Package exports are only `PyDAS`, `diff1d`, and `data_change_fs`. Everything else is an instance method or lives under `pydas.waveModel`.

## Requirements

- Python >= 3.11
- numpy, pandas (>= 2.0), scipy, matplotlib, plotly, numba, and the other packages listed in `pyproject.toml`

## Install

```bash
git clone https://github.com/XiaoxG/pydas.git
cd pydas
pip install -e .
pip install -e ".[dev]"          # pytest, sphinx, formatters
pip install -e ".[performance]"  # optional large-dataset extras
```

## Ten-minute start

Do **not** treat “load → lowpass → spectrum → write” as the laboratory path.
The processing spine is:

```
read .out → inspect → cut_series → detect_bad_events → preview_repair
  → apply_repair → qc_report → remove_mean / detrend → filter
  → spectrum / extremes / Excel → write .out
  + sidecar qc.xlsx and repair_log.csv (audit is not stored in the pack)
```

Detect and repair **before** filtering. `qc_report` means a short burst is
repaired only **after** `apply_repair`. If MPM/EEV in `channel_report` is NaN,
open the qc table (`limited` / `bad`), do not start by changing the formula.
`cutoffull` is full-scale rad/s, not Hertz.

```python
from pydas import PyDAS

# Preferred laboratory path: binary .out (constructor does not guess CSV/MAT)
data = PyDAS(filename="case.out", lam=36)
data.print_info()
data.plot_channel("Wave1", plotbackend="matplotlib", show=False, save_path="wave1.png")
```

The full chain — including planted defects, `tz`, refused clip/edge events,
and the four-file delivery set — is in [`docs/user-guide.md`](docs/user-guide.md)
(Chinese teaching guide) and [`examples/lab_workflow.py`](examples/lab_workflow.py).
[`examples/basic_usage.py`](examples/basic_usage.py) is only a smoke-test of
imports, plotting, and a spectrum; it is not the basin workflow.

When you do not have an `.out` file:

```python
import numpy as np
import pandas as pd
from pydas import PyDAS

fs = 50.0
t = np.arange(0, 20, 1 / fs)
df = pd.DataFrame({"eta": np.sin(2 * np.pi * 0.5 * t)})
data = PyDAS.from_dataframe(df, fs=fs, lam=1.0, units={"eta": "m"})
# or: data = PyDAS.read_csv("eta.csv", fs=50.0, lam=1.0, units={"eta": "m"})
```

## What the object stores

| Attribute | Meaning |
|-----------|---------|
| `__filename__`, `__date__`, `__desc__` | Source file metadata |
| `__fs__` | Sampling frequency in Hz |
| `__chN__`, `__segN__` | Channel and segment counts |
| `__lam__`, `__scale__` | Length scale factor and `'model'` / `'full'` |
| `chInfo` | DataFrame of `Name`, `Unit`, `Coef` |
| `data` | `list` of per-segment DataFrames (channel columns) |
| `segInfo` | Segment timing / sample counts |
| `segStatis` | Per-segment stats with columns `Mean`, `STD`, `Max`, `Min`, `Unit` |

Empty objects (`PyDAS(filename=None, lam=1)`) start with one empty segment so `add_channel` can fill them.

## Public API (method names are frozen)

Historical mixin parameter names stay camelCase (`chName`, `chOld`, `newOrder`, `updateST`, `cutoffull`, `plotbackend`). Do not rename them in calling code.

**Channels:** `add_channel`, `delete_channel`, `select_channels`, `rename_channel`, `change_channel_order`, `copy_channel`, `channel_calculate`, `channel_apply_function`, `updateChN` (alias `update_channel_count`)

**Processing:** `apply_lowpass_filter`, `apply_highpass_filter`, `remove_mean`, `detrend`, `add_value`, `multiply_value`, `move_data`, `data_wash`, `add_diff1`, `add_diff2`, `cut_series`, `move_ccor`, `find_move_ccor`, `fix_unit`, `to_fullscale`, `channel2fullscale`, `updateST` (alias `update_statistics`)

**Quality (1.4):** `detect_bad_events`, `preview_repair`, `apply_repair`, `qc_report` (grades `good` / `repaired` / `limited` / `bad`)

**I/O:** `write` (`.out`), `to_dat`, `to_mat`, `to_feather`, `to_parquet`, `to_hdf5`, `read_waveCal`, `read_motion`, classmethods `from_dataframe` / `read_csv`

**Plot / analysis / report:** `plot_channel`, `plot_histogram`, `boxplot_channel`, `plot_xy`, `spectral_analysis`, `statistic_analysis`, `extreme_analysis`, `print_info`, `print_channel_info`, `print_statistics`, `channel_report`, `wave_report`

**waveModel:** `jonswap_spectrum` (alias `jonswap`), `pm_spectrum` (`PM`), `torsethaugen_spectrum`, `TimeSeries`, `SpecData1D`, `spectrum_to_timeseries`, and related DNV-RP-C205 helpers.

Report column meanings are documented in Chinese in [`docs/channel_report_metrics.md`](docs/channel_report_metrics.md) and [`docs/report_appendix_metrics.md`](docs/report_appendix_metrics.md).

## Pitfalls that used to be in the old README

- The constructor reads **binary `.out` only**. CSV/DAT/MAT are not auto-detected. Use `from_dataframe`, `read_csv`, or the export methods.
- Plotting uses `plotbackend='plotly'|'matplotlib'|'seaborn'`. There is no `use_plotly=True`.
- `cutoffull` is **full-scale rad/s**. In model scale the implemented cutoff is `cutoffull / (2π) * sqrt(λ)` in Hz. It is not a Hertz argument.
- `spectral_analysis(..., method='cov')` is the autocovariance estimator; `method='psd'` is Welch.
- `examples/historical/proc.py` is a historical lab notebook (`CaseData`, `addCh`, …). Those names are not on `PyDAS`. Do not run it.
- The `.out` on-disk layout is **frozen**. Other software reads the same pack. Do not change header widths, reserved bytes, int16 scaling, or 128-byte alignment.
- `data_wash` is a global 3σ interpolator. Do not use it on irregular-wave crests. Use `detect_bad_events` / `apply_repair` (`short_only`) for bursts. Audit is `repair_log`, not the `.out` file.

## Data quality and short-gap repair

These methods sit **on the spine above**, between cutting the window and
removing the mean. Skipping them is how a 25-count spike reaches MPM.

```python
events = data.detect_bad_events("eta", tz=1.0)   # read-only event table
preview = data.preview_repair("eta", tz=1.0)     # does not write data
data.apply_repair("eta", tz=1.0, preview=preview)
qc = data.qc_report(tz=1.0)                      # good/repaired/limited/bad
data.detrend("eta", kind="linear")               # independent of repair
# limited/bad skip MPM/EEV in extreme_analysis and channel_report
```

Default policy is `short_only`: only short spike/dropout bursts are filled (linear if `n<=3`, otherwise PCHIP). Clip, file-edge runs, and medium/long gaps are reported, not invented. A grade of `repaired` is only a completed repair after `apply_repair`. See `docs/user-guide.md`.

## Package layout

```
src/pydas/
  __init__.py          # PyDAS, diff1d, data_change_fs, __version__
  core/                # PyDAS facade + thin mixins + state / channels / io_format
  process.py           # filters, scaling, stats, correlation, detrend
  quality/             # bad-event detection, short-gap repair, qc grades
  analysis.py          # spectral / statistic / extreme analysis
  output.py            # writers that consume core.io_format
  reporting.py         # Excel channel / wave reports
  plot/                # matplotlib / plotly helpers
  waveModel/           # spectra, TimeSeries, DNV-RP-C205 models
  utils.py, logger.py
examples/              # runnable scripts for the current API
tests/                 # pytest (tests/legacy is not collected)
docs/                  # user guide and report metric catalogues
```

Mixins on `PyDAS` are thin proxies. Shared kernels live in `core/state.py`, `core/channels.py`, `core/io_format.py`, `process.py`, and `quality/`.

## Examples

```bash
python examples/lab_workflow.py    # full laboratory spine (planted defects)
python examples/basic_usage.py     # smallest API smoke-test, not the basin path
```

See [`examples/README.md`](examples/README.md). Do not run `examples/historical/proc.py`.

## Tests

```bash
pytest
```

`pytest.ini` sets `pythonpath=src` and `testpaths=tests`. `tests/legacy` is excluded on purpose.

## Coding conventions

English NumPy-style docstrings and English log messages. Public method names and historical camelCase parameters stay unchanged. New internal functions use snake_case. Details: [`docs/coding-standards.md`](docs/coding-standards.md).

## License

MIT. See [`LICENSE`](LICENSE).
