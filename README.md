# PyDAS - Python Data Analysis System

A comprehensive data analysis system for processing and analyzing large time series data in Python.

## Overview

PyDAS is a powerful Python library designed for analyzing and processing time series data, with a focus on signal processing, spectral analysis, and data visualization. It provides a comprehensive set of tools for engineers and scientists working with large datasets, particularly in the field of oceanic engineering.

## Features

- **Data Processing & Analysis**
  - Multiple data format support (DAT, MAT, CSV, etc.)
  - Advanced filtering (lowpass, highpass, custom)
  - Statistical analysis and data cleaning
  - Downsampling and resampling capabilities
  - Outlier detection and handling

- **Visualization**
  - Interactive time series plots
  - Spectral analysis with customizable parameters
  - Histograms with statistical information
  - XY scatter plots with density visualization
  - Full-scale analysis visualization

- **Performance Optimization**
  - Numba JIT compilation for computationally intensive operations
  - Automatic downsampling for large datasets
  - Vectorized operations for faster processing
  - WebGL rendering for interactive visualization of large datasets

- **Scientific Computing**
  - Spectral density estimation
  - Moment calculation for spectral analysis
  - Cross-correlation between channels
  - Signal derivatives and transformations

## Installation

```bash
# Install from PyPI
pip install pydas

# Or install from source
git clone https://gitee.com/xiaoxianguo/pydas.git
cd pydas
pip install -e .
```

## Quick Start

```python
from pydas import PyDAS
import numpy as np

# Load data file
data = PyDAS(filename="your_data_file.csv")

# Basic info
print(f"Loaded data with {len(data.channels)} channels")
print(f"Available channels: {data.channels}")
print(f"Sampling frequency: {data.fs} Hz")

# Plot a channel
data.plot_channel("channel1", use_plotly=True, save_html="channel_plot.html")

# Apply a filter
data.apply_lowpass_filter("channel1", cutoff=2.0)  # 2.0 Hz cutoff
data.plot_channel("channel1", title="Filtered Data")

# Perform spectral analysis
spec, fig = data.spectral_analysis(
    channel_name="channel1",
    method="cov",  # Covariance method
    L=1024,        # Window size
    use_plotly=True
)

# Extract spectral characteristics
m0 = float(spec.moment(0))
print(f"Zeroth moment (m0): {m0:.5f}")
print(f"Significant wave height: {4.0 * np.sqrt(m0):.5f}")
```

## Usage Examples

### Channel Operations

```python
# Add a new channel
data.add_channel("new_channel", values=np.sin(np.linspace(0, 10*np.pi, len(data["channel1"]))), unit="m")

# Add derivative of a channel
data.add_diff1("channel1", new_ch_name="channel1_derivative")

# Remove mean from a channel
data.remove_mean("channel1")

# Cut time series to a specific range
data.cut_series(start_t=10, end_t=50)  # Cut between 10s and 50s
```

### Advanced Visualization

```python
# Interactive time series plot with statistics
data.plot_channel(
    "channel1", 
    use_plotly=True, 
    stats=True, 
    downsampling=True, 
    max_points=20000
)

# XY scatter plot with density visualization
data.plot_xy(
    x_ch_idx="channel1", 
    y_ch_idx="channel2", 
    density_plot=True, 
    fit_line=True,
    use_plotly=True
)

# Histogram with Gaussian fitting
data.plot_histogram(
    "channel1", 
    bins=50, 
    fit_gaussian=True, 
    show_stats=True
)

# Multi-channel spectral analysis
results = data.spectral_analysis(
    channel_name=["channel1", "channel2", "channel3"], 
    method="psd", 
    subplot_layout=(2, 2),
    use_plotly=True
)
```

### Data Import and Export

```python
# Import data
wave_data = PyDAS(filename="wave_data.csv")

# Export to MAT file
wave_data.to_mat("processed_data.mat")

# Export to DAT file
wave_data.to_dat("processed_data.dat")

# Export to CSV
wave_data.to_csv("processed_data.csv")
```

### Full-Scale Analysis

```python
# Perform spectral analysis with model scale
spec_model, fig_model = data.spectral_analysis(
    channel_name="wave_height",
    method="cov",
    use_plotly=True,
    title="Model Scale Spectrum"
)

# Perform full-scale spectral analysis (with scale factor λ)
spec_full, fig_full = data.spectral_analysis(
    channel_name="wave_height",
    method="cov",
    use_plotly=True,
    fullscale=True,  # Enable full scale
    title="Full Scale Spectrum"
)

# Compare results
Hm0_model = 4.0 * np.sqrt(float(spec_model.moment(0)))
Hm0_full = 4.0 * np.sqrt(float(spec_full.moment(0)))
print(f"Model scale Hm0: {Hm0_model:.4f} m")
print(f"Full scale Hm0: {Hm0_full:.4f} m")
```

## API Reference

### Core Functions

| Category | Function | Description |
|----------|----------|-------------|
| **Channel Operations** | `add_channel()` | Add a new channel to the dataset |
| | `delete_channel()` | Delete a channel from the dataset |
| | `select_channels()` | Select and keep specific channels |
| | `rename_channel()` | Rename an existing channel |
| | `change_channel_order()` | Change the order of channels |
| **Data Processing** | `remove_mean()` | Remove mean value from channel data |
| | `add_value()` | Add constant value to channel data |
| | `multiply_value()` | Multiply channel data by constant value |
| | `cut_series()` | Cut time series to specified range |
| | `move_data()` | Move channel data by specified offset |
| | `data_wash()` | Clean data, detect and interpolate outliers |
| **Differential Operations** | `add_diff1()` | Calculate and add first derivative |
| | `add_diff2()` | Calculate and add second derivative |
| **Filtering** | `apply_lowpass_filter()` | Apply lowpass filter to channel data |
| | `apply_highpass_filter()` | Apply highpass filter to channel data |
| **Data Alignment** | `move_ccor()` | Move channel data using cross-correlation |
| | `find_move_ccor()` | Find points to move between channels |
| **Data Output** | `to_dat()` | Export data to DAT file |
| | `to_mat()` | Export data to MAT file |
| | `to_csv()` | Export data to CSV file |
| | `write()` | Write data to generic file |
| **Visualization** | `plot_channel()` | Plot channel time series |
| | `plot_histogram()` | Generate histogram with statistics |
| | `plot_xy()` | Create XY scatter plot |
| | `spectral_analysis()` | Perform spectral analysis on channel |
| **Data Conversion** | `fix_unit()` | Fix channel unit |
| | `to_fullscale()` | Convert model scale data to prototype scale |

### Utility Functions

| Function | Description |
|----------|-------------|
| `diff1d()` | Calculate derivative of one-dimensional array |
| `data_change_fs()` | Change data sampling frequency |
| `print_info()` | Print basic information about dataset |
| `print_channel_info()` | Print detailed channel information |
| `print_statistics()` | Print statistical information for channels |
| `updateST()` | Update statistical information for all channels |
| `updateChN()` | Update channel count information |

## Dependencies

- numpy (≥1.19.0)
- pandas (≥1.1.0)
- scipy (≥1.5.0)
- matplotlib (≥3.3.0)
- plotly (≥5.0.0)
- numba (≥0.50.0)
- dask (≥2021.6.0)
- kaleido (≥0.2.0)
- scikit-learn (≥0.24.0)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use PyDAS in your research, please cite:

```
@software{PyDAS2024,
  author = {Guo, Xiaoxiang},
  title = {PyDAS: Python Data Analysis System},
  url = {https://gitee.com/xiaoxianguo/pydas},
  version = {1.0.2},
  year = {2024},
}
``` 