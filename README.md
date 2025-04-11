# PyDAS - Python Data Analysis System

A comprehensive data analysis system for processing and analyzing large time series data in Python.

## Overview

PyDAS is a powerful Python library designed for analyzing and processing time series data, with a focus on signal processing, spectral analysis, and data visualization. It provides a comprehensive set of tools for engineers and scientists working with large datasets, particularly in the field of oceanic engineering.

## Project Structure

```
pydas/
├── src/              # Main package directory
│   ├── __init__.py   # Package initialization 
│   ├── pydas.py      # Core functionality
│   ├── pydas_plot.py # Plotting utilities
│   └── waveModel/    # Wave modeling subpackage
│       ├── __init__.py
│       ├── core.py
│       ├── specdata.py
│       ├── specmodels.py
│       └── ...
├── examples/         # Example scripts
├── tests/            # Test suite
├── docs/             # Documentation
├── setup.py          # Installation script
├── requirements.txt  # Dependencies
└── README.md         # This file
```

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

### From Source

```bash
# Clone the repository
git clone https://gitee.com/xiaoxianguo/pydas.git
cd pydas

# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

### Requirements

PyDAS depends on the following packages:
- numpy (≥1.19.0)
- pandas (≥1.1.0)
- scipy (≥1.5.0)
- matplotlib (≥3.3.0)
- plotly (≥5.0.0)
- numba (≥0.50.0)
- dask (≥2021.6.0)
- kaleido (≥0.2.0)
- scikit-learn (≥0.24.0)

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

### Using waveModel Directly

The `waveModel` subpackage can be imported directly for spectral modeling and wave analysis:

```python
# Import the wave modeling subpackage
import pydas.waveModel as wm

# Create a JONSWAP spectrum
freq = np.linspace(0.05, 2, 100)  # Frequency array in Hz
Hs = 4.0  # Significant wave height in meters
Tp = 10.0  # Peak period in seconds
gamma = 3.3  # Peakedness parameter

# Generate spectrum
S = wm.jonswap(freq, Hs, Tp, gamma)

# Calculate spectral moments
m0 = wm.moment(freq, S, 0)  # Zeroth moment
m1 = wm.moment(freq, S, 1)  # First moment
m2 = wm.moment(freq, S, 2)  # Second moment

# Calculate wave parameters
Hm0 = 4.0 * np.sqrt(m0)  # Significant wave height
Tm01 = m0/m1  # Mean period
Tm02 = np.sqrt(m0/m2)  # Zero-crossing period

print(f"Significant wave height: {Hm0:.2f} m")
print(f"Mean period: {Tm01:.2f} s")
print(f"Zero-crossing period: {Tm02:.2f} s")
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

## Development

### Setting Up a Development Environment

```bash
# Clone the repository
git clone https://gitee.com/xiaoxianguo/pydas.git
cd pydas

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all extra dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_specific.py

# Run with coverage report
pytest --cov=pydas
```

### Code Style

PyDAS follows PEP 8 style guidelines. You can check your code with:

```bash
# Check code style
flake8 src tests

# Auto-format code
black src tests
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

## Contributing

Contributions to PyDAS are welcome! Here's how you can contribute:

1. **Fork the Repository**: Create your own fork of the project
2. **Create a Branch**: Make your changes in a new branch
3. **Write Tests**: Add tests for new features or bug fixes
4. **Follow Style Guidelines**: Ensure your code follows PEP 8
5. **Submit a Pull Request**: Open a PR to merge your changes

### Contribution Guidelines

- Keep the code well-documented
- Maintain backward compatibility when possible
- Write unit tests for new features
- Update documentation to reflect changes

## License

MIT License

## Citation

If you use PyDAS in your research, please cite:

```
@software{PyDAS2024,
  author = {Guo, Xiaoxiang},
  title = {PyDAS: Python Data Analysis System},
  url = {https://gitee.com/xiaoxianguo/pydas},
  version = {1.0.3},
  year = {2025},
}
```

## 中文说明

PyDAS是一个用于分析和处理大型时间序列数据的Python库，专注于信号处理、谱分析和数据可视化。它为工程师和科学家提供了一套全面的工具，特别适用于海洋工程领域的大型数据集处理。

### 主要功能

- **数据处理与分析**：支持多种数据格式，提供高级过滤、统计分析和数据清洗功能
- **可视化**：交互式时间序列图、谱分析、直方图和XY散点图
- **性能优化**：使用Numba JIT编译和向量化操作，自动下采样大型数据集
- **科学计算**：谱密度估计、谱矩计算、通道间互相关和信号导数

### 安装方法

```bash
# 从PyPI安装
pip install pydas

# 或从源代码安装
git clone https://gitee.com/xiaoxianguo/pydas.git
cd pydas
pip install -e .
```

### 快速入门

```python
from pydas import PyDAS

# 加载数据文件
data = PyDAS(filename="your_data_file.csv")

# 绘制通道数据
data.plot_channel("channel1", use_plotly=True)

# 应用滤波器
data.apply_lowpass_filter("channel1", cutoff=2.0)

# 执行谱分析
spec, fig = data.spectral_analysis(
    channel_name="channel1",
    method="cov",
    L=1024,
    use_plotly=True
)
```

有关更详细的说明和示例，请参阅上面的英文文档部分。 