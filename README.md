# PyDAS - Python Data Analysis System

A comprehensive data analysis system for processing and analyzing large time series data in Python.

## Overview

PyDAS is a powerful Python library designed for analyzing and processing time series data, with a focus on signal processing, spectral analysis, and data visualization. It provides a comprehensive set of tools for engineers and scientists working with large datasets, particularly in the field of oceanic engineering.

## Project Structure

PyDAS has been refactored to use a modular architecture, improving maintainability and organization:

```
pydas/
├── src/                    # Main package directory
│   ├── __init__.py         # Package initialization
│   ├── pydas.py            # Core PyDAS class and methods
│   ├── process.py          # Data processing functions
│   ├── plot.py             # Visualization functions
│   ├── output.py           # Data export functions
│   ├── utils.py            # Utility functions
│   ├── logger.py           # Logging functionality
│   └── waveModel/          # Wave modeling subpackage
│       ├── __init__.py
│       ├── timeseries.py
│       └── ...
├── examples/               # Example scripts
├── tests/                  # Test suite
├── docs/                   # Documentation
├── setup.py                # Installation script
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## Modules and Functions

### Core Module (pydas.py)

The `PyDAS` class is the main entry point for working with time series data:

- **Initialization and Data Import**
  - `__init__(filename, lam, sseg, log_level)`: Initialize PyDAS object and read data
  - `__read__(sseg)`: Read data from file

- **Channel Management**
  - `add_channel(name, unit, series, fs, coef, point_of_move, sseg)`: Add a new channel
  - `delete_channel(name)`: Delete a specified channel
  - `select_channels(chnames)`: Select and keep specified channels
  - `rename_channel(chOld, chNew, sseg)`: Rename a channel
  - `change_channel_order(newOrder, sseg)`: Change the order of channels
  - `copy_channel(chName, new_chName, sseg)`: Create a copy of existing channel

- **Data Alignment**
  - `move_ccor(to_move_chName, base_chName, reference_ch, sseg)`: Move channel using cross-correlation
  - `find_move_ccor(base_chName, reference_ch, sseg)`: Find points to move between channels
  - `cut_series(start, stop, sseg)`: Cut time series to specified range

- **Data Import**
  - `read_waveCal(wavefname, sseg, YBname, YBcalname, alignFlag)`: Read wave calibration data
  - `read_motion(motionfname, alignAccName, alignMethod, zerofilename, lowpassfilter, rotation, NameList)`: Read motion data

- **Data Conversion**
  - `fix_unit(chName, newunit, pInfo)`: Fix channel unit
  - `to_fullscale(rho, g, pInfo)`: Convert model scale data to prototype scale
  - `channel2fullscale(channel_name, lam, rho, g)`: Convert a single channel to fullscale

- **Information Output**
  - `print_info(printTxt, printExcel)`: Print basic information
  - `print_channel_info(printTxt, printExcel)`: Print channel information
  - `print_statistics(printTxt, printExcel)`: Print statistical information
  - `statistic_analysis(ch_name, sseg, advanced, visualization, ...)`: Perform comprehensive statistical analysis with visualization

- **Data Maintenance**
  - `updateST(chName, sseg)`: Update statistical information
  - `updateChN(sseg)`: Update channel count

- **Channel Calculation**
  - `channel_calculate(ch1, ch2, operation, new_chName, sseg)`: Perform arithmetic operation between two channels
  - `channel_apply_function(ch, func, new_chName, unit, sseg)`: Apply custom function to a single channel

### Process Module (process.py)

Data processing functions for filtering, transformation, and cleaning:

- **Filtering**
  - `apply_lowpass_filter(pydas_obj, chName, cutoffull, replace, returnValue, sseg, order, plot)`: Apply lowpass filter
  - `apply_highpass_filter(pydas_obj, chName, cutoffull, replace, returnValue, sseg, order, plot)`: Apply highpass filter

- **Channel Data Processing**
  - `remove_mean(pydas_obj, chName, sseg)`: Remove mean from channel data
  - `add_value(pydas_obj, chName, value2add, sseg)`: Add constant value to channel data
  - `multiply_value(pydas_obj, chName, value2mul, sseg)`: Multiply channel data by constant
  - `move_data(pydas_obj, chName, point_of_move, sseg)`: Move channel data by specified points
  - `data_wash(pydas_obj, ChName, method, order, threshold, sseg)`: Clean data by detecting and interpolating outliers

- **Differential Operations**
  - `add_diff1(pydas_obj, name, sseg, filter, filter_cutoff)`: Calculate and add first derivative
  - `add_diff2(pydas_obj, name, sseg, filter, filter_cutoff)`: Calculate and add second derivative
  - `diff1d(data, dt)`: Calculate derivative of one-dimensional array

### Plot Module (plot.py)

Visualization functions with support for interactive and high-performance rendering:

- **Core Visualization**
  - `plot_channel(pydas_obj, ch_name, sseg, title, xlabel, ylabel, ...)`: Plot channel data
  - `plot_histogram(pydas_obj, ch_name, sseg, bins, fit_gaussian, ...)`: Generate histograms with statistics
  - `plot_xy(pydas_obj, x_ch_name, y_ch_name, sseg, ...)`: Create XY scatter plots

- **Analysis Visualization**
  - `spectral_analysis(pydas_obj, channel_name, method, L, plot, ...)`: Perform spectral analysis

### Output Module (output.py)

Functions for exporting data to various formats:

- `write_data(pydas_obj, filename, sseg, ch)`: Write data to file
- `export_to_dat(pydas_obj, Time, sseg)`: Export data to DAT format
- `export_to_mat(pydas_obj, sseg)`: Export data to MAT format

### Utils Module (utils.py)

Utility functions for general data processing:

- `data_change_fs(data, old_fs, new_fs)`: Change data sampling frequency

### Logger Module (logger.py)

Logging functionality for the PyDAS system:

- `setup_logger(level)`: Configure the logger
- `get_logger(name)`: Get a named logger

### Reporting Module (reporting.py)

| Function | Description |
|----------|-------------|
| `analyze_channel_data(data_scaled, mean_val, std_val, ...)` | Analyze channel data and return statistical results |
| `channel_report(pydas_obj, output_file, sseg, fullscale, ...)` | Generate detailed Excel analysis report for all channels |

### Reporting Examples

```python
# 基本统计分析报告
results = data.channel_report(
    output_file='basic_analysis.xlsx',
    header_text='Basic Analysis Report'
)

# 全尺度分析报告
results = data.channel_report(
    output_file='fullscale_analysis.xlsx',
    fullscale=True,
    lam=36,  # 尺度因子
    rho=1.025,  # 水密度
    g=9.807,  # 重力加速度
    header_text='Full Scale Analysis Report'
)

# 波浪分析报告（包含高低频分离）
results = data.channel_report(
    output_file='wave_analysis.xlsx',
    header_text='Wave Analysis Report',
    cutoffperiod=15.0,  # 高低频分离的截止周期（秒）
    significant_percentile=33.0,  # 显著值的百分位数
    wave_analysis=True,
    zerocrossing_analysis=True,
    amplitude_analysis=True,
    n_hr_forecast=3  # 3小时极值预测
)

# 自定义分析报告
results = data.channel_report(
    output_file='custom_analysis.xlsx',
    header_text='Custom Analysis Report',
    include_charts=False,  # 不包含图表
    format_sheet=True,  # 格式化Excel表格
    significant_percentile=33.0,  # 显著值的百分位数
    wave_analysis=True,  # 进行波浪分析
    zerocrossing_analysis=True,  # 进行过零分析
    amplitude_analysis=True,  # 进行振幅分析
    n_hr_forecast=3,  # 3小时极值预测
    cutoffperiod=20.0  # 高低频分离的截止周期
)

# 分析结果处理
if results:
    # 获取总统计结果
    total_stats = results[0]  # 总统计结果DataFrame
    low_freq_stats = results[1]  # 低频统计结果DataFrame
    high_freq_stats = results[2]  # 高频统计结果DataFrame
    
    # 打印特定通道的统计信息
    channel_name = "wave_height"
    channel_stats = total_stats[total_stats['Name'] == channel_name]
    if not channel_stats.empty:
        print(f"\n{channel_name} 统计信息:")
        print(f"最大值: {channel_stats['maximum'].values[0]:.3f}")
        print(f"最小值: {channel_stats['minimum'].values[0]:.3f}")
        print(f"平均值: {channel_stats['mean'].values[0]:.3f}")
        print(f"标准差: {channel_stats['STD'].values[0]:.3f}")
        print(f"显著双振幅: {channel_stats['sign.\ndouble\namplitude'].values[0]:.3f}")
        print(f"零上穿数: {channel_stats['number\nof zero\nupcross'].values[0]}")
        print(f"平均零上穿周期: {channel_stats['mean\nzerocro.\nperiod'].values[0]:.3f}")
        print(f"预估3小时最大值: {channel_stats['estimated\n3hr\nmaximum'].values[0]:.3f}")
        print(f"预估3小时最小值: {channel_stats['estimated\n3hr\nminimum'].values[0]:.3f}")
```

## Features

- **Data Processing & Analysis**
  - Multiple data format support (DAT, MAT, CSV, etc.)
  - Advanced filtering (lowpass, highpass)
  - Statistical analysis and data cleaning
  - Outlier detection and interpolation
  - Differential calculation and signal transformation

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
  - Memory-mapped file reading for large datasets
  - Chunk-based processing for huge datasets

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

# Install with high-performance dependencies for large datasets
pip install -e ".[performance]"

# Install with all dependencies
pip install -e ".[dev,performance]"
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
- distributed (≥2021.6.0)
- kaleido (≥0.2.0)
- scikit-learn (≥0.24.0)

## Quick Start

```python
from pydas import PyDAS

# Load data file
data = PyDAS(filename="your_data_file.out", lam=36)

# Basic info
data.print_info()

# Print statistics
data.print_statistics()

# Plot a channel
data.plot_channel("channel1", use_plotly=True, save_html="channel_plot.html")

# Apply a filter
data.apply_lowpass_filter("channel1", cutoffull=2.0)  # 2.0 Hz cutoff
data.plot_channel("channel1", title="Filtered Data")

# Add first derivative
data.add_diff1("channel1", filter=True, filter_cutoff=2.0)
data.plot_channel(["channel1", "channel1_d1"], title="Original and Derivative")

# Perform spectral analysis
data.spectral_analysis(
    channel_name="channel1",
    method="cov",  # Covariance method
    L=1024,        # Window size
    plot=True,
    use_plotly=True
)
```

## Usage Examples

### Channel Operations

```python
# Add a new channel
import numpy as np
time = np.linspace(0, 10, int(data.__fs__ * 10))
data.add_channel("new_channel", unit="m", series=np.sin(2 * np.pi * 0.5 * time), fs=data.__fs__)

# Remove mean from a channel
data.remove_mean("channel1")

# Add a constant value to a channel
data.add_value("channel1", value2add=1.5)

# Multiply a channel by a constant
data.multiply_value("channel1", value2mul=2.0)

# Clean data (detect and interpolate outliers)
data.data_wash("channel1", method="linear", threshold=3)

# Cut time series to a specific range
data.cut_series(start=10, stop=50)  # Cut between 10s and 50s
```

### Advanced Visualization

```python
# Interactive time series plot
data.plot_channel(
    "channel1", 
    use_plotly=True, 
    stats=True
)

# Multiple channel plot
data.plot_channel(
    ["channel1", "channel2"],
    use_plotly=True,
    downsampling=True,     # Enable downsampling for large datasets
    max_points=10000,      # Maximum number of points to plot
    title="Multiple Channel Comparison"
)

# XY scatter plot with density visualization
data.plot_xy(
    x_ch_idx="channel1", 
    y_ch_idx="channel2", 
    density_plot=True, 
    fit_line=True,
    use_plotly=True,
    use_webgl=True  # Improve rendering performance for large datasets
)

# Histogram with Gaussian fitting
data.plot_histogram(
    "channel1", 
    bins=50, 
    fit_gaussian=True, 
    use_plotly=True
)
```

### Channel Calculation

```python
# 通道之间的运算
# 加法运算 - 将两个相同单位的通道相加
data.channel_calculate("wave1", "wave2", "+", "wave_sum")
# 或使用名称
data.channel_calculate("wave1", "wave2", "add", "wave_sum")

# 减法运算 - 计算两个通道的差值
data.channel_calculate("force1", "force2", "-", "force_diff") 

# 乘法运算 - 例如力和臂长相乘得到力矩
data.channel_calculate("force", "arm_length", "*", "moment")
# 或使用名称
data.channel_calculate("force", "arm_length", "multiply", "moment")

# 除法运算 - 例如计算阻抗(电压除以电流)
data.channel_calculate("voltage", "current", "/", "resistance")
# 自动处理除零错误

# 单通道函数运算
# 使用字符串表达式
data.channel_apply_function("displacement", "x**2", "displacement_squared")
data.channel_apply_function("voltage", "np.log10(x)", "voltage_log")
data.channel_apply_function("signal", "np.abs(x)", "signal_magnitude")

# 使用函数对象
import numpy as np
data.channel_apply_function("acceleration", np.square, "accel_squared")
data.channel_apply_function("velocity", lambda x: x**3, "velocity_cubed")

# 指定单位(默认会自动推断)
data.channel_apply_function("force", "x**2", "force_squared", unit="N²")

# 应用三角函数
data.channel_apply_function("angle", "np.sin(x)", "sin_component")
data.channel_apply_function("signal", "np.arctan(x)", "phase")

# 分段处理
data.channel_apply_function("wave_height", "x**2", "wave_energy", sseg=[0, 1, 2]) 
```

### Statistical Analysis

```python
# 基本统计分析
stats = data.statistic_analysis("wave_height")
print(stats)  # 显示统计结果DataFrame

# 分析多个通道
stats_multi = data.statistic_analysis(["wave_height", "wave_period", "current_speed"])

# 高级统计量 (包括偏度、峰度、分位数、峰值因子等)
stats_advanced = data.statistic_analysis("acceleration", advanced=True)

# 带可视化的统计分析
data.statistic_analysis("wave_force", 
                        advanced=True,
                        visualization=True,  # 启用可视化
                        bins=100,            # 直方图箱数
                        use_plotly=True)     # 使用plotly生成交互式图表

# 保存统计分析结果和图表
data.statistic_analysis("mooring_tension", 
                        sseg=2,               # 分析第3个数据段
                        advanced=True, 
                        visualization=True,
                        save_fig=True,        # 保存图形
                        save_path="./results") # 保存路径

# 使用matplotlib可视化 (替代plotly)
data.statistic_analysis("current_profile", 
                        visualization=True,
                        use_plotly=False)     # 使用matplotlib代替plotly
```

### Extreme Value Analysis

```python
# 基本极值分析
results = data.extreme_analysis("wave_height")

# 自定义峰值检测参数
results = data.extreme_analysis("wave_height",
                              peak_prominence=2.0,    # 设置峰值检测的突出度
                              peak_distance=50)       # 设置峰值之间的最小距离

# 全尺度极值分析
results = data.extreme_analysis("wave_height",
                              fullscale=True,        # 转换为原型尺度
                              lam=50)                # 设置尺度系数

# 自定义可视化选项
results = data.extreme_analysis("wave_height",
                              visualization=True,
                              plotbackend='plotly',  # 使用Plotly后端
                              save_path='extreme_analysis.png',
                              save_html='extreme_analysis.html')

# 分析结果处理
if results:
    # 获取检测到的峰值
    pos_peaks = results['peaks_positive']  # 正峰值
    neg_peaks = results['peaks_negative']  # 负峰值
    
    # 获取极值估计
    if 'return_values' in results:
        # 100年一遇极值
        return_100y = results['return_values']['100_year']
        print(f"100年一遇极值: {return_100y:.2f}")
        
        # 置信区间
        if 'return_value_confidence_intervals' in results:
            ci = results['return_value_confidence_intervals']['100_year']
            print(f"95%置信区间: [{ci['lower_95']:.2f}, {ci['upper_95']:.2f}]")
```

### Data Export

```python
# Export to MAT file
data.to_mat("processed_data.mat")

# Export to DAT file
data.to_dat("processed_data.dat")

# Write data file
data.write("processed_data.out")
```

## License

PyDAS is distributed under the MIT License. See the LICENSE file for more information.

## API Reference

### Core Module (pydas.py)

| Category | Function | Description |
|----------|----------|-------------|
| **Initialization** | `__init__(filename, lam, sseg, log_level)` | Initialize PyDAS object and read data file |
| | `__read__(sseg)` | Read data from file |
| **Channel Management** | `add_channel(name, unit, series, fs, ...)` | Add a new channel to the dataset |
| | `delete_channel(name)` | Delete a channel from the dataset |
| | `select_channels(chnames)` | Select and keep specified channels |
| | `rename_channel(chOld, chNew, sseg)` | Rename an existing channel |
| | `change_channel_order(newOrder, sseg)` | Change the order of channels |
| | `copy_channel(chName, new_chName, sseg)` | Create a copy of existing channel |
| **Channel Calculation** | `channel_calculate(ch1, ch2, operation, new_chName, sseg)` | Perform arithmetic operation between two channels |
| | `channel_apply_function(ch, func, new_chName, unit, sseg)` | Apply custom function to a single channel |
| **Data Alignment** | `move_ccor(to_move_chName, base_chName, reference_ch, sseg)` | Move channel using cross-correlation |
| | `find_move_ccor(base_chName, reference_ch, sseg)` | Find points to move between channels |
| | `cut_series(start, stop, sseg)` | Cut time series to specified range |
| **Data Import** | `read_waveCal(wavefname, sseg, YBname, YBcalname, alignFlag)` | Read wave calibration data |
| | `read_motion(motionfname, alignAccName, alignMethod, ...)` | Read motion data and add as channels |
| **Data Conversion** | `fix_unit(chName, newunit, pInfo)` | Fix channel unit |
| | `to_fullscale(rho, g, pInfo)` | Convert model scale data to prototype scale |
| | `channel2fullscale(channel_name, lam, rho, g)` | Convert channel to fullscale |
| **Information Output** | `print_info(printTxt, printExcel)` | Print basic information |
| | `print_channel_info(printTxt, printExcel)` | Print channel information |
| | `print_statistics(printTxt, printExcel)` | Print statistical information |
| | `statistic_analysis(ch_name, sseg, advanced, visualization, ...)` | Perform comprehensive statistical analysis with visualization |
| **Data Maintenance** | `updateST(chName, sseg)` | Update statistical information |
| | `updateChN(sseg)` | Update channel count information |

### Process Module (process.py)

| Function | Description |
|----------|-------------|
| `apply_lowpass_filter(pydas_obj, chName, cutoffull, ...)` | Apply lowpass filter to channel data |
| `apply_highpass_filter(pydas_obj, chName, cutoffull, ...)` | Apply highpass filter to channel data |
| `remove_mean(pydas_obj, chName, sseg)` | Remove mean from channel data |
| `add_value(pydas_obj, chName, value2add, sseg)` | Add constant value to channel data |
| `multiply_value(pydas_obj, chName, value2mul, sseg)` | Multiply channel data by constant |
| `move_data(pydas_obj, chName, point_of_move, sseg)` | Move channel data by specified points |
| `data_wash(pydas_obj, ChName, method, order, threshold, sseg)` | Clean data by detecting and interpolating outliers |
| `add_diff1(pydas_obj, name, sseg, filter, filter_cutoff)` | Calculate and add first derivative |
| `add_diff2(pydas_obj, name, sseg, filter, filter_cutoff)` | Calculate and add second derivative |
| `diff1d(data, dt)` | Calculate derivative of one-dimensional array |

### Plot Module (plot.py)

| Function | Description |
|----------|-------------|
| `plot_channel(pydas_obj, ch_name, sseg, ...)` | Plot channel time series data |
| `plot_histogram(pydas_obj, ch_name, sseg, bins, ...)` | Generate histogram with statistics |
| `plot_xy(pydas_obj, x_ch_name, y_ch_name, sseg, ...)` | Create XY scatter plot |
| `spectral_analysis(pydas_obj, channel_name, method, L, ...)` | Perform spectral analysis on channel |
| `boxplot_channel(pydas_obj, ch_name, sseg, use_peaks, ...)` | Create boxplot visualization of channel data |

### Analysis Module (analysis.py)

| Function | Description |
|----------|-------------|
| `spectral_analysis(pydas_obj, channel_name, method, L, ...)` | Perform spectral analysis and return spectrum |
| `statistic_analysis(pydas_obj, ch_name, sseg, advanced, ...)` | Perform statistical analysis with visualization |
| `extreme_analysis(pydas_obj, ch_name, sseg, visualization, ...)` | Perform extreme value analysis on channel peaks with GEV/Gumbel distribution fitting and return period calculation |

### Output Module (output.py)

| Function | Description |
|----------|-------------|
| `write_data(pydas_obj, filename, sseg, ch)` | Write data to file |
| `export_to_dat(pydas_obj, Time, sseg)` | Export data to DAT format |
| `export_to_mat(pydas_obj, sseg)` | Export data to MAT format |

### Utils Module (utils.py)

| Function | Description |
|----------|-------------|
| `data_change_fs(data, old_fs, new_fs)` | Change data sampling frequency |

### Logger Module (logger.py)

| Function | Description |
|----------|-------------|
| `setup_logger(level)` | Configure the logger for the PyDAS system |
| `get_logger(name)` | Get a named logger instance |

### Reporting Module (reporting.py)

| Function | Description |
|----------|-------------|
| `analyze_channel_data(data_scaled, mean_val, std_val, ...)` | Analyze channel data and return statistical results |
| `channel_report(pydas_obj, output_file, sseg, fullscale, ...)` | Generate detailed Excel analysis report for all channels |

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
data = PyDAS(filename="your_data_file.out", lam=36)

# 绘制通道数据
data.plot_channel("channel1", use_plotly=True)

# 应用滤波器
data.apply_lowpass_filter("channel1", cutoffull=2.0)

# 执行谱分析
spec, fig = data.spectral_analysis(
    channel_name="channel1",
    method="cov",
    L=1024,
    use_plotly=True
)
```

有关更详细的说明和示例，请参阅上面的英文文档部分。 

### Large-Scale Data Processing

PyDAS provides optimized methods for handling large datasets (>100,000 points):

```python
# 导入必要模块
import dask.dataframe as dd
import numpy as np
from pydas import PyDAS

# 加载大型数据集
data = PyDAS(filename="large_dataset.csv", use_dask=True)

# 使用dask进行并行计算
df = dd.from_pandas(data.data[0], npartitions=8)

# 通道计算优化
result = df['channel1'].map_partitions(lambda x: x.rolling(window=100).mean()).compute()
data.add_channel('channel1_smoothed', result, unit='m')

# 高性能可视化
data.plot_channel(
    'channel1_smoothed',
    use_plotly=True,
    use_webgl=True,
    data_decimation='lttb',
    chunk_size=20000
)

# 使用datashader和holoviews进行超大数据集可视化（依赖performance扩展）
try:
    import datashader as ds
    import holoviews as hv
    from holoviews.operation.datashader import datashade
    
    hv.extension('bokeh')
    
    # 创建holoviews曲线
    curve = hv.Curve((data.data[0].index, data.data[0]['channel1']))
    
    # 使用datashader进行可视化
    shaded = datashade(curve, width=800, height=400)
    
    # 显示图表
    hv.save(shaded, 'large_dataset_visualization.html')
    
except ImportError:
    print("Performance visualization requires datashader and holoviews.")
    print("Install with: pip install -e '.[performance]'")

### Full-Scale Analysis
