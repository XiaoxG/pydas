# PyDAS - Python Data Analysis System

PyDAS is a powerful Python library for processing and analyzing waveform data, offering robust capabilities for data import, processing, analysis, and visualization.

## Key Features

- Support for multiple data formats import and export
- Powerful time series data processing and analysis
- Advanced filtering capabilities (lowpass, highpass filters)
- Various visualization tools including:
  - Time series visualization
  - Spectrum analysis plots
  - Histograms
  - XY scatter plots (with density visualization)
- Interactive plotting (based on Plotly and Matplotlib)
- Flexible data management and channel operations

## Latest Updates

### XY Scatter Plot Enhancements

The XY scatter plot functionality has been enhanced with the following features:

- **Automatic Downsampling**: Intelligently reduces point count for large datasets while preserving data patterns
- **Density Visualization**: Provides heatmap/contour overlay to show data concentration in dense point clouds
- **Statistical Information**: Optional display of key statistics for both channels and their relationship
- **Linear Regression**: Fit a trend line to visualize correlation between channels
- **1:1 Aspect Ratio**: Maintains equal scaling on both axes for accurate spatial relationships

### Spectral Analysis with Plotly Support

The spectral analysis feature in PyDAS allows users to perform frequency domain analysis on time series data. It utilizes the `waveModel` library to compute spectral characteristics of channel data. Key features include:

- **Multiple Analysis Methods**: Support for covariance-based ('cov') and Welch's periodogram ('psd') methods
- **Filtering Options**: Apply low-pass filtering before analysis
- **Interactive Visualization**: Use Plotly for interactive plots or Matplotlib for static plots
- **Spectral Characteristics**: Calculate and display key spectral parameters like significant wave height and peak period
- **Customization**: Customize plot appearance, frequency ranges, and output options

## Installation

```bash
pip install pydas
```

Or install from source:

```bash
git clone https://github.com/yourusername/pydas.git
cd pydas
pip install -e .
```

## Basic Usage

```python
from pydas import PyDAS

# Load data
data = PyDAS('my_data_file.out')

# Plot channel data
data.plot_channel('Channel1')

# Create XY scatter plot with new density feature
data.plot_xy(
    x_ch_idx='Channel1',
    y_ch_idx='Channel2',
    density_plot=True,
    downsampling=True,
    max_points=10000
)

# Create histogram
data.plot_histogram('Channel1', bins=50, fit_gaussian=True)

# Apply filters
data.apply_lowpass_filter('Channel1', cutoff=0.1)

# Perform spectral analysis
spec, fig = data.spectral_analysis(
    channel_name='Channel1',  # Channel name or index
    method='cov',             # Analysis method: 'cov' or 'psd'
    L=1024,                   # Window size
    use_plotly=True           # Use Plotly for interactive plotting
)

# Perform spectral analysis on multiple channels
results = data.spectral_analysis(
    channel_name=['Channel1', 'Channel2', 'Channel3'],  # List of channels
    method='psd',                                        # Analysis method
    subplot_layout=(2, 2),                               # Optional layout control
    use_plotly=True                                      # Use Plotly for interactive plotting
)

# Access results for individual channels
spec1 = results['Channel1'][0]  # Get spectrum object for Channel1
fig1 = results['Channel1'][1]   # Get figure for Channel1 (if individual plots were created)
```

## Core Functions

### Data Processing Functions

- **diff1d**: Calculate the derivative of a one-dimensional array
- **data_change_fs**: Change the sampling frequency of data
- **add_channel**: Add a new channel to the dataset
- **delete_channel**: Remove a channel from the dataset
- **filter_channel**: Apply filter to a channel
- **add_diff**: Add derivative of a channel as a new channel
- **add_diff2**: Add second derivative of a channel as a new channel
- **to_fullscale**: Convert data to full scale based on unit conversion
- **updateST**: Update statistical information for all channels

### Visualization Functions

#### plot_channel Function

The `plot_channel` function provides powerful data visualization capabilities, supporting both single-channel and multi-channel data with interactive features.

```python
data.plot_channel(
    ch_name,           # Channel name or list
    sseg=0,            # Segment index
    title=None,        # Title
    xlabel='Time (s)', # X-axis label
    ylabel=None,       # Y-axis label
    xlim=None,         # X-axis range
    ylim=None,         # Y-axis range
    grid=True,         # Show grid
    show=True,         # Show chart
    save_path=None,    # Save path
    use_plotly=True,   # Use Plotly
    downsampling=True, # Apply downsampling
    max_points=10000,  # Maximum points
    save_html=None,    # HTML save path
    dpi=300,           # Image DPI
    width=None,        # Chart width
    height=None,       # Chart height
    color=None,        # Line color
    alpha=0.8,         # Transparency
    linewidth=1,       # Line width
    figsize=(12, 4),   # Figure size
    stats=True,        # Show statistics
    table_width=0.3,   # Statistics table width
    column_widths=None # Column widths
)
```

#### plot_xy Function

```python
data.plot_xy(
    x_ch_idx,              # X-axis channel 
    y_ch_idx,              # Y-axis channel
    title=None,            # Plot title
    xlabel=None,           # X-axis label
    ylabel=None,           # Y-axis label
    xlim=None,             # X-axis limits
    ylim=None,             # Y-axis limits
    grid=True,             # Show grid
    show=True,             # Display plot
    use_plotly=True,       # Use Plotly for interactive plot
    downsampling=True,     # Apply downsampling for large datasets
    max_points=10000,      # Maximum points to display
    density_plot=False,    # Show density contours
    density_colorscale='Viridis', # Colorscale for density
    show_stats=False,      # Show statistical information
    fit_line=False,        # Fit linear regression line
    fit_color='red'        # Color for fit line
)
```

#### spectral_analysis Function

The `spectral_analysis` method allows for detailed frequency domain analysis of time series data.

```python
spec, fig = data.spectral_analysis(
    channel_name='Channel1',  # Channel name or index (or list of channels)
    method='cov',             # Analysis method: 'cov' or 'psd'
    L=1024,                   # Window size for spectral analysis
    filtered=False,           # Apply low-pass filtering
    cutoff_freq=None,         # Cutoff frequency for low-pass filter
    plot=True,                # Generate plot
    title=None,               # Plot title (string or list for multiple channels)
    xlim=None,                # X-axis limits (single tuple or list of tuples)
    ylim=None,                # Y-axis limits (single tuple or list of tuples)
    figsize=(10, 6),          # Figure size for Matplotlib
    show=True,                # Display plot
    save_path=None,           # Path to save static plot
    dpi=300,                  # DPI for saved static plot
    use_plotly=True,          # Use Plotly for interactive plotting
    save_html=None,           # Path to save interactive HTML plot
    width=None,               # Width of plot in pixels (Plotly only)
    height=None,              # Height of plot in pixels (Plotly only)
    subplot_layout=None       # Custom layout for multiple channel plots (rows, cols)
)
```

## Spectral Analysis Usage Examples

### Basic Usage

```python
from pydas import PyDAS
import numpy as np

# Load data
data = PyDAS('your_data_file.out')

# Perform basic spectral analysis on a channel
spec, fig = data.spectral_analysis(
    channel_name='Channel1',  # Channel name or index
    method='cov',             # Analysis method: 'cov' or 'psd'
    L=1024,                   # Window size
    use_plotly=True           # Use Plotly for interactive plotting
)

# Get spectral characteristics
# Handle the case where moment returns a tuple
moment_0 = spec.moment(0)
if isinstance(moment_0, tuple) and len(moment_0) > 0:
    if isinstance(moment_0[0], list):
        m0 = float(moment_0[0][0])
    else:
        m0 = float(moment_0[0])
else:
    m0 = float(moment_0)

# Calculate significant wave height
Hm0 = 4.0 * np.sqrt(m0)
print(f"Significant wave height: {Hm0:.3f} m")
```

### Advanced Spectral Analysis

```python
# Perform filtered spectral analysis
spec, fig = data.spectral_analysis(
    channel_name='Channel1',
    method='psd',            # Use Welch's method
    filtered=True,           # Apply low-pass filtering
    cutoff_freq=1.5,         # Cutoff frequency in Hz
    L=2048,                  # Larger window for better resolution
    title="Filtered Spectrum Analysis",
    xlim=(0, 2),             # Limit x-axis range
    use_plotly=True,
    save_html="spectrum.html",  # Save interactive plot as HTML
    width=1000,              # Plot width in pixels
    height=600               # Plot height in pixels
)
```

### Multi-Channel Spectral Analysis

```python
# Analyze multiple channels with subplots in a single figure
results = data.spectral_analysis(
    channel_name=['Channel1', 'Channel2', 'Channel3', 'Channel4'],
    method='cov',
    L=1024,
    use_plotly=True,
    subplot_layout=(2, 2),  # 2 rows, 2 columns layout
    save_html="multi_channel_spectrum.html"
)

# The results are returned as a dictionary with channel names as keys
for channel, (spec, fig) in results.items():
    if channel != 'combined':  # 'combined' key contains the combined figure
        # Process individual channel results
        if spec is not None:
            # Get spectral moments for this channel
            moment_0 = spec.moment(0)
            # Process the moment appropriately (handle tuple case)
            if isinstance(moment_0, tuple) and len(moment_0) > 0:
                if isinstance(moment_0[0], list):
                    m0 = float(moment_0[0][0])
                else:
                    m0 = float(moment_0[0])
            else:
                m0 = float(moment_0)
                
            # Calculate significant wave height
            Hm0 = 4.0 * np.sqrt(m0)
            print(f"Channel {channel} - Significant wave height: {Hm0:.3f} m")

# Access the combined figure from the results
combined_fig = results.get('combined', (None, None))[1]

# Analyze multiple channels with individual plots for each
individual_results = data.spectral_analysis(
    channel_name=['Channel1', 'Channel2'],
    method='psd',
    subplot_layout=(1, 1),  # Force individual plots (1x1 layout)
    save_html="individual_spectrums.html"  # Channel names will be appended
)
```

## Performance Optimization

PyDAS includes several optimizations for handling large datasets:

- Numba-accelerated computation-heavy functions
- Vectorized operations to replace loops
- Caching optimizations to avoid redundant calculations
- Memory-mapped file reading for large data files
- Adaptive downsampling for visualization
- WebGL rendering for large datasets
- Parallel processing to improve multi-core performance

## Additional Resources

- [API Documentation](docs/api.md)
- [Example Scripts](examples/) 