"""
PyDAS Plot Module
================
This module contains plotting functions for PyDAS data.
"""

import logging
import numpy as np
import pandas as pd
import os

# Set up logging
logger = logging.getLogger('pydas.plot')

# Check whether the optional plotly-resampler library is available
try:
    import plotly_resampler
    HAS_PLOTLY_RESAMPLER = True
    logger.info("plotly-resampler loaded; large-dataset optimisation enabled.")
except ImportError:
    HAS_PLOTLY_RESAMPLER = False
    logger.info("plotly-resampler not found. Install for large-dataset optimisation: pip install plotly-resampler")

def use_webgl_rendering(fig, data_length=None, threshold=10000):
    """
    Convert a Plotly figure to use WebGL rendering for better performance on large datasets.

    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        The Plotly figure to optimise.
    data_length : int, optional
        Number of data points. Estimated from the figure data when *None*.
    threshold : int, optional
        Data-point count above which WebGL rendering is activated, default is 10000.

    Returns:
    --------
    plotly.graph_objects.Figure
        The optimised figure object.
    """
    try:
        import plotly.graph_objects as go
        
        # Estimate data length if not provided
        if data_length is None:
            data_length = 0
            for trace in fig.data:
                if hasattr(trace, 'x') and trace.x is not None:
                    data_length = max(data_length, len(trace.x))
                    
        # Skip optimisation when below the threshold
        if data_length < threshold:
            return fig
            
        # Upgrade all Scatter traces to ScatterGL for WebGL acceleration
        for i, trace in enumerate(fig.data):
            if hasattr(trace, 'type') and trace.type == 'scatter':
                # Copy all trace attributes
                trace_dict = trace.to_plotly_json()
                # Switch renderer to WebGL
                trace_dict['type'] = 'scattergl'
                # Replace the original trace
                fig.data[i] = trace_dict
                
        # Additional WebGL layout settings
        fig.update_layout(
            uirevision='constant',  # Preserve UI state across updates
            hovermode='closest',    # Optimise hover performance
        )
        
        logger.info(f"WebGL rendering enabled ({data_length} data points).")
        return fig
    except Exception as e:
        logger.warning(f"Failed to enable WebGL rendering: {e}")
        return fig  # Return unmodified figure

def create_resampable_plot(x, y, name=None, title=None, n_shown_samples=5000):
    """
    Create a dynamically resampable plot suitable for very large time series datasets.

    Parameters:
    -----------
    x : numpy.ndarray
        X-axis data.
    y : numpy.ndarray
        Y-axis data.
    name : str, optional
        Trace name shown in the legend.
    title : str, optional
        Figure title.
    n_shown_samples : int, optional
        Number of samples displayed initially, default is 5000.

    Returns:
    --------
    FigureResampler or None
        Resampable figure object, or *None* if the library is unavailable.
    """
    if not HAS_PLOTLY_RESAMPLER:
        logger.warning("plotly_resampler is not installed; dynamic resampling is unavailable.")
        return None
        
    try:
        from plotly_resampler import FigureResampler
        import plotly.graph_objects as go
        
        # Build a base figure
        fig = go.Figure()
        
        # Add the data trace
        trace_name = name if name else "data"
        fig.add_trace(go.Scatter(x=x, y=y, name=trace_name))
        
        # Apply title if provided
        if title:
            fig.update_layout(title=title)
            
        # Wrap in a FigureResampler for dynamic downsampling
        fig_resampler = FigureResampler(
            fig, 
            default_n_shown_samples=n_shown_samples,
            resampled_trace_prefix_suffix=(None, " (resampled)")
        )
        
        logger.info(f"Resampable figure created (data points: {len(x)}, shown: {n_shown_samples}).")
        return fig_resampler
    except Exception as e:
        logger.warning(f"Failed to create resampable figure: {e}")
        return None

def lttb_downsample(x, y, n_out):
    """
    Downsample data using the LTTB (Largest-Triangle-Three-Buckets) algorithm,
    preserving the visual shape of the signal.

    Parameters:
    -----------
    x : numpy.ndarray
        X-axis data.
    y : numpy.ndarray
        Y-axis data.
    n_out : int
        Number of output points.

    Returns:
    --------
    tuple
        ``(x_sampled, y_sampled)`` – downsampled data points.
    """
    n = len(x)
    if n <= n_out:
        return x, y
        
    # Always keep the first and last points
    sampled_x = np.zeros(n_out)
    sampled_y = np.zeros(n_out)
    sampled_x[0] = x[0]
    sampled_y[0] = y[0]
    sampled_x[n_out-1] = x[n-1]
    sampled_y[n_out-1] = y[n-1]
    
    # Compute bucket width
    bucket_size = (n - 2) / (n_out - 2)
    
    # For each output point, find the sample that forms the largest triangle
    for i in range(1, n_out-1):
        # Bucket boundaries for points a, b, c
        a = int((i - 1) * bucket_size) + 1
        b = int(i * bucket_size) + 1
        c = int((i + 1) * bucket_size) + 1 if i < n_out-2 else n-1
        
        # Point a: last selected point
        point_a_x = sampled_x[i-1]
        point_a_y = sampled_y[i-1]
        
        # Point c: average of the next bucket
        point_c_x = x[c-1]
        point_c_y = y[c-1]
        
        # Search bucket b for the point that maximises triangle area
        max_area = -1
        max_idx = b
        
        for j in range(a, b):
            area = abs(
                (point_a_x - point_c_x) * (y[j] - point_a_y) - 
                (point_a_x - x[j]) * (point_c_y - point_a_y)
            ) * 0.5
            if area > max_area:
                max_area = area
                max_idx = j
        
        # Store the best point
        sampled_x[i] = x[max_idx]
        sampled_y[i] = y[max_idx]
    
    return sampled_x, sampled_y

# Global plot configuration
PLOT_CONFIG = {
    # Figure size presets
    'figsize': {
        'small': (8, 6),
        'medium': (12, 8),
        'large': (16, 10),
        'wide': (12, 4),
        'square': (8, 8),
        'tall': (6, 8),
    },
    
    # Font settings
    'font': {
        'family': 'Arial, sans-serif',
        'size': {
            'small': 8,
            'medium': 10,
            'large': 12,
            'title': 14,
            'label': 10,
            'tick': 8,
            'legend': 9,
            'annotation': 9,
        },
        'weight': 'normal',
    },
    
    # Plot style themes
    'style': {
        'matplotlib': {
            'default': 'seaborn-v0_8-whitegrid',
            'light': 'seaborn-v0_8-whitegrid',
            'dark': 'seaborn-v0_8-dark',
            'paper': 'seaborn-v0_8-paper',
            'talk': 'seaborn-v0_8-talk',
            'colorblind': 'seaborn-v0_8-colorblind',
        },
        'plotly': {
            'default': 'plotly_white',
            'light': 'plotly_white',
            'dark': 'plotly_dark',
            'paper': 'ggplot2',
        },
        'seaborn': {
            'default': 'whitegrid',
            'light': 'whitegrid',
            'dark': 'darkgrid',
            'paper': 'ticks',
            'talk': 'whitegrid',
        },
    },
    
    # Default colours
    'colors': {
        'default': 'tab10',  # matplotlib colormap name
        'sequential': 'viridis',
        'diverging': 'coolwarm',
        'qualitative': 'tab10',
        'single': 'blue',
        'fit': 'red',
        'background': 'white',
        'grid': '#CCCCCC',
        'annotation': 'gray',
    },
    
    # Plot element defaults
    'elements': {
        'line_width': 1.5,
        'marker_size': 5,
        'alpha': 0.8,
        'grid': True,
        'dpi': 300,
        'edge_color': '#000000',
    },
    
    # Statistics table configuration
    'stats': {
        'table_width': 0.3,
        'table_height': 0.2,
        'table_font_size': 10,
        'header_color': '#EEEEEE',
    },
}

# Re-export commonly used config values for convenience
DEFAULT_FIGSIZE = PLOT_CONFIG['figsize']['medium']
DEFAULT_FONT_SIZE = PLOT_CONFIG['font']['size']['medium']
DEFAULT_DPI = PLOT_CONFIG['elements']['dpi']

def get_plot_backend(backend=None):
    """
    Return the requested plotting backend, falling back to available alternatives.

    Parameters:
    -----------
    backend : str or None
        Backend to use: 'plotly', 'matplotlib', 'seaborn', or *None* (auto-select).

    Returns:
    --------
    str
        Name of the backend that will be used.
    """
    if backend is None:
        # Try backends in order of preference
        try:
            import plotly
            return 'plotly'
        except ImportError:
            try:
                import seaborn
                return 'seaborn'
            except ImportError:
                try:
                    import matplotlib
                    return 'matplotlib'
                except ImportError:
                    logger.error("No available plotting backend found. Install plotly, seaborn, or matplotlib.")
                    return None
    
    # Check whether the requested backend is available
    if backend.lower() == 'plotly':
        try:
            import plotly
            return 'plotly'
        except ImportError:
            logger.warning("Plotly not available. Trying alternative backends.")
            return get_plot_backend(None)
    
    elif backend.lower() == 'seaborn':
        try:
            import seaborn
            return 'seaborn'
        except ImportError:
            logger.warning("Seaborn not available. Trying alternative backends.")
            return get_plot_backend(None)
    
    elif backend.lower() == 'matplotlib':
        try:
            import matplotlib
            return 'matplotlib'
        except ImportError:
            logger.warning("Matplotlib not available. Trying alternative backends.")
            return get_plot_backend(None)
    
    else:
        logger.warning(f"Unknown backend '{backend}'. Using default.")
        return get_plot_backend(None)

def apply_style(backend, style=None):
    """
    Apply the specified plotting style for the given backend.

    Parameters:
    -----------
    backend : str
        Plotting backend: 'plotly', 'matplotlib', or 'seaborn'.
    style : str or None
        Style name. Uses the default style when *None*.

    Returns:
    --------
    None
    """
    if backend is None:
        return
    
    # Default to the 'default' style when none is specified
    if style is None:
        style = 'default'
    
    if backend.lower() == 'matplotlib':
        try:
            import matplotlib.pyplot as plt
            plt.style.use(PLOT_CONFIG['style']['matplotlib'].get(style.lower(), 
                                                              PLOT_CONFIG['style']['matplotlib']['default']))
        except Exception as e:
            logger.warning(f"Failed to apply matplotlib style: {e}")
    
    elif backend.lower() == 'seaborn':
        try:
            import seaborn as sns
            sns.set_theme(style=PLOT_CONFIG['style']['seaborn'].get(style.lower(), 
                                                               PLOT_CONFIG['style']['seaborn']['default']))
        except Exception as e:
            logger.warning(f"Failed to apply seaborn style: {e}")
    
    # Plotly styles are applied when the figure is created

def validate_channel(pydas_obj, ch_idx):
    """
    Validate and convert channel index or name to channel name.
    
    Parameters:
    -----------
    pydas_obj : PyDAS
        PyDAS object containing the data
    ch_idx : int, str, or list
        Channel index, name, or list of indices/names
        
    Returns:
    --------
    str or list or None
        Channel name(s), or None if invalid
    """
    if isinstance(ch_idx, int):
        if ch_idx < len(pydas_obj.chInfo):
            return pydas_obj.chInfo.iloc[ch_idx]['Name']
        else:
            logger.error(f"Channel index {ch_idx} out of bounds.")
            return None
    elif isinstance(ch_idx, str):
        if ch_idx in pydas_obj.chInfo['Name'].values:
            return ch_idx
        else:
            logger.error(f"Channel '{ch_idx}' not found.")
            return None
    elif isinstance(ch_idx, list):
        # Process channel list - support both index lists and name lists
        result = []
        
        # If list of strings (channel names), validate each name
        if all(isinstance(item, str) for item in ch_idx):
            for name in ch_idx:
                if name in pydas_obj.chInfo['Name'].values:
                    result.append(name)
                else:
                    logger.warning(f"Channel '{name}' not found, skipping.")
            
            if not result:
                logger.error("No valid channels to plot.")
                return None
            return result
            
        # If list of integers (channel indices), convert to names
        elif all(isinstance(item, int) for item in ch_idx):
            for idx in ch_idx:
                if idx < len(pydas_obj.chInfo):
                    result.append(pydas_obj.chInfo.iloc[idx]['Name'])
                else:
                    logger.warning(f"Channel index {idx} out of bounds, skipping.")
            
            if not result:
                logger.error("No valid channels to plot.")
                return None
            return result
        else:
            logger.error("Channel list must contain all strings or all integers.")
            return None
    else:
        logger.error("Channel identifier must be an integer, string, or list.")
        return None


# Re-export from submodules for backward compatibility
from .timeseries import plot_channel
from .statistics import plot_histogram, _plot_statistics_mpl, _plot_statistics_plotly, boxplot_channel
from .scatter import plot_xy, _lttb_downsample
from .extreme import _detect_peaks, plot_extreme_analysis

__all__ = [
    # Config
    "PLOT_CONFIG", "DEFAULT_FIGSIZE", "DEFAULT_FONT_SIZE", "DEFAULT_DPI",
    # Utilities
    "use_webgl_rendering", "create_resampable_plot", "lttb_downsample",
    "get_plot_backend", "apply_style", "validate_channel",
    "HAS_PLOTLY_RESAMPLER",
    # Submodule functions
    "plot_channel",
    "plot_histogram", "_plot_statistics_mpl", "_plot_statistics_plotly", "boxplot_channel",
    "plot_xy", "_lttb_downsample",
    "_detect_peaks", "plot_extreme_analysis",
]
