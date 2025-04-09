#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for multi-channel spectral analysis - direct approach
This script tests the multi-channel functionality without depending on PyDAS file loading
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import waveModel directly
try:
    from waveModel.timeseries import TimeSeries
    from waveModel.datacontainer import PlotData
except ImportError:
    print("waveModel module not found. Make sure it's properly installed.")
    sys.exit(1)
    
# Import for plotting
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Plotly not available, will use Matplotlib only.")

def create_test_data():
    """Create synthetic data with multiple channels for testing."""
    # Time settings
    fs = 10.0  # Hz
    t = np.arange(0, 120, 1/fs)  # 120 seconds of data
    
    # Create different signals with known frequency components
    # Channel 1: Low frequency components
    f1_1, f1_2 = 0.1, 0.2  # Hz
    a1_1, a1_2 = 1.0, 0.5  # Amplitudes
    signal1 = a1_1 * np.sin(2 * np.pi * f1_1 * t) + a1_2 * np.sin(2 * np.pi * f1_2 * t) + 0.1 * np.random.randn(len(t))
    
    # Channel 2: Medium frequency components
    f2_1, f2_2 = 0.5, 0.7  # Hz
    a2_1, a2_2 = 0.8, 0.6  # Amplitudes
    signal2 = a2_1 * np.sin(2 * np.pi * f2_1 * t) + a2_2 * np.sin(2 * np.pi * f2_2 * t) + 0.1 * np.random.randn(len(t))
    
    # Channel 3: High frequency components
    f3_1, f3_2 = 1.0, 1.5  # Hz
    a3_1, a3_2 = 0.7, 0.4  # Amplitudes
    signal3 = a3_1 * np.sin(2 * np.pi * f3_1 * t) + a3_2 * np.sin(2 * np.pi * f3_2 * t) + 0.1 * np.random.randn(len(t))
    
    # Channel 4: Mixed frequency components
    signal4 = 0.5 * signal1 + 0.3 * signal2 + 0.2 * signal3
    
    # Create dictionary of signals
    signals = {
        'LowFreq': signal1,
        'MedFreq': signal2,
        'HighFreq': signal3,
        'MixedFreq': signal4
    }
    
    return t, signals, fs

def create_spectrum(signal, t, fs, L=512, method='cov'):
    """Create spectrum from signal data."""
    # Create PlotData and TimeSeries objects
    plot_data = PlotData(signal, t)
    ts = TimeSeries(plot_data.data, plot_data.args)
    
    # Compute spectrum
    spec = ts.tospecdata(L=L, method=method)
    return spec

def plot_spectrum_matplotlib(specs, channel_names, subplot_layout=None):
    """Plot spectra using Matplotlib."""
    n_channels = len(specs)
    
    # Determine subplot layout if not provided
    if subplot_layout is None:
        n_cols = min(3, n_channels)
        n_rows = int(np.ceil(n_channels / n_cols))
        subplot_layout = (n_rows, n_cols)
    else:
        n_rows, n_cols = subplot_layout
    
    # Create figure and axes
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), squeeze=False)
    
    # Plot each spectrum
    for i, (spec, channel) in enumerate(zip(specs, channel_names)):
        row = i // n_cols
        col = i % n_cols
        ax = axs[row, col]
        
        # Get frequency and spectral density
        if hasattr(spec, 'args'):
            if isinstance(spec.args, tuple) and len(spec.args) > 0:
                f = spec.args[0] / (2 * np.pi)
            else:
                f = spec.args / (2 * np.pi)
                
            if hasattr(spec, 'data'):
                S = spec.data
            elif hasattr(spec, 'S'):
                S = spec.S
            else:
                print(f"Warning: No spectral density data found for {channel}")
                continue
            
            # Ensure f and S have matching dimensions
            if f.ndim > 1:
                f = f.flatten()
            if S.ndim > 1:
                S = S.flatten()
            
            if len(f) != len(S):
                min_len = min(len(f), len(S))
                f = f[:min_len]
                S = S[:min_len]
            
            # Plot
            ax.plot(f, S, 'b-', linewidth=2)
            ax.set_title(f"Spectrum of {channel}")
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Spectral Density')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Try to calculate and display spectral characteristics
            try:
                # Extract the moment - handle tuple case
                moment_0 = spec.moment(0)
                if isinstance(moment_0, tuple) and len(moment_0) > 0:
                    if isinstance(moment_0[0], list) and len(moment_0[0]) > 0:
                        m0 = float(moment_0[0][0])
                    else:
                        m0 = float(moment_0[0])
                else:
                    m0 = float(moment_0)
                    
                # Calculate significant wave height
                Hm0 = 4.0 * np.sqrt(m0)
                
                # Display on the plot
                stats_text = f"Hm0 = {Hm0:.2f}\nm0 = {m0:.4f}"
                ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
                print(f"Channel {channel} - Significant height: {Hm0:.3f}")
            except Exception as e:
                print(f"Could not compute spectral characteristics for {channel}: {e}")
    
    # Remove any unused subplots
    for i in range(n_channels, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        fig.delaxes(axs[row, col])
    
    plt.tight_layout()
    plt.savefig("test_multi_direct_matplotlib.png", dpi=300)
    plt.show()
    
    return fig

def plot_spectrum_plotly(specs, channel_names, subplot_layout=None):
    """Plot spectra using Plotly."""
    if not HAS_PLOTLY:
        print("Plotly is not available, skipping Plotly plot.")
        return None
        
    n_channels = len(specs)
    
    # Determine subplot layout if not provided
    if subplot_layout is None:
        n_cols = min(3, n_channels)
        n_rows = int(np.ceil(n_channels / n_cols))
        subplot_layout = (n_rows, n_cols)
    else:
        n_rows, n_cols = subplot_layout
    
    # Create subplots
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[f"Spectrum of {ch}" for ch in channel_names])
    
    # Add each spectrum
    for i, (spec, channel) in enumerate(zip(specs, channel_names)):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        # Get frequency and spectral density
        if hasattr(spec, 'args'):
            if isinstance(spec.args, tuple) and len(spec.args) > 0:
                f = spec.args[0] / (2 * np.pi)
            else:
                f = spec.args / (2 * np.pi)
                
            if hasattr(spec, 'data'):
                S = spec.data
            elif hasattr(spec, 'S'):
                S = spec.S
            else:
                print(f"Warning: No spectral density data found for {channel}")
                continue
            
            # Ensure f and S have matching dimensions
            if f.ndim > 1:
                f = f.flatten()
            if S.ndim > 1:
                S = S.flatten()
            
            if len(f) != len(S):
                min_len = min(len(f), len(S))
                f = f[:min_len]
                S = S[:min_len]
            
            # Add trace
            fig.add_trace(
                go.Scatter(
                    x=f,
                    y=S,
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name=channel
                ),
                row=row,
                col=col
            )
            
            # Try to calculate spectral characteristics
            try:
                # Extract the moment - handle tuple case
                moment_0 = spec.moment(0)
                if isinstance(moment_0, tuple) and len(moment_0) > 0:
                    if isinstance(moment_0[0], list) and len(moment_0[0]) > 0:
                        m0 = float(moment_0[0][0])
                    else:
                        m0 = float(moment_0[0])
                else:
                    m0 = float(moment_0)
                    
                # Calculate significant wave height
                Hm0 = 4.0 * np.sqrt(m0)
                
                # Add annotation
                stats_text = f"Hm0 = {Hm0:.2f}<br>m0 = {m0:.4f}"
                fig.add_annotation(
                    x=0.95,
                    y=0.95,
                    xref=f"x{i+1} domain",
                    yref=f"y{i+1} domain",
                    text=stats_text,
                    showarrow=False,
                    align="right",
                    bgcolor="rgba(255, 255, 255, 0.7)",
                    bordercolor="black",
                    borderwidth=1,
                )
            except Exception as e:
                print(f"Could not compute spectral characteristics for {channel}: {e}")
    
    # Update layout
    fig.update_layout(
        width=800,
        height=600,
        showlegend=False
    )
    
    # Show and save
    fig.write_html("test_multi_direct_plotly.html")
    fig.show()
    
    return fig

def test_direct_multi_channel():
    """Test multi-channel spectral analysis using direct TimeSeries creation."""
    print("\n===== Testing Direct Multi-Channel Spectral Analysis =====")
    
    # Create test data
    t, signals, fs = create_test_data()
    
    # Select channels to analyze
    channels = ['LowFreq', 'MedFreq', 'HighFreq', 'MixedFreq']
    
    # Compute spectra for each channel
    specs = []
    for channel in channels:
        signal = signals[channel]
        spec = create_spectrum(signal, t, fs, L=512, method='cov')
        specs.append(spec)
        print(f"Created spectrum for {channel}")
    
    # Plot with Matplotlib
    print("\nCreating Matplotlib plot...")
    plot_spectrum_matplotlib(specs, channels, subplot_layout=(2, 2))
    
    # Plot with Plotly if available
    if HAS_PLOTLY:
        print("\nCreating Plotly plot...")
        plot_spectrum_plotly(specs, channels, subplot_layout=(2, 2))

if __name__ == "__main__":
    test_direct_multi_channel() 