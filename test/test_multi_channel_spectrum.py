#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for multi-channel spectral analysis
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import PyDAS
from pydas import PyDAS

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
    
    # Create DataFrame with time and signals
    df = pd.DataFrame({
        'Time': t,
        'LowFreq': signal1,
        'MedFreq': signal2,
        'HighFreq': signal3,
        'MixedFreq': signal4
    })
    
    # Save to CSV
    filename = "test/synthetic_multi_channel.csv"
    df.to_csv(filename, index=False)
    print(f"Created test data file: {filename}")
    return filename

def test_single_channel_spectrum():
    """Test spectral analysis on a single channel."""
    print("\n===== Testing Single Channel Spectral Analysis =====")
    
    # Create or use test data
    data_file = create_test_data()
    
    # Load data
    data = PyDAS(data_file)
    print(f"Loaded data with {len(data.channels)} channels")
    print(f"Available channels: {data.channels}")
    
    # Analyze a single channel
    channel = 'LowFreq'
    print(f"\nAnalyzing channel: {channel}")
    
    try:
        # With matplotlib
        spec, fig = data.spectral_analysis(
            channel_name=channel,
            method='cov',
            L=512,
            use_plotly=False,
            save_path=f"test_single_{channel}_spectrum.png"
        )
        
        print(f"Successfully created spectrum for {channel}")
        
        # With plotly
        spec, fig = data.spectral_analysis(
            channel_name=channel,
            method='psd',
            L=512,
            use_plotly=True,
            save_html=f"test_single_{channel}_spectrum.html"
        )
        
        print(f"Successfully created interactive spectrum for {channel}")
        
    except Exception as e:
        print(f"Error in single channel test: {e}")

def test_multi_channel_spectrum():
    """Test spectral analysis on multiple channels with subplots."""
    print("\n===== Testing Multi-Channel Spectral Analysis =====")
    
    # Use previously created test data
    data_file = "test/synthetic_multi_channel.csv"
    
    # Load data
    data = PyDAS(data_file)
    
    # List of channels to analyze
    channels = ['LowFreq', 'MedFreq', 'HighFreq', 'MixedFreq']
    print(f"\nAnalyzing channels: {channels}")
    
    try:
        # With matplotlib - 2x2 layout
        results = data.spectral_analysis(
            channel_name=channels,
            method='cov',
            L=512,
            use_plotly=False,
            subplot_layout=(2, 2),
            save_path="test_multi_channel_spectrum.png"
        )
        
        print(f"Successfully created combined matplotlib spectrum for {len(channels)} channels")
        
        # With plotly - 2x2 layout
        results = data.spectral_analysis(
            channel_name=channels,
            method='psd',
            L=512,
            use_plotly=True,
            subplot_layout=(2, 2),
            save_html="test_multi_channel_spectrum.html"
        )
        
        print(f"Successfully created combined interactive spectrum for {len(channels)} channels")
        
        # Process each channel's results
        for channel, (spec, fig) in results.items():
            if channel != 'combined' and spec is not None:
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
                    print(f"Channel {channel} - Significant height: {Hm0:.3f}")
                except Exception as e:
                    print(f"Error processing results for {channel}: {e}")
                    
    except Exception as e:
        print(f"Error in multi-channel test: {e}")

def test_individual_plots():
    """Test creating individual plots for multiple channels."""
    print("\n===== Testing Individual Plots for Multiple Channels =====")
    
    # Use previously created test data
    data_file = "test/synthetic_multi_channel.csv"
    
    # Load data
    data = PyDAS(data_file)
    
    # List of channels to analyze
    channels = ['LowFreq', 'MedFreq']
    print(f"\nCreating individual plots for channels: {channels}")
    
    try:
        # Create individual plots
        results = data.spectral_analysis(
            channel_name=channels,
            method='cov',
            L=512,
            use_plotly=True,
            subplot_layout=(1, 1),  # Force individual plots
            save_html="test_individual_spectrum.html"
        )
        
        print(f"Successfully created individual plots for {len(channels)} channels")
        
        # Check that we have individual figures for each channel
        for channel, (spec, fig) in results.items():
            if channel != 'combined':
                if fig is not None:
                    print(f"Verified individual figure for {channel}")
                else:
                    print(f"Warning: No individual figure for {channel}")
                    
    except Exception as e:
        print(f"Error in individual plots test: {e}")

def test_custom_parameters():
    """Test customized parameters for multi-channel spectrum."""
    print("\n===== Testing Custom Parameters =====")
    
    # Use previously created test data
    data_file = "test/synthetic_multi_channel.csv"
    
    # Load data
    data = PyDAS(data_file)
    
    # Custom titles and axis limits
    channels = ['LowFreq', 'HighFreq']
    titles = ['Low Frequency Spectrum', 'High Frequency Spectrum']
    xlims = [(0, 0.5), (0, 2)]
    
    print(f"\nTesting custom parameters for channels: {channels}")
    
    try:
        # Create customized plots
        results = data.spectral_analysis(
            channel_name=channels,
            method='cov',
            L=512,
            title=titles,
            xlim=xlims,
            use_plotly=True,
            subplot_layout=(1, 2),
            save_html="test_custom_spectrum.html"
        )
        
        print(f"Successfully created custom plots with specific titles and xlim values")
                    
    except Exception as e:
        print(f"Error in custom parameters test: {e}")

if __name__ == "__main__":
    # Run tests
    test_single_channel_spectrum()
    test_multi_channel_spectrum()
    test_individual_plots()
    test_custom_parameters()
    
    print("\nAll tests completed.") 