#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for PyDAS spectral analysis with Plotly support
"""

import os
import sys
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import PyDAS
from pydas import PyDAS

def create_synthetic_data():
    """Create synthetic data for testing spectral analysis"""
    # Time vector: 20 seconds at 10 Hz sampling rate
    fs = 10.0  # Hz
    t = np.arange(0, 20, 1/fs)
    
    # Create signal with multiple frequency components
    f1, f2, f3 = 0.5, 1.0, 2.0  # Hz
    a1, a2, a3 = 1.0, 0.5, 0.25  # Amplitudes
    
    # Signal with three spectral peaks
    signal = a1 * np.sin(2 * np.pi * f1 * t) + \
             a2 * np.sin(2 * np.pi * f2 * t) + \
             a3 * np.sin(2 * np.pi * f3 * t) + \
             0.1 * np.random.randn(len(t))  # Add some noise
    
    # Create a DataFrame
    df = pd.DataFrame({
        'Time': t,
        'signal': signal,
        'signal2': signal * 0.8 + 0.2 * np.random.randn(len(t))
    })
    
    # Save to formatted file
    filename = "test/synthetic_data.csv"
    df.to_csv(filename, index=False)
    
    return filename

def main():
    """Test PyDAS Spectral Analysis with Plotly functionality"""
    
    # Create synthetic data
    print("Creating synthetic test data...")
    test_file = create_synthetic_data()
    print(f"Synthetic data saved to: {test_file}")
    
    # Load the test data
    try:
        data = PyDAS(filename=test_file, lam=50)  # 正确的初始化方式
        print(f"Loaded data with {len(data.channels)} channels")
        print(f"Available channels: {data.channels}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # If load is successful, proceed with tests
    if len(data.channels) > 0:
        print("Channels available:", data.channels)
        # Use the 'signal' channel for testing
        channel_name = 'signal'
        if channel_name not in data.channels:
            channel_name = data.channels[0]  # Fall back to first channel
            
        print(f"Using channel: {channel_name} for testing")
        
        # Test 1: Basic spectral analysis with Plotly
        print("\nTest 1: Basic Spectral Analysis with Plotly")
        try:
            spec, fig = data.spectral_analysis(
                channel_name, 
                L=128,  # Smaller window for synthetic data
                method='cov',
                use_plotly=True,
                title="Spectral Analysis with Plotly (Cov)",
                save_html="test_cov_spectrum.html"
            )
            print("Spectral analysis completed")
            if spec is not None:
                print("Spectrum computed successfully")
                try:
                    m0 = float(spec.moment(0))
                    print(f"Zeroth moment (m0): {m0:.5f}")
                    print(f"Significant wave height: {4.0 * np.sqrt(m0):.5f}")
                except Exception as e:
                    print(f"Could not compute spectral characteristics: {e}")
            if fig is not None:
                print("Plot created successfully")
        except Exception as e:
            print(f"Error in Test 1: {e}")
        
        # Test 2: Spectral analysis with filtered data
        print("\nTest 2: Filtered Spectral Analysis with Plotly")
        try:
            spec, fig = data.spectral_analysis(
                channel_name, 
                L=128,
                method='psd',
                filtered=True,
                cutoff_freq=1.5,
                use_plotly=True,
                title="Filtered Spectral Analysis with Plotly (PSD)",
                save_html="test_filtered_spectrum.html"
            )
            print("Filtered spectral analysis completed")
        except Exception as e:
            print(f"Error in Test 2: {e}")
        
        # Test 3: Matplotlib comparison
        print("\nTest 3: Matplotlib Comparison")
        try:
            spec, fig = data.spectral_analysis(
                channel_name, 
                L=128,
                method='cov',
                use_plotly=False,
                title="Spectral Analysis with Matplotlib",
                save_path="test_matplotlib_spectrum.png"
            )
            print("Matplotlib spectral analysis completed")
        except Exception as e:
            print(f"Error in Test 3: {e}")
        
        # Test 4: Full Scale analysis with Plotly
        print("\nTest 4: Full Scale Spectral Analysis with Plotly")
        try:
            spec_fullscale, fig_fullscale = data.spectral_analysis(
                channel_name, 
                L=128,
                method='cov',
                use_plotly=True,
                title="Full Scale Spectral Analysis",
                save_html="test_fullscale_spectrum.html",
                fullscale=True  # Enable full scale analysis
            )
            
            # Compare with model scale results
            if spec is not None and spec_fullscale is not None:
                try:
                    # 获取模型尺度谱矩
                    m0_model = spec.moment(0)
                    if isinstance(m0_model, tuple) and len(m0_model) > 0:
                        if isinstance(m0_model[0], list) and len(m0_model[0]) > 0:
                            m0_model = float(m0_model[0][0])
                        else:
                            m0_model = float(m0_model[0])
                    else:
                        m0_model = float(m0_model)
                    
                    # 获取全尺度谱矩
                    m0_full = spec_fullscale.moment(0)
                    if isinstance(m0_full, tuple) and len(m0_full) > 0:
                        if isinstance(m0_full[0], list) and len(m0_full[0]) > 0:
                            m0_full = float(m0_full[0][0]) 
                        else:
                            m0_full = float(m0_full[0])
                    else:
                        m0_full = float(m0_full)
                    
                    # 计算对应的波高
                    Hm0_model = 4.0 * np.sqrt(m0_model)
                    Hm0_full = 4.0 * np.sqrt(m0_full)
                    
                    # 获取更高阶谱矩用于计算周期
                    m1_model = spec.moment(1)
                    if isinstance(m1_model, tuple) and len(m1_model) > 0:
                        if isinstance(m1_model[0], list) and len(m1_model[0]) > 0:
                            m1_model = float(m1_model[0][0])
                        else:
                            m1_model = float(m1_model[0])
                    else:
                        m1_model = float(m1_model)
                    
                    m2_model = spec.moment(2)
                    if isinstance(m2_model, tuple) and len(m2_model) > 0:
                        if isinstance(m2_model[0], list) and len(m2_model[0]) > 0:
                            m2_model = float(m2_model[0][0])
                        else:
                            m2_model = float(m2_model[0])
                    else:
                        m2_model = float(m2_model)
                    
                    m1_full = spec_fullscale.moment(1)
                    if isinstance(m1_full, tuple) and len(m1_full) > 0:
                        if isinstance(m1_full[0], list) and len(m1_full[0]) > 0:
                            m1_full = float(m1_full[0][0])
                        else:
                            m1_full = float(m1_full[0])
                    else:
                        m1_full = float(m1_full)
                    
                    m2_full = spec_fullscale.moment(2)
                    if isinstance(m2_full, tuple) and len(m2_full) > 0:
                        if isinstance(m2_full[0], list) and len(m2_full[0]) > 0:
                            m2_full = float(m2_full[0][0])
                        else:
                            m2_full = float(m2_full[0])
                    else:
                        m2_full = float(m2_full)
                    
                    # 计算周期
                    Tm01_model = 2 * np.pi * m0_model / m1_model if m0_model is not None and m1_model is not None and m1_model != 0 else None
                    Tm02_model = 2 * np.pi * np.sqrt(m0_model / m2_model) if m0_model is not None and m2_model is not None and m2_model != 0 else None
                    
                    Tm01_full = 2 * np.pi * m0_full / m1_full if m0_full is not None and m1_full is not None and m1_full != 0 else None
                    Tm02_full = 2 * np.pi * np.sqrt(m0_full / m2_full) if m0_full is not None and m2_full is not None and m2_full != 0 else None
                    
                    # 打印比较结果
                    print(f"Model scale Hm0: {Hm0_model:.4f}")
                    print(f"Full scale Hm0: {Hm0_full:.4f}")
                    print(f"Ratio Hm0 (should be close to λ=50): {Hm0_full/Hm0_model:.4f}")
                    
                    print(f"Model scale Tm01: {Tm01_model:.4f}")
                    print(f"Full scale Tm01: {Tm01_full:.4f}")
                    print(f"Ratio Tm01 (should be close to √λ=7.07): {Tm01_full/Tm01_model:.4f}")
                    
                    print(f"Model scale Tm02: {Tm02_model:.4f}")
                    print(f"Full scale Tm02: {Tm02_full:.4f}")
                    print(f"Ratio Tm02 (should be close to √λ=7.07): {Tm02_full/Tm02_model:.4f}")
                except Exception as e:
                    print(f"Error comparing spectral moments: {e}")
            
            print("Full scale spectral analysis completed")
        except Exception as e:
            print(f"Error in Test 4: {e}")
        
        print("\nTests completed.")
    else:
        print("No channels available for analysis")

if __name__ == "__main__":
    main() 