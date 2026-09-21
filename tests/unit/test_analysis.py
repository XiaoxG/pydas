# tests/unit/test_analysis.py
import pytest
import numpy as np
import pandas as pd
from pydas.analysis import spectral_analysis, statistic_analysis, extreme_analysis

def test_spectral_analysis_basic(pydas_instance):
    """Test basic spectral analysis invocation."""
    # Run spectral analysis
    spec = spectral_analysis(pydas_instance, channel_name='Wave1', method='cov')
    
    # Check return type and content
    assert spec is not None
    assert hasattr(spec, 'args')  # Frequencies
    assert hasattr(spec, 'data')  # Energy Density
    assert len(spec.args) > 0
    assert len(spec.data) > 0

def test_spectral_analysis_fullscale(pydas_instance):
    """Test spectral analysis with fullscale conversion."""
    pydas_instance.__lam__ = 25.0
    spec = spectral_analysis(pydas_instance, channel_name='Wave1', fullscale=True, freq_range=(0, 5))
    
    assert spec is not None
    assert len(spec.args) > 0

def test_statistic_analysis_basic(pydas_instance):
    """Test basic statistical analysis."""
    stats_df = statistic_analysis(pydas_instance, ch_name='Wave1', sseg=0)
    
    assert stats_df is not None
    assert isinstance(stats_df, pd.DataFrame)
    
    # Check core columns
    expected_cols = ['Mean', 'Std', 'Min', 'Max', 'Median', 'RMS', 'Range', 'Peak-to-Peak', 'Zero-Crossings', 'Unit']
    for col in expected_cols:
        assert col in stats_df.columns
        
    # Test value plausibility (mean should be close to 0 for the synthetic data)
    assert np.isclose(stats_df.loc['Wave1', 'Mean'], 0, atol=0.5)

def test_statistic_analysis_advanced(pydas_instance):
    """Test advanced statistical analysis with higher order moments."""
    stats_df = statistic_analysis(pydas_instance, ch_name='Wave1', sseg=0, advanced=True)
    
    assert stats_df is not None
    assert 'Skewness' in stats_df.columns
    assert 'Kurtosis' in stats_df.columns
    assert 'Crest Factor' in stats_df.columns

def test_extreme_analysis_basic(pydas_instance):
    """Test basic extreme analysis."""
    # Since our synthetic data is a simple sine wave + noise, extreme_analysis should find peaks.
    res = extreme_analysis(pydas_instance, ch_name='Wave1', peak_height=0.1, visualization=False)
    
    # It should return a dictionary
    assert res is not None
    assert isinstance(res, dict)
    
    # Check keys
    assert 'all_peaks' in res
    assert 'duration_seconds' in res
    
    # Check if peaks were found
    if 'peaks_positive' in res and len(res['peaks_positive']) > 0:
        assert len(res['all_peaks']) > 0
