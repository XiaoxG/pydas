# tests/unit/test_reporting.py
import pytest
import numpy as np
import pandas as pd
import os
from pydas.reporting import analyze_channel_data, channel_report, wave_report

def test_analyze_channel_data():
    """Test channel statistics calculation."""
    # Create simple sine wave
    t = np.linspace(0, 10, 1000)
    data = 2.0 * np.sin(2 * np.pi * 1.0 * t) # 1Hz sine wave, amp=2.0
    
    # Calculate dt
    dt = t[1] - t[0]
    duration = 10.0 / 3600.0 # hours
    
    res = analyze_channel_data(
        data_scaled=data, 
        zerocrossing_analysis=True, 
        amplitude_analysis=True,
        data_duration_hours=duration,
        dt=dt
    )
    
    # Check return type
    assert isinstance(res, dict)
    
    # Check values
    assert np.isclose(res['maximum'], 2.0, atol=0.1)
    assert np.isclose(res['minimum'], -2.0, atol=0.1)
    assert np.isclose(res['mean'], 0.0, atol=0.1)
    assert res['zero_upcross'] > 5  # Should find around 10 upcrossings

def test_channel_report(pydas_instance, tmp_path):
    """Test channel_report Excel generation."""
    out_file = tmp_path / "test_channel_report.xlsx"
    
    pydas_instance.__lam__ = 1.0
    
    # Run channel report
    result = channel_report(
        pydas_instance, output_file=str(out_file), sseg=0, fullscale=False
    )
    
    assert isinstance(result, pd.DataFrame)
    # Check if file was created
    assert out_file.exists()
    assert os.path.getsize(out_file) > 100

def test_wave_report(pydas_instance, tmp_path):
    """Test wave_report generation."""
    out_file = tmp_path / "test_wave_report.png"
    
    pydas_instance.__lam__ = 1.0
    
    # Run wave report
    fig = wave_report(pydas_instance, ch_name='Wave1', sseg=0, save_path=str(out_file))
    
    assert fig is not None
    # Check if file was created
    assert out_file.exists()
    assert os.path.getsize(out_file) > 100


def test_print_info_returns_dataframe(pydas_instance):
    df = pydas_instance.print_info()
    assert isinstance(df, pd.DataFrame)
    assert "Filename" in df.index


def test_channel_report_regular_excludes_mpm(pydas_instance, tmp_path):
    """wave_type='regular' must not include MPM / EEV columns."""
    out_file = tmp_path / "regular_report.xlsx"
    result = channel_report(
        pydas_instance,
        output_file=str(out_file),
        sseg=0,
        fullscale=False,
        wave_type="regular",
        include_charts=False,
    )
    assert isinstance(result, pd.DataFrame)
    columns = list(result.columns)
    for banned in ("MPM_pos", "MPM_neg", "EEV_pos", "EEV_neg"):
        assert banned not in columns
