# tests/integration/test_wc01_full_flow.py
import pytest
import os
import numpy as np
from pydas import PyDAS
from pydas.analysis import spectral_analysis, statistic_analysis, extreme_analysis

def test_wc01_full_workflow(tmp_path):
    """
    Test the full workflow on the legacy WC01.out file.
    Validates end-to-end functionality including data loading, washing, 
    processing, analysis, reports, and exporting.
    """
    file_path = "c:/coding/pydas/tests/legacy/WC01.out"
    
    # 1. Load Data
    assert os.path.exists(file_path), f"Test file {file_path} not found"
    data = PyDAS(filename=file_path, lam=25)
    
    # Ensure it's loaded properly
    assert data.__segN__ >= 1
    assert data.__chN__ >= 1
    
    # Identify a wave channel (assuming there's one named 'Wave' or similar)
    wave_ch = None
    for col in data.data[0].columns:
        if 'Wave' in col or col.lower() in ['wave1', 'wave']:
            wave_ch = col
            break
            
    if wave_ch is None:
        wave_ch = data.data[0].columns[0] # Fallback to first channel
        
    # 2. Data Wash
    # In fullscale, data wash with default 10 std might trim extremes
    data.data_wash(ChName=wave_ch, threshold=5)
    
    # 3. Data Processing
    # Remove mean using full data mode
    data.remove_mean(chName=wave_ch)
    
    # Apply lowpass filter (e.g., above 3 Hz in model scale is noise)
    data.apply_lowpass_filter(chName=wave_ch, cutoffull=3.0)
    
    # 4. Statistical Analysis
    stats_df = statistic_analysis(data, ch_name=wave_ch, advanced=True)
    assert not stats_df.empty
    
    # 5. Spectral Analysis
    spec = spectral_analysis(data, channel_name=wave_ch, fullscale=True)
    assert spec is not None
    assert len(spec.args) > 0 # Frequencies should exist
    
    # 6. Extreme Analysis
    # Fast basic analysis
    ext_res = extreme_analysis(data, ch_name=wave_ch, visualization=False)
    assert ext_res is not None
    assert 'exceedance_table' in ext_res
    
    # 7. Output Export Testing
    # Export to DAT in tmp_path
    dat_out = tmp_path / "test_wc01_export"
    # Temporarily change working dir to temp so it saves there
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        data.__filename__ = "test_wc01_export.out" # Overwrite to affect export names
        from pydas.output import export_to_dat
        export_to_dat(data)
        assert (tmp_path / "test_wc01_export_seg00-model.dat").exists()
        
        # Test plotting without showing
        from pydas.plot import plot_channel
        plot_channel(data, ch_name=wave_ch, plotbackend='matplotlib', show=False)
    finally:
        os.chdir(cwd)

