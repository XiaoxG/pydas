# tests/integration/test_pydas_flow.py
import pytest
import numpy as np
from pydas import PyDAS

def test_channel_operations(pydas_instance):
    """Test adding, renaming, and deleting channels."""
    original_chN = pydas_instance.__chN__
    
    # Add channel
    new_data = np.zeros(len(pydas_instance.data[0]))
    pydas_instance.add_channel('NewCh', 'm', new_data, pydas_instance.__fs__)
    assert pydas_instance.__chN__ == original_chN + 1
    assert 'NewCh' in pydas_instance.chInfo['Name'].values
    
    # Rename channel
    pydas_instance.rename_channel('NewCh', 'RenamedCh')
    assert 'RenamedCh' in pydas_instance.chInfo['Name'].values
    assert 'NewCh' not in pydas_instance.chInfo['Name'].values
    
    # Delete channel
    pydas_instance.delete_channel('RenamedCh')
    assert pydas_instance.__chN__ == original_chN
    assert 'RenamedCh' not in pydas_instance.chInfo['Name'].values

def test_data_manipulation(pydas_instance):
    """Test basic data manipulation methods."""
    ch_name = pydas_instance.chInfo['Name'].iloc[0]
    original_mean = np.mean(pydas_instance.data[0][ch_name])
    
    # Remove mean
    pydas_instance.remove_mean(ch_name)
    new_mean = np.mean(pydas_instance.data[0][ch_name])
    assert np.isclose(new_mean, 0.0, atol=1e-6)
    
    # Add value
    pydas_instance.add_value(ch_name, 5.0)
    assert np.isclose(np.mean(pydas_instance.data[0][ch_name]), 5.0, atol=1e-6)

def test_filtering(pydas_instance):
    """Test filtering integration."""
    ch_name = pydas_instance.chInfo['Name'].iloc[0]
    
    # Apply lowpass filter
    # Just check it runs without error and returns data of same length
    filtered = pydas_instance.apply_lowpass_filter(ch_name, cutoffull=0.1, replace=False, returnValue=True)
    assert len(filtered) == len(pydas_instance.data[0][ch_name])

def test_spectral_analysis_integration(pydas_instance):
    """Test spectral analysis method on PyDAS object."""
    ch_name = pydas_instance.chInfo['Name'].iloc[0]
    
    # Run analysis
    spec = pydas_instance.spectral_analysis(ch_name, plot=False)
    
    # Check result type
    # It should be a SpecData1D object (or compatible)
    assert hasattr(spec, 'data')
    assert hasattr(spec, 'args')
    assert len(spec.data) > 0

def test_full_workflow(pydas_instance, tmp_path):
    """Test a complete analysis workflow."""
    ch_name = pydas_instance.chInfo['Name'].iloc[0]
    
    # 1. Preprocessing
    pydas_instance.remove_mean(ch_name)
    
    # 2. Filtering
    pydas_instance.apply_lowpass_filter(ch_name, cutoffull=2.0)
    
    # 3. Spectral Analysis
    spec = pydas_instance.spectral_analysis(ch_name, plot=False)
    
    # 4. Statistics
    stats = pydas_instance.statistic_analysis(ch_name, visualization=False)
    assert stats is not None
    
    # 5. Export (mock or check output)
    # Just check no exceptions raised
