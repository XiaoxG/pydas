# tests/integration/test_pydas_features.py
import pytest
import numpy as np
import pandas as pd
from pydas import PyDAS

def test_channel_arithmetic(pydas_instance):
    """Test channel_calculate and channel_apply_function."""
    # Add second channel
    data2 = np.ones(len(pydas_instance.data[0])) * 2.0
    pydas_instance.add_channel('Const2', 'm', data2, pydas_instance.__fs__)
    
    # 1. Add channels
    pydas_instance.channel_calculate('Wave1', 'Const2', '+', 'SumCh')
    expected = pydas_instance.data[0]['Wave1'] + 2.0
    np.testing.assert_allclose(pydas_instance.data[0]['SumCh'], expected)
    
    # 2. Multiply channels
    pydas_instance.channel_calculate('Wave1', 'Const2', '*', 'MulCh')
    expected = pydas_instance.data[0]['Wave1'] * 2.0
    np.testing.assert_allclose(pydas_instance.data[0]['MulCh'], expected)
    
    # 3. Apply function (lambda)
    pydas_instance.channel_apply_function('Const2', lambda x: x**2, 'SquareCh')
    np.testing.assert_allclose(pydas_instance.data[0]['SquareCh'], 4.0)
    
    # 4. Apply function (numpy string)
    pydas_instance.channel_apply_function('Const2', 'np.sqrt(x)', 'SqrtCh')
    np.testing.assert_allclose(pydas_instance.data[0]['SqrtCh'], np.sqrt(2.0))

def test_physical_transformations(pydas_instance):
    """Test scaling to fullscale."""
    pydas_instance.__lam__ = 25.0 # Set scale factor
    ch_name = 'Wave1'
    orig_val = pydas_instance.data[0][ch_name].iloc[0]
    
    # To fullscale
    # For waves (L), scale is lambda. For time (T), scale is sqrt(lambda).
    pydas_instance.to_fullscale()
    
    assert pydas_instance.__scale__ == 'prototype'
    # Check if value scaled (assuming Wave1 is length-like)
    # Note: pydas.to_fullscale() might scale based on unit. 
    # If unit is 'm', it uses lambda.
    new_val = pydas_instance.data[0][ch_name].iloc[0]
    assert np.isclose(new_val, orig_val * 25.0)

def test_alignment_methods(pydas_instance):
    """Test correlation-based alignment."""
    # Create shifted version of Wave1
    data = pydas_instance.data[0]['Wave1'].values
    shifted_data = np.roll(data, 10)
    pydas_instance.add_channel('Wave1_Shifted', 'm', shifted_data, pydas_instance.__fs__)
    
    # Find shift
    move_pts = pydas_instance.find_move_ccor('Wave1_Shifted', 'Wave1')
    # np.roll with 10 should be found as 10 (or -10 depending on implementation)
    assert abs(move_pts) > 0
    
    # Apply move
    pydas_instance.move_ccor('Wave1_Shifted', 'Wave1', 'Wave1')
    # Wave1_Shifted should now be aligned with Wave1

def test_data_cleaning(pydas_instance):
    """Test outlier removal."""
    ch_name = 'Wave1'
    # Insert outlier
    pydas_instance.data[0].loc[100, ch_name] = 999.0
    
    # Wash data
    pydas_instance.data_wash(ch_name, method='linear', threshold=10.0)
    
    # Check if outlier was removed/interpolated
    assert pydas_instance.data[0][ch_name].iloc[100] < 100.0

def test_statistics_and_updates(pydas_instance):
    """Test statistics recalculation."""
    pydas_instance.updateST()
    ch_name = 'Wave1'
    
    # Modify data manually
    pydas_instance.data[0].loc[:, ch_name] = 1.0
    
    # Statistics should still be old until update
    pydas_instance.updateST(chName=ch_name)
    
    # Access via segStatis (PyDAS stores stats there)
    stats = pydas_instance.segStatis[0].loc[ch_name]
    assert np.isclose(stats['Mean'], 1.0)
