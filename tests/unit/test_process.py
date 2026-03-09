# tests/unit/test_process.py
import pytest
import numpy as np
from pydas.process import (
    apply_lowpass_filter, 
    apply_highpass_filter, 
    remove_mean, 
    add_value, 
    multiply_value,
    move_data
)

def test_remove_mean(pydas_instance):
    ch = 'Wave1'
    # Initial mean is likely non-zero due to noise
    remove_mean(pydas_instance, ch)
    assert np.isclose(np.mean(pydas_instance.data[0][ch]), 0.0, atol=1e-10)

def test_add_value(pydas_instance):
    ch = 'Wave1'
    orig = pydas_instance.data[0][ch].copy()
    add_value(pydas_instance, ch, 10.0)
    np.testing.assert_allclose(pydas_instance.data[0][ch], orig + 10.0)

def test_multiply_value(pydas_instance):
    ch = 'Wave1'
    orig = pydas_instance.data[0][ch].copy()
    multiply_value(pydas_instance, ch, 2.0)
    np.testing.assert_allclose(pydas_instance.data[0][ch], orig * 2.0)

def test_lowpass_filter_logic(pydas_instance):
    ch = 'Wave1'
    # Create high frequency noise
    fs = pydas_instance.__fs__
    t = np.arange(len(pydas_instance.data[0])) / fs
    pydas_instance.data[0][ch] = np.sin(2*np.pi*0.1*t) + 0.5*np.sin(2*np.pi*4.0*t)
    
    # Apply lowpass at 1.0 Hz
    # Note: cutoffull in apply_lowpass_filter is interpreted based on scale.
    # In model scale, cutoff = cutoffull / 2pi * sqrt(lam)
    # Our fixture lam=1.0. So cutoff = cutoffull / 2pi.
    # If we want 1Hz cutoff, cutoffull should be 2*pi.
    filtered = apply_lowpass_filter(pydas_instance, ch, cutoffull=2*np.pi, replace=False, returnValue=True)
    
    # High frequency part (4Hz) should be significantly attenuated
    # Low frequency part (0.1Hz) should remain
    assert len(filtered) == len(t)
    # Simple check: STD should decrease
    assert np.std(filtered) < np.std(pydas_instance.data[0][ch])

def test_move_data(pydas_instance):
    ch = 'Wave1'
    orig = pydas_instance.data[0][ch].copy().values
    move_pts = 5
    move_data(pydas_instance, ch, move_pts)
    
    new_data = pydas_instance.data[0][ch].values
    # First 5 should be 0
    assert np.all(new_data[:move_pts] == 0)
    # Remaining should be shifted orig
    np.testing.assert_array_equal(new_data[move_pts:], orig[:-move_pts])
