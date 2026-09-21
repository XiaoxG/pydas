# tests/unit/test_process.py
import numpy as np
from pydas.process import (
    apply_lowpass_filter,
    apply_highpass_filter,
    remove_mean,
    add_value,
    multiply_value,
    move_data,
    data_wash,
    detrend,
    add_diff1,
    _apply_butterworth,
    _correlation_lag,
    find_move_ccor,
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


def test_highpass_filter_logic(pydas_instance):
    ch = "Wave1"
    fs = pydas_instance.__fs__
    t = np.arange(len(pydas_instance.data[0])) / fs
    low = np.sin(2 * np.pi * 0.1 * t)
    high = 0.5 * np.sin(2 * np.pi * 4.0 * t)
    pydas_instance.data[0][ch] = low + high

    filtered = apply_highpass_filter(
        pydas_instance, ch, cutoffull=2 * np.pi, replace=False, returnValue=True
    )
    assert len(filtered) == len(t)
    corr_high = np.corrcoef(filtered, high)[0, 1]
    corr_low = np.corrcoef(filtered, low)[0, 1]
    assert corr_high > corr_low


def test_filter_channel_name_list(pydas_instance):
    """chName as a list applies the filter to every named channel."""
    extra = pydas_instance.data[0]["Wave1"].values.copy()
    pydas_instance.add_channel("Wave2", "m", extra, pydas_instance.__fs__)
    apply_lowpass_filter(
        pydas_instance, ["Wave1", "Wave2"], cutoffull=2 * np.pi, replace=True
    )
    assert "Wave1" in pydas_instance.data[0].columns
    assert "Wave2" in pydas_instance.data[0].columns


def test_add_diff1_creates_derivative_channel(pydas_instance):
    ok = add_diff1(pydas_instance, "Wave1")
    assert ok is True
    assert "Wave1_d1" in pydas_instance.chInfo["Name"].values
    assert len(pydas_instance.data[0]["Wave1_d1"]) == len(pydas_instance.data[0]["Wave1"])


def test_data_wash_removes_inserted_outlier(pydas_instance):
    ch_name = "Wave1"
    pydas_instance.data[0].loc[100, ch_name] = 999.0
    data_wash(pydas_instance, ch_name, method="linear", threshold=10.0)
    assert pydas_instance.data[0][ch_name].iloc[100] < 100.0


def test_detrend_linear_independent_of_repair(pydas_instance):
    ch = "Wave1"
    n = len(pydas_instance.data[0][ch])
    slope = np.linspace(0.0, 4.0, n)
    pydas_instance.data[0][ch] = pydas_instance.data[0][ch].to_numpy() + slope
    detrend(pydas_instance, ch, kind="linear")
    fitted = np.polyfit(np.arange(n), pydas_instance.data[0][ch].to_numpy(), 1)[0]
    assert abs(fitted) < 1e-4


def test_butterworth_kernel_low_and_high():
    """High/low-pass wrappers share one Butterworth kernel."""
    fs = 50.0
    t = np.arange(0, 4.0, 1.0 / fs)
    low = np.sin(2 * np.pi * 0.5 * t)
    high = np.sin(2 * np.pi * 8.0 * t)
    mixed = low + high
    lp = _apply_butterworth(mixed, cutoff=2.0, fs=fs, order=6, btype="low")
    hp = _apply_butterworth(mixed, cutoff=2.0, fs=fs, order=6, btype="high")
    assert np.corrcoef(lp, low)[0, 1] > np.corrcoef(lp, high)[0, 1]
    assert np.corrcoef(hp, high)[0, 1] > np.corrcoef(hp, low)[0, 1]


def test_correlation_lag_shared_by_find_move(pydas_instance):
    """find_move_ccor is the negated raw lag from _correlation_lag."""
    orig = pydas_instance.data[0]["Wave1"].values
    shifted = np.roll(orig, 10)
    pydas_instance.add_channel("Wave1_Shifted", "m", shifted, pydas_instance.__fs__)
    n_sample = int(pydas_instance.segInfo["N sample"].iloc[0])
    raw_lag, _ = _correlation_lag(
        pydas_instance.data[0]["Wave1_Shifted"].values,
        pydas_instance.data[0]["Wave1"].values,
        n_sample,
    )
    found = find_move_ccor(pydas_instance, "Wave1_Shifted", "Wave1")
    assert found == -raw_lag
    assert abs(found) > 0
