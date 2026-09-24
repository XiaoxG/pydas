# tests/unit/test_utils.py
import numpy as np

from pydas.utils import diff1d, data_change_fs, findtrans, get_default_transDict
import pydas
import pydas.utils as utils


def test_package_version():
    """Installed / source package version is exported on the package."""
    assert pydas.__version__ == "1.4.3"


def test_diff1d_sine_error():
    """diff1d of sin(x) should stay close to cos(x)."""
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    dy = diff1d(y, x[1] - x[0])
    assert dy is not None
    assert len(dy) == len(y)
    max_error = np.max(np.abs(dy - np.cos(x)))
    assert max_error < 0.05


def test_diff1d_numpy_fallback(monkeypatch):
    """NumPy fallback must return an array for every n (not implicit None)."""
    monkeypatch.setattr(utils, "NUMBA_AVAILABLE", False)
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    dy = utils.diff1d(y, x[1] - x[0])
    assert dy is not None
    assert len(dy) == len(y)
    max_error = np.max(np.abs(dy - np.cos(x)))
    assert max_error < 0.05


def test_data_change_fs_length():
    """Resampled length should track the fs ratio within a few samples."""
    fs_orig = 100
    fs_new = 50
    t = np.arange(0, 1, 1 / fs_orig)
    signal = np.sin(2 * np.pi * 5 * t)
    resampled = data_change_fs(signal, fs_orig, fs_new)
    expected_length = int(len(signal) * fs_new / fs_orig)
    assert expected_length - 5 <= len(resampled) <= expected_length + 5


def test_findtrans_composite_velocity():
    """Composite unit m/s combines length and time scale powers."""
    trans_dict = get_default_transDict()
    new_unit, coeffs = findtrans("m/s", trans_dict)
    assert new_unit == "m/s"
    assert len(coeffs) == 3
    np.testing.assert_allclose(coeffs[0], 1.0)
    np.testing.assert_allclose(coeffs[1], 0.0)
    np.testing.assert_allclose(coeffs[2], 0.5)
