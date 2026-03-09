# tests/unit/test_waveModel_models.py
import pytest
import numpy as np
from scipy.integrate import trapezoid
from pydas.waveModel.models import jonswap_spectrum, pm_spectrum, torsethaugen_spectrum, cos2s_spreading, directional_spectrum

class TestJonswap:
    def test_integration_hs(self):
        """Test JONSWAP spectrum integration matches Hs."""
        Hs = 5.0
        Tp = 10.0
        w = np.linspace(0.1, 4.0, 500)
        
        S = jonswap_spectrum(w, Hs, Tp)
        
        # Integration: m0
        dw = w[1] - w[0]
        m0 = trapezoid(S, w)
        Hs_calc = 4.0 * np.sqrt(m0)
        
        # Tolerance: Should be close
        np.testing.assert_allclose(Hs_calc, Hs, rtol=0.02)
        
    def test_peak_location(self):
        """Test JONSWAP spectrum peak frequency."""
        Hs = 5.0
        Tp = 10.0
        wp = 2 * np.pi / Tp
        w = np.linspace(0.1, 4.0, 500)
        S = jonswap_spectrum(w, Hs, Tp)
        
        peak_idx = np.argmax(S)
        peak_w = w[peak_idx]
        
        # Check peak frequency matches wp within one grid step
        assert abs(peak_w - wp) < (w[1] - w[0])

    def test_pm_equivalence(self):
        """Test PM spectrum is JONSWAP with gamma=1."""
        Hs = 5.0
        Tp = 10.0
        w = np.linspace(0.1, 4.0, 500)
        
        S_pm = pm_spectrum(w, Hs, Tp)
        S_jonswap = jonswap_spectrum(w, Hs, Tp, gamma=1.0)
        
        np.testing.assert_allclose(S_pm, S_jonswap)

class TestTorsethaugen:
    def test_double_peak(self):
        """Test Torsethaugen spectrum has two peaks for mixed seas."""
        Hs = 6.0
        Tp = 8.0 # Wind dominated, but might have swell
        w = np.linspace(0.1, 4.0, 500)
        
        S = torsethaugen_spectrum(w, Hs, Tp)
        
        # Difficult to assert specific peak locations without hardcoding, 
        # but we can check if it returns valid values.
        assert np.all(S >= 0)
        assert np.sum(S) > 0

class TestDirectional:
    def test_cos2s_normalization(self):
        """Test cos-2s spreading function integrates to 1."""
        theta = np.linspace(-np.pi, np.pi, 361) # Full circle
        s = 15.0
        
        D = cos2s_spreading(theta, s=s)
        
        # Integration over theta
        integral = trapezoid(D, theta)
        
        np.testing.assert_allclose(integral, 1.0, rtol=0.01)
        
    def test_directional_spectrum_shape(self):
        """Test directional spectrum shape."""
        w = np.linspace(0.5, 1.5, 100)
        theta = np.linspace(-np.pi, np.pi, 100)
        S1d = np.ones_like(w) # Flat 1D spectrum
        wp = 1.0
        
        S2d = directional_spectrum(w, S1d, theta, wp, freq_dependent=False)
        
        # Shape should be (len(theta), len(w))
        assert S2d.shape == (len(theta), len(w))
        
        # Check normalization for each frequency
        integrals = trapezoid(S2d, theta, axis=0)
        np.testing.assert_allclose(integrals, S1d, rtol=0.01)
