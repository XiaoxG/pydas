# tests/unit/test_waveModel_core.py
import pytest
import numpy as np
from pydas.waveModel.core import w2k, spectral_moments, wave_parameters, max_wave_height

class TestDispersion:
    def test_deep_water(self):
        """Test dispersion relation in deep water (h -> inf)."""
        w = np.linspace(0.1, 2.0, 100)
        h = np.inf
        k = w2k(w, h=h)
        # Deep water: w^2 = g k -> k = w^2 / g
        expected_k = w**2 / 9.81
        np.testing.assert_allclose(k, expected_k, rtol=1e-5)

    def test_shallow_water(self):
        """Test dispersion relation in shallow water (kh << 1)."""
        # Shallow water limit: w^2 = g k^2 h -> k = w / sqrt(gh)
        h = 5.0
        g = 9.81
        w_shallow = np.linspace(0.01, 0.1, 10) # Low frequency -> long wave -> shallow water approx better
        k = w2k(w_shallow, h=h, g=g)
        expected_k = w_shallow / np.sqrt(g * h)
        # Relax tolerance slightly as tanh(kh) approx kh is only valid for very small kh
        np.testing.assert_allclose(k, expected_k, rtol=0.05)

    def test_zero_frequency(self):
        """Test behavior at w=0."""
        w = np.array([0.0])
        k = w2k(w)
        assert k[0] == 0.0 or np.isclose(k[0], 0.0)

class TestSpectralMoments:
    def test_simple_box_spectrum(self):
        """Test moments integration with a simple box spectrum."""
        # Box spectrum: S(w) = 1 for w in [1, 2], else 0
        w = np.linspace(0, 3, 10001) # Fine grid
        dw = w[1] - w[0]
        S = np.zeros_like(w)
        mask = (w >= 1.0) & (w <= 2.0)
        S[mask] = 1.0
        
        # Theoretical moments:
        # m0 = int(1 dw) from 1 to 2 = 1.0
        # m1 = int(w dw) from 1 to 2 = [w^2/2] = 2^2/2 - 1^2/2 = 2 - 0.5 = 1.5
        # m2 = int(w^2 dw) from 1 to 2 = [w^3/3] = 8/3 - 1/3 = 7/3 = 2.333...
        
        moments = spectral_moments(w, S, orders=(0, 1, 2))
        
        np.testing.assert_allclose(moments[0], 1.0, rtol=1e-3)
        np.testing.assert_allclose(moments[1], 1.5, rtol=1e-3)
        np.testing.assert_allclose(moments[2], 7.0/3.0, rtol=1e-3)

class TestWaveParameters:
    def test_hs_tz(self):
        """Test Hs and Tz calculation."""
        # Using the same box spectrum logic
        w = np.linspace(0, 3, 3001)
        S = np.zeros_like(w)
        mask = (w >= 1.0) & (w <= 2.0)
        S[mask] = 1.0
        
        # Expected:
        # m0 = 1.0 -> Hs = 4 * sqrt(1) = 4.0
        # m2 = 7/3
        # Tz = 2*pi * sqrt(m0/m2) = 2*pi * sqrt(1 / (7/3)) = 2*pi * sqrt(3/7)
        expected_hs = 4.0
        expected_tz = 2 * np.pi * np.sqrt(3.0/7.0)
        
        params = wave_parameters(w, S)
        
        np.testing.assert_allclose(params['Hs'], expected_hs, rtol=1e-3)
        np.testing.assert_allclose(params['Tz'], expected_tz, rtol=1e-3)
        
    def test_bandwidth(self):
        """Test bandwidth parameter epsilon."""
        # For a delta function spectrum (single frequency), bandwidth should be 0.
        # Ideally: m2^2 = m0 * m4 -> eps = 0
        w = np.linspace(0, 2, 201)
        S = np.zeros_like(w)
        S[100] = 100.0 # Delta spike at w=1.0
        
        params = wave_parameters(w, S)
        # Numerical integration of delta function is tricky, but let's check if it's small
        assert params['eps'] < 0.1

class TestMaxWaveHeight:
    def test_max_wave_height(self):
        """Test max wave height formula."""
        Hs = 5.0
        Tz = 10.0
        duration = 3600.0 # 1 hour
        # N = 3600 / 10 = 360
        # Hmax_mode = 5.0 * sqrt(0.5 * ln(360))
        # ln(360) approx 5.886
        # sqrt(0.5 * 5.886) approx sqrt(2.943) approx 1.715
        # Hmax approx 5 * 1.715 = 8.575
        
        h_mode, h_mean = max_wave_height(Hs, Tz, duration)
        
        N = duration / Tz
        expected_mode = Hs * np.sqrt(0.5 * np.log(N))
        
        np.testing.assert_allclose(h_mode, expected_mode)
        assert h_mean > h_mode # Mean is slightly larger than mode for this distribution
