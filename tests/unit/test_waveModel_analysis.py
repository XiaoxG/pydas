# tests/unit/test_waveModel_analysis.py
import pytest
import numpy as np
from scipy.integrate import trapezoid
from pydas.waveModel.analysis import timeseries_to_spectrum, timeseries_to_acf, spectrum_to_acf, acf_to_spectrum

class TestSpectrumAnalysis:
    def test_sine_wave_spectrum(self):
        """Test Welch spectrum of a sine wave."""
        fs = 100.0
        T = 10.0 # 10 seconds
        t = np.arange(0, T, 1/fs)
        f0 = 5.0 # 5 Hz
        A = 2.0
        eta = A * np.sin(2 * np.pi * f0 * t)
        
        # Expected energy (variance) = A^2 / 2 = 2.0
        expected_m0 = 2.0
        
        w, S = timeseries_to_spectrum(t, eta, nperseg=256, freqtype='w')
        
        # Check peak location
        peak_idx = np.argmax(S)
        peak_w = w[peak_idx]
        expected_w = 2 * np.pi * f0
        
        # Allow some spectral leakage/resolution error
        # dw = 2*pi * fs / nperseg = 2*pi * 100 / 256 approx 2.45 rad/s
        dw = w[1] - w[0]
        assert abs(peak_w - expected_w) <= dw
        
        # Check total energy (approximate due to windowing/leakage)
        m0_calc = trapezoid(S, w)
        assert abs(m0_calc - expected_m0) < 0.2 * expected_m0 # 20% tolerance for short signal

class TestACF:
    def test_sine_wave_acf(self):
        """Test ACF of a sine wave."""
        fs = 100.0
        T = 10.0
        t = np.arange(0, T, 1/fs)
        f0 = 2.0
        A = 1.0
        eta = A * np.sin(2 * np.pi * f0 * t)
        
        # Use biased=False to match theoretical cosine
        tau, R = timeseries_to_acf(t, eta, lag=100, biased=False)
        
        # Expected ACF: R(tau) = (A^2/2) * cos(2*pi*f0*tau)
        expected_R = (A**2 / 2) * np.cos(2 * np.pi * f0 * tau)
        
        # Compare first few lags (avoid windowing effects at large lags)
        # Using a window like 'parzen' (default in timeseries_to_acf) dampens the ACF.
        # Let's set window=None to check raw ACF.
        tau, R = timeseries_to_acf(t, eta, lag=100, window=None, biased=False)
        
        np.testing.assert_allclose(R[:50], expected_R[:50], atol=0.1)

    def test_spectrum_acf_roundtrip(self):
        """Test round trip: Spectrum -> ACF -> Spectrum."""
        w = np.linspace(0, 10, 513) # Need 2^N + 1 points for best FFT behavior usually
        # Gaussian spectrum
        w0 = 5.0
        sigma = 1.0
        S = np.exp(-(w - w0)**2 / (2 * sigma**2))
        
        # Forward: S -> R
        tau, R = spectrum_to_acf(w, S)
        
        # Inverse: R -> S
        w_rec, S_rec = acf_to_spectrum(tau, R)
        
        # Interpolate reconstructed spectrum to original grid
        S_rec_interp = np.interp(w, w_rec, S_rec)
        
        # Check agreement
        max_error = np.max(np.abs(S_rec_interp - S))
        assert max_error < 0.05 # 5% error tolerance
