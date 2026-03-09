import numpy as np
from scipy.signal import welch


def timeseries_to_spectrum(t, eta, nperseg=None, noverlap=None, window="hann", freqtype="w"):
    """Estimate power spectral density (PSD) from time series (Welch method)."""
    eta = np.asarray(eta, dtype=float) - np.mean(eta)
    dt = t[1] - t[0] if len(t) > 1 else 1.0
    fs = 1.0 / dt
    if nperseg is None:
        nperseg = min(len(eta), 1024)
    if noverlap is None:
        noverlap = nperseg // 2
    f_hz, S_f = welch(
        eta, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,
        return_onesided=True, scaling="density",
    )
    if freqtype == "w":
        return 2.0 * np.pi * f_hz, S_f / (2.0 * np.pi)
    return f_hz, S_f


def spectrum_to_acf(w, S, nt=None):
    """Spectrum to Autocovariance via IFFT (Wiener-Khintchine)."""
    w, S = np.asarray(w), np.asarray(S)
    nf = len(w)
    nt = nt if nt is not None else nf - 1
    dt = np.pi / w[-1]
    nfft = 2 ** int(np.ceil(np.log2(2 * nf - 2)))
    specn = S * w[-1]
    rper = np.zeros(nfft)
    rper[:nf] = specn
    rper[nfft - nf + 2:] = specn[nf - 2:0:-1]
    r = np.fft.fft(rper, nfft).real / (2 * nf - 2)
    tau = np.arange(min(nt, nf - 1) + 1) * dt * (2 * nf - 2) / nfft
    return tau, r[:len(tau)]


def acf_to_spectrum(tau, R, nugget=1e-12):
    """Autocovariance to Spectrum via FFT (Wiener-Khintchine)."""
    R, tau = np.asarray(R).copy(), np.asarray(tau)
    n = len(R)
    R[0] += nugget
    nfft = 2 ** int(np.ceil(np.log2(2 * n - 2)))
    nf = nfft // 2
    dt = tau[1] - tau[0] if len(tau) > 1 else 1.0
    acf_circ = np.zeros(nfft)
    acf_circ[:n] = R
    acf_circ[nfft - n + 2:] = R[n - 2:0:-1]
    S = np.fft.fft(acf_circ, nfft).real[:nf + 1] * dt / np.pi
    w = np.linspace(0, np.pi / dt, nf + 1)
    return w, np.clip(S, 0, None)


def timeseries_to_acf(t, eta, lag=None, window="parzen", biased=True):
    """Estimate ACF from time series via FFT."""
    from scipy.signal.windows import get_window
    eta = np.asarray(eta, dtype=float) - np.mean(eta)
    n = len(eta)
    dt = t[1] - t[0] if len(t) > 1 else 1.0
    nfft = 2 ** int(np.ceil(np.log2(n)))
    raw_psd = np.abs(np.fft.fft(eta, nfft)) ** 2 / n
    acf_full = np.fft.fft(raw_psd).real / nfft
    if not biased:
        acf_full[:n] *= n / np.arange(n, 0, -1)
    if lag is None:
        lag = min(300, n - 2)
    lag = min(lag, n - 2)
    if window:
        win = get_window(window, 2 * lag - 1)
        acf_full[:lag] *= win[lag - 1:]
        acf_full[lag] = 0
    return np.arange(lag + 1) * dt, acf_full[:lag + 1]
