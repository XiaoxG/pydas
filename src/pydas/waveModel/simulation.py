import numpy as np


def spectrum_to_timeseries(w, S, duration, dt=None, seed=None):
    """Generate Gaussian wave elevation from spectrum via IFFT."""
    w, S = np.asarray(w), np.asarray(S)
    if dt is None:
        dt = np.pi / w[-1]
    ns = int(np.ceil(duration / dt))
    ns += ns % 2  # ensure even
    rng = np.random.default_rng(seed)
    f_fft = np.arange(1, ns // 2) / (ns * dt)
    f_spec = w / (2.0 * np.pi)
    S_interp = np.interp(f_fft, f_spec, S * 2.0 * np.pi, left=0, right=0)
    amplitude = np.sqrt(S_interp / (2.0 * ns * dt)) * ns
    z = rng.standard_normal(ns // 2 - 1) + 1j * rng.standard_normal(ns // 2 - 1)
    X = np.zeros(ns, dtype=complex)
    X[1:ns // 2] = amplitude * z
    X[ns // 2 + 1:] = np.conj(X[ns // 2 - 1:0:-1])
    eta = np.fft.ifft(X).real * np.sqrt(2.0)
    return np.arange(ns) * dt, eta


def directional_sim(w, theta, S2d, x, y, t, h=np.inf, g=9.81, seed=None):
    """Simulate short-crested wave field at point (x, y) from directional spectrum."""
    from .core import w2k
    w, theta, t = np.asarray(w), np.asarray(theta), np.asarray(t)
    rng = np.random.default_rng(seed)
    dw = np.diff(w, prepend=0)
    dw[0] = dw[1] if len(dw) > 1 else 1.0
    dtheta = np.abs(np.diff(theta, prepend=theta[0] - (theta[1] - theta[0])))
    dtheta[0] = dtheta[1] if len(dtheta) > 1 else 2 * np.pi / len(theta)
    k = w2k(w, h=h, g=g)
    phi = rng.uniform(0, 2 * np.pi, size=S2d.shape)
    eta = np.zeros_like(t)
    for j in range(len(w)):
        if dw[j] <= 0 or w[j] <= 0:
            continue
        for i in range(len(theta)):
            amp = np.sqrt(2.0 * S2d[i, j] * dw[j] * dtheta[i])
            if amp > 1e-15:
                spatial_phase = k[j] * (x * np.cos(theta[i]) + y * np.sin(theta[i]))
                eta += amp * np.cos(w[j] * t - spatial_phase + phi[i, j])
    return eta
