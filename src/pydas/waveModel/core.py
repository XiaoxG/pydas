import numpy as np


def w2k(w, h=np.inf, g=9.81, tol=1e-7, max_iter=100):
    """Solve dispersion relation: angular frequency → wave number.

    Parameters
    ----------
    w : array_like
        Angular frequency [rad/s].
    h : float
        Water depth [m]. Use np.inf for deep water.
    g : float
        Gravitational acceleration [m/s^2].

    Returns
    -------
    k : ndarray
        Wave number [rad/m].
    """
    w = np.asarray(w, dtype=float)
    k = w**2 / g  # deep water initial guess
    if np.isinf(h):
        return k
    for _ in range(max_iter):
        kh = k * h
        coshkh2 = np.where(np.abs(kh) < 300, np.cosh(kh) ** 2, np.inf)
        f = k * np.tanh(kh) - w**2 / g
        fp = np.tanh(kh) + kh / coshkh2
        dk = f / fp
        k = k - dk
        k = np.where(k <= 0, 1e-16, k)
        if np.all(np.abs(dk) < tol * np.abs(k) + tol):
            break
    return k


def spectral_moments(w, S, orders=(0, 1, 2, 4)):
    """Compute spectral moments m_n = ∫ w^n S(w) dw.

    Parameters
    ----------
    w : array_like
        Angular frequencies [rad/s].
    S : array_like
        Spectral density S(w).
    orders : tuple of int
        Which moments to compute.

    Returns
    -------
    dict : {n: m_n} for each n in orders.
    """
    w = np.asarray(w, dtype=float)
    S = np.asarray(S, dtype=float)
    from scipy.integrate import trapezoid
    return {n: trapezoid(w**n * S, w) for n in orders}


def wave_parameters(w, S):
    """Compute standard wave parameters from spectrum.

    Returns
    -------
    dict with keys:
        Hs   : Significant wave height [m]
        Tp   : Peak period [s]
        Tz   : Mean zero-crossing period [s]
        T1   : Energy mean period [s]
        m0   : Zeroth spectral moment (variance)
        eps  : Bandwidth parameter ε = sqrt(1 - m2²/(m0·m4))
    """
    m = spectral_moments(w, S, orders=(0, 1, 2, 4))
    m0, m1, m2, m4 = m[0], m[1], m[2], m[4]
    Hs = 4.0 * np.sqrt(m0)
    Tz = 2.0 * np.pi * np.sqrt(m0 / m2) if m2 > 0 else np.inf
    T1 = 2.0 * np.pi * m0 / m1 if m1 > 0 else np.inf
    Tp = 2.0 * np.pi / w[np.argmax(S)] if np.any(S > 0) else np.inf
    eps = np.sqrt(1.0 - m2**2 / (m0 * m4)) if (m0 > 0 and m4 > 0) else 0.0
    return dict(Hs=Hs, Tp=Tp, Tz=Tz, T1=T1, m0=m0, eps=eps)


def max_wave_height(Hs, Tz, duration=10800.0):
    """Most probable maximum wave height in a stationary sea state.

    H_max ≈ Hs * sqrt(0.5 * ln(N)), where N = duration / Tz.

    Parameters
    ----------
    Hs : float
        Significant wave height [m].
    Tz : float
        Mean zero-crossing period [s].
    duration : float
        Sea state duration [s] (default 3 hours = 10800 s).

    Returns
    -------
    Hmax_mode : float
        Most probable maximum wave height [m].
    Hmax_mean : float
        Expected maximum wave height [m].
    """
    N = duration / Tz
    lnN = np.log(N)
    Hmax_mode = Hs * np.sqrt(0.5 * lnN)
    Hmax_mean = Hs * np.sqrt(0.5 * lnN) * np.sqrt(1.0 + 0.29 / lnN)
    return Hmax_mode, Hmax_mean
