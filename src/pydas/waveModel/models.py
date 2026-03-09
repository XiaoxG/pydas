import numpy as np
from scipy.special import gammaln


def jonswap_peakfact(Hs, Tp):
    """Estimate JONSWAP gamma from Hs [m] and Tp [s] (DNV-RP-C205 Sec. 3.5.5.5)."""
    x = Tp / np.sqrt(Hs)
    if x > 5.0:
        return 1.0
    elif x < 3.6:
        return 5.0
    D = 0.036 - 0.0056 * x
    gamma = np.exp(3.484 * (1.0 - 0.1975 * D * x**4))
    return float(np.clip(gamma, 1.0, 7.0))


def jonswap_spectrum(w, Hs, Tp, gamma=None, sigma_a=0.07, sigma_b=0.09):
    """JONSWAP spectral density S(w) [m^2 s/rad]."""
    w = np.asarray(w, dtype=float)
    if gamma is None:
        gamma = jonswap_peakfact(Hs, Tp)
    wp = 2.0 * np.pi / Tp
    wn = np.where(w > 0, w / wp, 1e-30)

    # Simplified Bretschneider base (N=5, M=4)
    N, M = 5, 4
    B = N / M
    C = (N - 1.0) / M
    log_wn = np.log(wn)
    log_A = C * np.log(B) + np.log(M) - gammaln(C)
    G0 = np.exp(log_A - N * log_wn - B * np.exp(-M * log_wn))

    # Peak enhancement factor Gf
    sigma = np.where(w <= wp, sigma_a, sigma_b)
    Gf = gamma ** np.exp(-0.5 * ((wn - 1.0) / sigma) ** 2)

    # Normalization factor Ag (parametric shortcut for standard sigma)
    if sigma_a == 0.07 and sigma_b == 0.09:
        Ag = 1.0 - 0.287 * np.log(gamma)
    else:
        from scipy.integrate import quad

        def _local(wn_val):
            if wn_val <= 0:
                return 0.0
            s = sigma_a if wn_val <= 1.0 else sigma_b
            gf = gamma ** np.exp(-0.5 * ((wn_val - 1.0) / s) ** 2)
            lw = np.log(wn_val)
            return gf * np.exp(log_A - N * lw - B * np.exp(-M * lw))

        area, _ = quad(_local, 0, 10.0)
        Ag = 1.0 / area if area > 0 else 1.0

    S = (Hs / 4.0) ** 2 / wp * Ag * Gf * G0
    return np.where(w > 0, S, 0.0)


def torsethaugen_spectrum(w, Hs, Tp, g=9.81):
    """Torsethaugen double-peaked spectrum S(w) [m^2 s/rad]."""
    from scipy.special import gamma as _gamma_func

    Af = 6.6
    AL = 2.0
    Au = 25.0
    KG, KG0, KG1 = 35.0, 3.5, 1.0
    r = 6.0 / 7.0
    K0, K00 = 0.5, 3.2
    M0 = 4
    B1, B2, B3 = 2.0, 0.7, 3.0
    S0, S1 = 0.08, 3.0
    A10, A1, A20, A2, A3 = 0.7, 0.5, 0.6, 0.3, 6.0

    Tf = Af * Hs ** (1.0 / 3.0)
    Tl = AL * np.sqrt(Hs)
    Tu = Au
    El = np.clip((Tf - Tp) / (Tf - Tl), 0, 1)
    Eu = np.clip((Tp - Tf) / (Tu - Tf), 0, 1)

    if Tp < Tf:  # Wind dominated
        Nw = K0 * np.sqrt(Hs) + K00
        Mw = M0
        Rpw = min((1 - A10) * np.exp(-((El / A1) ** 2)) + A10, 1.0)
        Hpw = Rpw * Hs
        Tpw = Tp
        gammaw = KG * (1 + KG0 * np.exp(-Hs / KG1)) * (2 * np.pi / g * Rpw * Hs / Tp**2) ** r
        gammaw = max(gammaw, 1.0)
        Rps = np.sqrt(1.0 - Rpw**2)
        Hps = Rps * Hs
        Tps = Tf + B1
        gammas = 1.0
    else:  # Swell dominated
        Rps = min((1 - A20) * np.exp(-((Eu / A2) ** 2)) + A20, 1.0)
        Hps = Rps * Hs
        Tps = Tp
        gammas = (
            KG * (1 + KG0 * np.exp(-Hs / KG1))
            * (2 * np.pi / g * Hs / Tf**2) ** r
            * (1 + A3 * Eu)
        )
        gammas = max(gammas, 1.0)
        Nw, Mw = K0 * np.sqrt(Hs) + K00, M0 * (1 - B2 * np.exp(-Hs / B3))
        Rpw = np.sqrt(1 - Rps**2)
        Hpw = Rpw * Hs
        C_ = (Nw - 1) / Mw
        B_ = Nw / Mw
        G0w = B_**C_ * Mw / _gamma_func(C_)
        Tpw = (
            (16 * S0 * (1 - np.exp(-Hs / S1)) * 0.4**Nw / (G0w * Hpw**2))
            ** (-1.0 / (Nw - 1.0))
            if Hpw > 0
            else np.inf
        )
        gammaw = 1.0

    S_wind = jonswap_spectrum(w, Hpw, Tpw, gamma=gammaw)
    S_swell = jonswap_spectrum(w, Hps, Tps, gamma=gammas)
    return S_wind + S_swell


def pm_spectrum(w, Hs, Tp):
    """Pierson-Moskowitz spectral density S(w) [m^2 s/rad]."""
    return jonswap_spectrum(w, Hs, Tp, gamma=1.0)


def cos2s_spreading(theta, s=15.0, theta0=0.0):
    """Cos-2s directional spreading function."""
    theta = np.asarray(theta, dtype=float)
    # Cs = Gamma(s+1) / (2 * sqrt(pi) * Gamma(s+0.5))
    log_Cs = gammaln(s + 1) - gammaln(s + 0.5) - 0.5 * np.log(np.pi) - np.log(2.0)
    Cs = np.exp(log_Cs)
    D = Cs * np.cos((theta - theta0) / 2.0) ** (2.0 * s)
    return D


def directional_spectrum(w, S1d, theta, wp, theta0=0.0, s_max=15.0, freq_dependent=True):
    """Build 2D directional spectrum S(w, theta)."""
    if freq_dependent:
        wn = w / wp
        s = np.where(wn <= 1.0, s_max * wn**5, s_max * wn**-2.5)
        s = np.maximum(s, 0.01)
        cos_half = np.cos((theta[:, np.newaxis] - theta0) / 2.0)
        log_Cs = gammaln(s + 1) - gammaln(s + 0.5) - 0.5 * np.log(np.pi)
        D = np.exp(log_Cs[np.newaxis, :]) * np.abs(cos_half) ** (2.0 * s[np.newaxis, :])
    else:
        D = cos2s_spreading(theta, s=s_max, theta0=theta0)[:, np.newaxis] * np.ones_like(w)
    return D * S1d[np.newaxis, :]
