import numpy as np
from .core import wave_parameters, spectral_moments


class SpecData1D:
    """Simplified 1D Spectrum object for compatibility."""

    def __init__(self, data, args, **kwargs):
        self.data = np.asarray(data)
        self.args = np.asarray(args)
        self.type = kwargs.get("type", "freq")
        self.h = kwargs.get("h", np.inf)
        self._freqtype = kwargs.get("freqtype", "w")

    @property
    def S(self):
        return self.data

    @S.setter
    def S(self, value):
        self.data = value

    @property
    def moments(self):
        return self.moment()

    def moment(self, orders=(0, 1, 2, 4)):
        return spectral_moments(self.args, self.data, orders)

    def wave_parameters(self):
        return wave_parameters(self.args, self.data)

    def tocovdata(self, nt=None):
        from .analysis import spectrum_to_acf
        tau, R = spectrum_to_acf(self.args, self.data, nt=nt)
        return CovData1D(R, tau)


class CovData1D:
    """Simplified 1D Covariance object for compatibility."""

    def __init__(self, data, args, **kwargs):
        self.data = np.asarray(data)
        self.args = np.asarray(args)

    def tospecdata(self):
        from .analysis import acf_to_spectrum
        w, S = acf_to_spectrum(self.args, self.data)
        return SpecData1D(S, w)


class TimeSeries:
    """Simplified TimeSeries object for compatibility."""

    def __init__(self, data, args=None, **kwargs):
        self.data = np.asarray(data)
        if args is None:
            self.args = np.arange(len(self.data))
        else:
            self.args = np.asarray(args)

    def sampling_period(self):
        if len(self.args) < 2:
            return 1.0
        return self.args[1] - self.args[0]

    def tospecdata(self, L=None, method="psd", **kwargs):
        """Estimate a 1-D spectrum from this time series.

        Parameters
        ----------
        L : int, optional
            For ``method='psd'``, Welch segment length. For ``method='cov'``,
            maximum lag of the autocovariance. Defaults depend on the method.
        method : {'psd', 'cov'}, optional
            ``'psd'`` uses Welch's method. ``'cov'`` uses the autocovariance
            (Wiener–Khinchin) path.

        Returns
        -------
        SpecData1D
        """
        method = (method or "psd").lower()
        if method in ("psd", "welch"):
            from .analysis import timeseries_to_spectrum
            nperseg = L if L is not None else min(len(self.data), 1024)
            w, S = timeseries_to_spectrum(self.args, self.data, nperseg=nperseg)
            return SpecData1D(S, w)
        if method in ("cov", "covariance"):
            lag = L if L is not None else min(300, max(len(self.data) - 2, 1))
            return self.tocovdata(lag=lag).tospecdata()
        raise ValueError(f"Unknown spectral method '{method}'. Use 'psd' or 'cov'.")

    def tocovdata(self, lag=None, **kwargs):
        from .analysis import timeseries_to_acf
        tau, R = timeseries_to_acf(self.args, self.data, lag=lag)
        return CovData1D(R, tau)


class Jonswap(SpecData1D):
    """Compatibility Jonswap class that acts like a SpecData1D."""

    def __init__(self, Hs, Tp, gamma=None, nw=257, w_max=None):
        if w_max is None:
            w_max = 33.0 / Tp
        w = np.linspace(0, w_max, nw)
        from .models import jonswap_spectrum
        S = jonswap_spectrum(w, Hs, Tp, gamma=gamma)
        super().__init__(S, w)
        self.Hs = Hs
        self.Tp = Tp
        self.gamma = gamma
