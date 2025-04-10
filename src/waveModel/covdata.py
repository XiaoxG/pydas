"""
Covariance functions for time series analysis.
"""
import numpy as np
from numpy import (pi, zeros, ones, sin, exp, log, sqrt, asarray,
                  sign, arctan2, arange, linspace, abs,
                  minimum, maximum, sin, cos, newaxis, where, vstack,
                  tanh, hstack, atleast_1d, inf, r_)
from numpy.fft import fft, ifft
from scipy.signal import welch, detrend, get_window, butter, filtfilt
import warnings


__all__ = ['CovData1D', 'CovarianceEstimator']


def _set_seed(iseed):
    """Set random seed."""
    if iseed is not None:
        try:
            np.random.set_state(iseed)
        except (TypeError, ValueError):
            np.random.seed(iseed)


class CovData1D():
    """
    Container class for 1D auto covariance data objects.
    
    Member variables
    ----------------
    data : array_like
        Covariance function values
    args : vector
        Time lags
    type : string
        Covariance type
        'none', 'mea', 'mem'
    norm : bool
        If False, indicating that ACF is not normalized
    sigma : real scalar
        Estimated large-lag standard deviation, assuming the time series is Gaussian
    """

    def __init__(self, *args, **kwds):
        """
        Initialize a CovData1D object.
        """
        self.name_ = kwds.pop('name', 'WAFO CovData1D Object')
        self.sigma = kwds.pop('sigma', None)
        self.type = kwds.pop('type', 'none')
        self.norm = kwds.pop('norm', False)
        self.tr = kwds.pop('tr', None)
        self.L = kwds.pop('L', None)
        self.lagtype = kwds.pop('lagtype', 't')
        super(CovData1D, self).__init__(*args, **kwds)

    def get_delta_lag(self):
        """Return sampling interval."""
        lag = self.args
        return lag[1] - lag[0]

    def set_lagtype(self, lagtype, delta_lag=None):
        """
        Set the lag type and lag values.
        
        Parameters
        ----------
        lagtype : string
            Type of lag units: 't' (time) or 'x' (space)
        delta_lag : scalar, optional
            Sampling interval in new units
            
        Returns
        -------
        self : CovData1D
            Object with updated lag units
        """
        if delta_lag is None:
            delta_lag = 1.0

        if delta_lag == 1.0 and (lagtype == self.lagtype):
            return self
        
        delta_lag_old = self.get_delta_lag()
        
        if self.lagtype == 't' and lagtype == 'x':
            # m = t*v
            delta_lag_old *= delta_lag
        elif self.lagtype == 'x' and lagtype == 't':
            # t = m/v
            delta_lag_old /= delta_lag
        
        self.args *= (delta_lag_old / delta_lag)
        self.lagtype = lagtype
        return self

    def get_l2spike(self, ind=None):
        """
        Return lag of 2'nd largest, L2, spurious correlation after lag 0.

        Parameters
        ----------
        ind : array-like, optional
            Vector of indices to the lags where the L2 is found
            Default ind = arange(len(self.args)/10, len(self.args)) if self.norm else None
            
        Returns
        -------
        L2_lag : int
            Lag of the 2nd spurious correlation
        L2_value : float
            Value of the 2nd spurious correlation
        """
        if ind is None:
            n = len(self.args)
            if self.norm:
                ind = list(range(n // 10, n))
            else:
                return None, None
        
        if len(ind) == 0:
            return None, None
        
        acf = atleast_1d(self.data)
        n = len(acf)
        
        sorted_acf = abs(acf[ind])
        ix = sorted_acf.argsort()
        i_largest = ind[ix[-1]]  # Index to the largest spurious corr. after lag 0
        
        if i_largest < 1 or i_largest >= n:
            lag_largest = 0
            val_largest = 0
        else:
            lag_largest = self.args[i_largest]
            val_largest = acf[i_largest]
        
        return lag_largest, val_largest

    def tospecdata(self, rate=1, ftype='w'):
        """
        Computes spectral density from auto covariance function.
        
        Parameters
        ----------
        rate : scalar, int
            Interpolation rate. If rate > 1, then FFT is used.
            (default = 1, no interpolation)
        ftype : string
            Type of frequency, 'w' or 'f', default 'w'.
            
        Returns
        -------
        S : SpecData1D
            Spectral density object.
        """
        # 延迟导入避免循环依赖
        from waveModel.specdata import SpecData1D
        
        # Check that correlation is defined for the negative lags
        # Otherwise do a even reflection
        acf = atleast_1d(self.data)
        lag = atleast_1d(self.args)
        
        if len(acf) != len(lag):
            raise ValueError('Size of acf and lag inconsistent!')
        
        if lag[0] > 0:
            if abs(lag[0]) > 1e-6:
                n1 = len(lag)
                acf2 = zeros(2 * n1 - 1)
                acf2[n1-1:2*n1] = acf[:]
                acf2[0:n1] = acf[::-1]
                lag2 = hstack((-lag[:0:-1], lag[:]))
            else:
                acf2 = hstack((acf[:0:-1], acf))
                lag2 = hstack((-lag[:0:-1], lag))
            acf = acf2
            lag = lag2
        
        n = len(lag)
        if rate > 1:
            # Linear interpolation of data using FFT
            Nfft = 2 ** (nextpow2(n) + rate)
            NNn = 2 * Nfft
            
            # Add zeros to the end of acf
            acf0 = zeros(NNn)
            acf0[0:n] = acf * 1.0
            
            # Using FFT to interpolate
            acfi = fft(acf0, NNn)
            acfi[0] = 0.5 * acfi[0]
            acfi = r_[acfi, zeros(NNn // 2 - 1)]
            acfi2 = ifft(acfi).real
            acfi = zeros(NNn)
            acfi[0:NNn] = acfi2[0:NNn]
            
            # Create frequency grid
            Nold = (n - 1) // 2
            delta_f = 1 / (lag[n-1] - lag[0])
            
            # The complete spectrum
            acf = acfi[0:n+1]
            
            # The mathematical definition of the spectrum gives the factor 2*pi
            if ftype == 'w':
                f = 2 * pi * delta_f * lag[Nold:Nold + n+1]
            else:
                f = delta_f * lag[Nold:Nold + n+1]
            
            S = SpecData1D(acf, f)
            S.lagtype = self.lagtype
            
            if hasattr(self, 'tr'):
                S.tr = self.tr
                
            if hasattr(self, 'h'):
                S.h = self.h
                
            if self.lagtype == 't':
                S.freqtype = ftype
                S.title = 'Spectral density'
                if ftype == 'w':
                    S.labels.xlab = 'Angular frequency [rad/s]'
                else:
                    S.labels.xlab = 'Frequency [Hz]'
                
                if self.type == 'none' or self.type.startswith('n'):
                    S.labels.ylab = 'Power Spectrum'
                elif self.type == 'mea' or self.type == 'mean':
                    S.labels.ylab = 'S(f)'
                elif self.type.startswith('m'):
                    S.labels.ylab = 'S(f) [m^2 s]'
            
            return S
        
        # Calling the old function
        if len(acf.shape) > 1 and acf.shape[1] > 1:
            msg = 'This function can currently only handle real ACVs'
            warnings.warn(msg)
            # TODO: Fix for complex ACVs
        
        corr = acf
        dt = lag[1] - lag[0]
        ix = lag < 0
        mir = lag[ix]
        v = fft(corr)
        n = len(v)
        v = 2 * v[:n//2].real
        
        if ftype == 'w':
            w = 2 * pi * linspace(0, 1/(2 * dt), n//2) / (2 * pi)
            w = 2 * pi * w  # Giving the spectrum in rad/s
        else:
            w = linspace(0, 1/(2 * dt), n//2)
        
        spec = abs(v) * dt
        S = SpecData1D(spec, w)
        S.freqtype = ftype
        
        if self.norm:
            if self.lagtype == 't':
                spec_title = 'Normalized Power Spectrum'
            
            if self.type == 'none' or self.type.startswith('n'):
                spec_label = 'Power Spectrum'
            elif self.type.startswith('mea'):
                spec_label = 'S(f)'
            elif self.type.startswith('m'):
                spec_label = 'S(f) [m^2 s]'
        
        if hasattr(self, 'tr'):
            S.tr = self.tr
        
        if hasattr(self, 'h'):
            S.h = self.h
        
        S.norm = self.norm
        return S


class CovarianceEstimator(object):
    """
    Estimate auto covariance function from data.
    
    Parameters
    ----------
    lag : scalar, int
        Maximum time-lag for which the ACF is estimated. (Default lag=n-1)
    lag_shift : int, scalar
        Offset for acf calculation (default=0)
    tr : transformation
        Transformation of the process (default=None)
    detrend : function
        Detrending function applied to the process before estimation. (default=detrend_mean)
    window : vector
        Window function applied to the process before estimation. (default=None)
    flag : string, 'biased' or 'unbiased'
        If 'unbiased' scales the raw correlation by 1/(n-abs(k)),
        where k is the index into the result, otherwise scales the raw
        cross-correlation by 1/n. (default 'biased')
    norm : bool
        True if normalize output to one
    dt : scalar
        Time step in the data (default=1)
        
    Returns
    -------
    R : CovData1D object
    """

    def __init__(self, lag=None, lag_shift=0, tr=None, detrend=None, window=None,
                flag='biased', norm=False, dt=None):
        """Initialize a CovarianceEstimator."""
        self.lag = lag
        self.lag_shift = lag_shift
        self.tr = tr
        self.detrend = detrend
        self.window = window
        self.flag = flag
        self.norm = norm
        self.dt = dt

    def _estimate_xcov(self, x, y=None, lag=None, lag_shift=0):
        """
        Calculates auto or cross covariance.
        
        Parameters
        ----------
        x, y : array-like
            Signal vectors
        lag : scalar
            Maximum lag size of the ACF.
        lag_shift : scalar
            Offset for acf calculation (default lag_shift=0)
            
        Returns
        -------
        acf : array
            Auto- or cross-covariance
        """
        if y is None:
            y = x
            
        if lag is None:
            lag = len(x) - 1 - abs(lag_shift)
        
        # Remove mean
        x = x - x.mean()
        y = y - y.mean()
        
        n = len(x)
        
        minshift = abs(min(-lag - lag_shift, 0))
        maxshift = max(lag - lag_shift, 0)
        nfft = 2 ** (nextpow2(n + minshift + maxshift))
        
        Cxy = ifft(fft(x, nfft) * fft(y, nfft).conj()).real
        # Normalize
        if self.flag.lower() == 'unbiased':
            scale = n - abs(arange(nfft) - maxshift)
            scale[scale <= 0] = 1
            Cxy = Cxy / scale
        else:
            Cxy = Cxy / n
            
        indi = arange(-minshift, maxshift + 1)
        return Cxy[indi]

    def __call__(self, xo, y=None):
        """
        Return the estimated auto covariance function from data.
        
        Parameters
        ----------
        xo : TimeSeries or ndarray
            Data vector or TimeSeries object
        y : TimeSeries or ndarray, optional
            If given, compute cross-covariance between xo and y
            
        Returns
        -------
        R : CovData1D
            Estimated auto- or cross-covariance function
        """
        lag = self.lag
        lag_shift = self.lag_shift
        tr = self.tr
        detrend_ = self.detrend
        window = self.window
        flag = self.flag
        norm = self.norm
        dt = self.dt
        
        # Extract the timeseries
        if hasattr(xo, 'data'):
            x_in = atleast_1d(xo.data).ravel()
            if dt is None and hasattr(xo, 'sampling_period'):
                dt = xo.sampling_period()
            lagtype = 't'
        else:
            x_in = atleast_1d(xo).ravel()
            lagtype = 'n'
        
        # Extract the timeseries
        if y is not None:
            if hasattr(y, 'data'):
                y_in = atleast_1d(y.data).ravel()
            else:
                y_in = atleast_1d(y).ravel()
        else:
            y_in = None
        
        if dt is None:
            dt = 1
        
        # Check if transformation is needed
        if tr is not None:
            x = self._transform(x_in, tr)
        else:
            x = x_in
            
        # Also transform y
        if y_in is not None and tr is not None:
            y = self._transform(y_in, tr)
        else:
            y = y_in
        
        n = len(x)
        if lag is None:
            lag = n - 1
        
        if detrend_ is not None:
            x = detrend_(x)
            if y is not None:
                y = detrend_(y)
        
        if window is not None:
            if isinstance(window, tuple):
                x = x * get_window(window, n)
            else:
                x = x * window
            
            if y is not None:
                if isinstance(window, tuple):
                    y = y * get_window(window, n)
                else:
                    y = y * window
        
        # Calculates the cross covariance
        acf = self._estimate_xcov(x, y, lag, lag_shift)
        
        # Create the lag-grid
        if dt is None:
            dt = 1.0
            
        r0 = acf[lag_shift] if y is None else sqrt(acf[lag_shift] * acf[lag_shift])
        
        if norm and r0 > 0:
            acf = acf / r0
            acv_title = 'Auto Correlation Function'
            acv_ylabel = 'R(tau)'
        else:
            if y is None and x is x_in:
                acv_title = 'Auto Covariance Function'
                acv_ylabel = 'ACF'
            else:
                acv_title = 'Cross Covariance Function'
                acv_ylabel = 'CCF'
        
        if lagtype == 't':
            acv_xlabel = 'Lag [s]'
        else:
            acv_xlabel = 'Lag'
            
        lags = dt * arange(-lag - lag_shift, lag - lag_shift + 1)
        
        R = CovData1D(acf, lags, xlab=acv_xlabel, ylab=acv_ylabel, title=acv_title,
                      norm=norm, lagtype=lagtype)
        
        if lag_shift == 0 and y is None:
            # Calculates the asymptotic variance
            if norm and acf[lag_shift] > 0.0:
                # Normalized asymptotic variance
                R.sigma = sqrt(2 * sum(acf[lag_shift + 1:] ** 2) + 1.0) / sqrt(n)
            else:
                # Asymptotic variance
                R.sigma = sqrt(2 * sum(acf[lag_shift + 1:] ** 2) + acf[lag_shift] ** 2) / sqrt(n)
        
        if y is None and (tr is not None):
            R.tr = tr
        
        if hasattr(xo, 'h'):
            R.h = xo.h
                    
        return R
    
    def _transform(self, x, transform):
        """Apply a transformation."""
        if transform is None:
            return x
        if hasattr(transform, '__call__'):
            return transform(x)
        if hasattr(transform, 'trans'):
            return transform.trans(x)
        if hasattr(transform, '__getitem__'):
            return transform[0](x, *transform[1:])
        
        warnings.warn('Unknown transformation, returning untransformed data')
        return x


def nextpow2(x):
    """Return the power of two greater than or equal to absolute value of x."""
    return int(np.ceil(np.log2(np.abs(x))))