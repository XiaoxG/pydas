import numpy as np
from numpy.fft import fft
from scipy.signal.windows import get_window, parzen
from waveModel.core import nextpow2, sub_dict_select, JITImport
import warnings
from numpy import (zeros, ones, sqrt, inf, where, nan,
                   atleast_1d, hstack, r_, linspace, flatnonzero, size,
                   isnan, finfo, diag, ceil, random, pi)
from waveModel.dataframe import PlotData

_specdata = JITImport('waveModel.specdata')

from scipy import integrate, interpolate

def _set_seed(iseed):
    if iseed is not None:
        try:
            random.set_state(iseed)
        except Exception:
            random.seed(iseed)
            
def sampling_period(t_vec):
    """
    Returns sampling interval
    Returns
    -------
    dt : scalar
        sampling interval, unit:
        [s] if lagtype=='t'
        [m] otherwise
    See also
    """
    dt1 = t_vec[1] - t_vec[0]
    n = len(t_vec) - 1
    t = t_vec[-1] - t_vec[0]
    dt = t / n
    if abs(dt - dt1) > 1e-10:
        warnings.warn('Data is not uniformly sampled!')
    return dt


class CovData1D(PlotData):
    """ Container class for 1D covariance data objects in WAFO
    Member variables
    ----------------
    data : array_like
    args : vector for 1D, list of vectors for 2D, 3D, ...
    type : string
        spectrum type, one of 'freq', 'k1d', 'enc' (default 'freq')
    lagtype : letter
        lag type, one of: 'x', 'y' or 't' (default 't')
    Examples
    --------
    >>> import numpy as np
    >>> import wafo.spectrum as sp
    >>> Sj = sp.models.Jonswap(Hm0=3,Tp=7)
    >>> w = np.linspace(0,4,256)
    >>> S = sp.SpecData1D(Sj(w),w) #Make spectrum object from numerical values
    See also
    --------
    PlotData
    CovData
    """

    def __init__(self, *args, **kwds):
        super(CovData1D, self).__init__(*args, **kwds)

        self.name = 'WAFO Covariance Object'
        self.type = 'time'
        self.lagtype = 't'
        self.h = inf
        self.tr = None
        self.phi = 0.
        self.v = 0.
        self.norm = 0
        somekeys = ['phi', 'name', 'h', 'tr', 'lagtype', 'v', 'type', 'norm']

        self.__dict__.update(sub_dict_select(kwds, somekeys))

    def tospecdata(self, rate=None, method='fft', nugget=0.0, trunc=1e-5,
                   fast=True):
        '''
        Computes spectral density from the auto covariance function
        Parameters
        ----------
        rate = scalar, int
            1,2,4,8...2^r, interpolation rate for f (default 1)
        method : string
            interpolation method 'stineman', 'linear', 'cubic', 'fft'
        nugget : scalar, real
            nugget effect to ensure that round off errors do not result in
            negative spectral estimates. Good choice might be 10^-12.
        trunc : scalar, real
            truncates all spectral values where spec/max(spec) < trunc
                      0 <= trunc <1   This is to ensure that high frequency
                      noise is not added to the spectrum.  (default 1e-5)
        fast : bool
             if True : zero-pad to obtain power of 2 length ACF (default)
             otherwise  no zero-padding of ACF, slower but more accurate.
        Returns
        --------
        spec : SpecData1D object
            spectral density
         NB! This routine requires that the covariance is evenly spaced
             starting from zero lag. Currently only capable of 1D matrices.
        Examples
        --------
        >>> import wafo.spectrum.models as sm
        >>> import numpy as np
        >>> import scipy.signal as st
        >>> import pylab
        >>> L = 129
        >>> t = np.linspace(0,75,L)
        >>> R = np.zeros(L)
        >>> win = st.parzen(41)
        >>> R[0:21] = win[20:41]
        >>> R0 = CovData1D(R,t)
        >>> S0 = R0.tospecdata()
        >>> Sj = sm.Jonswap()
        >>> spec = Sj.tospecdata()
        >>> R2 = spec.tocovdata()
        >>> S1 = R2.tospecdata()
        >>> abs(S1.data-spec.data).max() < 1e-4
        True
        S1.plot('r-')
        spec.plot('b:')
        pylab.show()
        See also
        --------
        spec2cov
        datastructures
        '''

        dt = self.sampling_period()
        # dt = time-step between data points.

        acf, unused_ti = atleast_1d(self.data, self.args)

        if self.lagtype in 't':
            spectype = 'freq'
            ftype = 'w'
        else:
            spectype = 'k1d'
            ftype = 'k'

        if rate is None:
            rate = 1  # interpolation rate
        else:
            rate = 2 ** nextpow2(rate)  # make sure rate is a power of 2

        # add a nugget effect to ensure that round off errors
        # do not result in negative spectral estimates
        acf[0] = acf[0] + nugget
        n = acf.size
        # embedding a circulant vector and Fourier transform

        nfft = 2 ** nextpow2(2 * n - 2) if fast else 2 * n - 2

        if method == 'fft':
            nfft *= rate

        nf = int(nfft // 2)  # number of frequencies
        acf = r_[acf, zeros(nfft - 2 * n + 2), acf[n - 2:0:-1]]

        r_per = (fft(acf, nfft).real).clip(0)  # periodogram
        r_per_max = r_per.max()
        r_per = where(r_per < trunc * r_per_max, 0, r_per)

        spec = abs(r_per[0:(nf + 1)]) * dt / pi
        w = linspace(0, pi / dt, nf + 1)
        spec_out = _specdata.SpecData1D(spec, w, type=spectype, freqtype=ftype)
        spec_out.tr = self.tr
        spec_out.h = self.h
        spec_out.norm = self.norm

        if method != 'fft' and rate > 1:
            spec_out.args = linspace(0, pi / dt, nf * rate)
            intfun = interpolate.interp1d(w, spec, kind=method)
            spec_out.data = intfun(spec_out.args)
            spec_out.data = spec_out.data.clip(0)  # clip negative values to 0
        return spec_out

    def sampling_period(self):
        '''
        Returns sampling interval
        Returns
        ---------
        dt : scalar
            sampling interval, unit:
            [s] if lagtype=='t'
            [m] otherwise
        '''
        dt1 = self.args[1] - self.args[0]
        n = size(self.args) - 1
        t = self.args[-1] - self.args[0]
        dt = t / n
        if abs(dt - dt1) > 1e-10:
            warnings.warn('Data is not uniformly sampled!')
        return dt

    def _is_valid_acf(self):
        if self.data.argmax() != 0:
            raise ValueError('ACF does not have a maximum at zero lag')

    def sim(self, ns=None, cases=1, dt=None, iseed=None, derivative=False):
        '''
        Simulates a Gaussian process and its derivative from ACF
        Parameters
        ----------
        ns : scalar
            number of simulated points.  (default length(spec)-1=n-1).
                     If ns>n-1 it is assummed that R(k)=0 for all k>n-1
        cases : scalar
            number of replicates (default=1)
        dt : scalar
            step in grid (default dt is defined by the Nyquist freq)
        iseed : int or state
            starting state/seed number for the random number generator
            (default none is set)
        derivative : bool
            if true : return derivative of simulated signal as well
            otherwise
        Returns
        -------
        xs    = a cases+1 column matrix  ( t,X1(t) X2(t) ...).
        xsder = a cases+1 column matrix  ( t,X1'(t) X2'(t) ...).
        Details
        -------
        Performs a fast and exact simulation of stationary zero mean
        Gaussian process through circulant embedding of the covariance matrix.
        If the ACF has a non-empty field .tr, then the transformation is
        applied to the simulated data, the result is a simulation of a
        transformed Gaussian process.
        Note: The simulation may give high frequency ripple when used with a
                small dt.
        Examples
        --------
        >>> import wafo.spectrum.models as sm
        >>> Sj = sm.Jonswap()
        >>> spec = Sj.tospecdata()   #Make spec
        >>> R = spec.tocovdata()
        >>> x = R.sim(ns=1000,dt=0.2)
        See also
        --------
        spec2sdat, gaus2dat
        References
        ----------
        C.R Dietrich and G. N. Newsam (1997)
        "Fast and exact simulation of stationary
        Gaussian process through circulant embedding
        of the Covariance matrix"
        SIAM J. SCI. COMPT. Vol 18, No 4, pp. 1088-1107
        '''

        # TODO fix it, it does not work

        # Add a nugget effect to ensure that round off errors
        # do not result in negative spectral estimates
        nugget = 0  # 10**-12

        _set_seed(iseed)
        self._is_valid_acf()
        acf = self.data.ravel()
        n = acf.size
        acf.shape = (n, 1)

        dt = self.sampling_period()

        x = zeros((ns, cases + 1))

        if derivative:
            xder = x.copy()

        # add a nugget effect to ensure that round off errors
        # do not result in negative spectral estimates
        acf[0] = acf[0] + nugget

        # Fast and exact simulation of simulation of stationary
        # Gaussian process throug circulant embedding of the
        # Covariance matrix
        floatinfo = finfo(float)
        if (abs(acf[-1]) > floatinfo.eps):  # assuming acf(n+1)==0
            m2 = 2 * n - 1
            nfft = 2 ** nextpow2(max(m2, 2 * ns))
            acf = r_[acf, zeros((nfft - m2, 1)), acf[-1:0:-1, :]]
            # warnings,warn('I am now assuming that ACF(k)=0 for k>MAXLAG.')
        else:  # ACF(n)==0
            m2 = 2 * n - 2
            nfft = 2 ** nextpow2(max(m2, 2 * ns))
            acf = r_[acf, zeros((nfft - m2, 1)), acf[n - 1:1:-1, :]]

        # m2=2*n-2
        spec = fft(acf, nfft, axis=0).real  # periodogram

        I = spec.argmax()
        k = flatnonzero(spec < 0)
        if k.size > 0:
            _msg = '''
                Not able to construct a nonnegative circulant vector from ACF.
                Apply parzen windowfunction to the ACF in order to avoid this.
                The returned result is now only an approximation.'''

            # truncating negative values to zero to ensure that
            # that this noise is not added to the simulated timeseries

            spec[k] = 0.

            ix = flatnonzero(k > 2 * I)
            if ix.size > 0:
                # truncating all oscillating values above 2 times the peak
                # frequency to zero to ensure that
                # that high frequency noise is not added to
                # the simulated timeseries.
                ix0 = k[ix[0]]
                spec[ix0:-ix0] = 0.0

        trunc = 1e-5
        max_spec = spec[I]
        k = flatnonzero(spec[I:-I] < max_spec * trunc)
        if k.size > 0:
            spec[k + I] = 0.
            # truncating small values to zero to ensure that
            # that high frequency noise is not added to
            # the simulated timeseries

        cases1 = int(cases / 2)
        cases2 = int(ceil(cases / 2))
        # Generate standard normal random numbers for the simulations

        # randn = np.random.randn
        epsi = random.randn(nfft, cases2) + 1j * random.randn(nfft, cases2)
        sqrt_spec = sqrt(spec / (nfft))  # sqrt(spec(wn)*dw )
        ephat = epsi * sqrt_spec  # [:,np.newaxis]
        y = fft(ephat, nfft, axis=0)
        x[:, 1:cases + 1] = hstack((y[2:ns + 2, 0:cases2].real,
                                    y[2:ns + 2, 0:cases1].imag))

        x[:, 0] = linspace(0, (ns - 1) * dt, ns)  # (0:dt:(dt*(np-1)))'

        if derivative:
            sqrt_spec = sqrt_spec * \
                r_[0:(nfft / 2 + 1), -(nfft / 2 - 1):0] * 2 * pi / nfft / dt
            ephat = epsi * sqrt_spec  # [:,newaxis]
            y = fft(ephat, nfft, axis=0)
            xder[:, 1:(cases + 1)] = hstack((y[2:ns + 2, 0:cases2].imag -
                                             y[2:ns + 2, 0:cases1].real))
            xder[:, 0] = x[:, 0]

        if self.tr is not None:
            print('   Transforming data.')
            g = self.tr
            if derivative:
                for ix in range(cases):
                    tmp = g.gauss2dat(x[:, ix + 1], xder[:, ix + 1])
                    x[:, ix + 1] = tmp[0]
                    xder[:, ix + 1] = tmp[1]
            else:
                for ix in range(cases):
                    x[:, ix + 1] = g.gauss2dat(x[:, ix + 1])

        if derivative:
            return x, xder
        else:
            return x

    def _get_lag_where_acf_is_almost_zero(self):
        acf = self.data.ravel()
        r0 = acf[0]
        n = len(acf)
        sigma = sqrt(r_[0, r0 ** 2,
                        r0 ** 2 + 2 * np.cumsum(acf[1:n - 1] ** 2)] / n)
        k = flatnonzero(np.abs(acf) > 0.1 * sigma)
        if k.size > 0:
            lag = min(k.max() + 3, n)
            return lag
        return n

    def _get_acf(self, smooth=False):
        self._is_valid_acf()
        acf = atleast_1d(self.data).ravel()
        n = self._get_lag_where_acf_is_almost_zero()
        if smooth:
            rwin = parzen(2 * n + 1)
            return acf[:n] * rwin[n:2 * n]
        else:
            return acf[:n]

    @staticmethod
    def _split_cov(sigma, i_known, i_unknown):
        '''
        Split covariance matrix between known/unknown observations
        Returns
        -------
        soo  covariance between known observations
        s1o = covariance between known and unknown obs
        s11 = covariance between unknown observations
        '''
        soo, so1 = sigma[i_known][:, i_known], sigma[i_known][:, i_unknown]
        s11 = sigma[i_unknown][:, i_unknown]
        return soo, so1, s11

    @staticmethod
    def _update_window(idx, i_unknown, num_x, num_acf,
                       overlap, nw, num_restored):
        num_sig = len(idx)
        start_max = num_x - num_sig
        if (nw == 0) and (num_restored < len(i_unknown)):
            # move to the next missing data
            start_ix = min(i_unknown[num_restored + 1] - overlap, start_max)
        else:
            start_ix = min(idx[0] + num_acf, start_max)

        return idx + start_ix - idx[0]


class CovarianceEstimator(object):
    """
    Class for estimating AutoCovariance from timeseries
    Parameters
    ----------
    lag : scalar, int
        maximum time-lag for which the ACF is estimated.
        (Default lag where ACF is zero)
    tr : transformation object
        the transformation assuming that x is a sample of a transformed
        Gaussian process. If g is None then x  is a sample of a Gaussian
        process (Default)
    detrend : function
        defining detrending performed on the signal before estimation.
        (default detrend_mean)
    window : vector of length NFFT or function
        To create window vectors see numpy.blackman, numpy.hamming,
        numpy.bartlett, scipy.signal, scipy.signal.get_window etc.
    flag : string, 'biased' or 'unbiased'
        If 'unbiased' scales the raw correlation by 1/(n-abs(k)),
        where k is the index into the result, otherwise scales the raw
        cross-correlation by 1/n. (default)
    norm : bool
        True if normalize output to one
    dt : scalar
        time-step between data points (default see sampling_period).
    """
    def __init__(self, lag=None, tr=None, detrend=None, window='boxcar',
                 flag='biased', norm=False, dt=None):
        self.lag = lag
        self.tr = tr
        self.detrend = detrend
        self.window = window
        self.flag = flag
        self.norm = norm
        self.dt = dt

    def _estimate_lag(self, R, Ncens):
        Lmax = min(300, len(R) - 1)  # maximum lag if L is undetermined
        # finding where ACF is less than 2 st. deviations.
        sigma = np.sqrt(np.r_[0, R[0] ** 2,
                              R[0] ** 2 + 2 * np.cumsum(R[1:] ** 2)] / Ncens)
        lag = Lmax + 2 - (np.abs(R[Lmax::-1]) > 2 * sigma[Lmax::-1]).argmax()
        if self.window == 'parzen':
            lag = int(4 * lag / 3)
        # print('The default L is set to %d' % L)
        return lag
    
    def tocovdata(self, timeseries):
        """
        Return auto covariance function from data.
        Return
        -------
        acf : CovData1D object
            with attributes:
            data : ACF vector length L+1
            args : time lags  length L+1
            sigma : estimated large lag standard deviation of the estimate
                    assuming x is a Gaussian process:
                    if acf[k]=0 for all lags k>q then an approximation
                    of the variance for large samples due to Bartlett
                     var(acf[k])=1/N*(acf[0]**2+2*acf[1]**2+2*acf[2]**2+ ..+2*acf[q]**2)
                     for  k>q and where  N=length(x). Special case is
                     white noise where it equals acf[0]**2/N for k>0
            norm : bool
                If false indicating that auto_cov is not normalized
         Examples
         --------
         >>> import wafo.data
         >>> import wafo.objects as wo
         >>> x = wafo.data.sea()
         >>> ts = wo.mat2timeseries(x)
         >>> acf = ts.tocovdata(150)
         h = acf.plot()
        """
        lag = self.lag
        window = self.window
        detrend = self.detrend

        try:
            x = timeseries.data.flatten('F')
            dt = timeseries.sampling_period()
        except Exception:
            x = timeseries[:, 1:].flatten('F')
            dt = sampling_period(timeseries[:, 0])
        if self.dt is not None:
            dt = self.dt

        if self.tr is not None:
            x = self.tr.dat2gauss(x)

        n = len(x)
        indnan = np.isnan(x)
        if any(indnan):
            x = x - x[1 - indnan].mean()
            Ncens = n - indnan.sum()
            x[indnan] = 0.
        else:
            Ncens = n
            x = x - x.mean()
        if hasattr(detrend, '__call__'):
            x = detrend(x)

        nfft = 2 ** nextpow2(n)
        raw_periodogram = abs(fft(x, nfft)) ** 2 / Ncens
        # ifft = fft/nfft since raw_periodogram is real!
        auto_cov = np.real(fft(raw_periodogram)) / nfft

        if self.flag.startswith('unbiased'):
            # unbiased result, i.e. divide by n-abs(lag)
            auto_cov = auto_cov[:Ncens] * Ncens / np.arange(Ncens, 1, -1)

        if self.norm:
            auto_cov = auto_cov / auto_cov[0]

        if lag is None:
            lag = self._estimate_lag(auto_cov, Ncens)
        lag = min(lag, n - 2)
        if isinstance(window, str) or type(window) is tuple:
            win = get_window(window, 2 * lag - 1)
        else:
            win = np.asarray(window)
        auto_cov[:lag] = auto_cov[:lag] * win[lag - 1::]
        auto_cov[lag] = 0
        lags = slice(0, lag + 1)
        t = np.linspace(0, lag * dt, lag + 1)
        acf = CovData1D(auto_cov[lags], t)
        acf.sigma = np.sqrt(np.r_[0, auto_cov[0] ** 2,
                            auto_cov[0] ** 2 + 2 * np.cumsum(auto_cov[1:] ** 2)] / Ncens)
        acf.children = [PlotData(-2. * acf.sigma[lags], t),
                        PlotData(2. * acf.sigma[lags], t)]
        acf.plot_args_children = ['r:']
        acf.norm = self.norm
        return acf

    __call__ = tocovdata