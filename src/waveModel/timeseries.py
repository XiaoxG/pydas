from scipy.signal import welch
import warnings
import numpy as np
from numpy import (pi, zeros, ones, sqrt, where, log, exp, cos, sin,
                   arcsin, mod,linspace, arange, sort, all, abs, vstack, hstack,
                   atleast_1d, finfo, polyfit, r_, nonzero,
                   cumsum, ravel, isnan, ceil, diff, array)
from numpy.random import randn
from matplotlib.mlab import detrend_mean
from scipy.signal.windows import parzen


from waveModel.core import nextpow2
from waveModel.covdata import CovarianceEstimator
from waveModel.specdata import SpecData1D
from waveModel.datacontainer import DataContainer

def array2timeseries(x):
    """
    Convert 2D arrays to TimeSeries object
        assuming 1st column is time and the remaining columns contain data.
    """
    return TimeSeries(x[:, 1::], x[:, 0].ravel())

class TimeSeries(DataContainer):
    '''
    Container class for 1D TimeSeries data objects in WAFO
    Member variables
    ----------------
    data : array_like
    args : vector for 1D, list of vectors for 2D, 3D, ...
    sensortypes : list of integers or strings
        sensor type for time series (default ['n']    : Surface elevation)
        see sensortype for more options
    position : vector of size 3
        instrument position relative to the coordinate system
    Examples
    --------
    >>> import wafo.data
    >>> import wafo.objects as wo
    >>> x = wafo.data.sea()
    >>> ts = wo.mat2timeseries(x)
    >>> rf = ts.tocovdata(lag=150)
    >>> S = ts.tospecdata()
    >>> tp = ts.turning_points()
    >>> mm = tp.cycle_pairs()
    >>> lc = mm.level_crossings()
    h = rf.plot()
    h1 = mm.plot(marker='x')
    h2 = lc.plot()
    '''

    def __init__(self, *args, **kwds):
        self.name_ = kwds.pop('name', 'WAFO TimeSeries Object')
        self.sensortypes = kwds.pop('sensortypes', ['n', ])
        self.position = kwds.pop('position', [zeros(3), ])

        super(TimeSeries, self).__init__(*args, **kwds)

        if not any(self.args):
            n = len(self.data)
            self.args = range(0, n)

    def sampling_period(self):
        '''
        Returns sampling interval
        Returns
        -------
        dt : scalar
            sampling interval, unit:
            [s] if lagtype=='t'
            [m] otherwise
        See also
        '''
        t_vec = self.args
        dt1 = t_vec[1] - t_vec[0]
        n = len(t_vec) - 1
        t = t_vec[-1] - t_vec[0]
        dt = t / n
        if abs(dt - dt1) > 1e-10:
            warnings.warn('Data is not uniformly sampled!')
        return dt

    def tocovdata(self, lag=None, tr=None, detrend=detrend_mean,
                  window='boxcar', flag='biased', norm=False, dt=None):
        '''
        Return auto covariance function from data.
        Parameters
        ----------
        lag : scalar, int
            maximum time-lag for which the ACF is estimated. (Default lag=n-1)
        flag : string, 'biased' or 'unbiased'
            If 'unbiased' scales the raw correlation by 1/(n-abs(k)),
            where k is the index into the result, otherwise scales the raw
            cross-correlation by 1/n. (default)
        norm : bool
            True if normalize output to one
        dt : scalar
            time-step between data points (default see sampling_period).
        Return
        -------
        R : CovData1D object
            with attributes:
            data : ACF vector length L+1
            args : time lags  length L+1
            sigma : estimated large lag standard deviation of the estimate
                     assuming x is a Gaussian process:
                     if R(k)=0 for all lags k>q then an approximation
                     of the variance for large samples due to Bartlett
                     var(R(k))=1/N*(R(0)^2+2*R(1)^2+2*R(2)^2+ ..+2*R(q)^2)
                     for  k>q and where  N=length(x). Special case is
                     white noise where it equals R(0)^2/N for k>0
            norm : bool
                If false indicating that R is not normalized
         Examples
         --------
         >>> import wafo.data
         >>> import wafo.objects as wo
         >>> x = wafo.data.sea()
         >>> ts = wo.mat2timeseries(x)
         >>> acf = ts.tocovdata(150)
         >>> np.allclose(acf.data[:3], [ 0.22368637,  0.20838473,  0.17110733])
         True
         h = acf.plot()
        '''
        estimate_cov = CovarianceEstimator(
            lag=lag, tr=tr, detrend=detrend, window=window, flag=flag,
            norm=norm, dt=dt)
        return estimate_cov(self)

    def _get_bandwidth_and_dof(self, wname, n, L, dt, ftype='w'):
        '''Returns bandwidth (rad/sec) and degrees of freedom
            used in chi^2 distribution
        '''
        if isinstance(wname, tuple):
            wname = wname[0]
        dof = int(dict(parzen=3.71,
                       hanning=2.67,
                       bartlett=3).get(wname, np.nan) * n / L)
        Be = dict(parzen=1.33, hanning=1,
                  bartlett=1.33).get(wname, np.nan) * 2 * pi / (L * dt)
        if ftype == 'f':
            Be = Be / (2 * pi)  # bandwidth in Hz
        return Be, dof

    def tospecdata(self, L=None, method='cov', detrend=detrend_mean,
                   window='parzen', noverlap=0, ftype='w', alpha=None):
        '''
        Estimate one-sided spectral density from data.
        Parameters
        ----------
        L : scalar integer
            maximum lag size of the window function. As L decreases the
            estimate becomes smoother and Bw increases. If we want to resolve
            peaks in S which is Bf (Hz or rad/sec) apart then Bw < Bf. If no
            value is given the lag size is set to be the lag where the auto
            correlation is less than 2 standard deviations. (maximum 300)
        tr : transformation object
            the transformation assuming that x is a sample of a transformed
            Gaussian process. If g is None then x  is a sample of a Gaussian
            process (Default)
        method : string
            defining estimation method. Options are
            'cov' :  Frequency smoothing using the window function
                    on the estimated autocovariance function.  (default)
            'psd' : Welch's averaged periodogram method with no overlapping
                batches
        detrend : function
            defining detrending performed on the signal before estimation.
            (default detrend_mean)
        window : vector of length NFFT or function
            To create window vectors see numpy.blackman, numpy.hamming,
            numpy.bartlett, scipy.signal, scipy.signal.get_window etc.
        noverlap : scalar int
             gives the length of the overlap between segments.
        ftype : character
            defining frequency type: 'w' or 'f'  (default 'w')
        Returns
        ---------
        spec : SpecData1D  object
        Examples
        --------
        >>> import wafo.data as wd
        >>> import wafo.objects as wo
        >>> x = wd.sea()
        >>> ts = wo.mat2timeseries(x)
        >>> S0 = ts.tospecdata(method='psd', L=150)
        >>> np.allclose(S0.data[21:25],
        ...     [0.1948925209459276, 0.19124901618176282, 0.1705625876220829, 0.1471870958122376],
        ...     rtol=1e-2)
        True
        >>> S = ts.tospecdata(L=150)
        >>> np.allclose(S.data[21:25],
        ...    [0.13991863694982026, 0.15264493584526717, 0.160156678854338, 0.1622894414741913],
        ...    rtol=1e-2)
        True
        >>> h = S.plot()
        See also
        --------
        dat2tr, dat2cov
        References:
        -----------
        Georg Lindgren and Holger Rootzen (1986)
        "Stationara stokastiska processer",  pp 173--176.
        Gareth Janacek and Louise Swift (1993)
        "TIME SERIES forecasting, simulation, applications",
        pp 75--76 and 261--268
        Emanuel Parzen (1962),
        "Stochastic Processes", HOLDEN-DAY,
        pp 66--103
        '''
        x = atleast_1d(self.data).ravel()
        dt = self._check_dt(dt=None)

        if L is None:
            acf = self.tocovdata(lag=300, dt=dt)
            L = min(300, acf.get_l2spike()[0])

        if method.startswith('cov'):
            acf = self.tocovdata(lag=L, dt=dt)
            acf.data = acf.data * parzen(2 * L + 1)
            spec = acf.tospecdata(ftype=ftype)
        else:  # method='psd'
            if x.ndim != 1:
                raise ValueError('Input array must be one dimensional!')

            n = len(x)
            nfft = 2 ** nextpow2(L)
            nfft = min(nfft, n)
            if noverlap is None:
                noverlap = 0
            # Fs=1./dt
            Fs = 1
            freq, specdens = welch(
                x * dt, fs=Fs, window=window, nperseg=nfft,
                noverlap=noverlap, nfft=None, detrend=detrend,
                return_onesided=True, scaling='density', axis=-1)

            if ftype == 'w':
                freq1 = freq * (2 * pi)
                specdens = specdens / (2 * pi)
            else:
                freq1 = freq

            specdens = specdens * 2  # Make it one-sided
            spec = SpecData1D(specdens.ravel(), freq1.ravel())
            spec.freqtype = ftype

        # spec.name = 'S(w)'
        spec.tr = None

        # The confidence interval method:
        spec.Bw, spec.dof = self._get_bandwidth_and_dof(window, n, L, dt,
                                                        ftype=ftype)
        spec.CI = None
        spec.alpha = alpha
        spec.L = L

        return spec

    def _check_dt(self, dt=None):
        if dt is None:
            dt = self.sampling_period()
        return dt
