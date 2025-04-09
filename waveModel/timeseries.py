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
from waveModel.dataframe import PlotData
from waveModel.plotbackend import plotbackend as plt

def array2timeseries(x):
    """
    Convert 2D arrays to TimeSeries object
        assuming 1st column is time and the remaining columns contain data.
    """
    return TimeSeries(x[:, 1::], x[:, 0].ravel())

class TimeSeries(PlotData):
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

        nugget = 1e-12
        rate = 2  # interpolationrate for frequency
        dt = self.sampling_period()

        yy = self.data.ravel()
        yy = detrend(yy) if hasattr(detrend, '__call__') else yy
        n = len(yy)

        estimate_L = L is None
        if method == 'cov' or estimate_L:
            tsy = TimeSeries(yy, self.args)
            R = tsy.tocovdata(lag=L, window=window)
            L = len(R.data) - 1
            if method == 'cov':
                # add a nugget effect to ensure that round off errors
                # do not result in negative spectral estimates
                spec = R.tospecdata(rate=rate, nugget=nugget)
        L = min(L, n - 1)
        if method == 'psd':
            nfft = 2 ** nextpow2(L)
            pad_to = rate * nfft  # Interpolate the spectrum with rate
            f, S = welch(yy, fs=1.0 / dt, window=window, nperseg=nfft,
                         noverlap=noverlap, nfft=pad_to, detrend=detrend,
                         return_onesided=True, scaling='density', axis=-1)
#             S, f = psd(yy, Fs=1. / dt, NFFT=nfft, detrend=detrend,
#                        window=win, noverlap=noverlap, pad_to=pad_to,
#                        scale_by_freq=True)
            fact = 2.0 * pi
            w = fact * f
            spec = SpecData1D(S / fact, w)
        elif method == 'cov':
            pass
        else:
            raise ValueError('Unknown method (%s)' % method)

        Be, _ = self._get_bandwidth_and_dof(window, n, L, dt, ftype)
        spec.Bw = Be
        spec.L = L
        spec.norm = False
        spec.note = 'method=%s' % method
        return spec

    # def plot_wave(self, sym1='k.', ts=None, sym2='k+', nfig=None, nsub=None, sigma=None, vfact=3):
    #     '''
    #     Plots the surface elevation of timeseries.
    #     Parameters
    #     ----------
    #     sym1, sym2 : string
    #         plot symbol and color for data and ts, respectively
    #                   (see PLOT)  (default 'k.' and 'k+')
    #     ts : TimeSeries or TurningPoints object
    #         to overplot data. default zero-separated troughs and crests.
    #     nsub : scalar integer
    #         Number of subplots in each figure. By default nsub is such that
    #         there are about 20 mean down crossing waves in each subplot.
    #         If nfig is not given and nsub is larger than 6 then nsub is
    #         changed to nsub=min(6,ceil(nsub/nfig))
    #     nfig : scalar integer
    #         Number of figures. By default nfig=ceil(Nsub/6).
    #     sigma : real scalar
    #         standard deviation of data.
    #     vfact : real scalar
    #         how large in stdev the vertical scale should be (default 3)
    #     Examples
    #     --------
    #     Plot x1 with red lines and mark troughs and crests with blue circles.
    #     >>> import wafo
    #     >>> x = wafo.data.sea()
    #     >>> ts150 = wafo.objects.mat2timeseries(x[:150,:])
    #     >>> h = ts150.plot_wave('r-', sym2='bo')
    #     See also
    #     --------
    #     findtc, plot
    #     '''

    #     nw = 20
    #     tn = self.args
    #     xn = self.data.ravel()
    #     indmiss = isnan(xn)  # indices to missing points
    #     indg = where(1 - indmiss)[0]
    #     if ts is None:
    #         tc_ix = findtc(xn[indg], 0, 'tw')[0]
    #         xn2 = xn[tc_ix]
    #         tn2 = tn[tc_ix]
    #     else:
    #         xn2 = ts.data
    #         tn2 = ts.args

    #     if sigma is None:
    #         sigma = xn[indg].std()

    #     if nsub is None:
    #         # about Nw mdc waves in each plot
    #         nsub = int(len(xn2) / (2 * nw)) + 1
    #     if nfig is None:
    #         nfig = int(ceil(nsub / 6))
    #         nsub = min(6, int(ceil(nsub / nfig)))

    #     n = len(xn)
    #     Ns = int(n / (nfig * nsub))
    #     ind = r_[0:Ns]

    #     XlblTxt = 'Time [sec]'
    #     dT = 1
    #     timespan = tn[ind[-1]] - tn[ind[0]]
    #     if abs(timespan) > 18000:  # more than 5 hours
    #         dT = 1 / (60 * 60)
    #         XlblTxt = 'Time (hours)'
    #     elif abs(timespan) > 300:  # more than 5 minutes
    #         dT = 1 / 60
    #         XlblTxt = 'Time (minutes)'

    #     if np.max(abs(xn[indg])) > 5 * sigma:
    #         XlblTxt = XlblTxt + ' (Spurious data since max > 5 std.)'

    #     plot = plt.plot
    #     subplot = plt.subplot
    #     figs = []
    #     for unused_iz in range(nfig):
    #         figs.append(plt.figure())
    #         plt.title('Surface elevation from mean water level (MWL).')
    #         for ix in range(nsub):
    #             if nsub > 1:
    #                 subplot(nsub, 1, ix + 1)
    #             h_scale = array([tn[ind[0]], tn[ind[-1]]])
    #             ind2 = where((h_scale[0] <= tn2) & (tn2 <= h_scale[1]))[0]
    #             plot(tn[ind] * dT, xn[ind], sym1)
    #             if len(ind2) > 0:
    #                 plot(tn2[ind2] * dT, xn2[ind2], sym2)
    #             plot(h_scale * dT, [0, 0], 'k-')
    #             # plt.axis([h_scale*dT, v_scale])
    #             for iy in [-2, 2]:
    #                 plot(h_scale * dT, iy * sigma * ones(2), ':')
    #             ind = ind + Ns
    #         plt.xlabel(XlblTxt)

    #     return figs

    # def plot_sp_wave(self, wave_idx_, *args, **kwds):
    #     """
    #     Plot specified wave(s) from timeseries
    #     Parameters
    #     ----------
    #     wave_idx : integer vector
    #         of indices to waves we want to plot, i.e., wave numbers.
    #     tz_idx : integer vector
    #         of indices to the beginning, middle and end of
    #         defining wave, i.e. for zero-downcrossing waves, indices to
    #         zerocrossings (default trough2trough wave)
    #     Examples
    #     --------
    #     Plot waves nr. 6,7,8 and waves nr. 12,13,...,17
    #     >>> import wafo
    #     >>> x = wafo.data.sea()
    #     >>> ts = wafo.objects.mat2timeseries(x[0:500,...])
    #     >>> h = ts.plot_sp_wave(np.r_[6:9,12:18])
    #     See also
    #     --------
    #     plot_wave, findtc
    #     """
    #     wave_idx = atleast_1d(wave_idx_).flatten()
    #     tz_idx = kwds.pop('tz_idx', None)
    #     if tz_idx is None:
    #         # finding trough to trough waves
    #         unused_tc_ind, tz_idx = findtc(self.data, 0, 'tw')

    #     dw = nonzero(abs(diff(wave_idx)) > 1)[0]
    #     Nsub = dw.size + 1
    #     Nwp = zeros(Nsub, dtype=int)
    #     if Nsub > 1:
    #         dw = dw + 1
    #         Nwp[Nsub - 1] = wave_idx[-1] - wave_idx[dw[-1]] + 1
    #         wave_idx[dw[-1] + 1:] = -2
    #         for ix in range(Nsub - 2, 1, -2):
    #             # of waves pr subplot
    #             Nwp[ix] = wave_idx[dw[ix] - 1] - wave_idx[dw[ix - 1]] + 1
    #             wave_idx[dw[ix - 1] + 1:dw[ix]] = -2

    #         Nwp[0] = wave_idx[dw[0] - 1] - wave_idx[0] + 1
    #         wave_idx[1:dw[0]] = -2
    #         wave_idx = wave_idx[wave_idx > -1]
    #     else:
    #         Nwp[0] = wave_idx[-1] - wave_idx[0] + 1

    #     Nsub = min(6, Nsub)
    #     Nfig = int(ceil(Nsub / 6))
    #     Nsub = min(6, int(ceil(Nsub / Nfig)))
    #     figs = []
    #     for unused_iy in range(Nfig):
    #         figs.append(plt.figure())
    #         for ix in range(Nsub):
    #             plt.subplot(Nsub, 1, mod(ix, Nsub) + 1)
    #             ind = r_[tz_idx[2 * wave_idx[ix] - 1]:tz_idx[
    #                 2 * wave_idx[ix] + 2 * Nwp[ix] - 1]]
    #             # indices to wave
    #             plt.plot(self.args[ind], self.data[ind], *args, **kwds)
    #             plt.hold('on')
    #             xi = [self.args[ind[0]], self.args[ind[-1]]]
    #             plt.plot(xi, [0, 0])

    #             if Nwp[ix] == 1:
    #                 plt.ylabel('Wave %d' % wave_idx[ix])
    #             else:
    #                 plt.ylabel(
    #                     'Wave %d - %d' % (wave_idx[ix],
    #                                       wave_idx[ix] + Nwp[ix] - 1))
    #         plt.xlabel('Time [sec]')
    #         # wafostamp
    #     return figs