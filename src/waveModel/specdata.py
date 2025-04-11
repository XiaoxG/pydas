from waveModel.core import ecross, now, nextpow2, discretize

import warnings
import os
import numpy as np
from numpy import (pi, inf, zeros, ones, where, nonzero,
                   flatnonzero, ceil, sqrt, exp, log, arctan2,
                   tanh, cosh, sinh, random, atleast_1d,
                   minimum, diff, isnan, r_, conj, mod,
                   hstack, vstack, interp, ravel, finfo, linspace,
                   arange, array, nan, newaxis, sign, meshgrid)
from scipy.integrate import simps, trapz
import scipy.interpolate as interpolate
from scipy.interpolate.interpolate import interp1d, interp2d
from numpy.fft import fft
from waveModel.plotbackend import plotbackend as plt
from waveModel.covdata import CovData1D
from waveModel.dispersion_relation import k2w, w2k
from waveModel.dataframe import PlotData
from waveModel.misc import (cart2polar, polar2cart, sub_dict_select, gravity as _gravity)
_EPS = np.finfo(float).eps
_TINY = np.finfo(float).tiny



def qtf(w, h=inf, g=9.81, method='winterstein', rtol=1e-7, atol=0):
    """
    Return Quadratic Transfer Function
    Parameters
    ------------
    w : array-like
        angular frequencies
    h : scalar
        water depth
    g : scalar
        acceleration of gravity
    Returns
    -------
    h_s   = sum frequency effects
    h_d   = difference frequency effects
    h_dii = diagonal of h_d
    Examples
    --------
    >>> w = np.r_[0.1, 1./3, 2./3, 1]
    >>> hs, hd, hdi = qtf(w, h=np.inf, g=9.81)
    >>> np.allclose(hs, [[ 0.00050968,  0.00308642,  0.01158115,  0.02573904],
    ...                  [ 0.00308642,  0.00566316,  0.01415789,  0.02831578],
    ...                  [ 0.01158115,  0.01415789,  0.02265262,  0.03681051],
    ...                  [ 0.02573904,  0.02831578,  0.03681051,  0.0509684 ]])
    True
    >>> np.allclose(hd, [[-0.        , -0.00257674, -0.01107147, -0.02522936],
    ...                  [-0.00257674, -0.        , -0.00849473, -0.02265262],
    ...                  [-0.01107147, -0.00849473, -0.        , -0.01415789],
    ...                  [-0.02522936, -0.02265262, -0.01415789, -0.        ]])
    True
    >>> hs2, hd2, hdi2 = qtf(w, h=1e+6, g=9.81, method='winterstein')
    >>> np.allclose(hs2, [[0.00050968,  0.00308642,  0.01158115,  0.02573904],
    ...                   [0.00308642,  0.00566316,  0.01415789,  0.02831578],
    ...                   [0.01158115,  0.01415789,  0.02265262,  0.03681051],
    ...                   [0.02573904,  0.02831578,  0.03681051,  0.0509684 ]])
    True
    >>> np.allclose(hd2, [[-2.50061328e-07,   1.38729557e-03,   8.18314621e-03, 2.06421189e-02],
    ...                   [1.38729557e-03,  -2.50005518e-07,   2.83135545e-03, 1.13261230e-02],
    ...                   [8.18314621e-03,   2.83135545e-03,  -2.50001380e-07, 2.83133750e-03],
    ...                   [2.06421189e-02,   1.13261230e-02,   2.83133750e-03, -2.50000613e-07]])
    True
    >>> w = np.r_[0, 1e-10, 1e-5, 1e-1]
    >>> hs, hd, hdi = qtf(w, h=np.inf, g=9.81)
    >>> np.allclose(hs, [[0.00000000e+00,   2.54841998e-22,   2.54841998e-12, 2.54841998e-04],
    ...                  [2.54841998e-22,   5.09683996e-22,   2.54841998e-12, 2.54841998e-04],
    ...                  [2.54841998e-12,   2.54841998e-12,   5.09683996e-12, 2.54842001e-04],
    ...                  [2.54841998e-04,   2.54841998e-04,   2.54842001e-04, 5.09683996e-04]])
    True
    >>> np.allclose(hd, [[-0.00000000e+00,  -2.54841998e-22,  -2.54841998e-12, -2.54841998e-04],
    ...                  [-2.54841998e-22,  -0.00000000e+00,  -2.54841998e-12, -2.54841998e-04],
    ...                  [-2.54841998e-12,  -2.54841998e-12,  -0.00000000e+00, -2.54841995e-04],
    ...                  [-2.54841998e-04,  -2.54841998e-04,  -2.54841995e-04, -0.00000000e+00]])
    True
    >>> hs2, hd2, hdi2 = qtf(w, h=1e+100, g=9.81, method='winterstein')
    >>> np.allclose(hs2, [[1.50234572e-63,   2.54841998e-22,   2.54841998e-12, 2.54841998e-04],
    ...                   [2.54841998e-22,   5.09683996e-22,   2.54841998e-12, 2.54841998e-04],
    ...                   [2.54841998e-12,   2.54841998e-12,   5.09683996e-12, 2.54842001e-04],
    ...                   [2.54841998e-04,   2.54841998e-04,   2.54842001e-04, 5.09683996e-04]],
    ...             atol=0)
    True
    >>> np.allclose(hd2, [[-2.50000000e-101,  2.54841998e-022,   2.54841998e-012, 2.54841998e-004],
    ...                   [2.54841998e-022,  -2.50000000e-101,   2.54836901e-012, 2.54841997e-004],
    ...                   [2.54841998e-012,   2.54836901e-012,  -2.50000000e-101, 2.54791032e-004],
    ...                   [2.54841998e-004,   2.54841997e-004,   2.54791032e-004, -2.500000e-101]],
    ...             atol=0)
    True
    References
    ----------
    Langley, RS (1987)
    'A statistical analysis of nonlinear random waves'
    Ocean Engineering, Vol 14, No 5, pp 389-407
    Marthinsen, T. and Winterstein, S.R (1992)
    'On the skewness of random surface waves'
    In proceedings of the 2nd ISOPE Conference, San Francisco, 14-19 june.
    """
#     >>> hs3, hd3, hdi3 = qtf(w, h=200, g=9.81, method='winterstein')
#     >>> hs3
#
#     >>> hd3
#
#     >>> np.allclose(hs3, [[ 0.        ,  0.00283158,  0.01132631,  0.0254842 ],
#     ...                  [ 0.00283158,  0.00566316,  0.01415789,  0.02831578],
#     ...                  [ 0.01132631,  0.01415789,  0.02265262,  0.03681051],
#     ...                  [ 0.0254842 ,  0.02831578,  0.03681051,  0.0509684 ]])
#
#     >>> np.allclose(hd3, [[-0.        , -0.00283158, -0.01132631, -0.0254842 ],
#     ...                  [-0.00283158, -0.        , -0.00849473, -0.02265262],
#     ...                  [-0.01132631, -0.00849473, -0.        , -0.01415789],
#     ...                  [-0.0254842 , -0.02265262, -0.01415789, -0.        ]])

    w = atleast_1d(w)
    num_w = w.size

    if h == inf:  # go here for faster calculations
        k_w = w2k(w, theta=0, h=h, g=g, rtol=rtol, atol=atol)[0]
        k_1, k_2 = meshgrid(k_w, k_w, sparse=True)
        h_s = 0.25 * (abs(k_1) + abs(k_2))
        h_d = -0.25 * abs(abs(k_1) - abs(k_2))
        h_dii = zeros(num_w)
        return h_s, h_d, h_dii

    w1 = w + _TINY ** (1. / 10) * (np.sign(w) * np.int_(np.abs(w) < _EPS) + np.int_(w == 0))

    w = w1
    # k_w += _TINY ** (1./3) * (np.sign(k_w) * np.int_(np.abs(k_w) < _EPS) + np.int_(k_w==0))
    k_w = w2k(w, theta=0, h=h, g=g, rtol=rtol, atol=atol)[0]
    k_1, k_2 = meshgrid(k_w, k_w, sparse=True)
    w_1, w_2 = meshgrid(w, w, sparse=True)

    w12 = w_1 * w_2
    w1p2 = w_1 + w_2
    w1m2 = w_1 - w_2
    k12 = k_1 * k_2
    k1p2 = k_1 + k_2
    k1m2 = abs(k_1 - k_2)

    if method.startswith('langley'):
        p_1 = (-2 * w1p2 * (k12 * g ** 2. - w12 ** 2.) +
               w_1 * (w_2 ** 4. - g ** 2 * k_2 ** 2) +
               w_2 * (w_1 ** 4 - g * 2. * k_1 ** 2)) / (4. * w12 + _TINY)

        p_2 = w1p2 ** 2. * cosh((k1p2) * h) - g * (k1p2) * sinh((k1p2) * h)
        p_2 += _TINY * np.int_(p_2 == 0)

        h_s = (-p_1 / p_2 * w1p2 * cosh((k1p2) * h) / g -
               (k12 * g ** 2 - w12 ** 2.) / (4 * g * w12 + _TINY) +
               (w_1 ** 2 + w_2 ** 2) / (4 * g))

        p_3 = (-2 * w1m2 * (k12 * g ** 2 + w12 ** 2) -
               w_1 * (w_2 ** 4 - g ** 2 * k_2 ** 2) +
               w_2 * (w_1 ** 4 - g ** 2 * k_1 ** 2)) / (4. * w12 + _TINY)
        p_4 = w1m2 ** 2. * cosh(k1m2 * h) - g * (k1m2) * sinh((k1m2) * h)
        p_4 += _TINY * np.int_(p_4 == 0)

        h_d = (-p_3 / p_4 * (w1m2) * cosh((k1m2) * h) / g -
               (k12 * g ** 2 + w12 ** 2) / (4 * g * w12 + _TINY) +
               (w_1 ** 2. + w_2 ** 2.) / (4. * g))

    else:  # Marthinsen & Winterstein
        tmp1 = 2.0 * g * k12 / (w12 + 0)
        tmp2 = (w_1 ** 2. + w_2 ** 2. + w12) / g

        h_s = 0.25 * ((tmp1 - tmp2
                       + g * (w_1 * k_2 ** 2. + w_2 * k_1 ** 2) / (w12 * w1p2 + 0))
                      / (1. - g * h * k1p2 / (w1p2 ** 2. + 0) * tanh(k1p2))
                      + tmp2 - 0.5 * tmp1)  # OK

        tiny_diag = _TINY * np.diag(np.ones(num_w))  # Avoid division by zero on diagonal
        tmp3 = (w_1 ** 2 + w_2 ** 2 - w12) / g  # OK
        numerator = (tmp1 - tmp3 - g * (w_1 * k_2 ** 2
                                        - w_2 * k_1 ** 2) / (w12 * w1m2 + tiny_diag))
        h_d = 0.25 * (numerator / (1. - g * h * k1m2 / (w1m2 ** 2. + tiny_diag) * tanh(k1m2))
                      + tmp3 - 0.5 * tmp1)  # OK

#         h_d = 0.25 * ((tmp1 - tmp3
#                        - g * (w_1 * k_2 ** 2 - w_2 * k_1 ** 2) / (w12 * w1m2 + tiny_diag))
#                       / (1. - g * h * k1m2 / (w1m2 ** 2. + tiny_diag) * tanh(k1m2))
#                       + tmp3 - 0.5 * tmp1)  # OK

    # tmp1 = 2 * g * (k_w./w)^2
    # tmp2 = w.^2/g
    # Wave group velocity
    k_h = k_w * h
    c_g = 0.5 * g * (tanh(k_h) + k_h / np.cosh(k_h) ** 2) / w
    numerator2 = (g * (k_w / w) ** 2. - w ** 2 / g + 2 * g * k_w / (w * c_g + 0))
    h_dii = 0.25 * (numerator2 / (1. - g * h / (c_g ** 2. + 0))
                    - 2 * k_w / sinh(2 * k_h))  # OK
#     c_g = 0.5 * g * (tanh(k_w * h) + k_w * h * (1.0 - tanh(k_w * h) ** 2)) / w
#     h_dii = (0.5 * (0.5 * g * (k_w / w) ** 2. - 0.5 * w ** 2 / g + g * k_w / (w * c_g + 0))
#              / (1. - g * h / (c_g ** 2. + 0))
#              - 0.5 * k_w / sinh(2 * k_w * h))  # OK
    h_d.flat[0::num_w + 1] = h_dii

    # infinite water
    #     >>> np.allclose(hs, [[ 0.        ,  0.00283158,  0.01132631,  0.0254842 ],
    #     ...                  [ 0.00283158,  0.00566316,  0.01415789,  0.02831578],
    #     ...                  [ 0.01132631,  0.01415789,  0.02265262,  0.03681051],
    #     ...                  [ 0.0254842 ,  0.02831578,  0.03681051,  0.0509684 ]])
    #     True
    #
    #     >>> np.allclose(hd, [[-0.        , -0.00283158, -0.01132631, -0.0254842 ],
    #     ...                  [-0.00283158, -0.        , -0.00849473, -0.02265262],
    #     ...                  [-0.01132631, -0.00849473, -0.        , -0.01415789],
    #     ...                  [-0.0254842 , -0.02265262, -0.01415789, -0.        ]])
    #     True

    # winterstein
    #     h_s =
    #      [[  0.00000000e+00  -1.64775418e+00  -6.95612056e-01  -4.18817231e-01 -3.15690232e-01]
    #  [ -1.64775418e+00  -7.98574421e-01  -2.21051428e-01  -6.73808482e-02 -1.69373060e-02]
    #  [ -6.95612056e-01  -2.21051428e-01  -1.29139936e-01  -4.11797418e-02 1.12063541e-03]
    #  [ -4.18817231e-01  -6.73808482e-02  -4.11797418e-02  -3.51718594e-03 2.44489725e-02]

    #     h_d =
    #  [[ 0.         -1.64775418 -0.69561206 -0.41881723 -0.31569023]
    #  [-1.6103978   0.0130494  -0.12128861 -0.05785645 -0.02806048]
    #  [-0.65467824 -0.12128861  0.01494093 -0.04402996 -0.02595442]
    #  [-0.35876732 -0.05785645 -0.04402996  0.01905565 -0.02218373]
    #  [        inf -0.02806048 -0.02595442 -0.02218373  0.02705736]]
    # langley
    # h_d = [[  0.00000000e+00  -8.87390092e+14  -3.87924869e+14  -1.66844106e+15 inf]
    #  [ -8.87390092e+14  -1.14566397e-02  -1.50113192e-01  -1.11791139e-01 -1.13090565e-01]
    #  [ -3.87924869e+14  -1.50113192e-01  -8.56987798e-03  -5.10233013e-02 -4.93936523e-02]
    #  [ -1.66844106e+15  -1.11791139e-01  -5.10233013e-02  -4.72078473e-03 -2.74040590e-02]
    #  [             inf  -1.13090565e-01  -4.93936523e-02  -2.74040590e-02 -1.57316125e-03]]
    #  h_s =
    #  [[  0.00000000e+00  -8.62422934e+14  -3.76136070e+14  -1.61053099e+15 inf]
    #  [ -8.87390092e+14   2.59936788e-01   1.22409408e-01   7.97392657e-02 6.16999831e-02]
    #  [ -3.87924869e+14   1.46564082e-01   7.02793126e-02   4.62059958e-02 3.58607610e-02]
    #  [ -1.66844106e+15   1.18356989e-01   5.82970744e-02   3.92688958e-02 3.13685586e-02]
    #  [             inf   1.25606419e-01   6.35218804e-02   4.41902963e-02 3.69195895e-02]]

    # k    = find(w_1==w_2)
    # h_d(k) = h_dii

    # The NaN's occur due to division by zero. => Set the isnans to zero

    h_dii = where(isnan(h_dii), 0, h_dii)
    h_d = where(isnan(h_d), 0, h_d)
    h_s = where(isnan(h_s), 0, h_s)

    return h_s, h_d, h_dii
def _set_seed(iseed):
    '''Set seed of random generator'''
    if iseed is not None:
        try:
            random.set_state(iseed)
        except (KeyError, TypeError):
            random.seed(iseed)

class SpecData1D(PlotData):
    """
    Container class for 1D spectrum data objects in WAFO
    Member variables
    ----------------
    data : array-like
        One sided Spectrum values, size nf
    args : array-like
        freguency/wave-number-lag values of freqtype, size nf
    type : String
        spectrum type, one of 'freq', 'k1d', 'enc' (default 'freq')
    freqtype : letter
        frequency type, one of: 'f', 'w' or 'k' (default 'w')
    tr : Transformation function (default (none)).
    h : real scalar
        Water depth (default inf).
    v : real scalar
        Ship speed, if type = 'enc'.
    norm : bool
        Normalization flag, True if S is normalized, False if not
    date : string
        Date and time of creation or change.
    Examples
    --------
    >>> import numpy as np
    >>> import wafo.spectrum.models as sm
    >>> Sj = sm.Jonswap(Hm0=3)
    >>> w = np.linspace(0,4,256)
    >>> S1 = Sj.tospecdata(w)   #Make spectrum object from numerical values
    >>> S = sm.SpecData1D(Sj(w),w) # Alternatively do it manually
    See also
    --------
    DataFrame
    CovData
    """

    def __init__(self, *args, **kwds):
        super(SpecData1D, self).__init__(*args, **kwds)
        self.name_ = kwds.pop('name', 'WAFO Spectrum Object')
        self.type = kwds.pop('type', 'freq')
        self._freqtype = kwds.pop('freqtype', 'w')
        self.angletype = ''
        self.h = kwds.pop('h', inf)
        self.tr = kwds.pop('tr', None)  # TrLinear()
        self.phi = kwds.pop('phi', 0.0)
        self.v = kwds.pop('v', 0.0)
        self.norm = kwds.pop('norm', False)
        self.w = kwds.pop('w', None)

        self.setlabels()

    @property
    def freqtype(self):
        return self._freqtype

    @freqtype.setter
    def freqtype(self, freqtype):
        if self._freqtype == freqtype:
            return  # do nothind
        if freqtype == 'w' and self._freqtype == 'f':
            self.args *= 2 * np.pi
            self.data /= 2 * np.pi
            self._freqtype = 'w'
            self.setlabels()
        elif freqtype == 'f' and self._freqtype == 'w':
            self.args /= 2 * np.pi
            self.data *= 2 * np.pi
            self._freqtype = 'f'
            self.setlabels()

    def _get_default_dt_and_rate(self, dt):
        dt_old = self.sampling_period()
        if dt is None:
            return dt_old, 1
        rate = max(round(dt_old * 1. / dt), 1.)
        return dt, int(rate)

    def _check_dt(self, dt):
        freq = self.args
        checkdt = 1.2 * min(diff(freq)) / 2. / pi
        if self.freqtype in 'f':
            checkdt *= 2 * pi
        if (checkdt < 2. ** -16 / dt):
            print('Step dt = %g in computation of the density is ' +
                  'too small.' % dt)
            print('The computed covariance (by FFT(2^K)) may differ from the')
            print('theoretical. Solution:')
            raise ValueError('use larger dt or sparser grid for spectrum.')

    @staticmethod
    def _check_cov_matrix(acfmat, nt, dt):
        eps0 = 0.0001
        if nt + 1 >= 5:
            cc2 = acfmat[0, 0] - acfmat[4, 0] * (acfmat[4, 0] / acfmat[0, 0])
            if (cc2 < eps0):
                warnings.warn('Step dt = %g in computation of the density ' +
                              'is too small.' % dt)
        cc1 = acfmat[0, 0] - acfmat[1, 0] * (acfmat[1, 0] / acfmat[0, 0])
        if (cc1 < eps0):
            warnings.warn('Step dt = %g is small, and may cause numerical ' +
                          'inaccuracies.' % dt)

    @property
    def lagtype(self):
        if self.freqtype in 'k':  # options are 'f' and 'w' and 'k'
            return 'x'
        return 't'

    def tocov_matrix(self, nr=0, nt=None, dt=None):
        '''
        Computes covariance function and its derivatives, alternative version
        Parameters
        ----------
        nr : scalar integer
            number of derivatives in output, nr<=4          (default 0)
        nt : scalar integer
            number in time grid, i.e., number of time-lags.
            (default rate*(n_f-1)) where rate = round(1/(2*f(end)*dt)) or
                     rate = round(pi/(w(n_f)*dt)) depending on S.
        dt : real scalar
            time spacing for acfmat
        Returns
        -------
        acfmat : [R0, R1,...Rnr], shape Nt+1 x Nr+1
            matrix with autocovariance and its derivatives, i.e., Ri (i=1:nr)
            are column vectors with the 1'st to nr'th derivatives of R0.
        NB! This routine requires that the spectrum grid is equidistant
           starting from zero frequency.
        Examples
        --------
        >>> import wafo.spectrum.models as sm
        >>> Sj = sm.Jonswap()
        >>> S = Sj.tospecdata()
        >>> acfmat = S.tocov_matrix(nr=3, nt=256, dt=0.1)
        >>> np.round(acfmat[:2,:],3)
        array([[ 3.061,  0.   , -1.677,  0.   ],
               [ 3.052, -0.167, -1.668,  0.187]])
        See also
        --------
        cov,
        resample,
        objects
        '''

        dt, rate = self._get_default_dt_and_rate(dt)
        self._check_dt(dt)

        freq = self.args
        n_f = len(freq)
        if nt is None:
            nt = rate * (n_f - 1)
        else:  # check if Nt is ok
            nt = minimum(nt, rate * (n_f - 1))

        spec = self.copy()
        spec.resample(dt)

        acf = spec.tocovdata(nr, nt, rate=1)
        acfmat = zeros((nt + 1, nr + 1), dtype=float)
        acfmat[:, 0] = acf.data[0:nt + 1]
        fieldname = 'R' + self.lagtype * nr
        for i in range(1, nr + 1):
            fname = fieldname[:i + 1]
            r_i = getattr(acf, fname)
            acfmat[:, i] = r_i[0:nt + 1]

        self._check_cov_matrix(acfmat, nt, dt)
        return acfmat

    def tocovdata(self, nr=0, nt=None, rate=None):
        '''
        Computes covariance function and its derivatives
        Parameters
        ----------
        nr : number of derivatives in output, nr<=4 (default = 0).
        nt : number in time grid, i.e., number of time-lags
              (default rate*(length(S.data)-1)).
        rate = 1,2,4,8...2**r, interpolation rate for R
               (default = 1, no interpolation)
        Returns
        -------
        R : CovData1D
            auto covariance function
        The input 'rate' with the spectrum gives the time-grid-spacing:
            dt=pi/(S.w[-1]*rate),
            S.w[-1] is the Nyquist freq.
        This results in the time-grid: 0:dt:Nt*dt.
        What output is achieved with different S and choices of Nt, Nx and Ny:
        1) S.type='freq' or 'dir', Nt set, Nx,Ny not set => R(time) (one-dim)
        2) S.type='k1d' or 'k2d', Nt set, Nx,Ny not set: => R(x) (one-dim)
        3) Any type, Nt and Nx set => R(x,time); Nt and Ny set => R(y,time)
        4) Any type, Nt, Nx and Ny set => R(x,y,time)
        5) Any type, Nt not set, Nx and/or Ny set
            => Nt set to default, goto 3) or 4)
        NB! This routine requires that the spectrum grid is equidistant
         starting from zero frequency.
        NB! If you are using a model spectrum, spec, with sharp edges
         to calculate covariances then you should probably round off the sharp
         edges like this:
        Examples
        --------
        >>> import wafo.spectrum.models as sm
        >>> Sj = sm.Jonswap()
        >>> S = Sj.tospecdata()
        >>> S.data[0:40] = 0.0
        >>> S.data[100:-1] = 0.0
        >>> Nt = len(S.data)-1
        >>> acf = S.tocovdata(nr=0, nt=Nt)
        >>> S1 = acf.tospecdata()
        h = S.plot('r')
        h1 = S1.plot('b:')
        R   = spec2cov(spec,0,Nt)
        win = parzen(2*Nt+1)
        R.data = R.data.*win(Nt+1:end)
        S1  = cov2spec(acf)
        R2  = spec2cov(S1)
        figure(1)
        plotspec(S),hold on, plotspec(S1,'r')
        figure(2)
        covplot(R), hold on, covplot(R2,[],[],'r')
        figure(3)
        semilogy(abs(R2.data-R.data)), hold on,
        semilogy(abs(S1.data-S.data)+1e-7,'r')
        See also
        --------
        cov2spec
        '''

        freq = self.args
        n_f = len(freq)

        if freq[0] > 0:
            txt = '''Spectrum does not start at zero frequency/wave number.
            Correct it with resample, for example.'''
            raise ValueError(txt)
        d_w = abs(diff(freq, n=2, axis=0))
        if np.any(d_w > 1.0e-8):
            txt = '''Not equidistant frequencies/wave numbers in spectrum.
            Correct it with resample, for example.'''
            raise ValueError(txt)

        if rate is None:
            rate = 1  # interpolation rate
        elif rate > 16:
            rate = 16
        else:  # make sure rate is a power of 2
            rate = 2 ** nextpow2(rate)

        if nt is None:
            nt = int(rate * (n_f - 1))
        else:  # check if Nt is ok
            nt = int(minimum(nt, rate * (n_f - 1)))

        spec = self.copy()

        lagtype = self.lagtype

        d_t = spec.sampling_period()
        # normalize spec so that sum(specn)/(n_f-1)=acf(0)=var(X)
        specn = spec.data * freq[-1]
        if spec.freqtype in 'f':
            w = freq * 2 * pi
        else:
            w = freq

        nfft = rate * 2 ** nextpow2(2 * n_f - 2)

        # periodogram
        rper = r_[
            specn, zeros(nfft - (2 * n_f) + 2), conj(specn[n_f - 2:0:-1])]
        time = r_[0:nt + 1] * d_t * (2 * n_f - 2) / nfft

        r = fft(rper, nfft).real / (2 * n_f - 2)
        acf = CovData1D(r[0:nt + 1], time, lagtype=lagtype)
        acf.tr = spec.tr
        acf.h = spec.h
        acf.norm = spec.norm

        if nr > 0:
            w = r_[w, zeros(nfft - 2 * n_f + 2), -w[n_f - 2:0:-1]]
            fieldname = 'R' + lagtype[0] * nr
            for i in range(1, nr + 1):
                rper = -1j * w * rper
                d_acf = fft(rper, nfft).real / (2 * n_f - 2)
                setattr(acf, fieldname[0:i + 1], d_acf[0:nt + 1])
        return acf

    def to_specnorm(self):
        S = self.copy()
        S.normalize()
        return S

    def sim(self, ns=None, cases=1, dt=None, iseed=None, method='random',
            derivative=False):
        ''' Simulates a Gaussian process and its derivative from spectrum
        Parameters
        ----------
        ns : scalar
            number of simulated points.  (default length(spec)-1=n-1).
                     If ns>n-1 it is assummed that acf(k)=0 for all k>n-1
        cases : scalar
            number of replicates (default=1)
        dt : scalar
            step in grid (default dt is defined by the Nyquist freq)
        iseed : int or state
            starting state/seed number for the random number generator
            (default none is set)
        method : string
            if 'exact'  : simulation using cov2sdat
            if 'random' : random phase and amplitude simulation (default)
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
        Gaussian process through circulant embedding of the covariance matrix
        or by summation of sinus functions with random amplitudes and random
        phase angle.
        If the spectrum has a non-empty field .tr, then the transformation is
        applied to the simulated data, the result is a simulation of a
        transformed Gaussian process.
        Note: The method 'exact' simulation may give high frequency ripple when
        used with a small dt. In this case the method 'random' works better.
        Examples
        --------
        >>> import wafo.spectrum.models as sm
        >>> Sj = sm.Jonswap();S = Sj.tospecdata()
        >>> ns =100; dt = .2
        >>> x1 = S.sim(ns,dt=dt)
        >>> import numpy as np
        >>> import scipy.stats as st
        >>> x2 = S.sim(20000,20)
        >>> truth1 = [0,np.sqrt(S.moment(1)[0]),0., 0.]
        >>> funs = [np.mean,np.std,st.skew,st.kurtosis]
        >>> for fun,trueval in zip(funs,truth1):
        ...     res = fun(x2[:,1::],axis=0)
        ...     m = res.mean()
        ...     sa = res.std()
        ...     #trueval, m, sa
        ...     np.abs(m-trueval)<sa
        True
        array([ True], dtype=bool)
        True
        True
        waveplot(x1,'r',x2,'g',1,1)
        See also
        --------
        cov2sdat, gaus2dat
        Reference
        -----------
        C.S Dietrich and G. N. Newsam (1997)
        "Fast and exact simulation of stationary
        Gaussian process through circulant embedding
        of the Covariance matrix"
        SIAM J. SCI. COMPT. Vol 18, No 4, pp. 1088-1107
        Hudspeth, S.T. and Borgman, L.E. (1979)
        "Efficient FFT simulation of Digital Time sequences"
        Journal of the Engineering Mechanics Division, ASCE, Vol. 105, No. EM2,
        '''

        spec = self.copy()
        if dt is not None:
            spec.resample(dt)

        ftype = spec.freqtype
        freq = spec.args

        d_t = spec.sampling_period()
        Nt = freq.size

        if ns is None:
            ns = Nt - 1

        if method in 'exact':

            # nr=0,Nt=None,dt=None
            acf = spec.tocovdata(nr=0)
            T = Nt * d_t
            i = flatnonzero(acf.args > T)

            # Trick to avoid adding high frequency noise to the spectrum
            if i.size > 0:
                acf.data[i[0]::] = 0.0

            return acf.sim(ns=ns, cases=cases, iseed=iseed,
                           derivative=derivative)

        _set_seed(iseed)

        ns = ns + mod(ns, 2)  # make sure it is even

        f_i = freq[1:-1]
        s_i = spec.data[1:-1]
        if ftype in ('w', 'k'):
            fact = 2. * pi
            s_i = s_i * fact
            f_i = f_i / fact

        x = zeros((ns, cases + 1))

        d_f = 1 / (ns * d_t)

        # interpolate for freq.  [1:(N/2)-1]*d_f and create 2-sided, uncentered
        # spectra
        ns2 = ns // 2
        f = arange(1, ns2) * d_f

        f_u = hstack((0., f_i, d_f * ns2))
        s_u = hstack((0., abs(s_i) / 2, 0.))

        s_i = interp(f, f_u, s_u)
        s_u = hstack((0., s_i, 0, s_i[ns2 - 2::-1]))
        del(s_i, f_u)

        # Generate standard normal random numbers for the simulations
        randn = random.randn
        z_r = randn(ns2 + 1, cases)
        z_i = vstack(
            (zeros((1, cases)), randn(ns2 - 1, cases), zeros((1, cases))))

        amp = zeros((ns, cases), dtype=complex)
        amp[0:ns2 + 1, :] = z_r - 1j * z_i
        del(z_r, z_i)
        amp[ns2 + 1:ns, :] = amp[ns2 - 1:0:-1, :].conj()
        amp[0, :] = amp[0, :] * sqrt(2.)
        amp[ns2, :] = amp[ns2, :] * sqrt(2.)

        # Make simulated time series
        T = (ns - 1) * d_t
        Ssqr = sqrt(s_u * d_f / 2.)

        # stochastic amplitude
        amp = amp * Ssqr[:, newaxis]

        # Deterministic amplitude
        # amp =
        # sqrt[1]*Ssqr(:,ones(1,cases)) * \
        #            exp(sqrt(-1)*atan2(imag(amp),real(amp)))
        del(s_u, Ssqr)

        x[:, 1::] = fft(amp, axis=0).real
        x[:, 0] = linspace(0, T, ns)  # ' %(0:d_t:(np-1)*d_t).'

        if derivative:
            xder = zeros(ns, cases + 1)
            w = 2. * pi * hstack((0, f, 0., -f[-1::-1]))
            amp = -1j * amp * w[:, newaxis]
            xder[:, 1:(cases + 1)] = fft(amp, axis=0).real
            xder[:, 0] = x[:, 0]

        if spec.tr is not None:
            # print('   Transforming data.')
            g = spec.tr
            if derivative:
                for i in range(cases):
                    x[:, i + 1], xder[:, i + 1] = g.gauss2dat(x[:, i + 1],
                                                              xder[:, i + 1])
            else:
                for i in range(cases):
                    x[:, i + 1] = g.gauss2dat(x[:, i + 1])

        if derivative:
            return x, xder
        else:
            return x

    def stats_nl(self, h=None, moments='sk', method='approximate', g=9.81):
        """
        Statistics of 2'nd order waves to the leading order.
        Parameters
        ----------
        h : scalar
            water depth (default self.h)
        moments : string (default='sk')
            composed of letters ['mvsk'] specifying which moments to compute:
                   'm' = mean,
                   'v' = variance,
                   's' = skewness,
                   'k' = (Pearson's) kurtosis.
        method : string
            'approximate' method due to Marthinsen & Winterstein (default)
            'eigenvalue'  method due to Kac and Siegert
        Skewness = kurtosis-3 = 0 for a Gaussian process.
        The mean, sigma, skewness and kurtosis are determined as follows:
        method == 'approximate':  due to Marthinsen and Winterstein
        mean  = 2 * int Hd(w1,w1)*S(w1) dw1
        sigma = sqrt(int S(w1) dw1)
        skew  = 6 * int int [Hs(w1,w2)+Hd(w1,w2)]*S(w1)*S(w2) dw1*dw2/m0^(3/2)
        kurt  = (4*skew/3)^2
        where Hs = sum frequency effects  and Hd = difference frequency effects
        method == 'eigenvalue'
        mean  = sum(E)
        sigma = sqrt(sum(C^2)+2*sum(E^2))
        skew  = sum((6*C^2+8*E^2).*E)/sigma^3
        kurt  = 3+48*sum((C^2+E^2).*E^2)/sigma^4
        where
            h1 = sqrt(S*dw/2)
            C  = (ctranspose(V)*[h1;h1])
        and E and V is the eigenvalues and eigenvectors, respectively, of the
        2'order transfer matrix.
        S is the spectrum and dw is the frequency spacing of S.
        Examples
        --------
        # Simulate a Transformed Gaussian process:
        >>> import wafo.spectrum.models as sm
        >>> import wafo.transform.models as wtm
        >>> Hs = 7.
        >>> Sj = sm.Jonswap(Hm0=Hs, Tp=11)
        >>> S = Sj.tospecdata()
        >>> me, va, sk, ku = S.stats_nl(moments='mvsk')
        >>> g = wtm.TrHermite(mean=me, sigma=Hs/4, skew=sk, kurt=ku,
        ...                    ysigma=Hs/4)
        >>> ys = S.sim(15000)         # Simulated in the Gaussian world
        >>> xs = g.gauss2dat(ys[:,1]) # Transformed to the real world
        See also
        ---------
        transform.TrHermite
        transform.TrOchi
        objects.LevelCrossings.trdata
        objects.TimeSeries.trdata
        References:
        -----------
        Langley, RS (1987)
        'A statistical analysis of nonlinear random waves'
        Ocean Engineering, Vol 14, No 5, pp 389-407
        Marthinsen, T. and Winterstein, S.R (1992)
        'On the skewness of random surface waves'
        In proceedings of the 2nd ISOPE Conference, San Francisco, 14-19 june.
        Winterstein, S.R, Ude, T.C. and Kleiven, G. (1994)
        'Springing and slow drift responses:
        predicted extremes and fatigue vs. simulation'
        In Proc. 7th International behaviour of Offshore structures, (BOSS)
        Vol. 3, pp.1-15
        """

        #  default options
        if h is None:
            h = self.h

        # S = ttspec(S,'w')
        w = ravel(self.args)
        S = ravel(self.data)
        if self.freqtype in ['f', 'w']:
            # vari = 't'
            if self.freqtype == 'f':
                w = 2. * pi * w
                S = S / (2. * pi)
        # m0 = self.moment(nr=0)
        m0 = simps(S, w)
        sa = sqrt(m0)
        # Nw = w.size

        Hs, Hd, Hdii = qtf(w, h, g)

        # return
        # skew=6/sqrt(m0)^3*simpson(S.w,
        #            simpson(S.w,(Hs+Hd).*S1(:,ones(1,Nw))).*S1.')

        Hspd = trapz(trapz((Hs + Hd) * S[newaxis, :], w) * S, w)
        output = []
        # %approx : Marthinsen, T. and Winterstein, S.R (1992) method
        if method[0] == 'a':
            if 'm' in moments:
                output.append(2. * trapz(Hdii * S, w))
            if 'v' in moments:
                output.append(m0)
            skew = 6. / sa ** 3 * Hspd
            if 's' in moments:
                output.append(skew)
            if 'k' in moments:
                output.append((4. * skew / 3.) ** 2. + 3.)
        else:
            raise ValueError('Unknown option!')
        return output

    def moment(self, nr=2, even=True, j=0):
        ''' Calculates spectral moments from spectrum
        Parameters
        ----------
        nr: int
            order of moments (recomended maximum 4)
        even : bool
            False for all moments,
            True for only even orders
        j: int
            0 or 1
        Returns
        -------
        m     : list of moments
        mtext : list of strings describing the elements of m, see below
        Details
        -------
        Calculates spectral moments of up to order NR by use of
        Simpson-integration.
                 /                                  /
        mj_t^i = | w^i S(w)^(j+1) dw,  or  mj_x^i = | k^i S(k)^(j+1) dk
                 /                                  /
        where k=w^2/gravity, i=0,1,...,NR
        The strings in output mtext have the same position in the list
        as the corresponding numerical value has in output m
        Notation in mtext: 'm0' is the variance,
                        'm0x' is the first-order moment in x,
                       'm0xx' is the second-order moment in x,
                       'm0t'  is the first-order moment in t,
                             etc.
        For the calculation of moments see Baxevani et al.
        Examples
        --------
        >>> import numpy as np
        >>> import wafo.spectrum.models as sm
        >>> Sj = sm.Jonswap(Hm0=3, Tp=7)
        >>> w = np.linspace(0,4,256)
        >>> S = SpecData1D(Sj(w),w) #Make spectrum object from numerical values
        >>> mom, mom_txt = S.moment()
        >>> np.allclose(mom, [0.5616342024616453, 0.7309966918203602])
        True
        >>> mom_txt == ['m0', 'm0tt']
        True
        References
        ----------
        Baxevani A. et al. (2001)
        Velocities for Random Surfaces
        '''
        one_dim_spectra = ['freq', 'enc', 'k1d']
        if self.type not in one_dim_spectra:
            raise ValueError('Unknown spectrum type!')

        f = ravel(self.args)
        S = ravel(self.data)
        if self.freqtype in ['f', 'w']:
            vari = 't'
            if self.freqtype == 'f':
                f = 2. * pi * f
                S = S / (2. * pi)
        else:
            vari = 'x'
        S1 = abs(S) ** (j + 1.)
        m = [simps(S1, x=f)]
        mtxt = 'm%d' % j
        mtext = [mtxt]
        step = mod(even, 2) + 1
        df = f ** step
        for i in range(step, nr + 1, step):
            S1 = S1 * df
            m.append(simps(S1, x=f))
            mtext.append(mtxt + vari * i)
        return m, mtext

    def nyquist_freq(self):
        """
        Return Nyquist frequency
        Examples
        --------
        >>> import wafo.spectrum.models as sm
        >>> Sj = sm.Jonswap(Hm0=5)
        >>> S = Sj.tospecdata() #Make spectrum ob
        >>> S.nyquist_freq()
        3.0
        """
        return self.args[-1]

    def sampling_period(self):
        ''' Returns sampling interval from Nyquist frequency of spectrum
        Returns
        ---------
        dT : scalar
            sampling interval, unit:
            [m] if wave number spectrum,
            [s] otherwise
        Let wm be maximum frequency/wave number in spectrum, then
            dT=pi/wm
        if angular frequency,
            dT=1/(2*wm)
        if natural frequency (Hz)
        Examples
        --------
        >>> import wafo.spectrum.models as sm
        >>> Sj = sm.Jonswap(Hm0=5)
        >>> S = Sj.tospecdata() #Make spectrum ob
        >>> S.sampling_period()
        1.0471975511965976
        See also
        '''

        if self.freqtype == 'f':
            wmdt = 0.5  # Nyquist to sampling interval factor
        else:  # ftype == w og ftype == k
            wmdt = pi

        wm = self.args[-1]  # Nyquist frequency
        dt = wmdt / wm  # sampling interval = 1/Fs
        return dt

    def resample(self, dt=None, Nmin=0, Nmax=2 ** 13 + 1, method='stineman'):
        '''
        Interpolate and zero-padd spectrum to change Nyquist freq.
        Parameters
        ----------
        dt : real scalar
            wanted sampling interval (default as given by S, see spec2dt)
            unit: [s] if frequency-spectrum, [m] if wave number spectrum
        Nmin, Nmax : scalar integers
            minimum and maximum number of frequencies, respectively.
        method : string
            interpolation method (options are 'linear', 'cubic' or 'stineman')
        To be used before simulation (e.g. spec2sdat) or evaluation of
        covariance function (spec2cov) to get the wanted sampling interval.
        The input spectrum is interpolated and padded with zeros to reach
        the right max-frequency, w[-1]=pi/dt, f(end)=1/(2*dt), or k[-1]=pi/dt.
        The objective is that output frequency grid should be at least as dense
        as the input grid, have equidistant spacing and length equal to
        2^k+1 (>=Nmin). If the max frequency is changed, the number of points
        in the spectrum is maximized to 2^13+1.
        Note: Also zero-padding down to zero freq, if S does not start there.
        If empty input dt, this is the only effect.
        See also
        --------
        spec2cov, spec2sdat, covinterp, spec2dt
        '''

        ftype = self.freqtype
        w = self.args.ravel()
        n = w.size

        # doInterpolate = 0
        # Nyquist to sampling interval factor
        Cnf2dt = 0.5 if ftype == 'f' else pi  # % ftype == w og ftype == k

        wnOld = w[-1]         # Old Nyquist frequency
        dTold = Cnf2dt / wnOld  # sampling interval=1/Fs
        # dTold = self.sampling_period()

        if dt is None:
            dt = dTold

        # Find how many points that is needed
        nfft = 2 ** nextpow2(max(n - 1, Nmin - 1))
        dttest = dTold * (n - 1) / nfft

        while (dttest > dt) and (nfft < Nmax - 1):
            nfft = nfft * 2
            dttest = dTold * (n - 1) / nfft

        nfft = nfft + 1

        wnNew = Cnf2dt / dt  # % New Nyquist frequency
        dWn = wnNew - wnOld
        doInterpolate = dWn > 0 or w[1] > 0 or (
            nfft != n) or dt != dTold or np.any(abs(diff(w, axis=0)) > 1.0e-8)

        if doInterpolate > 0:
            S1 = self.data

            dw = min(diff(w))

            if dWn > 0:
                # add a zero just above old max-freq, and a zero at new
                # max-freq to get correct interpolation there
                Nz = 1 + (dWn > dw)  # % Number of zeros to add
                if Nz == 2:
                    w = hstack((w, wnOld + dw, wnNew))
                else:
                    w = hstack((w, wnNew))

                S1 = hstack((S1, zeros(Nz)))

            if w[0] > 0:
                # add a zero at freq 0, and, if there is space, a zero just
                # below min-freq
                Nz = 1 + (w[0] > dw)  # % Number of zeros to add
                if Nz == 2:
                    w = hstack((0, w[0] - dw, w))
                else:
                    w = hstack((0, w))

                S1 = hstack((zeros(Nz), S1))

            # Do a final check on spacing in order to check that the gridding
            # is sufficiently dense:
            # np1 = S1.size
            dwMin = finfo(float).max
            # wnc = min(wnNew,wnOld-1e-5)
            wnc = wnNew
            # specfun = lambda xi : stineman_interp(xi, w, S1)
            specfun = interpolate.interp1d(w, S1, kind='cubic')
            x = discretize(specfun, 0, wnc)[0]
            dwMin = minimum(min(diff(x)), dwMin)

            newNfft = 2 ** nextpow2(ceil(wnNew / dwMin)) + 1
            if newNfft > nfft:
                # if (nfft <= 2 ** 15 + 1) and (newNfft > 2 ** 15 + 1):
                #    warnings.warn('Spectrum matrix is very large (>33k). ' +
                #        'Memory problems may occur.')
                nfft = newNfft
            self.args = linspace(0, wnNew, nfft)
            intfun = interpolate.interp1d(w, S1, kind=method)
            self.data = intfun(self.args)
            self.data = self.data.clip(0)  # clip negative values to 0

    def interp(self, dt):
        S = self.copy()
        S.resample(dt)
        return S

    def normalize(self, gravity=9.81):
        '''
        Normalize a spectral density such that m0=m2=1
        Paramter
        --------
        gravity=9.81
        Notes
        -----
        Normalization performed such that
            INT S(freq) dfreq = 1       INT freq^2  S(freq) dfreq = 1
        where integration limits are given by  freq  and  S(freq)  is the
        spectral density; freq can be frequency or wave number.
        The normalization is defined by
            A=sqrt(m0/m2); B=1/A/m0; freq'=freq*A; S(freq')=S(freq)*B
        If S is a directional spectrum then a normalized gravity (.g) is added
        to Sn, such that mxx normalizes to 1, as well as m0 and mtt.
        (See spec2mom for notation of moments)
        If S is complex-valued cross spectral density which has to be
        normalized, then m0, m2 (suitable spectral moments) should be given.
        Examples
        --------
        >>> import wafo.spectrum.models as sm
        >>> Sj = sm.Jonswap(Hm0=5)
        >>> S = Sj.tospecdata() #Make spectrum ob
        >>> np.allclose(S.moment(2)[0],
        ...    [1.5614600345079888, 0.95567089481941048])
        True
        >>> Sn = S.copy(); Sn.normalize()
        Now the moments should be one
        >>> np.allclose(Sn.moment(2)[0], [1.0, 1.0])
        True
        '''
        mom = self.moment(nr=4, even=True)[0]
        m0 = mom[0]
        m2 = mom[1]
        m4 = mom[2]

        SM0 = sqrt(m0)
        SM2 = sqrt(m2)
        A = SM0 / SM2
        B = SM2 / (SM0 * m0)

        if self.freqtype == 'f':
            self.args = self.args * A / 2 / pi
            self.data = self.data * B * 2 * pi
        elif self.freqtype == 'w':
            self.args = self.args * A
            self.data = self.data * B
            m02 = m4 / gravity ** 2
            m20 = m02
            self.g = gravity * sqrt(m0 * m20) / m2
        self.A = A
        self.norm = True
        self.date = now()

    def bandwidth(self, factors=0):
        '''
        Return some spectral bandwidth and irregularity factors
        Parameters
        -----------
        factors : array-like
            Input vector 'factors' correspondence:
            0 alpha=m2/sqrt(m0*m4)                        (irregularity factor)
            1 eps2 = sqrt(m0*m2/m1^2-1)                   (narrowness factor)
            2 eps4 = sqrt(1-m2^2/(m0*m4))=sqrt(1-alpha^2) (broadness factor)
            3 Qp=(2/m0^2)int_0^inf f*S(f)^2 df            (peakedness factor)
        Returns
        --------
        bw : arraylike
            vector of bandwidth factors
            Order of output is the same as order in 'factors'
        Examples
        --------
        >>> import numpy as np
        >>> import wafo.spectrum.models as sm
        >>> Sj = sm.Jonswap(Hm0=3, Tp=7)
        >>> w = np.linspace(0,4,256)
        >>> S = SpecData1D(Sj(w),w) #Make spectrum object from numerical values
        >>> S.bandwidth([0,'eps2',2,3])
        array([ 0.73062845,  0.34476034,  0.68277527,  2.90817052])
        '''

        m = self.moment(nr=4, even=False)[0]
        if isinstance(factors, str):
            factors = [factors]
        fact_dict = dict(alpha=0, eps2=1, eps4=3, qp=3, Qp=3)
        fact = array([fact_dict.get(idx, idx)
                      for idx in list(factors)], dtype=int)

        # fact = atleast_1d(fact)
        alpha = m[2] / sqrt(m[0] * m[4])
        eps2 = sqrt(m[0] * m[2] / m[1] ** 2. - 1.)
        eps4 = sqrt(1. - m[2] ** 2. / m[0] / m[4])
        f = self.args
        S = self.data
        Qp = 2 / m[0] ** 2. * simps(f * S ** 2, x=f)
        bw = array([alpha, eps2, eps4, Qp])
        return bw[fact]

    def characteristic(self, fact='Hm0', T=1200, g=9.81):
        """
        Returns spectral characteristics and their covariance
        Parameters
        ----------
        fact : vector with factor integers or a string or a list of strings
            defining spectral characteristic, see description below.
        T  : scalar
            recording time (sec) (default 1200 sec = 20 min)
        g : scalar
            acceleration of gravity [m/s^2]
        Returns
        -------
        ch : vector
            of spectral characteristics
        R  : matrix
            of the corresponding covariances given T
        chtext : a list of strings
            describing the elements of ch, see example.
        Description
        ------------
        If input spectrum is of wave number type, output are factors for
        corresponding 'k1D', else output are factors for 'freq'.
        Input vector 'factors' correspondence:
        1 Hm0   = 4*sqrt(m0)                         Significant wave height
        2 Tm01  = 2*pi*m0/m1                         Mean wave period
        3 Tm02  = 2*pi*sqrt(m0/m2)                   Mean zero-crossing period
        4 Tm24  = 2*pi*sqrt(m2/m4)                   Mean period between maxima
        5 Tm_10 = 2*pi*m_1/m0                        Energy period
        6 Tp    = 2*pi/{w | max(S(w))}               Peak period
        7 Ss    = 2*pi*Hm0/(g*Tm02^2)                Significant wave steepness
        8 Sp    = 2*pi*Hm0/(g*Tp^2)                  Average wave steepness
        9 Ka    = abs(int S(w)*exp(i*w*Tm02) dw ) /m0  Groupiness parameter
        10 Rs    = (S(0.092)+S(0.12)+S(0.15)/(3*max(S(w)))
                                                     Quality control parameter
        11 Tp1   = 2*pi*int S(w)^4 dw                Peak Period
                  ------------------                 (robust estimate for Tp)
                  int w*S(w)^4 dw
        12 alpha = m2/sqrt(m0*m4)                          Irregularity factor
        13 eps2  = sqrt(m0*m2/m1^2-1)                      Narrowness factor
        14 eps4  = sqrt(1-m2^2/(m0*m4))=sqrt(1-alpha^2)    Broadness factor
        15 Qp    = (2/m0^2)int_0^inf w*S(w)^2 dw           Peakedness factor
        Order of output is same as order in 'factors'
        The covariances are computed with a Taylor expansion technique
        and is currently only available for factors 1, 2, and 3. Variances
        are also available for factors 4,5,7,12,13,14 and 15
        Quality control:
        ----------------
        Critical value for quality control parameter Rs is Rscrit = 0.02
        for surface displacement records and Rscrit=0.0001 for records of
        surface acceleration or slope. If Rs > Rscrit then probably there
        are something wrong with the lower frequency part of S.
        Ss may be used as an indicator of major malfunction, by checking that
        it is in the range of 1/20 to 1/16 which is the usual range for
        locally generated wind seas.
        Examples:
        ---------
        >>> import wafo.spectrum.models as sm
        >>> Sj = sm.Jonswap(Hm0=5)
        >>> S = Sj.tospecdata() #Make spectrum ob
        >>> S.characteristic(1)
        (array([ 8.59007646]), array([[ 0.03040216]]), ['Tm01'])
        >>> [ch, R, txt] = S.characteristic([1,2,3])  # fact vector of integers
        >>> S.characteristic('Ss')               # fact a string
        (array([ 0.04963112]), array([[  2.63624782e-06]]), ['Ss'])
        >>> S.characteristic(['Hm0','Tm02'])   # fact a list of strings
        (array([ 4.99833578,  8.03139757]), array([[ 0.05292989,  0.02511371],
               [ 0.02511371,  0.0274645 ]]), ['Hm0', 'Tm02'])
        See also
        ---------
        bandwidth,
        moment
        References
        ----------
        Krogstad, H.E., Wolf, J., Thompson, S.P., and Wyatt, L.R. (1999)
        'Methods for intercomparison of wave measurements'
        Coastal Enginering, Vol. 37, pp. 235--257
        Krogstad, H.E. (1982)
        'On the covariance of the periodogram'
        Journal of time series analysis, Vol. 3, No. 3, pp. 195--207
        Tucker, M.J. (1993)
        'Recommended standard for wave data sampling and near-real-time
        processing'
        Ocean Engineering, Vol.20, No.5, pp. 459--474
        Young, I.R. (1999)
        "Wind generated ocean waves"
        Elsevier Ocean Engineering Book Series, Vol. 2, pp 239
        """

        # TODO: Need more checking on computing the variances for Tm24,alpha,
        #       eps2 and eps4
        # TODO: Covariances between Tm24,alpha, eps2 and eps4 variables are
        #        also needed

        tfact = dict(Hm0=0, Tm01=1, Tm02=2, Tm24=3, Tm_10=4, Tp=5, Ss=6, Sp=7,
                     Ka=8, Rs=9, Tp1=10, Alpha=11, Eps2=12, Eps4=13, Qp=14)
        tfact1 = ('Hm0', 'Tm01', 'Tm02', 'Tm24', 'Tm_10', 'Tp', 'Ss', 'Sp',
                  'Ka', 'Rs', 'Tp1', 'Alpha', 'Eps2', 'Eps4', 'Qp')

        if isinstance(fact, str):
            fact = list((fact,))
        if isinstance(fact, (list, tuple)):
            nfact = []
            for k in fact:
                if isinstance(k, str):
                    nfact.append(tfact.get(k.capitalize(), 15))
                else:
                    nfact.append(k)
        else:
            nfact = fact

        nfact = atleast_1d(nfact)

        if np.any((nfact > 14) | (nfact < 0)):
            raise ValueError('Factor outside range (0,...,14)')

        # vari = self.freqtype
        f = self.args.ravel()
        S1 = self.data.ravel()
        m, unused_mtxt = self.moment(nr=4, even=False)

        # moments corresponding to freq  in Hz
        for k in range(1, 5):
            m[k] = m[k] / (2 * pi) ** k

        # pi = np.pi
        ind = flatnonzero(f > 0)
        m.append(simps(S1[ind] / f[ind], f[ind]) * 2. * pi)  # = m_1
        m_10 = simps(S1[ind] ** 2 / f[ind], f[ind]) * \
            (2 * pi) ** 2 / T  # = COV(m_1,m0|T=t0)
        m_11 = simps(S1[ind] ** 2. / f[ind] ** 2, f[ind]) * \
            (2 * pi) ** 3 / T  # = COV(m_1,m_1|T=t0)

        # sqrt = np.sqrt
        #     Hm0        Tm01        Tm02             Tm24         Tm_10
        Hm0 = 4. * sqrt(m[0])
        Tm01 = m[0] / m[1]
        Tm02 = sqrt(m[0] / m[2])
        Tm24 = sqrt(m[2] / m[4])
        Tm_10 = m[5] / m[0]

        Tm12 = m[1] / m[2]

        ind = S1.argmax()
        maxS = S1[ind]
        # [maxS ind] = max(S1)
        Tp = 2. * pi / f[ind]  # peak period /length
        Ss = 2. * pi * Hm0 / g / Tm02 ** 2  # Significant wave steepness
        Sp = 2. * pi * Hm0 / g / Tp ** 2  # Average wave steepness
        # groupiness factor
        Ka = abs(simps(S1 * exp(1J * f * Tm02), f)) / m[0]

        # Quality control parameter
        # critical value is approximately 0.02 for surface displacement records
        # If Rs>0.02 then there are something wrong with the lower frequency
        # part of S.
        Rs = np.sum(
            interp(r_[0.0146, 0.0195, 0.0244] * 2 * pi, f, S1)) / 3. / maxS
        Tp2 = 2 * pi * simps(S1 ** 4, f) / simps(f * S1 ** 4, f)

        alpha1 = Tm24 / Tm02  # m(3)/sqrt(m(1)*m(5))
        eps2 = sqrt(Tm01 / Tm12 - 1.)  # sqrt(m(1)*m(3)/m(2)^2-1)
        eps4 = sqrt(1. - alpha1 ** 2)  # sqrt(1-m(3)^2/m(1)/m(5))
        Qp = 2. / m[0] ** 2 * simps(f * S1 ** 2, f)

        ch = r_[Hm0, Tm01, Tm02, Tm24, Tm_10, Tp, Ss,
                Sp, Ka, Rs, Tp2, alpha1, eps2, eps4, Qp]

        # Select the appropriate values
        ch = ch[nfact]
        chtxt = [tfact1[i] for i in nfact]

        # if nargout>1,
        # covariance between the moments:
        # COV(mi,mj |T=t0) = int f^(i+j)*S(f)^2 df/T
        mij = self.moment(nr=8, even=False, j=1)[0]
        for ix, tmp in enumerate(mij):
            mij[ix] = tmp / T / ((2. * pi) ** (ix - 1.0))

        #  and the corresponding variances for
        # {'hm0', 'tm01', 'tm02', 'tm24', 'tm_10','tp','ss', 'sp', 'ka', 'rs',
        #  'tp1','alpha','eps2','eps4','qp'}
        R = r_[4 * mij[0] / m[0],
               mij[0] / m[1] ** 2. - 2. * m[0] * mij[1] /
               m[1] ** 3. + m[0] ** 2. * mij[2] / m[1] ** 4.,
               0.25 * (mij[0] / (m[0] * m[2]) - 2. * mij[2] / m[2] ** 2 +
                       m[0] * mij[4] / m[2] ** 3),
               0.25 * (mij[4] / (m[2] * m[4]) - 2 * mij[6] / m[4] ** 2 +
                       m[2] * mij[8] / m[4] ** 3),
               m_11 / m[0] ** 2 + (m[5] / m[0] ** 2) ** 2 *
               mij[0] - 2 * m[5] / m[0] ** 3 * m_10,
               nan, (8 * pi / g) ** 2 *
               (m[2] ** 2 / (4 * m[0] ** 3) *
                mij[0] + mij[4] / m[0] - m[2] / m[0] ** 2 * mij[2]),
               nan * ones(4),
               m[2] ** 2 * mij[0] / (4 * m[0] ** 3 * m[4]) + mij[4] /
               (m[0] * m[4]) + mij[8] * m[2] ** 2 / (4 * m[0] * m[4] ** 3) -
               m[2] * mij[2] / (m[0] ** 2 * m[4]) + m[2] ** 2 * mij[4] /
               (2 * m[0] ** 2 * m[4] ** 2) - m[2] * mij[6] / m[0] / m[4] ** 2,
               (m[2] ** 2 * mij[0] / 4 + (m[0] * m[2] / m[1]) ** 2 * mij[2] +
                m[0] ** 2 * mij[4] / 4 - m[2] ** 2 * m[0] * mij[1] / m[1] +
                m[0] * m[2] * mij[2] / 2 - m[0] ** 2 * m[2] / m[1] * mij[3]) /
               eps2 ** 2 / m[1] ** 4,
               (m[2] ** 2 * mij[0] / (4 * m[0] ** 2) + mij[4] + m[2] ** 2 *
                mij[8] / (4 * m[4] ** 2) - m[2] * mij[2] / m[0] + m[2] ** 2 *
                mij[4] / (2 * m[0] * m[4]) - m[2] * mij[6] / m[4]) *
               m[2] ** 2 / (m[0] * m[4] * eps4) ** 2,
               nan]

        # and covariances by a taylor expansion technique:
        # Cov(Hm0,Tm01) Cov(Hm0,Tm02) Cov(Tm01,Tm02)
        S0 = r_[2. / (sqrt(m[0]) * m[1]) * (mij[0] - m[0] * mij[1] / m[1]),
                1. / sqrt(m[2]) * (mij[0] / m[0] - mij[2] / m[2]),
                1. / (2 * m[1]) * sqrt(m[0] / m[2])
                * (mij[0] / m[0] - mij[2] / m[2] - mij[1] / m[1] + m[0] * mij[3] / (m[1] * m[2]))]

        R1 = ones((15, 15))
        R1[:, :] = nan
        for ix, Ri in enumerate(R):
            R1[ix, ix] = Ri

        R1[0, 2:4] = S0[:2]
        R1[1, 2] = S0[2]
        # make lower triangular equal to upper triangular part
        for ix in [0, 1]:
            R1[ix + 1:, ix] = R1[ix, ix + 1:]

        R = R[nfact]
        R1 = R1[nfact, :][:, nfact]

        # Needs further checking:
        # Var(Tm24)= 0.25*(mij[4]/(m[2]*m[4])-
        #                    2*mij[6]/m[4]**2+m[2]*mij[8]/m[4]**3)
        return ch, R1, chtxt

    def setlabels(self):
        ''' Set automatic title, x-,y- and z- labels on SPECDATA object
            based on type, angletype, freqtype
        '''

        N = len(self.type)
        if N == 0:
            raise ValueError(
                'Object does not appear to be initialized, it is empty!')

        labels = ['', '', '']
        if self.type.endswith('dir'):
            title = 'Directional Spectrum'
            if self.freqtype.startswith('w'):
                labels[0] = 'Frequency [rad/s]'
                labels[2] = r'S($\omega$,$\theta$) $[m^2 s / rad^2]$'
            else:
                labels[0] = 'Frequency [Hz]'
                labels[2] = r'S(f,$\theta$) $[m^2 s / rad]$'

            if self.angletype.startswith('r'):
                labels[1] = 'Wave directions [rad]'
            elif self.angletype.startswith('d'):
                labels[1] = 'Wave directions [deg]'
        elif self.type.endswith('freq'):
            title = 'Spectral density'
            if self.freqtype.startswith('w'):
                labels[0] = 'Frequency [rad/s]'
                labels[1] = r'S($\omega$) $[m^2 s/ rad]$'
            else:
                labels[0] = 'Frequency [Hz]'
                labels[1] = r'S(f) $[m^2 s]$'
        else:
            title = 'Wave Number Spectrum'
            labels[0] = 'Wave number [rad/m]'
            if self.type.endswith('k1d'):
                labels[1] = r'S(k) $[m^3/ rad]$'
            elif self.type.endswith('k2d'):
                labels[1] = labels[0]
                labels[2] = r'S(k1,k2) $[m^4/ rad^2]$'
            else:
                raise ValueError(
                    'Object does not appear to be initialized, it is empty!')
        if self.norm != 0:
            title = 'Normalized ' + title
            labels[0] = 'Normalized ' + labels[0].split('[')[0]
            if not self.type.endswith('dir'):
                labels[1] = labels[1].split('[')[0]
            labels[2] = labels[2].split('[')[0]

        self.labels.title = title
        self.labels.xlab = labels[0]
        self.labels.ylab = labels[1]
        self.labels.zlab = labels[2]


class SpecData2D(PlotData):

    """ Container class for 2D spectrum data objects in WAFO

    Member variables
    ----------------
    data : array_like
    args : vector for 1D, list of vectors for 2D, 3D, ...

    type : string
        spectrum type (default 'freq')
    freqtype : letter
        frequency type (default 'w')
    angletype : string
        angle type of directional spectrum (default 'radians')

    Examples
    --------
    >>> import numpy as np
    >>> import wafo.spectrum.models as sm
    >>> Sj = sm.Jonswap(Hm0=3, Tp=7)
    >>> w = np.linspace(0,4,256)
    >>> S = SpecData1D(Sj(w),w) #Make spectrum object from numerical values

    See also
    --------
    PlotData
    CovData
    """

    def __init__(self, *args, **kwds):

        super(SpecData2D, self).__init__(*args, **kwds)

        self.name = 'WAFO Spectrum Object'
        self.type = 'dir'
        self.freqtype = 'w'
        self.angletype = ''
        self.h = inf
        self.tr = None
        self.phi = 0.
        self.v = 0.
        self.norm = 0
        somekeys = ['angletype', 'phi', 'name', 'h',
                    'tr', 'freqtype', 'v', 'type', 'norm']

        self.__dict__.update(sub_dict_select(kwds, somekeys))

        if self.type.endswith('dir') and self.angletype == '':
            self.angletype = 'radians'

        self.setlabels()

    def toacf(self):
        pass

    def tospecdata(self, type=None):  # @ReservedAssignment
        pass

    def sim(self):
        pass

    def sim_nl(self):
        pass

    def rotate(self, phi=0, rotateGrid=False, method='linear'):
        '''
        Rotate spectrum clockwise around the origin.

        Parameters
        ----------
        phi : real scalar
            rotation angle (default 0)
        rotateGrid : bool
            True if rotate grid of Snew physically (thus Snew.phi=0).
            False if rotate so that only Snew.phi is changed
                        (the grid is not physically rotated)  (default)
        method : string
            interpolation method to use when ROTATEGRID==1, (default 'linear')

        Rotates the spectrum clockwise around the origin.
        This equals a anti-clockwise rotation of the cordinate system (x,y).
        The spectrum can be of any of the two-dimensional types.
        For spectrum in polar representation:
            newtheta = theta-phi, but circulant such that -pi<newtheta<pi
        For spectrum in Cartesian representation:
            If the grid is rotated physically, the size of it is preserved
            (maybe it must be increased such that no nonzero points are
           affected, but this is not implemented yet: i.e. corners are cut off)
        The spectrum is assumed to be zero outside original grid.
        NB! The routine does not change the type of spectrum, use spec2spec
            for this.

        Examples
        --------
          S=demospec('dir');
          plotspec(S), hold on
          plotspec(rotspec(S,pi/2),'r'), hold off

        See also
        --------
        spec2spec
        '''
        # TODO: Make physical grid rotation of cartesian coordinates more
        # robust.

        # Snew=S;

        self.phi = mod(self.phi + phi + pi, 2 * pi) - pi
        stype = self.type.lower()[-3::]
        if stype == 'dir':
            # any of the directinal types
            # Make sure theta is from -pi to pi
            theta = self.args[0]
            phi0 = theta[0] + pi
            self.args[0] = theta - phi0

            # make sure -pi<phi<pi
            self.phi = mod(self.phi + phi0 + pi, 2 * pi) - pi
            if (rotateGrid and (self.phi != 0)):
                # Do a physical rotation of spectrum
                # theta = Snew.args[0]
                ntOld = len(theta)
                if (mod(theta[0] - theta[-1], 2 * pi) == 0):
                    nt = ntOld - 1
                else:
                    nt = ntOld

                theta[0:nt] = mod(theta[0:nt] - self.phi + pi, 2 * pi) - pi
                self.phi = 0
                ind = theta.argsort()
                self.data = self.data[ind, :]
                self.args[0] = theta[ind]
                if (nt < ntOld):
                    if (self.args[0][0] == -pi):
                        self.data[ntOld, :] = self.data[0, :]
                    else:
                        # ftype = self.freqtype
                        freq = self.args[1]
                        theta = linspace(-pi, pi, ntOld)
                        # [F, T] = meshgrid(freq, theta)

                        dtheta = self.theta[1] - self.theta[0]
                        self.theta[nt] = self.theta[nt - 1] + dtheta
                        self.data[nt, :] = self.data[0, :]
                        self.data = interp2d(freq,
                                             np.vstack([self.theta[0] - dtheta,
                                                        self.theta]),
                                             np.vstack([self.data[nt, :],
                                                        self.data]),
                                             kind=method)(freq, theta)
                        self.args[0] = theta

        elif stype == 'k2d':
            # any of the 2D wave number types
            # Snew.phi   = mod(Snew.phi+phi+pi,2*pi)-pi
            if (rotateGrid and (self.phi != 0)):
                # Do a physical rotation of spectrum

                [k, k2] = meshgrid(*self.args)
                [th, r] = cart2polar(k, k2)
                [k, k2] = polar2cart(th + self.phi, r)
                ki1, ki2 = self.args
                Sn = interp2d(ki1, ki2, self.data, kind=method)(k, k2)
                self.data = np.where(np.isnan(Sn), 0, Sn)
                self.phi = 0
        else:
            raise ValueError('Can only rotate two dimensional spectra')
        return

    def moment(self, nr=2, vari='xt'):
        '''
        Calculates spectral moments from spectrum

        Parameters
        ----------
        nr   : int
            order of moments (maximum 4)
        vari : string
            variables in model, optional when two-dim.spectrum,
                   string with 'x' and/or 'y' and/or 't'
        Returns
        -------
        m     : list of moments
        mtext : list of strings describing the elements of m, see below

        Details
        -------
        Calculates spectral moments of up to order four by use of
        Simpson-integration.

           //
        m_jkl=|| k1^j*k2^k*w^l S(w,th) dw dth
           //

        where k1=w^2/gravity*cos(th-phi),  k2=w^2/gravity*sin(th-phi)
        and phi is the angle of the rotation in S.phi. If the spectrum
        has field .g, gravity is replaced by S.g.

        The strings in output mtext have the same position in the cell array
        as the corresponding numerical value has in output m
        Notation in mtext: 'm0' is the variance,
                        'mx' is the first-order moment in x,
                       'mxx' is the second-order moment in x,
                       'mxt' is the second-order cross moment between x and t,
                     'myyyy' is the fourth-order moment in y
                             etc.
        For the calculation of moments see Baxevani et al.

        Examples
        --------
        >>> import wafo.spectrum.models as sm
        >>> D = sm.Spreading()
        >>> SD = D.tospecdata2d(sm.Jonswap().tospecdata(),nt=101)
        >>> m,mtext = SD.moment(nr=2,vari='xyt')
        >>> np.allclose(np.round(m,3),
        ... [ 3.061,  0.132, -0.   ,  2.13 ,  0.011,  0.008,  1.677, -0.,
        ...     0.109,  0.109])
        True
        >>> mtext == ['m0', 'mx', 'my', 'mt', 'mxx', 'myy', 'mtt', 'mxy',
        ...            'mxt', 'myt']
        True

        References
        ----------
        Baxevani A. et al. (2001)
        Velocities for Random Surfaces
        '''

        two_dim_spectra = ['dir', 'encdir', 'k2d']
        if self.type not in two_dim_spectra:
            raise ValueError('Unknown 2D spectrum type!')

        if vari is None and nr <= 1:
            vari = 'x'
        elif vari is None:
            vari = 'xt'
        else:  # secure the mutual order ('xyt')
            vari = ''.join(sorted(vari.lower()))
            Nv = len(vari)

            if vari[0] == 't' and Nv > 1:
                vari = vari[1::] + vari[0]

        Nv = len(vari)

        if not self.type.endswith('dir'):
            S1 = self.tospecdata(self.type[:-2] + 'dir')
        else:
            S1 = self
        w = ravel(S1.args[0])
        theta = S1.args[1] - S1.phi
        S = S1.data
        Sw = simps(S, x=theta, axis=0)
        m = [simps(Sw, x=w)]
        mtext = ['m0']

        if nr > 0:
            vec = []
            g = np.atleast_1d(S1.__dict__.get('g', _gravity()))
            # maybe different normalization in x and y => diff. g
            kx = w ** 2 / g[0]
            ky = w ** 2 / g[-1]

            # nw = w.size

            if 'x' in vari:
                ct = np.cos(theta[:, None])
                Sc = simps(S * ct, x=theta, axis=0)
                vec.append(kx * Sc)
                mtext.append('mx')
            if 'y' in vari:
                st = np.sin(theta[:, None])
                Ss = simps(S * st, x=theta, axis=0)
                vec.append(ky * Ss)
                mtext.append('my')
            if 't' in vari:
                vec.append(w * Sw)
                mtext.append('mt')

            if nr > 1:
                if 'x' in vari:
                    Sc2 = simps(S * ct ** 2, x=theta, axis=0)
                    vec.append(kx ** 2 * Sc2)
                    mtext.append('mxx')
                if 'y' in vari:
                    Ss2 = simps(S * st ** 2, x=theta, axis=0)
                    vec.append(ky ** 2 * Ss2)
                    mtext.append('myy')
                if 't' in vari:
                    vec.append(w ** 2 * Sw)
                    mtext.append('mtt')
                if 'x' in vari and 'y' in vari:
                    Scs = simps(S * ct * st, x=theta, axis=0)
                    vec.append(kx * ky * Scs)
                    mtext.append('mxy')
                if 'x' in vari and 't' in vari:
                    vec.append(kx * w * Sc)
                    mtext.append('mxt')
                if 'y' in vari and 't' in vari:
                    vec.append(ky * w * Sc)
                    mtext.append('myt')

                if nr > 3:
                    if 'x' in vari:
                        Sc3 = simps(S * ct ** 3, x=theta, axis=0)
                        Sc4 = simps(S * ct ** 4, x=theta, axis=0)
                        vec.append(kx ** 4 * Sc4)
                        mtext.append('mxxxx')
                    if 'y' in vari:
                        Ss3 = simps(S * st ** 3, x=theta, axis=0)
                        Ss4 = simps(S * st ** 4, x=theta, axis=0)
                        vec.append(ky ** 4 * Ss4)
                        mtext.append('myyyy')
                    if 't' in vari:
                        vec.append(w ** 4 * Sw)
                        mtext.append('mtttt')

                    if 'x' in vari and 'y' in vari:
                        Sc2s = simps(S * ct ** 2 * st, x=theta, axis=0)
                        Sc3s = simps(S * ct ** 3 * st, x=theta, axis=0)
                        Scs2 = simps(S * ct * st ** 2, x=theta, axis=0)
                        Scs3 = simps(S * ct * st ** 3, x=theta, axis=0)
                        Sc2s2 = simps(S * ct ** 2 * st ** 2, x=theta, axis=0)
                        vec.extend((kx ** 3 * ky * Sc3s,
                                    kx ** 2 * ky ** 2 * Sc2s2,
                                    kx * ky ** 3 * Scs3))
                        mtext.extend(('mxxxy', 'mxxyy', 'mxyyy'))
                    if 'x' in vari and 't' in vari:
                        vec.extend((kx ** 3 * w * Sc3,
                                    kx ** 2 * w ** 2 * Sc2, kx * w ** 3 * Sc))
                        mtext.extend(('mxxxt', 'mxxtt', 'mxttt'))
                    if 'y' in vari and 't' in vari:
                        vec.extend((ky ** 3 * w * Ss3, ky ** 2 * w ** 2 * Ss2,
                                    ky * w ** 3 * Ss))
                        mtext.extend(('myyyt', 'myytt', 'myttt'))
                    if 'x' in vari and 'y' in vari and 't' in vari:
                        vec.extend((kx ** 2 * ky * w * Sc2s,
                                    kx * ky ** 2 * w * Scs2,
                                    kx * ky * w ** 2 * Scs))
                        mtext.extend(('mxxyt', 'mxyyt', 'mxytt'))
            # end % if nr>1
            m.extend([simps(vals, x=w) for vals in vec])
        return np.asarray(m), mtext

    def interp(self):
        pass

    def normalize(self):
        pass

    def bandwidth(self):
        pass

    def setlabels(self):
        ''' Set automatic title, x-,y- and z- labels on SPECDATA object

            based on type, angletype, freqtype
        '''

        N = len(self.type)
        if N == 0:
            raise ValueError(
                'Object does not appear to be initialized, it is empty!')

        labels = ['', '', '']
        if self.type.endswith('dir'):
            title = 'Directional Spectrum'
            if self.freqtype.startswith('w'):
                labels[0] = 'Frequency [rad/s]'
                labels[2] = r'$S(w,\theta) [m**2 s / rad**2]$'
            else:
                labels[0] = 'Frequency [Hz]'
                labels[2] = r'$S(f,\theta) [m**2 s / rad]$'

            if self.angletype.startswith('r'):
                labels[1] = 'Wave directions [rad]'
            elif self.angletype.startswith('d'):
                labels[1] = 'Wave directions [deg]'
        elif self.type.endswith('freq'):
            title = 'Spectral density'
            if self.freqtype.startswith('w'):
                labels[0] = 'Frequency [rad/s]'
                labels[1] = 'S(w) [m**2 s/ rad]'
            else:
                labels[0] = 'Frequency [Hz]'
                labels[1] = 'S(f) [m**2 s]'
        else:
            title = 'Wave Number Spectrum'
            labels[0] = 'Wave number [rad/m]'
            if self.type.endswith('k1d'):
                labels[1] = 'S(k) [m**3/ rad]'
            elif self.type.endswith('k2d'):
                labels[1] = labels[0]
                labels[2] = 'S(k1,k2) [m**4/ rad**2]'
            else:
                raise ValueError(
                    'Object does not appear to be initialized, it is empty!')
        if self.norm != 0:
            title = 'Normalized ' + title
            labels[0] = 'Normalized ' + labels[0].split('[')[0]
            if not self.type.endswith('dir'):
                labels[1] = labels[1].split('[')[0]
            labels[2] = labels[2].split('[')[0]

        self.labels.title = title
        self.labels.xlab = labels[0]
        self.labels.ylab = labels[1]
        self.labels.zlab = labels[2]

if __name__ == '__main__':
    pass
