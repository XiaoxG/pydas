"""
Wave Models and Dispersion Relations
------------------------------------

Dispersion relation
------------------
k2w - Translates from wave number to frequency
w2k - Translates from frequency to wave number

Model spectra
-------------
Jonswap          - JONSWAP spectral density
Torsethaugen     - Torsethaugen double peaked (swell + wind) spectrum model
"""

import warnings
import numpy as np
from numpy import (inf, atleast_1d, minimum, exp, log, sqrt, where, pi, ones_like, zeros_like, flatnonzero, tanh, cosh, 
                  sin, cos, arctan2, sign, finfo)
import scipy.special as sp
import scipy.integrate as integrate
from scipy.interpolate import interp1d
import scipy.optimize as optimize

# Constants
_EPS = finfo(float).eps

# =============================================================================
# Dispersion Relations
# =============================================================================

def lazywhere(cond, arrays, f, fillvalue=None, f2=None):
    """
    np.where(cond, x, fillvalue) always evaluates x even where cond is False.
    This one only evaluates f(arr1[cond], arr2[cond], ...).
    """
    if fillvalue is None:
        _assert(f2 is not None, "One of (fillvalue, f2) must be given.")
        fillvalue = np.nan
    else:
        _assert(f2 is None, "Only one of (fillvalue, f2) can be given.")

    arrays = np.broadcast_arrays(*arrays)
    temp = tuple(np.extract(cond, arr) for arr in arrays)
    out = np.full(np.shape(arrays[0]), fill_value=fillvalue)
    np.place(out, cond, f(*temp))
    if f2 is not None:
        temp = tuple(np.extract(~cond, arr) for arr in arrays)
        np.place(out, ~cond, f2(*temp))

    return out

def _assert(cond, msg):
    if not cond:
        raise ValueError(msg)

def _assert_warn(cond, msg):
    if not cond:
        warnings.warn(msg)

def k2w(k1, k2=0e0, h=inf, g=9.81, u1=0e0, u2=0e0):
    """Translates from wave number to frequency using the dispersion relation
    
    Parameters
    ----------
    k1 : array-like
        wave numbers [rad/m].
    k2 : array-like, optional
        second dimension wave number
    h : real scalar, optional
        water depth [m].
    g : real scalar, optional
        acceleration of gravity, see gravity
    u1, u2 : real scalars, optional
        current velocity [m/s] along dimension 1 and 2.
        note: when u1!=0 | u2!=0 then theta is not calculated correctly
        
    Returns
    -------
    w : ndarray
        angular frequency [rad/s].
    theta : ndarray
        direction [rad].
        
    Dispersion relation
    -------------------
        w     = sqrt(g*K*tanh(K*h))   (  0 <   w   < inf)
        theta = arctan2(k2,k1)        (-pi < theta <  pi)
    where
        K = sqrt(k1**2+k2**2)
        
    The shape of w and theta is the common shape of k1 and k2 according to the
    numpy broadcasting rules.
    """
    k1i, k2i, hi, gi, u1i, u2i = np.broadcast_arrays(k1, k2, h, g, u1, u2)

    if np.size(k1i) == 0:
        return zeros_like(k1i)
    ku1 = k1i * u1i
    ku2 = k2i * u2i

    theta = arctan2(k2, k1)

    k = sqrt(k1i ** 2 + k2i ** 2)
    w = where(k > 0, ku1 + ku2 + sqrt(gi * k * tanh(k * hi)), 0.0)

    cond = (0 <= w)
    _assert_warn(np.all(cond), """
        Waves and current are in opposite directions
        making some of the frequencies negative.
        Here we are forcing the negative frequencies to zero.
        """)

    w = where(cond, w, 0.0)  # force w to zero
    return w, theta


def w2k(w, theta=0.0, h=inf, g=9.81, count_limit=100, rtol=1e-7, atol=1e-14):
    """
    Translates from frequency to wave number using the dispersion relation
    
    Parameters
    ----------
    w : array-like
        angular frequency [rad/s].
    theta : array-like, optional
        direction [rad].
    h : real scalar, optional
        water depth [m].
    g : real scalar or array-like of size 2.
        constant of gravity [m/s**2] or 3D normalizing constant
        
    Returns
    -------
    k1, k2 : ndarray
        wave numbers [rad/m] along dimension 1 and 2.
        
    Description
    -----------
    Uses Newton Raphson method to find the wave number k in the dispersion
    relation
        w**2= g*k*tanh(k*h).
    The solution k(w) => k1 = k(w)*cos(theta)
                         k2 = k(w)*sin(theta)
    """
    gi = atleast_1d(g)
    wi, th, hi = np.broadcast_arrays(w, theta, h)
    if wi.size == 0:
        return zeros_like(wi)

    k = 1.0 * sign(wi) * wi ** 2.0 / gi[0]  # deep water
    if (hi > 1e25).all():
        k2 = k * sin(th) * gi[0] / gi[-1]  # size np x nf
        k1 = k * cos(th)
        return k1, k2
    _assert(gi.size == 1, 'Finite depth in combination with 3D normalization'
            ' (len(g)=2) is not implemented yet.')

    oshape = k.shape
    wi, k, hi = wi.ravel(), k.ravel(), hi.ravel()

    # Newton's Method
    # Permit no more than count_limit iterations.
    hi = hi * ones_like(k)
    hn = zeros_like(k)
    ix = flatnonzero(((wi < 0) | (0 < wi)) & (hi < 1e25))

    # Break out of the iteration loop for three reasons:
    #  1) the last update is very small (compared to k*rtol)
    #  2) the last update is very small (compared to atol)
    #  3) There are more than 100 iterations. This should NEVER happen.
    count = 0
    while (ix.size > 0 and count < count_limit):
        ki = k[ix]
        kh = ki * hi[ix]
        coshkh2 = lazywhere(np.abs(kh) < 350, (kh, ),
                            lambda kh: cosh(kh) ** 2.0, fillvalue=np.inf)
        hn[ix] = 0.5 * (tanh(kh) + kh * (1.0 - tanh(kh) ** 2.0))
        f = wi[ix] ** 2.0 - gi[0] * ki * tanh(kh)
        dfdk = -gi[0] * (tanh(kh) + ki * hi[ix] / coshkh2)
        dk = f / dfdk

        knew = ki - dk
        # Accept the new k if it acquires an imaginary part
        # or if it is closer to the answer.
        # Handle the delicate case where the answer is close to 0.
        # Avoid the case where we step from positive to negative k in the presence of
        # a sufficiently large h, as this will converge to the negative root which we don't want.
        cond1 = (np.abs(dk) < rtol * np.maximum(atol / rtol, np.abs(ki)))
        cond2 = (np.abs(dk) < atol)
        cond3 = ((knew > 0) | (hi[ix] < 1))
        cond = cond1 | cond2
        k[ix] = np.where(cond, ki, np.where(cond3, knew, ki / 2.0))
        ix = ix[~cond]
        count += 1

    k = k.reshape(oshape)
    k1 = k * cos(th)
    k2 = k * sin(th)
    return k1, k2

# =============================================================================
# Wave Spectrum Models
# =============================================================================

def sech(x):
    """
    Hyperbolic secant function.
    
    Parameters
    ----------
    x : array-like
        Input argument
        
    Returns
    -------
    y : array-like
        Hyperbolic secant of x
    """
    return 1.0 / cosh(x)

def _gengamspec(wn, N=5, M=4):
    """Return Generalized gamma spectrum in dimensionless form
    
    Parameters
    ----------
    wn : arraylike
        normalized frequencies, w/wp.
    N  : scalar
        defining the decay of the high frequency part.
    M  : scalar
        defining the spectral width around the peak.
        
    Returns
    -------
    S   : arraylike
        spectral values, same size as wn.
        
    Note that N = 5, M = 4 corresponds to a normalized Bretschneider spectrum.
    """
    w = atleast_1d(wn)
    S = zeros_like(w)

    k = flatnonzero(w > 0.0)
    if k.size > 0:
        B = N / M
        C = (N - 1.0) / M

        # A = Normalizing factor related to Bretschneider form
        # A = B**C*M/gamma(C)
        # S[k] = A*wn[k]**(-N)*exp(-B*wn[k]**(-M))
        logwn = log(w.take(k))
        logA = (C * log(B) + log(M) - sp.gammaln(C))  #pylint: disable=no-member
        S.put(k, exp(logA - N * logwn - B * exp(-M * logwn)))
    return S

def jonswap_peakfact(Hm0, Tp):
    """Jonswap peakedness factor, gamma, given Hm0 and Tp
    
    Parameters
    ----------
    Hm0 : significant wave height [m].
    Tp  : peak period [s]
    
    Returns
    -------
    gamma : Peakedness parameter of the JONSWAP spectrum
    
    Details
    -------
    A standard value for GAMMA is 3.3. However, a more correct approach is
    to relate GAMMA to Hm0 and Tp:
         D = 0.036-0.0056*Tp/sqrt(Hm0)
        gamma = exp(3.484*(1-0.1975*D*Tp**4/(Hm0**2)))
    This parameterization is based on qualitative considerations of deep water
    wave data from the North Sea, see Torsethaugen et. al. (1984)
    Here GAMMA is limited to 1..7.
    """
    Hm0, Tp = atleast_1d(Hm0, Tp)

    x = Tp / sqrt(Hm0)

    gam = ones_like(x)

    k1 = flatnonzero(x <= 5.14285714285714)
    if k1.size > 0:  # limiting gamma to [1 7]
        xk = x.take(k1)
        D = 0.036 - 0.0056 * xk  # approx 5.061*Hm0**2/Tp**4*(1-0.287*log(gam))
        # gamma
        gam.put(k1, minimum(exp(3.484 * (1.0 - 0.1975 * D * xk ** 4.0)), 7.0))

    return gam

class ModelSpectrum(object):
    """Base class for wave spectrum models"""
    type = 'ModelSpectrum'

    def __init__(self, Hm0=7.0, Tp=11.0, **kwds):  # @UnusedVariable
        self.Hm0 = Hm0
        self.Tp = Tp

    def tospecdata(self, w=None, wc=None, nw=257):
        """
        Return SpecData1D object from ModelSpectrum

        Parameter
        ---------
        w : arraylike
            vector of angular frequencies used in discretization of spectrum
        wc : scalar
            cut off frequency (default 33/Tp)
        nw : int
            number of frequencies

        Returns
        -------
        S : SpecData1D object
            member attributes of model spectrum are copied to S.workspace
        """
        # 延迟导入SpecData1D，避免循环导入
        from waveModel.specdata import SpecData1D

        if w is None:
            if wc is None:
                wc = 33. / self.Tp
            w = np.linspace(0, wc, nw)
        S = SpecData1D(self.__call__(w), w)
        try:
            S.h = self.h
        except AttributeError:
            pass
        S.labels.title = self.type + ' ' + S.labels.title
        S.workspace = self.__dict__.copy()
        return S

    def chk_seastate(self):
        """Check if seastate is OK for the wave model."""
        pass

    def _chk_extra_param(self):
        pass

    def __call__(self, w):
        """Return spectrum value for a given frequency"""
        pass


class Jonswap(ModelSpectrum):
    """
    Jonswap spectral density model
    
    Member variables
    ----------------
    Hm0    : significant wave height (default 7 (m))
    Tp     : peak period             (default 11 (sec))
    gamma  : peakedness factor determines the concentraton
            of the spectrum on the peak frequency.
            Usually in the range  1 <= gamma <= 7.
            default depending on Hm0, Tp, see jonswap_peakedness)
    sigmaA : spectral width parameter for w<wp (default 0.07)
    sigmaB : spectral width parameter for w<wp (default 0.09)
    Ag     : normalization factor used when gamma>1:
    N      : scalar defining decay of high frequency part. (default 5)
    M      : scalar defining spectral width around the peak. (default 4)
    method : String defining method used to estimate Ag when gamma>1
            'integration': Ag = 1/gaussq(Gf*ggamspec(wn,N,M),0,wnc) (default)
            'parametric' : Ag = (1+f1(N,M)*log(gamma)**f2(N,M))/gamma
            'custom'     : Ag = Ag
    wnc    : wc/wp normalized cut off frequency used when calculating Ag
                by integration (default 6)
                
    The JONSWAP spectrum is defined as
        S(w) = A * Gf * G0 * wn**(-N)*exp(-N/(M*wn**M))
    where
        G0  = Normalizing factor related to Bretschneider form
        A   = Ag * (Hm0/4)**2 / wp     (Normalization factor)
        Gf  = j**exp(-.5*((wn-1)/s)**2) (Peak enhancement factor)
        wn  = w/wp
        wp  = angular peak frequency
        s   = sigmaA      for wn <= 1
              sigmaB      for 1  <  wn
        j   = gamma,     (j=1, => Bretschneider spectrum)
    """

    type = 'Jonswap'

    def __init__(self, Hm0=7.0, Tp=11.0, gamma=None, sigmaA=0.07, sigmaB=0.09,
                 Ag=None, N=5, M=4, method='integration', wnc=6.0,
                 chk_seastate=True):
        super(Jonswap, self).__init__(Hm0, Tp)
        self.gamma = gamma
        self.sigmaA = sigmaA
        self.sigmaB = sigmaB
        self.Ag = Ag
        self.N = N
        self.M = M
        self.method = method
        self.wnc = wnc
        self.chk_seastate()
        self._chk_extra_param()

    def chk_seastate(self):
        """Check if seastate is OK for the Jonswap model."""
        Hm0 = atleast_1d(self.Hm0)
        Tp = atleast_1d(self.Tp)
        ones_like(Hm0)

        if hasattr(self, 'gamma') and self.gamma is not None:
            gamma = self.gamma
        else:
            gamma = jonswap_peakfact(Hm0, Tp)

        # (3.6 to 5) * sqrt(Hm0) < Tp for Jonswap -1
        return where((Tp < 3.6 * sqrt(Hm0)) | (5 * sqrt(Hm0) < Tp), gamma, 3.3)

    def _chk_extra_param(self):
        if self.gamma is None:
            # peakedness factor
            self.gamma = jonswap_peakfact(self.Hm0, self.Tp)
            gammai = self.gamma
        elif np.isscalar(self.gamma):
            gammai = ones_like(atleast_1d(self.Hm0)) * self.gamma
        else:
            gammai = self.gamma

        if (self.gamma != 1).any():
            self._pre_calculate_ag()
        elif np.isscalar(self.Ag):
            self.Ag = 1.0

    def _localspec(self, wn):
        return _gengamspec(wn, self.N, self.M)

    def _check_parametric_ag(self, N, M, gammai):
        return (1 + 0.5 * sqrt(0.5) * log(gammai) ** 1.05) / gammai

    def _parametric_ag(self):
        gamma = self.gamma
        gammai = gamma
        N = self.N
        M = self.M
        if np.isscalar(N) and np.isscalar(M):
            if (N, M) == (5, 4):
                # Approximate normalization if (M,N)=(4,5)
                # approx = (1 + 0.5*sqrt(0.5)*log(gamma)**1.05)/gamma
                # If gamma == 1 then Ag = 1
                # _assert(gamma != 1, 'JONSWAP with gamma=1, use Bretschneider spectrum')

                # gte = (gamma != 1).any()
                # if gte:
                if np.isscalar(gamma):
                    self.Ag = (1 + 0.5 * sqrt(0.5) * log(gamma) ** 1.05) / gamma
                    # else:
                    # self.Ag = ones_like(atleast_1d(gamma))
                elif (gamma == 1).all():
                    self.Ag = ones_like(atleast_1d(gamma))
                else:
                    ix = flatnonzero(gamma != 1)
                    tmp = gamma.take(ix)
                    ag = ones_like(atleast_1d(gamma))
                    ag.put(ix, (1 + 0.5 * sqrt(0.5) * log(tmp) ** 1.05) / tmp)
                    self.Ag = ag
            else:
                # More accurate normalization factor
                if np.isscalar(gamma):
                    self.Ag = self._check_parametric_ag(N, M, gammai)
                else:
                    ix = flatnonzero(gamma != 1)
                    tmp = gamma.take(ix)
                    ag = ones_like(atleast_1d(gamma))
                    ag.put(ix, self._check_parametric_ag(N, M, gammai.take(ix)))
                    self.Ag = ag

    def _custom_ag(self):
        """Normalization for Jonswap not needed for custom ag"""
        pass

    def _integrate_ag(self):
        # normalizing by integration
        gamma = self.gamma
        g_int = integrate.quad(
            lambda x: self.peak_e_factor(x) * _gengamspec(x, self.N, self.M),
            0, self.wnc)
        self.Ag = 1.0 / g_int[0]

    def _pre_calculate_ag(self):
        if self.Ag is None:
            method = self.method
            if method.startswith('integration'):  # integration
                self._integrate_ag()
            elif method.startswith('param'):  # parametric
                self._parametric_ag()
            else:  # Custom
                self._custom_ag()

    def peak_e_factor(self, wn):
        """Return peakedness factor for Jonswap normalized frequency"""
        wni = atleast_1d(wn)
        j = self.gamma
        sigmaj = where(wni <= 1, self.sigmaA, self.sigmaB)
        return j ** exp(-0.5 * ((wni - 1) / sigmaj) ** 2)

    def __call__(self, wi):
        """Return Jonswap spectral density for a given frequency"""
        w = atleast_1d(wi).ravel()
        s = np.zeros(w.size)

        # Water depth is assumed deep
        # Generates the modified JONSWAP spectral density
        wp = 2 * pi / self.Tp  # peak frequency [rad/s]
        wn = w / wp

        k2 = flatnonzero(wn > 0)
        if k2.size > 0:
            # compute modified jonswap
            wnk = wn.take(k2)
            spec2 = _gengamspec(wnk, self.N, self.M)

            # apply the peak enhancement
            if self.gamma != 1:
                spec2 = spec2 * self.peak_e_factor(wnk)
                if self.Ag is not None:
                    # Normalize with Ag
                    spec2 = spec2 * self.Ag

            # The JONSWAP (Bretschneider) spectra are normalized
            # to have the specified significant wave height
            # through the normalization below

            # Convert to spectral density
            spec2 = spec2 / wp * (self.Hm0 / 4) ** 2
            s.put(k2, spec2)

        return s


class Torsethaugen(ModelSpectrum):
    """
    Torsethaugen double peaked (swell + wind) spectrum model

    Member variables
    ----------------
    Hm0   : significant wave height (default 7 (m))
    Tp    : peak period (default 11 (sec))
    wnc   : wc/wp normalized cut off frequency used when calculating Ag
            by integration (default 6)
    method : String defining method used to estimate normalization factors, Ag,
             in the the modified JONSWAP spectra when gamma>1
            'integrate' : Ag = 1/quad(Gf.*gengamspec(wn,N,M),0,wnc)
            'parametric': Ag = (1+f1(N,M)*log(gamma)**f2(N,M))/gamma
    
    The double peaked (swell + wind) Torsethaugen spectrum is
    modelled as  S(w) = Ss(w) + Sw(w) where Ss and Sw are modified
    JONSWAP spectrums for swell and wind peak, respectively.
    The energy is divided between the two peaks according
    to empirical parameters, which peak that is primary depends on parameters.
    The empirical parameters are found for classes of Hm0 and Tp,
    originating from a dataset consisting of 20 000 spectra divided
    into 146 different classes of Hm0 and Tp. (Data measured at the
    Statfjord field in the North Sea in a period from 1980 to 1989.)
    The range of the measured  Hm0 and Tp for the dataset
    are from 0.5 to 11 meters and from 3.5 to 19 sec, respectively.
    """

    type = 'Torsethaugen'

    def __init__(self, Hm0=7, Tp=11, method='integration', wnc=6, gravity=9.81,
                 chk_seastate=True, **kwds):
        super(Torsethaugen, self).__init__(Hm0, Tp)
        self.method = method.lower()
        self.wnc = wnc
        self.gravity = gravity
        self._chk_extra_param()
        self._init_spec()

    def __call__(self, w):
        """Return Torsethaugen spectral density for a given frequency"""
        ss = self.spec_wind(w)
        sw = self.spec_swell(w)
        return ss + sw

    def _chk_extra_param(self):
        """Not used in Torsethaugen"""
        pass

    def _init_spec(self):
        """Initialize Torsethaugen spectrum"""
        self.h_w = Jonswap(Hm0=self.Hm0, Tp=self.Tp, gamma=1.0, sigmaA=0.07,
                          sigmaB=0.09, N=4, M=4, wnc=6.0, method='integration')
        self.h_s = Jonswap(Hm0=self.Hm0, Tp=self.Tp, gamma=7.0, sigmaA=0.07,
                          sigmaB=0.09, N=4, M=4, wnc=6.0, method='integration')

    def swell(self, w):
        """Return swell part of the spectrum"""
        return self.spec_swell(w)

    def wind(self, w):
        """Return wind part of the spectrum"""
        return self.spec_wind(w)

    def spec_wind(self, w):
        """Return wind part of the Torsethaugen spectrum"""
        # Not implemented
        return np.zeros_like(w)

    def spec_swell(self, w):
        """Return swell part of the Torsethaugen spectrum"""
        # Not implemented
        return np.zeros_like(w) 