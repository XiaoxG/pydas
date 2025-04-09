"""
Miscellaneous utility functions for the WaveModel package.

This module contains various utility functions for data analysis and processing.
"""

import collections
import fractions
import numbers
import sys
import warnings
from time import strftime, gmtime

import numpy as np
from numpy import (sqrt, arctan2, sin, cos, exp, log, log1p,
                   inf, pi, zeros, ones, meshgrid)
from scipy.special import gammaln, betaln  # pylint: disable=no-name-in-module
from scipy.integrate import trapz, simps
from numba import jit, float64, int64, int32, int8, void

# Constants
FLOATINFO = np.finfo(float)
_TINY = FLOATINFO.tiny
_EPS = FLOATINFO.eps

# Export list - only include functions that are actually used
__all__ = [
    # Core utility functions
    'now', 'moving_average', 'moment', 'findcross', 'findpeaks', 'findrfc',
    'gravity', 'polar2cart', 'cart2polar',
    
    # Less commonly used but important utilities
    'check_random_state', 'piecewise', 'discretize', 'lazywhere', 'lazyselect',
    'nextpow2', 'findextrema', 'findtp', 'findtc'
]


def xor(a, b):
    """Returns True only when inputs differ."""
    return a ^ b


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    If seed is None (or np.random), return the RandomState singleton used
    by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.

    Examples
    --------
    >>> rs0 = check_random_state(seed=None)
    >>> rs1 = check_random_state(seed=1)
    >>> rs2 = check_random_state(seed=np.random.RandomState(1))
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    msg = '{} cannot be used to seed a numpy.random.RandomState instance'
    raise ValueError(msg.format(seed))


def moment(spec, m=0, int_method=trapz):
    """
    Calculate spectral moment of order m for a spectral density.

    Parameters
    ----------
    spec : array-like
        Spectral density values
    m : int
        Order of moment
    int_method : function
        Integration method (default: trapz)

    Returns
    -------
    mom : float
        Spectral moment of order m
    """
    if not hasattr(spec, 'args'):
        f = spec[0]
        S = spec[1]
    else:
        f = spec.args
        S = spec.data
    
    if m != 0:
        S = S * f**m
    
    return int_method(S, f)


def moving_average(x, L, axis=0):
    """
    Calculate the moving average of x.

    Parameters
    ----------
    x : array_like
        Signal
    L : scalar, integer
        Window length
    axis : scalar, integer, optional
        If axis is None, x is treated as a flat array, regardless of
        its shape. Otherwise, averaging is performed along the given axis.

    Returns
    -------
    y : ndarray
        Moving average of x
    
    Notes
    -----
    The moving average is calculated using:
    y = sum(x(i-L+1:i))/L for i>=L

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(11)
    >>> moving_average(x, 3)
    array([1., 2., 3., 4., 5., 6., 7., 8., 9.])
    >>> moving_average(x, 3).shape
    (9,)
    """
    n = len(np.atleast_1d(x))
    if axis is None or n == 1:
        return np.convolve(x, np.ones(L)/float(L), 'valid')
    
    # Use numpy's stride_tricks to make a sliding window view
    from numpy.lib.stride_tricks import as_strided
    
    # Calculate new shape and strides
    new_shape = list(x.shape)
    new_shape[axis] = new_shape[axis] - L + 1
    new_shape.insert(axis+1, L)
    
    new_strides = list(x.strides)
    new_strides.insert(axis+1, new_strides[axis])
    
    # Create strided view and average along window dimension
    strided_x = as_strided(x, shape=new_shape, strides=new_strides)
    return strided_x.mean(axis=axis+1)


# Numba accelerated functions
@jit(int64(int64[:], int8[:]))
def _findcross(ind, y):
    """Returns indices to zero level crossings of y vector

    Notes
    -----
    Same implementation as findcross function found in c_functions.c.
    """
    ix, dcross, start, v = 0, 0, 0, 0
    n = len(y)
    if y[0] < v:
        dcross = -1  # first is a up-crossing
    elif y[0] > v:
        dcross = 1  # first is a down-crossing
    elif y[0] == v:
        # Find out what type of crossing we have next time..
        for i in range(1, n):
            start = i
            if y[i] < v:
                ind[ix] = i - 1  # first crossing is a down crossing
                ix += 1
                dcross = -1  # The next crossing is a up-crossing
                break
            elif y[i] > v:
                ind[ix] = i - 1  # first crossing is a up-crossing
                ix += 1
                dcross = 1  # The next crossing is a down-crossing
                break

    for i in range(start, n - 1):
        if ((dcross == -1 and y[i] <= v and v < y[i + 1])
                or (dcross == 1 and v <= y[i] and y[i + 1] < v)):

            ind[ix] = i
            ix += 1
            dcross = -dcross
    return ix


@jit(int32(float64, float64), nopython=True)
def a_le_b(a, b):
    return a <= b


@jit(int32(float64, float64), nopython=True)
def a_lt_b(a, b):
    return a < b


def _make_findrfc(cmp1, cmp2):
    @jit(int64(int64[:], float64[:], float64), nopython=True)
    def local_findrfc(t, y, h):
        """Returns indices, t, to RFC turningpoints of a vector y of turningpoints

        Notes
        -----
        Same implementation as rfcfilter function found in rfcfilter.m in wafo matlab.
        """
        n = len(y)
        j, t0, z0 = 0, 0, 0
        y0 = y[t0]
        # The rainflow filter
        for ti in range(1, n):
            fpi = y0 + h
            fmi = y0 - h
            yi = y[ti]

            if z0 == 0:
                if cmp1(yi, fmi):
                    z1 = -1
                elif cmp1(fpi, yi):
                    z1 = +1
                else:
                    z1 = 0
                t1, y1 = (t0, y0) if z1 == 0 else (ti, yi)
            else:
                if (((z0 == +1) and cmp1(yi, fmi))
                        or ((z0 == -1) and cmp2(yi, fpi))):
                    z1 = -1
                elif (((z0 == +1) and cmp2(fmi, yi)) or
                        ((z0 == -1) and cmp1(fpi, yi))):
                    z1 = +1
                else:
                    raise ValueError
                #     warnings.warn('Something wrong, i={}'.format(tim1))

                # Update y1
                if z1 != z0:
                    t1, y1 = ti, yi
                elif z1 == -1:
                    t1, y1 = (t0, y0) if y0 < yi else (ti, yi)
                elif z1 == +1:
                    t1, y1 = (t0, y0) if y0 > yi else (ti, yi)

            # Update y if y0 is a turning point
            if abs(z0 - z1) == 2:
                j += 1
                t[j] = t0

            # Update t0, y0, z0
            t0, y0, z0 = t1, y1, z1
        # end

        # Update y if last y0 is greater than (or equal) threshold
        if cmp2(h, abs(y0 - y[t[j]])):
            j += 1
            t[j] = t0
        return j + 1
    return local_findrfc


_findrfc_le = _make_findrfc(a_le_b, a_lt_b)
_findrfc_lt = _make_findrfc(a_lt_b, a_le_b)


@jit(int64(int64[:], float64[:], float64), nopython=True)
def _findrfc(ind, y, h):
    """Returns indices to RFC turningpoints of a vector y of turningpoints

    Notes
    -----
    Same implementation as findrfc function found in c_functions.c.
    """
    n = len(y)
    t_start = 0
    nc = n // 2
    ix = 0
    for i in range(nc):
        Tmi = t_start + 2 * i
        Tpl = t_start + 2 * i + 2
        xminus = y[2 * i]
        xplus = y[2 * i + 2]

        if(i != 0):
            j = i - 1
            while ((j >= 0) and (y[2 * j + 1] <= y[2 * i + 1])):
                if (y[2 * j] < xminus):
                    xminus = y[2 * j]
                    Tmi = t_start + 2 * j
                j -= 1
        if (xminus >= xplus):
            if (y[2 * i + 1] - xminus >= h):
                ind[ix] = Tmi
                ix += 1
                ind[ix] = (t_start + 2 * i + 1)
                ix += 1
            # Skip to next iteration
            continue
            
        j = i + 1
        while (j < nc):
            if (y[2 * j + 1] >= y[2 * i + 1]):
                break  # goto L170
            if((y[2 * j + 2] <= xplus)):
                xplus = y[2 * j + 2]
                Tpl = (t_start + 2 * j + 2)
            j += 1
        
        # Check if we didn't break out of the loop
        if j >= nc:
            if ((y[2 * i + 1] - xminus) >= h):
                ind[ix] = Tmi
                ix += 1
                ind[ix] = (t_start + 2 * i + 1)
                ix += 1
            # Skip to next iteration
            continue
        
        # If we get here, we broke out of the while loop
        # L170:
        if (xplus <= xminus):
            if ((y[2 * i + 1] - xminus) >= h):
                ind[ix] = Tmi
                ix += 1
                ind[ix] = (t_start + 2 * i + 1)
                ix += 1
        elif ((y[2 * i + 1] - xplus) >= h):
            ind[ix] = (t_start + 2 * i + 1)
            ix += 1
            ind[ix] = Tpl
            ix += 1

        # L180:
        # iy=i
    #  /* for i */
    return ix


def findcross(x, v=0.0, kind=None, method='numba'):
    """
    Returns indices to zero-crossings in a vector.

    Parameters
    ----------
    x : array-like
        Vector with sampled values.
    v : scalar, optional
        Level of up-crossing (default 0)
    kind : string, optional
        Type of crossing
        'u' Up-crossing.
        'd' Down-crossing.
        None Finds all crossings (default)
    method : string, optional
        'numba' uses numba compiled functions (default)
        'python' uses pure python functions

    Returns
    -------
    ind : ndarray
        Indices to the crossings.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 7*np.pi, 250)
    >>> y = np.sin(x)
    >>> ind = findcross(y, 0.75)
    >>> np.allclose(ind, [9, 25, 80, 97, 151, 168, 223, 239])
    True
    """
    xn = np.atleast_1d(x).ravel()
    if v is None:
        v = 0.0
        
    # Find relative crossings
    ix = np.zeros(len(xn), dtype=np.int64)
    xn = xn - v
    
    if method.startswith('numba'):
        ind = _findcross(ix, xn.astype(np.int8))
        return ix[:ind]
    else:
        # Use Python implementation
        ind = []
        n = len(xn)
        if n < 2:
            return np.array([])
            
        # Tabulate crossing types
        dcross = np.sign(np.diff(np.sign(xn)))
        
        # Find indices of crossings
        idx, = np.nonzero(dcross)
        
        # Handle different crossing types
        if kind == 'u':  # Keep only up-crossings
            ind = idx[dcross[idx] > 0]
        elif kind == 'd':  # Keep only down-crossings
            ind = idx[dcross[idx] < 0]
        else:  # Keep all crossings
            ind = idx
            
    return ind


def findpeaks(data, n=2, min_h=None, min_p=0.0):
    """
    Find peaks in a vector.

    Parameters
    ----------
    data : vector
        Data vector
    n : scalar, integer
        Minimum distance between peaks
    min_h : real scalar
        Minimum height from trough to peak
    min_p : real scalar
        Minimum prominence

    Returns
    -------
    indices : array-like
        Indices to peaks
    """
    # Initialize
    x = np.atleast_1d(data).ravel()
    
    # Find local maxima
    dx = np.diff(x)
    ind_max, = np.nonzero((dx[:-1] > 0) & (dx[1:] < 0))
    
    # Add end point if it's a peak
    if len(x) > 1 and dx[-1] > 0:
        ind_max = np.append(ind_max, len(x)-1)
    
    # Filter by minimum distance
    if len(ind_max) > 1 and n > 1:
        # Remove peaks that are too close
        too_close = np.diff(ind_max) < n
        if any(too_close):
            # Keep higher peaks
            heights = x[ind_max]
            indices_to_remove = []
            for i in range(len(too_close)):
                if too_close[i]:
                    if heights[i] < heights[i+1]:
                        indices_to_remove.append(i)
                    else:
                        indices_to_remove.append(i+1)
            ind_max = np.delete(ind_max, indices_to_remove)
    
    # Filter by minimum height
    if min_h is not None and len(ind_max) > 0:
        # Calculate prominence of peaks
        heights = x[ind_max]
        # Find preceding/following troughs
        troughs = []
        for i in ind_max:
            # Find preceding trough
            j = i
            while j > 0 and x[j-1] <= x[j]:
                j -= 1
            # Find following trough
            k = i
            while k < len(x)-1 and x[k+1] <= x[k]:
                k += 1
            troughs.append((j, k))
        
        # Filter peaks by height
        indices_to_keep = []
        for i, (j, k) in enumerate(troughs):
            height_before = heights[i] - x[j]
            height_after = heights[i] - x[k]
            min_height = min(height_before, height_after)
            if min_height >= min_h:
                indices_to_keep.append(i)
        ind_max = ind_max[indices_to_keep]
    
    # Filter by minimum prominence
    if min_p > 0 and len(ind_max) > 0:
        # TODO: Implement prominence filtering if needed
        pass
    
    return ind_max


def findrfc(tp, h=0.0, method='numba'):
    """
    Finds rainflow cycles.

    Parameters
    ----------
    tp : vector
        Vector of turningpoints
    h : scalar
        Threshold for rainflow filtering
    method : string, optional
        'numba' uses numba compiled functions (default)
        'python' uses pure python functions

    Returns
    -------
    rfc_out : ndarray
        Rainflow cycles with columns: [min, max, multiplicity]
    """
    if method.startswith('numba'):
        ind = np.zeros(len(tp), dtype=np.int64)
        ix = _findrfc(ind, np.asarray(tp), h)
        t = ind[:ix]
        return t
        
    # Pure Python implementation
    # Convert turning points to numpy array
    tp = np.asarray(tp)
    
    # Pre-allocate output array
    max_cycles = len(tp) // 2
    rfc_out = np.zeros((max_cycles, 3))
    
    # Apply rainflow algorithm
    cycle_count = 0
    stack = [0, 1]  # Start with first two turning points
    
    for i in range(2, len(tp)):
        while len(stack) >= 2:
            # Check if we have a cycle
            a, b, c = tp[stack[-2]], tp[stack[-1]], tp[i]
            if (b-a)*(c-b) <= 0 and abs(b-a) >= h:
                # We have a cycle
                min_val = min(a, b)
                max_val = max(a, b)
                rfc_out[cycle_count] = [min_val, max_val, 1.0]
                cycle_count += 1
                
                # Remove processed points
                stack.pop()
                if len(stack) == 1:
                    stack.append(i)
                    break
            else:
                # No cycle, add new point
                stack.append(i)
                break
        else:
            # Stack too small, add new point
            stack.append(i)
    
    # Process remaining stack
    while len(stack) >= 3:
        a, b = tp[stack[-2]], tp[stack[-1]]
        if abs(b-a) >= h:
            min_val = min(a, b)
            max_val = max(a, b)
            rfc_out[cycle_count] = [min_val, max_val, 1.0]
            cycle_count += 1
        stack.pop()
    
    # Return only the filled part of the array
    return rfc_out[:cycle_count]


def polar2cart(theta, rho, z=None):
    """
    Transform polar coordinates to Cartesian.

    Parameters
    ----------
    theta : array_like
        Angles in radians
    rho : array_like
        Radius
    z : array_like, optional
        Height (default 0)

    Returns
    -------
    x, y, z : array_like
        Cartesian coordinates
    """
    x = rho * cos(theta)
    y = rho * sin(theta)
    
    if z is None:
        return x, y
    return x, y, z


def cart2polar(x, y, z=None):
    """
    Transform Cartesian coordinates to polar.

    Parameters
    ----------
    x, y : array_like
        Cartesian coordinates
    z : array_like, optional
        Height (default None)

    Returns
    -------
    theta, rho, z : array_like
        Polar coordinates
    """
    theta = arctan2(y, x)
    rho = sqrt(x**2 + y**2)
    
    if z is None:
        return theta, rho
    return theta, rho, z


def gravity(phi=45):
    """
    Returns the constant acceleration of gravity.

    Parameters
    ----------
    phi : scalar
        Latitude in degrees.

    Returns
    -------
    g : scalar
        Acceleration of gravity [m/s^2]

    Notes
    -----
    The formula is from Geodetic Reference System 1967.
    g = 9.780327 * (1 + Ax * sin(phi)^2 + Bx * sin(phi)^4)
    where
    Ax = 0.0052792
    Bx = 0.0000232
    """
    phi = phi * pi / 180  # Convert to radians
    sin_phi = sin(phi)
    Ax = 0.0052792
    Bx = 0.0000232
    
    return 9.780327 * (1 + Ax * sin_phi**2 + Bx * sin_phi**4)


def now(show_seconds=True):
    """
    Return current date and time as a string.

    Parameters
    ----------
    show_seconds : bool
        True (default) displays the seconds, False does not.

    Returns
    -------
    time_str : str
        Current time formatted as "Day, DD Mon YYYY HH:MM:SS"
    """
    fmt = "%a, %d %b %Y %H:%M:%S" if show_seconds else "%a, %d %b %Y %H:%M"
    return strftime(fmt, gmtime())
