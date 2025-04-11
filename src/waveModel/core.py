import numpy as np
from time import strftime, gmtime
import warnings

from scipy import integrate, interpolate

from numpy import (amax, logical_and, arange, linspace, atleast_1d,
                   asarray, ceil, floor, frexp, hypot,
                   sqrt, arctan2, sin, cos, exp, log, log1p, mod, diff,
                   inf, pi, interp, isscalar, zeros, ones,
                   sign, unique, hstack, vstack, nonzero, where, extract,
                   meshgrid)

_TINY = np.finfo(float).tiny
_EPS = np.finfo(float).eps

class JITImport(object):

    '''
    Just In Time Import of module
    Examples
    --------
    >>> np = JITImport('numpy')
    >>> np.exp(0)==1.0
    True
    '''

    def __init__(self, module_name):
        self._module_name = module_name
        self._module = None

    def __getattr__(self, attr):
        try:
            return getattr(self._module, attr)
        except AttributeError as exc:
            if self._module is None:
                self._module = __import__(self._module_name, None, None, ['*'])
                # assert(isinstance(self._module, types.ModuleType), 'module')
                return getattr(self._module, attr)
            raise exc

def discretize(fun, a, b, tol=0.005, n=5, method='linear'):
    '''
    Automatic discretization of function
    Parameters
    ----------
    fun : callable
        function to discretize
    a,b : real scalars
        evaluation limits
    tol : real, scalar
        absoute error tolerance
    n : scalar integer
        number of values to start the discretization with.
    method : string
        defining method of gridding, options are 'linear' and 'adaptive'
    Returns
    -------
    x : discretized values
    y : fun(x)
    Examples
    --------
    >>> import wafo.misc as wm
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x,y = wm.discretize(np.cos, 0, np.pi)
    >>> np.allclose(x[:5], [0.,  0.19634954,  0.39269908,  0.58904862,  0.78539816])
    True
    >>> xa,ya = wm.discretize(np.cos, 0, np.pi, method='adaptive')
    >>> np.allclose(xa[:5], [0.,  0.19634954,  0.39269908,  0.58904862,  0.78539816])
    True
    t = plt.plot(x, y, xa, ya, 'r.')
    plt.show()
    plt.close('all')
    '''
    if method.startswith('a'):
        return _discretize_adaptive(fun, a, b, tol, n)
    else:
        return _discretize_linear(fun, a, b, tol, n)


def _discretize_linear(fun, a, b, tol=0.005, n=5):
    '''
    Automatic discretization of function, linear gridding
    '''
    x = linspace(a, b, n)
    y = fun(x)

    err0 = inf
    err = 10000
    nmax = 2 ** 20
    num_tries = 0
    while (num_tries < 5 and err > tol and n < nmax):
        err0 = err
        x0 = x
        y0 = y
        n = 2 * (n - 1) + 1
        x = linspace(a, b, n)
        y = fun(x)
        y00 = interp(x, x0, y0)
        err = 0.5 * amax(np.abs(y00 - y) / (np.abs(y00) + np.abs(y) + _TINY + tol))
        num_tries += int(abs(err - err0) <= tol / 2)
    return x, y


def _discretize_adaptive(fun, a, b, tol=0.005, n=5):
    '''
    Automatic discretization of function, adaptive gridding.
    '''
    n += (mod(n, 2) == 0)  # make sure n is odd
    x = linspace(a, b, n)
    fx = fun(x)

    n2 = (n - 1) // 2
    erri = hstack((zeros((n2, 1)), ones((n2, 1)))).ravel()
    err = erri.max()
    err0 = inf
    num_tries = 0
    # reltol = abstol = tol
    for j in range(50):
        if num_tries < 5 and err > tol:
            err0 = err
            # find top errors

            ix, = where(erri > tol)
            # double the sample rate in intervals with the most error
            y = (vstack(((x[ix] + x[ix - 1]) / 2,
                         (x[ix + 1] + x[ix]) / 2)).T).ravel()
            fy = fun(y)
            fy0 = interp(y, x, fx)

            abserr = np.abs(fy0 - fy)
            erri = 0.5 * (abserr / (np.abs(fy0) + np.abs(fy) + _TINY + tol))
            # converged = abserr <= np.maximum(abseps, releps * abs(fy))
            # converged = abserr <= np.maximum(tol, tol * abs(fy))
            err = erri.max()

            x = hstack((x, y))

            ix = x.argsort()
            x = x[ix]
            erri = hstack((zeros(len(fx)), erri))[ix]
            fx = hstack((fx, fy))[ix]
            num_tries += int(abs(err - err0) <= tol / 2)
        else:
            break
    else:
        warnings.warn('Recursion level limit reached j=%d' % j)

    return x, fx


def sub_dict_select(somedict, somekeys):
    '''
    Extracting a Subset from Dictionary
    Examples
    --------
    # Update options dict from keyword arguments if
    # the keyword exists in options
    >>> opt = dict(arg1=2, arg2=3)
    >>> kwds = dict(arg2=100,arg3=1000)
    >>> sub_dict = sub_dict_select(kwds, opt)
    >>> opt.update(sub_dict)
    >>> opt == {'arg1': 2, 'arg2': 100}
    True
    See also
    --------
    dict_intersection
    '''
    # slower: validKeys = set(somedict).intersection(somekeys)
    return type(somedict)((k, somedict[k]) for k in somekeys if k in somedict)

def nextpow2(x):
    '''
    Return next higher power of 2
    Examples
    --------
    >>> import wafo.misc as wm
    >>> wm.nextpow2(10)
    4
    >>> wm.nextpow2(np.arange(5))
    3
    '''
    t = np.isscalar(x) or len(x)
    if (t > 1):
        f, n = np.frexp(t)
    else:
        f, n = np.frexp(np.abs(x))

    if (f == 0.5):
        n = n - 1
    return n

def ecross(t, f, ind, v=0):
    '''
    Extracts exact level v crossings
    ECROSS interpolates t and f linearly to find the exact level v
    crossings, i.e., the points where f(t0) = v
    Parameters
    ----------
    t,f : vectors
        of arguments and functions values, respectively.
    ind : ndarray of integers
        indices to level v crossings as found by findcross.
    v : scalar or vector (of size(ind))
        defining the level(s) to cross.
    Returns
    -------
    t0 : vector
        of  exact level v crossings.
    Examples
    --------
    -------
    >>> from matplotlib import pyplot as plt
    >>> import wafo.misc as wm
    >>> ones = np.ones
    >>> t = np.linspace(0,7*np.pi,250)
    >>> x = np.sin(t)
    >>> ind = wm.findcross(x,0.75)
    >>> np.allclose(ind, [  9,  25,  80,  97, 151, 168, 223, 239])
    True
    >>> t0 = wm.ecross(t,x,ind,0.75)
    >>> np.allclose(t0, [0.84910514, 2.2933879 , 7.13205663, 8.57630119,
    ...        13.41484739, 14.85909194, 19.69776067, 21.14204343])
    True
    a = plt.plot(t, x, '.', t[ind], x[ind], 'r.', t, ones(t.shape)*0.75,
                  t0, ones(t0.shape)*0.75, 'g.')
    plt.close('all')
    See also
    --------
    findcross
    '''
    # Tested on: Python 2.5
    # revised pab Feb2004
    # By pab 18.06.2001
    return (t[ind] + (v - f[ind]) * (t[ind + 1] - t[ind]) /
            (f[ind + 1] - f[ind]))

def now(show_seconds=True):
    '''
    Return current date and time as a string
    '''
    if show_seconds:
        return strftime("%a, %d %b %Y %H:%M:%S", gmtime())
    return strftime("%a, %d %b %Y %H:%M", gmtime())

def empty_copy(obj):
    class Empty(obj.__class__):

        def __init__(self):
            pass
    newcopy = Empty()
    # pylint: disable=attribute-defined-outside-init
    newcopy.__class__ = obj.__class__
    return newcopy

def findtp(x, h=0.0, kind=None):
    '''
    Return indices to turning points (tp) of data, optionally rainflowfiltered.
    Parameters
    ----------
    x : vector
        signal
    h : real, scalar
        rainflow threshold
         if  h<0, then ind = range(len(x))
         if  h=0, then  tp  is a sequence of turning points (default)
         if  h>0, then all rainflow cycles with height smaller than
                  h  are removed.
    kind : string
        defines the type of wave or indicate the ASTM rainflow counting method.
        Possible options are 'astm' 'mw' 'Mw' or 'none'.
        If None all rainflow filtered min and max
        will be returned, otherwise only the rainflow filtered
        min and max, which define a wave according to the
        wave definition, will be returned.
    Returns
    -------
    ind : arraylike
        indices to the turning points in the original sequence.
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import wafo.misc as wm
    >>> t = np.linspace(0,30,500).reshape((-1,1))
    >>> x = np.hstack((t, np.cos(t) + 0.3 * np.sin(5*t)))
    >>> x1 = x[0:100,:]
    >>> itp = wm.findtp(x1[:,1],0,'Mw')
    >>> itph = wm.findtp(x1[:,1],0.3,'Mw')
    >>> tp = x1[itp,:]
    >>> tph = x1[itph,:]
    >>> np.allclose(itp, [ 5, 18, 24, 38, 46, 57, 70, 76, 91, 98, 99])
    True
    >>> np.allclose(itph, 91)
    True
    a = plt.plot(x1[:,0],x1[:,1],
                 tp[:,0],tp[:,1],'ro',
                 tph[:,0],tph[:,1],'k.')
    plt.close('all')
    See also
    ---------
    findtc
    findcross
    findextrema
    findrfc
    '''
    n = len(x)
    if h < 0.0:
        return arange(n)

    ind = findextrema(x)

    if ind.size < 2:
        return None

    # In order to get the exact up-crossing intensity from rfc by
    # mm2lc(tp2mm(rfc))  we have to add the indices to the last value
    # (and also the first if the sequence of turning points does not start
    # with a minimum).

    if kind == 'astm':
        # the Nieslony approach always put the first loading point as the first
        # turning point.
        # add the first turning point is the first of the signal
        if ind[0] != 0:
            ind = np.r_[0, ind, n - 1]
        else:  # only add the last point of the signal
            ind = np.r_[ind, n - 1]
    else:
        if x[ind[0]] > x[ind[1]]:  # adds indices to  first and last value
            ind = np.r_[0, ind, n - 1]
        else:  # adds index to the last value
            ind = np.r_[ind, n - 1]

    if h > 0.0:
        ind1 = findrfc(x[ind], h)
        ind = ind[ind1]

    if kind in ('mw', 'Mw'):
        # make sure that the first is a Max if wdef == 'Mw'
        # or make sure that the first is a min if wdef == 'mw'
        first_is_max = (x[ind[0]] > x[ind[1]])

        remove_first = xor(first_is_max, kind.startswith('Mw'))
        if remove_first:
            ind = ind[1::]

        # make sure the number of minima and Maxima are according to the
        # wavedef. i.e., make sure Nm=length(ind) is odd
        if (mod(ind.size, 2)) != 1:
            ind = ind[:-1]
    return ind


def findtc(x_in, v=None, kind=None):
    """
    Return indices to troughs and crests of data.
    Parameters
    ----------
    x : vector
        surface elevation.
    v : real scalar
        reference level (default  v = mean of x).
    kind : string
        defines the type of wave. Possible options are
        'dw', 'uw', 'tw', 'cw' or None.
        If None indices to all troughs and crests will be returned,
        otherwise only the paired ones will be returned
        according to the wavedefinition.
    Returns
    --------
    tc_ind : vector of ints
        indices to the trough and crest turningpoints of sequence x.
    v_ind : vector of ints
        indices to the level v crossings of the original
        sequence x. (d,u)
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import wafo.misc as wm
    >>> t = np.linspace(0,30,500).reshape((-1,1))
    >>> x = np.hstack((t, np.cos(t)))
    >>> x1 = x[0:200,:]
    >>> itc, iv = wm.findtc(x1[:,1],0,'dw')
    >>> tc = x1[itc,:]
    >>> np.allclose(itc, [ 52, 105])
    True
    >>> itc, iv = wm.findtc(x1[:,1],0,'uw')
    >>> np.allclose(itc, [ 105, 157])
    True
    a = plt.plot(x1[:,0],x1[:,1],tc[:,0],tc[:,1],'ro')
    plt.close('all')
    See also
    --------
    findtp
    findcross,
    wavedef
    """

    x = atleast_1d(x_in)
    if v is None:
        v = x.mean()

    v_ind = findcross(x, v, kind)
    n_c = v_ind.size
    if n_c <= 2:
        warnings.warn('There are no waves!')
        return zeros(0, dtype=np.int), zeros(0, dtype=np.int)

    # determine the number of trough2crest (or crest2trough) cycles
    is_even = mod(n_c + 1, 2)
    n_tc = int((n_c - 1 - is_even) / 2)

    # allocate variables before the loop increases the speed
    ind = zeros(n_c - 1, dtype=np.int)

    first_is_down_crossing = (x[v_ind[0]] > x[v_ind[0] + 1])
    if first_is_down_crossing:
        f1, f2 = np.argmin, np.argmax
    else:
        f1, f2 = np.argmax, np.argmin

    for i in range(n_tc):
        # trough or crest
        j = 2 * i
        ind[j] = f1(x[v_ind[j] + 1:v_ind[j + 1] + 1])
        # crest or trough
        ind[j + 1] = f2(x[v_ind[j + 1] + 1:v_ind[j + 2] + 1])

    if (2 * n_tc + 1 < n_c) and (kind in (None, 'tw', 'cw')):
        # trough or crest
        ind[n_c - 2] = f1(x[v_ind[n_c - 2] + 1:v_ind[n_c - 1] + 1])

    return v_ind[:n_c - 1] + ind + 1, v_ind


def findoutliers(x, zcrit=0.0, dcrit=None, ddcrit=None, verbose=False):
    """
    Return indices to spurious points of data
    Parameters
    ----------
    x : vector
        of data values.
    zcrit : real scalar
        critical distance between consecutive points.
    dcrit : real scalar
        critical distance of Dx used for determination of spurious
        points.  (Default 1.5 standard deviation of x)
    ddcrit : real scalar
        critical distance of DDx used for determination of spurious
        points.  (Default 1.5 standard deviation of x)
    Returns
    -------
    inds : ndarray of integers
        indices to spurious points.
    indg : ndarray of integers
        indices to the rest of the points.
    Notes
    -----
    Consecutive points less than zcrit apart  are considered as spurious.
    The point immediately after and before are also removed. Jumps greater than
    dcrit in Dxn and greater than ddcrit in D^2xn are also considered as
    spurious.
    (All distances to be interpreted in the vertical direction.)
    Another good choice for dcrit and ddcrit are:
        dcrit = 5*dT  and ddcrit = 9.81/2*dT**2
    where dT is the timestep between points.
    Examples
    --------
    >>> import numpy as np
    >>> import wafo.misc as wm
    >>> t = np.linspace(0,30,500).reshape((-1,1))
    >>> xx = np.hstack((t, np.cos(t)))
    >>> dt = np.diff(xx[:2,0])
    >>> dcrit = 5*dt
    >>> ddcrit = 9.81/2*dt*dt
    >>> zcrit = 0
    >>> inds, indg = wm.findoutliers(xx[:,1], verbose=True)
    Found 0 missing points
    dcrit is set to 1.05693
    ddcrit is set to 1.05693
    Found 0 spurious positive jumps of Dx
    Found 0 spurious negative jumps of Dx
    Found 0 spurious positive jumps of D^2x
    Found 0 spurious negative jumps of D^2x
    Found 0 consecutive equal values
    Found the total of 0 spurious points
    #waveplot(xx,'-',xx(inds,:),1,1,1)
    See also
    --------
    waveplot, reconstruct
    """

    def _find_nans(xn):
        i_missing = np.flatnonzero(np.isnan(xn))
        if verbose:
            print('Found %d missing points' % i_missing.size)
        return i_missing

    def _find_spurious_jumps(dxn, dcrit, name='Dx'):
        i_p = np.flatnonzero(dxn > dcrit)
        if i_p.size > 0:
            i_p += 1  # the point after the jump
        if verbose:
            print('Found {0:d} spurious positive jumps of {1}'.format(i_p.size,
                                                                      name))

        i_n = np.flatnonzero(dxn < -dcrit)  # the point before the jump
        if verbose:
            print('Found {0:d} spurious negative jumps of {1}'.format(i_n.size,
                                                                      name))
        if i_n.size > 0:
            return hstack((i_p, i_n))
        return i_p

    def _find_consecutive_equal_values(dxn, zcrit):

        mask_small = (np.abs(dxn) <= zcrit)
        i_small = np.flatnonzero(mask_small)
        if verbose:
            if zcrit == 0.:
                print('Found %d consecutive equal values' % i_small.size)
            else:
                print('Found %d consecutive values less than %g apart.' %
                      (i_small.size, zcrit))
        if i_small.size > 0:
            i_small += 1
            # finding the beginning and end of consecutive equal values
            i_step = np.flatnonzero((diff(mask_small))) + 1
            # indices to consecutive equal points
            # removing the point before + all equal points + the point after

            return hstack((i_step - 1, i_small, i_step, i_step + 1))
        return i_small

    xn = asarray(x).flatten()

    # _assert(2 < xn.size, 'The vector must have more than 2 elements!')

    i_missing = _find_nans(xn)
    if np.any(i_missing):
        xn[i_missing] = 0.  # set NaN's to zero
    if dcrit is None:
        dcrit = 1.5 * xn.std()
        if verbose:
            print('dcrit is set to %g' % dcrit)

    if ddcrit is None:
        ddcrit = 1.5 * xn.std()
        if verbose:
            print('ddcrit is set to %g' % ddcrit)

    dxn = diff(xn)
    ddxn = diff(dxn)

    ind = np.hstack((_find_spurious_jumps(dxn, dcrit, name='Dx'),
                     _find_spurious_jumps(ddxn, ddcrit, name='D^2x'),
                     _find_consecutive_equal_values(dxn, zcrit)))

    indg = ones(xn.size, dtype=bool)
    if ind.size > 1:
        ind = unique(ind)
        indg[ind] = 0
    indg, = nonzero(indg)

    if verbose:
        print('Found the total of %d spurious points' % np.size(ind))

    return ind, indg


def common_shape(*args, ** kwds):
    """Return the common shape of a sequence of arrays.
    Parameters
    -----------
    *args : arraylike
        sequence of arrays
    **kwds :
        shape
    Returns
    -------
    shape : tuple
        common shape of the elements of args.
    Raises
    ------
    An error is raised if some of the arrays do not conform
    to the common shape according to the broadcasting rules in numpy.
    Examples
    --------
    >>> import numpy as np
    >>> import wafo.misc as wm
    >>> A = np.ones((4,1))
    >>> B = 2
    >>> C = np.ones((1,5))*5
    >>> np.allclose(wm.common_shape(A,B,C), (4, 5))
    True
    >>> np.allclose(wm.common_shape(A,B,C,shape=(3,4,1)), (3, 4, 5))
    True
    See also
    --------
    broadcast, broadcast_arrays
    """
    shape = kwds.get('shape')
    x0 = 1 if shape is None else np.ones(shape)
    return tuple(np.broadcast(x0, *args).shape)


def argsreduce(condition, * args):
    """ Return the elements of each input array that satisfy some condition.
    Parameters
    ----------
    condition : array_like
        An array whose nonzero or True entries indicate the elements of each
        input array to extract. The shape of 'condition' must match the common
        shape of the input arrays according to the broadcasting rules in numpy.
    arg1, arg2, arg3, ... : array_like
        one or more input arrays.
    Returns
    -------
    narg1, narg2, narg3, ... : ndarray
        sequence of extracted copies of the input arrays converted to the same
        size as the nonzero values of condition.
    Examples
    --------
    >>> import wafo.misc as wm
    >>> import numpy as np
    >>> rand = np.random.random_sample
    >>> A = rand((4,5))
    >>> B = 2
    >>> C = rand((1,5))
    >>> cond = np.ones(A.shape)
    >>> [A1,B1,C1] = wm.argsreduce(cond,A,B,C)
    >>> np.allclose(B1.shape, (20,))
    True
    >>> cond[2,:] = 0
    >>> [A2,B2,C2] = wm.argsreduce(cond,A,B,C)
    >>> np.allclose(B2.shape, (15,))
    True
    See also
    --------
    numpy.extract
    """
    newargs = atleast_1d(*args)
    if not isinstance(newargs, list):
        newargs = [newargs, ]
    expand_arr = (condition == condition)
    return [extract(condition, arr1 * expand_arr) for arr1 in newargs]