"""
Data container classes for the WaveModel package.
"""
import warnings
import numpy as np
from scipy import interpolate
from scipy.integrate import cumtrapz
from scipy import integrate
from waveModel.core import now

__all__ = ['DataContainer', 'AxisLabels', 'PlotData']

def empty_copy(obj):
    """Create an empty copy of an object with the same class."""
    class Empty(obj.__class__):
        def __init__(self):
            pass
    newcopy = Empty()
    # pylint: disable=attribute-defined-outside-init
    newcopy.__class__ = obj.__class__
    return newcopy


def _set_seed(iseed):
    """Set random seed."""
    if iseed is not None:
        try:
            np.random.set_state(iseed)
        except ValueError:
            np.random.seed(iseed)


class DataContainer(object):
    """Container class for data with interpolation methods.
    
    Member variables
    ----------------
    data : array_like
        Data values
    args : vector for 1D, list of vectors for 2D, 3D, ...
        Arguments (e.g., time points, coordinates)
    labels : AxisLabels
        Labels for axes, title, etc.
    children : list of DataContainer objects
        Child data containers, e.g., for confidence intervals
    """

    def __init__(self, data=None, args=None, **kwds):
        self.data = data
        self.args = args
        self.date = now()
        self.children = kwds.pop('children', None)
        self.labels = AxisLabels(**kwds)

    def copy(self):
        """Return a copy of the object."""
        newcopy = empty_copy(self)
        newcopy.__dict__.update(self.__dict__)
        return newcopy

    def eval_points(self, *points, **kwds):
        """Interpolate data at points.
        
        Parameters
        ----------
        points :  ndarray of float, shape (..., ndim)
            Points where to interpolate data at.
        method : {'linear', 'nearest', 'cubic'}
            Method of interpolation. One of
            - ``nearest``: return the value at the data point closest to
              the point of interpolation.
            - ``linear``: tesselate the input point set to n-dimensional
              simplices, and interpolate linearly on each simplex.
            - ``cubic`` (1-D): return the value detemined from a cubic
              spline.
            - ``cubic`` (2-D): return the value determined from a
              piecewise cubic, continuously differentiable (C1), and
              approximately curvature-minimizing polynomial surface.
        fill_value : float, optional
            Value used to fill in for requested points outside of the
            convex hull of the input points.  If not provided, then the
            default is ``nan``. This option has no effect for the
            'nearest' method.
        """
        options = dict(method='linear')
        options.update(**kwds)
        if isinstance(self.args, (list, tuple)):  # Multidimensional data
            ndim = len(self.args)
            if ndim < 2:
                msg = '''
                Unable to determine data type, because len(self.args)<2.
                If the data is 1D, then self.args should be a vector!
                If the data is 2D, then length(self.args) should be 2.
                If the data is 3D, then length(self.args) should be 3.
                Unless you fix this, the interpolation will not work!'''
                warnings.warn(msg)
            else:
                xi = np.meshgrid(*self.args)
                return interpolate.griddata(xi, self.data.ravel(), points,
                                            **options)
        # One dimensional data
        return interpolate.griddata(self.args, self.data, points, **options)

    def to_cdf(self):
        """Convert to cumulative distribution function."""
        if isinstance(self.args, (list, tuple)):  # Multidimensional data
            raise NotImplementedError('integration for ndim>1 not implemented')
        cdf = np.hstack((0, cumtrapz(self.data, self.args)))
        return DataContainer(cdf, np.copy(self.args), xlab='x', ylab='F(x)')

    def _get_fi_xi(self, a, b):
        """Get function values and arguments in range [a, b]."""
        x = self.args
        if a is None:
            a = x[0]
        if b is None:
            b = x[-1]
        ix = np.flatnonzero((a < x) & (x < b))
        xi = np.hstack((a, x.take(ix), b))

        if self.data.ndim > 1:
            fi = np.vstack((self.eval_points(a),
                            self.data[ix, :],
                            self.eval_points(b))).T
        else:
            fi = np.hstack((self.eval_points(a), self.data.take(ix),
                            self.eval_points(b)))
        return fi, xi

    def integrate(self, a=None, b=None, **kwds):
        """
        Calculate the integral of the data over the specified interval.
        
        Parameters
        ----------
        a, b : float, optional
            Integration limits. Default is the full range of the data.
        method : str, optional
            Integration method ('trapz' by default)
        return_ci : bool, optional
            If True, also return confidence intervals for children
            
        Returns
        -------
        res : float or array
            Integral value, or array including confidence intervals if return_ci=True
        """
        method = kwds.pop('method', 'trapz')
        fun = getattr(integrate, method)
        if isinstance(self.args, (list, tuple)):  # Multidimensional data
            raise NotImplementedError('integration for ndim>1 not implemented')
        # One dimensional data
        return_ci = kwds.pop('return_ci', False)
        fi, xi = self._get_fi_xi(a, b)
        res = fun(fi, xi, **kwds)
        if return_ci:
            res_ci = [child.integrate(a, b, method=method)
                      for child in self.children]
            return np.hstack((res, np.ravel(res_ci)))
        return res
    
    interpolate = eval_points


class AxisLabels:
    """Container for axis labels."""
    
    def __init__(self, title='', xlab='', ylab='', zlab='', **kwds):
        self.title = title
        self.xlab = xlab
        self.ylab = ylab
        self.zlab = zlab

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '{0.__class__.__name__}(title={0.title!r}, xlab={0.xlab!r}, ylab={0.ylab!r}, zlab={0.zlab!r})'.format(self)

    def copy(self):
        """Return a copy of the labels."""
        newcopy = empty_copy(self)
        newcopy.__dict__.update(self.__dict__)
        return newcopy


# For backward compatibility
class PlotData(DataContainer):
    """
    Alias for DataContainer for backward compatibility.
    
    This class provides the same functionality as DataContainer
    but is kept for backward compatibility with code using PlotData.
    Use DataContainer for new code.
    """
    pass 