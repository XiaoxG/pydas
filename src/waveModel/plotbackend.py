import warnings
verbose = False
try:
    from matplotlib import pyplot as plotbackend
    plotbackend.interactive(True)
    if verbose:
        print('wafo: plotbackend is set to matplotlib.pyplot')
except ImportError:
    warnings.warn('wafo: Unable to load matplotlib.pyplot as plotbackend')
    plotbackend = None