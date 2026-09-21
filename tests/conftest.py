# tests/conftest.py
import pytest
import numpy as np

from pydas import PyDAS

@pytest.fixture(scope="session")
def synthetic_data_config():
    """Configuration for synthetic data generation."""
    return {
        'fs': 10.0,
        'duration': 600.0,
        'Hs': 5.0,
        'Tp': 10.0,
        'gamma': 3.3
    }

@pytest.fixture(scope="function")
def synthetic_timeseries(synthetic_data_config):
    """Generate a synthetic time series (simple sine wave + noise)."""
    fs = synthetic_data_config['fs']
    duration = synthetic_data_config['duration']
    t = np.arange(0, duration, 1/fs)
    
    Hs = synthetic_data_config['Hs']
    Tp = synthetic_data_config['Tp']
    
    # 3 sine waves
    f1 = 1.0/Tp
    f2 = 2.0/Tp
    f3 = 0.5/Tp
    
    eta = (Hs/2.0) * np.sin(2*np.pi*f1*t) + \
          (Hs/4.0) * np.sin(2*np.pi*f2*t + 1.0) + \
          (Hs/8.0) * np.sin(2*np.pi*f3*t + 2.0)
    
    # Add noise
    sigma = Hs / 20.0
    eta += np.random.normal(0, sigma, len(t))
          
    return t, eta

@pytest.fixture(scope="function")
def pydas_instance(synthetic_timeseries):
    """Create a PyDAS instance with synthetic data."""
    t, eta = synthetic_timeseries
    fs = 1.0 / (t[1] - t[0])
    
    pydas = PyDAS(None, lam=1.0)
    pydas.__filename__ = "synthetic.csv"
    pydas.__date__ = "01-01"
    pydas.__desc__ = "Synthetic test data"
    pydas.__fs__ = fs
    pydas.add_channel('Wave1', 'm', eta, fs)
    
    return pydas


def pytest_collection_modifyitems(config, items):
    """Assign declared markers from the tests/ directory layout."""
    for item in items:
        path = getattr(item, "path", None)
        rel = path.as_posix() if path is not None else str(item.fspath).replace("\\", "/")
        names = {marker.name for marker in item.iter_markers()}
        if "/tests/unit/" in rel and "unit" not in names:
            item.add_marker(pytest.mark.unit)
        elif "/tests/integration/" in rel and "integration" not in names:
            item.add_marker(pytest.mark.integration)
