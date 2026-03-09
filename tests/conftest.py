# tests/conftest.py
import pytest
import numpy as np
import pandas as pd
import os
import sys

# Ensure src is in path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

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
def pydas_instance(synthetic_timeseries, tmp_path):
    """Create a PyDAS instance with synthetic data."""
    t, eta = synthetic_timeseries
    fs = 1.0 / (t[1] - t[0])
    
    # Initialize empty PyDAS
    pydas = PyDAS(None, lam=1.0)
    pydas.__filename__ = "synthetic.csv"
    pydas.__date__ = "01-01"
    pydas.__desc__ = "Synthetic test data"
    
    # Manually populate basics
    pydas.__fs__ = fs
    pydas.__chN__ = 0
    pydas.__segN__ = 1
    # Initialize data structure
    pydas.data = {0: pd.DataFrame()} # Segment 0
    pydas.segInfo = pd.DataFrame([{'Start': '00:00:00.0', 'Stop': '00:10:00.0', 'Duration': 600.0, 'N sample': len(t), 'Type': 1, 'Note': ''}])
    pydas.segInfo.index = ['Seg 0']
    pydas.segStatis = {0: pd.DataFrame(columns=['Mean', 'STD', 'Max', 'Min', 'Unit'])}
    
    # Add the channel
    # add_channel expects: name, unit, series, fs, coef=1.0, point_of_move=0, sseg='all'
    pydas.add_channel('Wave1', 'm', eta, fs)
    
    # Also add Time channel if needed, but PyDAS usually handles time implicitly via index or fs
    
    return pydas
