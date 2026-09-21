# Simplified waveModel package for PyDAS

from .core import w2k, spectral_moments, wave_parameters, max_wave_height
from .models import (
    jonswap_spectrum,
    pm_spectrum,
    torsethaugen_spectrum,
    directional_spectrum,
    cos2s_spreading,
)
from .simulation import spectrum_to_timeseries, directional_sim
from .analysis import (
    timeseries_to_spectrum,
    timeseries_to_acf,
    spectrum_to_acf,
    acf_to_spectrum,
)
from .objects import TimeSeries, SpecData1D, CovData1D, Jonswap

# Aliases for WAFO compatibility
jonswap = jonswap_spectrum
torsethaugen = torsethaugen_spectrum
PM = pm_spectrum

__all__ = [
    "w2k", "spectral_moments", "wave_parameters", "max_wave_height",
    "jonswap_spectrum", "pm_spectrum", "torsethaugen_spectrum",
    "directional_spectrum", "cos2s_spreading",
    "spectrum_to_timeseries", "directional_sim",
    "timeseries_to_spectrum", "timeseries_to_acf",
    "spectrum_to_acf", "acf_to_spectrum",
    "TimeSeries", "SpecData1D", "CovData1D", "Jonswap",
    "jonswap", "torsethaugen", "PM",
]
