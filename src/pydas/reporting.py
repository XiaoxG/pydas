"""
PyDAS Reporting Module
======================
This module contains reporting functions for PyDAS data.
- Excel report generation
- Statistical summary reports
- Channel analysis reports

Author: Xiaoxian Guo
Date: 2025-04-12
"""

import os
import numpy as np
import pandas as pd
import logging
from openpyxl.styles import Font, Alignment, Border, Side, PatternFill
from openpyxl.utils import get_column_letter
import scipy.signal as signal
import copy
from scipy import stats as spstats
from openpyxl.worksheet.page import PageMargins, PrintPageSetup
import matplotlib.pyplot as plt
from .waveModel.objects import Jonswap
import matplotlib.gridspec as gridspec
from scipy.stats import norm

# Set up logging
logger = logging.getLogger('pydas.reporting')

# Euler-Mascheroni constant for expected extreme value calculation
GAMMA = 0.57721566

# ---------------------------------------------------------------------------
# Metric catalog
# ---------------------------------------------------------------------------
# Each entry maps a metric ID to (column header for Excel, key in the result
# dict returned by ``analyze_channel_data``).
#
# Column headers may contain explicit ``\n`` line breaks, which Excel will
# render as multi-line headers when ``wrap_text=True`` is applied.
# ---------------------------------------------------------------------------
METRIC_CATALOG = {
    # --- Basic statistics ---
    'maximum':                  ('maximum',                 'maximum'),
    'minimum':                  ('minimum',                 'minimum'),
    'mean':                     ('mean',                    'mean'),
    'STD':                      ('STD',                     'STD'),
    # --- Zero-crossing analysis ---
    'zero_upcross':             ('number\nof zero\nupcross', 'zero_upcross'),
    'mean_zerocross_period':    ('mean\nzerocro.\nperiod',  'mean_zerocross_period'),
    # --- Wave-by-wave amplitude (irregular wave) ---
    'maximum_double_amplitude': ('maximum\ndouble\namplitude', 'maximum_double_amplitude'),
    'sign_double_amplitude':    ('sign.\ndouble\namplitude',   'sign_double_amplitude'),
    'pos_sign_amplitude':       ('Pos. sign.\namplitude',   'pos_sign_amplitude'),
    'neg_sign_amplitude':       ('Neg. sign.\namplitude',   'neg_sign_amplitude'),
    # --- STD-based amplitude (regular wave; assumes sinusoidal: A = sqrt(2)*sigma) ---
    # Internal IDs keep the ``_std`` suffix to preserve the calculation
    # semantics, but the Excel column headers are intentionally simplified.
    'amplitude_std':            ('amplitude',               'amplitude_std'),
    'double_amplitude_std':     ('double\namplitude',       'double_amplitude_std'),
    # --- Extreme value estimates ---
    'mpm_pos':                  ('MPM_pos',                 'mpm_pos'),
    'mpm_neg':                  ('MPM_neg',                 'mpm_neg'),
    'eev_pos':                  ('EEV_pos',                 'eev_pos'),
    'eev_neg':                  ('EEV_neg',                 'eev_neg'),
    # --- Signal characteristics ---
    'irregularity_factor':      ('irregularity\nfactor',    'irregularity_factor'),
    'crest_factor':             ('crest\nfactor',           'crest_factor'),
}

# Default metric sets per wave-type. Order in the list defines column order.
DEFAULT_METRICS_IRREGULAR = [
    'zero_upcross',
    'maximum', 'minimum', 'mean', 'STD',
    'maximum_double_amplitude', 'sign_double_amplitude',
    'pos_sign_amplitude', 'neg_sign_amplitude',
    'mpm_pos', 'mpm_neg', 'eev_pos', 'eev_neg',
    'irregularity_factor', 'crest_factor',
    'mean_zerocross_period',
]

DEFAULT_METRICS_REGULAR = [
    'zero_upcross',
    'maximum', 'minimum', 'mean', 'STD',
    'amplitude_std', 'double_amplitude_std',
    'mean_zerocross_period',
]

# Metric IDs that require Most-Probable-Maximum / Expected-Extreme-Value
# computation (used to decide whether to run the heavy POT/STD analysis).
_MPM_METRIC_IDS = {
    'mpm_pos', 'mpm_neg', 'eev_pos', 'eev_neg',
    'pos_sign_amplitude', 'neg_sign_amplitude',
}

# Metric IDs that require wave-by-wave peak detection.
_PEAK_METRIC_IDS = {
    'maximum_double_amplitude', 'sign_double_amplitude',
} | _MPM_METRIC_IDS


def _resolve_metrics(wave_type, metrics):
    """Return a validated list of metric IDs based on ``wave_type``/``metrics``.

    Parameters
    ----------
    wave_type : str
        Either ``'irregular'`` or ``'regular'``. Determines the default metric
        set when ``metrics`` is *None*.
    metrics : list of str or None
        User-specified metric IDs. When *None*, the default set for
        ``wave_type`` is used.

    Returns
    -------
    list of str
        Validated list of metric IDs (unknown IDs are dropped with a warning).
    """
    wave_type = (wave_type or 'irregular').lower()
    if wave_type not in ('irregular', 'regular'):
        logger.warning(
            f"Unknown wave_type '{wave_type}'. Falling back to 'irregular'.")
        wave_type = 'irregular'

    if metrics is None:
        metrics = (DEFAULT_METRICS_IRREGULAR if wave_type == 'irregular'
                   else DEFAULT_METRICS_REGULAR)

    validated = []
    for m in metrics:
        if m in METRIC_CATALOG:
            validated.append(m)
        else:
            logger.warning(f"Unknown metric ID '{m}', ignored.")
    return validated

def analyze_channel_data(data_scaled, mean_val=None, std_val=None, zerocrossing_analysis=True, 
                      amplitude_analysis=True, significant_percentile=33.0,
                      data_duration_hours=None, dt=None, peak_distance=130, 
                      pot_threshold_factor=1.5, mpm_method='POT',
                      wave_type='irregular', compute_extremes=None):
    """
    Analyze channel data and return statistical results including MPM, EEV, and ocean engineering parameters

    Parameters
    ----------
    data_scaled : numpy.ndarray
        The scaled data to analyze
    mean_val : float, optional
        Pre-calculated mean value
    std_val : float, optional
        Pre-calculated standard deviation
    zerocrossing_analysis : bool, default=True
        Whether to perform zero-crossing analysis
    amplitude_analysis : bool, default=True
        Whether to perform amplitude analysis
    significant_percentile : float, default=33.0
        Percentile for significant value calculation
    data_duration_hours : float, optional
        Duration of data in hours
    dt : float, optional
        Time step
    peak_distance : int, default=130
        Minimum distance between peaks for peak detection
    pot_threshold_factor : float, default=1.5
        Threshold coefficient for the Peak-Over-Threshold (POT) method.
        Actual threshold = pot_threshold_factor x STD / sqrt(2) (one-sided).
        Typical values: 1.0 (aggressive), 1.5 (moderate), 2.0 (conservative).
    mpm_method : str, default='POT'
        Method for computing the Most Probable Maximum (MPM).
        - 'POT': Peak-Over-Threshold with Weibull distribution fit (default; more accurate
          but computationally intensive).
        - 'STD': Simplified STD-based method (assumes narrow-band process, suitable for
          linear waves).
    wave_type : str, default='irregular'
        Wave-type hint that controls the default analysis depth:
        - 'irregular': perform full peak detection, MPM/EEV estimation and
          wave-by-wave amplitude analysis.
        - 'regular': skip MPM/EEV (and the heavy peak/Weibull pipeline) by
          default. Only basic statistics, zero-crossing analysis and the
          STD-based amplitude estimates are produced.
    compute_extremes : bool, optional
        Override for ``wave_type``. When *None* (default), MPM/EEV computation
        is enabled for ``wave_type='irregular'`` and disabled for ``'regular'``.
        Set to *True* / *False* to force-enable / disable independently of the
        wave type (useful when the caller only needs a subset of metrics).

    Returns
    -------
    dict
        Dictionary containing all statistical results including:
        - Basic statistics: maximum, minimum, mean, STD
        - STD-based amplitudes: amplitude_std (= sqrt(2)*STD),
          double_amplitude_std (= 2*sqrt(2)*STD)
        - Wave parameters: maximum_double_amplitude, sign_double_amplitude,
          mean_zerocross_period, zero_upcross
        - Extreme values: MPM (Most Probable Maximum) and EEV (Expected Extreme Value)
          for positive and negative peaks
        - Ocean engineering parameters: irregularity_factor, crest_factor
    """
    # Resolve compute_extremes from wave_type if not specified explicitly
    if compute_extremes is None:
        compute_extremes = (str(wave_type).lower() != 'regular')
    
    # Helper functions for MPM calculation (Weibull EVD)
    def _weibull_evd_pdf(x, mu, sigma, k, n):
        """Calculate the PDF of extreme value distribution for Weibull."""
        if x <= mu:
            return 0.0
        u = (x - mu) / sigma
        if u <= 0:
            return 0.0
        cdf_base = 1 - np.exp(-u**k)
        pdf_base = k/sigma * u**(k-1) * np.exp(-u**k)
        if cdf_base <= 0 or cdf_base >= 1:
            return 0.0
        return n * cdf_base**(n-1) * pdf_base

    def _negative_log_pdf(x, mu, sigma, k, n):
        """Negative log PDF for optimization (to find MPM)."""
        pdf = _weibull_evd_pdf(x, mu, sigma, k, n)
        if pdf <= 0:
            return 1e10
        return -np.log(pdf)

    # Calculate basic statistics if not provided
    if mean_val is None:
        mean_val = np.mean(data_scaled)
    if std_val is None:
        std_val = np.std(data_scaled)
    max_val = np.max(data_scaled)
    min_val = np.min(data_scaled)
    
    # Initialize results (removed skewness, kurtosis, and Weibull parameters)
    results = {
        'maximum': max_val,
        'minimum': min_val,
        'mean': mean_val,
        'STD': std_val,
        # STD-based amplitudes (theoretical values for a sinusoidal signal):
        #   single amplitude  A  = sqrt(2) * sigma
        #   double amplitude  2A = 2*sqrt(2) * sigma
        # These are always available because they are essentially free to compute.
        'amplitude_std': float(np.sqrt(2.0) * std_val),
        'double_amplitude_std': float(2.0 * np.sqrt(2.0) * std_val),
        'maximum_double_amplitude': 0,
        'sign_double_amplitude': 0,
        'pos_sign_amplitude': np.nan,  # Positive significant amplitude
        'neg_sign_amplitude': np.nan,  # Negative significant amplitude
        'mean_zerocross_period': 0,
        'zero_upcross': 0,
        # MPM and expected extremes
        'mpm_pos': np.nan,
        'mpm_neg': np.nan,
        'eev_pos': np.nan,
        'eev_neg': np.nan,
        # New ocean engineering parameters
        'irregularity_factor': np.nan,
        'crest_factor': np.nan
    }
    
    # Remove mean for MPM analysis
    data_centered = data_scaled - mean_val
    
    # MPM Analysis will use peaks from zerocrossing_analysis if available
    
    if zerocrossing_analysis:
        # Calculate mean crossings (already have data_centered)
        zero_crossings = np.where(np.diff(np.signbit(data_centered)))[0]
        upcrossings = [i for i in zero_crossings if data_centered[i+1] > data_centered[i]]
        results['zero_upcross'] = len(upcrossings)
        
        # Calculate mean period first (before peak detection)
        mean_period_samples = 0
        if results['zero_upcross'] > 1:
            periods = np.diff(upcrossings) * dt
            results['mean_zerocross_period'] = np.mean(periods)
            # Convert to samples
            mean_period_samples = int(results['mean_zerocross_period'] / dt)
        
        # Set peak detection distance based on mean zero-crossing period
        # Use 0.5 times the mean period as minimum peak distance
        if mean_period_samples > 0:
            min_peak_distance = int(0.5 * mean_period_samples)
            logger.debug(f"Using peak distance of {min_peak_distance} samples (0.5 × mean period of {mean_period_samples} samples)")
        else:
            # Fallback to user-provided or default value
            min_peak_distance = peak_distance
            logger.debug(f"No valid mean period, using default peak distance of {min_peak_distance} samples")
        
        # Find peaks and troughs on original data with appropriate distance constraint
        peaks, _ = signal.find_peaks(data_scaled, distance=min_peak_distance)
        troughs, _ = signal.find_peaks(-data_scaled, distance=min_peak_distance)
        
        if len(peaks) > 0 and len(troughs) > 0:
            # Sort peaks and troughs
            peaks = np.sort(peaks)
            troughs = np.sort(troughs)
            
            # Calculate double amplitudes (using original data)
            double_amplitudes = []
            
            # Method 1: Calculate double amplitude for each complete wave (trough-peak-trough pattern)
            # First, merge and sort peaks and troughs with their types
            extrema_indices = []
            extrema_types = []
            extrema_values = []
            
            for idx in peaks:
                extrema_indices.append(idx)
                extrema_types.append('peak')
                extrema_values.append(data_scaled[idx])
                
            for idx in troughs:
                extrema_indices.append(idx)
                extrema_types.append('trough')
                extrema_values.append(data_scaled[idx])
            
            # Sort by index
            if len(extrema_indices) > 0:
                sort_order = np.argsort(extrema_indices)
                extrema_indices = np.array(extrema_indices)[sort_order]
                extrema_types = np.array(extrema_types)[sort_order]
                extrema_values = np.array(extrema_values)[sort_order]
                
                # Find trough-peak-trough patterns
                for i in range(1, len(extrema_types) - 1):
                    if (extrema_types[i-1] == 'trough' and 
                        extrema_types[i] == 'peak' and 
                        extrema_types[i+1] == 'trough'):
                        # Found a complete wave pattern
                        trough1_val = extrema_values[i-1]
                        peak_val = extrema_values[i]
                        trough2_val = extrema_values[i+1]
                        
                        # Double amplitude is from the lower trough to the peak
                        lower_trough = min(trough1_val, trough2_val)
                        double_amp = peak_val - lower_trough
                        
                        if double_amp > 0:
                            double_amplitudes.append(double_amp)
            
            # Method 2: Use zero-crossing waves for more accurate wave-by-wave analysis
            if zerocrossing_analysis and len(upcrossings) > 1:
                wave_amplitudes = []
                for i in range(len(upcrossings) - 1):
                    start_idx = upcrossings[i]
                    end_idx = upcrossings[i+1]
                    wave_segment = data_scaled[start_idx:end_idx+1]
                    if len(wave_segment) > 2:
                        # Double amplitude is max minus min within the wave
                        wave_max = np.max(wave_segment)
                        wave_min = np.min(wave_segment)
                        wave_amp = wave_max - wave_min
                        if wave_amp > 0:
                            wave_amplitudes.append(wave_amp)
                
                # Use zero-crossing method if it provides more complete analysis
                if len(wave_amplitudes) > 0:
                    logger.debug(f"Using zero-crossing method for double amplitude calculation ({len(wave_amplitudes)} waves)")
                    double_amplitudes = wave_amplitudes
                elif len(double_amplitudes) > 0:
                    logger.debug(f"Using trough-peak-trough method for double amplitude calculation ({len(double_amplitudes)} waves)")
            
            if len(double_amplitudes) > 0:
                results['maximum_double_amplitude'] = np.max(double_amplitudes)
                
                # Calculate significant double amplitude
                n_waves = len(double_amplitudes)
                sorted_amps = np.sort(double_amplitudes)[::-1]
                n_significant = max(1, int(n_waves * significant_percentile / 100))
                results['sign_double_amplitude'] = np.mean(sorted_amps[:n_significant])
                
                logger.debug(f"Calculated significant double amplitude from {n_significant} highest waves")
    
    # Calculate irregularity factor (STD / mean zero-crossing period)
    if results['mean_zerocross_period'] > 0:
        results['irregularity_factor'] = std_val / results['mean_zerocross_period']
    else:
        results['irregularity_factor'] = np.nan
    
    # Calculate crest factor (max deviation from mean / STD)
    if std_val > 0:
        max_deviation = max(abs(max_val - mean_val), abs(min_val - mean_val))
        results['crest_factor'] = max_deviation / std_val
    else:
        results['crest_factor'] = np.nan
    
    logger.debug(f"Calculated parameters: irregularity_factor={results['irregularity_factor']:.3f}, crest_factor={results['crest_factor']:.3f}")

    # Skip the heavy extreme-value pipeline when not required (e.g. regular waves).
    if not compute_extremes:
        logger.debug(
            f"compute_extremes=False (wave_type='{wave_type}'); "
            "MPM/EEV analysis skipped.")
        return results

    # MPM Analysis using selected method
    if mpm_method == 'STD':
        # Simplified method based on standard deviation (assumes narrow-band process)
        # MPM = mean + sqrt(2) * STD * sqrt(ln(N))
        # where N is the number of waves/cycles
        
        # Calculate number of waves/cycles
        if results['zero_upcross'] > 0:
            N_waves = results['zero_upcross']
        else:
            # Estimate from data length and mean period if available
            if data_duration_hours is not None and results['mean_zerocross_period'] > 0:
                N_waves = int(data_duration_hours * 3600 / results['mean_zerocross_period'])
            else:
                # Fallback: estimate from data length
                N_waves = len(data_scaled) // 100  # Rough estimate
        
        if N_waves > 1:
            # Separate positive and negative data for individual STD calculation
            # Using centered data for proper separation
            positive_data = data_centered[data_centered > 0]
            negative_data = -data_centered[data_centered < 0]  # Convert to positive for STD calculation
            
            # Calculate separate standard deviations for positive and negative data
            std_positive = np.std(positive_data) if len(positive_data) > 0 else std_val
            std_negative = np.std(negative_data) if len(negative_data) > 0 else std_val
            
            logger.debug(f"STD method: std_total={std_val:.3f}, std_positive={std_positive:.3f}, std_negative={std_negative:.3f}")
            
            # Calculate significant amplitudes if peaks are available
            if len(peaks) > 0:
                # Positive peaks
                positive_peaks = data_centered[peaks]
                positive_peaks = positive_peaks[positive_peaks > 0]
                if len(positive_peaks) > 0:
                    sorted_pos_peaks = np.sort(positive_peaks)[::-1]
                    n_significant_pos = max(1, int(len(positive_peaks) * significant_percentile / 100))
                    results['pos_sign_amplitude'] = mean_val + np.mean(sorted_pos_peaks[:n_significant_pos])
                    
            if len(troughs) > 0:
                # Negative peaks
                negative_peaks = data_centered[troughs]
                negative_peaks_abs = -negative_peaks[negative_peaks < 0]
                if len(negative_peaks_abs) > 0:
                    sorted_neg_peaks = np.sort(negative_peaks_abs)[::-1]
                    n_significant_neg = max(1, int(len(negative_peaks_abs) * significant_percentile / 100))
                    results['neg_sign_amplitude'] = mean_val - np.mean(sorted_neg_peaks[:n_significant_neg])
            
            # Calculate MPM using Rice distribution formulas with separate STDs
            # For narrow-band processes, using spectral width parameter epsilon
            # If epsilon = 0 (narrow band), the distribution reduces to Rayleigh
            
            # Estimate spectral width parameter epsilon (for narrow-band assumption, epsilon ≈ 0)
            # This implementation assumes narrow-band process (epsilon = 0)
            
            # Calculate sqrt(ln(N)) for both positive and negative
            sqrt_ln_N = np.sqrt(np.log(N_waves))
            
            # Positive MPM (using positive STD)
            # X_max = mean + sqrt(2 * sigma_x^2) * sqrt(ln(Ne)) [B.80]
            # For narrow band: X_max ≈ mean + sqrt(2) * sigma_x * sqrt(ln(N))
            results['mpm_pos'] = mean_val + np.sqrt(2) * std_positive * sqrt_ln_N
            
            # EEV correction for positive (using positive STD)
            # For STD method, EEV includes a small correction based on Euler-Mascheroni constant
            # EEV ≈ MPM + γ * std / sqrt(2 * ln(N))
            eev_correction_pos = GAMMA * std_positive / np.sqrt(2 * np.log(N_waves))
            results['eev_pos'] = results['mpm_pos'] + eev_correction_pos
            
            # Negative MPM (using negative STD)
            # X_min = mean - sqrt(2 * sigma_x^2) * sqrt(ln(Ne)) [B.81]
            # For narrow band: X_min ≈ mean - sqrt(2) * sigma_x * sqrt(ln(N))
            results['mpm_neg'] = mean_val - np.sqrt(2) * std_negative * sqrt_ln_N
            
            # EEV correction for negative (using negative STD)
            eev_correction_neg = GAMMA * std_negative / np.sqrt(2 * np.log(N_waves))
            results['eev_neg'] = results['mpm_neg'] - eev_correction_neg
            
            logger.debug(f"STD method: N_waves={N_waves}, MPM_pos={results['mpm_pos']:.3f}, EEV_pos={results['eev_pos']:.3f}")
            logger.debug(f"STD method: MPM_neg={results['mpm_neg']:.3f}, EEV_neg={results['eev_neg']:.3f}")
        else:
            logger.warning("STD method: Not enough waves for MPM calculation")
            
    elif mpm_method == 'POT':
        # Peak Over Threshold method with Weibull distribution fitting
        # Following the standard procedure from the image:
        # 1. Extract all peaks (not just those over threshold)
        # 2. Use Weibull plotting position for cumulative probability
        # 3. Fit both Rayleigh and Weibull distributions using linear regression
        # 4. Calculate MPM based on EVD theory
        
        if len(peaks) > 0:
            # Extract positive peaks from centered data
            positive_peaks = data_centered[peaks]
            positive_peaks = positive_peaks[positive_peaks > 0]
            
            # Calculate positive significant amplitude (mean of highest 1/3 peaks)
            if len(positive_peaks) > 0:
                sorted_pos_peaks = np.sort(positive_peaks)[::-1]  # Sort descending
                n_significant_pos = max(1, int(len(positive_peaks) * significant_percentile / 100))
                results['pos_sign_amplitude'] = mean_val + np.mean(sorted_pos_peaks[:n_significant_pos])
                logger.debug(f"Calculated positive significant amplitude from {n_significant_pos} highest peaks: {results['pos_sign_amplitude']:.3f}")
            
            # Apply improved Weibull analysis to all positive peaks
            if len(positive_peaks) >= 10:
                try:
                    # Select only the largest 10% of peaks for extreme value analysis
                    n_extreme = max(int(0.1 * len(positive_peaks)), 10)  # At least 10 peaks
                    sorted_all_peaks = np.sort(positive_peaks)[::-1]  # Sort descending
                    extreme_peaks = sorted_all_peaks[:n_extreme]  # Take largest 10%
                    
                    logger.debug(f"Analyzing largest {n_extreme} positive peaks ({100*n_extreme/len(positive_peaks):.1f}%) for extreme value fitting")
                    
                    # Sort peaks in ascending order for distribution fitting
                    sorted_peaks = np.sort(extreme_peaks)
                    n_peaks = len(sorted_peaks)
                    
                    # Calculate cumulative probability using Weibull plotting position
                    j_values = np.arange(1, n_peaks + 1)
                    P_empirical = j_values / (n_peaks + 1)
                    
                    # 1. Fit Rayleigh distribution
                    valid_idx = P_empirical < 0.999
                    Y_rayleigh = -np.log(1 - P_empirical[valid_idx])
                    X_rayleigh = sorted_peaks[valid_idx]**2
                    
                    a_rayleigh = np.sum(X_rayleigh * Y_rayleigh) / np.sum(X_rayleigh**2)
                    sigma_R = np.sqrt(1 / (2 * a_rayleigh))
                    
                    logger.debug(f"Positive Rayleigh fit (10% largest): σR = {sigma_R:.3f}")
                    
                    # 2. Fit Weibull distribution with location parameter
                    best_r2 = -np.inf
                    best_params = None
                    
                    for mu_factor in [0.0, 0.5, 0.8, 0.9, 0.95]:
                        mu_trial = mu_factor * sorted_peaks[0]
                        
                        if mu_trial >= sorted_peaks[0]:
                            continue
                            
                        Y_weibull = np.log(-np.log(1 - P_empirical[valid_idx]))
                        X_weibull = np.log(sorted_peaks[valid_idx] - mu_trial)
                        
                        k_trial, b_trial = np.polyfit(X_weibull, Y_weibull, 1)
                        
                        y_pred = k_trial * X_weibull + b_trial
                        ss_res = np.sum((Y_weibull - y_pred)**2)
                        ss_tot = np.sum((Y_weibull - np.mean(Y_weibull))**2)
                        r2 = 1 - (ss_res / ss_tot)
                        
                        if r2 > best_r2:
                            best_r2 = r2
                            best_params = (mu_trial, k_trial, b_trial)
                    
                    if best_params is not None:
                        mu_weibull, k_weibull, b_weibull = best_params
                        sigma_w = np.exp(-b_weibull / k_weibull)
                        logger.debug(f"Positive Weibull fit (10% largest): μ = {mu_weibull:.3f}, k = {k_weibull:.3f}, σw = {sigma_w:.3f}, R² = {best_r2:.3f}")
                    else:
                        mu_weibull = 0
                        Y_weibull = np.log(-np.log(1 - P_empirical[valid_idx]))
                        X_weibull = np.log(sorted_peaks[valid_idx])
                        k_weibull, b_weibull = np.polyfit(X_weibull, Y_weibull, 1)
                        sigma_w = np.exp(-b_weibull / k_weibull)
                        logger.debug(f"Positive Weibull fit (10% largest, μ=0): k = {k_weibull:.3f}, σw = {sigma_w:.3f}")
                    
                    # 3. Calculate MPM based on EVD theory
                    # Number of extreme peaks (10% largest)
                    N = len(extreme_peaks)
                    
                    # Find MPM by minimizing negative log PDF
                    # Initial guess: use the approximate formula
                    x0 = mu_weibull + sigma_w * (np.log(N))**(1/k_weibull)
                    
                    try:
                        from scipy.optimize import minimize_scalar
                        # Search in a reasonable range around the initial guess
                        bounds = (mu_weibull + 0.1*(x0-mu_weibull), 
                                 mu_weibull + 3.0*(x0-mu_weibull))
                        
                        result = minimize_scalar(_negative_log_pdf, 
                                               bounds=bounds,
                                               method='bounded',
                                               args=(mu_weibull, sigma_w, k_weibull, N))
                        
                        if result.success:
                            mpm_pos_weibull = result.x - mu_weibull  # Subtract mu to get excess
                            logger.debug(f"  MPM found at x={result.x:.3f} (exact method)")
                        else:
                            # Fallback to approximate formula
                            mpm_pos_weibull = sigma_w * (np.log(N))**(1/k_weibull)
                            logger.debug(f"  MPM calculation failed, using approximate formula")
                    except Exception as e:
                        # Fallback to approximate formula
                        mpm_pos_weibull = sigma_w * (np.log(N))**(1/k_weibull)
                        logger.debug(f"  MPM optimization error: {e}, using approximate formula")
                    
                    # For Rayleigh, the MPM is well-known:
                    # MPM ≈ σR * sqrt(2*ln(N))
                    mpm_pos_rayleigh = sigma_R * np.sqrt(2 * np.log(N))
                    
                    # Use Weibull MPM as the primary result
                    results['mpm_pos'] = mean_val + mu_weibull + mpm_pos_weibull
                    
                    # Calculate EEV (Expected Extreme Value)
                    # For Weibull: EEV includes the Euler-Mascheroni constant correction
                    if k_weibull > 0:
                        eev_correction = sigma_w * GAMMA * (np.log(N))**(1/k_weibull - 1) / k_weibull
                    else:
                        eev_correction = 0
                    results['eev_pos'] = results['mpm_pos'] + eev_correction
                    
                    logger.debug(f"Positive peaks analysis completed (N={N} peaks):")
                    logger.debug(f"  Weibull MPM: {mpm_pos_weibull:.3f}, Total MPM: {results['mpm_pos']:.3f}")
                    logger.debug(f"  Rayleigh MPM: {mpm_pos_rayleigh:.3f}")
                    logger.debug(f"  EEV: {results['eev_pos']:.3f}")
                    
                except Exception as e:
                    logger.warning(f"Weibull/Rayleigh analysis failed for positive peaks: {e}")
            else:
                logger.warning(f"Too few positive peaks (< 10) for Weibull analysis: {len(positive_peaks)} peaks found")
            
            if len(troughs) > 0:
                # Extract negative peaks from centered data and convert to positive values
                negative_peaks = data_centered[troughs]
                negative_peaks = negative_peaks[negative_peaks < 0]
                negative_peaks_abs = -negative_peaks  # Convert to positive for analysis
                
                # Calculate negative significant amplitude (mean of highest 1/3 peaks in absolute value)
                if len(negative_peaks_abs) > 0:
                    sorted_neg_peaks = np.sort(negative_peaks_abs)[::-1]  # Sort descending
                    n_significant_neg = max(1, int(len(negative_peaks_abs) * significant_percentile / 100))
                    results['neg_sign_amplitude'] = mean_val - np.mean(sorted_neg_peaks[:n_significant_neg])
                    logger.debug(f"Calculated negative significant amplitude from {n_significant_neg} highest peaks: {results['neg_sign_amplitude']:.3f}")
                
            # Apply improved Weibull analysis to all negative peaks
            if len(negative_peaks_abs) >= 10:
                try:
                    # Select only the largest 10% of peaks for extreme value analysis
                    n_extreme = max(int(0.1 * len(negative_peaks_abs)), 10)  # At least 10 peaks
                    sorted_all_peaks = np.sort(negative_peaks_abs)[::-1]  # Sort descending
                    extreme_peaks = sorted_all_peaks[:n_extreme]  # Take largest 10%
                    
                    logger.debug(f"Analyzing largest {n_extreme} negative peaks ({100*n_extreme/len(negative_peaks_abs):.1f}%) for extreme value fitting")
                    
                    # Sort peaks in ascending order for distribution fitting
                    sorted_peaks = np.sort(extreme_peaks)
                    n_peaks = len(sorted_peaks)
                    
                    # Calculate cumulative probability using Weibull plotting position
                    j_values = np.arange(1, n_peaks + 1)
                    P_empirical = j_values / (n_peaks + 1)
                    
                    # 1. Fit Rayleigh distribution
                    valid_idx = P_empirical < 0.999
                    Y_rayleigh = -np.log(1 - P_empirical[valid_idx])
                    X_rayleigh = sorted_peaks[valid_idx]**2
                    
                    a_rayleigh = np.sum(X_rayleigh * Y_rayleigh) / np.sum(X_rayleigh**2)
                    sigma_R = np.sqrt(1 / (2 * a_rayleigh))
                    
                    logger.debug(f"Negative Rayleigh fit (10% largest): σR = {sigma_R:.3f}")
                    
                    # 2. Fit Weibull distribution with location parameter
                    best_r2 = -np.inf
                    best_params = None
                    
                    for mu_factor in [0.0, 0.5, 0.8, 0.9, 0.95]:
                        mu_trial = mu_factor * sorted_peaks[0]
                        
                        if mu_trial >= sorted_peaks[0]:
                            continue
                            
                        Y_weibull = np.log(-np.log(1 - P_empirical[valid_idx]))
                        X_weibull = np.log(sorted_peaks[valid_idx] - mu_trial)
                        
                        k_trial, b_trial = np.polyfit(X_weibull, Y_weibull, 1)
                        
                        y_pred = k_trial * X_weibull + b_trial
                        ss_res = np.sum((Y_weibull - y_pred)**2)
                        ss_tot = np.sum((Y_weibull - np.mean(Y_weibull))**2)
                        r2 = 1 - (ss_res / ss_tot)
                        
                        if r2 > best_r2:
                            best_r2 = r2
                            best_params = (mu_trial, k_trial, b_trial)
                    
                    if best_params is not None:
                        mu_weibull, k_weibull, b_weibull = best_params
                        sigma_w = np.exp(-b_weibull / k_weibull)
                        logger.debug(f"Negative Weibull fit (10% largest): μ = {mu_weibull:.3f}, k = {k_weibull:.3f}, σw = {sigma_w:.3f}, R² = {best_r2:.3f}")
                    else:
                        mu_weibull = 0
                        Y_weibull = np.log(-np.log(1 - P_empirical[valid_idx]))
                        X_weibull = np.log(sorted_peaks[valid_idx])
                        k_weibull, b_weibull = np.polyfit(X_weibull, Y_weibull, 1)
                        sigma_w = np.exp(-b_weibull / k_weibull)
                        logger.debug(f"Negative Weibull fit (10% largest, μ=0): k = {k_weibull:.3f}, σw = {sigma_w:.3f}")
                    
                    # 3. Calculate MPM based on EVD theory
                    # Number of extreme peaks (10% largest)
                    N = len(extreme_peaks)
                    
                    # Find MPM by minimizing negative log PDF
                    # Initial guess: use the approximate formula
                    x0 = mu_weibull + sigma_w * (np.log(N))**(1/k_weibull)
                    
                    try:
                        from scipy.optimize import minimize_scalar
                        # Search in a reasonable range around the initial guess
                        bounds = (mu_weibull + 0.1*(x0-mu_weibull), 
                                 mu_weibull + 3.0*(x0-mu_weibull))
                        
                        result = minimize_scalar(_negative_log_pdf, 
                                               bounds=bounds,
                                               method='bounded',
                                               args=(mu_weibull, sigma_w, k_weibull, N))
                        
                        if result.success:
                            mpm_neg_weibull = result.x - mu_weibull  # Subtract mu to get excess
                            logger.debug(f"  MPM found at x={result.x:.3f} (exact method)")
                        else:
                            # Fallback to approximate formula
                            mpm_neg_weibull = sigma_w * (np.log(N))**(1/k_weibull)
                            logger.debug(f"  MPM calculation failed, using approximate formula")
                    except Exception as e:
                        # Fallback to approximate formula
                        mpm_neg_weibull = sigma_w * (np.log(N))**(1/k_weibull)
                        logger.debug(f"  MPM optimization error: {e}, using approximate formula")
                    
                    # For Rayleigh, the MPM is well-known:
                    # MPM ≈ σR * sqrt(2*ln(N))
                    mpm_neg_rayleigh = sigma_R * np.sqrt(2 * np.log(N))
                    
                    # Use Weibull MPM as the primary result (negative direction)
                    results['mpm_neg'] = mean_val - (mu_weibull + mpm_neg_weibull)
                    
                    # Calculate EEV
                    if k_weibull > 0:
                        eev_correction = sigma_w * GAMMA * (np.log(N))**(1/k_weibull - 1) / k_weibull
                    else:
                        eev_correction = 0
                    results['eev_neg'] = mean_val - (mu_weibull + mpm_neg_weibull + eev_correction)
                    
                    logger.debug(f"Negative peaks analysis completed (N={N} peaks):")
                    logger.debug(f"  Weibull MPM: {mpm_neg_weibull:.3f}, Total MPM: {results['mpm_neg']:.3f}")
                    logger.debug(f"  Rayleigh MPM: {mpm_neg_rayleigh:.3f}")
                    logger.debug(f"  EEV: {results['eev_neg']:.3f}")
                    
                except Exception as e:
                    logger.warning(f"Weibull/Rayleigh analysis failed for negative peaks: {e}")
            else:
                logger.warning(f"Too few negative peaks (< 10) for Weibull analysis: {len(negative_peaks_abs)} peaks found")
    else:
        logger.error(f"Unknown MPM method: {mpm_method}. Using default values.")
    
    return results

def _build_results_row(ch_idx, ch_name, ch_unit, ch_results, metric_ids):
    """Build one row of the results DataFrame from a metric-id list."""
    row = [ch_idx, ch_name, ch_unit]
    for mid in metric_ids:
        _, result_key = METRIC_CATALOG[mid]
        row.append(ch_results.get(result_key, np.nan))
    return row


def _format_report_sheet(ws, columns, header_text, sheet_name, results_df):
    """Apply standard Excel formatting to a report sheet (fonts, borders, widths)."""
    # Add header rows
    ws.insert_rows(0, 2)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(columns))
    title_cell = ws.cell(row=1, column=1, value=f"{header_text} - {sheet_name}")

    # Format title
    title_cell.font = Font(bold=True, size=14)
    title_cell.alignment = Alignment(horizontal='center', vertical='center')

    # Define borders
    thick_border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin'),
    )

    # Format data rows
    for row in range(3, ws.max_row + 1):
        for col in range(1, ws.max_column + 1):
            cell = ws.cell(row=row, column=col)
            cell.border = thick_border

            if col == 1:      # ID
                cell.alignment = Alignment(horizontal='center')
            elif col == 2:    # Name
                cell.alignment = Alignment(horizontal='left')
            elif col == 3:    # Unit
                cell.alignment = Alignment(horizontal='center')
            else:             # Numeric values
                cell.alignment = Alignment(horizontal='center')
                if isinstance(cell.value, (int, float)) and col >= 4:
                    if abs(cell.value) < 0.001:
                        cell.value = "0.000"
                    else:
                        cell.value = f"{cell.value:.3f}"

    # Format column headers
    header_row = 3
    header_fill = PatternFill(start_color='E9E9E9', end_color='E9E9E9', fill_type='solid')
    for col in range(1, ws.max_column + 1):
        cell = ws.cell(row=header_row, column=col)
        cell.font = Font(bold=True, size=10)
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        cell.border = thick_border

    # Page layout
    ws.page_setup.orientation = 'landscape'
    ws.page_setup.paperSize = 9  # A4
    ws.page_margins = PageMargins(left=0.5, right=0.5, top=0.5, bottom=0.5)
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.print_area = f'A1:{get_column_letter(ws.max_column)}{ws.max_row}'
    ws.print_title_rows = '1:3'

    # Column widths: fixed for first 3 columns (ID/Name/Unit), data-driven for the rest
    fixed_widths = {1: 6, 2: 20, 3: 8}
    default_width = 12
    for i in range(1, ws.max_column + 1):
        col_letter = get_column_letter(i)
        if i in fixed_widths:
            ws.column_dimensions[col_letter].width = fixed_widths[i]
        else:
            try:
                col_name = columns[i - 1]
                content_width = max(
                    len(str(c)) for c in results_df[col_name].astype(str)
                ) * 1.2
                header_width = max(len(s) for s in str(col_name).split('\n')) * 1.2
                width = max(content_width, header_width, default_width)
                ws.column_dimensions[col_letter].width = min(width, 20)
            except Exception:
                ws.column_dimensions[col_letter].width = default_width
        ws.column_dimensions[col_letter].bestFit = True

    ws.sheet_properties.pageSetUpPr.fitToPage = True


def channel_report(pydas_obj, output_file='channel_report.xlsx', sseg=0, fullscale=True,
                  lam=None, rho=1.025, g=9.807, header_text=None, include_charts=False,
                  significant_percentile=33.0, wave_analysis=True, format_sheet=True,
                  zerocrossing_analysis=True, amplitude_analysis=True,
                  cutoffperiod=15.0, peak_distance=10, pot_threshold_factor=1.5,
                  mpm_method='POT', frequency_separation=False,
                  wave_type='irregular', metrics=None):
    """
    Generate a detailed Excel analysis report for all channels in a PyDAS object,
    with optional high/low frequency separation.

    The report content is now controlled by two layered parameters:

    * ``wave_type`` selects the high-level analysis preset
      (``'irregular'`` for the full ocean-engineering report,
      ``'regular'`` for a lean basic + zero-crossing + STD-amplitude report).
    * ``metrics`` lets you fine-tune the exact set of statistics that appear as
      columns in the report. When *None*, the default set for ``wave_type`` is
      used; pass an explicit list of metric IDs (see :data:`METRIC_CATALOG`)
      to override.

    Parameters
    ----------
    pydas_obj : PyDAS
        PyDAS object containing the data to analyse.
    output_file : str, default='channel_report.xlsx'
        Path to the output Excel file.
    sseg : int, default=0
        Index of the data segment to analyse.
    fullscale : bool, default=True
        Whether to convert data to full (prototype) scale before analysis.
    lam : float, optional
        Scale factor. Used only when ``fullscale=True`` and the PyDAS object has no
        ``__lam__`` attribute.
    rho : float, default=1.025
        Water density in kg/m^3. Used only when ``fullscale=True``.
    g : float, default=9.807
        Gravitational acceleration in m/s^2. Used only when ``fullscale=True``.
    header_text : str, optional
        Title text for the report. Auto-generated from the filename if *None*.
    include_charts : bool, default=False
        Whether to embed charts in the Excel report.
    significant_percentile : float, default=33.0
        Percentile used to compute significant values (e.g. 33 -> top 1/3).
    wave_analysis : bool, default=True
        Whether to perform wave-by-wave analysis.
    format_sheet : bool, default=True
        Whether to apply Excel formatting (fonts, borders, column widths).
    zerocrossing_analysis : bool, default=True
        Whether to perform zero-crossing analysis.
    amplitude_analysis : bool, default=True
        Whether to perform amplitude analysis.
    cutoffperiod : float, default=15.0
        Cut-off period in seconds for separating low- and high-frequency components.
    peak_distance : int, default=10
        Minimum sample distance between peaks used in Weibull peak detection.
    pot_threshold_factor : float, default=1.5
        Threshold coefficient for the POT method.
        Actual threshold = pot_threshold_factor * STD / sqrt(2) (one-sided).
        Typical values: 1.0 (aggressive), 1.5 (moderate), 2.0 (conservative).
    mpm_method : str, default='POT'
        MPM (Most Probable Maximum) calculation method.
        - 'POT': Peak-Over-Threshold with Weibull fit (default; more accurate).
        - 'STD': Simplified STD-based method (assumes narrow-band process).
    frequency_separation : bool, default=False
        If *True*, analyse total, low-frequency (T > cutoffperiod), and high-frequency
        (T < cutoffperiod) components separately. If *False*, analyse total data only.
    wave_type : {'irregular', 'regular'}, default='irregular'
        Analysis preset:
        - 'irregular': full report with peak-based amplitudes and MPM/EEV
          extreme-value estimates.
        - 'regular': basic statistics + zero-crossing + amplitudes derived from
          ``sqrt(2) * STD`` (single) and ``2*sqrt(2) * STD`` (double). MPM/EEV
          and other peak-based metrics are excluded by default and the heavy
          extreme-value pipeline is skipped for performance.
    metrics : list of str, optional
        Explicit list of metric IDs that defines the exact set / order of
        report columns. When *None*, the default set of ``wave_type`` is used.
        See :data:`METRIC_CATALOG` for valid IDs.

    Returns
    -------
    tuple of pandas.DataFrame or pandas.DataFrame
        - If ``frequency_separation=True``: tuple of (total_stats, low_freq_stats,
          high_freq_stats).
        - If ``frequency_separation=False``: total statistics DataFrame only.

    Notes
    -----
    - Default behaviour for ``wave_type='irregular'`` reproduces the original
      19-column report.
    - Setting ``wave_type='regular'`` yields a lean report focused on linear /
      regular wave tests where MPM/EEV are not meaningful.
    - Analysis may take time for large datasets.
    """
    # Validate segment index
    if sseg >= pydas_obj.__segN__:
        logger.error(f"Segment index {sseg} exceeds maximum ({pydas_obj.__segN__ - 1})")
        return None

    # Resolve wave_type and metric set
    wave_type = (wave_type or 'irregular').lower()
    metric_ids = _resolve_metrics(wave_type, metrics)
    if not metric_ids:
        logger.error("No valid metrics resolved; aborting report generation.")
        return None

    # Decide whether to compute the heavy MPM/EEV pipeline:
    # only run it if at least one MPM-dependent metric was requested.
    compute_extremes = bool(set(metric_ids) & _MPM_METRIC_IDS)
    logger.info(
        f"Channel report wave_type='{wave_type}', "
        f"{len(metric_ids)} metric column(s), compute_extremes={compute_extremes}.")

    if fullscale:
        # Resolve scale factor
        if lam is None:
            if hasattr(pydas_obj, '__lam__'):
                lam = pydas_obj.__lam__
            else:
                logger.warning("Scale factor lam not provided and object has no __lam__ attribute")
                return None

        logger.info(f"Creating data copy and converting to full scale (lam={lam}, rho={rho}, g={g})...")
        pydas_analysis = copy.deepcopy(pydas_obj)
        pydas_analysis.to_fullscale(rho=rho, g=g, pInfo=False)
        scale_text = "Full Scale"
    else:
        pydas_analysis = pydas_obj
        scale_text = "Model Scale"

    if header_text is None:
        wave_label = "Regular" if wave_type == 'regular' else "Irregular"
        if hasattr(pydas_obj, 'filename'):
            filename = os.path.basename(pydas_obj.filename)
            header_text = f"{wave_label} wave [{filename}, {scale_text}]"
        else:
            header_text = f"{wave_label} Wave Analysis Report [{scale_text}]"

    # Build dynamic column list: fixed identification columns + selected metrics
    fixed_columns = ['channel\nID', 'Name', 'unit']
    metric_columns = [METRIC_CATALOG[mid][0] for mid in metric_ids]
    columns = fixed_columns + metric_columns

    # Initialise results DataFrame(s)
    results_total = pd.DataFrame(columns=columns)
    if frequency_separation:
        results_low = pd.DataFrame(columns=columns)
        results_high = pd.DataFrame(columns=columns)

    # Retrieve channel info
    ch_info = pydas_analysis.chInfo

    # Compute time step and data duration
    dt = 1.0 / pydas_analysis.__fs__
    data_duration_seconds = len(pydas_analysis.data[sseg]) * dt
    data_duration_hours = data_duration_seconds / 3600

    # Convert cut-off period to angular frequency (rad/s)
    cutoff_freq = 2 * np.pi / cutoffperiod

    # Common kwargs for analyze_channel_data
    common_kwargs = dict(
        zerocrossing_analysis=zerocrossing_analysis,
        amplitude_analysis=amplitude_analysis,
        significant_percentile=significant_percentile,
        data_duration_hours=data_duration_hours,
        dt=dt,
        peak_distance=peak_distance,
        pot_threshold_factor=pot_threshold_factor,
        mpm_method=mpm_method,
        wave_type=wave_type,
        compute_extremes=compute_extremes,
    )

    # Analyse each channel
    for ch_idx, (_, row) in enumerate(ch_info.iterrows(), 1):
        ch_name = row['Name']
        ch_unit = row['Unit']

        # Get scaled channel data
        data_scaled = pydas_analysis.data[sseg][ch_name].values

        # Analyse total (unfiltered) data
        results_total_ch = analyze_channel_data(data_scaled, **common_kwargs)
        results_total.loc[ch_idx] = _build_results_row(
            ch_idx, ch_name, ch_unit, results_total_ch, metric_ids)

        # Frequency separation analysis (only when requested)
        if frequency_separation:
            data_low = pydas_analysis.apply_lowpass_filter(ch_name, cutoff_freq, returnValue=True)
            data_high = pydas_analysis.apply_highpass_filter(ch_name, cutoff_freq, returnValue=True)

            results_low_ch = analyze_channel_data(data_low, **common_kwargs)
            results_high_ch = analyze_channel_data(data_high, **common_kwargs)

            results_low.loc[ch_idx] = _build_results_row(
                ch_idx, ch_name, ch_unit, results_low_ch, metric_ids)
            results_high.loc[ch_idx] = _build_results_row(
                ch_idx, ch_name, ch_unit, results_high_ch, metric_ids)

    # Create Excel file
    if output_file:
        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                results_total.to_excel(writer, sheet_name='Total Statistics', index=False)

                if frequency_separation:
                    results_low.to_excel(writer, sheet_name=f'Low Freq (T>{cutoffperiod}s)', index=False)
                    results_high.to_excel(writer, sheet_name=f'High Freq (T<{cutoffperiod}s)', index=False)

                if format_sheet:
                    sheet_specs = [('Total Statistics', results_total)]
                    if frequency_separation:
                        sheet_specs.append((f'Low Freq (T>{cutoffperiod}s)', results_low))
                        sheet_specs.append((f'High Freq (T<{cutoffperiod}s)', results_high))

                    for sheet_name, df in sheet_specs:
                        ws = writer.sheets[sheet_name]
                        _format_report_sheet(ws, columns, header_text, sheet_name, df)

            logger.info(f"Report successfully exported to {output_file}")

        except Exception as e:
            logger.error(f"Error exporting Excel file: {str(e)}")

    # Return results
    if frequency_separation:
        return results_total, results_low, results_high
    return results_total

def wave_report(pydas_obj, ch_name, sseg=0, save_path=None, title=None, L=1024,
                Hs=None, Tp=None, gamma=None, bins=50, fullscale=False, lam=None, 
                rho=1.025, g=9.807):
    """
    Generate a wave analysis report including time series, spectral analysis, and peak statistics.

    Parameters
    ----------
    pydas_obj : PyDAS
        PyDAS object containing the data.
    ch_name : str
        Name of the channel to analyse.
    sseg : int, optional
        Segment index, default is 0.
    save_path : str, optional
        Path to save the figure. If *None*, the figure is not saved.
    title : str, optional
        Figure title. If *None*, no title is used.
    L : int, optional
        Block length for spectral analysis, default is 1024.
    Hs : float, optional
        Significant wave height for JONSWAP reference spectrum. If *None*, no reference
        spectrum is plotted.
    Tp : float, optional
        Peak period for JONSWAP reference spectrum.
    gamma : float, optional
        Peak enhancement factor for JONSWAP spectrum. Defaults to 3.3 when *None*.
    bins : int, optional
        Number of histogram bins, default is 50.
    fullscale : bool, optional
        Whether to convert to full-scale data, default is *False*.
    lam : float, optional
        Scale factor. Inferred from ``pydas_obj.__lam__`` when *None*.
    rho : float, optional
        Water density in kg/m³, default is 1.025.
    g : float, optional
        Gravitational acceleration in m/s², default is 9.807.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object.
    """
    # Validate channel name
    if ch_name not in pydas_obj.data[sseg]:
        raise ValueError(f"Channel {ch_name} not found in segment {sseg}")
    
    # Load channel data
    data = pydas_obj.data[sseg][ch_name].values
    unit = 'cm'
    T = np.arange(len(data)) / pydas_obj.__fs__

    # Convert to full scale if requested
    if fullscale:
        if lam is None:
            lam = pydas_obj.__lam__
        ts = pydas_obj.channel2fullscale(ch_name, lam, rho, g)
        data = ts.data
        T = ts.args
        unit = 'm'  # Update unit after scaling
    
    # Build the figure layout
    fig = plt.figure(figsize=(10, 11))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])
    
    # 1. Time series plot (full-width top row)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(T/3600, data)
    ax1.set_xlabel('Time (hr)')
    ax1.set_ylabel(f'Amplitude ({unit})')
    ax1.grid(True, axis='x')
    ax1.set_title('Time Series')
    ax1.set_xlim(T[0]/3600, T[-1]/3600)
    if title:
        fig.suptitle(title, y=0.95)
    
    # 2. Spectral analysis (middle row, left)
    ax2 = fig.add_subplot(gs[1, 0])
    spec = pydas_obj.spectral_analysis(ch_name, method='cov', L=L, plot=False, 
                                     fullscale=fullscale, lam=lam, rho=rho, g=g)
    
    # Extract spectral data
    freq = spec.args
    psd = spec.data
    
    # Plot measured spectrum
    ax2.plot(freq, psd, label='Measured')
    
    # Overlay JONSWAP reference spectrum if parameters are provided
    if Hs is not None and Tp is not None:
        from .waveModel.objects import Jonswap
        jonswap_spec = Jonswap(Hs, Tp, gamma=gamma if gamma is not None else 3.3)
        ax2.plot(jonswap_spec.args, jonswap_spec.data, 'r--', label='JONSWAP')

    ax2.set_xlim(0, 2)
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel('Frequency (rad/s)')
    ax2.set_ylabel(f'PSD ({unit}$^2$ s/rad)')
    # ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Wave Spectrum')
    
    # Compute and annotate spectral characteristics
    m0 = np.trapezoid(psd, freq) if hasattr(np, 'trapezoid') else np.trapz(psd, freq)  # Zeroth spectral moment via numerical integration
    Hm0 = 4.0 * np.sqrt(m0)
    text = f'Hm0 = {Hm0:.2f} {unit}'
    ax2.text(0.05, 0.95, text, transform=ax2.transAxes, verticalalignment='top')
    
    # 3. Amplitude histogram with normal fit (middle row, right)
    ax3 = fig.add_subplot(gs[1, 1])
    n, bins, patches = ax3.hist(data, bins=bins, density=True, alpha=0.6)
    
    # Normal distribution fit
    mu, std = np.mean(data), np.std(data)
    x = np.linspace(min(data), max(data), 100)
    p = norm.pdf(x, mu, std)
    ax3.plot(x, p, 'r-', linewidth=2, label=f'Normal (mu={mu:.2f}, sigma={std:.2f})')
    
    ax3.set_xlabel(f'Amplitude ({unit})')
    ax3.set_ylabel('Probability Density')
    ax3.grid(True)
    ax3.set_title('Amplitude Distribution')
    ax3.legend()
    
    # 4. Peak statistics
    peaks, _ = signal.find_peaks(data, prominence=1.0)
    peak_values = data[peaks]
    
    # Peak histogram
    ax4 = fig.add_subplot(gs[2, 0])
    n_peaks, bins_peaks, patches_peaks = ax4.hist(peak_values, bins=bins, density=True, alpha=0.6)
    
    # Weibull fit for peaks
    shape, loc, scale = spstats.weibull_min.fit(peak_values, floc=0)
    x_peaks = np.linspace(0, max(peak_values), 100)
    p_peaks = spstats.weibull_min.pdf(x_peaks, shape, loc, scale)
    ax4.plot(x_peaks, p_peaks, 'r-', linewidth=2, 
             label=f'Weibull (k={shape:.2f}, lambda={scale:.2f})')
    
    ax4.set_xlabel(f'Peak Amplitude ({unit})')
    ax4.set_ylabel('Probability Density')
    ax4.grid(True)
    ax4.set_title('Peak Distribution')
    ax4.set_xlim(left=0)
    ax4.legend()
    
    # Display statistics summary
    text = f'Peak Count: {len(peaks)}\nMean: {np.mean(peak_values):.2f}\nMax: {max(peak_values):.2f}\nMin: {min(peak_values):.2f}'
    ax4.text(0.05, 0.95, text, transform=ax4.transAxes, verticalalignment='top')
    
    # Q-Q plot
    ax5 = fig.add_subplot(gs[2, 1])
    spstats.probplot(peak_values, dist="norm", plot=ax5)
    ax5.set_title('Q-Q Plot of Peaks')
    ax5.grid(True)
    
    plt.tight_layout()
    
    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig