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
from waveModel.specmodels import Jonswap
import matplotlib.gridspec as gridspec
from scipy.stats import norm

# Set up logging
logger = logging.getLogger('pydas.reporting')

# Euler-Mascheroni constant for expected extreme value calculation
GAMMA = 0.57721566

def analyze_channel_data(data_scaled, mean_val=None, std_val=None, zerocrossing_analysis=True, 
                      amplitude_analysis=True, significant_percentile=33.0,
                      data_duration_hours=None, dt=None, peak_distance=130, 
                      pot_threshold_factor=1.5, mpm_method='POT'):
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
        峰值超阈值（POT）方法的阈值系数
        实际阈值 = pot_threshold_factor × 标准差 / √2（考虑单侧分布）
        常用值：1.0（激进）、1.5（适中）、2.0（保守）
    mpm_method : str, default='POT'
        MPM（最可能最大值）的计算方法
        - 'POT': 峰值超阈值方法，使用Weibull分布拟合（默认，更精确但计算复杂）
        - 'STD': 基于标准差的简化方法（假设窄带过程，适用于线性波浪）
        
    Returns
    -------
    dict
        Dictionary containing all statistical results including:
        - Basic statistics: maximum, minimum, mean, STD
        - Wave parameters: maximum_double_amplitude, sign_double_amplitude, mean_zerocross_period, zero_upcross
        - Extreme values: MPM (Most Probable Maximum) and EEV (Expected Extreme Value) for positive and negative peaks
        - Ocean engineering parameters: irregularity_factor, crest_factor
    """
    
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
                    
                    # For Weibull distribution, MPM is the value where PDF of EVD reaches maximum
                    def weibull_evd_pdf(x, mu, sigma, k, n):
                        """Calculate the PDF of extreme value distribution for Weibull"""
                        if x <= mu:
                            return 0.0
                        u = (x - mu) / sigma
                        if u <= 0:
                            return 0.0
                        
                        # Components of the PDF
                        cdf_base = 1 - np.exp(-u**k)
                        pdf_base = k/sigma * u**(k-1) * np.exp(-u**k)
                        
                        # Complete PDF
                        if cdf_base <= 0 or cdf_base >= 1:
                            return 0.0
                        
                        pdf = n * cdf_base**(n-1) * pdf_base
                        return pdf
                    
                    def negative_log_pdf(x, mu, sigma, k, n):
                        """Negative log PDF for optimization (to find maximum)"""
                        pdf = weibull_evd_pdf(x, mu, sigma, k, n)
                        if pdf <= 0:
                            return 1e10
                        return -np.log(pdf)
                    
                    # Find MPM by minimizing negative log PDF
                    # Initial guess: use the approximate formula
                    x0 = mu_weibull + sigma_w * (np.log(N))**(1/k_weibull)
                    
                    try:
                        from scipy.optimize import minimize_scalar
                        # Search in a reasonable range around the initial guess
                        bounds = (mu_weibull + 0.1*(x0-mu_weibull), 
                                 mu_weibull + 3.0*(x0-mu_weibull))
                        
                        result = minimize_scalar(negative_log_pdf, 
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
                    
                    # For Weibull distribution, MPM is the value where PDF of EVD reaches maximum
                    def weibull_evd_pdf(x, mu, sigma, k, n):
                        """Calculate the PDF of extreme value distribution for Weibull"""
                        if x <= mu:
                            return 0.0
                        u = (x - mu) / sigma
                        if u <= 0:
                            return 0.0
                        
                        # Components of the PDF
                        cdf_base = 1 - np.exp(-u**k)
                        pdf_base = k/sigma * u**(k-1) * np.exp(-u**k)
                        
                        # Complete PDF
                        if cdf_base <= 0 or cdf_base >= 1:
                            return 0.0
                        
                        pdf = n * cdf_base**(n-1) * pdf_base
                        return pdf
                    
                    def negative_log_pdf(x, mu, sigma, k, n):
                        """Negative log PDF for optimization (to find maximum)"""
                        pdf = weibull_evd_pdf(x, mu, sigma, k, n)
                        if pdf <= 0:
                            return 1e10
                        return -np.log(pdf)
                    
                    # Find MPM by minimizing negative log PDF
                    # Initial guess: use the approximate formula
                    x0 = mu_weibull + sigma_w * (np.log(N))**(1/k_weibull)
                    
                    try:
                        from scipy.optimize import minimize_scalar
                        # Search in a reasonable range around the initial guess
                        bounds = (mu_weibull + 0.1*(x0-mu_weibull), 
                                 mu_weibull + 3.0*(x0-mu_weibull))
                        
                        result = minimize_scalar(negative_log_pdf, 
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

def channel_report(pydas_obj, output_file='channel_report.xlsx', sseg=0, fullscale=True, 
                  lam=None, rho=1.025, g=9.807, header_text=None, include_charts=False, 
                  significant_percentile=33.0, wave_analysis=True, format_sheet=True, 
                  zerocrossing_analysis=True, amplitude_analysis=True, 
                  cutoffperiod=15.0, peak_distance=10, pot_threshold_factor=1.5,
                  mpm_method='POT', frequency_separation=False):
    """
    为PyDAS对象的所有通道生成详细的Excel分析报告，包括高低频分离分析
    
    Parameters
    ----------
    pydas_obj : PyDAS
        PyDAS对象，包含要分析的数据
    output_file : str, default='channel_report.xlsx'
        输出Excel文件的路径
    sseg : int, default=0
        要分析的数据段索引
    fullscale : bool, default=True
        是否使用实际尺度（原型尺度）值
    lam : float, optional
        尺度因子，仅在fullscale=True且PyDAS对象未设置__lam__属性时使用
    rho : float, default=1.025
        水密度 (kg/m³)，仅用于fullscale=True时
    g : float, default=9.807
        重力加速度 (m/s²)，仅用于fullscale=True时
    header_text : str, optional
        报告中的标题文本
    include_charts : bool, default=False
        是否在报告中包含图表
    significant_percentile : float, default=33.0
        计算显著值的百分位数
    wave_analysis : bool, default=True
        是否进行波浪分析
    format_sheet : bool, default=True
        是否设置Excel格式
    zerocrossing_analysis : bool, default=True
        是否进行过零分析
    amplitude_analysis : bool, default=True
        是否进行振幅分析
    cutoffperiod : float, default=15.0
        高低频分离的截止周期（秒），用于分离高频和低频成分
    peak_distance : int, default=130
        峰值检测的最小距离参数，用于 Weibull 分析中的峰值检测
    pot_threshold_factor : float, default=1.5
        峰值超阈值（POT）方法的阈值系数
        实际阈值 = pot_threshold_factor × 标准差 / √2（考虑单侧分布）
        常用值：1.0（激进）、1.5（适中）、2.0（保守）
    mpm_method : str, default='POT'
        MPM（最可能最大值）的计算方法
        - 'POT': 峰值超阈值方法，使用Weibull分布拟合（默认，更精确但计算复杂）
        - 'STD': 基于标准差的简化方法（假设窄带过程，适用于线性波浪）
    frequency_separation : bool, default=False
        是否进行高低频分离分析。如果为True，将分别分析总体、低频（T>cutoffperiod）
        和高频（T<cutoffperiod）成分；如果为False，只分析总体数据
        
    Returns
    -------
    tuple of pandas.DataFrame or pandas.DataFrame
        - 如果frequency_separation=True：返回包含三个DataFrame的元组(总统计, 低频统计, 高频统计)
        - 如果frequency_separation=False：只返回总统计的DataFrame
        
    Notes
    -----
    - 生成一个包含所有通道统计分析的Excel报告
    - 报告包括基本统计值、过零分析、振幅分析和极值估计
    - 可以选择是否使用实际尺度值（原型尺度）
    - 分析可能需要一些时间，特别是对于大型数据集
    """
    # 检查输入
    if sseg >= pydas_obj.__segN__:
        logger.error(f"段索引 {sseg} 超出最大段数 ({pydas_obj.__segN__-1})")
        return None

    if fullscale:
        # 确定尺度因子
        if lam is None:
            if hasattr(pydas_obj, '__lam__'):
                lam = pydas_obj.__lam__
            else:
                logger.warning("Scale factor lam not provided and object has no __lam__ attribute")
                return None
        
        logger.info(f"Creating data copy and converting to full scale (λ={lam}, ρ={rho}, g={g})...")
        pydas_analysis = copy.deepcopy(pydas_obj)
        pydas_analysis.to_fullscale(rho=rho, g=g, pInfo=False)
        scale_text = "Full Scale"
    else:
        pydas_analysis = pydas_obj
        scale_text = "Model Scale"
    
    if header_text is None:
        if hasattr(pydas_obj, 'filename'):
            filename = os.path.basename(pydas_obj.filename)
            header_text = f"Wave only [{filename}, {scale_text}]"
        else:
            header_text = f"Wave Analysis Report [{scale_text}]"
    
    # 创建结果DataFrame的列
    columns = [
        'channel\nID', 'Name', 'unit', 'number\nof zero\nupcross', 
        'maximum', 'minimum', 'mean', 'STD',
        'maximum\ndouble\namplitude', 'sign.\ndouble\namplitude', 
        'Pos. sign.\namplitude', 'Neg. sign.\namplitude',
        'MPM_pos', 'MPM_neg', 'EEV_pos', 'EEV_neg', 'irregularity\nfactor', 'crest\nfactor',
        'mean\nzerocro.\nperiod'
    ]
    
    # 创建结果DataFrame
    results_total = pd.DataFrame(columns=columns)
    
    # 只在需要时创建高低频结果DataFrame
    if frequency_separation:
        results_low = pd.DataFrame(columns=columns)
        results_high = pd.DataFrame(columns=columns)
    
    # 获取通道信息
    ch_info = pydas_analysis.chInfo
    
    # 计算分析周期
    dt = 1.0 / pydas_analysis.__fs__
    data_duration_seconds = len(pydas_analysis.data[sseg]) * dt
    data_duration_hours = data_duration_seconds / 3600
    
    # 计算截止频率（从周期转换为rad/s）
    cutoff_freq = 2 * np.pi / cutoffperiod
    
    # 对每个通道进行分析
    for ch_idx, (_, row) in enumerate(ch_info.iterrows(), 1):
        ch_name = row['Name']
        ch_unit = row['Unit']
        
        # 获取原始数据
        data_scaled = pydas_analysis.data[sseg][ch_name].values
        
        # 分析总体数据
        results_total_ch = analyze_channel_data(
            data_scaled, zerocrossing_analysis=zerocrossing_analysis,
            amplitude_analysis=amplitude_analysis, significant_percentile=significant_percentile,
            data_duration_hours=data_duration_hours, dt=dt, peak_distance=peak_distance,
            pot_threshold_factor=pot_threshold_factor, mpm_method=mpm_method
        )
        
        # 将结果添加到总体DataFrame
        results_total.loc[ch_idx] = [
            ch_idx, ch_name, ch_unit, results_total_ch['zero_upcross'],
            results_total_ch['maximum'], results_total_ch['minimum'],
            results_total_ch['mean'], results_total_ch['STD'],
            results_total_ch['maximum_double_amplitude'],
            results_total_ch['sign_double_amplitude'],
            results_total_ch['pos_sign_amplitude'], results_total_ch['neg_sign_amplitude'],
            results_total_ch['mpm_pos'], results_total_ch['mpm_neg'],
            results_total_ch['eev_pos'], results_total_ch['eev_neg'],
            results_total_ch['irregularity_factor'], results_total_ch['crest_factor'],
            results_total_ch['mean_zerocross_period']
        ]
        
        # 只在需要时进行高低频分离分析
        if frequency_separation:
            # 分离高低频
            data_low = pydas_analysis.apply_lowpass_filter(ch_name, cutoff_freq, returnValue=True)
            data_high = pydas_analysis.apply_highpass_filter(ch_name, cutoff_freq, returnValue=True)
            
            # 分析低频数据
            results_low_ch = analyze_channel_data(
                data_low, zerocrossing_analysis=zerocrossing_analysis,
                amplitude_analysis=amplitude_analysis, significant_percentile=significant_percentile,
                data_duration_hours=data_duration_hours, dt=dt, peak_distance=peak_distance,
                pot_threshold_factor=pot_threshold_factor, mpm_method=mpm_method
            )
            
            # 分析高频数据
            results_high_ch = analyze_channel_data(
                data_high, zerocrossing_analysis=zerocrossing_analysis,
                amplitude_analysis=amplitude_analysis, significant_percentile=significant_percentile,
                data_duration_hours=data_duration_hours, dt=dt, peak_distance=peak_distance,
                pot_threshold_factor=pot_threshold_factor, mpm_method=mpm_method
            )
            
            # 将结果添加到相应的DataFrame
            results_low.loc[ch_idx] = [
                ch_idx, ch_name, ch_unit, results_low_ch['zero_upcross'],
                results_low_ch['maximum'], results_low_ch['minimum'],
                results_low_ch['mean'], results_low_ch['STD'],
                results_low_ch['maximum_double_amplitude'],
                results_low_ch['sign_double_amplitude'],
                results_low_ch['pos_sign_amplitude'], results_low_ch['neg_sign_amplitude'],
                results_low_ch['mpm_pos'], results_low_ch['mpm_neg'],
                results_low_ch['eev_pos'], results_low_ch['eev_neg'],
                results_low_ch['irregularity_factor'], results_low_ch['crest_factor'],
                results_low_ch['mean_zerocross_period']
            ]
            
            results_high.loc[ch_idx] = [
                ch_idx, ch_name, ch_unit, results_high_ch['zero_upcross'],
                results_high_ch['maximum'], results_high_ch['minimum'],
                results_high_ch['mean'], results_high_ch['STD'],
                results_high_ch['maximum_double_amplitude'],
                results_high_ch['sign_double_amplitude'],
                results_high_ch['pos_sign_amplitude'], results_high_ch['neg_sign_amplitude'],
                results_high_ch['mpm_pos'], results_high_ch['mpm_neg'],
                results_high_ch['eev_pos'], results_high_ch['eev_neg'],
                results_high_ch['irregularity_factor'], results_high_ch['crest_factor'],
                results_high_ch['mean_zerocross_period']
            ]

    # 创建Excel文件
    if output_file:
        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # 总是写入总体统计
                results_total.to_excel(writer, sheet_name='Total Statistics', index=False)
                
                # 只在进行频率分离时写入高低频统计
                if frequency_separation:
                    results_low.to_excel(writer, sheet_name=f'Low Freq (T>{cutoffperiod}s)', index=False)
                    results_high.to_excel(writer, sheet_name=f'High Freq (T<{cutoffperiod}s)', index=False)
                
                if format_sheet:
                    # 确定要格式化的表名列表
                    sheet_names = ['Total Statistics']
                    if frequency_separation:
                        sheet_names.extend([f'Low Freq (T>{cutoffperiod}s)', f'High Freq (T<{cutoffperiod}s)'])
                    
                    # 格式化每个表
                    for sheet_name in sheet_names:
                        ws = writer.sheets[sheet_name]
                        
                        # 添加标题行
                        ws.insert_rows(0, 2)
                        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(columns))
                        title_cell = ws.cell(row=1, column=1, value=f"{header_text} - {sheet_name}")
                        
                        # 设置标题行格式
                        title_cell.font = Font(bold=True, size=14)
                        title_cell.alignment = Alignment(horizontal='center', vertical='center')
                        
                        # 设置列宽
                        for i, column in enumerate(columns, 1):
                            col_width = max(len(str(c)) for c in results_total[column].astype(str)) * 1.2
                            col_width = max(col_width, len(column) * 1.2)
                            ws.column_dimensions[get_column_letter(i)].width = min(col_width, 20)
                        
                        # 添加粗边框样式
                        thick_border = Border(
                            left=Side(style='thin'), 
                            right=Side(style='thin'),
                            top=Side(style='thin'),
                            bottom=Side(style='thin')
                        )
                        
                        # 设置数据行格式
                        for row in range(3, ws.max_row + 1):
                            for col in range(1, ws.max_column + 1):
                                cell = ws.cell(row=row, column=col)
                                
                                # 应用边框
                                cell.border = thick_border
                                
                                # 对齐
                                if col == 1:  # 通道编号
                                    cell.alignment = Alignment(horizontal='center')
                                elif col in [2]:  # 名称
                                    cell.alignment = Alignment(horizontal='left')
                                else:  # 数值
                                    cell.alignment = Alignment(horizontal='center')
                                    
                                    # 格式化数值
                                    if isinstance(cell.value, (int, float)) and col >= 4:
                                        if abs(cell.value) < 0.001:
                                            # 对于接近0的值直接显示0.000
                                            cell.value = "0.000"
                                        else:
                                            # 普通数字，3位小数
                                            cell.value = f"{cell.value:.3f}"
                        
                        # 设置标题行格式
                        header_row = 3
                        header_fill = PatternFill(start_color='E9E9E9', end_color='E9E9E9', fill_type='solid')
                        
                        for col in range(1, ws.max_column + 1):
                            cell = ws.cell(row=header_row, column=col)
                            cell.font = Font(bold=True, size=10)
                            cell.fill = header_fill
                            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
                            cell.border = thick_border
                        
                        # 设置页面布局
                        ws.page_setup.orientation = 'landscape'
                        ws.page_setup.paperSize = 9  # A4
                        ws.page_margins = PageMargins(left=0.5, right=0.5, top=0.5, bottom=0.5)
                        ws.page_setup.fitToWidth = 1
                        ws.page_setup.fitToHeight = 0
                        ws.print_area = f'A1:{get_column_letter(ws.max_column)}{ws.max_row}'
                        ws.print_title_rows = '1:3'
                        
                        # 设置列宽
                        col_width_map = {
                            1: 6,   # 通道编号
                            2: 20,  # 通道名称
                            3: 8    # 单位
                        }
                        
                        default_width = 12
                        special_widths = {
                            4: 10,  # 零上穿数
                            9: 15,  # 最大双振幅
                            10: 15, # 显著双振幅
                            11: 12, # Pos. sign. amplitude
                            12: 12, # Neg. sign. amplitude
                            13: 12, # MPM_pos
                            14: 12, # MPM_neg
                            15: 10, # EEV_pos
                            16: 10, # EEV_neg
                            17: 12, # irregularity factor
                            18: 10, # crest factor
                            19: 12  # 平均零上穿周期
                        }
                        
                        for i in range(1, ws.max_column + 1):
                            col_letter = get_column_letter(i)
                            if i in col_width_map:
                                ws.column_dimensions[col_letter].width = col_width_map[i]
                            elif i in special_widths:
                                ws.column_dimensions[col_letter].width = special_widths[i]
                            else:
                                ws.column_dimensions[col_letter].width = default_width
                            
                            ws.column_dimensions[col_letter].bestFit = True
                        
                        ws.sheet_properties.pageSetUpPr.fitToPage = True
            
            logger.info(f"Report successfully exported to {output_file}")
        
        except Exception as e:
            logger.error(f"Error exporting Excel file: {str(e)}")
    
    # 返回结果
    if frequency_separation:
        # 返回所有三个DataFrame
        return results_total, results_low, results_high
    else:
        # 只返回总体统计DataFrame
        return results_total

def wave_report(pydas_obj, ch_name, sseg=0, save_path=None, title=None, L=1024,
                Hs=None, Tp=None, gamma=None, bins=50, fullscale=False, lam=None, 
                rho=1.025, g=9.807):
    """
    生成波浪分析报告，包括时间序列、谱分析和峰值统计
    
    Parameters
    ----------
    pydas_obj : PyDAS object
        PyDAS对象
    ch_name : str
        要分析的通道名称
    sseg : int, optional
        数据段索引，默认为0
    save_path : str, optional
        保存图片的路径，默认为None
    title : str, optional
        图表标题，默认为None
    L : int, optional
        谱分析的数据块长度，默认为1024
    Hs : float, optional
        JONSWAP谱的有效波高，默认为None
    Tp : float, optional
        JONSWAP谱的峰值周期，默认为None
    gamma : float, optional
        JONSWAP谱的峰值增强因子，默认为None
    bins : int, optional
        直方图的bin数量，默认为50
    fullscale : bool, optional
        是否使用实际尺度数据，默认为False
    lam : float, optional
        尺度因子，默认为None
    rho : float, optional
        水密度 (kg/m3)，默认为1.025
    g : float, optional
        重力加速度 (m/s2)，默认为9.807
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        生成的图表对象
    """
    # 检查通道是否存在
    if ch_name not in pydas_obj.data[sseg]:
        raise ValueError(f"Channel {ch_name} not found in segment {sseg}")
    
    # 获取数据
    data = pydas_obj.data[sseg][ch_name].values
    unit = 'cm'
    T = np.arange(len(data)) / pydas_obj.__fs__

    # 如果需要转换为实际尺度
    if fullscale:
        if lam is None:
            lam = pydas_obj.__lam__
        ts = pydas_obj.channel2fullscale(ch_name, lam, rho, g)
        data = ts.data
        T = ts.args
        unit = 'm'  # 更新单位
    
    # 创建图表
    fig = plt.figure(figsize=(10, 11))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])
    
    # 1. 时间序列图（第一行，全宽）
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(T/3600, data)
    ax1.set_xlabel('Time (hr)')
    ax1.set_ylabel(f'Amplitude ({unit})')
    ax1.grid(True, axis='x')
    ax1.set_title('Time Series')
    ax1.set_xlim(T[0]/3600, T[-1]/3600)
    if title:
        fig.suptitle(title, y=0.95)
    
    # 2. 谱分析（第二行，左）
    ax2 = fig.add_subplot(gs[1, 0])
    spec = pydas_obj.spectral_analysis(ch_name, method='cov', L=L, plot=False, 
                                     fullscale=fullscale, lam=lam, rho=rho, g=g)
    
    # 获取谱数据
    freq = spec.args
    psd = spec.data
    
    # 绘制测量谱
    ax2.plot(freq, psd, label='Measured')
    
    # 如果提供了JONSWAP参数，绘制理论谱
    if Hs is not None and Tp is not None:
        from waveModel.specmodels import Jonswap
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
    
    # 计算并显示谱特征
    m0 = np.trapz(psd, freq)  # 使用数值积分计算零阶矩
    Hm0 = 4.0 * np.sqrt(m0)
    text = f'Hm0 = {Hm0:.2f} {unit}'
    ax2.text(0.05, 0.95, text, transform=ax2.transAxes, verticalalignment='top')
    
    # 3. 直方图和正态拟合（第二行，右）
    ax3 = fig.add_subplot(gs[1, 1])
    n, bins, patches = ax3.hist(data, bins=bins, density=True, alpha=0.6)
    
    # 拟合正态分布
    mu, std = np.mean(data), np.std(data)
    x = np.linspace(min(data), max(data), 100)
    p = norm.pdf(x, mu, std)
    ax3.plot(x, p, 'r-', linewidth=2, label=f'Normal (μ={mu:.2f}, σ={std:.2f})')
    
    ax3.set_xlabel(f'Amplitude ({unit})')
    ax3.set_ylabel('Probability Density')
    ax3.grid(True)
    ax3.set_title('Amplitude Distribution')
    ax3.legend()
    
    # 4. 峰值统计（第三行）
    # 检测峰值
    peaks, _ = signal.find_peaks(data, prominence=1.0)
    peak_values = data[peaks]
    
    # 峰值直方图（第三行，左）
    ax4 = fig.add_subplot(gs[2, 0])
    n_peaks, bins_peaks, patches_peaks = ax4.hist(peak_values, bins=bins, density=True, alpha=0.6)
    
    # 拟合Weibull分布
    shape, loc, scale = spstats.weibull_min.fit(peak_values, floc=0)
    x_peaks = np.linspace(0, max(peak_values), 100)
    p_peaks = spstats.weibull_min.pdf(x_peaks, shape, loc, scale)
    ax4.plot(x_peaks, p_peaks, 'r-', linewidth=2, 
             label=f'Weibull (k={shape:.2f}, λ={scale:.2f})')
    
    ax4.set_xlabel(f'Peak Amplitude ({unit})')
    ax4.set_ylabel('Probability Density')
    ax4.grid(True)
    ax4.set_title('Peak Distribution')
    ax4.set_xlim(left=0)
    ax4.legend()
    
    # 显示峰值统计
    text = f'Peak Count: {len(peaks)}\nMean: {np.mean(peak_values):.2f}\nMax: {max(peak_values):.2f}\nMin: {min(peak_values):.2f}'
    ax4.text(0.05, 0.95, text, transform=ax4.transAxes, verticalalignment='top')
    
    # Q-Q图（第三行，右）
    ax5 = fig.add_subplot(gs[2, 1])
    spstats.probplot(peak_values, dist="norm", plot=ax5)
    ax5.set_title('Q-Q Plot of Peaks')
    ax5.grid(True)
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig