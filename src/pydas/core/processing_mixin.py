"""PyDAS Core - Processing Mixin"""
import datetime
import numpy as np
import pandas as pd
import logging

from scipy.signal import correlate

from ..process import (
    apply_lowpass_filter, apply_highpass_filter, remove_mean, 
    add_value, multiply_value, move_data, data_wash, add_diff1, add_diff2
)
from ..utils import get_default_transDict, findtrans
from ..waveModel.objects import TimeSeries

logger = logging.getLogger(__name__)

class ProcessingMixin:
    def fix_unit(self, chName, newunit, pInfo=False):
        """
        Fix channel unit.
        
        Parameters:
        -----------
        chName : str
            Channel name
        newunit : str
            New unit to set
        pInfo : bool, optional
            Whether to print information, default is False
            
        Notes:
        ------
        - Updates channel unit in information
        - Validates unit conversion
        - Maintains data integrity
        """
        # Validate channel name
        if chName not in self.chInfo['Name'].values:
            logger.warning(f"Channel '{chName}' does not exist.")
            return None
            
        # Locate the row index for this channel
        idx = self.chInfo.index[self.chInfo['Name'] == chName].tolist()[0]
        
        # Update the unit field
        self.chInfo.loc[idx, 'Unit'] = newunit
        logger.info(f"Channel '{chName}' unit updated to: {newunit}")
        
        if pInfo:
            logger.info('-' * 50)
            logger.info('\n' + self.chInfo.to_string(justify='center'))
            logger.info('-' * 50)

        return None

    def to_fullscale(self, rho=1.025, g=9.807, pInfo=False):
        """
        Convert model scale data to prototype scale.
        
        Parameters
        ----------
        rho : float, optional
            Water density in kg/m³, default is 1.025.
        g : float, optional
            Gravitational acceleration in m/s², default is 9.807.
        pInfo : bool, optional
            Whether to print information, default is False.

        Notes
        -----
        Uses ``self.__lam__`` as the length scale factor. There is no ``lam``
        argument; set ``obj.__lam__`` before calling this method.
            
        Notes:
        ------
        - Applies Froude scaling laws
        - Handles unit conversions
        - Updates channel information
        - Maintains data consistency
        """
        if self.__scale__ == 'prototype':
            logger.warning('The data is already upscaled.')
            return None
        else:
            logger.info('Please make sure the channel units are all checked!')
            if pInfo:
                logger.info(self.chInfo.to_string(
                    justify='center', columns=['Name', 'Unit']))
            self.rho = rho
            self.__scale__ = 'prototype'
            
            # Get unit conversion dictionary using utility function
            transDict = get_default_transDict(g)
            
            # Clear global unit-conversion cache to ensure a fresh calculation
            # Use a safe non-empty string instead of an empty string
            findtrans('none', transDict, clear_cache=True)
            
            # Preprocess: batch get all unit conversions
            unique_units = self.chInfo['Unit'].unique()
            unit_to_trans = {}
            
            # Parallel precompute all unique unit conversions
            for unit in unique_units:
                if not pd.isna(unit):  # Handle potential NaN values
                    unit_to_trans[unit] = findtrans(unit, transDict)
            
            # Prepare batch update data
            transUnit = []
            transCoeffUnit = np.zeros(self.__chN__)
            transCoeffRho = np.zeros(self.__chN__)
            transCoeffLam = np.zeros(self.__chN__)
            
            # Use vectorized operations to update channel information
            for idx, unit in enumerate(self.chInfo['Unit']):
                if pd.isna(unit):  # Handle potential NaN values
                    transUnit.append('')
                    transCoeffUnit[idx] = 1.0
                    transCoeffRho[idx] = 0.0
                    transCoeffLam[idx] = 0.0
                    continue
                    
                trans_temp = unit_to_trans[unit]
                transUnit.append(trans_temp[0])
                transCoeffUnit[idx] = trans_temp[1][0]
                transCoeffRho[idx] = trans_temp[1][1]
                transCoeffLam[idx] = trans_temp[1][2]
            
            # Update channel information
            self.chInfo['Unit'] = transUnit
            self.chInfo['CoeffUnit'] = transCoeffUnit
            self.chInfo['CoeffRho'] = transCoeffRho
            self.chInfo['CoeffLam'] = transCoeffLam
            
            # Update sampling rate
            self.__fs__ = self.__fs__ / np.sqrt(self.__lam__)
            logger.info(f'lambda = {self.__lam__:.2f}')
            
            if pInfo:
                logger.info(self.chInfo.to_string(justify='center'))

            # Use vectorized operations to update data (process all channels at once)
            for idx1 in range(self.__segN__):
                # Create conversion coefficient array
                coeffs = np.ones(self.__chN__)
                
                # Calculate conversion coefficient for each channel
                for idx2 in range(self.__chN__):
                    C1 = self.chInfo['CoeffUnit'].iloc[idx2]
                    C2 = rho ** self.chInfo['CoeffRho'].iloc[idx2]
                    C3 = self.__lam__ ** self.chInfo['CoeffLam'].iloc[idx2]
                    coeffs[idx2] = C1 * C2 * C3
                
                # Apply conversion coefficients (vectorized operation)
                for idx2, name in enumerate(self.chInfo['Name']):
                    self.data[idx1][name] *= coeffs[idx2]
            
            # Update statistical information
            self.updateST(sseg=0)

        return None

    def move_ccor(self,
                  to_move_chName,
                  base_chName,
                  reference_ch,
                  sseg=0):
        """
        Move channel data using cross-correlation.
        
        Parameters:
        -----------
        to_move_chName : str
            Name of channel to move
        base_chName : str
            Name of base channel
        reference_ch : str
            Name of reference channel
        sseg : int, optional
            Segment index to process, default is 0
            
        Notes:
        ------
        - Uses cross-correlation for alignment
        - Handles data shifting
        - Maintains data quality
        - Updates channel information
        """
        try:
            # Check if channels exist
            if to_move_chName not in self.data[sseg].columns:
                logger.error(f"Channel to move '{to_move_chName}' not found in segment {sseg}")
                raise KeyError(f"Channel to move '{to_move_chName}' not found")
                
            if reference_ch not in self.data[sseg].columns:
                logger.error(f"Reference channel '{reference_ch}' not found in segment {sseg}")
                raise KeyError(f"Reference channel '{reference_ch}' not found")
                
            if base_chName not in self.data[sseg].columns:
                logger.error(f"Base channel '{base_chName}' not found in segment {sseg}")
                raise KeyError(f"Base channel '{base_chName}' not found")
            
            # Get channel data
            reference = self.data[sseg][reference_ch]
            base = self.data[sseg][base_chName]
            
            # Remove mean from signals for better correlation
            try:
                base_rmmean = base.values - np.mean(base.values)
                reference_remean = reference.values - np.mean(reference.values)
            except Exception as e:
                logger.error(f"Error removing mean from signals: {str(e)}")
                raise ValueError(f"Failed to prepare signals for correlation: {str(e)}")
            
            # Get number of samples
            try:
                n_sample = self.segInfo['N sample'].iloc[sseg]
            except Exception as e:
                logger.error(f"Error getting sample count: {str(e)}")
                raise ValueError(f"Failed to get sample count: {str(e)}")
            
            # Calculate cross-correlation and find lag
            try:
                correlation = correlate(base_rmmean, reference_remean, method='fft')
                max_corr_idx = np.argmax(correlation)
                lag = max_corr_idx - n_sample + 1
                
                # Log correlation strength
                max_corr = correlation[max_corr_idx]
                norm_factor = np.sqrt(np.sum(base_rmmean**2) * np.sum(reference_remean**2))
                if norm_factor > 0:
                    normalized_corr = max_corr / norm_factor
                    logger.debug(f"Maximum correlation between '{base_chName}' and '{reference_ch}': {normalized_corr:.4f} at lag {lag}")
                else:
                    logger.warning("Could not normalize correlation (division by zero)")
            except Exception as e:
                logger.error(f"Error calculating correlation: {str(e)}")
                raise ValueError(f"Correlation calculation failed: {str(e)}")
            
            # Move the target channel by the calculated lag
            try:
                self.move_data(to_move_chName, -lag, sseg=sseg)
                logger.info(f"Moved channel '{to_move_chName}' by {-lag} points based on correlation")
            except Exception as e:
                logger.error(f"Error moving channel '{to_move_chName}': {str(e)}")
                raise ValueError(f"Failed to move channel: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error in move_ccor: {str(e)}")
            raise

    def find_move_ccor(self,
                  base_chName,
                  reference_ch,
                  sseg=0):
        """
        Find the number of points to move between channels using cross-correlation.
        
        Parameters:
        -----------
        base_chName : str
            Name of base channel
        reference_ch : str
            Name of reference channel
        sseg : int, optional
            Segment index to process, default is 0
            
        Returns:
        --------
        int
            Number of points to move
            
        Notes:
        ------
        - Uses cross-correlation for alignment
        - Handles large datasets efficiently
        - Maintains numerical accuracy
        """
        try:
            # Check if channels exist
            if reference_ch not in self.data[sseg].columns:
                logger.error(f"Reference channel '{reference_ch}' not found in segment {sseg}")
                raise KeyError(f"Reference channel '{reference_ch}' not found")
                
            if base_chName not in self.data[sseg].columns:
                logger.error(f"Base channel '{base_chName}' not found in segment {sseg}")
                raise KeyError(f"Base channel '{base_chName}' not found")
            
            # Get channel data
            reference = self.data[sseg][reference_ch]
            base = self.data[sseg][base_chName]
            
            # Remove mean from signals for better correlation
            try:
                base_rmmean = base.values - np.mean(base.values)
                reference_remean = reference.values - np.mean(reference.values)
            except Exception as e:
                logger.error(f"Error removing mean from signals: {str(e)}")
                raise ValueError(f"Failed to prepare signals for correlation: {str(e)}")
            
            # Get number of samples
            try:
                n_sample = self.segInfo['N sample'].iloc[sseg]
            except Exception as e:
                logger.error(f"Error getting sample count: {str(e)}")
                raise ValueError(f"Failed to get sample count: {str(e)}")
            
            # Calculate cross-correlation and find lag
            try:
                correlation = correlate(base_rmmean, reference_remean, method='fft')
                max_corr_idx = np.argmax(correlation)
                lag = max_corr_idx - n_sample + 1
                lag = - lag
                
                # Log correlation strength
                max_corr = correlation[max_corr_idx]
                norm_factor = np.sqrt(np.sum(base_rmmean**2) * np.sum(reference_remean**2))
                if norm_factor > 0:
                    normalized_corr = max_corr / norm_factor
                    logger.info(f"Maximum correlation between '{base_chName}' and '{reference_ch}': {normalized_corr:.4f} at lag {lag}")
                else:
                    logger.warning("Could not normalize correlation (division by zero)")
                    
                return lag
            except Exception as e:
                logger.error(f"Error calculating correlation: {str(e)}")
                raise ValueError(f"Correlation calculation failed: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error in find_move_ccor: {str(e)}")
            raise

    def apply_lowpass_filter(self, chName, cutoffull=2, replace=True, returnValue=False, 
                          sseg=0, order=6, plot=False):
        """
        Apply a lowpass filter to a channel.
        
        Parameters:
        -----------
        chName : str or list
            Name of the channel to filter, or list of channel names
        cutoffull : float, optional
            Full-scale cutoff in rad/s, default is 2. In model scale this is
            converted as ``cutoffull / (2*pi) * sqrt(lam)``. This is **not** Hertz.
        replace : bool, optional
            Whether to replace original data, default is True
        returnValue : bool, optional
            Whether to return filtered data, default is False
        sseg : int, optional
            Segment index, default is 0
        order : int, optional
            Filter order, default is 6
        plot : bool, optional
            Whether to plot before/after comparison, default is False
            
        Returns:
        --------
        numpy.ndarray, optional
            Filtered data if returnValue is True
        """
        return apply_lowpass_filter(self, chName, cutoffull, replace, returnValue, sseg, order, plot)

    def apply_highpass_filter(self, chName, cutoffull=2, replace=True, returnValue=False, 
                           sseg=0, order=6, plot=False):
        """
        Apply a highpass filter to a channel.
        
        Parameters:
        -----------
        chName : str or list
            Name of the channel to filter, or list of channel names
        cutoffull : float, optional
            Full-scale cutoff in rad/s, default is 2. In model scale this is
            converted as ``cutoffull / (2*pi) * sqrt(lam)``. This is **not** Hertz.
        replace : bool, optional
            Whether to replace original data, default is True
        returnValue : bool, optional
            Whether to return filtered data, default is False
        sseg : int, optional
            Segment index, default is 0
        order : int, optional
            Filter order, default is 6
        plot : bool, optional
            Whether to plot before/after comparison, default is False
            
        Returns:
        --------
        numpy.ndarray, optional
            Filtered data if returnValue is True
        """
        return apply_highpass_filter(self, chName, cutoffull, replace, returnValue, sseg, order, plot)

    def remove_mean(self, chName, sseg=0):
        """
        Remove the mean value from one or more channels.
        
        Parameters:
        -----------
        chName : str or list
            Name of the channel(s) to process
        sseg : int, optional
            Segment index, default is 0
        """
        return remove_mean(self, chName, sseg)

    def add_value(self, chName, value2add, sseg=0):
        """
        Add a constant value to one or more channels.
        
        Parameters:
        -----------
        chName : str or list
            Name of the channel(s) to process
        value2add : float
            Value to add to the channel(s)
        sseg : int, optional
            Segment index, default is 0
        """
        return add_value(self, chName, value2add, sseg)

    def multiply_value(self, chName, value2mul, sseg=0):
        """
        Multiply one or more channels by a constant value.
        
        Parameters:
        -----------
        chName : str or list
            Name of the channel(s) to process
        value2mul : float
            Value to multiply the channel(s) by
        sseg : int, optional
            Segment index, default is 0
        """
        return multiply_value(self, chName, value2mul, sseg)

    def move_data(self, chName, point_of_move, sseg=0):
        """
        Move data in a channel by a specified number of points.
        
        Parameters:
        -----------
        chName : str
            Name of the channel to move
        point_of_move : int
            Number of points to move the data (positive for forward, negative for backward)
        sseg : int, optional
            Segment index, default is 0
        """
        return move_data(self, chName, point_of_move, sseg)

    def data_wash(self, ChName, method='linear', order=5, threshold=3, sseg=0):
        """
        Clean data by detecting and interpolating outliers.
        
        Parameters:
        -----------
        ChName : str
            Name of the channel to clean
        method : str, optional
            Interpolation method ('linear' or 'polynomial'), default is 'linear'
        order : int, optional
            Order of polynomial interpolation, default is 5
        threshold : float, optional
            Standard deviation threshold for outlier detection, default is 3
        sseg : int, optional
            Segment index to process, default is 0
        """
        return data_wash(self, ChName, method, order, threshold, sseg)

    def add_diff1(self, name, sseg=0, filter=False, filter_cutoff=2):
        """
        Calculate and add first derivative of a channel.
        
        Parameters:
        -----------
        name : str
            Name of the channel to differentiate
        sseg : int, optional
            Segment index to process, default is 0
        filter : bool, optional
            Whether to apply lowpass filter, default is False
        filter_cutoff : float, optional
            Cutoff frequency for filtering in Hz, default is 2
        """
        return add_diff1(self, name, sseg, filter, filter_cutoff)

    def add_diff2(self, name, sseg=0, filter=False, filter_cutoff=2):
        """
        Calculate and add second derivative of a channel.
        
        Parameters:
        -----------
        name : str
            Name of the channel to differentiate
        sseg : int, optional
            Segment index to process, default is 0
        filter : bool, optional
            Whether to apply lowpass filter, default is False
        filter_cutoff : float, optional
            Cutoff frequency for filtering in Hz, default is 2
        """
        return add_diff2(self, name, sseg, filter, filter_cutoff)

    def updateST(self, chName='all', sseg=0):
        """
        Update statistical information for channels.
        
        Parameters:
        -----------
        chName : str, optional
            Channel name to update, 'all' for all channels, default is 'all'
        sseg : int, optional
            Segment index to process, default is 0
            
        Notes:
        ------
        - Calculates basic statistics (mean, std, min, max)
        - Updates channel information
        - Handles multiple channels efficiently
        - Uses vectorized operations for performance
        """
        if chName == 'all':
            # Compute all channel statistics at once using pandas aggregation
            try:
                # Fetch the segment DataFrame
                data_frame = self.data[sseg]
                
                # Use chunked processing for large DataFrames (adjustable threshold)
                if data_frame.shape[0] * data_frame.shape[1] > 10000000:
                    # Prefer Dask for parallel computation on large data
                    try:
                        import dask.dataframe as dd
                        
                        # Convert pandas DataFrame to a Dask DataFrame
                        dask_df = dd.from_pandas(data_frame, npartitions=min(32, data_frame.shape[1]))
                        
                        # Compute statistics in parallel
                        mean_result = dask_df.mean().compute()
                        std_result = dask_df.std().compute()
                        max_result = dask_df.max().compute()
                        min_result = dask_df.min().compute()
                        
                        # Assemble the result DataFrame
                        stats = pd.DataFrame({
                            'Mean': mean_result,
                            'STD': std_result,
                            'Max': max_result,
                            'Min': min_result
                        })
                        
                        # Attach unit column
                        stats['Unit'] = self.chInfo.set_index('Name')['Unit']
                        
                        # Store updated statistics
                        self.segStatis[sseg] = stats
                        
                    except ImportError:
                        # Fall back to manual chunked processing if Dask is unavailable
                        logger.info("Dask not available, using chunked processing for large dataset")
                        
                        # Chunk size in rows
                        chunk_size = 1000000 // data_frame.shape[1]
                        chunk_size = max(chunk_size, 1000)  # Guarantee at least 1000 rows per chunk
                        
                        # Initialise accumulators (Welford online algorithm)
                        means = pd.Series(index=data_frame.columns)
                        stds = pd.Series(index=data_frame.columns)
                        maxs = pd.Series(index=data_frame.columns)
                        mins = pd.Series(index=data_frame.columns)
                        
                        # Chunk iteration
                        n_chunks = (data_frame.shape[0] + chunk_size - 1) // chunk_size
                        
                        # Welford online mean and variance
                        count = 0
                        M2 = pd.Series(0, index=data_frame.columns)
                        mean = pd.Series(0, index=data_frame.columns)
                        
                        # Initialise min/max from first row
                        maxs = data_frame.iloc[0]
                        mins = data_frame.iloc[0]
                        
                        for i in range(n_chunks):
                            start_idx = i * chunk_size
                            end_idx = min((i + 1) * chunk_size, data_frame.shape[0])
                            chunk = data_frame.iloc[start_idx:end_idx]
                            
                            # Update running min/max
                            maxs = np.maximum(maxs, chunk.max())
                            mins = np.minimum(mins, chunk.min())
                            
                            # Update mean and variance via Welford algorithm
                            for _, row in chunk.iterrows():
                                count += 1
                                delta = row - mean
                                mean += delta / count
                                delta2 = row - mean
                                M2 += delta * delta2
                        
                        # Compute standard deviation
                        stds = np.sqrt(M2 / count)
                        means = mean
                        
                        # Assemble result DataFrame
                        stats = pd.DataFrame({
                            'Mean': means,
                            'STD': stds,
                            'Max': maxs,
                            'Min': mins,
                            'Unit': self.chInfo.set_index('Name')['Unit']
                        })
                        
                        # Store updated statistics
                        self.segStatis[sseg] = stats
                else:
                    # Small DataFrame: use pandas aggregation directly
                    stats = data_frame.agg(['mean', 'std', 'max', 'min'])
                    
                    # Transpose to match the expected (channels × stats) layout
                    stats = stats.T
                    stats.columns = ['Mean', 'STD', 'Max', 'Min']
                    
                    # Attach unit column
                    stats['Unit'] = self.chInfo.set_index('Name')['Unit']
                    
                    # Store updated statistics
                    self.segStatis[sseg] = stats
                
            except Exception as e:
                logger.error(f"Statistics calculation error: {str(e)}")
                # Fall back to simple pandas aggregation
                data_frame = self.data[sseg]
                means = data_frame.mean()
                stds = data_frame.std()
                maxs = data_frame.max()
                mins = data_frame.min()
                
                # Assemble fallback result DataFrame
                stats_data = {
                    'Mean': means,
                    'STD': stds,
                    'Max': maxs,
                    'Min': mins,
                    'Unit': self.chInfo.set_index('Name')['Unit']
                }
                
                # Store updated statistics
                self.segStatis[sseg] = pd.DataFrame(stats_data)
        else:
            # Update statistics for a single specified channel
            if chName in self.chInfo['Name'].values:
                # Use pandas Series methods for fast single-channel statistics
                series = self.data[sseg][chName]
                
                # Chunked processing for very large series (adjustable threshold)
                if len(series) > 10000000:
                    # Chunk size in samples
                    chunk_size = 1000000
                    
                    # Initialise Welford accumulators
                    count = 0
                    mean = 0
                    M2 = 0
                    max_val = series.iloc[0]
                    min_val = series.iloc[0]
                    
                    # Iterate over chunks
                    n_chunks = (len(series) + chunk_size - 1) // chunk_size
                    
                    for i in range(n_chunks):
                        start_idx = i * chunk_size
                        end_idx = min((i + 1) * chunk_size, len(series))
                        chunk = series.iloc[start_idx:end_idx]
                        
                        # Update running min/max
                        max_val = max(max_val, chunk.max())
                        min_val = min(min_val, chunk.min())
                        
                        # Update mean and variance via Welford algorithm
                        for val in chunk:
                            count += 1
                            delta = val - mean
                            mean += delta / count
                            delta2 = val - mean
                            M2 += delta * delta2
                    
                    # Compute standard deviation
                    std = np.sqrt(M2 / count)
                    
                    stats = {
                        'mean': mean,
                        'std': std,
                        'max': max_val,
                        'min': min_val
                    }
                else:
                    # Small series: direct pandas aggregation
                    stats = series.agg(['mean', 'std', 'max', 'min'])
                
                # Retrieve the channel unit
                unit_idx = self.chInfo['Name'].values == chName
                unit = self.chInfo.loc[unit_idx, 'Unit'].values[0]
                
                # Write back to the statistics table
                self.segStatis[sseg].loc[chName] = [
                    stats['mean'], stats['std'], stats['max'], stats['min'], unit]
            else:
                logger.error(f'ERROR! {chName:8s} not found.')

        return None

    def cut_series(self, start, stop, sseg=0):
        """Cut a time series to a specified time range in seconds.

        Parameters
        ----------
        start : float
            Start time in seconds (inclusive).
        stop : float
            End time in seconds (exclusive of the sample at ``stop``).
        sseg : int, optional
            Segment index, default is 0.

        Raises
        ------
        ValueError
            If the segment is missing or the range is empty.
        """
        def move_timestr(timestr, seconds_float):
            seconds = int(seconds_float)
            milliseconds = int((seconds_float - seconds) * 1000)
            start_time = datetime.datetime.strptime(timestr, "%H:%M:%S.%f")
            start_time_new = (
                start_time + datetime.timedelta(seconds=seconds, milliseconds=milliseconds)
            ).strftime("%H:%M:%S.%f")
            return start_time_new[:-5]

        if sseg < 0 or sseg >= self.__segN__:
            raise ValueError(f"Invalid segment index: {sseg}")

        start_idx = int(start * self.__fs__)
        stop_idx = int(stop * self.__fs__)
        n_rows = len(self.data[sseg])
        start_idx = max(0, min(start_idx, n_rows))
        stop_idx = max(start_idx, min(stop_idx, n_rows))
        if stop_idx <= start_idx:
            raise ValueError("cut_series range is empty after applying start/stop.")

        self.data[sseg] = self.data[sseg].iloc[start_idx:stop_idx].reset_index(drop=True)
        samp_num = self.data[sseg].shape[0]
        seg_label = self.segInfo.index[sseg]
        orig_start = self.segInfo['Start'].iloc[sseg]

        self.segInfo.loc[seg_label, 'Start'] = move_timestr(orig_start, start)
        self.segInfo.loc[seg_label, 'Stop'] = move_timestr(orig_start, stop)
        self.segInfo.loc[seg_label, 'Duration'] = '{0:8.1f}s'.format((samp_num - 1) / self.__fs__)
        self.segInfo.loc[seg_label, 'N sample'] = samp_num
        self.updateST(sseg=sseg)
        logger.info('Cut time series from {0:5.2f}s to {1:5.2f}s'.format(start, stop))
        return None

    def channel2fullscale(self, channel_name, lam, rho=1.025, g=9.807):
        """
        Convert a single channel from model scale to prototype scale and return timeseries.
        
        Parameters:
        -----------
        channel_name : str
            Name of the channel to convert
        lam : float
            Scale factor
        rho : float, optional
            Water density in kg/m³, default is 1.025
        g : float, optional
            Gravitational acceleration in m/s², default is 9.807
            
        Returns:
        --------
        ts : waveModel.TimeSeries
            TimeSeries object containing the converted data with time in seconds
        
        Notes:
        ------
        - Applies Froude scaling laws without modifying the original data
        - Uses the same conversion logic as to_fullscale method
        - Returns a TimeSeries object from waveModel module
        """

        # Validate channel exists
        if channel_name not in self.chInfo['Name'].values:
            logger.error(f"Channel {channel_name} not found")
            return None
            
        # Get channel index
        ch_idx = self.chInfo[self.chInfo['Name'] == channel_name].index[0]
        
        # Get channel unit
        unit = self.chInfo.loc[ch_idx, 'Unit']
        
        # Get unit conversion dictionary using utility function
        transDict = get_default_transDict(g)
        
        # Get conversion factors using utility function
        trans_temp = findtrans(unit, transDict)
        logger.debug(f"Conversion result for unit {unit}: {trans_temp}")
        
        # Conversion coefficients as float
        try:
            C1 = float(trans_temp[1][0])  # CoeffUnit
            C2 = float(rho ** trans_temp[1][1])  # CoeffRho
            C3 = float(lam ** trans_temp[1][2])  # CoeffLam
            coeff = C1 * C2 * C3
            logger.debug(f"Conversion coefficients: C1={C1}, C2={C2}, C3={C3}, total={coeff}")
        except Exception as e:
            logger.error(f"Error converting coefficients: {str(e)}")
            # Fall back to identity scaling
            coeff = 1.0
            logger.warning(f"Using default coefficient value: {coeff}")
        
        # Calculate time array based on the scaling
        fs_scaled = self.__fs__ / np.sqrt(lam)
        
        # Only the first segment is converted
        idx1 = 0
        if self.__segN__ > 1:
            logger.info(f"Multiple segments found. Only converting first segment.")
            
        # Extract original data and ensure it's a float64 numpy array
        try:
            # ndarray, not Series
            data_raw = self.data[idx1][channel_name]
            if hasattr(data_raw, 'values'):
                data = data_raw.values
            else:
                data = np.array(data_raw)
                
            # float64 for scaling
            if not np.issubdtype(data.dtype, np.floating):
                logger.debug(f"Converting data from {data.dtype} to float64")
                data = data.astype(np.float64)
            else:
                data = data.copy()
                
            # Flag non-finite samples
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                logger.warning(f"Data contains NaN or Inf values")
                
            logger.debug(f"Data shape: {data.shape}, type: {data.dtype}")
            
            # Apply conversion coefficient
            data_scaled = data * coeff
            logger.debug(f"Scaled data range: {np.min(data_scaled)} to {np.max(data_scaled)}")
            
        except Exception as e:
            logger.error(f"Error processing data in channel2fullscale: {str(e)}")
            return None
        
        # Create time array
        T = np.arange(0, len(data)) / fs_scaled
        
        # Build a waveModel.TimeSeries
        try:
            # Unit label as str
            unit_name = str(trans_temp[0]) if trans_temp and trans_temp[0] is not None else unit
            
            # args is the time vector
            ts = TimeSeries(data_scaled, T)
            return ts
        except Exception as e:
            logger.error(f"Error creating TimeSeries object: {str(e)}")
            return None

    def channel_calculate(self, ch1, ch2, operation, new_chName, sseg=0):
        """Create a new channel from an arithmetic operation on two channels.

        Parameters
        ----------
        ch1, ch2 : str
            Operand channel names.
        operation : str
            ``'add'``/``'+'``, ``'subtract'``/``'-'``, ``'multiply'``/``'*'``,
            or ``'divide'``/``'/'``.
        new_chName : str
            Name of the result channel.
        sseg : int, list, or 'all', optional
            Segment(s) to process, default is 0.

        Returns
        -------
        bool
            True on success, False otherwise.

        Notes
        -----
        Addition and subtraction require matching units. Multiplication and
        division compose a new unit string. An existing ``new_chName`` cancels
        the operation.
        """
        # Validate channel name
        if ch1 not in self.chInfo['Name'].values:
            logger.warning(f"Channel '{ch1}' does not exist.")
            return False
            
        if ch2 not in self.chInfo['Name'].values:
            logger.warning(f"Channel '{ch2}' does not exist.")
            return False
            
        # Normalise operator aliases
        ops_map = {
            '+': 'add',
            '-': 'subtract',
            '*': 'multiply',
            '/': 'divide'
        }
        
        if operation in ops_map:
            operation = ops_map[operation]
        elif operation not in ['add', 'subtract', 'multiply', 'divide']:
            valid_operations = ["'add' or '+'", "'subtract' or '-'", "'multiply' or '*'", "'divide' or '/'"]
            logger.warning(f"Invalid operation '{operation}'. Valid operations are: {valid_operations}")
            return False
            
        # Reject colliding names
        if new_chName in self.chInfo['Name'].values:
            logger.warning(f"Channel '{new_chName}' already exists. Operation canceled.")
            return False
            
        # Operand metadata
        ch1_idx = self.chInfo.index[self.chInfo['Name'] == ch1].tolist()[0]
        ch2_idx = self.chInfo.index[self.chInfo['Name'] == ch2].tolist()[0]
        
        ch1_unit = self.chInfo.loc[ch1_idx, 'Unit']
        ch2_unit = self.chInfo.loc[ch2_idx, 'Unit']
        
        # Add/subtract require identical units
        if operation in ['add', 'subtract'] and ch1_unit != ch2_unit:
            logger.warning(f"Cannot {operation} channels with different units: '{ch1_unit}' and '{ch2_unit}'")
            return False
            
        # Result unit
        if operation in ['add', 'subtract']:
            new_unit = ch1_unit
        elif operation == 'multiply':
            # Multiply units
            if ch1_unit == '-' or ch2_unit == '-':
                new_unit = ch1_unit if ch2_unit == '-' else ch2_unit
            elif ch1_unit == '' or ch2_unit == '':
                new_unit = ch1_unit if ch2_unit == '' else ch2_unit
            else:
                new_unit = f"{ch1_unit}·{ch2_unit}"
        elif operation == 'divide':
            # Divide units
            if ch1_unit == '-' or ch1_unit == '':
                new_unit = '-'
            elif ch2_unit == '-' or ch2_unit == '':
                new_unit = ch1_unit
            else:
                new_unit = f"{ch1_unit}/{ch2_unit}"
                
        # Segment list
        if sseg == 'all':
            segments = list(range(self.__segN__))
        elif isinstance(sseg, int):
            if sseg < self.__segN__:
                segments = [sseg]
            else:
                logger.warning(f"Segment {sseg} exceeds the maximum segment number ({self.__segN__ - 1}).")
                return False
        elif isinstance(sseg, list):
            segments = [s for s in sseg if s < self.__segN__]
            if len(segments) != len(sseg):
                logger.warning("Some segment indices were invalid and will be skipped.")
        else:
            logger.warning("Invalid segment selection. Use an integer, list, or 'all'.")
            return False
            
        # Success flag
        success = True
            
        # First segment creates the channel
        first_seg = segments[0]
        
        try:
            # Apply the operator
            if operation == 'add':
                result = self.data[first_seg][ch1] + self.data[first_seg][ch2]
            elif operation == 'subtract':
                result = self.data[first_seg][ch1] - self.data[first_seg][ch2]
            elif operation == 'multiply':
                result = self.data[first_seg][ch1] * self.data[first_seg][ch2]
            elif operation == 'divide':
                # Guard divide-by-zero
                divisor = self.data[first_seg][ch2].copy()
                # Replace 0 with NaN before division
                divisor = divisor.replace(0, np.nan)
                result = self.data[first_seg][ch1] / divisor
                # Fill NaN (from /0) with 0
                result = result.fillna(0)
                
            # Register the result channel
            # Coefficient is 1 after the computed result
            self.add_channel(new_chName, new_unit, result.values, self.__fs__, 1.0, 0, first_seg)
                
            # Additional segments
            for seg in segments[1:]:
                if ch1 in self.data[seg].columns and ch2 in self.data[seg].columns:
                    # Apply the operator
                    if operation == 'add':
                        result = self.data[seg][ch1] + self.data[seg][ch2]
                    elif operation == 'subtract':
                        result = self.data[seg][ch1] - self.data[seg][ch2]
                    elif operation == 'multiply':
                        result = self.data[seg][ch1] * self.data[seg][ch2]
                    elif operation == 'divide':
                        # Guard divide-by-zero
                        divisor = self.data[seg][ch2].copy()
                        # Replace 0 with NaN before division
                        divisor = divisor.replace(0, np.nan)
                        result = self.data[seg][ch1] / divisor
                        # Fill NaN (from /0) with 0
                        result = result.fillna(0)
                        
                    # Store result samples
                    self.data[seg][new_chName] = result
                    
                    # Update segment statistics
                    self.segStatis[seg].loc[new_chName] = [
                        np.mean(result), 
                        np.std(result),
                        np.amax(result), 
                        np.amin(result), 
                        new_unit
                    ]
        except Exception as e:
            logger.error(f"Error performing {operation} operation: {str(e)}")
            # Roll back a partial channel
            if new_chName in self.chInfo['Name'].values:
                self.delete_channel(new_chName)
            success = False
            
        if success:
            ops_dict = {'add': '+', 'subtract': '-', 'multiply': '*', 'divide': '/'}
            ops_symbol = ops_dict.get(operation, operation)
            logger.info(f"Created new channel '{new_chName}' as {ch1} {ops_symbol} {ch2}")
            
        return success

    def channel_apply_function(self, ch, func, new_chName, unit=None, sseg=0):
        """Apply a function to one channel and store the result as a new channel.

        Parameters
        ----------
        ch : str
            Source channel name.
        func : callable or str
            Vectorized callable (e.g. ``np.square``, ``lambda x: x**2``) or a
            restricted expression string where ``x`` is the channel array
            (e.g. ``"np.abs(x)"``).
        new_chName : str
            Name of the new channel.
        unit : str, optional
            Unit of the result. Defaults to the source channel unit.
        sseg : int, list, or 'all', optional
            Segment(s) to process, default is 0.

        Returns
        -------
        bool
            True on success, False otherwise.

        Notes
        -----
        Prefer a vectorized callable. String expressions are evaluated once on
        the full ndarray, not sample-by-sample.
        """
        import math

        if ch not in self.chInfo['Name'].values:
            logger.warning(f"Channel '{ch}' does not exist.")
            return False

        if new_chName in self.chInfo['Name'].values:
            logger.warning(f"Channel '{new_chName}' already exists. Operation canceled.")
            return False

        ch_idx = self.chInfo.index[self.chInfo['Name'] == ch].tolist()[0]
        ch_unit = self.chInfo.loc[ch_idx, 'Unit']

        if unit is None:
            unit = ch_unit
            logger.info(
                f"Using source unit '{ch_unit}' for '{new_chName}'. "
                "Adjust the unit if the transform changes dimensionality."
            )

        if sseg == 'all':
            segments = list(range(self.__segN__))
        elif isinstance(sseg, int):
            if sseg < self.__segN__:
                segments = [sseg]
            else:
                logger.warning(f"Segment {sseg} exceeds the maximum segment number ({self.__segN__ - 1}).")
                return False
        elif isinstance(sseg, list):
            segments = [s for s in sseg if s < self.__segN__]
            if len(segments) != len(sseg):
                logger.warning("Some segment indices were invalid and will be skipped.")
        else:
            logger.warning("Invalid segment selection. Use an integer, list, or 'all'.")
            return False

        try:
            if isinstance(func, str):
                unsafe_terms = [
                    'import', 'eval', 'exec', 'compile', 'open', 'file',
                    'os.', 'sys.', 'subprocess', 'shutil', '__',
                ]
                if any(term in func for term in unsafe_terms):
                    logger.error(f"Unsafe expression detected: {func}")
                    return False

                def apply_func(arr):
                    local_vars = {"np": np, "math": math, "x": np.asarray(arr)}
                    return eval(func, {"__builtins__": {}}, local_vars)
            else:
                apply_func = func

            for seg_idx, seg in enumerate(segments):
                if ch not in self.data[seg].columns:
                    logger.warning(f"Channel '{ch}' not found in segment {seg}, skipping.")
                    continue

                values = np.asarray(self.data[seg][ch].values, dtype=np.float64)
                try:
                    result = np.asarray(apply_func(values), dtype=np.float64)
                    if result.shape != values.shape:
                        result = np.asarray([apply_func(v) for v in values], dtype=np.float64)
                except TypeError:
                    result = np.asarray([apply_func(v) for v in values], dtype=np.float64)

                if seg_idx == 0:
                    self.add_channel(new_chName, unit, result, self.__fs__, 1.0, 0, seg)
                else:
                    self.data[seg][new_chName] = result

                self.segStatis[seg].loc[new_chName] = [
                    np.mean(result),
                    np.std(result),
                    np.amax(result),
                    np.amin(result),
                    unit,
                ]

            logger.info(f"Created channel '{new_chName}' by applying a function to '{ch}'")
            return True

        except Exception as e:
            logger.error(f"Error applying function to channel: {str(e)}")
            if new_chName in self.chInfo['Name'].values:
                self.delete_channel(new_chName)
            return False

    def _findtrans(self, unit, transDict):
        """
        Compatibility wrapper for the findtrans function in utils.py.
        
        Parameters:
        -----------
        unit : str
            Unit to be converted
        transDict : dict
            Dictionary of unit conversion rules
            
        Returns:
        --------
        list
            Result from utils.findtrans
            
        Notes:
        ------
        This method is maintained for backward compatibility.
        New code should use utils.findtrans directly.
        """
        return findtrans(unit, transDict)

    def _get_default_transDict(self, g=9.807):
        """
        Compatibility wrapper for the get_default_transDict function in utils.py.
        
        Parameters:
        -----------
        g : float, optional
            Gravitational acceleration in m/s², default is 9.807
            
        Returns:
        --------
        dict
            Default unit conversion dictionary
            
        Notes:
        ------
        This method is maintained for backward compatibility.
        New code should use utils.get_default_transDict directly.
        """
        return get_default_transDict(g)

