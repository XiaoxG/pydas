# -*- coding: utf-8 -*-
"""
PyDAS - Python Data Analysis System
A comprehensive data analysis system for processing and analyzing time series data.

Module Organization:
------------------
PyDAS now adopts a modular structure, separating different functionalities into specialized modules:
- pydas.py: Core class and method definitions
- process.py: Data processing functions (filtering, differentiation, data cleaning, etc.)
- plot.py: Visualization functions (plotting, histograms, scatter plots, etc.)
- output.py: Data output functions (file exports, etc.)
- utils.py: General utility functions
- logger.py: Logging functionality

Function Categories:
------------------
1. Channel Operations
   a) Channel Management:
      - add_channel: Add a new channel
      - delete_channel: Delete specified channel
      - select_channels: Select and keep specified channels
      - rename_channel: Rename channel
      - change_channel_order: Change channel order
   
   b) Channel Data Processing (imported from process module):
      - remove_mean: Remove mean from channel data
      - add_value: Add constant value to channel data
      - multiply_value: Multiply channel data by constant
      - move_data: Move channel data
      - data_wash: Clean data, detect and interpolate outliers

2. Channel Calculations
   a) Differential Operations (imported from process module):
      - add_diff1: Calculate and add first derivative
      - add_diff2: Calculate and add second derivative
   
   b) Filtering (imported from process module):
      - apply_lowpass_filter: Apply lowpass filter
      - apply_highpass_filter: Apply highpass filter
   
   c) Data Alignment:
      - move_ccor: Move channel data using cross-correlation
      - find_move_ccor: Find points to move between channels
      - cut_series: Cut time series to specified range

3. Data Output (partially imported from output module)
   a) File Output:
      - to_dat: Export data to DAT file
      - to_mat: Export data to MAT file
      - to_feather: Export data to Feather file
      - to_parquet: Export data to Parquet file
      - write: Write data file
   
   b) Information Output:
      - print_info: Print basic information
      - print_channel_info: Print channel information
      - print_statistics: Print statistical information

4. Data Visualization (partially imported from plot module)
   - plot_channel: Plot channel data with interactive features and performance optimization
   - plot_histogram: Generate histograms with statistics and Gaussian fitting capabilities
   - plot_xy: Create XY scatter plots with density visualization, downsampling, and linear regression
   - spectral_analysis: Perform spectral analysis on channels with customizable parameters and fullscale conversion

5. Data Import
   - read_waveCal: Read wave calibration data
   - read_motion: Read motion data and add as channels

6. Data Conversion
   - fix_unit: Fix channel unit
   - to_fullscale: Convert model scale data to prototype scale
   - channel2fullscale: Convert individual channel to fullscale for spectral analysis

7. Data Update
   - updateST: Update statistical information
   - updateChN: Update channel count

8. Global Utility Functions (partially imported from utils module)
   - diff1d: Calculate derivative of one-dimensional array with adaptive optimization
   - data_change_fs: Change data sampling frequency with optimized implementation

Performance Optimizations:
------------------------
1. Numba Acceleration
   - JIT compilation for compute-intensive functions
   - Parallel processing for large datasets
   - Adaptive algorithm selection based on data size
   - Applied to derivative calculation and data resampling

2. Vectorized Operations
   - Pandas vectorized operations for statistics
   - Numpy vectorized operations for dataset processing
   - Batch processing for large datasets

3. Cache Optimization
   - Cache for unit conversion calculations
   - Pre-calculation of unique unit conversions
   - Memory-mapped file reading for large datasets

4. Memory Management
   - Efficient data loading with memory mapping
   - Chunk-based processing for huge datasets
   - Reduced data copying and conversion

5. Visualization Optimizations
   - Automatic downsampling for large datasets
   - WebGL rendering for interactive visualization
   - Adaptive sampling algorithms (LTTB)
   - Memory-efficient plotting modes

Dependencies:
------------
- numpy: Numerical computing
- pandas: Data manipulation
- numba: JIT compilation
- plotly: Interactive visualization
- scipy: Scientific computing
- matplotlib: Static visualization and fallback rendering

Author: Xiaoxian Guo
Date: 2025-04-12
Version: 1.0.3
"""
import sys
import os
import struct
import math
import numpy as np
import pandas as pd
from scipy.signal import correlate
from scipy.spatial.transform import Rotation as R
from waveModel.timeseries import TimeSeries
import datetime
from logger import logger, setup_logger, LOG_LEVELS  # 修改回非相对导入
from utils import diff1d, data_change_fs, get_default_transDict, findtrans  # 添加导入get_default_transDict和findtrans
from output import write_data, export_to_dat, export_to_mat, export_to_feather, export_to_parquet  # 修改回非相对导入
from plot import validate_channel, plot_channel, plot_histogram, plot_xy  # 修改回非相对导入
from analysis import spectral_analysis, statistic_analysis  # 从analysis模块导入分析函数
from process import (
    apply_lowpass_filter, 
    apply_highpass_filter, 
    remove_mean, 
    add_value, 
    multiply_value, 
    move_data, 
    data_wash,
    add_diff1,
    add_diff2
)

# Import numba for acceleration
try:
    from numba import jit, float64, int32, void, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available. Some functions will run slower.")

class PyDAS:
    """
    Python Data Analysis System for processing and analyzing time series data.
    
    This class provides comprehensive tools for reading, processing, and analyzing
    experimental data with support for filtering, scaling, and visualization.
    
    Attributes:
    -----------
    data : list of pandas.DataFrame
        List of data segments, each containing channel data
    chInfo : pandas.DataFrame
        Channel information including names, units, and coefficients
    segInfo : pandas.DataFrame
        Segment information including sample counts and timestamps
    __fs__ : float
        Sampling frequency in Hz
    """
    
    def __init__(self, filename, lam, sseg='all', log_level='info'):
        """
        Initialize PyDAS object and read data file.
        
        Parameters:
        -----------
        filename : str
            Path to the data file
        lam : float, optional
            Scale factor for data conversion, default is 1
        sseg : int or str, optional
            Segment index to read, 'all' for all segments, default is 'all'
        log_level : str, optional
            Logging level ('debug', 'info', 'warning', 'error', 'critical'), default is 'info'
            
        Notes:
        ------
        - Automatically detects file format and reads accordingly
        - Processes channel information and data segments
        - Calculates basic statistics for each channel
        """
        # Configure logger
        # self.set_logger(log_level)  # 使用新的setup_logger函数
        setup_logger(log_level)
        
        # Initialize basic properties
        self.__lam__ = lam
        self.__fs__ = 1  # Default sampling frequency, will be updated during reading
        self.__chN__ = 0  # Number of channels
        self.__segN__ = 0  # Number of segments
        self.__scale__ = 'model'  # Default scale is model scale
        
        # Validate segment selection
        if not (isinstance(sseg, int) or sseg == 'all'):
            logger.error("Input 'sseg' is illegal (should be int or 'all').")
            raise ValueError("sseg must be an integer or 'all'")
        
        # If filename is provided, read the data file
        if filename is not None:
            # Validate file existence
            if os.path.exists(filename):
                self.__filename__ = filename
            else:
                logger.error(f"File {filename} does not exist. Breaking!")
                sys.exit()
            
            # Read the data file
            self.__read__(sseg)
        else:
            # Initialize empty data structures for manual data setting
            self.__filename__ = None
            self.chInfo = pd.DataFrame(columns=['Name', 'Unit'])
            self.data = {}
            logger.info("Created empty PyDAS object. Use load() method to read data or set data manually.")

    def __read__(self, sseg):
        """
        Read data from the *.out file.
        
        This method reads the file header, channel information, and data segments.
        It populates the object's properties with the read data.
        
        Parameters:
        -----------
        sseg : int or 'all'
            Selected segment number to load, or 'all' to load all segments
        """
        with open(self.__filename__, 'rb') as fIn:
            # Read file header (256 bytes)
            fmtstr = '=hhlhh2s2s240s'
            buf = fIn.read(256)
            if not buf:
                logger.warning(f"Reading data file {self.__filename__} failed, exiting...")
                return None
                
            # Unpack header data
            tmp = struct.unpack(fmtstr, buf)
            index, self.__chN__, self.__fs__, self.__segN__ = tmp[0], tmp[1], tmp[3], tmp[4]
            
            # Extract date information
            datemm, datedd = tmp[5].decode('utf-8'), tmp[6].decode('utf-8')
            self.__date__ = f'{datemm}-{datedd}'
            
            # Extract global description
            self.__desc__ = tmp[7].decode('utf-8').rstrip()

            # Read channel names (16 bytes per channel)
            chName = [namei.decode('utf-8').rstrip() for namei in
                      struct.unpack(self.__chN__ * '16s', fIn.read(self.__chN__ * 16))]
            
            # Read channel units (4 bytes per channel)
            chUnit = [uniti.decode('utf-8').rstrip() for uniti in
                      struct.unpack(self.__chN__ * '4s', fIn.read(self.__chN__ * 4))]
            
            # Read channel coefficients (4 bytes per channel)
            chCoef = struct.unpack('=' + self.__chN__ * 'f',
                                   fIn.read(self.__chN__ * 4))

            # Read channel IDs if available
            if (index < -1):
                chIdx = struct.unpack(
                    '=' + self.__chN__ * 'h', fIn.read(self.__chN__ * 2))
            else:
                chIdx = list(range(1, self.__chN__ + 1))

            # Create channel information dictionary and DataFrame
            chInfoDict = {'Index': chIdx, 'Name': chName, 'Unit': chUnit,
                          'Coef': chCoef}
            column = ['Name', 'Unit', 'Coef']
            self.chInfo = pd.DataFrame(chInfoDict, columns=column)
            self.chInfo.index = range(1, self.__chN__ + 1)

            # Initialize arrays for segment data
            sampNum = [0] * self.__segN__  # Number of samples in each segment
            segInfo = [[] for _ in range(self.__segN__)]  # Segment information
            segStatis = [[] for _ in range(self.__segN__)]  # Statistical values
            dataRaw = [[] for _ in range(self.__segN__)]  # Raw data
            note = [[] for _ in range(self.__segN__)]  # Notes for each segment
            
            # 记录文件位置，用于后续内存映射
            segment_positions = []
            segment_sizes = []

            # 首先读取所有段的信息，但不立即读取数据
            for iseg in range(self.__segN__):
                # Align to 128-byte boundary
                p_cur = fIn.tell()
                aligned_pos = 128 * math.ceil(p_cur / 128)
                fIn.seek(aligned_pos)
                segment_positions.append(aligned_pos)

                # Read segment information (256 bytes)
                fmtstr = '=hhlBBBBBBBB240s'
                buf = fIn.read(256)
                segInfo[iseg] = struct.unpack(fmtstr, buf)

                # Extract segment details
                segChN = segInfo[iseg][1]  # Number of channels in this segment
                sampNum[iseg] = segInfo[iseg][2] - 5  # Number of samples (minus 5)
                note[iseg] = segInfo[iseg][11].decode('utf-8').rstrip()  # Segment note

                # Read statistical values for each channel
                fmtstr = '=' + segChN * 'h' + segChN * 'f' + segChN * 2 * 'h'
                buf = fIn.read(segChN * (2 * 3 + 4))
                segStatis[iseg] = struct.unpack(fmtstr, buf)
                
                # 记录数据段的位置和大小，但不立即读取
                data_pos = fIn.tell()
                data_size = sampNum[iseg] * segChN * 2
                segment_sizes.append(data_size)
                
                # 跳过数据段
                fIn.seek(data_pos + data_size)

        # 使用内存映射读取大数据段
        with open(self.__filename__, 'rb') as fIn:
            for iseg in range(self.__segN__):
                segChN = segInfo[iseg][1]
                
                # 使用内存映射读取数据
                fIn.seek(segment_positions[iseg] + 256 + segChN * (2 * 3 + 4))
                
                # 对于大数据段，使用内存映射
                if sampNum[iseg] * segChN > 1000000:  # 阈值可以根据实际情况调整
                    # 使用numpy的memmap直接从文件读取数据
                    mm = np.memmap(self.__filename__, dtype=np.int16, mode='r',
                                  offset=fIn.tell(),
                                  shape=(sampNum[iseg], segChN))
                    # 复制到内存中以避免后续操作影响原文件
                    dataRaw[iseg] = np.array(mm, dtype=np.int16)
                    # 关闭内存映射
                    del mm
                else:
                    # 对于小数据段，直接读取
                    dataRaw[iseg] = np.frombuffer(
                        fIn.read(sampNum[iseg] * segChN * 2),
                        dtype=np.int16
                    ).reshape((sampNum[iseg], segChN))

        # Process segment information
        segType = []
        startTime = []
        stopTime = []
        index = []
        duration = []
        
        for n in range(self.__segN__):
            # Segment type: 0-sampling, 1-pre-calibration, 2-post-calibration
            segType.append(segInfo[n][0])
            
            # Format start and stop times
            startTime.append('{0:02d}:{1:02d}:{2:02d}.{3:1d}'.format(
                segInfo[n][6], segInfo[n][5], segInfo[n][4], segInfo[n][3]))
            stopTime.append('{0:02d}:{1:02d}:{2:02d}.{3:1d}'.format(
                segInfo[n][10], segInfo[n][9], segInfo[n][8], segInfo[n][7]))
            
            # Segment index and duration
            index.append(f'Seg{n:2d}')
            duration.append(f'{(sampNum[n] - 1) / self.__fs__:8.1f}s')
        
        # Create segment information DataFrame
        segInfoDict = {
            'Type': segType,
            'Start': startTime,
            'Stop': stopTime,
            'Duration': duration,
            'N sample': sampNum,
            'Note': note
        }
        column = ['Type', 'Start', 'Stop', 'Duration', 'N sample', 'Note']
        segInfo = pd.DataFrame(segInfoDict, index=index, columns=column)

        # 使用并行处理转换统计数据和原始数据
        from concurrent.futures import ThreadPoolExecutor
        import multiprocessing
        
        # 确定使用的CPU核心数
        num_cores = min(multiprocessing.cpu_count(), self.__segN__)
        
        # 转换统计数据的函数
        def process_statistics(iseg):
            segStatis_temp = np.reshape(
                np.array(segStatis[iseg], dtype='float64'),
                (4, self.__chN__)).transpose()
            
            for m in range(self.__chN__):
                segStatis_temp[m] *= chCoef[m]
            
            # Create DataFrame with statistics
            column = ['Mean', 'STD', 'Max', 'Min']
            stats_df = pd.DataFrame(
                segStatis_temp, index=chName, columns=column)
            stats_df['Unit'] = chUnit
            return stats_df
        
        # 转换原始数据的函数
        def process_raw_data(iseg):
            # 使用numpy的向量化操作加速
            data_temp = dataRaw[iseg].astype('float64')
            
            # 使用广播机制一次性应用所有系数
            coef_array = np.array(chCoef, dtype='float64')
            data_temp = data_temp * coef_array
            
            # 创建DataFrame
            return pd.DataFrame(data_temp, columns=chName, dtype='float64')
        
        # 并行处理统计数据
        self.segStatis = [None] * self.__segN__
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            for iseg, result in enumerate(executor.map(process_statistics, range(self.__segN__))):
                self.segStatis[iseg] = result
        
        # 并行处理原始数据
        self.data = [None] * self.__segN__
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            for iseg, result in enumerate(executor.map(process_raw_data, range(self.__segN__))):
                self.data[iseg] = result

        # Handle segment selection
        if sseg == 'all':
            self.segInfo = segInfo
        else:
            # If a specific segment is selected, keep only that segment
            self.__segN__ = 1
            self.segInfo = segInfo[sseg:sseg + 1]
            self.segInfo = segInfo[sseg:sseg + 1].rename(index={'Seg{0:2d}'.format(sseg): 'Seg 0'})
            self.segStatis = [self.segStatis[sseg]]
            self.data = [self.data[sseg]]

        return None

    def write(self, filename, sseg='all', ch='all'):
        """
        Write data to a new *.out file.
        
        Parameters:
        -----------
        filename : str
            Path to the output *.out file
        sseg : int, list, or 'all', optional
            Segment(s) to write to the file, default is 'all'
        ch : list or 'all', optional
            Channels to write to the file, default is 'all'
        
        Notes:
        ------
        This method will automatically append '.out' extension if not provided.
        """
        return write_data(self, filename, sseg, ch)

    def to_dat(self, Time=True, sseg='all'):
        """
        Export data to DAT file format.
        
        Parameters:
        -----------
        Time : bool, optional
            If True, include time column in the output, default is True
        sseg : int or 'all', optional
            Segment(s) to export, default is 'all'
            
        Notes:
        ------
        The output file will be named based on the original filename with
        segment number and scale (model or full) appended.
        """
        return export_to_dat(self, Time, sseg)

    def to_mat(self, sseg=0):
        """
        Export data to MATLAB MAT file format.
        
        Parameters:
        -----------
        sseg : int, optional
            Segment index to export, default is 0
            
        Returns:
        --------
        bool
            True if export was successful, False otherwise
            
        Notes:
        ------
        The output file will be named based on the original filename.
        """
        return export_to_mat(self, sseg)

    def to_feather(self, sseg='all', compression='zstd'):
        """
        Export data to feather file format

        Parameters:
        -----------
        sseg : int, list, or 'all', optional
            Segment(s) to export
        compression : str, optional
            Compression to use, default is 'zstd', other options include 'lz4' and 'uncompressed'

        Returns:
        --------
        bool
            True if export was successful
        """
        return export_to_feather(self, sseg, compression)

    def to_parquet(self, sseg='all', compression='zstd', compression_level=9):
        """
        Export data to Apache Parquet file format.
        
        Parameters:
        -----------
        sseg : int, list, or 'all', optional
            Segment(s) to export, default is 'all'
        compression : str, optional
            Compression type to use. Options include: 'snappy', 'gzip', 'brotli', 'zstd', 'lz4', 'none'
            Default is 'zstd' which offers a good balance between compression ratio and speed.
        compression_level : int, optional
            Compression level for 'gzip', 'brotli', and 'zstd' compressors.
            Higher values mean better compression, but slower processing.
            Default is 9 (range typically 1-22 for zstd).
            
        Returns:
        --------
        bool
            True if export was successful, False otherwise
            
        Notes:
        ------
        The output file(s) will be named based on the original filename with segment number appended.
        Parquet format is optimized for columnar data and offers excellent compression and read performance.
        """
        return export_to_parquet(self, sseg, compression, compression_level)

    def add_channel(self, name, unit, series, fs, coef=1, point_of_move=0, sseg=0):
        """
        Add a new channel to the data.
        
        Parameters:
        -----------
        name : str
            Name of the new channel
        unit : str
            Unit of measurement
        series : numpy.ndarray
            Channel data
        fs : float
            Sampling frequency in Hz
        coef : float, optional
            Coefficient for data scaling, default is 1
        point_of_move : int, optional
            Number of points to shift the data, default is 0
        sseg : int, optional
            Segment index to add channel to, default is 0
            
        Notes:
        ------
        - Validates input data and parameters
        - Handles data alignment and scaling
        - Updates channel information and statistics
        """
        if name not in self.chInfo['Name'].values:
            # Resample if necessary
            if fs != self.__fs__:
                series = data_change_fs(series, fs, self.__fs__)
                
            # Adjust data length if necessary
            n_sample = self.segInfo.iloc[sseg]['N sample']
            if len(series) > n_sample:
                series = series[:n_sample]
            elif len(series) < n_sample:
                series = np.pad(series, (0, n_sample - len(series)), 'constant', constant_values=0)
            
            # Add to data
            self.data[sseg][name] = series
            
            # Update channel info
            new_idx = self.__chN__ + 1
            self.chInfo.loc[new_idx] = [name, unit, coef]
            
            # Update statistics
            self.segStatis[sseg].loc[name] = [
                np.mean(series), np.std(series), np.amax(series), np.amin(series), unit]
                
            # Update channel count
            self.__chN__ += 1
            
            # Move data if requested
            if point_of_move != 0:
                self.move_data(name, point_of_move, sseg=sseg)
                
            logger.info(f"Channel '{name}' has been added")
        else:
            logger.warning(f"Channel '{name}' already exists.")

    def delete_channel(self, name):
        """
        Delete a specified channel from the data.
        
        Parameters:
        -----------
        name : str
            Name of the channel to delete
            
        Notes:
        ------
        - Removes channel from all data segments
        - Updates channel information
        - Recalculates statistics
        """
        if name in self.chInfo['Name'].values:
            # Find the index of the channel
            idx = self.chInfo.index[self.chInfo['Name'] == name].tolist()[0]
            
            # Remove from channel info
            self.chInfo = self.chInfo.drop(idx)
            
            # Remove from data in all segments
            for sseg in range(self.__segN__):
                self.data[sseg] = self.data[sseg].drop(name, axis=1)
                self.segStatis[sseg] = self.segStatis[sseg].drop(name)
            
            # Update channel count
            self.__chN__ -= 1
            
            # Reset index
            self.chInfo.index = range(1, self.__chN__ + 1)
            
            logger.info(f"Channel '{name}' has been removed")
        else:
            logger.warning(f"Channel '{name}' does not exist.")

    def select_channels(self, chnames):
        """
        Select and keep only specified channels, removing others.
        
        Parameters:
        -----------
        chnames : str or list of str
            Channel name(s) to keep. Can be a single string for one channel
            or a list of strings for multiple channels.
            
        Returns:
        --------
        bool
            True if successful, False otherwise
            
        Notes:
        ------
        - Validates channel names
        - Removes all channels not in the specified list
        - Updates channel information and statistics
        """
        # Check if input is a string (single channel) and convert to list
        if isinstance(chnames, str):
            chnames = [chnames]
            
        # Validate channel names
        valid_chnames = []
        for name in chnames:
            if name in self.chInfo['Name'].values:
                valid_chnames.append(name)
            else:
                logger.warning(f"Channel '{name}' does not exist and will be ignored.")
        
        if not valid_chnames:
            logger.warning("No valid channels specified.")
            return False
            
        # Get indices of channels to keep
        keep_indices = []
        for name in valid_chnames:
            idx = self.chInfo.index[self.chInfo['Name'] == name].tolist()[0]
            keep_indices.append(idx)
            
        # Keep only selected channels in channel info
        self.chInfo = self.chInfo.loc[keep_indices]
        
        # Keep only selected channels in data and statistics
        for sseg in range(self.__segN__):
            self.data[sseg] = self.data[sseg][valid_chnames]
            self.segStatis[sseg] = self.segStatis[sseg].loc[valid_chnames]
            
        # Update channel count and reset indices
        self.__chN__ = len(valid_chnames)
        self.chInfo.index = range(1, self.__chN__ + 1)
        
        logger.info(f"Selected {len(valid_chnames)} channels: {', '.join(valid_chnames)}")
        return True

    def print_info(self, printTxt=False, printExcel=False):
        """
        Print and optionally export general information about the data.
        
        Parameters:
        -----------
        printTxt : bool, optional
            If True, export information to a text file, default is False
        printExcel : bool, optional
            If True, export information to an Excel file, default is False
            
        Returns:
        --------
        DataFrame
            DataFrame containing general information about the data
        """
        # Create information DataFrame
        info = pd.DataFrame(columns=['Value'])
        info.loc['Filename'] = self.__filename__
        info.loc['Date'] = self.__date__
        info.loc['Scale'] = self.__scale__
        info.loc['Lambda'] = self.__lam__
        info.loc['Sampling frequency'] = '{0:5.2f} Hz'.format(self.__fs__)
        info.loc['Number of channels'] = self.__chN__
        info.loc['Number of segments'] = self.__segN__
        
        # Print to console
        logger.info('\nGeneral Information:')
        logger.info(info.to_string())
        
        # Export to text file if requested
        if printTxt:
            txt_filename = os.path.splitext(self.__filename__)[0] + '_info.txt'
            with open(txt_filename, 'w') as f:
                f.write('General Information:\n')
                f.write(info.to_string())
                f.write('\n\nSegment Information:\n')
                f.write(self.segInfo.to_string())
                f.write('\n\nChannel Information:\n')
                f.write(self.chInfo.to_string())
            logger.info(f"Information exported to: {txt_filename}")
            
        # Export to Excel file if requested
        if printExcel:
            excel_filename = os.path.splitext(self.__filename__)[0] + '_info.xlsx'
            with pd.ExcelWriter(excel_filename) as writer:
                info.to_excel(writer, sheet_name='General Info')
                self.segInfo.to_excel(writer, sheet_name='Segment Info')
                self.chInfo.to_excel(writer, sheet_name='Channel Info')
            logger.info(f"Information exported to: {excel_filename}")
            
        return None

    def print_channel_info(self, printTxt=False, printExcel=False):
        """
        Print and optionally export channel information.
        
        Parameters:
        -----------
        printTxt : bool, optional
            If True, export information to a text file, default is False
        printExcel : bool, optional
            If True, export information to an Excel file, default is False
            
        Returns:
        --------
        DataFrame
            DataFrame containing channel information
        """
        # Print to console
        logger.info('Channel Information:')
        logger.info('\n' + self.chInfo.to_string())
        
        # Export to text file if requested
        if printTxt:
            txt_filename = os.path.splitext(self.__filename__)[0] + '_channel_info.txt'
            with open(txt_filename, 'w') as f:
                f.write('Channel Information:\n')
                f.write(self.chInfo.to_string())
            logger.info(f"Channel information exported to: {txt_filename}")
            
        # Export to Excel file if requested
        if printExcel:
            excel_filename = os.path.splitext(self.__filename__)[0] + '_channel_info.xlsx'
            self.chInfo.to_excel(excel_filename)
            logger.info(f"Channel information exported to: {excel_filename}")
            
        return None

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
        # 检查通道名是否存在
        if chName not in self.chInfo['Name'].values:
            logger.warning(f"Channel '{chName}' does not exist.")
            return None
            
        # 找到对应的索引
        idx = self.chInfo.index[self.chInfo['Name'] == chName].tolist()[0]
        
        # 更新单位
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
        
        Parameters:
        -----------
        lam : float
            Scale factor
        rho : float, optional
            Water density in kg/m³, default is 1.025
        g : float, optional
            Gravitational acceleration in m/s², default is 9.807
        pInfo : bool, optional
            Whether to print information, default is False
            
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
            
            # Clear global trans cache to ensure fresh calculation
            findtrans('', transDict, clear_cache=True)
            
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
            logger.info(f'lambda = {self.__lam__:2d}')
            
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

    def read_waveCal(self, wavefname, sseg=0, YBname='YBS', YBcalname='YBS', alignFlag=True):
        """
        Read wave calibration data.
        
        Parameters:
        -----------
        wavefname : str
            Path to wave calibration file
        sseg : int, optional
            Segment index to process, default is 0
        YBname : str, optional
            Name of wave gauge channel, default is 'YBS'
        YBcalname : str, optional
            Name of calibration wave gauge channel, default is 'YBS'
        alignFlag : bool, optional
            Whether to align data, default is True
            
        Notes:
        ------
        - Reads wave calibration data from file
        - Supports data alignment
        - Handles multiple wave gauges
        - Updates channel information
        """
        wavecase_cal = PyDAS(wavefname, lam = self.__lam__)
        fs_cal = wavecase_cal.__fs__
        nch = wavecase_cal.data[0].shape[1]
        for i in range(nch):
            iname = wavecase_cal.chInfo['Name'].loc[i+1]
            iunit = wavecase_cal.chInfo['Unit'].loc[i+1]
            icoef = wavecase_cal.chInfo['Coef'].loc[i+1]
            iseries = wavecase_cal.data[0][iname].values
            self.add_channel('Cal.'+iname, iunit, iseries, fs_cal, coef=icoef, point_of_move=0, sseg=sseg)
        if alignFlag:
            for ich in wavecase_cal.chInfo['Name'].values:
                self.move_ccor('Cal.'+ich, 'Cal.'+YBcalname, YBname, sseg=sseg)

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

    def change_channel_order(self,
                     newOrder,
                     sseg=0):
        """
        Change channel order in the data.
        
        Parameters:
        -----------
        newOrder : list of str
            New order of channel names
        sseg : int, optional
            Segment index to process, default is 0
            
        Notes:
        ------
        - Reorders channels in data structure
        - Updates channel information
        - Maintains data integrity
        - Validates channel names
        """
        if len(newOrder) == self.__chN__:
            indexNew = []
            for inewOrder in newOrder:
                indexNew.append(
                    list(
                        self.data[sseg].columns).index(inewOrder) +
                    1)
            self.chInfo = self.chInfo.reindex(indexNew)
            self.segStatis[sseg] = self.segStatis[sseg].reindex(newOrder)
            self.chInfo.index = np.arange(1, len(self.chInfo) + 1)
            self.data[sseg] = self.data[sseg][newOrder]
            self.updateChN()
            logger.info('Changed the Channel order.')
        else:
            raise ValueError("Number of channels does not match!")
        #self.chInfo.index = range(1,self.__chN__+1)

    def apply_lowpass_filter(self, chName, cutoffull=2, replace=True, returnValue=False, 
                          sseg=0, order=6, plot=False):
        """
        Apply a lowpass filter to a channel.
        
        Parameters:
        -----------
        chName : str or list
            Name of the channel to filter, or list of channel names
        cutoffull : float, optional
            Cutoff frequency in Hz, default is 2
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
            Cutoff frequency in Hz, default is 2
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


    def read_motion(self, motionfname, alignAccName=None, alignMethod='acc', zerofilename='', lowpassfilter=-1, rotation=True, NameList=['Platform']):
        """
        Read motion data and add as channels.
        
        Parameters:
        -----------
        motionfname : str
            Path to motion data file
        alignAccName : str, optional
            Acceleration channel name for alignment, default is None
        alignMethod : str, optional
            Alignment method ('time' or 'cross_correlation'), default is 'time'
        zerofilename : str, optional
            Path to zero reference file, default is ''
        lowpassfilter : float, optional
            Lowpass filter cutoff frequency, default is -1 (no filter)
        rotation : bool, optional
            Whether to apply rotation, default is True
        NameList : list of str, optional
            List of object names to process, default is ['Platform']
            
        Notes:
        ------
        - Reads motion data from file
        - Supports data alignment and filtering
        - Handles coordinate transformations
        - Updates channel information
        """
        try:
            # Try to open and read the motion file
            with open(motionfname, 'r') as f:
                try:
                    f.seek(0)
                    lines = f.readlines()
                    # Parse header information
                    try:
                        n_body = int(lines[2].replace('\n', '').split('\t')[1])
                        # n_frames = int(lines[0].replace('\n', '').split('\t')[1])
                        motion_fs = float(lines[3].replace('\n', '').split('\t')[1])
                        Time_start = pd.Timestamp(lines[7].replace('\n', '').split('\t')[1])
                        rotationname = lines[10].replace('\n', '').split('\t')[3:6]
                    except (IndexError, ValueError) as e:
                        logger.error(f"Invalid motion file format: {str(e)}")
                        raise ValueError(f"Motion file format is invalid: {str(e)}")
                except Exception as e:
                    logger.error(f"Error reading motion file: {str(e)}")
                    raise
        except FileNotFoundError:
            logger.error(f"Motion file not found: {motionfname}")
            raise

        motionName = ['Surge','Sway','Heave'] + rotationname
        motionDataRawList = []
        
        # Process zero reference file if provided
        if zerofilename:
            try:
                for ibody in range(n_body):
                    try:
                        # Read zero reference data
                        ZeromotionDataRaw = np.genfromtxt(
                            zerofilename, 
                            skip_header=12, 
                            delimiter='\t', 
                            usecols=(0+ibody*17, 1+ibody*17, 2+ibody*17, 3+ibody*17, 4+ibody*17, 5+ibody*17)
                        )
                        Zeromean = ZeromotionDataRaw.mean(axis=0)
                        Zeromean[3] = 0  # Don't apply zero correction to yaw
                        logger.info(f"Zero reference data: {Zeromean}")
                        # Read motion data and apply zero correction
                        motionDataRawList.append(
                            np.genfromtxt(
                                motionfname, 
                                skip_header=12, 
                                delimiter='\t', 
                                usecols=(0+ibody*17, 1+ibody*17, 2+ibody*17, 3+ibody*17, 4+ibody*17, 5+ibody*17)
                            ) - Zeromean
                        )
                    except Exception as e:
                        logger.error(f"Error processing data for body {ibody}: {str(e)}")
                        raise ValueError(f"Failed to process motion data for body {ibody}")
            except FileNotFoundError:
                logger.error(f"Zero reference file not found: {zerofilename}")
                raise
        else:
            # Read motion data without zero correction
            try:
                for ibody in range(n_body):
                    motionDataRawList.append(
                        np.genfromtxt(
                            motionfname, 
                            skip_header=12, 
                            delimiter='\t', 
                            usecols=(0+ibody*17, 1+ibody*17, 2+ibody*17, 3+ibody*17, 4+ibody*17, 5+ibody*17)
                        )
                    )
            except Exception as e:
                logger.error(f"Error reading motion data: {str(e)}")
                raise ValueError(f"Failed to read motion data: {str(e)}")

        # Process each body's motion data
        for ibody, motionDataRaw in enumerate(motionDataRawList):
            try:
                # Convert units from mm to cm for position data
                motionDataRaw[:, 0:3] /= 10
                
                # Apply rotation if requested
                if rotation:
                    try:
                        Yaw = np.mean(motionDataRaw[:, motionName.index('Yaw')])
                        # Yaw = 180
                        r = R.from_euler('z', Yaw, degrees=True)
                        motionDataRaw[:, 0:3] = r.apply(motionDataRaw[:, 0:3])
                        logger.info(f'motion rotated: {Yaw: 0.2f} DEG')
                    except Exception as e:
                        logger.warning(f"Failed to apply rotation: {str(e)}")
                        # Continue without rotation

                # Add channels for each motion component
                unit = ['cm']*3 + ['deg']*3
                for i, iName in enumerate(motionName):
                    try:
                        self.add_channel(
                            NameList[ibody] + '.' + iName, 
                            unit[i], 
                            motionDataRaw[:, i], 
                            fs=motion_fs
                        )
                    except Exception as e:
                        logger.error(f"Failed to add channel {NameList[ibody]}.{iName}: {str(e)}")
            except Exception as e:
                logger.error(f"Error processing motion data for body {ibody}: {str(e)}")
                # Continue with next body
        
        # Align motion data with existing data
        try:
            if alignMethod == 'acc':
                try:
                    # Calculate acceleration from heave motion
                    heave = self.apply_lowpass_filter(NameList[0]+'.Heave', replace=False, returnValue=True)
                    vz = diff1d(heave/100, 1 / self.__fs__)
                    az = diff1d(vz, 1 / self.__fs__) * -1
                    n_sample = self.segInfo.iloc[0]['N sample']
                    
                    # Find correlation with acceleration channel
                    base = self.apply_lowpass_filter(alignAccName, replace=False, returnValue=True)
                    lag = np.argmax(correlate(base, az, method='fft')) - n_sample + 1
                    
                    # Add calculated acceleration channel
                    self.add_channel('azfromHeave', unit='m/s2', series=az, point_of_move=lag, fs=self.__fs__)
                    
                    # Move all motion channels by the calculated lag
                    for ibody in NameList:
                        for iname in motionName:
                            try:
                                self.move_data(ibody+'.'+iname, point_of_move=lag)
                            except Exception as e:
                                logger.warning(f"Failed to move channel {ibody}.{iname}: {str(e)}")
                except Exception as e:
                    logger.error(f"Failed to align using acceleration method: {str(e)}")
                    logger.warning("Continuing without alignment")
            elif alignMethod == 'time':
                try:
                    # Calculate time difference and convert to sample points
                    timedelta = Time_start - pd.Timestamp('2023-'+self.__date__+' '+self.segInfo['Start']['Seg 0'])
                    lag = round(timedelta.total_seconds() * self.__fs__)
                    
                    # Move all motion channels by the calculated lag
                    for ibody in NameList:
                        for iname in motionName:
                            try:
                                self.move_data(ibody+'.'+iname, point_of_move=lag)
                            except Exception as e:
                                logger.warning(f"Failed to move channel {ibody}.{iname}: {str(e)}")
                except Exception as e:
                    logger.error(f"Failed to align using time method: {str(e)}")
                    logger.warning("Continuing without alignment")
            elif alignMethod != 'none':
                logger.warning(f"Unknown alignment method: {alignMethod}. No alignment applied.")
        except Exception as e:
            logger.error(f"Error during alignment: {str(e)}")
            logger.warning("Continuing without alignment")

        # Apply lowpass filter if requested
        if lowpassfilter > 0:
            try:
                for ibody in NameList:
                    for iname in motionName:
                        try:
                            self.apply_lowpass_filter(ibody+'.'+iname, cutoffull=lowpassfilter)
                        except Exception as e:
                            logger.warning(f"Failed to apply filter to {ibody}.{iname}: {str(e)}")
            except Exception as e:
                logger.error(f"Error applying lowpass filter: {str(e)}")
                logger.warning("Continuing without filtering")

        return None

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
            # 使用pandas的优化方法一次性计算所有统计量
            try:
                # 获取数据帧
                data_frame = self.data[sseg]
                
                # 检查数据大小，对于大型数据使用分块处理
                if data_frame.shape[0] * data_frame.shape[1] > 10000000:  # 阈值可调整
                    # 使用dask进行大数据并行计算
                    try:
                        import dask.dataframe as dd
                        
                        # 将pandas DataFrame转换为dask DataFrame
                        dask_df = dd.from_pandas(data_frame, npartitions=min(32, data_frame.shape[1]))
                        
                        # 并行计算统计量
                        mean_result = dask_df.mean().compute()
                        std_result = dask_df.std().compute()
                        max_result = dask_df.max().compute()
                        min_result = dask_df.min().compute()
                        
                        # 创建结果DataFrame
                        stats = pd.DataFrame({
                            'Mean': mean_result,
                            'Std': std_result,
                            'Max': max_result,
                            'Min': min_result
                        })
                        
                        # 添加单位列
                        stats['Unit'] = self.chInfo.set_index('Name')['Unit']
                        
                        # 更新统计信息
                        self.segStatis[sseg] = stats
                        
                    except ImportError:
                        # 如果dask不可用，使用分块处理
                        logger.info("Dask not available, using chunked processing for large dataset")
                        
                        # 分块大小
                        chunk_size = 1000000 // data_frame.shape[1]
                        chunk_size = max(chunk_size, 1000)  # 确保至少有1000行
                        
                        # 初始化结果
                        means = pd.Series(index=data_frame.columns)
                        stds = pd.Series(index=data_frame.columns)
                        maxs = pd.Series(index=data_frame.columns)
                        mins = pd.Series(index=data_frame.columns)
                        
                        # 分块处理
                        n_chunks = (data_frame.shape[0] + chunk_size - 1) // chunk_size
                        
                        # 使用Welford算法进行在线计算均值和标准差
                        count = 0
                        M2 = pd.Series(0, index=data_frame.columns)
                        mean = pd.Series(0, index=data_frame.columns)
                        
                        # 初始化最大最小值
                        maxs = data_frame.iloc[0]
                        mins = data_frame.iloc[0]
                        
                        for i in range(n_chunks):
                            start_idx = i * chunk_size
                            end_idx = min((i + 1) * chunk_size, data_frame.shape[0])
                            chunk = data_frame.iloc[start_idx:end_idx]
                            
                            # 更新最大最小值
                            maxs = pd.concat([maxs, chunk.max()]).max(level=0)
                            mins = pd.concat([mins, chunk.min()]).min(level=0)
                            
                            # 更新均值和方差（Welford算法）
                            for _, row in chunk.iterrows():
                                count += 1
                                delta = row - mean
                                mean += delta / count
                                delta2 = row - mean
                                M2 += delta * delta2
                        
                        # 计算标准差
                        stds = np.sqrt(M2 / count)
                        means = mean
                        
                        # 创建结果DataFrame
                        stats = pd.DataFrame({
                            'Mean': means,
                            'Std': stds,
                            'Max': maxs,
                            'Min': mins,
                            'Unit': self.chInfo.set_index('Name')['Unit']
                        })
                        
                        # 更新统计信息
                        self.segStatis[sseg] = stats
                else:
                    # 对于小型数据，使用pandas的优化方法
                    # 并行计算统计量
                    stats = data_frame.agg(['mean', 'std', 'max', 'min'])
                    
                    # 转置结果，使其与所需格式匹配
                    stats = stats.T
                    stats.columns = ['Mean', 'Std', 'Max', 'Min']
                    
                    # 添加单位列
                    stats['Unit'] = self.chInfo.set_index('Name')['Unit']
                    
                    # 更新统计信息
                    self.segStatis[sseg] = stats
                
            except Exception as e:
                logger.error(f"统计计算错误: {str(e)}")
                # 回退到原始方法
                data_frame = self.data[sseg]
                means = data_frame.mean()
                stds = data_frame.std()
                maxs = data_frame.max()
                mins = data_frame.min()
                
                # 创建统计数据DataFrame
                stats_data = {
                    'Mean': means,
                    'Std': stds,
                    'Max': maxs,
                    'Min': mins,
                    'Unit': self.chInfo.set_index('Name')['Unit']
                }
                
                # 更新统计信息
                self.segStatis[sseg] = pd.DataFrame(stats_data)
        else:
            # 只更新指定通道的统计信息
            if chName in self.chInfo['Name'].values:
                # 使用pandas的Series方法快速计算统计量
                series = self.data[sseg][chName]
                
                # 对于大型序列，使用分块处理
                if len(series) > 10000000:  # 阈值可调整
                    # 分块大小
                    chunk_size = 1000000
                    
                    # 初始化结果
                    count = 0
                    mean = 0
                    M2 = 0
                    max_val = series.iloc[0]
                    min_val = series.iloc[0]
                    
                    # 分块处理
                    n_chunks = (len(series) + chunk_size - 1) // chunk_size
                    
                    for i in range(n_chunks):
                        start_idx = i * chunk_size
                        end_idx = min((i + 1) * chunk_size, len(series))
                        chunk = series.iloc[start_idx:end_idx]
                        
                        # 更新最大最小值
                        max_val = max(max_val, chunk.max())
                        min_val = min(min_val, chunk.min())
                        
                        # 更新均值和方差（Welford算法）
                        for val in chunk:
                            count += 1
                            delta = val - mean
                            mean += delta / count
                            delta2 = val - mean
                            M2 += delta * delta2
                    
                    # 计算标准差
                    std = np.sqrt(M2 / count)
                    
                    stats = {
                        'mean': mean,
                        'std': std,
                        'max': max_val,
                        'min': min_val
                    }
                else:
                    # 对于小型序列，直接使用pandas方法
                    stats = series.agg(['mean', 'std', 'max', 'min'])
                
                # 获取单位
                unit_idx = self.chInfo['Name'].values == chName
                unit = self.chInfo.loc[unit_idx, 'Unit'].values[0]
                
                # 更新统计信息
                self.segStatis[sseg].loc[chName] = [
                    stats['mean'], stats['std'], stats['max'], stats['min'], unit]
            else:
                logger.error(f'ERROR! {chName:8s} not found.')

        return None

    def updateChN(self, sseg=0):
        """
        Update channel count information.
        
        Parameters:
        -----------
        sseg : int, optional
            Segment index to process, default is 0
            
        Notes:
        ------
        - Updates channel count in segment information
        - Validates channel consistency
        - Maintains data integrity
        """
        if self.data[sseg].shape[1] == self.chInfo.shape[0] == self.segStatis[0].shape[0]:
            self.__chN__ = self.chInfo.shape[0]
        else:
            raise ValueError("Number of channels does not match!")

        return None

    def rename_channel(self,
                     chOld,
                     chNew,
                     sseg=0):
        """
        Rename a channel in the dataset.
        
        Parameters:
        -----------
        chOld : str
            Original channel name
        chNew : str
            New channel name
        sseg : int, optional
            Segment index to process, default is 0
            
        Notes:
        ------
        - Updates channel name in data, chInfo and statistics
        - Maintains all data and properties
        """
        try:
            # Check if the old channel exists
            if chOld not in self.data[sseg].columns:
                logger.error(f"Channel '{chOld}' not found in segment {sseg}")
                raise KeyError(f"Channel '{chOld}' not found")
                
            # Check if the new channel name already exists
            if chNew in self.data[sseg].columns:
                logger.error(f"Channel '{chNew}' already exists in segment {sseg}")
                raise ValueError(f"Channel '{chNew}' already exists")
                
            # 1. 重命名数据DataFrame中的列
            self.data[sseg].rename(columns={chOld: chNew}, inplace=True)
            
            # 2. 更新chInfo中的通道名
            # 找到具有旧通道名的行索引
            ch_mask = self.chInfo['Name'] == chOld
            # 直接使用布尔掩码更新名称
            self.chInfo.loc[ch_mask, 'Name'] = chNew
            
            # 3. 重命名统计表中的索引
            if chOld in self.segStatis[sseg].index:
                self.segStatis[sseg].rename(index={chOld: chNew}, inplace=True)
            
            logger.info(f"Renamed channel '{chOld}' to '{chNew}' in segment {sseg}")
            return True
            
        except Exception as e:
            logger.error(f"Channel renaming failed: {str(e)}")
            return False
    
    def cut_series(self,
                   start,
                   stop,
                 sseg=0):
        """
        Cut a time series to a specified range.
        
        Parameters:
        -----------
        start : str or float
            Start time of the cut (time string or seconds)
        stop : str or float
            End time of the cut (time string or seconds)
        sseg : int, optional
            Segment index, default is 0
            
        Raises:
        -------
        ValueError
            If start or stop times are invalid
        """
        def moveTimestr(Timestr, seconds_float):
            seconds = int(seconds_float)
            milliseconds = int((seconds_float-seconds)*1000)
            startTime = datetime.datetime.strptime(Timestr,"%H:%M:%S.%f")
            startTime_new = (startTime + datetime.timedelta(seconds=seconds, milliseconds=milliseconds)).strftime("%H:%M:%S.%f")
            return startTime_new[:-5]

        startIndx = int(start * self.__fs__)
        stopIndx = int(stop * self.__fs__)
        lngth = self.data[sseg].index[-1]
        self.data[sseg] = self.data[sseg].drop(range(startIndx + 1))
        self.data[sseg] = self.data[sseg].drop(range(stopIndx, lngth + 1))
        self.data[sseg] = self.data[sseg].reset_index(drop=True)

        sampNum = self.data[sseg].shape[0]

        self.segInfo.loc['Seg{0:2d}'.format(
            sseg),'Start'] = moveTimestr(self.segInfo['Start'].values[0], start)
        self.segInfo.loc['Seg{0:2d}'.format(
            sseg),'Stop'] = moveTimestr(self.segInfo['Start'].values[0], stop)
        self.segInfo.loc['Seg{0:2d}'.format(
            sseg),'Duration'] = '{0:8.1f}s'.format((sampNum - 1) / self.__fs__)
        self.segInfo.loc['Seg{0:2d}'.format(
            sseg),'N sample'] = sampNum
        self.updateST(sseg=sseg)
        logger.info('Cut time series from {0:5.2f}s to {1:5.2f}s'.format(
                start, stop))

        return None

    def plot_channel(self, ch_idx, sseg=0, figsize=(10, 6), title=None, 
                    xlabel='Time (s)', ylabel=None, grid=True, 
                    color=None, linewidth=1.0, alpha=0.8,
                    xlim=None, ylim=None, 
                    downsampling=True, max_points=40000, 
                    save_path=None, show=True, plotbackend=None,
                    height=None, width=None, save_html=None,
                    table_width=0.25, column_widths=None,
                    stats=True, dpi=300):
        """
        Plot a channel from the PyDAS object with interactive features.
        
        Parameters:
        -----------
        ch_idx : int, str, or list
            Channel index, name, or list of channel names to plot
        sseg : int, optional
            Segment index to plot, default is 0
        figsize : tuple, optional
            Figure size in inches (width, height), default is (10, 6)
        title : str, optional
            Plot title, default is None (auto-generated)
        xlabel : str, optional
            X-axis label, default is 'Time (s)'
        ylabel : str, optional
            Y-axis label, default is None (auto-generated)
        grid : bool, optional
            Whether to show grid, default is True
        color : str, optional
            Line color for single channel plot, default is 'blue'
        linewidth : float, optional
            Line width, default is 1.0
        alpha : float, optional
            Line transparency, default is 0.8
        xlim : tuple, optional
            X-axis limits (min, max), default is None (auto)
        ylim : tuple, optional
            Y-axis limits (min, max), default is None (auto)
        downsampling : bool, optional
            Whether to enable downsampling for large datasets, default is True
        max_points : int, optional
            Maximum number of points to plot before downsampling, default is 20000
        save_path : str, optional
            Path to save the figure, default is None (don't save)
        show : bool, optional
            Whether to display the plot, default is True
        plotbackend : str, optional
            Plotting backend to use ('plotly', 'matplotlib', 'seaborn', or None for auto), default is None (auto)
        height : int, optional
            Height in pixels for plot, default is None (auto)
        width : int, optional
            Width in pixels for plot, default is None (auto)
        save_html : str, optional
            Path to save the interactive HTML plot, default is None (don't save)
        table_width : float, optional
            Relative width of the statistics table (0-1), default is 0.25
        column_widths : list, optional
            Relative widths for each column in the statistics table, default is None
        stats : bool, optional
            Whether to show statistics, default is True
        dpi : int, optional
            DPI for saved image, default is 300
            
        Returns:
        --------
        Figure object (matplotlib.figure.Figure or plotly.graph_objects.Figure)
        """
        try:
            # 验证通道并转换为通道名称
            ch_name = validate_channel(self, ch_idx)
            if ch_name is None:
                return None
            
            # 调用plot模块的plot_channel函数
            plot_channel(
                pydas_obj=self,
                ch_name=ch_name,
                sseg=sseg,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                xlim=xlim,
                ylim=ylim,
                grid=grid,
                show=show,
                save_path=save_path,
                plotbackend=plotbackend,
                downsampling=downsampling,
                max_points=max_points,
                save_html=save_html,
                dpi=dpi,
                width=width,
                height=height,
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                figsize=figsize,
                stats=stats,
                table_width=table_width,
                column_widths=column_widths
            )
            return None
        except ImportError as e:
            logger.error(f"Plot module not found: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error in plot_channel: {str(e)}")
            return None

    def plot_histogram(self, ch_idx, sseg=0, title=None, xlabel=None, ylabel='Count',
                    bins=50, xlim=None, ylim=None, grid=True, show=True, 
                    save_path=None, plotbackend=None, save_html=None, 
                    dpi=300, width=None, height=None, color=None, 
                    alpha=0.6, figsize=(12, 6), fit_gaussian=True, fit_color='red'):
        """
        Plot a histogram of a channel from the PyDAS object.
        
        Parameters:
        -----------
        ch_idx : int, str, or list
            Channel index, name, or list of channel indices/names to plot histogram for
        sseg : int, optional
            Segment index to plot, default is 0
        title : str, optional
            Plot title, default is None (auto-generated)
        xlabel : str, optional
            X-axis label, default is None (auto-generated from channel name and unit)
        ylabel : str, optional
            Y-axis label, default is 'Count'
        bins : int, optional
            Number of histogram bins, default is 50
        xlim : tuple, optional
            X-axis limits as (min, max), default is None (auto)
        ylim : tuple, optional
            Y-axis limits as (min, max), default is None (auto)
        grid : bool, optional
            Whether to show grid, default is True
        show : bool, optional
            Whether to display the plot, default is True
        save_path : str, optional
            Path to save the plot, default is None (don't save)
        plotbackend : str, optional
            Plotting backend to use ('plotly', 'matplotlib', 'seaborn', or None for auto), default is None (auto)
        save_html : str, optional
            Path to save the interactive HTML plot, default is None (don't save)
        dpi : int, optional
            DPI for saved image, default is 300
        width : int, optional
            Width in pixels for plot, default is None (auto)
        height : int, optional
            Height in pixels for plot, default is None (auto)
        color : str or list, optional
            Histogram color or list of colors, default is None (auto-generated)
        alpha : float, optional
            Histogram transparency, default is 0.6
        figsize : tuple, optional
            Figure size for matplotlib in inches, default is (12, 6)
        fit_gaussian : bool, optional
            Whether to fit a Gaussian distribution to the data, default is True
        fit_color : str or list, optional
            Color of the Gaussian fit curve or list of colors, default is 'red'
            
        Returns:
        --------
        Figure object (matplotlib.figure.Figure or plotly.graph_objects.Figure)
        """
        try:
            # 验证通道并转换为通道名称
            ch_name = validate_channel(self, ch_idx)
            if ch_name is None:
                return None
            
            # 调用plot模块的plot_histogram函数
            plot_histogram(
                pydas_obj=self,
                ch_name=ch_name,
                sseg=sseg,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                bins=bins,
                xlim=xlim,
                ylim=ylim,
                grid=grid,
                show=show,
                save_path=save_path,
                plotbackend=plotbackend,
                save_html=save_html,
                dpi=dpi,
                width=width,
                height=height,
                color=color,
                alpha=alpha,
                figsize=figsize,
                fit_gaussian=fit_gaussian,
                fit_color=fit_color
            )
            return None
        except ImportError as e:
            logger.error(f"Plot module not found: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error in plot_histogram: {str(e)}")
            return None
            
    def plot_xy(self, x_ch_idx, y_ch_idx, sseg=0, title=None, 
              xlabel=None, ylabel=None, xlim=None, ylim=None, grid=True, 
              show=True, save_path=None, plotbackend=None, save_html=None,
              dpi=300, width=None, height=None, color='blue', alpha=0.8, 
              marker_size=5, figsize=(8, 8), line=False, fit_line=False,
              fit_color='red', fit_line_width=2, fit_alpha=0.8,
              show_stats=False, downsampling=True, max_points=10000,
              density_plot=False, density_colorscale='Viridis', 
              density_opacity=0.7, use_webgl=True, adaptive_sampling=False,
              datashade=False, contour_levels=20, sampling_algorithm='lttb',
              memory_efficient=True, bin_size=None, sns_style=None,
              sns_bins=50, sns_pthresh=0.1, sns_cmap=None,
              sns_contour_levels=5, sns_contour_color=None, sns_linewidths=None):
        """
        Create an XY scatter plot with one channel on the X-axis and another on the Y-axis.
        
        Args:
            x_ch_idx (int, str): Index or name of the channel to plot on the X-axis.
            y_ch_idx (int, str): Index or name of the channel to plot on the Y-axis.
            sseg (int, optional): Segment index. Defaults to 0.
            title (str, optional): Plot title. Defaults to None.
            xlabel (str, optional): X-axis label. Defaults to None (uses channel name).
            ylabel (str, optional): Y-axis label. Defaults to None (uses channel name).
            xlim (tuple, optional): X-axis limits (min, max). Defaults to None.
            ylim (tuple, optional): Y-axis limits (min, max). Defaults to None.
            grid (bool, optional): Show grid. Defaults to True.
            show (bool, optional): Display the plot. Defaults to True.
            save_path (str, optional): Path to save the plot. Defaults to None.
            plotbackend (str, optional): Plotting backend to use ('plotly', 'matplotlib', 'seaborn', or None for auto). Defaults to None.
            save_html (str, optional): Path to save interactive HTML plot. Defaults to None.
            dpi (int, optional): DPI for saved plot. Defaults to 300.
            width (int, optional): Width of the plot in pixels. Defaults to None.
            height (int, optional): Height of the plot in pixels. Defaults to None.
            color (str, optional): Color for the scatter points. Defaults to 'blue'.
            alpha (float, optional): Opacity for the scatter points. Defaults to 0.8.
            marker_size (float, optional): Size of the scatter points. Defaults to 5.
            figsize (tuple, optional): Figure size for matplotlib. Defaults to (8, 8).
            line (bool, optional): Connect points with lines. Defaults to False.
            fit_line (bool, optional): Show linear regression fit line. Defaults to False.
            fit_color (str, optional): Color for fit line. Defaults to 'red'.
            fit_line_width (float, optional): Width of fit line. Defaults to 2.
            fit_alpha (float, optional): Opacity of fit line. Defaults to 0.8.
            show_stats (bool, optional): Show statistical information on the plot. Defaults to False.
            downsampling (bool, optional): Apply downsampling for large datasets. Defaults to True.
            max_points (int, optional): Maximum number of points to show before downsampling. Defaults to 10000.
            density_plot (bool, optional): Show density contour plot for large datasets. Defaults to False.
            density_colorscale (str, optional): Colorscale for density plot. Defaults to 'Viridis'.
            density_opacity (float, optional): Opacity for density contours. Defaults to 0.7.
            use_webgl (bool, optional): Use WebGL rendering for better performance with large datasets. Defaults to True.
            adaptive_sampling (bool, optional): Use adaptive sampling to preserve signal features. Defaults to False.
            datashade (bool, optional): Use datashading for very large datasets (>100k points). Defaults to False.
            contour_levels (int, optional): Number of contour levels for density plot. Defaults to 20.
            sampling_algorithm (str, optional): Algorithm for downsampling: 'lttb' (Largest Triangle Three Buckets), 
                                               'uniform', or 'peak'. Defaults to 'lttb'.
            memory_efficient (bool, optional): Use memory-efficient methods for very large datasets. Defaults to True.
            bin_size (tuple, optional): Bin size for 2D histogram (x_bins, y_bins). Defaults to None (auto).
            sns_style (str, optional): Seaborn style theme. Defaults to None.
            sns_bins (int, optional): Number of bins for Seaborn histplot. Defaults to 50.
            sns_pthresh (float, optional): Threshold for Seaborn histplot. Defaults to 0.1.
            sns_cmap (str, optional): Colormap for Seaborn histplot. Defaults to None.
            sns_contour_levels (int, optional): Number of levels for Seaborn kdeplot. Defaults to 5.
            sns_contour_color (str, optional): Color of contour lines for Seaborn kdeplot. Defaults to None.
            sns_linewidths (float, optional): Line width for Seaborn kdeplot. Defaults to None.
            
        Returns:
            tuple: (pandas.DataFrame with x and y data, figure object)
        """
        try:
            # 验证X和Y通道并转换为通道名称
            x_ch_name = validate_channel(self, x_ch_idx)
            y_ch_name = validate_channel(self, y_ch_idx)
            
            if x_ch_name is None or y_ch_name is None:
                return None
            
            # 调用plot模块的plot_xy函数
            return plot_xy(
                pydas_obj=self,
                x_ch_name=x_ch_name,
                y_ch_name=y_ch_name,
                sseg=sseg,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                xlim=xlim,
                ylim=ylim,
                grid=grid,
                show=show,
                save_path=save_path,
                plotbackend=plotbackend,
                save_html=save_html,
                dpi=dpi,
                width=width,
                height=height,
                color=color,
                alpha=alpha,
                marker_size=marker_size,
                figsize=figsize,
                line=line,
                fit_line=fit_line,
                fit_color=fit_color,
                fit_line_width=fit_line_width,
                fit_alpha=fit_alpha,
                show_stats=show_stats,
                downsampling=downsampling,
                max_points=max_points,
                density_plot=density_plot,
                density_colorscale=density_colorscale,
                density_opacity=density_opacity,
                use_webgl=use_webgl,
                adaptive_sampling=adaptive_sampling,
                datashade=datashade,
                contour_levels=contour_levels,
                sampling_algorithm=sampling_algorithm,
                memory_efficient=memory_efficient,
                bin_size=bin_size,
                sns_style=sns_style,
                sns_bins=sns_bins,
                sns_pthresh=sns_pthresh,
                sns_cmap=sns_cmap,
                sns_contour_levels=sns_contour_levels,
                sns_contour_color=sns_contour_color,
                sns_linewidths=sns_linewidths
            )
        except ImportError as e:
            logger.error(f"Plot module not found: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error in plot_xy: {str(e)}")
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
        
        # 确保系数是浮点数
        try:
            C1 = float(trans_temp[1][0])  # CoeffUnit
            C2 = float(rho ** trans_temp[1][1])  # CoeffRho
            C3 = float(lam ** trans_temp[1][2])  # CoeffLam
            coeff = C1 * C2 * C3
            logger.debug(f"Conversion coefficients: C1={C1}, C2={C2}, C3={C3}, total={coeff}")
        except Exception as e:
            logger.error(f"Error converting coefficients: {str(e)}")
            # 使用默认值
            coeff = 1.0
            logger.warning(f"Using default coefficient value: {coeff}")
        
        # Calculate time array based on the scaling
        fs_scaled = self.__fs__ / np.sqrt(lam)
        
        # 只处理第一段数据（如果用户需要多段，可以拓展此功能）
        idx1 = 0
        if self.__segN__ > 1:
            logger.info(f"Multiple segments found. Only converting first segment.")
            
        # Extract original data and ensure it's a float64 numpy array
        try:
            # 确保获取的是numpy数组而不是pandas Series
            data_raw = self.data[idx1][channel_name]
            if hasattr(data_raw, 'values'):
                data = data_raw.values
            else:
                data = np.array(data_raw)
                
            # 检查数据类型并转换为float64
            if not np.issubdtype(data.dtype, np.floating):
                logger.debug(f"Converting data from {data.dtype} to float64")
                data = data.astype(np.float64)
            else:
                data = data.copy()
                
            # 检查数据是否有nan或inf
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
        
        # 创建TimeSeries对象
        try:
            # 确保单位名称是字符串
            unit_name = str(trans_temp[0]) if trans_temp and trans_temp[0] is not None else unit
            
            # TimeSeries构造函数需要data和args参数，其中args是时间向量
            ts = TimeSeries(data_scaled, T)
            return ts
        except Exception as e:
            logger.error(f"Error creating TimeSeries object: {str(e)}")
            return None

    def spectral_analysis(self, channel_name, method='cov', L=1024, plot=False, title=None, 
                         save_path=None, plotbackend=None, save_html=None,
                         fullscale=False, lam=None, rho=1.025, g=9.807, freq_range=(0, 2)):
        """
        Perform spectral analysis on a single channel and return a spectral data object.
        This method calls the spectral_analysis function from the analysis module.
        
        See analysis.spectral_analysis for full documentation.
        """
        return spectral_analysis(self, channel_name, method, L, plot, title, save_path, 
                               plotbackend, save_html, fullscale, lam, rho, g, freq_range)
    
    def statistic_analysis(self, ch_name, sseg=0, advanced=False, visualization=False, bins=50, 
                          save_fig=False, save_path=None, plotbackend=None, fullscale=False, lam=None, 
                          rho=1.025, g=9.807):
        """
        对通道进行时域统计分析。此方法调用analysis模块中的statistic_analysis函数。
        
        Parameters:
        -----------
        ch_name : str
            要分析的通道名称
        sseg : int, optional
            要分析的数据段索引，默认为0
        advanced : bool, optional
            是否计算高级统计量（偏度、峰度、分位数等），默认为False
        visualization : bool, optional
            是否显示统计量可视化，默认为False
        bins : int, optional
            直方图的箱数，默认为50
        save_fig : bool, optional
            是否保存图形，默认为False
        save_path : str, optional
            图形保存路径，默认为None（当前目录）
        plotbackend : str, optional
            绘图后端 ('plotly', 'matplotlib', 'seaborn' 或 None 自动选择)，默认为None
        fullscale : bool, optional
            是否转换为原型尺度，默认为False
        lam : float, optional
            尺度系数，仅在fullscale=True时使用，默认为None（使用对象的__lam__属性）
        rho : float, optional
            水密度(kg/m³)，仅在fullscale=True时使用，默认为1.025
        g : float, optional
            重力加速度(m/s²)，仅在fullscale=True时使用，默认为9.807
            
        完整文档请参见analysis.statistic_analysis。
        """
        # 如果fullscale=True但未提供lam参数，使用对象的__lam__属性
        if fullscale and lam is None:
            lam = self.__lam__
            
        return statistic_analysis(self, ch_name, sseg, advanced, visualization, bins, 
                                save_fig, save_path, plotbackend, fullscale, lam, rho, g)

    def print_statistics(self, printTxt=False, printExcel=False):
        """
        Print and optionally export statistical information for all channels.
        
        Parameters:
        -----------
        printTxt : bool, optional
            If True, export statistics to a text file, default is False
        printExcel : bool, optional
            If True, export statistics to an Excel file, default is False
            
        Returns:
        --------
        None
            Statistics are printed to the console and optionally exported to files
        """
        # Update statistics for all segments
        self.updateST(sseg=0)
        
        # Print separator line and segment count
        logger.info(f'Segment total: {self.__segN__:02d}')
        
        # Print statistics for each segment
        for idx, segment_stats in enumerate(self.segStatis):
            logger.info(f'Seg{idx:02d}')
            logger.info('\n' + segment_stats.to_string(float_format=lambda x: f"% .3E" % x, justify='center'))
        
        
        # Export to files if requested
        if printTxt or printExcel:
            # Prepare file path
            path = os.getcwd()
            base_filename = os.path.splitext(self.__filename__)[0]
            
            # Export to text file
            if printTxt:
                txt_filename = f"{path}/{base_filename}_statistic.txt"
                
                # Write to file
                with open(txt_filename, 'w') as infoFile:
                    infoFile.write(f'Segment total: {self.__segN__:02d}\n')
                    
                    # Write statistics for each segment
                    for idx, segment_stats in enumerate(self.segStatis):
                        infoFile.write('\n')
                        infoFile.write(f'Seg{idx:02d}\n')
                        infoFile.write(segment_stats.to_string(
                            float_format=lambda x: f"% .3E" % x, justify='center'))
                
                logger.info(f"Statistics exported to: {txt_filename}")
            
            # Export to Excel file
            if printExcel:
                excel_filename = f"{path}/{base_filename}_statistic.xlsx"
                
                # Write each segment to a separate sheet
                with pd.ExcelWriter(excel_filename) as writer:
                    for idx, segment_stats in enumerate(self.segStatis):
                        segment_stats.to_excel(writer, sheet_name=f'SEG{idx:02d}')
                
                logger.info(f"Statistics exported to: {excel_filename}")

        return None

    def copy_channel(self, chName, new_chName=None, sseg='all'):
        """
        Copy an existing channel to create a new channel with the same data.
        
        Parameters:
        -----------
        chName : str
            Name of the channel to copy
        new_chName : str, optional
            Name for the new channel. If None, will use original name + "_copy"
        sseg : int, list, or 'all', optional
            Segment(s) to apply the copy operation, default is 'all'
            
        Returns:
        --------
        bool
            True if copy was successful, False otherwise
            
        Notes:
        ------
        - The copy will have the same unit and coefficient as the original channel
        - If a channel with the new name already exists, it will be overwritten
        """
        # Check if source channel exists
        if chName not in self.chInfo['Name'].values:
            logger.warning(f"Channel '{chName}' does not exist.")
            return False
            
        # Create new channel name if not provided
        if new_chName is None:
            new_chName = f"{chName}_copy"
            
        # Get unit and coefficient of original channel
        idx = self.chInfo.index[self.chInfo['Name'] == chName].tolist()[0]
        unit = self.chInfo.loc[idx, 'Unit']
        coef = self.chInfo.loc[idx, 'Coef']
            
        # Determine which segments to process
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
            
        # Delete the channel first if it already exists
        if new_chName in self.chInfo['Name'].values:
            logger.warning(f"Channel '{new_chName}' already exists. Operation canceled.")
            return False
            
        # Copy channel data for first segment
        first_seg = segments[0]
        series = self.data[first_seg][chName].copy()
        self.add_channel(new_chName, unit, series, self.__fs__, coef, 0, first_seg)
        
        # For additional segments (if any), manually copy the data
        for seg in segments[1:]:
            if chName in self.data[seg].columns:
                # Copy the data for this segment
                self.data[seg][new_chName] = self.data[seg][chName].copy()
                
                # Update statistics for this segment
                self.segStatis[seg].loc[new_chName] = [
                    np.mean(self.data[seg][new_chName]), 
                    np.std(self.data[seg][new_chName]),
                    np.amax(self.data[seg][new_chName]), 
                    np.amin(self.data[seg][new_chName]), 
                    unit
                ]
                
        logger.info(f"Channel '{chName}' copied to '{new_chName}'")
        return True
        
    def channel_calculate(self, ch1, ch2, operation, new_chName, sseg=0):
        """
        对两个通道执行数学运算并创建新的通道
        
        Parameters:
        -----------
        ch1 : str
            第一个通道名称
        ch2 : str
            第二个通道名称
        operation : str
            要执行的运算，可选值: 
            - 'add'或'+': 加法
            - 'subtract'或'-': 减法
            - 'multiply'或'*': 乘法
            - 'divide'或'/': 除法
        new_chName : str
            新通道的名称
        sseg : int, list, or 'all', optional
            要处理的数据段，默认为0
            
        Returns:
        --------
        bool
            如果操作成功返回True，否则返回False
            
        Notes:
        ------
        - 加减运算要求两个通道的单位相同
        - 乘除运算不要求通道单位相同，会自动计算新的单位
        - 如果新通道名称已存在，操作将被取消
        """
        # 检查通道是否存在
        if ch1 not in self.chInfo['Name'].values:
            logger.warning(f"Channel '{ch1}' does not exist.")
            return False
            
        if ch2 not in self.chInfo['Name'].values:
            logger.warning(f"Channel '{ch2}' does not exist.")
            return False
            
        # 检查操作类型并转换符号
        ops_map = {
            '+': 'add',
            '-': 'subtract',
            '*': 'multiply',
            '/': 'divide'
        }
        
        if operation in ops_map:
            operation = ops_map[operation]
        elif operation not in ['add', 'subtract', 'multiply', 'divide']:
            valid_operations = ["'add'或'+'", "'subtract'或'-'", "'multiply'或'*'", "'divide'或'/'"]
            logger.warning(f"Invalid operation '{operation}'. Valid operations are: {valid_operations}")
            return False
            
        # 检查新通道名是否已存在
        if new_chName in self.chInfo['Name'].values:
            logger.warning(f"Channel '{new_chName}' already exists. Operation canceled.")
            return False
            
        # 获取通道信息
        ch1_idx = self.chInfo.index[self.chInfo['Name'] == ch1].tolist()[0]
        ch2_idx = self.chInfo.index[self.chInfo['Name'] == ch2].tolist()[0]
        
        ch1_unit = self.chInfo.loc[ch1_idx, 'Unit']
        ch2_unit = self.chInfo.loc[ch2_idx, 'Unit']
        
        # 检查单位一致性（仅加减运算）
        if operation in ['add', 'subtract'] and ch1_unit != ch2_unit:
            logger.warning(f"Cannot {operation} channels with different units: '{ch1_unit}' and '{ch2_unit}'")
            return False
            
        # 确定新通道的单位
        if operation in ['add', 'subtract']:
            new_unit = ch1_unit
        elif operation == 'multiply':
            # 单位相乘
            if ch1_unit == '-' or ch2_unit == '-':
                new_unit = ch1_unit if ch2_unit == '-' else ch2_unit
            elif ch1_unit == '' or ch2_unit == '':
                new_unit = ch1_unit if ch2_unit == '' else ch2_unit
            else:
                new_unit = f"{ch1_unit}·{ch2_unit}"
        elif operation == 'divide':
            # 单位相除
            if ch1_unit == '-' or ch1_unit == '':
                new_unit = '-'
            elif ch2_unit == '-' or ch2_unit == '':
                new_unit = ch1_unit
            else:
                new_unit = f"{ch1_unit}/{ch2_unit}"
                
        # 确定要处理的数据段
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
            
        # 初始化成功标志
        success = True
            
        # 处理第一个段并创建新通道
        first_seg = segments[0]
        
        try:
            # 执行运算
            if operation == 'add':
                result = self.data[first_seg][ch1] + self.data[first_seg][ch2]
            elif operation == 'subtract':
                result = self.data[first_seg][ch1] - self.data[first_seg][ch2]
            elif operation == 'multiply':
                result = self.data[first_seg][ch1] * self.data[first_seg][ch2]
            elif operation == 'divide':
                # 处理除零问题
                divisor = self.data[first_seg][ch2].copy()
                # 将零值替换为NaN以避免除零错误
                divisor = divisor.replace(0, np.nan)
                result = self.data[first_seg][ch1] / divisor
                # 将NaN值替换为0
                result = result.fillna(0)
                
            # 添加新通道
            # 系数设为1.0，因为已经进行了计算
            self.add_channel(new_chName, new_unit, result.values, self.__fs__, 1.0, 0, first_seg)
                
            # 处理其他段（如果有）
            for seg in segments[1:]:
                if ch1 in self.data[seg].columns and ch2 in self.data[seg].columns:
                    # 执行运算
                    if operation == 'add':
                        result = self.data[seg][ch1] + self.data[seg][ch2]
                    elif operation == 'subtract':
                        result = self.data[seg][ch1] - self.data[seg][ch2]
                    elif operation == 'multiply':
                        result = self.data[seg][ch1] * self.data[seg][ch2]
                    elif operation == 'divide':
                        # 处理除零问题
                        divisor = self.data[seg][ch2].copy()
                        # 将零值替换为NaN以避免除零错误
                        divisor = divisor.replace(0, np.nan)
                        result = self.data[seg][ch1] / divisor
                        # 将NaN值替换为0
                        result = result.fillna(0)
                        
                    # 添加数据到新通道
                    self.data[seg][new_chName] = result
                    
                    # 更新统计信息
                    self.segStatis[seg].loc[new_chName] = [
                        np.mean(result), 
                        np.std(result),
                        np.amax(result), 
                        np.amin(result), 
                        new_unit
                    ]
        except Exception as e:
            logger.error(f"Error performing {operation} operation: {str(e)}")
            # 如果已经创建了通道，尝试删除它
            if new_chName in self.chInfo['Name'].values:
                self.delete_channel(new_chName)
            success = False
            
        if success:
            ops_dict = {'add': '+', 'subtract': '-', 'multiply': '*', 'divide': '/'}
            ops_symbol = ops_dict.get(operation, operation)
            logger.info(f"Created new channel '{new_chName}' as {ch1} {ops_symbol} {ch2}")
            
        return success
        
    def channel_apply_function(self, ch, func, new_chName, unit=None, sseg=0):
        """
        对单一通道应用自定义函数并创建新的通道
        
        Parameters:
        -----------
        ch : str
            要处理的通道名称
        func : callable 或 str
            要应用的函数。可以是:
            - 可调用对象(函数), 如 np.square, math.log, lambda x: x**2
            - 字符串表达式，如 "x**2", "np.log10(x)", "np.exp(x)"
        new_chName : str
            新通道的名称
        unit : str, optional
            新通道的单位。如果为None，将根据函数类型尝试推断
        sseg : int, list, or 'all', optional
            要处理的数据段，默认为0
            
        Returns:
        --------
        bool
            如果操作成功返回True，否则返回False
            
        Notes:
        ------
        - 如果函数是字符串表达式，将使用eval进行计算，x代表通道数据
        - 常见函数单位转换：
          - 平方(x^2): 原单位²
          - 开方(sqrt(x)): 原单位^(1/2)
          - 对数(log(x)): 无单位
          - 指数(exp(x)): 与x相关的单位
        - 对于不安全的字符串表达式，将拒绝执行
        """
        import numpy as np
        import math
        
        # 检查通道是否存在
        if ch not in self.chInfo['Name'].values:
            logger.warning(f"Channel '{ch}' does not exist.")
            return False
            
        # 检查新通道名是否已存在
        if new_chName in self.chInfo['Name'].values:
            logger.warning(f"Channel '{new_chName}' already exists. Operation canceled.")
            return False
            
        # 获取通道信息
        ch_idx = self.chInfo.index[self.chInfo['Name'] == ch].tolist()[0]
        ch_unit = self.chInfo.loc[ch_idx, 'Unit']
        
        # 确定要处理的数据段
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
        
        # 识别常见函数并推断单位（如果未提供）
        func_name = None
        if unit is None:
            # 如果是字符串表达式，分析它来推断单位
            if isinstance(func, str):
                func_expr = func.lower().strip()
                if any(x in func_expr for x in ['**2', 'square', 'x*x']):
                    unit = f"{ch_unit}²" if ch_unit not in ['-', ''] else ch_unit
                    func_name = "square"
                elif any(x in func_expr for x in ['**3', 'cube', 'x**3']):
                    unit = f"{ch_unit}³" if ch_unit not in ['-', ''] else ch_unit
                    func_name = "cube"
                elif any(x in func_expr for x in ['sqrt', 'x**0.5', 'x**(1/2)']):
                    unit = f"{ch_unit}^(1/2)" if ch_unit not in ['-', ''] else ch_unit
                    func_name = "square root"
                elif any(x in func_expr for x in ['log', 'ln']):
                    unit = '-'  # 对数无单位
                    func_name = "logarithm"
                elif any(x in func_expr for x in ['exp', 'e**']):
                    unit = '-'  # 指数函数通常改变单位
                    func_name = "exponential"
                elif any(x in func_expr for x in ['sin', 'cos', 'tan']):
                    unit = '-'  # 三角函数无单位
                    func_name = "trigonometric"
                elif any(x in func_expr for x in ['abs', 'fabs']):
                    unit = ch_unit  # 绝对值保持单位不变
                    func_name = "absolute"
                else:
                    unit = '-'  # 默认无法确定单位
                    func_name = "custom"
            else:
                # 如果是可调用对象，尝试通过函数名称推断
                func_str = str(func)
                if 'square' in func_str or 'pow' in func_str:
                    unit = f"{ch_unit}2" if ch_unit not in ['-', ''] else ch_unit
                    func_name = "square"
                elif 'cube' in func_str:
                    unit = f"{ch_unit}3" if ch_unit not in ['-', ''] else ch_unit
                    func_name = "cube"
                elif 'sqrt' in func_str:
                    unit = f"{ch_unit}^(1/2)" if ch_unit not in ['-', ''] else ch_unit
                    func_name = "square root"
                elif 'log' in func_str:
                    unit = '-'
                    func_name = "logarithm"
                elif 'exp' in func_str:
                    unit = '-'
                    func_name = "exponential"
                elif any(x in func_str for x in ['sin', 'cos', 'tan']):
                    unit = '-'
                    func_name = "trigonometric"
                elif 'abs' in func_str:
                    unit = ch_unit
                    func_name = "absolute"
                else:
                    unit = '-'
                    func_name = "custom"
        
        # 初始化成功标志
        success = True
        
        # 处理第一个段并创建新通道
        first_seg = segments[0]
        
        try:
            x = self.data[first_seg][ch].values
            
            # 应用函数
            if callable(func):
                # 直接调用函数
                result = func(x)
            elif isinstance(func, str):
                # 检查字符串表达式安全性
                unsafe_terms = ['import', 'eval', 'exec', 'compile', 'open', 'file', 
                              'os.', 'sys.', 'subprocess', 'shutil', '__']
                if any(term in func for term in unsafe_terms):
                    logger.error(f"Unsafe expression detected: {func}")
                    return False
                
                # 使用eval执行字符串表达式
                x_series = self.data[first_seg][ch]
                
                # 定义一个安全的本地命名空间
                local_vars = {'x': x_series, 'np': np, 'math': math}
                
                try:
                    result = eval(func, {"__builtins__": {}}, local_vars)
                    # 如果结果是pandas.Series，转换为numpy数组
                    if hasattr(result, 'values'):
                        result = result.values
                except Exception as e:
                    logger.error(f"Error evaluating expression '{func}': {str(e)}")
                    return False
            else:
                logger.error(f"Invalid function type: {type(func)}. Must be callable or string.")
                return False
            
            # 添加新通道
            self.add_channel(new_chName, unit, result, self.__fs__, 1.0, 0, first_seg)
            
            # 处理其他段（如果有）
            for seg in segments[1:]:
                if ch in self.data[seg].columns:
                    x = self.data[seg][ch]
                    
                    # 应用函数
                    if callable(func):
                        result = func(x)
                    elif isinstance(func, str):
                        local_vars = {'x': x, 'np': np, 'math': math}
                        result = eval(func, {"__builtins__": {}}, local_vars)
                        if hasattr(result, 'values'):
                            result = result.values
                    
                    # 添加数据到新通道
                    self.data[seg][new_chName] = result
                    
                    # 更新统计信息
                    self.segStatis[seg].loc[new_chName] = [
                        np.mean(result), 
                        np.std(result),
                        np.amax(result), 
                        np.amin(result), 
                        unit
                    ]
        except Exception as e:
            logger.error(f"Error applying function to channel: {str(e)}")
            # 如果已经创建了通道，尝试删除它
            if new_chName in self.chInfo['Name'].values:
                self.delete_channel(new_chName)
            success = False
        
        if success:
            if func_name:
                logger.info(f"Created new channel '{new_chName}' by applying {func_name} function to '{ch}'")
            else:
                logger.info(f"Created new channel '{new_chName}' by applying custom function to '{ch}'")
        
        return success

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