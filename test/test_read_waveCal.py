#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for PyDAS read_waveCal method.

This script specifically tests the read_waveCal method of the PyDAS class using WC01.out as test data.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

# Add parent directory to path to import PyDAS
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pydas import PyDAS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_read_waveCal.log')
    ]
)
logger = logging.getLogger(__name__)

# Test data file
TEST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'WC01.out')

# Result directory
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')
os.makedirs(RESULT_DIR, exist_ok=True)

def test_read_waveCal():
    """Test the read_waveCal method."""
    logger.info("Starting test for read_waveCal method")
    logger.info("Test data file: %s", TEST_FILE)
    
    # Check if test data file exists
    if not os.path.exists(TEST_FILE):
        logger.error("Test data file not found: %s", TEST_FILE)
        return False
    
    try:
        # Create a PyDAS object with the test data
        logger.info("Creating PyDAS object with test data")
        pydas = PyDAS(TEST_FILE)
        
        # Get original number of channels
        original_chN = pydas.__chN__
        logger.info("Original number of channels: %d", original_chN)
        
        # Use the same file as both the main data and the wave calibration data for testing
        # In a real scenario, these would be different files
        logger.info("Testing read_waveCal method")
        
        # Find a suitable channel for YBname
        yb_channel = None
        for ch_name in pydas.chInfo['Name'].values:
            if 'YB' in ch_name:
                yb_channel = ch_name
                break
        
        if yb_channel is None:
            # If no YB channel found, use the first channel
            yb_channel = pydas.chInfo['Name'].iloc[0]
        
        logger.info("Using channel %s as YBname", yb_channel)
        
        # Call read_waveCal method
        pydas.read_waveCal(
            wavefname=TEST_FILE,
            sseg=0,
            YBname=yb_channel,
            YBcalname=yb_channel,
            alignFlag=True
        )
        
        # Check if new channels were added
        new_chN = pydas.__chN__
        logger.info("New number of channels: %d", new_chN)
        
        # Check if calibration channels were added
        cal_channels = [ch for ch in pydas.chInfo['Name'].values if ch.startswith('Cal.')]
        logger.info("Added calibration channels: %s", cal_channels)
        
        # Verify that calibration channels were added
        if len(cal_channels) > 0:
            logger.info("read_waveCal test: PASSED - Calibration channels were added")
            
            # Plot one of the original channels and its calibration counterpart
            if len(cal_channels) > 0:
                original_channel = cal_channels[0][4:]  # Remove 'Cal.' prefix
                if original_channel in pydas.chInfo['Name'].values:
                    # Plot both channels for comparison
                    html_file = os.path.join(RESULT_DIR, 'test_read_waveCal_comparison.html')
                    # Check which plot_channel method is available with the correct parameters
                    try:
                        # Try plot_channel_optimized first
                        pydas.plot_channel_optimized([original_channel, cal_channels[0]], save_html=html_file)
                        logger.info("Created comparison plot using plot_channel_optimized: %s", html_file)
                    except Exception as e:
                        logger.warning("Could not use plot_channel_optimized: %s", str(e))
                        try:
                            # Try plot_channel with different parameters
                            pydas.plot_channel(ChName=[original_channel, cal_channels[0]], use_plotly=True)
                            logger.info("Created comparison plot using plot_channel")
                        except Exception as e:
                            logger.warning("Could not create comparison plot: %s", str(e))
            
            return True
        else:
            logger.error("read_waveCal test: FAILED - No calibration channels were added")
            return False
            
    except Exception as e:
        logger.error("Error testing read_waveCal method: %s", str(e))
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_read_waveCal()
    sys.exit(0 if success else 1) 