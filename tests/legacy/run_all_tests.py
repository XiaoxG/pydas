#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main test runner for PyDAS library.

This script runs all tests for the PyDAS library, including:
- Comprehensive tests
- Specific method tests
- Performance tests

Usage:
    python run_all_tests.py [test_name]
    
    test_name: Optional. Name of the specific test to run.
               If not provided, all tests will be run.
               
    Available tests:
    - comprehensive: Run comprehensive tests
    - read_wavecal: Test read_waveCal method
    - all: All tests (default)
"""

import os
import sys
import time
import importlib
from datetime import datetime

# Add parent directory to path to import PyDAS
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_test(test_name):
    """Run a specific test."""
    print(f"\n{'='*50}")
    print(f"Running {test_name} test...")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    # Convert test_name to lowercase for case-insensitive comparison
    test_name_lower = test_name.lower()
    
    if test_name_lower == 'comprehensive':
        # Import and run comprehensive tests
        try:
            from test_comprehensive import run_tests
            success = run_tests()
        except ImportError as e:
            print(f"Error importing comprehensive test module: {str(e)}")
            return False
    elif test_name_lower in ['read_wavecal', 'readwavecal']:
        # Import and run read_waveCal test
        try:
            from test_read_waveCal import test_read_waveCal
            success = test_read_waveCal()
        except ImportError as e:
            print(f"Error importing read_waveCal test module: {str(e)}")
            return False
    else:
        print(f"Unknown test: {test_name}")
        return False
    
    end_time = time.time()
    
    print(f"\n{test_name} test completed in {end_time - start_time:.2f} seconds")
    print(f"Result: {'SUCCESS' if success else 'FAILURE'}")
    
    return success

def run_all_tests():
    """Run all tests."""
    tests = ['comprehensive', 'read_wavecal']
    results = {}
    
    for test in tests:
        results[test] = run_test(test)
    
    # Print summary
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    
    all_success = True
    for test, success in results.items():
        print(f"{test}: {'SUCCESS' if success else 'FAILURE'}")
        all_success = all_success and success
    
    return all_success

def main():
    """Main function."""
    print(f"Starting PyDAS tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if a specific test was requested
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name.lower() == 'all':
            success = run_all_tests()
        else:
            success = run_test(test_name)
    else:
        # Run all tests by default
        success = run_all_tests()
    
    print(f"\nAll tests completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Overall result: {'SUCCESS' if success else 'FAILURE'}")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 