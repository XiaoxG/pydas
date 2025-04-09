#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyDAS installation script
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pydas",
    version="1.0.1",
    author="SJTU/SKLOE",
    author_email="xiaoxguo@sjtu.edu.cn",
    description="Python data analysis system, used for processing and analyzing large time series data @ SJTU/SKLOE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "scipy>=1.5.0",
        "matplotlib>=3.3.0",
        "plotly>=5.0.0",
        "numba>=0.50.0",
        "dask>=2021.6.0",
        "kaleido>=0.2.0",  # For saving Plotly charts
        "scikit-learn>=0.24.0",  # For machine learning functionality
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b2",
            "flake8>=3.9.0",
        ],
    },
    package_data={
        "pydas_viz": ["*.py"],
    },
) 