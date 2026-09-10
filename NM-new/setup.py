#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for nelder-mead-optimization package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_file(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), encoding='utf-8') as f:
        return f.read()

setup(
    name="nelder-mead-optimization",
    version="1.0.0",
    author="Marco Mollinetti",
    author_email="marco.mollinetti@gmail.com",
    description="A comprehensive implementation of Nelder-Mead optimization algorithms with constraint handling",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/mollinetti/nelder-mead-optimization",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.10.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "benchmarks": [
            "scikit-fem>=8.0.0",
            "cmaes>=0.10.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "hypothesis>=6.0.0",
        ],
        "plotting": [
            "matplotlib>=3.3.0",
        ],
    },
    keywords="optimization nelder-mead simplex constrained-optimization metaheuristics",
    project_urls={
        "Bug Reports": "https://github.com/mollinetti/nelder-mead-optimization/issues",
        "Source": "https://github.com/mollinetti/nelder-mead-optimization",
        "Documentation": "https://github.com/mollinetti/nelder-mead-optimization/tree/main/docs",
    },
)
