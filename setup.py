"""
Setup script for the TSAMS Prime Generator package.
"""

from setuptools import setup, find_packages

setup(
    name="tsams-prime-generator",
    version="0.1.0",
    description="A comprehensive toolkit for prime number generation, analysis, and visualization",
    long_description="""
    # TSAMS Prime Generator

    A comprehensive toolkit for prime number generation, analysis, and visualization,
    specifically designed for the Tibedo Structural Algebraic Modeling System (TSAMS) ecosystem.

    ## Features

    - Advanced prime generation algorithms including cyclotomic field-based approaches
    - Quantum-inspired primality testing
    - E8 lattice-based prime sieving
    - Modular forms and L-functions for prime analysis
    - Galois theory applications to prime generation
    - Visualization of prime patterns through various mathematical lenses
    - Comprehensive utilities for prime number analysis
    """,
    long_description_content_type="text/markdown",
    author="NinjaTech AI",
    author_email="info@ninjatech.ai",
    url="https://github.com/ninjatech-ai/tsams-prime-generator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "sympy>=1.8.0",
        "scipy>=1.7.0",
        "ipywidgets>=7.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.5b2",
            "isort>=5.9.0",
            "flake8>=3.9.0",
            "mypy>=0.812",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.2",
            "nbsphinx>=0.8.0",
            "jupyter>=1.0.0",
        ],
    },
)