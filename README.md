# TSAMS Prime Generator

A comprehensive toolkit for prime number generation, analysis, and visualization, specifically designed for the Tibedo Structural Algebraic Modeling System (TSAMS) ecosystem.

## Overview

The TSAMS Prime Generator is a specialized Python package that implements advanced mathematical approaches to prime number generation and analysis. It leverages concepts from cyclotomic fields, quantum computing, exceptional Lie algebras, modular forms, and L-functions to provide novel ways of working with prime numbers.

This package is part of the broader TSAMS ecosystem, which represents a paradigm shift from hardware-focused to mathematics-focused quantum computing approaches.

## Key Features

### Advanced Prime Generation Algorithms

- **Cyclotomic Sieve**: Generate primes using properties of cyclotomic fields
- **Quantum Prime Generator**: Simulate quantum interference patterns to identify primes
- **E8 Lattice Sieve**: Use the exceptional Lie algebra E8 and its associated lattice for prime generation
- **Zeta Zeros Generator**: Leverage the relationship between Riemann zeta function zeros and prime distribution

### Primality Testing

- **Modular Forms Test**: Test primality using properties of modular forms
- **L-Function Test**: Use L-functions to determine if a number is prime
- **Classical Methods**: Miller-Rabin, Fermat, Lucas-Lehmer, and other standard tests

### Visualization Tools

- **Cyclotomic Field Visualization**: Map primes onto the complex plane based on cyclotomic fields
- **Quantum Interference Patterns**: Visualize quantum signatures of prime numbers
- **E8 Lattice Projections**: Project primes onto the E8 lattice structure
- **Modular Forms Zeros**: Explore the relationship between modular form zeros and primes
- **L-Function Visualization**: Visualize how L-functions encode information about prime distribution
- **Classical Visualizations**: Ulam spiral, Sacks spiral, prime distribution plots, and more

### Utility Functions

- **Cyclotomic Field Extensions**: Analyze properties of cyclotomic fields and their relation to primes
- **Quantum Fourier Transform**: Implement QFT for prime-related calculations
- **E8 Root System**: Generate and work with the E8 root system
- **Modular Form Coefficients**: Calculate coefficients of modular forms
- **L-Function Zeros**: Compute zeros of L-functions
- **Special Prime Sequences**: Generate twin primes, Mersenne primes, and other special prime types

## Installation

```bash
pip install tsams-prime-generator
```

Or install from source:

```bash
git clone https://github.com/yourusername/tsams-prime-generator.git
cd tsams-prime-generator
pip install -e .
```

## Quick Start

```python
from prime_generator import cyclotomic_sieve, quantum_prime_generator, e8_lattice_sieve
from prime_generator.visualization import cyclotomic_field_visualization, quantum_interference_pattern

# Generate primes using different methods
primes_cyclotomic = cyclotomic_sieve(100, conductor=8)
primes_quantum = quantum_prime_generator(100, qubits=4)
primes_e8 = e8_lattice_sieve(100)

print(f"Cyclotomic sieve: {primes_cyclotomic}")
print(f"Quantum generator: {primes_quantum}")
print(f"E8 lattice sieve: {primes_e8}")

# Create visualizations
cyclotomic_field_visualization(conductor=8, limit=100)
quantum_interference_pattern(limit=100, qubits=4)
```

## Interactive Exploration

The package includes interactive tools for exploring prime numbers:

```python
from prime_generator.visualization import tsams_prime_explorer

# Launch the interactive explorer
tsams_prime_explorer(max_limit=1000)
```

## Mathematical Background

The TSAMS Prime Generator is based on advanced mathematical concepts:

- **Cyclotomic Fields**: Number fields obtained by adjoining roots of unity to the rational numbers
- **Quantum Computing**: Leveraging quantum interference patterns and phase estimation
- **E8 Lattice**: The exceptional Lie algebra E8 and its associated lattice structure
- **Modular Forms**: Complex analytic functions with special transformation properties
- **L-Functions**: Generalizations of the Riemann zeta function with deep connections to prime numbers

## Integration with TSAMS Ecosystem

This package is designed to work seamlessly with other components of the TSAMS ecosystem:

- **TSAMS Core**: Mathematical framework for structural algebraic modeling
- **TSAMS Quantum**: Quantum computing implementations
- **TSAMS Chemistry/Biology/Physics**: Domain-specific applications

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- SymPy
- SciPy
- IPyWidgets (for interactive visualizations)

## License

MIT License

## Citation

If you use this package in your research, please cite:

```
@software{tsams_prime_generator,
  title = {TSAMS Prime Generator},
  url = {https://github.com/yourusername/tsams-prime-generator},
  version = {0.1.0},
  year = {2025},
}
```

## Acknowledgments

This package is part of the Tibedo Structural Algebraic Modeling System (TSAMS).