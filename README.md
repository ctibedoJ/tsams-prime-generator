# Prime Indexed Möbius Transformation State Space Theory

A classical Python implementation of the Prime Indexed Möbius Transformation State Space Theory described in Chapters 16-22 of the textbook.

## Overview

This package provides computational tools for working with:

- Cyclotomic fields and their properties
- Prime indexed Möbius transformations and the 420-root structure
- State space theory and transformation mechanics
- The 441-dimensional nodal structure
- Hair braid dynamics and braid invariants
- Hyperbolic priming transformations and energy quantization
- Visualization of mathematical structures and transformations

## Installation

```bash
pip install prime_mobius
```

## Core Components

### 1. Cyclotomic Field Implementation
- `CyclotomicField` class for representing and working with cyclotomic fields Q(ζ_n)
- Operations for field elements (addition, multiplication, conjugation, norm)
- Methods for computing cyclotomic polynomials
- Galois group representation and operations

### 2. Prime Indexed Möbius Transformations
- `MoebiusTransformation` class for representing general Möbius transformations
- `PrimeIndexedMoebiusTransformation` class for the specific transformations indexed by primes
- The 420-root structure with all 81 transformations
- Methods for computing fixed points and other properties

### 3. State Space Theory
- `StateSpace` class for representing the state space
- State transformations and their compositions
- Methods for computing orbits and analyzing their properties
- The 441-dimensional nodal structure and its factorization

### 4. Hair Braid Dynamics
- The 21 hair braid nodes structure
- Braid operations and their algebraic properties
- Methods for computing braid invariants

### 5. Hyperbolic Priming Transformations
- Hyperbolic priming transformations
- Energy quantization functions
- Methods for analyzing the energy spectrum

### 6. Visualization Tools
- Riemann sphere visualization for Möbius transformations
- Orbit visualization in 2D and 3D
- Energy spectrum and statistical analysis plots
- Nodal structure and hair braid visualization
- Interactive animations for dynamic processes

## Usage Examples

### Creating a Cyclotomic Field

```python
from prime_mobius.cyclotomic import CyclotomicField

# Create the cyclotomic field Q(ζ_420)
field = CyclotomicField(420)

# Get the dimension of the field
dimension = field.dimension  # 96

# Get the Galois group structure
galois_group = field.galois_group_structure()
```

### Working with Prime Indexed Möbius Transformations

```python
from prime_mobius.moebius import PrimeIndexedMoebiusTransformation, Root420Structure

# Create a prime indexed Möbius transformation
transformation = PrimeIndexedMoebiusTransformation(11, 420)

# Apply the transformation to a complex number
result = transformation.apply(1 + 2j)

# Create the 420-root structure
root_structure = Root420Structure()

# Get all 81 transformations in the structure
transformations = root_structure.transformations
```

### State Space Analysis

```python
from prime_mobius.state_space import StateSpace, NodalStructure441

# Create a state space
state_space = StateSpace()

# Compute the orbit of a state under a transformation
orbit = state_space.orbit(1 + 2j, transformation)

# Check if the orbit is dense in the state space
is_dense = state_space.is_dense(orbit)

# Create the 441-dimensional nodal structure
nodal_structure = NodalStructure441()

# Get the 21 hair braid nodes
hair_braid_nodes = nodal_structure.get_hair_braid_nodes()
```

### Hair Braid Dynamics

```python
from prime_mobius.hair_braid import HairBraidSystem, BraidOperations

# Create a hair braid system
hair_braid_system = HairBraidSystem()

# Get a hair braid node
node = hair_braid_system.get_node(0)

# Apply the braid operation between two nodes
result = hair_braid_system.braid_operation(0, 1)

# Create braid operations
braid_operations = BraidOperations()

# Compute the Jones polynomial of a braid
jones_polynomial = braid_operations.jones_polynomial(braid)
```

### Energy Quantization

```python
from prime_mobius.hyperbolic import EnergyQuantization, EnergySpectrum

# Create an energy quantization calculator
energy_calculator = EnergyQuantization()

# Compute the energy of a transformation
energy = energy_calculator.energy(transformation)

# Create an energy spectrum analyzer
energy_spectrum = EnergySpectrum()

# Compute the energy spectrum of the 420-root structure
spectrum = energy_spectrum.compute_spectrum()

# Compute the total energy
total_energy = energy_spectrum.total_energy()
```

### Visualization Examples

```python
from prime_mobius.visualization import (
    plot_mobius_transformation,
    plot_orbit_on_sphere,
    plot_energy_spectrum,
    plot_orbit_2d,
    plot_nodal_structure
)

# Visualize a Möbius transformation on the Riemann sphere
fig, axes = plot_mobius_transformation(transformation, save_path="mobius_transformation.png")

# Visualize an orbit on the Riemann sphere
fig, ax = plot_orbit_on_sphere(orbit, save_path="orbit_on_sphere.png")

# Visualize the energy spectrum
fig, ax = plot_energy_spectrum(save_path="energy_spectrum.png")

# Visualize an orbit in the complex plane
fig, ax = plot_orbit_2d(orbit, save_path="orbit_2d.png")

# Visualize the nodal structure
fig, ax = plot_nodal_structure(save_path="nodal_structure.png")
```

## Running Tests

The package includes a comprehensive test suite to ensure correctness:

```bash
# Run all tests
python -m prime_mobius.tests.run_tests

# Run specific test modules
python -m unittest prime_mobius.tests.test_moebius
python -m unittest prime_mobius.tests.test_cyclotomic
python -m unittest prime_mobius.tests.test_state_space
```

## Demo Scripts

The package includes several demo scripts to showcase its capabilities:

```bash
# Run the basic demo
python -m prime_mobius.examples.prime_mobius_demo

# Run the visualization demo
python -m prime_mobius.examples.visualization_demo
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.