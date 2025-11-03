# Burgers Equation Solver: Rusanov Method with MPI Parallelization

Parallel numerical solver for the hydrodynamic [Burgers equation](https://en.wikipedia.org/wiki/Burgers%27_equation) using the Rusanov method with MPI, featuring shock wave visualization and performance analysis for high-performance computing studies.

## Project Overview

This project implements a numerical solution to the 1D Burgers equation using the **Rusanov method** (first-order finite volume scheme). The implementation includes:

- **Sequential baseline solver** (1_sequential_rusanov.py)
- **Parallel MPI solver** with domain decomposition (2_parallel_rusanov.py)
- **Visualization tools** for shock wave analysis (3_visualization.py)
- **Performance analysis** with speedup and efficiency metrics (4_performance_analysis.py)

### The Burgers Equation

The inviscid Burgers equation: `∂u/∂t + u·∂u/∂x = 0`

Or viscous form: `∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²`

This equation models fluid flow and exhibits shock wave formation (discontinuities in the solution).

### The Rusanov Method

A first-order finite volume scheme with numerical flux:

`F_{i+1/2} = 0.5*(F(u_i) + F(u_{i+1})) - 0.5*α*(u_{i+1} - u_i)`

where `α = max(|u_i|, |u_{i+1}|)` is the maximum wave speed.

Reference: Section 2.1.1 in "THE SOLUTION OF A BURGERS EQUATION.pdf"

## Quick Start

```bash
# Install dependencies
pip install numpy matplotlib mpi4py

# Run complete demonstration
python main.py demo

# Or run individual components
python main.py sequential --nx 200 --t-final 0.5
python main.py parallel --procs 4 --nx 400 --t-final 0.5
python main.py visualize results_parallel.npz --viz-type shock
python main.py performance --run-study --max-procs 8
```

See **USAGE.md** for detailed instructions.

## Project Structure

```
.
├── main.py                       # Unified entry point
├── 1_sequential_rusanov.py       # Sequential baseline solver
├── 2_parallel_rusanov.py         # MPI parallel solver
├── 3_visualization.py            # Visualization tools
├── 4_performance_analysis.py     # Performance metrics
├── README.md                     # This file
├── USAGE.md                      # Detailed usage guide
├── CLAUDE.md                     # AI assistant guidance
├── Makefile                      # Convenience commands
└── pyproject.toml                # Python dependencies
```

## Parallelization Strategy

**Domain Decomposition**: The spatial domain is divided among MPI processes. Each process handles a subdomain and exchanges boundary values (halo/ghost cells) with neighbors using MPI point-to-point communication.

```
Process 0: [ghost | x0, x1, x2, ... | ghost] →
Process 1: [ghost | x3, x4, x5, ... | ghost] →
Process 2: [ghost | x6, x7, x8, ... | ghost]
```

## Performance Metrics

- **Speedup**: S(P) = T(1) / T(P)
- **Efficiency**: E(P) = S(P) / P × 100%
- **Karp-Flatt**: Estimates serial fraction

## Shock Wave Detection

The code automatically detects discontinuities (shock waves) by analyzing solution gradients:
- Strong shocks: max |du/dx| > 10.0
- Moderate: max |du/dx| > 2.0
- Smooth: max |du/dx| < 2.0

## For Presentation

Generate all results for your presentation:
```bash
make demo                    # Run complete demonstration
make performance             # Run scaling study

# This creates:
# - demo_shock_analysis.png (shock wave visualization)
# - demo_comparison.png (sequential vs parallel)
# - scaling_results/scaling_analysis.png (performance metrics)
```

## Requirements

- Python 3.12+
- NumPy ≥ 1.26.0
- Matplotlib ≥ 3.8.0
- mpi4py ≥ 3.1.5
- MPI implementation (OpenMPI or MPICH)

## References

- THE SOLUTION OF A BURGERS EQUATION.pdf (included)
- [Wikipedia: Burgers' equation](https://en.wikipedia.org/wiki/Burgers%27_equation)
- [Rusanov scheme](https://en.wikipedia.org/wiki/Rusanov_scheme)
