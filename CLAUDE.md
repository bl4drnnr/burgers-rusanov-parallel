# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an academic project for implementing a numerical solution to the hydrodynamic Burgers equation using the Rusanov method. The project focuses on:
- Implementing the Rusanov method (as described in section 2.1.1 of the reference PDF)
- Parallelizing the algorithm for performance
- Visualizing results to identify discontinuities and shock waves

## Reference Material

The PDF file "THE SOLUTION OF A BURGERS EQUATION.pdf" contains the mathematical foundation and methodology. Section 2.1.1 specifically describes the Rusanov method implementation.

## Project Requirements

1. **Numerical Method**: Implement the Rusanov method for solving the Burgers equation
2. **Parallelization**: The algorithm must be parallelized (consider OpenMP, MPI, or language-specific parallel frameworks)
3. **Visualization**: Results must be presented graphically to show shock wave formation
4. **Analysis**: Identify and analyze discontinuities (shock waves) in the solution

## Development Considerations

- Choice of programming language will depend on performance requirements and available parallel computing libraries
- Common choices: Python (with NumPy/SciPy + multiprocessing/Numba), C/C++ (with OpenMP/MPI), MATLAB, or Julia
- Visualization should clearly show the temporal evolution of the solution and emergence of shock waves
- Consider using appropriate boundary and initial conditions as specified in the reference material
