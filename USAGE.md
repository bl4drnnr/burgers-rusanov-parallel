# Usage Guide: Burgers Equation Solver

This guide explains how to use the parallel Burgers equation solver with the Rusanov method.

## Installation

1. Install dependencies:
```bash
pip install -e .
```

Or manually:
```bash
pip install numpy matplotlib mpi4py
```

2. Ensure MPI is installed on your system (e.g., OpenMPI, MPICH)

## Quick Start

Run the complete demonstration:
```bash
python main.py demo
```

This will:
1. Run the sequential solver
2. Run the parallel solver (4 processes)
3. Generate shock wave analysis plots
4. Compare sequential vs parallel results

## Individual Components

### 1. Sequential Solver

Run the baseline sequential implementation:

```bash
# Basic usage
python 1_sequential_rusanov.py

# Custom parameters
python 1_sequential_rusanov.py --nx 400 --t-final 0.5 --nu 0.01

# Different initial conditions
python 1_sequential_rusanov.py --ic step      # Step function
python 1_sequential_rusanov.py --ic sine      # Sine wave (default)
python 1_sequential_rusanov.py --ic rarefaction  # Rarefaction wave
```

Via main script:
```bash
python main.py sequential --nx 200 --t-final 0.5 --output results.npz
```

### 2. Parallel MPI Solver

Run with multiple processes:

```bash
# Run with 4 processes
mpiexec -n 4 python 2_parallel_rusanov.py

# Custom configuration
mpiexec -n 8 python 2_parallel_rusanov.py --nx 1000 --t-final 0.3

# Via main script
python main.py parallel --procs 4 --nx 400 --t-final 0.5
```

### 3. Visualization

Create various types of plots:

```bash
# Shock wave analysis (default)
python 3_visualization.py results.npz --type shock --output shock_analysis.png

# Solution evolution
python 3_visualization.py results.npz --type evolution --output evolution.png

# Animation frames
python 3_visualization.py results.npz --type frames --frames-dir frames/

# Compare sequential vs parallel
python 3_visualization.py results_seq.npz --type compare --compare-with results_par.npz

# Via main script
python main.py visualize results.npz --viz-type shock
```

### 4. Performance Analysis

#### Option A: Run Full Scaling Study

Automatically run the solver with different process counts and analyze performance:

```bash
# Run scaling study (1, 2, 4, 8 processes)
python 4_performance_analysis.py --run-study --max-procs 8

# Custom problem size
python 4_performance_analysis.py --run-study --nx 2000 --max-procs 16

# Via main script
python main.py performance --run-study --max-procs 8
```

This will:
- Run simulations with 1, 2, 4, 8 processes
- Collect timing data
- Generate performance plots (speedup, efficiency, Karp-Flatt)
- Save results to `scaling_results/`

#### Option B: Analyze Existing Timing Data

```bash
python 4_performance_analysis.py --timing-file timing_results.json
```

## Complete Workflow Example

```bash
# 1. Run sequential baseline
python main.py sequential --nx 400 --t-final 0.5 --output seq.npz

# 2. Run parallel version (4 processes)
python main.py parallel --procs 4 --nx 400 --t-final 0.5 --output par.npz

# 3. Visualize shock waves
python main.py visualize par.npz --viz-type shock --output analysis.png

# 4. Compare results
python main.py visualize seq.npz --viz-type compare --compare-with par.npz

# 5. Run performance study
python main.py performance --run-study --nx 1000 --max-procs 8
```

## Parameters Reference

### Solver Parameters

- `--nx`: Number of spatial grid points (default: 200-400)
- `--t-final`: Final simulation time (default: 0.5)
- `--cfl`: CFL number for stability (default: 0.5, must be < 1.0)
- `--nu`: Viscosity coefficient (default: 0.01, set to 0 for inviscid)
- `--ic`: Initial condition type (`sine`, `step`, `rarefaction`)
- `--snapshots`: Number of solution snapshots to save (default: 10)

### Initial Conditions

- **sine**: Smooth sine wave that develops shock (best for demonstration)
- **step**: Step function with immediate discontinuity
- **rarefaction**: Rarefaction wave (expansion wave)

### Recommended Settings

For presentation/demonstration:
```bash
--nx 200 --t-final 0.5 --nu 0.01 --ic sine
```

For performance analysis:
```bash
--nx 1000 --t-final 0.2 --nu 0.01
```

For strong shock formation:
```bash
--nx 400 --t-final 0.8 --nu 0.005 --ic step
```

## Understanding the Output

### Shock Wave Detection

The code automatically detects shock waves by analyzing solution gradients:

- **Strong shocks**: Max |du/dx| > 10.0
- **Moderate discontinuities**: Max |du/dx| > 2.0
- **Smooth solution**: Max |du/dx| < 2.0

### Performance Metrics

- **Speedup S(P)**: T(1) / T(P) - how much faster with P processes
- **Efficiency E(P)**: S(P) / P × 100% - percentage of ideal speedup
- **Karp-Flatt metric e**: Estimates serial fraction of code (lower is better)

Good parallel performance:
- Speedup close to P (linear scaling)
- Efficiency > 80%
- Karp-Flatt < 0.1

## Troubleshooting

### MPI not found
```bash
# Install OpenMPI (macOS)
brew install open-mpi

# Install MPICH (Linux)
sudo apt-get install mpich

# Verify installation
mpiexec --version
```

### Dependencies missing
```bash
pip install numpy matplotlib mpi4py
```

### Numerical instability (solution blows up)
- Reduce CFL number: `--cfl 0.3`
- Increase viscosity: `--nu 0.02`
- Use finer grid: `--nx 400`

### Poor parallel performance
- Increase problem size: `--nx 2000`
- Check if enough work per process (nx/P should be > 100)
- Communication overhead dominates for small problems

## Creating Animations

Generate animation frames:
```bash
python 3_visualization.py results.npz --type frames --frames-dir frames/
```

Create video with ffmpeg:
```bash
ffmpeg -framerate 10 -pattern_type glob -i 'frames/frame_*.png' \
       -c:v libx264 -pix_fmt yuv420p burgers_evolution.mp4
```

## File Outputs

- `*.npz`: NumPy compressed archives with solution data
- `*.png`: Visualization plots
- `*.json`: Timing data for performance analysis
- `scaling_results/`: Directory with performance study results

## Advanced Usage

### Manual Timing Collection

```python
import time
import subprocess

timings = {}
for n_procs in [1, 2, 4, 8]:
    start = time.time()
    if n_procs == 1:
        subprocess.run(['python3', '1_sequential_rusanov.py', '--nx', '1000'])
    else:
        subprocess.run(['mpiexec', '-n', str(n_procs),
                       'python3', '2_parallel_rusanov.py', '--nx', '1000'])
    timings[n_procs] = time.time() - start

# Save for analysis
import json
with open('my_timings.json', 'w') as f:
    json.dump({'timings': timings, 'problem_size': 1000}, f)
```

### Custom Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.load('results.npz')
x = data['x']
u = data['u_final']

plt.plot(x, u)
plt.xlabel('x')
plt.ylabel('u')
plt.title('Custom Plot')
plt.savefig('custom.png')
```

## For Your Presentation

1. **Theory**: See `README.md` and PDF reference
2. **Parallelism Challenge**: Domain decomposition with halo exchange
3. **Results**: Run demo or scaling study, use generated plots
4. **Analysis**: Check performance metrics, discuss efficiency vs. process count

Key commands for presentation:
```bash
# Generate all results
python main.py demo
python main.py performance --run-study --max-procs 8

# This creates:
# - demo_shock_analysis.png (show shock waves)
# - demo_comparison.png (sequential vs parallel)
# - scaling_results/scaling_analysis.png (performance metrics)
```
