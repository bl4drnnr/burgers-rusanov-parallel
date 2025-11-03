#!/usr/bin/env python3
"""
Main entry point for the Burgers equation solver project.

This script provides a unified interface to run:
- Sequential Rusanov solver
- Parallel MPI Rusanov solver
- Visualization of results
- Performance analysis
"""

import argparse
import sys
import subprocess
from pathlib import Path


def run_sequential(args):
    """Run sequential solver."""
    cmd = [
        'python3', '1_sequential_rusanov.py',
        '--nx', str(args.nx),
        '--t-final', str(args.t_final),
        '--cfl', str(args.cfl),
        '--nu', str(args.nu),
        '--ic', args.ic,
        '--snapshots', str(args.snapshots),
        '--save', args.output
    ]

    print("Running sequential solver...")
    result = subprocess.run(cmd)
    return result.returncode


def run_parallel(args):
    """Run parallel MPI solver."""
    cmd = [
        'mpiexec', '-n', str(args.procs),
        'python3', '2_parallel_rusanov.py',
        '--nx', str(args.nx),
        '--t-final', str(args.t_final),
        '--cfl', str(args.cfl),
        '--nu', str(args.nu),
        '--ic', args.ic,
        '--snapshots', str(args.snapshots),
        '--save', args.output
    ]

    print(f"Running parallel solver with {args.procs} processes...")
    result = subprocess.run(cmd)
    return result.returncode


def run_visualization(args):
    """Run visualization."""
    cmd = [
        'python3', '3_visualization.py',
        args.input,
        '--type', args.viz_type
    ]

    if args.output:
        cmd.extend(['--output', args.output])

    if args.viz_type == 'compare' and args.compare_with:
        cmd.extend(['--compare-with', args.compare_with])

    if args.viz_type == 'frames':
        cmd.extend(['--frames-dir', args.frames_dir])

    print(f"Running visualization ({args.viz_type})...")
    result = subprocess.run(cmd)
    return result.returncode


def run_performance(args):
    """Run performance analysis."""
    cmd = ['python3', '4_performance_analysis.py']

    if args.run_study:
        cmd.extend([
            '--run-study',
            '--nx', str(args.nx),
            '--t-final', str(args.t_final),
            '--max-procs', str(args.max_procs),
            '--output-dir', args.output_dir
        ])
    elif args.timing_file:
        cmd.extend([
            '--timing-file', args.timing_file,
            '--output', args.output
        ])
    else:
        print("Error: Either --run-study or --timing-file required")
        return 1

    print("Running performance analysis...")
    result = subprocess.run(cmd)
    return result.returncode


def run_demo(args):
    """Run a complete demonstration."""
    print("\n" + "="*80)
    print("RUNNING BURGERS EQUATION DEMONSTRATION")
    print("="*80 + "\n")

    # Step 1: Run sequential
    print("Step 1: Sequential solver")
    print("-" * 40)
    seq_result = 'demo_sequential.npz'
    seq_args = argparse.Namespace(
        nx=200, t_final=0.5, cfl=0.5, nu=0.01,
        ic='sine', snapshots=10, output=seq_result
    )
    if run_sequential(seq_args) != 0:
        print("Sequential solver failed!")
        return 1

    # Step 2: Run parallel
    print("\n\nStep 2: Parallel solver (4 processes)")
    print("-" * 40)
    par_result = 'demo_parallel.npz'
    par_args = argparse.Namespace(
        nx=200, t_final=0.5, cfl=0.5, nu=0.01,
        ic='sine', snapshots=10, procs=4, output=par_result
    )
    if run_parallel(par_args) != 0:
        print("Parallel solver failed!")
        return 1

    # Step 3: Visualize
    print("\n\nStep 3: Visualization")
    print("-" * 40)
    viz_args = argparse.Namespace(
        input=par_result,
        viz_type='shock',
        output='demo_shock_analysis.png',
        compare_with=None,
        frames_dir='frames'
    )
    if run_visualization(viz_args) != 0:
        print("Visualization failed!")
        return 1

    # Step 4: Compare
    print("\n\nStep 4: Sequential vs Parallel comparison")
    print("-" * 40)
    comp_args = argparse.Namespace(
        input=seq_result,
        viz_type='compare',
        output='demo_comparison.png',
        compare_with=par_result,
        frames_dir='frames'
    )
    if run_visualization(comp_args) != 0:
        print("Comparison failed!")
        return 1

    print("\n\n" + "="*80)
    print("DEMONSTRATION COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print(f"  - {seq_result} (sequential results)")
    print(f"  - {par_result} (parallel results)")
    print(f"  - demo_shock_analysis.png (shock wave analysis)")
    print(f"  - demo_comparison.png (sequential vs parallel)")
    print()

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Burgers Equation Solver - Rusanov Method',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run sequential solver
  python main.py sequential --nx 200 --t-final 0.5

  # Run parallel solver with 4 processes
  python main.py parallel --procs 4 --nx 400 --t-final 0.5

  # Visualize results
  python main.py visualize results.npz --viz-type shock

  # Run performance study
  python main.py performance --run-study --max-procs 8

  # Run complete demonstration
  python main.py demo
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Sequential command
    seq_parser = subparsers.add_parser('sequential', help='Run sequential solver')
    seq_parser.add_argument('--nx', type=int, default=200, help='Grid points')
    seq_parser.add_argument('--t-final', type=float, default=0.5, help='Final time')
    seq_parser.add_argument('--cfl', type=float, default=0.5, help='CFL number')
    seq_parser.add_argument('--nu', type=float, default=0.01, help='Viscosity')
    seq_parser.add_argument('--ic', type=str, default='sine',
                           choices=['sine', 'step', 'rarefaction'])
    seq_parser.add_argument('--snapshots', type=int, default=10)
    seq_parser.add_argument('--output', type=str, default='results_sequential.npz')

    # Parallel command
    par_parser = subparsers.add_parser('parallel', help='Run parallel solver')
    par_parser.add_argument('--procs', type=int, default=4, help='Number of MPI processes')
    par_parser.add_argument('--nx', type=int, default=400, help='Global grid points')
    par_parser.add_argument('--t-final', type=float, default=0.5, help='Final time')
    par_parser.add_argument('--cfl', type=float, default=0.5, help='CFL number')
    par_parser.add_argument('--nu', type=float, default=0.01, help='Viscosity')
    par_parser.add_argument('--ic', type=str, default='sine',
                           choices=['sine', 'step', 'rarefaction'])
    par_parser.add_argument('--snapshots', type=int, default=10)
    par_parser.add_argument('--output', type=str, default='results_parallel.npz')

    # Visualization command
    viz_parser = subparsers.add_parser('visualize', help='Visualize results')
    viz_parser.add_argument('input', type=str, help='Input results file')
    viz_parser.add_argument('--viz-type', type=str, default='shock',
                           choices=['evolution', 'shock', 'frames', 'compare'])
    viz_parser.add_argument('--output', type=str, help='Output file')
    viz_parser.add_argument('--compare-with', type=str, help='File to compare with')
    viz_parser.add_argument('--frames-dir', type=str, default='frames')

    # Performance command
    perf_parser = subparsers.add_parser('performance', help='Performance analysis')
    perf_parser.add_argument('--run-study', action='store_true',
                            help='Run full scaling study')
    perf_parser.add_argument('--timing-file', type=str, help='JSON timing file')
    perf_parser.add_argument('--nx', type=int, default=1000)
    perf_parser.add_argument('--t-final', type=float, default=0.2)
    perf_parser.add_argument('--max-procs', type=int, default=8)
    perf_parser.add_argument('--output-dir', type=str, default='scaling_results')
    perf_parser.add_argument('--output', type=str, default='scaling_analysis.png')

    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run complete demonstration')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Execute command
    if args.command == 'sequential':
        return run_sequential(args)
    elif args.command == 'parallel':
        return run_parallel(args)
    elif args.command == 'visualize':
        return run_visualization(args)
    elif args.command == 'performance':
        return run_performance(args)
    elif args.command == 'demo':
        return run_demo(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
