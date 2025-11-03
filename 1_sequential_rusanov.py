#!/usr/bin/env python3
"""
Sequential implementation of the Rusanov method for solving the 1D Burgers equation.

The inviscid Burgers equation: ∂u/∂t + u·∂u/∂x = 0
Or viscous form: ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²

Rusanov method (first-order finite volume scheme):
- Numerical flux: F_{i+1/2} = 0.5*(F(u_i) + F(u_{i+1})) - 0.5*α*(u_{i+1} - u_i)
- Where α = max(|u_i|, |u_{i+1}|) is the maximum wave speed
- Explicit time integration with CFL condition

Reference: Section 2.1.1 of "THE SOLUTION OF A BURGERS EQUATION.pdf"
"""

import numpy as np
import time
import argparse
from typing import Tuple


class BurgersRusanovSolver:
    """Sequential Rusanov solver for the 1D Burgers equation."""

    def __init__(self, nx: int, domain: Tuple[float, float],
                 t_final: float, cfl: float = 0.5, nu: float = 0.01):
        """
        Initialize the Burgers equation solver.

        Args:
            nx: Number of spatial grid points
            domain: Spatial domain (x_min, x_max)
            t_final: Final simulation time
            cfl: CFL number for stability (typically < 1.0)
            nu: Viscosity coefficient (set to 0 for inviscid)
        """
        self.nx = nx
        self.x_min, self.x_max = domain
        self.t_final = t_final
        self.cfl = cfl
        self.nu = nu

        # Create spatial grid
        self.dx = (self.x_max - self.x_min) / (nx - 1)
        self.x = np.linspace(self.x_min, self.x_max, nx)

        # Solution array
        self.u = np.zeros(nx)

        # Time tracking
        self.t = 0.0
        self.dt = 0.0
        self.n_steps = 0

        # Storage for snapshots
        self.snapshots = []
        self.snapshot_times = []

    def set_initial_condition(self, ic_type: str = 'sine'):
        """
        Set initial condition for the solution.

        Args:
            ic_type: Type of initial condition
                - 'sine': Smooth sine wave that develops shock
                - 'step': Step function
                - 'rarefaction': Rarefaction wave test
        """
        if ic_type == 'sine':
            # Smooth sine wave - will develop shock
            self.u = 0.5 + 0.5 * np.sin(2 * np.pi * self.x)
        elif ic_type == 'step':
            # Step function - immediate shock
            self.u = np.where(self.x < 0.5 * (self.x_min + self.x_max), 1.0, 0.0)
        elif ic_type == 'rarefaction':
            # Rarefaction wave
            self.u = np.where(self.x < 0.5 * (self.x_min + self.x_max), 0.0, 1.0)
        else:
            raise ValueError(f"Unknown initial condition type: {ic_type}")

        # Store initial state
        self.snapshots = [self.u.copy()]
        self.snapshot_times = [0.0]

    def flux(self, u: np.ndarray) -> np.ndarray:
        """
        Compute the flux function F(u) = u²/2 for Burgers equation.

        Args:
            u: Solution values

        Returns:
            Flux values
        """
        return 0.5 * u**2

    def rusanov_flux(self, u_left: np.ndarray, u_right: np.ndarray) -> np.ndarray:
        """
        Compute Rusanov numerical flux at cell interfaces.

        F_{i+1/2} = 0.5*(F(u_L) + F(u_R)) - 0.5*α*(u_R - u_L)
        where α = max(|u_L|, |u_R|)

        Args:
            u_left: Solution values on left side of interface
            u_right: Solution values on right side of interface

        Returns:
            Numerical flux at interfaces
        """
        f_left = self.flux(u_left)
        f_right = self.flux(u_right)

        # Maximum wave speed (local Lax-Friedrichs)
        alpha = np.maximum(np.abs(u_left), np.abs(u_right))

        # Rusanov flux
        return 0.5 * (f_left + f_right) - 0.5 * alpha * (u_right - u_left)

    def compute_dt(self) -> float:
        """
        Compute time step based on CFL condition.

        For Burgers equation: dt <= CFL * dx / max(|u|)
        For viscous term: dt <= dx² / (2*ν)

        Returns:
            Time step size
        """
        # Convection CFL condition
        max_speed = np.max(np.abs(self.u))
        if max_speed > 1e-10:
            dt_convection = self.cfl * self.dx / max_speed
        else:
            dt_convection = self.cfl * self.dx / 1e-10

        # Diffusion stability condition (if viscous)
        if self.nu > 0:
            dt_diffusion = 0.5 * self.dx**2 / self.nu
            return min(dt_convection, dt_diffusion)

        return dt_convection

    def apply_boundary_conditions(self, u: np.ndarray) -> np.ndarray:
        """
        Apply periodic boundary conditions.

        Args:
            u: Solution array

        Returns:
            Solution with boundary conditions applied
        """
        # Periodic boundaries
        u[0] = u[-2]
        u[-1] = u[1]
        return u

    def step(self) -> None:
        """Perform one time step using the Rusanov method."""
        # Compute time step
        self.dt = self.compute_dt()

        # Don't overshoot final time
        if self.t + self.dt > self.t_final:
            self.dt = self.t_final - self.t

        # Create arrays for left and right states at interfaces
        u_left = self.u[:-1]   # u_i
        u_right = self.u[1:]   # u_{i+1}

        # Compute Rusanov fluxes at interfaces
        f_interfaces = self.rusanov_flux(u_left, u_right)

        # Update interior points using conservative form
        u_new = self.u.copy()
        u_new[1:-1] = self.u[1:-1] - (self.dt / self.dx) * (
            f_interfaces[1:] - f_interfaces[:-1]
        )

        # Add viscous term if present (central difference for diffusion)
        if self.nu > 0:
            u_new[1:-1] += self.nu * (self.dt / self.dx**2) * (
                self.u[2:] - 2*self.u[1:-1] + self.u[:-2]
            )

        # Apply boundary conditions
        self.u = self.apply_boundary_conditions(u_new)

        # Update time
        self.t += self.dt
        self.n_steps += 1

    def solve(self, n_snapshots: int = 10) -> Tuple[np.ndarray, list, list]:
        """
        Solve the Burgers equation until t_final.

        Args:
            n_snapshots: Number of solution snapshots to save

        Returns:
            Tuple of (final solution, list of snapshots, list of snapshot times)
        """
        # Determine snapshot interval
        snapshot_interval = self.t_final / n_snapshots
        next_snapshot_time = snapshot_interval

        print(f"Starting sequential Rusanov solver...")
        print(f"Grid points: {self.nx}")
        print(f"Domain: [{self.x_min}, {self.x_max}]")
        print(f"Final time: {self.t_final}")
        print(f"CFL number: {self.cfl}")
        print(f"Viscosity: {self.nu}")
        print()

        start_time = time.time()

        # Time integration loop
        while self.t < self.t_final:
            self.step()

            # Save snapshots
            if self.t >= next_snapshot_time or abs(self.t - self.t_final) < 1e-10:
                self.snapshots.append(self.u.copy())
                self.snapshot_times.append(self.t)
                next_snapshot_time += snapshot_interval
                print(f"Step {self.n_steps}: t = {self.t:.6f}, dt = {self.dt:.6e}, "
                      f"max(|u|) = {np.max(np.abs(self.u)):.6f}")

        elapsed_time = time.time() - start_time

        print()
        print(f"Simulation complete!")
        print(f"Total time steps: {self.n_steps}")
        print(f"Elapsed time: {elapsed_time:.6f} seconds")
        print(f"Average time per step: {elapsed_time/self.n_steps:.6e} seconds")

        return self.u, self.snapshots, self.snapshot_times


def main():
    """Main function to run sequential Rusanov solver."""
    parser = argparse.ArgumentParser(description='Sequential Rusanov solver for Burgers equation')
    parser.add_argument('--nx', type=int, default=200, help='Number of grid points')
    parser.add_argument('--domain', type=float, nargs=2, default=[0.0, 1.0],
                       help='Spatial domain [x_min, x_max]')
    parser.add_argument('--t-final', type=float, default=0.5,
                       help='Final simulation time')
    parser.add_argument('--cfl', type=float, default=0.5, help='CFL number')
    parser.add_argument('--nu', type=float, default=0.01, help='Viscosity coefficient')
    parser.add_argument('--ic', type=str, default='sine',
                       choices=['sine', 'step', 'rarefaction'],
                       help='Initial condition type')
    parser.add_argument('--snapshots', type=int, default=10,
                       help='Number of solution snapshots')
    parser.add_argument('--save', type=str, default='results_sequential.npz',
                       help='Output file for results')

    args = parser.parse_args()

    # Create solver
    solver = BurgersRusanovSolver(
        nx=args.nx,
        domain=tuple(args.domain),
        t_final=args.t_final,
        cfl=args.cfl,
        nu=args.nu
    )

    # Set initial condition
    solver.set_initial_condition(args.ic)

    # Solve
    u_final, snapshots, snapshot_times = solver.solve(n_snapshots=args.snapshots)

    # Save results
    np.savez(args.save,
             x=solver.x,
             u_final=u_final,
             snapshots=np.array(snapshots),
             times=np.array(snapshot_times),
             nx=args.nx,
             nu=args.nu,
             t_final=args.t_final)

    print(f"\nResults saved to {args.save}")

    # Check for shock formation (large gradients)
    gradients = np.abs(np.gradient(u_final, solver.x))
    max_gradient = np.max(gradients)
    print(f"\nMaximum gradient in solution: {max_gradient:.6f}")
    if max_gradient > 10.0:
        print("Strong discontinuities (shock waves) detected!")
    elif max_gradient > 2.0:
        print("Moderate discontinuities detected")
    else:
        print("Solution appears smooth")


if __name__ == "__main__":
    main()
