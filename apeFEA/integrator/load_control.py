import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from apeFEA.core.model import Model
from apeFEA.solver.newton_raphson import NewtonRaphsonSolver

class LoadControl:
    """
    Perform static nonlinear analysis using a load-controlled Newton–Raphson scheme.
    Tracks displacement, residuals, and iteration history.
    """

    def __init__(self, model: Model, solver: NewtonRaphsonSolver, t_end: float, steps: int):
        self.model = model
        self.solver = solver
        self.t_end = t_end
        self.steps = steps

        self.time_values = np.linspace(0, t_end, steps + 1)

        self.u_history: List[np.ndarray] = []
        self.residual_history_per_step: List[List[float]] = []
        self.iteration_counts: List[int] = []

    def run(self) -> None:
        """Execute the static analysis across load steps."""
        for i, t in enumerate(self.time_values):
            print(f"\n=== Load Step {i}/{self.steps} – Pseudo-time: {t:.3f} ===")
            try:
                u, residuals, n_iter = self.solver.solve(t)
                self.u_history.append(u.copy())
                self.residual_history_per_step.append(residuals)
                self.iteration_counts.append(n_iter)

                final_residual = residuals[-1] if residuals else float('nan')
                print(f" → Iterations: {n_iter:2d} | Final Residual Norm: {final_residual:.3e}")

            except RuntimeError as e:
                print(f"Step {i} failed: {e}")
                break

    def plot_convergence(self) -> None:
        """Plot convergence history and iteration counts."""
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        for i, residuals in enumerate(self.residual_history_per_step):
            axs[0].semilogy(residuals, label=f"Step {i}")
        axs[0].set_ylabel("Residual Norm")
        axs[0].set_title("Residual Convergence per Step")
        axs[0].grid(True)

        axs[1].bar(range(len(self.iteration_counts)), self.iteration_counts)
        axs[1].set_xlabel("Load Step")
        axs[1].set_ylabel("# Iterations")
        axs[1].set_title("Iterations per Load Step")
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()
        
        return fig, ax
