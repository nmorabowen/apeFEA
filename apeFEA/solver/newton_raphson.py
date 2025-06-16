import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    from apeFEA.core.model import Model


class NewtonRaphsonSolver:
    def __init__(
        self,
        model: "Model",
        tolerance: float = 1e-6,
        max_iterations: int = 20,
        verbose: bool = False,
    ):
        self.model = model
        self.tol = tolerance
        self.max_iter = max_iterations
        self.verbose = verbose
        self.residual_history = []

    def solve(self, t: float) -> tuple[np.ndarray, list[float], int]:
        u = self.model._assemble_displacement_vector_committed()
        self.model.reset_trial()
        residual = []

        if self.verbose:
            print(f"Initial committed displacement u_committed:\n{u.T}")

        for i in range(self.max_iter):
            if self.verbose:
                print("="*60)
                print(f"Iteration {i}")

            R = self.model.calculate_residual(t)
            K = self.model.get_stiffness_matrix()
            free = self.model.free_indices

            norm_R = self.model.residual_norm(t, norm_type='L2')
            residual.append(norm_R)

            if self.verbose:
                print(f"Residual vector R.T:\n{R.T}")
                print(f"Residual norm = {norm_R:.3e}")
                print(f"Stiffness submatrix (free DOFs):\n{K[np.ix_(free, free)]}")

            if np.isnan(norm_R) or np.isinf(norm_R):
                self.model.revert_to_start()
                raise RuntimeError("Residual norm is NaN or Inf – possible numerical instability.")

            if norm_R < self.tol:
                if self.verbose:
                    print("Converged. Committing state.")
                self.model.commit_state()
                self.residual_history = residual
                return u, residual, i + 1  # <-- return count and residuals

            du = np.zeros_like(u)

            try:
                cond_K = np.linalg.cond(K[np.ix_(free, free)])
                if self.verbose and cond_K > 1e12:
                    print(f"Warning: Ill-conditioned stiffness matrix (cond={cond_K:.3e})")
                du[free] = np.linalg.solve(K[np.ix_(free, free)], R[free])
            except np.linalg.LinAlgError as e:
                self.model.revert_to_start()
                raise RuntimeError(f"Linear solve failed: {e}")

            u += du
            self.model.update_trial_state(u, verbose=self.verbose, print_elements=self.verbose)

            if self.verbose:
                print(f"Δu.T =\n{du.T}")
                print(f"Δu L2 norm = {np.linalg.norm(du):.3e}")
                print(f"Updated u.T =\n{u.T}")
                for node in self.model.nodes:
                    print(f"Node {node.id} Trial Displacement: {node.u_trial.flatten()}")

        self.model.revert_to_start()
        raise RuntimeError("Newton–Raphson did not converge.")

    def plot_residual_convergence(self):
        import matplotlib.pyplot as plt
        plt.semilogy(self.residual_history, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Residual Norm')
        plt.title('Newton–Raphson Convergence')
        plt.grid(True)
        plt.show()
