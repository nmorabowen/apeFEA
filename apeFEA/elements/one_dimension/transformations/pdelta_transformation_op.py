import numpy as np
from numpy import ndarray
from typing import TYPE_CHECKING

from .transformation import Transformation           # abstract base
from apeFEA.core.node import Node

if TYPE_CHECKING:                                    # avoid circular imports
    from apeFEA.elements.one_dimension.frame_element import FrameElement


class PDeltaTransformation2D_OP(Transformation):
    """
    Small-rotation *consistent* P–Δ transformation (linear-geometry).

    Public API •identical• to your abstract `Transformation` base and to the
    previous `CorotationalTransformation2D`, so the element code can switch
    to it without further edits.
    """

    # ------------------------------------------------------------------ #
    # construction / cached state                                        #
    # ------------------------------------------------------------------ #
    def __init__(self, element: "FrameElement"):
        self.element: "FrameElement" = element
        self.node_i: Node = element.node_i
        self.node_j: Node = element.node_j

        self.ub_trial    = np.zeros((3, 1))
        self.ub_commit   = np.zeros((3, 1))
        self.ub_previous = np.zeros((3, 1))

        # store undeformed geometry once
        delta0           = self.node_j.coords - self.node_i.coords
        self.L0          = float(np.linalg.norm(delta0))
        self.c0          = delta0[0] / self.L0
        self.s0          = delta0[1] / self.L0

    # ------------------------------------------------------------------ #
    # interface required by `Transformation`                             #
    # ------------------------------------------------------------------ #
    def get_L0(self) -> float:
        return self.L0                      # constant

    def get_length(self) -> float:
        """Current deformed length (needed by some material models)."""
        xi = self.node_i.coords + self.node_i.u_trial[:2, 0]
        xj = self.node_j.coords + self.node_j.u_trial[:2, 0]
        return float(np.linalg.norm(xj - xi))

    def get_cosine_director(self) -> Tuple[float, float, float]:
        """(c, s, L) from the *initial* configuration (consistent P-Δ)."""
        return self.c0, self.s0, self.L0

    def get_Tlg(self) -> ndarray:
        """6×6 global→local rotation matrix based on initial axis."""
        c, s = self.c0, self.s0
        return np.array(
            [[ c,  s, 0, 0, 0, 0],
             [-s,  c, 0, 0, 0, 0],
             [ 0,  0, 1, 0, 0, 0],
             [ 0,  0, 0,  c,  s, 0],
             [ 0,  0, 0, -s,  c, 0],
             [ 0,  0, 0,  0,  0, 1]],
            dtype=float,
        )

    def get_Tbl(self) -> ndarray:
        """3×6 P-matrix Γᵀ_RBM,PΔ(Uᵉ) (yellow slide #1)."""
        # local trial displacements
        u_glob = np.vstack((self.node_i.u_trial, self.node_j.u_trial))  # 6×1
        u_loc  = self.get_Tlg() @ u_glob
        Δu_x2  = u_loc[4, 0] - u_loc[1, 0]

        L0 = self.L0
        return np.array(
            [
                [-1,           0,   0, +1,           0,   0],
                [ Δu_x2/L0,  1/L0, 1/L0, Δu_x2/L0, -1/L0, -1/L0],
                [ 0,           1,   0,  0,           0,  +1],
            ],
            dtype=float,
        )

    # ------------------------------------------------------------------ #
    # geometric stiffness K_T,g,PΔ (yellow slide #2)                      #
    # ------------------------------------------------------------------ #
    def geometric_transformation_matrix(self, P: float) -> ndarray:
        """
        Returns the 6×6 **global** geometric stiffness:

            K_geo =  (P / L0) · Tlgᵀ · K̂ · Tlg
        with   K̂  = [[0 0 0 0 0 0],
                      [0 1 0 0 -1 0],
                      [0 0 0 0  0 0],
                      [0 0 0 0  0 0],
                      [0 -1 0 0 1 0],
                      [0 0 0 0 0 0]]
        """
        if abs(P) < 1e-30:                     # early exit — saves a multiply
            return np.zeros((6, 6))

        Kg_loc = np.zeros((6, 6))
        Kg_loc[1, 1] = Kg_loc[4, 4] =  1
        Kg_loc[1, 4] = Kg_loc[4, 1] = -1

        Kg_loc *= (P / self.L0)
        Tlg = self.get_Tlg()
        return Tlg.T @ Kg_loc @ Tlg           # global 6×6 matrix

    # ------------------------------------------------------------------ #
    # state-vector bookkeeping (unchanged signatures)                     #
    # ------------------------------------------------------------------ #
    def update_trial(self) -> None:
        u_glob = np.vstack((self.node_i.u_trial, self.node_j.u_trial))
        u_loc  = self.get_Tlg() @ u_glob

        ΔL = u_loc[3, 0] - u_loc[0, 0]
        θi = self.node_i.u_trial[2, 0]
        θj = self.node_j.u_trial[2, 0]

        self.ub_previous[:] = self.ub_trial
        self.ub_trial[:, 0] = [ΔL, θi, θj]

    def commit_state(self)        -> None: self.ub_commit[:]   = self.ub_trial
    def reset_trial(self)         -> None: self.ub_trial[:]    = self.ub_commit
    def revert_to_start(self)     -> None:
        self.ub_trial[:] = self.ub_commit[:] = self.ub_previous[:] = 0.0

    # convenience getters (same as before)
    def get_basic_trial_disp(self)       -> ndarray: return self.ub_trial
    def get_basic_incr_disp(self)        -> ndarray: return self.ub_trial - self.ub_commit
    def get_basic_incr_delta_disp(self)  -> ndarray: return self.ub_trial - self.ub_previous