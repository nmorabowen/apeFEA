import numpy as np
from numpy import ndarray
from typing import TYPE_CHECKING

from .transformation import Transformation
from apeFEA.core.node import Node

if TYPE_CHECKING:
    from apeFEA.elements.one_dimension.frame_element import FrameElement

class PDeltaTransformation2D(Transformation):
    """
    P–Δ geometric transformation for 2D frame elements (consistent formulation).

    This transformation accounts for second-order effects using a consistent
    geometric formulation derived from the corotational kinematics. It captures
    large displacements and rotations in the element, but assumes small strains.

    The transformation maps between global, local, and basic coordinate systems,
    and updates internal deformation measures accordingly.

    Attributes
    ----------
    element : FrameElement
        The parent element for which this transformation applies.
    node_i, node_j : Node
        The two nodes defining the frame element geometry.
    ub_trial : ndarray
        Basic trial deformation vector (ΔL, θ_i, θ_j)ᵀ.
    ub_commit : ndarray
        Committed (converged) basic deformation vector.
    ub_previous : ndarray
        Basic deformation vector from previous iteration (ΔΔu computation).
    """
    def __init__(self, element: "FrameElement"):
        self.element = element
        self.node_i = element.node_i
        self.node_j = element.node_j
        
        # State vectors for basic system
        self.ub_trial = np.zeros((3,1))
        self.ub_commit = np.zeros((3,1))
        self.ub_previous = np.zeros((3,1))

    def _trans(self, u_trial: np.ndarray) -> np.ndarray:
        return u_trial[:2, 0]

    def _pos(self, node) -> np.ndarray:
        return node.coords + self._trans(node.u_trial)

    def get_L0(self) -> float:
        return np.linalg.norm(self.node_j.coords - self.node_i.coords)

    def get_length(self) -> float:
        self.get_L0

    def _get_corrotational_parameters(self):
        L0 = self.get_L0()
        u_trial_global = np.vstack((self.node_i.u_trial, self.node_j.u_trial))
        u_trial_local = self.get_Tlg() @ u_trial_global
        Delta_ul_x = u_trial_local[3, 0] - u_trial_local[0, 0]
        Delta_ul_y = u_trial_local[4, 0] - u_trial_local[1, 0]
        beta = np.arctan2(Delta_ul_y, (L0 + Delta_ul_x))
        return beta, Delta_ul_x, Delta_ul_y

    def get_cosine_director(self) -> tuple[float, float]:
        delta = self.node_j.coords - self.node_i.coords
        L = self.get_L0()
        alpha = np.arctan2(delta[1], delta[0])
        c = np.cos(alpha)
        s = np.sin(alpha)
        return c, s, L

    def get_Tbl(self) -> ndarray:
        L0 = self.get_L0()
        _, _, Delta_ul_y = self._get_corrotational_parameters()
        
        Tbl= np.array(
            [
                [-1, -Delta_ul_y/L0, 0, 1, Delta_ul_y/L0, 0],
                [0, 1/L0, 1, 0, -1/L0, 0],
                [0, 1/L0, 0, 0, -1/L0, 1]
            ]
        )
        
        # Tbl = np.array(
        #     [
        #         [-1, -Delta_ul_y/L0, 0, 1, Delta_ul_y/L0, 0],
        #         [-Delta_ul_y/L0**2, 1/L0, 1, Delta_ul_y/L0**2, -1/L0, 0],
        #         [-Delta_ul_y/L0**2, 1/L0, 0, Delta_ul_y/L0**2, -1/L0, 1]
        #     ]
        # )
        
        
        return Tbl

    def get_Tlg(self) -> ndarray:        
        c, s, L = self.get_cosine_director()
        
        return np.array([
            [ c,  s, 0, 0, 0, 0],
            [-s,  c, 0, 0, 0, 0],
            [ 0,  0, 1, 0, 0, 0],
            [ 0,  0, 0,  c,  s, 0],
            [ 0,  0, 0, -s,  c, 0],
            [ 0,  0, 0,  0,  0, 1]
        ])

    def update_trial(self):
        
        # self.ub_previous[:] = self.ub_trial
        # Tbl = self.get_Tbl()
        # u_trial_global = np.vstack((self.node_i.u_trial, self.node_j.u_trial))
        # u_trial_local = self.get_Tlg() @ u_trial_global
        # self.ub_trial[:] = Tbl @ u_trial_local  # basic system update
        
        """Update the basic deformation ub_trial."""
        L0 = self.get_L0()
        L = self.get_length()
        _, Delta_ul_x, Delta_ul_y = self._get_corrotational_parameters()

        self.ub_previous[:] = self.ub_trial
        self.ub_trial[0, 0] = Delta_ul_x + Delta_ul_y**2/(2*L0)
        self.ub_trial[1, 0] = self.node_i.u_trial[2, 0] - Delta_ul_y/L0
        self.ub_trial[2, 0] = self.node_j.u_trial[2, 0] - Delta_ul_y/L0


    def commit_state(self):
        self.ub_commit[:] = self.ub_trial

    def reset_trial(self):
        self.ub_trial[:] = self.ub_commit
        self.ub_previous[:] = self.ub_commit

    def revert_to_start(self):
        self.ub_trial[:] = 0.0
        self.ub_commit[:] = 0.0
        self.ub_previous[:] = 0.0

    def get_basic_trial_disp(self) -> ndarray:
        return self.ub_trial

    def get_basic_incr_disp(self) -> ndarray:
        return self.ub_trial - self.ub_commit

    def get_basic_incr_delta_disp(self) -> ndarray:
        return self.ub_trial - self.ub_previous

    def geometric_transformation_matrix(self) -> tuple[ndarray, ndarray]:
        L = self.get_length()
        L0 = self.get_L0()

        T_geo_Fb1 = (1 / L0) * np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, -1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        
        T_geo_Fb2 = (1 / L0**2) * np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 2, 1, 0, -2, -1],
            [0, 1, 2, 0, -1, -2],
            [0, 0, 0, 0, 0, 0],
            [0, -2, -1, 0, 2, 1],
            [0, -1, -2, 0, 1, 2],
        ])
        
        return T_geo_Fb1, T_geo_Fb2
