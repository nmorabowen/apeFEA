import numpy as np
from numpy import ndarray
from typing import TYPE_CHECKING

from .transformation import Transformation
from apeFEA.core.node import Node

if TYPE_CHECKING:
    from apeFEA.elements.one_dimension.frame_element import FrameElement


class LinearTransformation(Transformation):
    """
    Linear geometric transformation for 2D frame elements.

    This class implements a transformation scheme assuming small
    displacements and rotations. It provides the mappings between:
    
    - global system → local system (Tlg)
    - local system → basic system (Tbl)
    - basic deformations (axial, end rotations)

    It assumes the geometry remains fixed (undeformed configuration)
    and is therefore suitable for linear analysis or as the geometric
    transformation in a total Lagrangian formulation.

    Attributes:
    -----------
    element : FrameElement
        The parent element for which this transformation applies.
    node_i, node_j : Node
        The two nodes that define the element.
    ub_trial : ndarray
        Trial basic deformation vector [ΔL, θ_i, θ_j]ᵀ.
    ub_commit : ndarray
        Committed (last converged) basic deformation.
    ub_previous : ndarray
        Basic deformation from previous iteration (for ΔΔu).
    """
    def __init__(self, element: "FrameElement"):
        self.element = element
        self.node_i = element.node_i
        self.node_j = element.node_j
        
        # State vectors for basic system
        self.ub_trial = np.zeros((3,1))
        self.ub_commit = np.zeros((3,1))
        self.ub_previous = np.zeros((3,1))

    def get_length(self) -> float:
        return np.linalg.norm(self.node_j.coords - self.node_i.coords)
    
    def get_L0(self) -> float:
        return np.linalg.norm(self.node_j.coords - self.node_i.coords)

    def get_cosine_director(self) -> tuple[float, float, float]:
        delta_vector = self.node_j.coords - self.node_i.coords
        length = np.linalg.norm(delta_vector)
        unit_vector = delta_vector / length if length > 0 else np.zeros_like(delta_vector)
        c = unit_vector[0]
        s = unit_vector[1]
        return c, s, length

    def get_Tbl(self) -> ndarray:
        L = self.get_length()
        Tbl = np.array([
            [-1,  0, 0,  1,  0, 0],
            [ 0, 1/L, 1,  0, -1/L, 0],
            [ 0, 1/L, 0,  0, -1/L, 1],
        ])
        return Tbl

    def get_Tlg(self) -> ndarray:
        c, s, _ = self.get_cosine_director()
        Tlg = np.array([
            [ c,  s, 0,  0, 0, 0],
            [-s,  c, 0,  0, 0, 0],
            [ 0,  0, 1,  0, 0, 0],
            [ 0,  0, 0,  c, s, 0],
            [ 0,  0, 0, -s, c, 0],
            [ 0,  0, 0,  0, 0, 1],
        ])
        return Tlg

    def geometric_transformation_matrix(self) -> ndarray:
        """Return geometric stiffness for linear transformation (classical axial P–Δ only)."""
        L = self.get_length()
        c, s, _ = self.get_cosine_director()

        T1=np.zeros((6, 6))  # Initialize T1 as a zero matrix

        T2 = np.zeros((6, 6))  # Initialize T2 as a zero matrix

        return T1, T2
    
    def reset_trial(self):
        self.ub_trial[:] = self.ub_commit
        self.ub_previous[:] = self.ub_commit
        
    def update_trial(self):
        """
        Updates the basic trial displacement `ub_trial` assuming small displacements.
        """
        self.ub_previous[:] = self.ub_trial

        u_global = np.vstack([self.node_i.u_trial, self.node_j.u_trial])  # 6x1
        Tlg = self.get_Tlg()  # 6x6
        u_local = Tlg @ u_global  # transform to local coords

        Tbl = self.get_Tbl()  # 3x6
        self.ub_trial[:] = Tbl @ u_local  # basic system update
        
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
    
    def commit_state(self):
        self.ub_commit[:] = self.ub_trial
