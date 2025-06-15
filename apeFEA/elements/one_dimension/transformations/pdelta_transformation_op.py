import numpy as np
from numpy import ndarray
from typing import TYPE_CHECKING, Tuple

from .transformation import Transformation

if TYPE_CHECKING:
    from apeFEA.elements.one_dimension.frame_element import FrameElement


class PDeltaTransformation2D_OP(Transformation):
    """
    OpenSees-style P–Δ transformation (linear kinematics + geometric stiffness).

    Assumptions:
    - Geometry is fixed (undeformed)
    - Transformation matrix is linear (based on initial geometry)
    - Geometric nonlinearity is captured through ul14 and P–Δ force/stiffness
    """

    def __init__(self, element: "FrameElement"):
        self.element = element
        self.node_i = element.node_i
        self.node_j = element.node_j

        # Reference geometry
        delta = self.node_j.coords - self.node_i.coords
        self.L0 = np.linalg.norm(delta)
        self.cos_theta = delta[0] / self.L0
        self.sin_theta = delta[1] / self.L0

        self.ub_trial = np.zeros((3, 1))
        self.ub_commit = np.zeros((3, 1))
        self.ub_previous = np.zeros((3, 1))
        self.ul14 = 0.0  # for leaning-column effect

    def get_L0(self) -> float:
        return self.L0

    def get_length(self) -> float:
        return self.L0  # No updated length in OpenSees-style

    def get_cosine_director(self) -> Tuple[float, float, float]:
        return self.cos_theta, self.sin_theta, self.L0

    def get_Tlg(self) -> ndarray:
        c, s = self.cos_theta, self.sin_theta
        return np.array([
            [ c,  s, 0, 0, 0, 0],
            [-s,  c, 0, 0, 0, 0],
            [ 0,  0, 1, 0, 0, 0],
            [ 0,  0, 0,  c,  s, 0],
            [ 0,  0, 0, -s,  c, 0],
            [ 0,  0, 0,  0,  0, 1]
        ])

    def get_Tbl(self) -> ndarray:
        L = self.L0
        return np.array([
            [-1,  0,  0,  1,  0, 0],
            [ 0,  1,  1,  0,  0, 0],
            [ 0,  1,  0,  0,  0, 1]
        ]) / L

    def update_trial(self) -> None:
        """Update the basic deformation ub_trial and ul14 for P–Δ effect."""
        ui = self.node_i.u_trial
        uj = self.node_j.u_trial

        self.ub_previous[:] = self.ub_trial

        # Axial deformation along element axis
        delta_ux = self.cos_theta * (uj[0, 0] - ui[0, 0]) + self.sin_theta * (uj[1, 0] - ui[1, 0])
        self.ub_trial[0, 0] = delta_ux

        # Rotations from nodes
        self.ub_trial[1, 0] = ui[2, 0]
        self.ub_trial[2, 0] = uj[2, 0]

        # Local vertical displacements for ul14
        ul1 = -self.sin_theta * ui[0, 0] + self.cos_theta * ui[1, 0]
        ul4 = -self.sin_theta * uj[0, 0] + self.cos_theta * uj[1, 0]
        self.ul14 = ul1 - ul4

    def commit_state(self) -> None:
        self.ub_commit[:] = self.ub_trial

    def reset_trial(self) -> None:
        self.ub_trial[:] = self.ub_commit
        self.ub_previous[:] = self.ub_commit

    def revert_to_start(self) -> None:
        self.ub_trial[:] = 0.0
        self.ub_commit[:] = 0.0
        self.ub_previous[:] = 0.0

    def get_basic_trial_disp(self) -> ndarray:
        return self.ub_trial

    def get_basic_incr_disp(self) -> ndarray:
        return self.ub_trial - self.ub_commit

    def get_basic_incr_delta_disp(self) -> ndarray:
        return self.ub_trial - self.ub_previous

    def get_ul14(self) -> float:
        """Return Δy = ul1 - ul4 for leaning-column moment effect."""
        return self.ul14

    def geometric_transformation_matrix(self, P: float) -> ndarray:
        """
        Compute the geometric stiffness matrix (global) due to axial force P.
        Matches OpenSees-style PDelta contribution.
        """
        L = self.L0
        c, s = self.cos_theta, self.sin_theta

        # Local geometric stiffness matrix (6×6)
        kG_local = (P / L) * np.array([
            [ 0,  0,  0,  0,  0, 0],
            [ 0,  1,  0,  0, -1, 0],
            [ 0,  0,  0,  0,  0, 0],
            [ 0,  0,  0,  0,  0, 0],
            [ 0, -1,  0,  0,  1, 0],
            [ 0,  0,  0,  0,  0, 0]
        ])

        # Rotate to global system
        T = self.get_Tlg()
        kg = T.T @ kG_local @ T
        return kg
