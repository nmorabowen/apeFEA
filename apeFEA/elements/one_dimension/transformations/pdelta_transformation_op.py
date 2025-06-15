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

    def get_Tbl(self) -> np.ndarray:
        """
        Transformation matrix from local DOFs to basic DOFs
        for OpenSees-style linear PDelta transformation (constant matrix).
        """
        L = self.get_length()  # or self.L0 for fixed geometry
        delta_ul_y= self.ul14  # ul1 - ul4 for leaning-column effect
        Tbl = np.array([
            [-1,  -delta_ul_y/L,  0,   1,   delta_ul_y/L,  0],     # axial deformation (u_jx - u_ix)
            [ 0,  1/L,  1,   0,  -1/L,  0],     # curvature at node i
            [ 0,  1/L,  0,   0,  -1/L,  1],     # curvature at node j
        ])
        
        return Tbl

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

    def geometric_transformation_matrix(self) -> tuple[ndarray, ndarray]:
        """
        Return geometric stiffness transformation pattern matrices (6×6 each).
        These are used in the element as:
            K_geo = Fb[0] * T_geo_Fb1 + (Fb[1] + Fb[2]) * T_geo_Fb2
        """
        L = self.get_L0()

        T_geo_Fb1 = (1 / L) * np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, -1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0]
        ])

        T_geo_Fb2 = np.zeros((6, 6))  # not used in OpenSees PDelta

        return T_geo_Fb1, T_geo_Fb2
