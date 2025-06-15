import numpy as np
from numpy import ndarray
from typing import TYPE_CHECKING

from .transformation import Transformation
from apeFEA.core.node import Node

if TYPE_CHECKING:
    from apeFEA.elements.one_dimension.frame_element import FrameElement


class PDeltaTransformation2D_OP(Transformation):
    """
    OpenSees-style P–Δ transformation (linear kinematics + geometric stiffness).
    """
    def __init__(self, element: "FrameElement"):
        self.element = element
        self.node_i = element.node_i
        self.node_j = element.node_j

        # Precompute fixed geometry
        delta = self.node_j.coords - self.node_i.coords
        self.L0 = np.linalg.norm(delta)
        self.cos_theta = delta[0] / self.L0
        self.sin_theta = delta[1] / self.L0

        self.ub_trial = np.zeros((3, 1))
        self.ub_commit = np.zeros((3, 1))
        self.ub_previous = np.zeros((3, 1))

    def get_L0(self) -> float:
        return self.L0

    def get_length(self) -> float:
        return self.L0  # No large deformation update

    def get_Tlg(self) -> ndarray:
        c = self.cos_theta
        s = self.sin_theta
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
        """Update ul14 and ub_trial."""
        ui = self.node_i.u_trial
        uj = self.node_j.u_trial

        # Local vertical displacements
        ul1 = -self.sin_theta * ui[0, 0] + self.cos_theta * ui[1, 0]
        ul4 = -self.sin_theta * uj[0, 0] + self.cos_theta * uj[1, 0]
        self.ul14 = ul1 - ul4  # for P–Δ effect

        # Basic trial deformation (OpenSees-style)
        L = self.L0
        delta_ux = self.cos_theta * (uj[0, 0] - ui[0, 0]) + self.sin_theta * (uj[1, 0] - ui[1, 0])
        self.ub_previous[:] = self.ub_trial
        self.ub_trial[0, 0] = delta_ux
        self.ub_trial[1, 0] = ui[2, 0]
        self.ub_trial[2, 0] = uj[2, 0]

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

    def geometric_transformation_matrix(self) -> tuple[ndarray, ndarray]:
        """Returns geometric transformation matrices T_geo_Fb1, T_geo_Fb2."""
        L = self.L0
        T_geo_Fb1 = (1 / L) * np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, -1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        T_geo_Fb2 = np.zeros((6, 6))  # Not used in OpenSees
        return T_geo_Fb1, T_geo_Fb2

    def get_ul14(self) -> float:
        """Return Δy = ul1 - ul4 used in leaning column effect."""
        return self.ul14
