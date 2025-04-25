# apeFEA/elements/elastic2D_beam.py
import numpy as np
from .element import Element

class Beam2D(Element):
    """
    2D Bernoulli-Euler beam element (2 nodes, 3 DOFs per node: UX, UY, RZ).
    """

    def get_length(self):
        x1, y1 = self.nodes[0].coords
        x2, y2 = self.nodes[1].coords
        return np.hypot(x2 - x1, y2 - y1)

    def get_direction_cosines(self):
        x1, y1 = self.nodes[0].coords
        x2, y2 = self.nodes[1].coords
        L = self.get_length()
        c = (x2 - x1) / L
        s = (y2 - y1) / L
        return c, s

    def get_local_stiffness_matrix(self):
        E = self.material.E
        I = self.section.get_inertia()
        A = self.section.get_area()
        L = self.get_length()

        # Element stiffness matrix in local coordinates
        k = np.array([
            [ A*E/L,       0,          0,     -A*E/L,       0,          0],
            [ 0,     12*E*I/L**3,  6*E*I/L**2,  0, -12*E*I/L**3,  6*E*I/L**2],
            [ 0,     6*E*I/L**2,   4*E*I/L,     0, -6*E*I/L**2,   2*E*I/L],
            [-A*E/L,      0,          0,      A*E/L,       0,          0],
            [ 0, -12*E*I/L**3, -6*E*I/L**2,  0,  12*E*I/L**3, -6*E*I/L**2],
            [ 0,     6*E*I/L**2,   2*E*I/L,     0, -6*E*I/L**2,   4*E*I/L]
        ])
        return k

    def get_transformation_matrix(self):
        c, s = self.get_direction_cosines()
        T = np.array([
            [ c,  s, 0,  0,  0, 0],
            [-s,  c, 0,  0,  0, 0],
            [ 0,  0, 1,  0,  0, 0],
            [ 0,  0, 0,  c,  s, 0],
            [ 0,  0, 0, -s,  c, 0],
            [ 0,  0, 0,  0,  0, 1],
        ])
        return T
