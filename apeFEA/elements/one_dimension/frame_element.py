from __future__ import annotations  # if using forward type hints (Python <3.10)

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

from apeFEA.core.node import Node
from .one_dim_element import Element
from apeFEA.elements.one_dimension.transformations.transformation import Transformation
from apeFEA.sections.section import Section
from apeFEA.elements.one_dimension.transformations.linear_transformation import LinearTransformation

class FrameElement(Element):
    """
    2D frame element supporting nonlinear material and geometric effects.

    This class represents a 2-node 2D frame element with three degrees of freedom per node (ux, uy, θ).
    It supports corotational or linear geometric transformations and nonlinear section behavior
    through a material-based `Section` object.

    Parameters
    ----------
    id : int
        Unique element identifier.
    nodes : list[Node]
        List of two Node objects defining the element ends.
    section : Section
        Cross-sectional object defining stiffness via material behavior.
    transformation : type[Transformation], optional
        Transformation class (e.g., Linear, Corotational), default is LinearTransformation.

    Attributes
    ----------
    node_i, node_j : Node
        References to the two end nodes.
    section : Section
        Section object (material + A/I).
    idx : ndarray
        Global DOF indices for this element.
    restraints : ndarray
        DOF restraint flags (e.g., 'r' or 'f').
    transformation : Transformation
        Instantiated transformation object for this element.
    """
    
    def __init__(self, 
                 id: int, 
                 nodes: list[Node], 
                 section: Section, 
                 transformation: type[Transformation]=LinearTransformation):
        
        super().__init__(id, nodes)

        self.node_i = nodes[0]
        self.node_j = nodes[1]
        self.section = section
        
        # Get indices and restraints
        self.idx, self.restraints = self._elementIndices()

        self.transformation = transformation(element=self)


    def _elementIndices(self):
        idx=np.concatenate([self.node_i.idx,self.node_j.idx])
        restraints=np.concatenate([self.node_i.restraints.restraints, self.node_j.restraints.restraints])
        return idx, restraints
    
    def get_basic_stiffness_matrix(self) -> ndarray:
        EA, EI = self.section.get_stiffness_matrix()
        L = self.transformation.get_length()

        kb=np.array([
            [EA/L, 0, 0],
            [0, 4*EI/L, 2*EI/L],
            [0, 2*EI/L, 4*EI/L]
        ])

        return kb
    
    def get_local_stiffness_matrix(self) -> ndarray:
        
        # Get the transformation matrices
        Tbl = self.transformation.get_Tbl()
        Tlg = self.transformation.get_Tlg()
        
        # Get the material stiffness matrix
        kb_material = self.get_basic_stiffness_matrix()
        kl_material = Tbl.T @ kb_material @ Tbl
        
        # Get the geometric stiffness matrix        
        _, results = self.force_recovery()
        Fb = results['Fb']
        T_geo_Fb1, T_geo_Fb2 = self.transformation.geometric_transformation_matrix()
        
        kl_geometric = Fb[0] * T_geo_Fb1  + (Fb[1]+Fb[2]) * T_geo_Fb2
        
        # Tangent stiffness matrix
        kl= kl_material + kl_geometric
        
        return kl
    
    def get_global_stiffness_matrix(self) -> ndarray:
        Tlg = self.transformation.get_Tlg()
        kl = self.get_local_stiffness_matrix()
        kg = Tlg.T @ kl @ Tlg
        return kg
    
    def get_assembly_stiffness_matrix(self) -> ndarray:
        """ 
        This method returns the stiffness matrix needed for assembly into the global system.
        """
        return self.get_global_stiffness_matrix()
    
    def force_recovery(self) -> tuple[ndarray, dict]:
        """
        Returns:
            F_assembly: Assembled force vector (global)
            results: Dictionary with intermediate vectors
        """
        self.transformation.update_trial()
        
        u_global = np.vstack([self.node_i.u_trial, self.node_j.u_trial])
        Tlg = self.transformation.get_Tlg()
        u_local = Tlg @ u_global
        Tbl = self.transformation.get_Tbl()
        u_basic = self.transformation.get_basic_trial_disp()

        kb = self.get_basic_stiffness_matrix()
        Fb = kb @ u_basic
        Fl = Tbl.T @ Fb
        Fg = Tlg.T @ Fl

        F_assembly = Fg

        results = {
            'Fb': Fb,
            'Fl': Fl,
            'Fg': Fg,
            'u_basic': u_basic,
            'u_local': u_local,
            'u_global': u_global
        }

        return F_assembly, results
    
    def commit_state(self) -> None:
        self.transformation.commit_state()

    def reset_trial(self) -> None:
        self.transformation.reset_trial()

    def revert_to_start(self) -> None:
        self.transformation.revert_to_start()
    
    def plot(self, ax: plt.Axes, color: str = "black", linewidth: float = 2.0, show_id: bool = True, **kwargs) -> None:
        xi = self.node_i.coords
        xj = self.node_j.coords

        ax.plot(
            [xi[0], xj[0]],
            [xi[1], xj[1]],
            color=color,
            linewidth=linewidth,
            **kwargs
        )

        if show_id:
            center = 0.5 * (xi + xj)
            ax.text(center[0], center[1], f"E{self.id}", fontsize=10, ha="center", va="center", color=color)

    def __str__(self):
        return f"FrameElement {self.id}: Node {self.node_i.id} → Node {self.node_j.id}"

    def __repr__(self):
        return self.__str__()

