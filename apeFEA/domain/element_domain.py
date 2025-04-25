# apeFEA/domain/element_domain.py

from ..core.config import FEMConfig
from ..core.line import Line 
from ..elements.element import Element
from ..materials.material import Material
from ..section.section import Section


class ElementDomain:
    def __init__(
        self,
    999999999999999999999999999    line: Line,
        node_list: list,
        element_type: Element,
        material: Material,
        section: Section,
    ):
        """
        This is the domain element object.
        """
        self.line = line
        self.node_list=node_list
        self.material=material
        self.section=section
        
        # Get the nodes domains objects from the geometric nodes
        self.geometric_nodes = [self.line.node_start, self.line.node_end]
        

        # Instantiate the actual element and assign geometry + physics
        self.element = element_type(
            id=self.line.id,
            nodes=node_list,
            material=material,
            section=section
        )


    def get_global_stiffness_matrix(self):
        """Return global stiffness matrix of the embedded element."""
        return self.element.get_global_stiffness_matrix()

    def __repr__(self):
        return f"ElementDomain(id={self.id_domain}, type={self.element.__class__.__name__})"
