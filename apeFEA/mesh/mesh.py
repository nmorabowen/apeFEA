"""
MeshBuilder Module
------------------
This module defines the MeshBuilder class, which generates a 1D mesh of FrameElements between two given nodes. 
It preserves original nodes and their properties (e.g., boundary conditions or loads) and automatically adds 
intermediate nodes based on a specified mesh size.

Author: Patricio Palacios B., M.Sc y Caiza.
Version: 1.0
"""

from math import sqrt, ceil
import numpy as np
from typing import Optional

from apeFEA.core.node import Node
from apeFEA.elements.one_dimension.frame_element import FrameElement
from apeFEA.sections.section import Section
from apeFEA.elements.one_dimension.transformations.transformation import Transformation


class MeshBuilder:
    def __init__(self):
        self.nodes: list[Node] = []
        self.elements: list[FrameElement] = []
        self._node_id_counter: int = 1
        self._element_id_counter: int = 1

    def mesh_line(self, start: Node, end: Node, mesh_size: float, section: Section, transformation) -> None:
        """Create a mesh of elements between two coordinates with a given section and transformation."""
        x1, y1 = start.coords
        x2, y2 = end.coords
        total_length = sqrt((x2 - x1)**2 + (y2 - y1)**2)
        n_div = max(1, ceil(total_length / mesh_size))
        dx = (x2 - x1) / n_div
        dy = (y2 - y1) / n_div

        # Reuse or add the first node
        prev = self._get_or_add_node(start.coords)

        for i in range(1, n_div):
            x = x1 + i * dx
            y = y1 + i * dy
            next_node = self._get_or_add_node([x, y])
            self._add_element(prev, next_node, section, transformation)
            prev = next_node

        # Reuse or add the last node
        final = self._get_or_add_node(end.coords)
        self._add_element(prev, final, section, transformation)

    def _get_or_add_node(self, coords: list[float], tol: float = 1e-6) -> Node:
        for node in self.nodes:
            if np.linalg.norm(np.array(node.coords) - np.array(coords)) < tol:
                return node
        new_node = Node(id=self._node_id_counter, coords=coords)
        new_node.set_node_id(self._node_id_counter)
        self.nodes.append(new_node)
        self._node_id_counter += 1
        return new_node

    def _add_element(self, ni: Node, nj: Node, section: Section, transformation):
        if ni is nj:
            print(f"[⚠️] Skipping zero-length element between Node {ni.id} and itself.")
            return
        element = FrameElement(
            id=self._element_id_counter,
            nodes=[ni, nj],
            section=section,
            transformation=transformation,
        )
        self.elements.append(element)
        self._element_id_counter += 1

