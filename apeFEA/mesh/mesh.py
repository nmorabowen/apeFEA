"""
MeshBuilder class for generating 1D meshes between two nodes using maximum mesh size.
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

    def mesh_line(self, ni: Node, nj: Node, mesh_size: float, section: Section, transformation: Optional[Transformation] = None):
        xi, yi = ni.coords
        xj, yj = nj.coords

        length = sqrt((xj - xi)**2 + (yj - yi)**2)
        n_div = ceil(length / mesh_size)

        dx = (xj - xi) / n_div
        dy = (yj - yi) / n_div

        if not any(n.id == ni.id for n in self.nodes):
            self.nodes.append(ni)
        if not any(n.id == nj.id for n in self.nodes):
            self.nodes.append(nj)

        used_ids = {n.id for n in self.nodes}
        current_id = max(used_ids) + 1

        node_list = [ni]

        for i in range(1, n_div):
            x = xi + i * dx
            y = yi + i * dy

            if any(np.allclose(n.coords, [x, y]) for n in self.nodes):
                continue

            while current_id in used_ids:
                current_id += 1

            new_node = Node(current_id, [x, y])
            self.nodes.append(new_node)
            node_list.append(new_node)
            used_ids.add(current_id)
            current_id += 1

        node_list.append(nj)

        # Generate FrameElements between consecutive nodes
        next_element_id = max((e.id for e in self.elements), default=0) + 1

        for i in range(len(node_list) - 1):
            n_start = node_list[i]
            n_end = node_list[i + 1]
            element = FrameElement(
                id=next_element_id,
                nodes=[n_start, n_end],
                section=section,
                transformation=transformation
            )
            self.elements.append(element)
            next_element_id += 1
