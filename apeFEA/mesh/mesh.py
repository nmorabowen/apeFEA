import numpy as np
from math import sqrt, ceil

from apeFEA.core.node import Node
from apeFEA.elements.one_dimension.frame_element import FrameElement
from apeFEA.sections.section import Section  # if needed for type hinting
from apeFEA.elements.one_dimension.transformations.transformation import Transformation  # for type hinting


class MeshBuilder:
    def __init__(self):
        self.nodes: list[Node] = []
        self.elements: list[FrameElement] = []

    def mesh_line(self, ni: Node, nj: Node, mesh_size: float, section: Section, transformation: Transformation = None):
        """
        Generate mesh of FrameElements between two nodes with given mesh size.
        Updates internal node and element lists.
        """
        from math import sqrt, ceil

        xi, yi = ni.coords
        xj, yj = nj.coords

        # Length and division
        total_length = sqrt((xj - xi) ** 2 + (yj - yi) ** 2)
        n_div = ceil(total_length / mesh_size)

        dx = (xj - xi) / n_div
        dy = (yj - yi) / n_div

        node_list = [ni]
        next_node_id = max((n.id for n in self.nodes), default=0) + 1

        for i in range(1, n_div):
            x = xi + i * dx
            y = yi + i * dy
            node = Node(next_node_id, [x, y])
            self.nodes.append(node)
            node_list.append(node)
            next_node_id += 1

        node_list.append(nj)

        next_ele_id = max((e.id for e in self.elements), default=0) + 1
        for i in range(n_div):
            n_start = node_list[i]
            n_end = node_list[i + 1]
            ele = FrameElement(id=next_ele_id, nodes=[n_start, n_end], section=section, transformation=transformation)
            self.elements.append(ele)
            next_ele_id += 1

        # Optional: warn about duplicates
        if any(n1.id == n2.id and n1.coords.tolist() == n2.coords.tolist() for i, n1 in enumerate(self.nodes) for n2 in self.nodes[i+1:]):
            print("⚠️ Duplicate nodes detected.")

    def get_nodes(self):
        return self.nodes

    def get_elements(self):
        return self.elements

