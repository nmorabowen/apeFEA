"""
MeshBuilder Module
------------------
This module defines the MeshBuilder class, which generates a 1D mesh of FrameElements between two given nodes. 
It preserves original nodes and their properties (e.g., boundary conditions or loads) and automatically adds 
intermediate nodes based on a specified mesh size.

Author: Patricio Palacios B., M.Sc.
Version: 1.0
"""

import numpy as np
from math import sqrt, ceil

from apeFEA.core.node import Node
from apeFEA.elements.one_dimension.frame_element import FrameElement
from apeFEA.sections.section import Section
from apeFEA.elements.one_dimension.transformations.transformation import Transformation


class MeshBuilder:
    def __init__(self):
        self.nodes: list[Node] = []
        self.elements: list[FrameElement] = []

    def mesh_line(self, ni: Node, nj: Node, mesh_size: float, section: Section, transformation: Transformation = None):
        xi, yi = ni.coords
        xj, yj = nj.coords

        total_length = sqrt((xj - xi) ** 2 + (yj - yi) ** 2)
        n_div = ceil(total_length / mesh_size)

        dx = (xj - xi) / n_div
        dy = (yj - yi) / n_div

        if not any(n.id == ni.id for n in self.nodes):
            self.nodes.append(ni)
        if not any(n.id == nj.id for n in self.nodes):
            self.nodes.append(nj)

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

        if any(n1.id == n2.id and np.allclose(n1.coords, n2.coords) for i, n1 in enumerate(self.nodes) for n2 in self.nodes[i+1:]):
            print("⚠️ Duplicate nodes detected.")

    def renumber_nodes(self, start_id=1):
        """
        Renumber all nodes with new sequential IDs starting from `start_id`.
        Updates references in all FrameElements accordingly.
        """
        id_map = {}
        for new_id, node in enumerate(self.nodes, start=start_id):
            id_map[node.id] = new_id
            node.id = new_id

        for ele in self.elements:
            ele.nodes = [next(n for n in self.nodes if n.id == id_map[old.id]) for old in ele.nodes]

    def assign_dof_indices(self, ndof: int):
        """
        Assign sequential degrees of freedom (DoF) indices to each node.
        """
        for i, node in enumerate(self.nodes):
            node.idx = np.array([i * ndof + j for j in range(ndof)])

    def get_nodes(self):
        return self.nodes

    def get_elements(self):
        return self.elements
