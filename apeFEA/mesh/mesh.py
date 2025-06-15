import numpy as np
from math import sqrt, ceil

from apeFEA.core.node import Node
from apeFEA.elements.one_dimension.frame_element import FrameElement
from apeFEA.sections.section import Section  # if needed for type hinting
from apeFEA.elements.one_dimension.transformations.transformation import Transformation  # for type hinting


def mesh_line(ni:Node, nj:Node, mesh_size, section:Section, transformation:Transformation = None):
    """
    Genera una malla de elementos entre dos nodos (ni, nj), 
    subdividida por tamaño máximo mesh_size. Agrega directamente
    a las listas globales 'nodes' y 'elements' con IDs automáticos.
    """
    global nodes, elements
    from math import sqrt, ceil

    xi, yi = ni.coords
    xj, yj = nj.coords
    
    # Longitud total
    total_length = sqrt((xj - xi) ** 2 + (yj - yi) ** 2)
    n_div = ceil(total_length / mesh_size)

    dx = (xj - xi) / n_div
    dy = (yj - yi) / n_div

    # Lista de nodos para este tramo
    node_list = [ni]
    next_node_id = max(n.id for n in nodes) + 1 if nodes else 1

    for i in range(1, n_div):
        x = xi + i * dx
        y = yi + i * dy
        node = Node(next_node_id, [x, y])
        nodes.append(node)
        node_list.append(node)
        next_node_id += 1

    node_list.append(nj)

    # ID inicial del nuevo elemento (último existente + 1)
    if elements:
        next_ele_id = max(e.id for e in elements) + 1
    else:
        next_ele_id = 1

    # Crear elementos
    for i in range(n_div):
        n_start = node_list[i]
        n_end = node_list[i + 1]
        ele = FrameElement(id=next_ele_id, nodes=[n_start, n_end], section=section, transformation=transformation)
        elements.append(ele)
        next_ele_id += 1
