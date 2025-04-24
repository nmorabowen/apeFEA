# apeFEM/geometry/geometry.py

from .node_manager import NodeManager
from .plot_geometry import PlotGeometry

from collections import OrderedDict
import numpy as np
import logging



logger = logging.getLogger(__name__)

class Geometry(PlotGeometry):
    """
    Represents the geometric model used for FEM analysis. This class handles the
    creation and management of nodes and line elements independently of any FEM domain.

    """

    def __init__(self, name: str):
        """
        Initialize a new Geometry object.

        Args:
            name (str): Name or identifier for this geometry.
        """
        self.name = name
        self.nodes:NodeManager = NodeManager(self)    # Call the NodeManager Class to handle nodes creation and management
        #self.lines=LineMagager()    # Call the LineManager Class to handle lines creation and management

    def __repr__(self):
        """
        Returns a string summary of the geometry.
        """
        return f"Geometry(n_nodes={len(self.nodes.nodes_array)})"

