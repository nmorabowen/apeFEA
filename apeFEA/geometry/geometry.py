# apeFEM/geometry/geometry.py

from .node_manager import NodeManager
from .line_manager import LineManager
from .plot_geometry import PlotGeometry
from .plot_config import PlotConfig

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
        self.lines:LineManager=LineManager(self)    # Call the LineManager Class to handle lines creation and management
        self.plot_config = PlotConfig()
        
    def get_lines_list(self):
        """
        Returns a list of all lines in the geometry.

        Returns:
            list: List of Line objects.
        """
        return self.lines.lines_array
    
    def get_nodes_list(self):
        """
        Returns a list of all nodes in the geometry.

        Returns:
            list: List of Node objects.
        """
        return self.nodes.nodes_array
    
    def __repr__(self):
        """
        Returns a string summary of the geometry.
        """
        return f"Geometry(n_nodes={len(self.nodes.nodes_array)})"

