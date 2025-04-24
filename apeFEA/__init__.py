# Upper Folder (apeFEA/apeFEA/__init__.py)
"""apeFEA - Finite-element module for structural analysis"""

__version__ = "0.1.0"

# Import core components for easy access
from .core.node import Node
from .core.config import FEMConfig


# Import geometry components
from .geometry import Geometry

# Import domain components
from .domain.domain import Domain
from .domain.node_domain import NodeDomain

