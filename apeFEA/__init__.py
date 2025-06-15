"""
apeFEA — Nonlinear FEM framework for structural analysis.
LARGA VIDA AL LADRUÑO!!!

Public API shortcut imports for convenience:
    - TimeSeries classes
    - Solver interfaces
    - Core modeling components
"""

# Core components
from .core.node import Node
from .core.nodal_load import NodalLoad
from .core.restraints import Restraints
from .core.model import Model

# Material imports
from .materials.linear_elastic import LinearElastic
from .materials.elasto_plastic import EPP
from .materials.concrete01 import Concrete01

# Section imports
from .sections.section import Section

# Transformation imports
from .elements.one_dimension.transformations.linear_transformation import LinearTransformation
from .elements.one_dimension.transformations.corrotational_transformation import CorotationalTransformation2D
from .elements.one_dimension.transformations.pdelta_transformation import PDeltaTransformation2D

# Frame elements import
from .elements.one_dimension.frame_element import FrameElement

# TimeSeries models
from .timeseries import ConstantTimeSeries, LinearRampTimeSeries

# Solver imports
from .solver.newton_raphson import NewtonRaphsonSolver

# Integrator imports
from .integrator.load_control import LoadControl

# Meshing utilities
from .mesh.mesh import MeshBuilder

__all__ = [
    "Node",
    "NodalLoad",
    "Restraints",
    "ConstantTimeSeries",
    "LinearRampTimeSeries",
    "LinearElastic",
    "EPP",
    "Concrete01",
    "Section",
    "LinearTransformation",
    "CorotationalTransformation2D",
    "PDeltaTransformation2D",
    "FrameElement",
    "Model",
    "NewtonRaphsonSolver",
    "LoadControl",
    "MeshBuilder"
]
