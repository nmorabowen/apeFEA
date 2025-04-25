# apeFEA/sections/section.py
from abc import ABC, abstractmethod

class Section(ABC):
    """
    Base abstract class for all cross-section types.
    """

    def __init__(self, name=None):
        self.name = name

    @abstractmethod
    def get_area(self):
        """Return cross-sectional area A."""
        pass

    @abstractmethod
    def get_inertia(self):
        """Return moment of inertia I (or Iy, Iz, etc. depending on the section)."""
        pass

    def __str__(self):
        return f"{self.name} ({self.__class__.__name__})"

    def __repr__(self):
        return self.__str__()
