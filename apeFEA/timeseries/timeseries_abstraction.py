from abc import ABC, abstractmethod

class TimeSeries(ABC):
    """
    Abstract base class for all time-dependent load profiles.
    Subclasses implement `get_factor(t)` which returns a scale factor at time `t`.
    """

    @abstractmethod
    def get_factor(self, t: float) -> float:
        """
        Return the scaling factor at time `t`.
        """
        ...
