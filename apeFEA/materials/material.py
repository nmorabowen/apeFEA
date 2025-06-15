import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


# ----------------------------------------------------------------------------- #
#                           Abstract Base Class                                 #
# ----------------------------------------------------------------------------- #

class Material(ABC):
    """
    Abstract base class for 1D uniaxial constitutive models.
    Defines interface for trial stress/strain evaluation and state updates.
    """

    @abstractmethod
    def set_trial_strain(self, eps: float) -> None: ...
    
    @abstractmethod
    def get_trial_stress(self) -> float: ...
    
    @abstractmethod
    def get_tangent(self) -> float: ...

    @abstractmethod
    def commit_state(self) -> None: ...
    
    @abstractmethod
    def reset_trial(self) -> None: ...

    def plot(
        self,
        strain_range: np.ndarray,
        ax: Optional[plt.Axes] = None,
        **kwargs,
    ):
        """
        Quick 1-D stress-strain plot for inspection.

        Parameters
        ----------
        strain_range : np.ndarray
            Strain values to evaluate.
        ax : matplotlib Axes, optional
            Axis to plot into (will create one if not provided).
        kwargs : dict
            Passed to `ax.plot()` for styling.

        Returns
        -------
        fig, ax : Figure and Axes objects for further modification
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        stress_vals = np.empty_like(strain_range, dtype=float)
        for i, eps in enumerate(strain_range):
            self.set_trial_strain(float(eps))
            stress_vals[i] = self.get_trial_stress()
            self.reset_trial()

        ax.plot(strain_range, stress_vals, **kwargs)
        ax.set_xlabel("Strain, ε")
        ax.set_ylabel("Stress, σ")
        ax.set_title(type(self).__name__)
        ax.grid(True)
        return fig, ax


