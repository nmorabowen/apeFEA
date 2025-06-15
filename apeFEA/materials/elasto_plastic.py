from dataclasses import dataclass, field
import numpy as np

from .material import Material


@dataclass
class EPP(Material):
    """
    Elastic-perfectly plastic 1D material model (symmetric tension/compression).
    """
    E: float               # Young's modulus
    fy: float              # Yield stress

    _eps_c: float = field(default=0.0, init=False)
    _sig_c: float = field(default=0.0, init=False)
    _eps_t: float = field(default=0.0, init=False)
    _sig_t: float = field(default=0.0, init=False)
    _eps_p_c: float = field(default=0.0, init=False)  # Committed plastic strain

    def _yield_f(self, sigma: float) -> float:
        return abs(sigma) - self.fy

    def set_trial_strain(self, eps: float) -> None:
        self._eps_t = eps
        trial_stress = self.E * (eps - self._eps_p_c)
        f = self._yield_f(trial_stress)

        if f <= 0.0:  # Elastic
            self._sig_t = trial_stress
        else:  # Plastic correction (radial return)
            sign = np.sign(trial_stress)
            self._sig_t = sign * self.fy
            self._eps_p_inc = (f / self.E) * sign

    def get_trial_stress(self) -> float:
        return self._sig_t

    def get_tangent(self) -> float:
        return self.E if self._yield_f(self._sig_t) < 0.0 else 0.0

    def commit_state(self) -> None:
        self._eps_c = self._eps_t
        self._sig_c = self._sig_t
        if hasattr(self, "_eps_p_inc"):
            self._eps_p_c += self._eps_p_inc
            del self._eps_p_inc

    def reset_trial(self) -> None:
        self._eps_t = self._eps_c
        self._sig_t = self._sig_c
