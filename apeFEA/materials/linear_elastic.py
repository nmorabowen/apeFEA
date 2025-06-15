from dataclasses import dataclass, field
from .material import Material  # assumes Material is in base.py


@dataclass
class LinearElastic(Material):
    """
    Linear elastic material with constant stiffness E.
    """
    E: float  # Young's modulus

    _eps_c: float = field(default=0.0, init=False)
    _sig_c: float = field(default=0.0, init=False)
    _eps_t: float = field(default=0.0, init=False)
    _sig_t: float = field(default=0.0, init=False)

    def set_trial_strain(self, eps: float) -> None:
        self._eps_t = eps
        self._sig_t = self.E * eps

    def get_trial_stress(self) -> float:
        return self._sig_t

    def get_tangent(self) -> float:
        return self.E

    def commit_state(self) -> None:
        self._eps_c = self._eps_t
        self._sig_c = self._sig_t

    def reset_trial(self) -> None:
        self._eps_t = self._eps_c
        self._sig_t = self._sig_c
