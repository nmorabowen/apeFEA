from dataclasses import dataclass, field
from .material import Material


@dataclass
class Concrete01(Material):
    """
    Compressive-only concrete model (similar to OpenSees Concrete01).
    Only covers the loading envelope (no unloading/reloading paths).

    Parameters
    ----------
    E0 : float
        Initial elastic stiffness (secant modulus).
    fc : float
        Peak compressive strength (negative).
    eps_c0 : float
        Strain at peak strength.
    fcu : float
        Residual crushing strength (more negative than fc).
    eps_u : float
        Strain at ultimate crushing strength.
    """
    E0: float
    fc: float
    eps_c0: float
    fcu: float
    eps_u: float

    _eps_c: float = field(default=0.0, init=False)
    _sig_c: float = field(default=0.0, init=False)
    _eps_t: float = field(default=0.0, init=False)
    _sig_t: float = field(default=0.0, init=False)

    def _envelope(self, eps: float) -> float:
        """
        Stress-strain envelope curve: ascending and descending compressive branches.
        No tensile resistance.
        """
        if eps >= 0:
            return 0.0
        if eps > self.eps_c0:
            # Ascending parabolic branch
            return self.fc * (eps / self.eps_c0) * (2 - eps / self.eps_c0)
        if eps > self.eps_u:
            # Linear descending branch
            return (self.fc - self.fcu) * ((eps - self.eps_c0) / (self.eps_u - self.eps_c0)) + self.fc
        return self.fcu  # Crushed state

    def set_trial_strain(self, eps: float) -> None:
        self._eps_t = eps
        self._sig_t = self._envelope(eps)

    def get_trial_stress(self) -> float:
        return self._sig_t

    def get_tangent(self) -> float:
        # Return secant stiffness as a simple approximation
        if self._eps_t == 0.0:
            return self.E0
        return self._sig_t / self._eps_t

    def commit_state(self) -> None:
        self._eps_c = self._eps_t
        self._sig_c = self._sig_t

    def reset_trial(self) -> None:
        self._eps_t = self._eps_c
        self._sig_t = self._sig_c
