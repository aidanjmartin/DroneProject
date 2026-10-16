"""Nonlinear planar plant: the Betaflight-stabilized drone as the ground station sees it.

States (SPEC Section 3.2): x, y, psi, omega, v, theta.

    domega/dt = (omega_c - omega) / tau_psi
    dpsi/dt   = omega
    dtheta/dt = (theta_c - theta) / tau_theta
    dv/dt     = g*tan(theta) - k_v*v + a_d
    dx/dt     = v*cos(psi)
    dy/dt     = v*sin(psi)
"""

from __future__ import annotations

import math

import numpy as np

from .params import PlantParams

STATE_NAMES = ("x", "y", "psi", "omega", "v", "theta")


def saturate(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


class Plant:
    def __init__(self, p: PlantParams, x: float = 0.0, y: float = 0.0, psi: float = 0.0):
        self.p = p
        self.state = np.array([x, y, psi, 0.0, 0.0, 0.0])

    def __getattr__(self, name: str) -> float:
        if name in STATE_NAMES:
            return float(self.state[STATE_NAMES.index(name)])
        raise AttributeError(name)

    def _deriv(self, s, omega_c: float, theta_c: float, a_d: float):
        p = self.p
        _, _, psi, omega, v, theta = s
        return (
            v * math.cos(psi),
            v * math.sin(psi),
            omega,
            (omega_c - omega) / p.tau_psi,
            p.g * math.tan(theta) - p.k_v * v + a_d,
            (theta_c - theta) / p.tau_theta,
        )

    def step(self, omega_c: float, theta_c: float, a_d: float = 0.0) -> np.ndarray:
        """Advance one RK4 step of length dt. Commands are held across the step."""
        omega_c = saturate(omega_c, self.p.omega_max)
        theta_c = saturate(theta_c, self.p.theta_max)
        h = self.p.dt
        s = self.state.tolist()
        k1 = self._deriv(s, omega_c, theta_c, a_d)
        k2 = self._deriv([si + 0.5 * h * ki for si, ki in zip(s, k1)], omega_c, theta_c, a_d)
        k3 = self._deriv([si + 0.5 * h * ki for si, ki in zip(s, k2)], omega_c, theta_c, a_d)
        k4 = self._deriv([si + h * ki for si, ki in zip(s, k3)], omega_c, theta_c, a_d)
        self.state = np.array(
            [si + h / 6 * (a + 2 * b + 2 * c + d) for si, a, b, c, d in zip(s, k1, k2, k3, k4)]
        )
        return self.state
