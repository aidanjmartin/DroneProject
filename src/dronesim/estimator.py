"""Range estimator: constant-velocity Kalman filter on [r, dr/dt] (SPEC Section 4.1).

Imports only numpy and the standard library. The ground station imports this
file directly.

Times passed to ``update`` are frame capture times. Measurement variance
propagates pixel noise through the inverse range mapping:
R = (r_hat^2 * sigma_d / (f*L))^2.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class RangeEstimate(NamedTuple):
    r: float        # range estimate, m
    r_dot: float    # range-rate estimate, m/s
    t_meas: float   # capture time of the last measurement used, s


class RangeEstimator:
    def __init__(
        self,
        q_a: float,
        sigma_d: float,
        focal: float,
        torso_length: float,
        v0_std: float,
        t_vision: float = 0.0,
        predict_latency: bool = False,
    ):
        self.q_a = q_a
        self.sigma_d = sigma_d
        self.focal = focal
        self.torso_length = torso_length
        self.v0_std = v0_std
        self.t_vision = t_vision
        self.predict_latency = predict_latency
        self.reset()

    def reset(self) -> None:
        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self._t: float | None = None
        self._t_meas: float | None = None

    def measurement_variance(self, r: float) -> float:
        return (r * r * self.sigma_d / (self.focal * self.torso_length)) ** 2

    def _predict(self, t: float) -> None:
        dt = t - self._t
        if dt <= 0:
            return
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = self.q_a * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self._t = t

    def update(self, r_m: float | None, t: float) -> RangeEstimate | None:
        """Fold in a range measurement captured at time t, or coast to t when r_m is None.

        Returns None until the first measurement arrives.
        """
        if self.x is None:
            if r_m is None:
                return None
            self.x = np.array([r_m, 0.0])
            self.P = np.diag([self.measurement_variance(r_m), self.v0_std**2])
            self._t = self._t_meas = t
            return self.estimate()

        self._predict(t)
        if r_m is not None:
            S = self.P[0, 0] + self.measurement_variance(self.x[0])
            K = self.P[:, 0] / S
            self.x = self.x + K * (r_m - self.x[0])
            self.P = self.P - np.outer(K, self.P[0, :])
            self.P = 0.5 * (self.P + self.P.T)
            self._t_meas = t
        return self.estimate()

    def estimate(self) -> RangeEstimate | None:
        if self.x is None:
            return None
        r, r_dot = float(self.x[0]), float(self.x[1])
        if self.predict_latency:
            r += r_dot * self.t_vision
        return RangeEstimate(r, r_dot, self._t_meas)
