"""Yaw and range controllers (SPEC Sections 4.2 to 4.4).

Imports only numpy and the standard library. The ground station imports this
file directly.

Both controllers run once per camera frame, with t the frame capture time.
When no measurement arrives for longer than loss_timeout, each commands zero
and resets its integrator. Each anti-windup clamp limits the integral term to
the output limit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # type hints only, so the file stands alone on the ground station
    from .estimator import RangeEstimate


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


class YawController:
    """omega_c = Kp*beta_m + Ki*integral(beta_m dt). Positive bearing (person left) turns left."""

    def __init__(self, kp: float, ki: float, omega_max: float, loss_timeout: float):
        self.kp = kp
        self.ki = ki
        self.omega_max = omega_max
        self.loss_timeout = loss_timeout
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self.command = 0.0
        self.lost = False
        self._t_meas: float | None = None

    def update(self, bearing: float | None, t: float) -> float:
        """Return the yaw-rate command (rad/s) for a bearing measurement (rad) or None."""
        if bearing is None:
            if self._t_meas is not None and t - self._t_meas > self.loss_timeout:
                self.integral = 0.0
                self.command = 0.0
                self.lost = True
            return self.command  # hold the last command before the timeout

        dt = 0.0 if self._t_meas is None else t - self._t_meas
        self._t_meas = t
        self.lost = False
        self.integral += bearing * dt
        if self.ki != 0.0:
            self.integral = _clamp(self.integral, self.omega_max / abs(self.ki))
        self.command = _clamp(self.kp * bearing + self.ki * self.integral, self.omega_max)
        return self.command


class RangeController:
    """theta_c = Kp*e + Kd*r_dot_hat + Ki*integral(e dt), e = r_hat - r_ref.

    Positive error (drone too far) gives a positive pitch command (nose down, forward).
    """

    def __init__(
        self,
        kp: float,
        kd: float,
        ki: float,
        r_ref: float,
        theta_max: float,
        loss_timeout: float,
    ):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.r_ref = r_ref
        self.theta_max = theta_max
        self.loss_timeout = loss_timeout
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self.command = 0.0
        self.lost = False
        self._t_prev: float | None = None

    def update(self, estimate: RangeEstimate | None, t: float) -> float:
        """Return the pitch command (rad) from the range estimator output, or None before track."""
        if estimate is None or t - estimate.t_meas > self.loss_timeout:
            self.integral = 0.0
            self.command = 0.0
            self.lost = estimate is not None
            self._t_prev = None
            return self.command

        dt = 0.0 if self._t_prev is None else t - self._t_prev
        self._t_prev = t
        self.lost = False
        e = estimate.r - self.r_ref
        self.integral += e * dt
        if self.ki != 0.0:
            self.integral = _clamp(self.integral, self.theta_max / abs(self.ki))
        self.command = _clamp(
            self.kp * e + self.kd * estimate.r_dot + self.ki * self.integral, self.theta_max
        )
        return self.command
