"""Camera keypoint sensor: pinhole range and bearing, noise, detection rules, delay queue.

Range (SPEC Section 3.4):  d = f*L/r + n_d,  r_m = f*L/d
Bearing:                   u = W/2 - f*tan(beta) + n_u,  beta_m = atan((W/2 - u)/f)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from .params import CameraParams


def wrap_angle(a: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def relative_geometry(x: float, y: float, psi: float, px: float, py: float) -> tuple[float, float]:
    """True range and bearing of the person at (px, py) from a drone at (x, y) with heading psi.

    Bearing is positive when the person is to the left of the nose.
    """
    dx, dy = px - x, py - y
    return math.hypot(dx, dy), wrap_angle(math.atan2(dy, dx) - psi)


@dataclass(frozen=True)
class Measurement:
    range: float      # r_m, m
    bearing: float    # beta_m, rad
    d_px: float       # torso pixel separation, px
    u_px: float       # torso center horizontal pixel from the left edge, px


class Camera:
    """Generates one keypoint measurement per frame, or None when detection fails."""

    def __init__(self, p: CameraParams, rng: np.random.Generator, noise: bool = True):
        self.p = p
        self.rng = rng
        self.noise = noise
        self.focal = p.focal

    def capture(self, r_true: float, beta_true: float) -> Measurement | None:
        p = self.p
        # Draw every random number each frame so the noise sequence does not
        # depend on which detection rule fires.
        n_d, n_u = self.rng.standard_normal(2)
        dropped = self.rng.random() < p.p_drop
        if not self.noise:
            n_d = n_u = 0.0
            dropped = False

        if abs(beta_true) > p.hfov / 2 or dropped:
            return None
        d = self.focal * p.torso_length / r_true + p.sigma_d * n_d
        if d < p.d_min:
            return None
        u = p.width / 2 - self.focal * math.tan(beta_true) + p.sigma_u * n_u
        return Measurement(
            range=self.focal * p.torso_length / d,
            bearing=math.atan((p.width / 2 - u) / self.focal),
            d_px=d,
            u_px=u,
        )

    def range_std(self, r: float | np.ndarray) -> float | np.ndarray:
        """First-order range standard deviation, r^2 * sigma_d / (f*L)."""
        return np.asarray(r) ** 2 * self.p.sigma_d / (self.focal * self.p.torso_length)


class DelayQueue:
    """FIFO that releases each item a fixed delay after it was pushed.

    ``tol`` absorbs floating-point error in the time grid; the simulation
    passes half a physics step.
    """

    def __init__(self, delay: float, tol: float):
        self.delay = delay
        self.tol = tol
        self._items: deque[tuple[float, Any]] = deque()

    def push(self, t: float, item: Any) -> None:
        self._items.append((t + self.delay, item))

    def pop_ready(self, t: float) -> list[Any]:
        ready = []
        while self._items and self._items[0][0] <= t + self.tol:
            ready.append(self._items.popleft()[1])
        return ready

    def __len__(self) -> int:
        return len(self._items)
