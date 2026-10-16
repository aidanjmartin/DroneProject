"""Person trajectories and initial conditions (SPEC Section 6).

The drone starts at the origin facing +x. Each scenario's ``person(t)``
returns (x, y, vx, vy) in meters and m/s.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .params import Params

PersonFn = Callable[[float], tuple[float, float, float, float]]


def _no_disturbance(t: float) -> float:
    return 0.0


@dataclass(frozen=True)
class Scenario:
    name: str
    duration: float
    person: PersonFn
    drone_psi0: float = 0.0
    disturbance: Callable[[float], float] = field(default=_no_disturbance)


def walk(x0: float, y0: float, vx: float, vy: float, t_walk: float) -> PersonFn:
    """Walk at constant velocity from (x0, y0) for t_walk seconds, then stand still."""

    def person(t: float) -> tuple[float, float, float, float]:
        tc = min(max(t, 0.0), t_walk)
        moving = 0.0 <= t < t_walk
        return x0 + vx * tc, y0 + vy * tc, vx if moving else 0.0, vy if moving else 0.0

    return person


def s1_yaw_step(p: Params) -> Scenario:
    """Person stationary at r_ref ahead; drone heading starts offset so the person is to the left."""
    return Scenario(
        name="S1",
        duration=p.scenarios.duration,
        person=walk(p.control.r_ref, 0.0, 0.0, 0.0, 0.0),
        drone_psi0=-p.scenarios.s1_heading_offset,
    )


def s2_range_step(p: Params, sign: int) -> Scenario:
    """Person stationary and aligned; drone starts at r_ref + sign * offset."""
    r0 = p.control.r_ref + sign * p.scenarios.s2_range_offset
    return Scenario(
        name="S2_far" if sign > 0 else "S2_near",
        duration=p.scenarios.duration,
        person=walk(r0, 0.0, 0.0, 0.0, 0.0),
    )


def build_scenarios(p: Params) -> dict[str, Scenario]:
    scenarios = [s1_yaw_step(p), s2_range_step(p, +1), s2_range_step(p, -1)]
    return {s.name: s for s in scenarios}
