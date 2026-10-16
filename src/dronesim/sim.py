"""Multi-rate simulation loop and logging.

Rates:
- Plant: RK4 at dt (1 ms).
- Camera: one frame every 1/f_cam s. Each frame result (a measurement or a
  missed detection) reaches the ground station T_vision after capture.
- Controllers: run once per arriving frame, keyed on the frame capture time.
- Command link: samples the latest command at f_link; each sample reaches the
  plant T_link later.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .controllers import RangeController, YawController
from .estimator import RangeEstimator
from .link import CommandLink
from .params import Gains, Params
from .plant import STATE_NAMES, Plant
from .scenarios import Scenario
from .sensor import Camera, DelayQueue, relative_geometry

TICK_FIELDS = STATE_NAMES + (
    "t", "person_x", "person_y", "person_vx", "person_vy",
    "r_true", "bearing_true", "omega_cmd", "theta_cmd", "omega_applied", "theta_applied",
    "lost",
)
FRAME_FIELDS = (
    "t_capture", "t_arrive", "detected", "r_true", "bearing_true", "r_meas", "bearing_meas",
    "r_est", "r_dot_est", "omega_cmd", "theta_cmd", "lost",
)


@dataclass
class SimResult:
    ticks: dict[str, np.ndarray]
    frames: dict[str, np.ndarray]
    meta: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        arrays = {f"tick_{k}": v for k, v in self.ticks.items()}
        arrays.update({f"frame_{k}": v for k, v in self.frames.items()})
        np.savez_compressed(path, meta=json.dumps(self.meta), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "SimResult":
        with np.load(path) as data:
            ticks = {k[5:]: data[k] for k in data.files if k.startswith("tick_")}
            frames = {k[6:]: data[k] for k in data.files if k.startswith("frame_")}
            meta = json.loads(str(data["meta"]))
        return cls(ticks, frames, meta)


def run(scenario: Scenario, params: Params, gains: Gains, seed: int, noise: bool = True) -> SimResult:
    p = params
    dt = p.plant.dt
    tol = dt / 2
    n_steps = round(scenario.duration / dt)
    frame_period = 1.0 / p.camera.f_cam

    rng = np.random.default_rng(seed)
    plant = Plant(p.plant, psi=scenario.drone_psi0)
    camera = Camera(p.camera, rng, noise=noise)
    vision_queue = DelayQueue(p.camera.t_vision, tol)
    link = CommandLink(p.link.t_link, p.link.f_link, initial=(0.0, 0.0), tol=tol)
    estimator = RangeEstimator(
        q_a=p.estimator.q_a,
        sigma_d=p.camera.sigma_d,
        focal=p.camera.focal,
        torso_length=p.camera.torso_length,
        v0_std=p.estimator.v0_std,
        t_vision=p.camera.t_vision,
        predict_latency=p.estimator.predict_latency,
    )
    yaw = YawController(gains.yaw.kp, gains.yaw.ki, p.plant.omega_max, p.control.loss_timeout)
    rng_ctl = RangeController(
        gains.range.kp, gains.range.kd, gains.range.ki,
        p.control.r_ref, p.plant.theta_max, p.control.loss_timeout,
    )

    ticks = {k: np.zeros(n_steps + 1) for k in TICK_FIELDS}
    frames: dict[str, list] = {k: [] for k in FRAME_FIELDS}
    command = (0.0, 0.0)
    n_frame = 0

    for k in range(n_steps + 1):
        t = k * dt
        px, py, pvx, pvy = scenario.person(t)
        r_true, bearing_true = relative_geometry(plant.x, plant.y, plant.psi, px, py)

        if t >= n_frame * frame_period - tol:
            meas = camera.capture(r_true, bearing_true)
            vision_queue.push(t, (t, r_true, bearing_true, meas))
            n_frame += 1

        for t_cap, r_cap, b_cap, meas in vision_queue.pop_ready(t):
            estimate = estimator.update(meas.range if meas else None, t_cap)
            omega_c = yaw.update(meas.bearing if meas else None, t_cap)
            theta_c = rng_ctl.update(estimate, t_cap)
            command = (omega_c, theta_c)
            link.set_command(command)
            row = (
                t_cap, t, meas is not None, r_cap, b_cap,
                meas.range if meas else math.nan, meas.bearing if meas else math.nan,
                estimate.r if estimate else math.nan, estimate.r_dot if estimate else math.nan,
                omega_c, theta_c, yaw.lost or rng_ctl.lost,
            )
            for name, value in zip(FRAME_FIELDS, row):
                frames[name].append(value)

        applied = link.update(t)

        for i, name in enumerate(STATE_NAMES):
            ticks[name][k] = plant.state[i]
        row = (
            t, px, py, pvx, pvy, r_true, bearing_true,
            command[0], command[1], applied[0], applied[1], yaw.lost or rng_ctl.lost,
        )
        for name, value in zip(TICK_FIELDS[len(STATE_NAMES):], row):
            ticks[name][k] = value

        if k < n_steps:
            plant.step(applied[0], applied[1], scenario.disturbance(t))

    meta = {
        "scenario": scenario.name,
        "seed": seed,
        "noise": noise,
        "total_delay": p.total_delay,
        "effective_delay": p.effective_delay,
        "gains": {
            "yaw": vars(gains.yaw),
            "range": vars(gains.range),
        },
    }
    return SimResult(ticks, {k: np.asarray(v) for k, v in frames.items()}, meta)
