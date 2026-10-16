"""Load and validate config/params.yaml and config/gains.yaml into dataclasses.

Keys ending in ``_deg`` are converted to radians and stored without the suffix.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
PARAMS_PATH = CONFIG_DIR / "params.yaml"
GAINS_PATH = CONFIG_DIR / "gains.yaml"

DEG_SUFFIX = "_deg"


@dataclass(frozen=True)
class PlantParams:
    g: float
    tau_psi: float
    tau_theta: float
    k_v: float
    theta_max: float
    omega_max: float
    dt: float


@dataclass(frozen=True)
class CameraParams:
    hfov: float
    width: float
    torso_length: float
    sigma_d: float
    sigma_u: float
    d_min: float
    f_cam: float
    t_vision: float
    p_drop: float

    @property
    def focal(self) -> float:
        """Focal length in pixels, f = (W/2) / tan(HFOV/2)."""
        return (self.width / 2) / math.tan(self.hfov / 2)


@dataclass(frozen=True)
class LinkParams:
    f_link: float
    t_link: float


@dataclass(frozen=True)
class ControlParams:
    r_ref: float
    loss_timeout: float


@dataclass(frozen=True)
class EstimatorParams:
    q_a: float
    v0_std: float
    predict_latency: bool


@dataclass(frozen=True)
class ScenarioParams:
    duration: float
    s1_heading_offset: float
    s2_range_offset: float
    s3_speed: float
    s3_walk_time: float
    s4_speed: float
    s4_walk_time: float
    s5_start_offset: float
    s5_speed: float
    s5_walk_time: float


@dataclass(frozen=True)
class MonteCarloParams:
    n_seeds: int
    base_seed: int


@dataclass(frozen=True)
class Params:
    plant: PlantParams
    camera: CameraParams
    link: LinkParams
    control: ControlParams
    estimator: EstimatorParams
    scenarios: ScenarioParams
    monte_carlo: MonteCarloParams

    @property
    def total_delay(self) -> float:
        """Total delay: vision latency plus command link latency, s."""
        return self.camera.t_vision + self.link.t_link

    @property
    def effective_delay(self) -> float:
        """Effective delay: total delay plus the camera zero-order hold, s."""
        return self.total_delay + 0.5 / self.camera.f_cam

    def replace(self, section: str, **changes: Any) -> "Params":
        """Return a copy with fields in one section changed (for sweeps)."""
        new_section = dataclasses.replace(getattr(self, section), **changes)
        return dataclasses.replace(self, **{section: new_section})


@dataclass(frozen=True)
class YawGains:
    kp: float
    ki: float


@dataclass(frozen=True)
class RangeGains:
    kp: float
    kd: float
    ki: float


@dataclass(frozen=True)
class Gains:
    yaw: YawGains
    range: RangeGains


def _convert(section: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in section.items():
        if key.endswith(DEG_SUFFIX):
            out[key[: -len(DEG_SUFFIX)]] = math.radians(float(value))
        else:
            out[key] = value
    return out


def _build(cls: type, raw: dict[str, Any], name: str) -> Any:
    if name not in raw:
        raise ValueError(f"config is missing section '{name}'")
    values = _convert(raw[name])
    fields = {f.name: f.type for f in dataclasses.fields(cls)}
    missing = fields.keys() - values.keys()
    extra = values.keys() - fields.keys()
    if missing or extra:
        raise ValueError(
            f"section '{name}': missing keys {sorted(missing)}, unknown keys {sorted(extra)}"
        )
    typed = {}
    for key, value in values.items():
        if fields[key] == "bool":
            typed[key] = bool(value)
        elif fields[key] == "int":
            typed[key] = int(value)
        else:
            typed[key] = float(value)
    return cls(**typed)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(p: Params) -> None:
    for section in (p.plant, p.camera, p.link, p.control, p.scenarios):
        for f in dataclasses.fields(section):
            value = getattr(section, f.name)
            if f.name in ("p_drop", "t_link"):
                _require(value >= 0, f"{f.name} must be >= 0, got {value}")
            else:
                _require(value > 0, f"{f.name} must be > 0, got {value}")
    _require(p.camera.p_drop <= 1, "p_drop must be <= 1")
    _require(p.camera.hfov < math.pi, "hfov must be below 180 deg")
    _require(p.estimator.q_a > 0, "q_a must be > 0")
    _require(p.estimator.v0_std > 0, "v0_std must be > 0")
    _require(p.monte_carlo.n_seeds > 0, "n_seeds must be > 0")


def load_params(path: str | Path = PARAMS_PATH) -> Params:
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    p = Params(
        plant=_build(PlantParams, raw, "plant"),
        camera=_build(CameraParams, raw, "camera"),
        link=_build(LinkParams, raw, "link"),
        control=_build(ControlParams, raw, "control"),
        estimator=_build(EstimatorParams, raw, "estimator"),
        scenarios=_build(ScenarioParams, raw, "scenarios"),
        monte_carlo=_build(MonteCarloParams, raw, "monte_carlo"),
    )
    validate(p)
    return p


def load_gains(path: str | Path = GAINS_PATH) -> Gains:
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    return Gains(
        yaw=_build(YawGains, raw, "yaw"),
        range=_build(RangeGains, raw, "range"),
    )
