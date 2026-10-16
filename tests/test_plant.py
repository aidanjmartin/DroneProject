import math

import pytest

from dronesim.plant import Plant


def run(plant, omega_c, theta_c, duration):
    for _ in range(round(duration / plant.p.dt)):
        plant.step(omega_c, theta_c)


def test_constant_pitch_settles_speed(params):
    p = params.plant
    theta_c = math.radians(10.0)
    plant = Plant(p)
    run(plant, 0.0, theta_c, 30.0)  # 15 drag time constants
    assert plant.v == pytest.approx(p.g * math.tan(theta_c) / p.k_v, rel=1e-4)
    assert plant.x > 0  # positive pitch (nose down) moves the drone forward


def test_yaw_rate_step_gives_heading_ramp_after_lag(params):
    p = params.plant
    omega_c = math.radians(30.0)
    plant = Plant(p)
    t = 1.0
    run(plant, omega_c, 0.0, t)
    expected = omega_c * (t - p.tau_psi * (1 - math.exp(-t / p.tau_psi)))
    assert plant.psi == pytest.approx(expected, rel=1e-6)
    assert plant.omega == pytest.approx(omega_c, rel=1e-6)


def test_commands_saturate_at_plant_input(params):
    p = params.plant
    plant = Plant(p)
    run(plant, 100.0, 100.0, 2.0)
    assert plant.omega == pytest.approx(p.omega_max, rel=1e-6)
    assert plant.theta == pytest.approx(p.theta_max, rel=1e-6)
