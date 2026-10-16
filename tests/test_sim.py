import math

import numpy as np
import pytest

from dronesim.scenarios import build_scenarios
from dronesim.sim import SimResult, run


@pytest.fixture
def scenarios(params):
    return build_scenarios(params)


def test_s1_noise_free_bearing_converges(params, gains, scenarios):
    res = run(scenarios["S1"], params, gains, seed=0, noise=False)
    b = res.ticks["bearing_true"]
    assert b[0] == pytest.approx(params.scenarios.s1_heading_offset)
    assert abs(b[-1]) < math.radians(0.5)
    assert not res.ticks["lost"].any()


@pytest.mark.parametrize("name", ["S2_far", "S2_near"])
def test_s2_noise_free_range_converges(params, gains, scenarios, name):
    res = run(scenarios[name], params, gains, seed=0, noise=False)
    assert abs(res.ticks["r_true"][-1] - params.control.r_ref) < 0.05


def test_frames_arrive_t_vision_after_capture(params, gains, scenarios):
    res = run(scenarios["S1"], params, gains, seed=0, noise=False)
    lag = res.frames["t_arrive"] - res.frames["t_capture"]
    np.testing.assert_allclose(lag, params.camera.t_vision, atol=params.plant.dt / 2)


def test_seeded_runs_repeat_and_save(params, gains, scenarios, tmp_path):
    a = run(scenarios["S2_far"], params, gains, seed=7)
    b = run(scenarios["S2_far"], params, gains, seed=7)
    np.testing.assert_array_equal(a.ticks["x"], b.ticks["x"])
    a.save(tmp_path / "run.npz")
    loaded = SimResult.load(tmp_path / "run.npz")
    assert loaded.meta["seed"] == 7
    np.testing.assert_array_equal(loaded.frames["r_meas"], a.frames["r_meas"])
