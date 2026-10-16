import numpy as np
import pytest

from dronesim.estimator import RangeEstimator


def make(params, **kw):
    c = params.camera
    return RangeEstimator(params.estimator.q_a, c.sigma_d, c.focal, c.torso_length,
                          params.estimator.v0_std, c.t_vision, **kw)


def test_tracks_constant_velocity_noise_free(params):
    est = make(params)
    dt = 1 / params.camera.f_cam
    for i in range(300):
        t = i * dt
        out = est.update(3.0 + 1.4 * t, t)
    assert out.r == pytest.approx(3.0 + 1.4 * t, abs=1e-3)
    assert out.r_dot == pytest.approx(1.4, abs=1e-3)


def test_coasts_without_measurement(params):
    est = make(params)
    dt = 1 / params.camera.f_cam
    for i in range(300):
        est.update(3.0 + 1.0 * i * dt, i * dt)
    t_last = 299 * dt
    out = est.update(None, t_last + 0.3)
    assert out.r == pytest.approx(3.0 + 1.0 * (t_last + 0.3), abs=1e-2)
    assert out.t_meas == pytest.approx(t_last)


def test_predict_latency_shifts_estimate(params):
    est = make(params, predict_latency=True)
    dt = 1 / params.camera.f_cam
    for i in range(300):
        t = i * dt
        out = est.update(3.0 + 1.0 * t, t)
    assert out.r == pytest.approx(3.0 + 1.0 * (t + params.camera.t_vision), abs=1e-2)


def test_smooths_noise(params):
    rng = np.random.default_rng(0)
    est = make(params)
    sigma = 0.15
    errs = []
    for i in range(600):
        out = est.update(3.0 + sigma * rng.standard_normal(), i / params.camera.f_cam)
        if i > 100:
            errs.append(out.r - 3.0)
    assert np.std(errs) < 0.5 * sigma


def test_no_estimate_before_first_measurement(params):
    est = make(params)
    assert est.update(None, 0.0) is None
