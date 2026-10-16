import math

import numpy as np
import pytest

from dronesim.sensor import Camera, relative_geometry


@pytest.fixture
def camera(params):
    return Camera(params.camera, np.random.default_rng(0), noise=False)


@pytest.mark.parametrize("r", [1.0, 3.0, 5.0, 8.0])
@pytest.mark.parametrize("beta_deg", [-50.0, -10.0, 0.0, 25.0, 55.0])
def test_noise_free_inversion(camera, r, beta_deg):
    m = camera.capture(r, math.radians(beta_deg))
    assert m.range == pytest.approx(r, rel=1e-12)
    assert m.bearing == pytest.approx(math.radians(beta_deg), abs=1e-12)


def test_person_outside_fov_not_measured(camera, params):
    edge = params.camera.hfov / 2
    assert camera.capture(3.0, edge + 0.01) is None
    assert camera.capture(3.0, -edge - 0.01) is None
    assert camera.capture(3.0, edge - 0.01) is not None


def test_person_too_far_not_measured(camera, params):
    c = params.camera
    r_limit = camera.focal * c.torso_length / c.d_min
    assert camera.capture(r_limit * 0.99, 0.0) is not None
    assert camera.capture(r_limit * 1.01, 0.0) is None


def test_bearing_sign_left_is_positive():
    # Person to the left of a drone facing +x.
    r, beta = relative_geometry(0.0, 0.0, 0.0, 3.0, 1.0)
    assert beta > 0
    assert r == pytest.approx(math.hypot(3.0, 1.0))


def test_person_left_appears_left_of_image_center(camera, params):
    m = camera.capture(3.0, math.radians(20.0))
    assert m.u_px < params.camera.width / 2


def test_range_noise_matches_spec(params):
    # SPEC Section 5: about 0.15 m at 3 m and 0.41 m at 5 m.
    camera = Camera(params.camera, np.random.default_rng(1))
    assert camera.range_std(3.0) == pytest.approx(0.146, abs=0.002)
    assert camera.range_std(5.0) == pytest.approx(0.406, abs=0.002)
    samples = [camera.capture(3.0, 0.0).range for _ in range(20000)]
    assert np.std(samples) == pytest.approx(camera.range_std(3.0), rel=0.05)


def test_dropout_probability(params):
    camera = Camera(params.replace("camera", p_drop=0.1).camera, np.random.default_rng(2))
    misses = sum(camera.capture(3.0, 0.0) is None for _ in range(20000))
    assert misses / 20000 == pytest.approx(0.1, abs=0.01)
