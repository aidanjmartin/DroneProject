import math

import pytest

from dronesim.params import load_params


def test_degrees_converted_to_radians(params):
    assert params.plant.theta_max == pytest.approx(math.radians(15.0))
    assert params.plant.omega_max == pytest.approx(math.radians(90.0))
    assert params.camera.hfov == pytest.approx(math.radians(120.0))


def test_focal_length_matches_spec(params):
    # SPEC Section 5: f is about 185 px at nominal values.
    assert params.camera.focal == pytest.approx(184.75, abs=0.01)


def test_effective_delay(params):
    assert params.total_delay == pytest.approx(0.120)
    assert params.effective_delay == pytest.approx(0.120 + 0.5 / 30.0)


def test_invalid_value_rejected(tmp_path):
    from dronesim.params import PARAMS_PATH

    text = PARAMS_PATH.read_text().replace("k_v: 0.5", "k_v: -0.5")
    bad = tmp_path / "params.yaml"
    bad.write_text(text)
    with pytest.raises(ValueError, match="k_v"):
        load_params(bad)
