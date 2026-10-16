import math

from dronesim.controllers import RangeController, YawController
from dronesim.estimator import RangeEstimate


def test_person_left_gives_positive_yaw_rate(params, gains):
    yaw = YawController(gains.yaw.kp, gains.yaw.ki, params.plant.omega_max, params.control.loss_timeout)
    assert yaw.update(math.radians(10.0), 0.0) > 0
    yaw.reset()
    assert yaw.update(math.radians(-10.0), 0.0) < 0


def test_person_too_far_gives_positive_pitch(params, gains):
    g = gains.range
    ctl = RangeController(g.kp, g.kd, g.ki, params.control.r_ref, params.plant.theta_max,
                          params.control.loss_timeout)
    far = RangeEstimate(params.control.r_ref + 1.0, 0.0, 0.0)
    assert ctl.update(far, 0.0) > 0
    near = RangeEstimate(params.control.r_ref - 1.0, 0.0, 0.0)
    assert ctl.update(near, 0.0) < 0
