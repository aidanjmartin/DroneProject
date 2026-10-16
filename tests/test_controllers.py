import math

import pytest

from dronesim.controllers import RangeController, YawController
from dronesim.estimator import RangeEstimate

TIMEOUT = 0.5


def test_yaw_holds_then_zeroes_after_loss_timeout():
    yaw = YawController(kp=2.0, ki=1.0, omega_max=10.0, loss_timeout=TIMEOUT)
    yaw.update(0.1, 0.0)
    cmd = yaw.update(0.1, 0.1)
    assert yaw.update(None, 0.5) == cmd  # before timeout: hold
    assert not yaw.lost
    assert yaw.update(None, 0.7) == 0.0  # after timeout: zero
    assert yaw.lost and yaw.integral == 0.0


def test_yaw_integrator_and_output_clamped():
    yaw = YawController(kp=1.0, ki=2.0, omega_max=1.0, loss_timeout=TIMEOUT)
    t = 0.0
    for _ in range(1000):
        cmd = yaw.update(1.0, t)
        t += 0.1
    assert cmd == pytest.approx(1.0)
    assert abs(yaw.ki * yaw.integral) <= 1.0 + 1e-12


def test_range_zeroes_after_loss_timeout():
    ctl = RangeController(kp=0.1, kd=0.1, ki=0.05, r_ref=3.0, theta_max=0.3, loss_timeout=TIMEOUT)
    est = RangeEstimate(4.0, 0.0, 0.0)
    assert ctl.update(est, 0.0) > 0
    assert ctl.update(est, 0.4) > 0  # coasting estimate, before timeout
    assert ctl.update(est, 0.6) == 0.0
    assert ctl.lost and ctl.integral == 0.0


def test_range_output_saturates():
    ctl = RangeController(kp=1.0, kd=0.0, ki=0.0, r_ref=3.0, theta_max=math.radians(15), loss_timeout=TIMEOUT)
    assert ctl.update(RangeEstimate(30.0, 0.0, 0.0), 0.0) == pytest.approx(math.radians(15))
