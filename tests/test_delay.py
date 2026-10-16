import pytest

from dronesim.link import CommandLink
from dronesim.sensor import DelayQueue

DT = 0.001
TOL = DT / 2


def test_measurement_available_exactly_t_vision_after_capture(params):
    t_vision = params.camera.t_vision
    q = DelayQueue(t_vision, TOL)
    t_capture = 0.4
    q.push(t_capture, "frame")
    assert q.pop_ready(t_capture + t_vision - DT) == []
    assert q.pop_ready(t_capture + t_vision) == ["frame"]
    assert len(q) == 0


def test_command_takes_effect_exactly_t_link_after_issue(params):
    t_link = params.link.t_link
    link = CommandLink(t_link, params.link.f_link, initial=0.0, tol=TOL)
    t_issue = 5 * link.period  # issued on a link tick
    t = 0.0
    while t < t_issue - TOL:
        assert link.update(t) == 0.0
        t += DT
    link.set_command(1.0)
    assert link.update(t_issue) == 0.0
    assert link.update(t_issue + t_link - DT) == 0.0
    assert link.update(t_issue + t_link) == 1.0


def test_link_zero_order_hold(params):
    link = CommandLink(0.0, params.link.f_link, initial=0.0, tol=TOL)
    link.update(0.0)
    link.set_command(1.0)  # issued between ticks: waits for the next tick
    assert link.update(link.period - DT) == 0.0
    assert link.update(link.period) == 1.0
