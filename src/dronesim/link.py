"""Command link: zero-order hold at f_link followed by a fixed latency T_link (SPEC Section 3.5).

At each link tick the link samples the latest controller command. That sample
reaches the plant T_link later and holds until the next sample arrives.
Commands stay in physical units.
"""

from __future__ import annotations

from typing import Any

from .sensor import DelayQueue


class CommandLink:
    def __init__(self, latency: float, rate: float, initial: Any, tol: float):
        self.period = 1.0 / rate
        self.tol = tol
        self._queue = DelayQueue(latency, tol)
        self._latest = initial
        self._applied = initial
        self._n_tx = 0

    def set_command(self, command: Any) -> None:
        """Record the controller's newest command; the next link tick transmits it."""
        self._latest = command

    def update(self, t: float) -> Any:
        """Transmit any link ticks due by time t and return the command applied at the plant."""
        while self._n_tx * self.period <= t + self.tol:
            self._queue.push(self._n_tx * self.period, self._latest)
            self._n_tx += 1
        for command in self._queue.pop_ready(t):
            self._applied = command
        return self._applied
