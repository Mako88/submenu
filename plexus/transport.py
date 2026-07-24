"""Transport: the seam where a local run becomes a distributed one.

Nothing in the model may require globally synchronised state. That is the whole
design discipline, and this module is where it is enforced: a column talks to
the rest of the world only through this interface, and the interface offers
exactly two things -- delayed access to other units' past emissions, and a
low-bandwidth broadcast channel for neuromodulatory signals.

Neither operation is a barrier. A column never waits for another column's
current state; it reads history through a conduction delay. That is precisely
what a real axon does, and it is why hundreds of milliseconds of latency are
survivable.

``LocalTransport`` runs everything in one process. A future ``NetworkTransport``
implements the same three methods over QUIC or UDP; because reads are indexed
by emission time, the arithmetic is identical either way.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .events import EventBuffer


class Transport(ABC):
    """Abstract event substrate."""

    n_sources: int
    depth: int

    @abstractmethod
    def publish(self, t: int, values: np.ndarray) -> None:
        """Announce the activity emitted by locally-owned units at time ``t``."""

    @abstractmethod
    def gather(self, t: int, sources: np.ndarray, delays: np.ndarray) -> np.ndarray:
        """Read what ``sources`` emitted ``delays`` steps before ``t``."""

    @abstractmethod
    def broadcast(self, t: int, modulator: np.ndarray) -> None:
        """Publish the neuromodulatory vector for time ``t``."""

    @abstractmethod
    def modulator(self, t: int) -> np.ndarray:
        """Read the most recent modulatory vector visible at time ``t``."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all buffered state."""


class LocalTransport(Transport):
    """Single-process transport. Delays are honoured; latency is zero.

    ``modulator_lag`` deliberately delays the broadcast signal even in local
    runs. It defaults to zero, but setting it is the cheapest way to confirm
    that learning still works when the modulator arrives late -- which is the
    property the distributed version depends on.
    """

    def __init__(
        self,
        n_sources: int,
        depth: int,
        modulator_dim: int,
        modulator_lag: int = 0,
    ):
        self.n_sources = n_sources
        self.depth = depth
        self.modulator_dim = modulator_dim
        self.modulator_lag = modulator_lag
        self._events = EventBuffer(n_sources, depth)
        self._mod = EventBuffer(modulator_dim, max(depth, modulator_lag + 1))
        self._mod_idx = np.arange(modulator_dim)
        self._mod_lag = np.full(modulator_dim, modulator_lag)

    # Written densely every step and zeroed on reset, so consumers may use the
    # unchecked precomputed-index path.
    dense = True

    def publish(self, t: int, values: np.ndarray) -> None:
        self._events.write(t, values)

    def publish_slice(self, t: int, start: int, values: np.ndarray) -> None:
        self._events.write_slice(t, start, values)

    def begin(self, t: int) -> None:
        self._events.begin(t)

    def gather(self, t: int, sources: np.ndarray, delays: np.ndarray) -> np.ndarray:
        return self._events.gather(t, sources, delays)

    def take(self, flat_index: np.ndarray) -> np.ndarray:
        return self._events.take(flat_index)

    def broadcast(self, t: int, modulator: np.ndarray) -> None:
        self._mod.write(t, modulator)

    def modulator(self, t: int) -> np.ndarray:
        return self._mod.gather(t, self._mod_idx, self._mod_lag)

    def reset(self) -> None:
        self._events.reset()
        self._mod.reset()
