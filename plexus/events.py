"""Valued events: the communication primitive.

Biology's spike is all-or-none because the axon is a lossy analog wire and the
action potential is regenerative signalling that fights attenuation. We are on
reliable digital transport, so we drop that constraint and let each event carry
a payload.

The consequence is large. A rate code needs 10-50 spikes to convey one scalar;
a valued event conveys it in one. We keep everything that makes spiking cheap
-- sparsity, asynchrony, event-driven transmission -- and discard only the part
that was a workaround for wetware.

Events are indexed by the timestep they were *emitted*, never by the timestep
they arrived. That single decision is what makes network jitter benign: a
packet that shows up late still lands in the correct slot of history, so a
distributed run computes the same thing a local one does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Event:
    """A single valued event.

    Attributes:
        source: index of the emitting unit within the global source space.
        t: timestep at which the event was *emitted* (not received).
        value: the payload -- suprathreshold magnitude of the emission.
    """

    source: int
    t: int
    value: float


class EventBuffer:
    """Timestamp-indexed ring buffer of recent activity.

    Holds the last ``depth`` timesteps of values for ``n_sources`` units. Reads
    are performed by (source, delay) pairs, which is exactly how a synapse
    addresses its input: "give me what unit ``i`` emitted ``d`` steps ago".

    Because slots are addressed by emission time modulo ``depth``, a writer may
    fill a slot late (a delayed network packet) without corrupting the read
    path, provided it arrives within ``depth`` steps. ``depth`` is therefore
    both the maximum conduction delay and the tolerance for transport lateness.
    """

    def __init__(self, n_sources: int, depth: int):
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.n_sources = n_sources
        self.depth = depth
        self._buf = np.zeros((depth, n_sources), dtype=np.float32)
        # Emission time currently occupying each slot, so a stale slot that was
        # never written for this cycle reads as silence instead of as an echo
        # from `depth` steps ago.
        self._stamp = np.full(depth, -1, dtype=np.int64)

    def write(self, t: int, values: np.ndarray) -> None:
        """Store the activity vector emitted at time ``t``."""
        slot = t % self.depth
        self._buf[slot] = values
        self._stamp[slot] = t

    def begin(self, t: int) -> None:
        """Open the slot for emission time ``t``, clearing stale contents once.

        Multiple writers contribute to one timestep when the model is split
        across columns, so the slot is cleared by whoever touches it first
        rather than by each writer.
        """
        slot = t % self.depth
        if self._stamp[slot] != t:
            self._buf[slot] = 0.0
            self._stamp[slot] = t

    def write_slice(self, t: int, start: int, values: np.ndarray) -> None:
        """Write one writer's own span of the source space for time ``t``.

        This is what makes ownership explicit: a column publishes the slice it
        owns and never touches anyone else's. In a distributed run the same
        call becomes a send, and the slices are filled by different machines.
        """
        self.begin(t)
        self._buf[t % self.depth, start : start + len(values)] = values

    def scatter(self, t: int, sources: np.ndarray, values: np.ndarray) -> None:
        """Accumulate sparse events emitted at time ``t``.

        Used by network transports, which receive events piecemeal rather than
        as a dense vector. Clears the slot the first time it is touched for
        this ``t`` so late arrivals accumulate instead of overwriting.
        """
        slot = t % self.depth
        if self._stamp[slot] != t:
            self._buf[slot] = 0.0
            self._stamp[slot] = t
        np.add.at(self._buf[slot], sources, values)

    def gather(self, t: int, sources: np.ndarray, delays: np.ndarray) -> np.ndarray:
        """Read values emitted at ``t - delays`` by ``sources``.

        ``sources`` and ``delays`` broadcast against each other and the result
        takes their shape, so a whole synapse tensor is read in one call.
        """
        emitted_at = t - delays
        slots = emitted_at % self.depth
        vals = self._buf[slots, sources]
        # Mask reads of slots that do not actually hold the requested emission
        # time: before warm-up, or when a delayed packet never arrived.
        return np.where(self._stamp[slots] == emitted_at, vals, 0.0)

    def take(self, flat_index: np.ndarray) -> np.ndarray:
        """Unchecked gather by precomputed flat index.

        Skips the emission-time check, so it is only valid on a buffer that is
        written densely every step and zeroed on reset -- which is exactly the
        local case. Network transports must use :meth:`gather`, where the check
        is what distinguishes a genuinely silent unit from a dropped packet.
        """
        return np.take(self._buf.reshape(-1), flat_index)

    def reset(self) -> None:
        self._buf.fill(0.0)
        self._stamp.fill(-1)
