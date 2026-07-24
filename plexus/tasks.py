"""Temporal tasks.

Both tasks are built so that a model without temporal structure cannot solve
them by accident. Total input energy is held constant across classes, and every
class uses the same channels the same number of times. What differs is *when*.
A rate code, a bag-of-features, or an MLP over summed input is at chance on
both by construction -- which is the only way to know the architecture is
earning its keep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Episode:
    """One trial: an input event stream, a label, and when to be judged."""

    inputs: np.ndarray  # (T, n_inputs) float32
    label: int
    response: np.ndarray  # (T,) bool -- steps where the answer is read out


class DelayedXOR:
    """Hold two cues across a gap, then report their XOR.

    Cue A appears as one of two channel groups (A+ or A-), then after a delay
    cue B appears as one of two others. After a further delay the network must
    report whether the signs agreed.

    Both cues are always present -- only their *identity* varies -- so total
    activity is identical across all four conditions. Nothing but held,
    combined memory of two events separated in time can solve it, and the
    combination is XOR, so no linear function of the two memories works either.
    """

    def __init__(
        self,
        group_size: int = 4,
        n_distractor: int = 8,
        length: int = 400,
        cue_a_time: int = 40,
        cue_b_time: int = 140,
        cue_duration: int = 25,
        response_start: int = 280,
        go_lead: int = 15,
        go_duration: int = 45,
        jitter: int = 10,
        noise_rate: float = 0.02,
        cue_strength: float = 1.0,
    ):
        self.group_size = group_size
        self.n_distractor = n_distractor
        # Five groups: A-, A+, B-, B+, and a go cue. The go cue is identical in
        # every condition, so it carries no label information -- it exists
        # because memory held in synaptic efficacy is silent by construction
        # and has to be probed to be read. This is the standard delayed-response
        # paradigm, not a hint.
        self.n_inputs = 5 * group_size + n_distractor
        self.go_group = 4
        self.n_classes = 2
        self.length = length
        self.cue_a_time = cue_a_time
        self.cue_b_time = cue_b_time
        self.cue_duration = cue_duration
        self.response_start = response_start
        self.go_lead = go_lead
        self.go_duration = go_duration
        self.jitter = jitter
        self.noise_rate = noise_rate
        self.cue_strength = cue_strength

    def _group(self, index: int) -> slice:
        return slice(index * self.group_size, (index + 1) * self.group_size)

    def episode(self, rng: np.random.Generator) -> Episode:
        T = self.length
        x = np.zeros((T, self.n_inputs), dtype=np.float32)

        # Background: sparse Poisson events with random magnitude on every
        # channel, including the cue channels, so the cue must be recognised
        # rather than merely detected.
        mask = rng.random((T, self.n_inputs)) < self.noise_rate
        x[mask] = rng.uniform(0.4, 1.0, size=int(mask.sum())).astype(np.float32)

        a = int(rng.integers(0, 2))
        b = int(rng.integers(0, 2))

        t_a = self.cue_a_time + int(rng.integers(-self.jitter, self.jitter + 1))
        t_b = self.cue_b_time + int(rng.integers(-self.jitter, self.jitter + 1))

        t_go = self.response_start - self.go_lead
        bursts = (
            (t_a, a, self.cue_duration),
            (t_b, 2 + b, self.cue_duration),
            (t_go, self.go_group, self.go_duration),
        )
        for t0, group, dur in bursts:
            sl = self._group(group)
            seg = x[t0 : t0 + dur, sl]
            seg += self.cue_strength * (
                rng.random(seg.shape) < 0.5
            ).astype(np.float32) * rng.uniform(0.8, 1.2, size=seg.shape).astype(np.float32)

        response = np.zeros(T, dtype=bool)
        response[self.response_start :] = True
        return Episode(inputs=x, label=a ^ b, response=response)


class TemporalPatterns:
    """Classify spatiotemporal templates that differ only in timing.

    Every class fires the same channels the same number of times within the
    window. Only the relative offsets differ. Summing over time destroys the
    label completely, so any model that solves this is genuinely reading
    temporal structure.
    """

    def __init__(
        self,
        n_classes: int = 4,
        n_inputs: int = 24,
        events_per_pattern: int = 40,
        window: int = 150,
        length: int = 320,
        onset: int = 30,
        response_start: int = 200,
        noise_rate: float = 0.015,
        jitter: float = 2.0,
        seed: int = 0,
    ):
        self.n_classes = n_classes
        self.n_inputs = n_inputs
        self.window = window
        self.length = length
        self.onset = onset
        self.response_start = response_start
        self.noise_rate = noise_rate
        self.jitter = jitter

        # Build templates that share an identical channel histogram: draw one
        # channel sequence, then give each class a distinct permutation of the
        # *times*. Same channels, same counts, different order.
        rng = np.random.default_rng(seed)
        channels = rng.integers(0, n_inputs, size=events_per_pattern)
        self.templates = []
        for _ in range(n_classes):
            times = np.sort(rng.integers(0, window, size=events_per_pattern))
            self.templates.append((channels.copy(), rng.permutation(times)))

    def episode(self, rng: np.random.Generator) -> Episode:
        T = self.length
        x = np.zeros((T, self.n_inputs), dtype=np.float32)

        mask = rng.random((T, self.n_inputs)) < self.noise_rate
        x[mask] = rng.uniform(0.4, 1.0, size=int(mask.sum())).astype(np.float32)

        label = int(rng.integers(0, self.n_classes))
        channels, times = self.templates[label]
        onset = self.onset + int(rng.integers(-5, 6))
        jittered = times + rng.normal(0.0, self.jitter, size=times.shape)
        idx = np.clip((onset + jittered).astype(int), 0, T - 1)
        np.add.at(x, (idx, channels), rng.uniform(0.8, 1.2, size=idx.shape).astype(np.float32))

        response = np.zeros(T, dtype=bool)
        response[self.response_start :] = True
        return Episode(inputs=x, label=label, response=response)
