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
    # The underlying cue pattern, when the task has one. Never shown to the
    # model -- it exists so diagnostics can ask questions the label cannot
    # answer, such as whether two episodes the model must give the *same*
    # answer to were nonetheless different events (++ and -- are both parity 0).
    bits: tuple[int, ...] | None = None


class DelayedParity:
    """Hold N cues across gaps, then report their parity.

    Each cue appears as one of two channel groups (a "+" group or a "-" group),
    separated in time, and after a further delay the network must report the
    parity of the signs.

    Every cue is always present -- only its *identity* varies -- so total
    activity is identical across all conditions. Nothing but held, combined
    memory of events separated in time can solve it, and parity is the maximally
    nonlinear combination: no linear function of the individual memories works,
    and flipping any single cue flips the answer.

    Difficulty scales sharply with ``n_cues``. Two cues (delayed XOR) turn out
    to be roughly 0.85 decodable from a *random* column, leaving a learning rule
    almost nothing to improve. Three cues is where a frozen reservoir starts to
    run out, which is the regime a plasticity rule has to earn its place in.
    """

    def __init__(
        self,
        n_cues: int = 2,
        group_size: int = 4,
        n_distractor: int = 8,
        length: int | None = None,
        first_cue: int = 40,
        cue_spacing: int = 100,
        cue_duration: int = 25,
        response_gap: int = 140,
        go_lead: int = 15,
        go_duration: int = 45,
        jitter: int = 10,
        noise_rate: float = 0.02,
        cue_strength: float = 1.0,
        channel_seed: int | None = None,
    ):
        # A permutation of the input channels. Two tasks that differ only in
        # this are the *same computation over a different sensory mapping* --
        # identical difficulty, identical statistics, disjoint in what they ask
        # the column's fixed wiring to do. That makes a task stream out of one
        # benchmark without introducing a difficulty confound, which is what
        # continual learning needs and what a stream of unrelated tasks cannot
        # give: if task B is simply harder, forgetting and difficulty are not
        # separable in the result.
        self.channel_seed = channel_seed
        self.n_cues = n_cues
        self.group_size = group_size
        self.n_distractor = n_distractor
        # Two groups per cue plus a go cue. The go cue is identical in every
        # condition, so it carries no label information -- it exists because
        # memory held in synaptic efficacy is silent by construction and has to
        # be probed to be read. Standard delayed-response paradigm, not a hint.
        self.n_inputs = (2 * n_cues + 1) * group_size + n_distractor
        self.go_group = 2 * n_cues
        self.n_classes = 2
        self.cue_times = [first_cue + i * cue_spacing for i in range(n_cues)]
        self.response_start = self.cue_times[-1] + cue_duration + response_gap
        self.length = length if length is not None else self.response_start + 120
        self.cue_duration = cue_duration
        self.go_lead = go_lead
        self.go_duration = go_duration
        self.jitter = jitter
        self.noise_rate = noise_rate
        self.cue_strength = cue_strength
        self.channel_perm = (
            None if channel_seed is None
            else np.random.default_rng(channel_seed).permutation(self.n_inputs)
        )

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

        bits = [int(rng.integers(0, 2)) for _ in range(self.n_cues)]
        bursts = [
            (
                self.cue_times[i] + int(rng.integers(-self.jitter, self.jitter + 1)),
                2 * i + bits[i],
                self.cue_duration,
            )
            for i in range(self.n_cues)
        ]
        bursts.append((self.response_start - self.go_lead, self.go_group, self.go_duration))

        for t0, group, dur in bursts:
            sl = self._group(group)
            seg = x[t0 : t0 + dur, sl]
            seg += self.cue_strength * (
                rng.random(seg.shape) < 0.5
            ).astype(np.float32) * rng.uniform(0.8, 1.2, size=seg.shape).astype(np.float32)

        response = np.zeros(T, dtype=bool)
        response[self.response_start :] = True
        label = 0
        for b in bits:
            label ^= b
        if self.channel_perm is not None:
            # Applied last, so everything above -- constant total energy across
            # classes, every class using the same channels equally often -- is
            # untouched. A permutation cannot change any of those properties,
            # only which physical channel carries which role.
            x = x[:, self.channel_perm]
        return Episode(inputs=x, label=label, response=response, bits=tuple(bits))


class DelayedXOR(DelayedParity):
    """Two-cue parity: hold two cues across a gap, then report their XOR."""

    def __init__(self, **kwargs):
        kwargs.setdefault("n_cues", 2)
        kwargs.setdefault("cue_spacing", 100)
        kwargs.setdefault("response_gap", 115)
        super().__init__(**kwargs)


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
