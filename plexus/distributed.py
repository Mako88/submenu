"""Splitting a model across several columns, each of which could be a machine.

This is the claim the whole design exists to support: because no operation
needs globally synchronised state, the model can be cut into pieces that only
ever exchange sparse events through a conduction delay -- and a piece that
lives 80ms away is just a longer axon.

The per-step loop makes the ownership explicit:

    transport.begin(t)              -- open the timestep
    publish external drive          -- nobody's column owns the sensors
    for each column: publish(t)     -- each writes only the slice it owns
    for each column: step(t)        -- each reads history, never the present

Only the middle two lines would cross a network. Every column publishes before
any column reads, and since every synapse carries a delay of at least one step,
a column never observes another's current state -- so the order in which the
columns are stepped cannot change the result. That is the property that makes
the loop safe to run on separate machines rather than merely tidy.

What distribution costs the model is one number: ``peer_delay``, the conduction
delay on synapses reaching a neuron another column owns. Turning it up is the
simulation of moving those columns further apart.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .column import Column, ColumnConfig
from .readout import LinearReadout, softmax
from .tasks import Episode
from .transport import LocalTransport


class DistributedPlexus:
    """A model split across ``n_columns`` columns sharing one event substrate.

    ``peer_delay`` is expressed in steps, and one step is 1ms, so it reads
    directly as network latency: 3 is a rack, 30 is a region, 150 is
    intercontinental. Cortex itself runs 0.5-30ms conduction delays, so the
    lower end of that range is not a compromise -- it is the regime the
    architecture was designed around.
    """

    def __init__(
        self,
        n_inputs: int,
        n_classes: int,
        n_columns: int = 4,
        column: ColumnConfig | None = None,
        peer_delay: int = 30,
        peer_delay_min: int | None = None,
        readout_tau: float = 60.0,
        readout_lr: float = 0.5,
        feedback_mode: str = "symmetric",
        readout_rule: str = "delta",
        modulator_lag: int = 0,
        answer_steps: int = 50,
        seed: int = 0,
    ):
        base = column or ColumnConfig()
        per_column = base.n_neurons
        total_neurons = per_column * n_columns
        self.n_columns = n_columns
        self.n_classes = n_classes
        self.answer_steps = answer_steps
        self.peer_delay = peer_delay

        depth = max(base.delay_max, peer_delay) + 2
        self.transport = LocalTransport(
            n_sources=n_inputs + total_neurons,
            depth=depth,
            modulator_dim=n_classes,
            modulator_lag=modulator_lag,
        )

        self.columns = [
            Column(
                replace(
                    base,
                    n_external=n_inputs,
                    modulator_dim=n_classes,
                    seed=seed * 1000 + i,
                    source_offset=n_inputs + i * per_column,
                    peer_span=total_neurons,
                    peer_delay_min=peer_delay_min if peer_delay_min is not None else peer_delay,
                    peer_delay_max=peer_delay,
                ),
                self.transport,
            )
            for i in range(n_columns)
        ]
        self.readout = LinearReadout(
            n_inputs=total_neurons,
            n_outputs=n_classes,
            tau=readout_tau,
            lr=readout_lr,
            feedback_mode=feedback_mode,
            rule=readout_rule,
            seed=seed + 1,
        )
        self.n_inputs = n_inputs
        self.per_column = per_column
        self._t = 0
        self._zero_mod = np.zeros(n_classes, dtype=np.float32)
        self._zero_input = np.zeros(n_inputs, dtype=np.float32)

    # -----------------------------------------------------------------
    @property
    def activity(self) -> np.ndarray:
        return np.concatenate([c.out for c in self.columns])

    def _distribute_feedback(self) -> None:
        """Hand each column the slice of the feedback matrix addressing it.

        The full matrix is (total_neurons, n_classes); a column only ever needs
        the rows for neurons it owns, so this is a slice, not a broadcast of
        everything to everyone.
        """
        fb = self.readout.feedback_matrix()
        if fb is None:
            return
        for i, col in enumerate(self.columns):
            col.feedback = fb[i * self.per_column : (i + 1) * self.per_column].copy()

    def _tick(self, t: int, external: np.ndarray) -> None:
        self.transport.begin(t)
        self.transport.publish_slice(t, 0, external)
        for col in self.columns:
            col.publish(t)
        for col in self.columns:
            col.step(t)

    # -----------------------------------------------------------------
    def run_episode(self, ep: Episode, learn: bool = True) -> tuple[float, bool]:
        for col in self.columns:
            col.learning = learn
            col.reset_state()
        self.readout.learning = learn
        self.readout.reset()
        self.transport.reset()
        self._distribute_feedback()

        loss = 0.0
        votes = np.zeros(self.n_classes, dtype=np.float64)
        answered = 0
        err = None

        for k in range(ep.inputs.shape[0]):
            t = self._t
            self._t += 1
            self._tick(t, ep.inputs[k])
            self.readout.observe(self.activity)

            if ep.response[k]:
                answered += 1
                if answered <= self.answer_steps:
                    if answered == self.answer_steps:
                        logits, z = self.readout.decide(self.readout.trace)
                        votes = softmax(logits)
                        err, loss = self.readout.error(logits, ep.label)
                        if learn:
                            self.readout.update(err, z, ep.label)
                        self.transport.broadcast(
                            t, self.readout.modulator(err) if learn else self._zero_mod
                        )
                    else:
                        self.transport.broadcast(t, self._zero_mod)
                else:
                    self.transport.broadcast(t, self._zero_mod)
            else:
                self.transport.broadcast(t, self._zero_mod)

            for col in self.columns:
                col.apply_modulator(t)

        for _ in range(self.transport.modulator_lag):
            t = self._t
            self._t += 1
            self._tick(t, self._zero_input)
            self.transport.broadcast(t, self._zero_mod)
            for col in self.columns:
                col.apply_modulator(t)

        return loss, bool(np.argmax(votes) == ep.label)

    # -----------------------------------------------------------------
    def train(self, task, episodes: int, rng: np.random.Generator | None = None) -> None:
        rng = rng or np.random.default_rng(0)
        for _ in range(episodes):
            self.run_episode(task.episode(rng), learn=True)

    def evaluate(self, task, episodes: int, rng: np.random.Generator | None = None) -> float:
        rng = rng or np.random.default_rng(12345)
        return sum(
            self.run_episode(task.episode(rng), learn=False)[1] for _ in range(episodes)
        ) / episodes

    @property
    def sparsity(self) -> float:
        return float(np.mean([c.sparsity for c in self.columns]))
