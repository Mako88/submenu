"""Assembling a column, a readout, and a transport into a trainable model.

The per-step loop is the honest statement of the architecture:

    column.step(t)          -- reads only delayed history
    readout.observe(...)    -- filters local activity
    transport.broadcast(m)  -- publishes a small vector
    column.apply_modulator  -- commits local weight changes

There is no backward pass and no point at which any component waits for a
global state to be assembled. Every line would still be correct if the column
and the readout lived on different continents; the only thing that would change
is how stale ``m`` is when it arrives, and the eligibility traces already
absorb that.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .column import Column, ColumnConfig
from .readout import LinearReadout, softmax
from .tasks import Episode
from .transport import LocalTransport


@dataclass
class TrainReport:
    episode: int
    loss: float
    accuracy: float
    sparsity: float
    mean_weight: float


class Plexus:
    """A single-column model with a linear readout.

    ``modulator_lag`` injects artificial staleness into the broadcast even in a
    local run. It exists to be turned up: if learning survives a lag of a few
    hundred steps, the model will survive a wide-area network.
    """

    def __init__(
        self,
        n_inputs: int,
        n_classes: int,
        column: ColumnConfig | None = None,
        readout_tau: float = 60.0,
        readout_lr: float = 0.5,
        feedback_mode: str = "symmetric",
        readout_rule: str = "delta",
        readout_block: int | None = None,
        modulator_lag: int = 0,
        drain_steps: int | None = None,
        answer_steps: int = 50,
        seed: int = 0,
    ):
        cfg = column or ColumnConfig()
        cfg = replace(cfg, n_external=n_inputs, modulator_dim=n_classes, seed=seed)
        self.cfg = cfg
        self.n_classes = n_classes
        self.answer_steps = answer_steps

        depth = cfg.delay_max + 2
        self.transport = LocalTransport(
            n_sources=n_inputs + cfg.n_neurons,
            depth=depth,
            modulator_dim=n_classes,
            modulator_lag=modulator_lag,
        )
        self.column = Column(cfg, self.transport)
        self.readout = LinearReadout(
            n_inputs=cfg.n_neurons,
            n_outputs=n_classes,
            tau=readout_tau,
            lr=readout_lr,
            feedback_mode=feedback_mode,
            rule=readout_rule,
            rls_block=readout_block,
            seed=seed + 1,
        )
        # How many silent steps end each episode. Defaults to the lag, which is
        # the minimum needed for an in-flight modulator to land -- but it must
        # be held CONSTANT when comparing lag settings, or the comparison is
        # confounded. The tail is not inert: homeostasis keeps adapting through
        # it on near-zero activity, so a longer tail lowers thresholds, and
        # `modulator_lag` ends up changing the column's whole operating point
        # rather than only when feedback arrives.
        #
        # Measured, on a *frozen* column that never reads the modulator at all:
        # lag 0 gives threshold 0.295, sparsity 0.0289, engagement 0.111; lag
        # 200 gives 0.231, 0.0189, 0.0926. End-to-end that was a -0.062 accuracy
        # difference on 17/20 paired seeds at p = 0.0011, in a condition where
        # the lag is supposed to be unobservable.
        self.drain_steps = modulator_lag if drain_steps is None else int(drain_steps)
        if self.drain_steps < modulator_lag:
            raise ValueError(
                f"drain_steps ({self.drain_steps}) < modulator_lag ({modulator_lag}): "
                "the episode would end before the modulator arrived, silently "
                "discarding every learning signal"
            )
        self._t = 0
        self._zero_mod = np.zeros(n_classes, dtype=np.float32)
        self._zero_input = np.zeros(n_inputs, dtype=np.float32)

    # -----------------------------------------------------------------
    def run_episode(self, ep: Episode, learn: bool = True) -> tuple[float, bool]:
        """Run one trial. Returns (mean loss over response window, correct)."""
        self.column.learning = learn
        self.readout.learning = learn
        self.column.reset_state()
        self.readout.reset()
        self.transport.reset()

        fb = self.readout.feedback_matrix()
        if fb is not None:
            self.column.feedback = fb

        loss = 0.0
        votes = np.zeros(self.n_classes, dtype=np.float64)
        answered = 0
        err = None

        for k in range(ep.inputs.shape[0]):
            t = self._t
            self._t += 1

            activity = self.column.step(t, ep.inputs[k])
            self.readout.observe(activity)

            # The response window splits into answer-then-feedback. Scoring and
            # learning must not overlap: if the readout updates while votes are
            # still being counted, it fits the current episode's label within a
            # few steps and the remaining votes are trivially correct. That
            # reads as perfect training accuracy on a model that has learned
            # nothing. Answering first and being told the answer afterwards is
            # also the honest order of events for a system learning from
            # feedback.
            if ep.response[k]:
                answered += 1
                if answered <= self.answer_steps:
                    # Answer phase: let evidence accumulate in the trace, then
                    # commit once at the end of it. The decision vector is the
                    # trace itself, not an average over the window -- the trace
                    # already applies ~60ms of exponential weighting, and
                    # averaging on top of that dilutes a signal concentrated
                    # just after the go cue (0.69 vs 0.85 decodable).
                    if answered == self.answer_steps:
                        logits, z = self.readout.decide(self.readout.trace)
                        votes = softmax(logits)
                        err, loss = self.readout.error(logits, ep.label)
                        if learn:
                            self.readout.update(err, z, ep.label)
                        # One decision, one feedback signal, released while the
                        # eligibility trace still holds what produced that
                        # decision. Sustaining the modulator across the whole
                        # feedback phase applied the same error ~70 times as the
                        # trace accumulated post-decision activity -- both far
                        # too strong and steadily more misdirected.
                        self.transport.broadcast(
                            t, self.readout.modulator(err) if learn else self._zero_mod
                        )
                    else:
                        self.transport.broadcast(t, self._zero_mod)
                else:
                    self.transport.broadcast(t, self._zero_mod)
            else:
                self.transport.broadcast(t, self._zero_mod)

            self.column.apply_modulator(t)

        # Drain the in-flight modulator. A signal broadcast during the response
        # window arrives `modulator_lag` steps later, which may be after the
        # inputs have run out; without this tail an episodic harness would
        # silently discard every delayed learning signal and the model would
        # appear not to learn at all. A continuously running system has no such
        # boundary -- the tail is what makes the episodic simulation faithful
        # to it, not a workaround for a real limitation.
        #
        # Its length is `drain_steps`, not the lag, so that two lag settings can
        # be compared without also comparing two different amounts of silent
        # homeostatic adaptation. See the constructor for what that cost.
        for _ in range(self.drain_steps):
            t = self._t
            self._t += 1
            self.column.step(t, self._zero_input)
            self.transport.broadcast(t, self._zero_mod)
            self.column.apply_modulator(t)

        correct = bool(np.argmax(votes) == ep.label)
        return loss, correct

    # -----------------------------------------------------------------
    def train(
        self,
        task,
        episodes: int,
        rng: np.random.Generator | None = None,
        report_every: int = 50,
        on_report=None,
    ) -> list[TrainReport]:
        rng = rng or np.random.default_rng(0)
        reports: list[TrainReport] = []
        window_loss: list[float] = []
        window_acc: list[bool] = []

        for i in range(1, episodes + 1):
            ep = task.episode(rng)
            loss, correct = self.run_episode(ep, learn=True)
            window_loss.append(loss)
            window_acc.append(correct)

            if i % report_every == 0:
                rep = TrainReport(
                    episode=i,
                    loss=float(np.mean(window_loss)),
                    accuracy=float(np.mean(window_acc)),
                    sparsity=self.column.sparsity,
                    mean_weight=float(self.column.W.mean()),
                )
                reports.append(rep)
                window_loss.clear()
                window_acc.clear()
                if on_report:
                    on_report(rep)
        return reports

    def evaluate(self, task, episodes: int, rng: np.random.Generator | None = None) -> float:
        rng = rng or np.random.default_rng(12345)
        correct = 0
        for _ in range(episodes):
            _, ok = self.run_episode(task.episode(rng), learn=False)
            correct += int(ok)
        return correct / episodes
