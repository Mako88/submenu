"""Readout and modulator generation.

The readout is where error becomes a broadcastable signal. It is deliberately
tiny: a linear map on a low-pass filter of column activity. Everything
expensive stays inside the column, learned locally; the only thing that ever
crosses the network is a vector the width of the output.

That asymmetry is the point. A backward pass moves gradient tensors
proportional to the parameter count. This moves a handful of floats, and the
eligibility traces inside each column mean those floats may arrive late.
Bandwidth scales with what the task outputs, not with how big the model is.
"""

from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class LinearReadout:
    """Linear decoder over filtered column activity, trained by a delta rule.

    ``feedback_mode`` selects how credit reaches the column:

    ``symmetric``
        Neurons read the modulator through the readout's own weights. Needs the
        readout matrix broadcast alongside the error, which is cheap for a small
        output space and still carries no synchronisation barrier.

    ``dfa``
        Neurons keep their random projection (direct feedback alignment).
        Nothing but the error vector is ever transmitted, and the column learns
        to align with whatever fixed projection it was born with. Strictly
        cheaper, usually slower to converge.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        tau: float = 60.0,
        lr: float = 0.5,
        dt: float = 1.0,
        feedback_mode: str = "symmetric",
        rule: str = "delta",
        rls_alpha: float = 1.0,
        rls_forget: float = 0.999,
        rls_block: int | None = None,
        seed: int = 0,
    ):
        if feedback_mode not in ("symmetric", "dfa"):
            raise ValueError("feedback_mode must be 'symmetric' or 'dfa'")
        if rule not in ("delta", "rls", "rls_diag", "rls_block"):
            raise ValueError("rule must be delta, rls, rls_diag or rls_block")
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0.0, 1.0 / np.sqrt(n_inputs), size=(n_outputs, n_inputs)).astype(
            np.float32
        )
        self.b = np.zeros(n_outputs, dtype=np.float32)
        self.n_outputs = n_outputs
        self.lr = lr
        self.decay = np.float32(np.exp(-dt / tau))
        self.gain = np.float32(1.0 - self.decay)  # unit DC gain, as in the column
        self.feedback_mode = feedback_mode
        self.trace = np.zeros(n_inputs, dtype=np.float32)
        # Per-neuron running mean and variance, used to standardise the trace.
        #
        # Centering is not cosmetic. Column activity is sparse and strictly
        # non-negative, so the population mean is by far the largest direction
        # in the state and it carries no label information at all. A delta rule
        # on uncentered input spends its whole budget on that direction and
        # crawls along the low-variance discriminative ones -- the readout sat
        # at chance while an offline decoder on *standardised* copies of the
        # very same states reached 0.86.
        #
        # Each dimension is normalised using only its own statistics, so this
        # stays a per-neuron operation with nothing pooled across the
        # population and nothing to synchronise.
        # Both start at zero and are bias-corrected on read. Seeding variance
        # at 1.0 instead looks harmless and is not: column traces are tiny, so
        # the leftover initialisation dominated the true variance for thousands
        # of episodes, shrinking standardised features to a standard deviation
        # of 0.003 and leaving the decoder at chance on separable data.
        self.mean = np.zeros(n_inputs, dtype=np.float32)
        self.var = np.zeros(n_inputs, dtype=np.float32)
        self.n_decisions = 0
        # Statistics now advance once per decision, not once per timestep,
        # so the window is measured in episodes.
        self.decay_stats = np.float32(0.99)
        self.learning = True

        # Recursive least squares (FORCE, Sussillo & Abbott 2009). A delta rule
        # takes a fixed step along the current sample and has to see a direction
        # many times to move along it; RLS carries the inverse correlation
        # matrix of the inputs, so one sample updates every direction by exactly
        # as much as the accumulated evidence warrants. On a reservoir readout
        # that is the difference between converging in hundreds of episodes and
        # in tens.
        #
        # The cost is honest: P is (n+1)x(n+1) and pools across the population,
        # which is a real exception to the locality rule. It is confined to the
        # readout -- the one place that already sees every neuron -- and it does
        # not cross the network, but it does not scale to a large column and it
        # should not be pretended otherwise.
        # Three variants, trading decorrelation against how much state has to be
        # pooled. The design rule forbids *globally synchronised* state, not all
        # aggregation, so where the matrix lives matters more than that it
        # exists:
        #   "rls"       -- one (n+1)x(n+1) matrix over every neuron. Pools across
        #                  the whole population, so in a split model it would
        #                  need state from every machine. This is the variant
        #                  that genuinely breaks the rule.
        #   "rls_block" -- one small matrix per column. A column is a machine, so
        #                  nothing crosses the network and nothing synchronises.
        #                  Keeps decorrelation within a column, drops it between.
        #   "rls_diag"  -- one scalar per input. Fully local to a single neuron,
        #                  no cross-neuron state at all. Keeps the per-dimension
        #                  adaptive step size, drops decorrelation entirely.
        # Which of those two effects carried the +0.032 is an empirical question,
        # and the whole point of having all three.
        self.rule = rule
        self.rls_forget = float(rls_forget)
        self.rls_block = int(rls_block) if rls_block else n_inputs
        if rule == "rls":
            self.P = np.eye(n_inputs + 1, dtype=np.float64) / float(rls_alpha)
        elif rule == "rls_block":
            edges = list(range(0, n_inputs, self.rls_block)) + [n_inputs]
            self.blocks = [slice(a, b) for a, b in zip(edges, edges[1:])]
            self.Pb = [
                np.eye(sl.stop - sl.start, dtype=np.float64) / float(rls_alpha)
                for sl in self.blocks
            ]
            self.pbias = 1.0 / float(rls_alpha)
        elif rule == "rls_diag":
            self.pdiag = np.full(n_inputs + 1, 1.0 / float(rls_alpha), dtype=np.float64)

    def observe(self, activity: np.ndarray) -> np.ndarray:
        """Filter incoming column activity. Returns the current trace."""
        self.trace = (self.decay * self.trace + self.gain * activity).astype(np.float32)
        return self.trace

    def decide(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Standardise a decision vector and classify it.

        The statistics are gathered over *decision vectors* -- one per episode,
        the same objects being classified -- not over per-timestep traces.
        Normalising by the wrong quantity is a silent killer here: per-timestep
        variance is dominated by within-episode dynamics, which averaging over
        the answer window removes, so dividing by it shrank the features to a
        standard deviation of 0.009 and did so unevenly across dimensions. The
        readout sat at chance on states an offline decoder read at 0.85.
        """
        state = state.astype(np.float32)
        d = self.decay_stats
        if self.learning:
            self.n_decisions += 1
            self.mean = (d * self.mean + (1.0 - d) * state).astype(np.float32)
            bias = 1.0 - d**self.n_decisions
            centered = state - self.mean / bias
            self.var = (d * self.var + (1.0 - d) * centered**2).astype(np.float32)
        bias = 1.0 - d ** max(self.n_decisions, 1)
        mean = self.mean / bias
        std = np.sqrt(self.var / bias)
        z = ((state - mean) / (std + 1e-12)).astype(np.float32)
        return (self.W @ z + self.b).astype(np.float32), z

    def error(self, logits: np.ndarray, target: int) -> tuple[np.ndarray, float]:
        """Cross-entropy error signal and loss for a given set of logits."""
        p = softmax(logits)
        onehot = np.zeros(self.n_outputs, dtype=np.float32)
        onehot[target] = 1.0
        loss = float(-np.log(max(p[target], 1e-9)))
        return (p - onehot).astype(np.float32), loss

    def update(self, err: np.ndarray, state: np.ndarray, target: int | None = None) -> None:
        """Delta rule against the state the answer was actually based on.

        The readout gets an eligibility trace of its own, for the same reason
        the column has one. Feedback arrives after the answer, by which point
        the network has moved on -- the go cue has ended and activity is
        decaying. Updating against the *current* state would train the decoder
        on a different distribution from the one it is scored on, which is
        enough on its own to hold accuracy at chance.
        """
        if not self.learning:
            return

        if self.rule.startswith("rls"):
            if target is None:
                raise ValueError("the rls rules need the target class")
            # Proper least squares against a one-hot target, with the bias
            # folded in as a constant input. The softmax error is left alone and
            # still feeds the column's modulator, so switching rule changes how
            # fast the readout learns and nothing about what the column is told.
            lam = self.rls_forget
            zf = state.astype(np.float64)
            onehot = np.zeros(self.n_outputs, dtype=np.float64)
            onehot[target] = 1.0
            residual = (self.W @ state + self.b).astype(np.float64) - onehot

            if self.rule == "rls":
                z = np.append(zf, 1.0)
                Pz = self.P @ z
                k = Pz / (lam + float(z @ Pz))
                self.W -= np.outer(residual, k[:-1]).astype(np.float32)
                self.b -= (residual * k[-1]).astype(np.float32)
                # Forgetting keeps P adapting: in the plastic condition the
                # column's representation drifts under it, so a fixed
                # correlation estimate would describe a network that no longer
                # exists.
                self.P = (self.P - np.outer(k, Pz)) / lam

            elif self.rule == "rls_block":
                k = np.empty_like(zf)
                for sl, P in zip(self.blocks, self.Pb):
                    zb = zf[sl]
                    Pz = P @ zb
                    kb = Pz / (lam + float(zb @ Pz))
                    k[sl] = kb
                    P -= np.outer(kb, Pz)
                    P /= lam
                self.W -= np.outer(residual, k).astype(np.float32)
                kb_bias = self.pbias / (lam + self.pbias)
                self.b -= (residual * kb_bias).astype(np.float32)
                self.pbias = (self.pbias - kb_bias * self.pbias) / lam

            else:  # rls_diag -- one scalar per input, nothing shared
                z = np.append(zf, 1.0)
                denom = lam + self.pdiag * z * z
                k = self.pdiag * z / denom
                self.W -= np.outer(residual, k[:-1]).astype(np.float32)
                self.b -= (residual * k[-1]).astype(np.float32)
                self.pdiag = (self.pdiag - k * z * self.pdiag) / lam
            return

        # Normalised LMS: dividing by the state's own energy makes the step
        # size mean "move the logit this fraction of the way" regardless of how
        # many neurons there are or how active they were. Without it the
        # effective rate scales with ||state||^2 (~192 here), and any fixed lr
        # is either inert or wildly divergent -- at lr=0.05 the readout was
        # confidently wrong, with cross-entropy near 5 on a two-class problem.
        energy = float(state @ state) + 1.0
        self.W -= (self.lr / energy) * np.outer(err, state)
        self.b -= self.lr * err

    def modulator(self, err: np.ndarray) -> np.ndarray:
        """The vector broadcast to every column. Sign flipped so that a
        positive component means 'strengthen what you just did'."""
        return (-err).astype(np.float32)

    def feedback_matrix(self) -> np.ndarray | None:
        """Per-neuron modulator projection, or None to keep the column's own."""
        return self.W.T.copy() if self.feedback_mode == "symmetric" else None

    def reset(self) -> None:
        self.trace.fill(0.0)
