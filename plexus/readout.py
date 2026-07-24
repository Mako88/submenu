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
        lr: float = 5e-3,
        dt: float = 1.0,
        feedback_mode: str = "symmetric",
        seed: int = 0,
    ):
        if feedback_mode not in ("symmetric", "dfa"):
            raise ValueError("feedback_mode must be 'symmetric' or 'dfa'")
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
        # Column activity is sparse by design, so the filtered trace has a tiny
        # magnitude that depends on the target firing rate. Rather than bake
        # that into the weight scale, track its RMS and normalise. This is the
        # one place in the model where anything is pooled across neurons, and
        # it is also the one place where pooling is free: the readout already
        # has to see the whole population.
        self.scale = np.float32(1e-3)
        self.decay_scale = np.float32(0.999)
        self.learning = True

    def observe(self, activity: np.ndarray) -> np.ndarray:
        """Filter incoming column activity and return current logits."""
        self.trace = (self.decay * self.trace + self.gain * activity).astype(np.float32)
        rms = float(np.sqrt(np.mean(self.trace**2)))
        if self.learning and rms > 0.0:
            self.scale = np.float32(
                self.decay_scale * self.scale + (1.0 - self.decay_scale) * rms
            )
        return self.logits

    @property
    def normalized(self) -> np.ndarray:
        return self.trace / max(float(self.scale), 1e-8)

    @property
    def logits(self) -> np.ndarray:
        return self.W @ self.normalized + self.b

    def error(self, target: int) -> tuple[np.ndarray, float]:
        """Cross-entropy error signal and loss for the current filtered state."""
        p = softmax(self.logits)
        onehot = np.zeros(self.n_outputs, dtype=np.float32)
        onehot[target] = 1.0
        loss = float(-np.log(max(p[target], 1e-9)))
        return (p - onehot).astype(np.float32), loss

    def update(self, err: np.ndarray) -> None:
        """Delta rule on the readout itself -- local, no backward pass."""
        if not self.learning:
            return
        self.W -= self.lr * np.outer(err, self.normalized)
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
