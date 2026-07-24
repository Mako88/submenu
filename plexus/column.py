"""The column: a population of dendritic neurons with local plasticity.

A "column" is the unit of ownership -- in a distributed run, one machine holds
one column. Everything here is local by construction: no array in this module
is ever indexed by another column's current state. Inputs arrive through a
delay, and the only external signal is a low-dimensional broadcast modulator.

Three departures from biology are load-bearing:

1.  Branches are the real computational unit. A biological neuron is not a
    summer -- each dendritic branch integrates its own synapses and applies a
    local NMDA-plateau nonlinearity, making one neuron closer to a two-layer
    network that detects *coincident, clustered* input. We keep this, because
    it buys computation per message, and messages are the expensive resource.

2.  Time constants are heterogeneous and learnable. Biology's are pinned by ion
    channel kinetics and adapt over days. Ours are parameters. Heterogeneity
    alone is a large win: a population with membrane constants spread over
    10-300ms holds working memory that a homogeneous one cannot.

3.  The modulator is a routed vector, not a diffuse chemical bath. Dopamine
    broadcasts one scalar to millions of synapses by diffusion. We broadcast a
    small vector and let every neuron read it through its own learned
    projection, so different neurons extract different credit from the same
    signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .transport import Transport


@dataclass
class ColumnConfig:
    """Structural and dynamical hyperparameters for a column."""

    n_neurons: int = 256
    n_branches: int = 8
    n_synapses: int = 16
    n_external: int = 0

    # Membrane and branch integration (milliseconds; dt is 1ms).
    tau_branch: float = 15.0
    tau_soma_min: float = 12.0
    tau_soma_max: float = 320.0

    # Conduction delays, in steps. Real axons run 0.5-30ms and the brain uses
    # that spread for temporal coding; we sample the same range.
    delay_min: int = 1
    delay_max: int = 24

    # Dendritic plateau, defined relative to its own knee so that it is
    # scale-free: phi(x) = x + plateau * knee * sigmoid(sharpness * (x/knee - 1)).
    # An absolute knee is unusable here -- branch potentials depend on fan-in,
    # weight scale and input sparsity, and a knee that sits above the operating
    # range leaves the nonlinearity permanently disengaged.
    plateau: float = 1.2
    plateau_sharpness: float = 6.0
    knee_init: float = 0.15
    # The knee adapts locally to keep each branch's plateau engaged a target
    # fraction of the time -- dendritic excitability homeostasis, and the same
    # trick as the somatic threshold one level up.
    plateau_engagement: float = 0.12
    knee_lr: float = 3e-3
    tau_engagement: float = 600.0

    # Emission. An event carries a baseline magnitude plus a graded component,
    # saturating at value_scale. The baseline matters: with a soft threshold,
    # neurons cross by a hair, so a purely graded payload would be ~0 for
    # almost every event. This way an event is never less informative than a
    # binary spike, and usually more.
    value_base: float = 0.6
    value_scale: float = 2.0
    threshold_init: float = 1.0
    surrogate_width: float = 0.5

    # Eligibility trace horizon. This is the credit-assignment window *and*
    # the tolerance for a late-arriving modulator -- they are the same number.
    tau_eligibility: float = 240.0

    # Short-term synaptic plasticity (Tsodyks-Markram). Facilitation outlasts
    # depression here, which puts terminals in the regime where a transient
    # burst leaves a lasting trace in synaptic *efficacy* rather than in
    # ongoing spiking -- working memory that costs no activity to hold.
    # A 2%-sparse network has far too few active units to bridge a long delay
    # with persistent firing; this is how biology does it instead.
    stp: bool = True
    stp_u: float = 0.2
    tau_facilitation: float = 1200.0
    tau_depression: float = 200.0

    # Inhibitory fraction (Dale's law is enforced on outgoing sign).
    inhibitory_frac: float = 0.2

    # Homeostasis. Threshold adaptation is multiplicative so that its step size
    # tracks the neuron's own operating scale rather than a fixed absolute
    # amount, which converges far faster across a heterogeneous population.
    target_rate: float = 0.03
    homeostatic_lr: float = 1.5e-2
    tau_rate: float = 300.0

    # Learning.
    lr: float = 4e-3
    lr_tau: float = 0.0  # set > 0 to learn membrane time constants
    weight_max: float = 4.0
    branch_budget: float = 6.0  # target L1 norm of |W| per branch
    scaling_lr: float = 2e-2
    scaling_every: int = 20  # steps between synaptic scaling sweeps

    seed: int = 0

    modulator_dim: int = 2
    dt: float = 1.0

    fanin_recurrent_frac: float = 0.55

    def __post_init__(self) -> None:
        if self.delay_min < 1:
            raise ValueError("delay_min must be >= 1 (a synapse cannot read the present)")
        if self.delay_max < self.delay_min:
            raise ValueError("delay_max must be >= delay_min")


class Column:
    """A population of branch-structured neurons with three-factor learning.

    The source space is ``[external inputs | this column's own neurons]``, so
    recurrence is just self-addressed synapses and needs no special machinery.
    In a multi-column run the space extends with remote neurons; the read path
    is unchanged.
    """

    def __init__(self, cfg: ColumnConfig, transport: Transport):
        self.cfg = cfg
        self.transport = transport
        rng = np.random.default_rng(cfg.seed)
        self.rng = rng

        N, B, S = cfg.n_neurons, cfg.n_branches, cfg.n_synapses
        self.n_sources = cfg.n_external + N
        if transport.n_sources < self.n_sources:
            raise ValueError(
                f"transport carries {transport.n_sources} sources, column needs {self.n_sources}"
            )
        if transport.depth <= cfg.delay_max:
            raise ValueError("transport depth must exceed delay_max")

        # --- Connectivity -------------------------------------------------
        # Each branch draws its synapses from a mix of external and recurrent
        # sources. Clustering matters: synapses on one branch are what the
        # plateau nonlinearity conjoins, so a branch is a coincidence detector
        # over whichever sources land on it.
        self.src = self._sample_sources(rng, N, B, S)
        self.delay = rng.integers(cfg.delay_min, cfg.delay_max + 1, size=(N, B, S)).astype(np.int64)

        # Dale's law: a unit is excitatory or inhibitory, and every synapse it
        # makes carries that sign for life. Learning moves magnitude only.
        sign = np.ones(self.n_sources, dtype=np.float32)
        n_inh = int(round(cfg.inhibitory_frac * N))
        if n_inh:
            inh = rng.choice(N, size=n_inh, replace=False) + cfg.n_external
            sign[inh] = -1.0
        self.source_sign = sign
        self.syn_sign = sign[self.src].astype(np.float32)

        # --- Parameters ---------------------------------------------------
        self.W = np.abs(rng.normal(0.0, 0.4, size=(N, B, S))).astype(np.float32)
        self.G = rng.uniform(0.4, 1.0, size=(N, B)).astype(np.float32)
        self.bias = np.zeros(N, dtype=np.float32)
        self.theta = np.full(N, cfg.threshold_init, dtype=np.float32)

        self.decay_branch = np.float32(np.exp(-cfg.dt / cfg.tau_branch))
        # Log-uniform spread of membrane constants across the population.
        tau_v = np.exp(
            rng.uniform(np.log(cfg.tau_soma_min), np.log(cfg.tau_soma_max), size=N)
        ).astype(np.float32)
        self.tau_soma = tau_v
        self.decay_soma = np.exp(-cfg.dt / tau_v).astype(np.float32)
        self.decay_elig = np.float32(np.exp(-cfg.dt / cfg.tau_eligibility))
        self.decay_rate = np.float32(np.exp(-cfg.dt / cfg.tau_rate))

        # Unit-DC-gain companions for every leaky filter above.
        self.gain_branch = np.float32(1.0 - self.decay_branch)
        self.gain_soma = (1.0 - self.decay_soma).astype(np.float32)
        self.gain_elig = np.float32(1.0 - self.decay_elig)

        # Per-neuron projection of the broadcast modulator. This is the
        # "routed vector modulator" -- each neuron reads the same broadcast
        # differently. Overwritten by the network when using symmetric
        # feedback; kept random for a DFA-style scheme.
        self.feedback = rng.normal(
            0.0, 1.0 / np.sqrt(cfg.modulator_dim), size=(N, cfg.modulator_dim)
        ).astype(np.float32)

        # --- State ----------------------------------------------------------
        self.b = np.zeros((N, B), dtype=np.float32)
        self.v = np.zeros(N, dtype=np.float32)
        self.out = np.zeros(N, dtype=np.float32)
        self.rate = np.full(N, cfg.target_rate, dtype=np.float32)
        self.knee = np.full((N, B), cfg.knee_init, dtype=np.float32)
        self.engagement = np.full((N, B), cfg.plateau_engagement, dtype=np.float32)
        self.decay_engage = np.float32(np.exp(-cfg.dt / cfg.tau_engagement))

        # Presynaptic terminal state: utilisation (facilitation) and available
        # resources (depression). Modelled per source rather than per synapse --
        # a real terminal's state is shared across the axon's targets, and it
        # keeps this to two length-N vectors instead of two (N,B,S) tensors.
        self.u = np.full(N, cfg.stp_u, dtype=np.float32)
        self.res = np.ones(N, dtype=np.float32)
        self.gain_fac = np.float32(1.0 - np.exp(-cfg.dt / cfg.tau_facilitation))
        self.gain_dep = np.float32(1.0 - np.exp(-cfg.dt / cfg.tau_depression))
        # Release from a fully rested terminal, used to normalise efficacy to
        # 1.0 at rest so enabling STP does not rescale the whole network.
        self._release_ref = np.float32(cfg.stp_u * (2.0 - cfg.stp_u))

        # --- Traces ---------------------------------------------------------
        self.pre = np.zeros((N, B, S), dtype=np.float32)  # branch-filtered input
        self.eps = np.zeros((N, B, S), dtype=np.float32)  # soma-filtered sensitivity
        self.elig = np.zeros((N, B, S), dtype=np.float32)  # reward-bridging trace
        self.dv_dlam = np.zeros(N, dtype=np.float32)
        self.elig_tau = np.zeros(N, dtype=np.float32)

        self.learning = True
        self._steps = 0

        # Synapse sources and delays never change, so the ring-buffer slot each
        # synapse reads depends only on t mod depth. Precomputing the whole
        # index table turns the per-step gather -- which profiled at ~46% of
        # runtime -- into a single flat take.
        self._fast = bool(getattr(transport, "dense", False))
        if self._fast:
            d = transport.depth
            slots = (np.arange(d)[:, None, None, None] - self.delay[None]) % d
            self._flat = (
                (slots * transport.n_sources + self.src[None]).astype(np.int32).reshape(d, -1)
            )
        self._shape = (N, B, S)

    # -----------------------------------------------------------------
    def _sample_sources(self, rng, N: int, B: int, S: int) -> np.ndarray:
        cfg = self.cfg
        n_ext = cfg.n_external
        src = np.empty((N, B, S), dtype=np.int64)
        n_rec = int(round(cfg.fanin_recurrent_frac * S)) if n_ext > 0 else S
        n_rec = min(n_rec, S)
        n_ext_syn = S - n_rec
        for n in range(N):
            for b in range(B):
                picks = []
                if n_ext_syn > 0:
                    picks.append(rng.integers(0, n_ext, size=n_ext_syn))
                if n_rec > 0:
                    picks.append(rng.integers(0, cfg.n_neurons, size=n_rec) + n_ext)
                src[n, b] = rng.permutation(np.concatenate(picks))
        return src

    # -----------------------------------------------------------------
    def _phi(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Dendritic plateau nonlinearity, its derivative, and engagement.

        Linear passthrough plus a saturating bump scaled to the knee: weak
        input propagates proportionally, but input that crosses the knee
        triggers a plateau that superlinearly amplifies coincident synapses on
        the same branch. Because both the bump height and the sharpness are
        expressed in units of the knee, the shape is invariant to the branch's
        operating scale -- only the knee itself has to adapt.
        """
        cfg = self.cfg
        k = np.maximum(self.knee, 1e-4)
        z = cfg.plateau_sharpness * (x / k - 1.0)
        sig = 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
        val = x + cfg.plateau * k * sig
        deriv = 1.0 + cfg.plateau * cfg.plateau_sharpness * sig * (1.0 - sig)
        return val.astype(np.float32), deriv.astype(np.float32), sig.astype(np.float32)

    def _emit(self, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Threshold and payload.

        Note the emitted value is continuous at threshold -- it rises from zero
        rather than jumping to one, as a binary spike would. The discontinuity
        that forces surrogate gradients in spiking networks is simply absent
        from the forward pass; the surrogate below only widens the window so
        that near-silent neurons still receive credit.
        """
        cfg = self.cfg
        u = v - self.theta
        fired = u > 0.0
        span = cfg.value_scale - cfg.value_base
        value = np.where(
            fired, cfg.value_base + span * np.tanh(np.maximum(u, 0.0) / span), 0.0
        ).astype(np.float32)
        susceptibility = (1.0 / (1.0 + np.abs(u) / cfg.surrogate_width) ** 2).astype(np.float32)
        return value, susceptibility

    def _short_term_plasticity(self, value: np.ndarray, fired: np.ndarray) -> np.ndarray:
        """Tsodyks-Markram terminal dynamics; returns efficacy-scaled output.

        Utilisation relaxes toward its baseline with ``tau_facilitation`` and
        jumps on every emission; resources deplete on release and recover with
        ``tau_depression``. With facilitation the slower of the two, a burst of
        activity leaves the terminal primed for far longer than the burst
        lasted -- so a cue can be held across a delay in which the neuron is
        silent, and read back out when something probes the network.
        """
        cfg = self.cfg
        self.u += (cfg.stp_u - self.u) * self.gain_fac
        self.res += (1.0 - self.res) * self.gain_dep
        u_post = np.where(fired, self.u + cfg.stp_u * (1.0 - self.u), self.u).astype(np.float32)
        release = np.where(fired, u_post * self.res, 0.0).astype(np.float32)
        self.res = np.where(fired, self.res - release, self.res).astype(np.float32)
        self.u = u_post
        return (value * release / self._release_ref).astype(np.float32)

    # -----------------------------------------------------------------
    def step(self, t: int, external: np.ndarray | None = None) -> np.ndarray:
        """Advance one timestep and return this column's emitted values.

        Ordering matters: we publish *last* step's output before gathering, so
        every read goes through a delay of at least one step and no unit ever
        observes the present. That is what removes the synchronisation barrier.
        """
        cfg = self.cfg

        # 1. Publish local activity for time t, alongside external drive.
        frame = np.zeros(self.transport.n_sources, dtype=np.float32)
        if cfg.n_external:
            if external is None:
                raise ValueError("column configured with external inputs but none supplied")
            frame[: cfg.n_external] = external
        frame[cfg.n_external : cfg.n_external + cfg.n_neurons] = self.out
        self.transport.publish(t, frame)

        # 2. Gather delayed input for every synapse, applying Dale sign.
        if self._fast:
            x = self.transport.take(self._flat[t % self.transport.depth]).reshape(self._shape)
        else:
            x = self.transport.gather(t, self.src, self.delay)
        x = x * self.syn_sign

        # 3. Branch integration and plateau nonlinearity.
        #    Every filter here is written with unit DC gain -- the input is
        #    scaled by (1 - decay) rather than added raw. Without this a leaky
        #    integrator multiplies sustained input by 1/(1-decay), so a neuron
        #    with a 320ms constant would simply be 320x louder than one with a
        #    12ms constant. Heterogeneous time constants must buy memory, not
        #    gain, or the population is just badly normalised.
        self.b *= self.decay_branch
        self.b += self.gain_branch * x.sum(axis=2)
        a, dphi, engaged = self._phi(self.b)

        # 4. Soma integration.
        drive = (self.G * a).sum(axis=1) + self.bias
        v_prev = self.v
        self.v = self.decay_soma * v_prev + self.gain_soma * drive

        # 5. Emission and soft reset. Subtracting the threshold rather than
        #    clearing to zero preserves what the integrator was holding, which
        #    matters for the long-constant neurons that carry working memory.
        value, h = self._emit(self.v)
        fired = value > 0.0
        if cfg.stp:
            value = self._short_term_plasticity(value, fired)
        self.out = value
        self.v = (self.v - self.theta * fired).astype(np.float32)

        # 6. Trace updates (all local: pre-activity, post-state, nothing else).
        self.pre *= self.decay_branch
        self.pre += self.gain_branch * x
        sens = (self.G * dphi)[:, :, None]  # d v / d b, per branch
        self.eps *= self.decay_soma[:, None, None]
        self.eps += self.gain_soma[:, None, None] * self.pre * sens
        self.elig *= self.decay_elig
        self.elig += self.gain_elig * h[:, None, None] * self.eps

        if cfg.lr_tau > 0.0:
            self.dv_dlam = (v_prev + self.decay_soma * self.dv_dlam).astype(np.float32)
            self.dv_dlam = np.where(fired, 0.0, self.dv_dlam).astype(np.float32)
            self.elig_tau = (self.decay_elig * self.elig_tau + h * self.dv_dlam).astype(np.float32)

        # 7. Homeostasis: hold each neuron near its target rate using only its
        #    own history. No global normalisation, so nothing to synchronise.
        self.rate = (self.decay_rate * self.rate + (1.0 - self.decay_rate) * fired).astype(
            np.float32
        )
        self.engagement = (
            self.decay_engage * self.engagement + (1.0 - self.decay_engage) * engaged
        ).astype(np.float32)
        if self.learning:
            self.theta *= 1.0 + cfg.homeostatic_lr * (self.rate - cfg.target_rate)
            np.clip(self.theta, 0.02, 50.0, out=self.theta)
            # Same idea one level down: hold each branch near a target plateau
            # engagement so the dendritic nonlinearity stays in its useful band.
            self.knee += cfg.knee_lr * self.knee * (self.engagement - cfg.plateau_engagement)
            np.clip(self.knee, 1e-3, 50.0, out=self.knee)
            if self._steps % cfg.scaling_every == 0:
                self._synaptic_scaling()

        self._steps += 1
        return self.out

    # -----------------------------------------------------------------
    def apply_modulator(self, t: int) -> None:
        """Consume the broadcast modulator and commit weight changes.

        The modulator may be arbitrarily stale. Eligibility traces span
        ``tau_eligibility``, so a signal that arrives hundreds of steps late
        still lands on the synapses that earned it. Biology evolved this to
        bridge delayed reward; we get delayed *packets* handled by the very
        same mechanism.
        """
        if not self.learning:
            return
        cfg = self.cfg
        m = self.transport.modulator(t)
        if not np.any(m):
            return

        signal = (self.feedback @ m).astype(np.float32)  # (N,)
        self.W += cfg.lr * signal[:, None, None] * self.elig
        np.clip(self.W, 0.0, cfg.weight_max, out=self.W)

        if cfg.lr_tau > 0.0:
            grad = cfg.lr_tau * signal * self.elig_tau
            lam = np.clip(self.decay_soma + grad * self.decay_soma * (1.0 - self.decay_soma), 0.5, 0.9995)
            self.decay_soma = lam.astype(np.float32)
            self.gain_soma = (1.0 - self.decay_soma).astype(np.float32)
            self.tau_soma = (-cfg.dt / np.log(self.decay_soma)).astype(np.float32)

    def _synaptic_scaling(self) -> None:
        """Local multiplicative scaling toward a per-branch weight budget.

        Unsupervised and continuous: it never consults the modulator, so it
        runs on its own schedule rather than being gated by reward. Each branch
        sees only its own synapses, so there is nothing to synchronise.
        """
        cfg = self.cfg
        if cfg.scaling_lr <= 0.0:
            return
        norm = np.abs(self.W).sum(axis=2, keepdims=True) + 1e-6
        self.W *= (1.0 + cfg.scaling_lr * (cfg.branch_budget / norm - 1.0)).astype(np.float32)

    # -----------------------------------------------------------------
    def reset_state(self, keep_homeostasis: bool = True) -> None:
        """Clear dynamical state and traces between episodes."""
        self.b.fill(0.0)
        self.v.fill(0.0)
        self.out.fill(0.0)
        self.pre.fill(0.0)
        self.eps.fill(0.0)
        self.elig.fill(0.0)
        self.dv_dlam.fill(0.0)
        self.elig_tau.fill(0.0)
        self.u.fill(self.cfg.stp_u)
        self.res.fill(1.0)
        if not keep_homeostasis:
            self.rate.fill(self.cfg.target_rate)
            self.theta.fill(self.cfg.threshold_init)
            self.knee.fill(self.cfg.knee_init)
            self.engagement.fill(self.cfg.plateau_engagement)

    @property
    def plateau_engagement(self) -> float:
        """Fraction of branches with the plateau active, recently averaged."""
        return float(self.engagement.mean())

    def clear_eligibility(self) -> None:
        self.elig.fill(0.0)
        self.elig_tau.fill(0.0)

    @property
    def sparsity(self) -> float:
        """Fraction of neurons emitting, averaged over the recent past."""
        return float(self.rate.mean())
