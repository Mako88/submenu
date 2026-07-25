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
    # How the per-step sensitivity d(out)/dv is computed.
    #   "window" -- a smooth bump around threshold, ignoring whether the neuron
    #               fired. Standard surrogate-gradient practice, and what the
    #               gradient check measured as carrying sign but not magnitude.
    #   "graded" -- the true derivative of the emitted value where the output is
    #               actually smooth. Valued events make the emission continuous
    #               above threshold, so this part is a real gradient, not a
    #               surrogate; below threshold it is exactly zero.
    #   "hybrid" -- graded where the neuron fired, plus a small window term to
    #               keep credit flowing to silent-but-close neurons.
    surrogate: str = "window"
    surrogate_mix: float = 0.3  # weight of the window term in "hybrid"
    value_base: float = 0.6
    value_scale: float = 2.0
    threshold_init: float = 1.0
    surrogate_width: float = 0.5

    # Eligibility trace horizon. This single number does two jobs that pull in
    # opposite directions: it sets how precisely credit is assigned, and it sets
    # how late a modulator may arrive. Matching it to the readout's filter
    # (~60ms) is what moved plasticity from actively harmful to neutral -- at
    # 240ms credit was smeared over activity predating the cue. Raising it buys
    # latency tolerance at the cost of credit precision, and separating the two
    # into distinct traces is the obvious next design step.
    tau_eligibility: float = 70.0

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

    # --- Engram allocation ------------------------------------------------
    # Biology does not spread a memory over whatever happened to be active. It
    # *allocates* it: in the hours before an event, neurons drift apart in
    # intrinsic excitability (CREB is the usual marker), the most excitable ~15%
    # win a competition to be recruited, and being recruited then drops their
    # excitability for hours so the next memory lands on a different set (Han
    # et al. 2007; Yiu et al. 2014; Cai et al. 2016). That refractory step is
    # what stops successive memories from overwriting each other, and it is
    # exactly the property a reservoir lacks.
    #
    # Our homeostasis currently does the opposite: it drives every neuron to
    # the same firing rate, which erases the excitability differences allocation
    # depends on. So this is not simply an addition -- it requires the
    # homeostatic setpoint to become per-neuron, tracking each unit's own
    # excitability rather than fighting it. `excite_setpoint` is that coupling,
    # and setting it to 0 recovers today's behaviour with the drift left in,
    # which is the control the mechanism has to beat.
    engram: bool = False
    excite_drift: float = 0.35  # stationary sd of the excitability bias
    tau_excite: float = 2000.0  # drift timescale, and allocation recovery
    excite_setpoint: float = 1.0  # how far the homeostatic target follows xi
    alloc_frac: float = 0.15  # target share of the population recruited
    alloc_drop: float = 0.6  # excitability lost on being recruited
    tau_tag_fast: float = 50.0  # window defining "active right now"
    tau_tag_slow: float = 5000.0  # each neuron's own activity baseline
    tag_lr: float = 0.05  # adaptation of the per-neuron recruitment margin
    hebb_lr: float = 0.02  # Hebbian binding within the recruited assembly
    # How strongly excitability biases the recruitment competition directly,
    # over and above what it already does by lowering the threshold. It needs a
    # direct path: routing it only through firing does not work, because a
    # neuron made more excitable raises its own activity baseline along with its
    # activity, and a criterion measured against that baseline cancels exactly
    # the effect it was supposed to detect. Measured at -0.40 correlation
    # between excitability and recruitment -- backwards -- before this existed.
    excite_gain: float = 1.0

    # Homeostasis. Threshold adaptation is multiplicative so that its step size
    # tracks the neuron's own operating scale rather than a fixed absolute
    # amount, which converges far faster across a heterogeneous population.
    target_rate: float = 0.03
    homeostatic_lr: float = 1.5e-2
    tau_rate: float = 300.0

    # Learning.
    #
    # `elig_norm` selects what the weight update is normalised by, and the
    # choice matters more than it looks:
    #   "column" -- one scale for the whole population. Makes `lr` meaningful
    #               while preserving the natural weighting in which strongly
    #               engaged neurons receive larger updates.
    #   "neuron" -- each neuron divides by its own eligibility magnitude. This
    #               destroys that relative weighting: a neuron that barely
    #               participated has tiny, mostly-noise eligibility, and
    #               normalising amplifies that noise to full scale alongside
    #               genuinely engaged ones.
    #   "none"   -- raw. `lr` then depends on trace magnitude and is not
    #               portable across configurations.
    elig_norm: str = "column"
    # What of the eligibility trace the update actually uses.
    #   "magnitude" -- the trace as-is.
    #   "sign"      -- direction only, gated by magnitude. A finite-difference
    #                  check (experiments/gradcheck.py) finds the trace agrees
    #                  with the true gradient in sign 0.64 of the time
    #                  (p = 0.004) and in magnitude not at all (r = -0.045), so
    #                  the magnitude is contributing noise of the same order as
    #                  the signal. Magnitude is still used as a participation
    #                  gate, since a near-zero trace means the synapse was not
    #                  involved and the sign check has nothing to say about it.
    elig_mode: str = "magnitude"
    elig_gate: float = 1.0  # gate at this multiple of the trace's own RMS
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
    # Shared across every column so they agree on which units inhibit.
    sign_seed: int = 12345

    # Placement in a shared source space, for splitting a model across columns.
    # `source_offset` is where this column's own neurons live; `peer_span` is
    # the full range of neurons it may draw synapses from, own and remote. The
    # defaults keep a single column self-contained.
    source_offset: int | None = None
    peer_span: int | None = None
    # Conduction delay for synapses onto neurons owned by another column. This
    # is the only thing that changes when a column moves to another machine:
    # a remote peer is just a longer axon.
    peer_delay_min: int | None = None
    peer_delay_max: int | None = None
    peer_frac: float = 0.35  # share of recurrent synapses drawn from peers

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
        # Where this column's neurons live in the shared source space, and how
        # far its synapses may reach. A lone column owns everything after the
        # external inputs; in a split model each owns its own slice.
        self.source_offset = cfg.n_external if cfg.source_offset is None else cfg.source_offset
        self.peer_span = N if cfg.peer_span is None else cfg.peer_span
        self.n_sources = max(cfg.n_external + self.peer_span, self.source_offset + N)
        if transport.n_sources < self.n_sources:
            raise ValueError(
                f"transport carries {transport.n_sources} sources, column needs {self.n_sources}"
            )


        # --- Connectivity -------------------------------------------------
        # Each branch draws its synapses from a mix of external and recurrent
        # sources. Clustering matters: synapses on one branch are what the
        # plateau nonlinearity conjoins, so a branch is a coincidence detector
        # over whichever sources land on it.
        self.src = self._sample_sources(rng, N, B, S)
        self.delay = rng.integers(cfg.delay_min, cfg.delay_max + 1, size=(N, B, S)).astype(np.int64)
        # Synapses reaching a neuron owned by another column carry the longer
        # peer delay. This is the whole of what distribution costs the model.
        if cfg.peer_delay_max is not None:
            own_lo, own_hi = self.source_offset, self.source_offset + N
            remote = (self.src >= cfg.n_external) & ((self.src < own_lo) | (self.src >= own_hi))
            lo = cfg.peer_delay_min if cfg.peer_delay_min is not None else cfg.peer_delay_max
            self.delay = np.where(
                remote,
                rng.integers(lo, cfg.peer_delay_max + 1, size=self.delay.shape),
                self.delay,
            ).astype(np.int64)
        self.max_delay = int(self.delay.max())
        if transport.depth <= self.max_delay:
            raise ValueError(
                f"transport depth {transport.depth} must exceed max delay {self.max_delay}"
            )

        # Dale's law: a unit is excitatory or inhibitory, and every synapse it
        # makes carries that sign for life. Learning moves magnitude only.
        #
        # Signs come from a dedicated shared RNG, not this column's seed,
        # because being excitatory is a property of the *emitting* neuron and
        # not of whoever reads it. Every column must therefore derive the same
        # sign for the same global index -- otherwise a neuron would excite one
        # column while inhibiting another, which no biological network does and
        # which nothing downstream could detect.
        sign_rng = np.random.default_rng(cfg.sign_seed)
        sign = np.ones(self.n_sources, dtype=np.float32)
        n_units = self.n_sources - cfg.n_external
        n_inh = int(round(cfg.inhibitory_frac * n_units))
        if n_inh:
            inh = sign_rng.choice(n_units, size=n_inh, replace=False) + cfg.n_external
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

        # --- Engram allocation state ---------------------------------------
        # `xi` is the excitability bias: a slow drift that makes the population
        # heterogeneous at any given moment, minus a drop each time the neuron
        # is recruited. It enters the threshold directly, so it acts on the very
        # next step rather than waiting for homeostasis to notice.
        #
        # `xi_slow` is the neuron's own long-run excitability. The homeostatic
        # setpoint follows xi *relative to it*, so a neuron that is transiently
        # more excitable than usual is allowed to fire more, while its baseline
        # rate is still regulated. Without that reference, the allocation drops
        # would push the whole population's setpoint down over time.
        self.xi = np.zeros(N, dtype=np.float32)
        self.xi_slow = np.zeros(N, dtype=np.float32)
        self.theta_eff = self.theta
        self.decay_xi = np.float32(np.exp(-cfg.dt / cfg.tau_excite))
        # Innovation scaled so the Ornstein-Uhlenbeck process has stationary
        # standard deviation `excite_drift` regardless of its timescale.
        self.sigma_xi = np.float32(cfg.excite_drift * np.sqrt(1.0 - self.decay_xi**2))
        # Fast and slow activity filters, plus the neuron's own variability.
        # Recruitment is a z-score against these -- how far *this* neuron is
        # from its own usual state, in units of its own usual fluctuation -- so
        # the criterion is dimensionless and needs no knowledge of anyone else.
        #
        # The variability term is what makes it work. Measured directly, 99.7%
        # of the column's state vector is an episode-independent common mode;
        # the part that identifies which cues were presented is a small residual
        # on top of it (pairwise correlation +0.23 within a cue pattern, -0.08
        # across). A criterion scaled by the neuron's *magnitude* cannot resolve
        # a residual that small, and an earlier version that used a ratio
        # against the mean recruited essentially the same neurons every episode.
        self.act_fast = np.zeros(N, dtype=np.float32)
        self.act_slow = np.zeros(N, dtype=np.float32)
        self.act_var = np.zeros(N, dtype=np.float32)
        self.n_samples = 0
        self.decay_fast = np.float32(np.exp(-cfg.dt / cfg.tau_tag_fast))
        self.decay_slow = np.float32(np.exp(-cfg.dt / cfg.tau_tag_slow))
        self.decay_stats = np.float32(0.99)  # per allocation, not per step
        # Per-neuron recruitment margin, adapted to hold each unit's own share
        # of allocations near `alloc_frac`. This is the local stand-in for the
        # lateral-inhibition competition biology uses: no neuron can see the
        # others, but each can see how often it has been winning.
        self.tag_margin = np.full(N, 1.0, dtype=np.float32)
        self.tag = np.zeros(N, dtype=bool)
        self.n_allocations = 0

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
        # Bias-corrected, and its window is measured in *updates*, not steps.
        # Both matter: seeding this at 1e-3 with a 0.999 decay was fine while
        # the modulator fired ~70x per episode, but once one decision produced
        # one release the EMA advanced 70x more slowly and stayed dominated by
        # its own initialisation -- tracking 4.6e-3 against an actual 2.4e-2, so
        # the effective learning rate ran ~5x high and drifted as it caught up.
        # The same failure as the readout's variance, one level down.
        self.elig_rms = np.zeros(N, dtype=np.float32)
        self.sensitivity = np.zeros((N, B, S), dtype=np.float32)
        self.decay_elig_rms = np.float32(0.99)
        self.n_updates = 0

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
                    n_peer = int(round(cfg.peer_frac * n_rec)) if self.peer_span > cfg.n_neurons else 0
                    if n_rec - n_peer > 0:
                        picks.append(
                            rng.integers(0, cfg.n_neurons, size=n_rec - n_peer) + self.source_offset
                        )
                    if n_peer > 0:
                        picks.append(rng.integers(0, self.peer_span, size=n_peer) + n_ext)
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
        u = v - self.theta_eff
        fired = u > 0.0
        span = cfg.value_scale - cfg.value_base
        value = np.where(
            fired, cfg.value_base + span * np.tanh(np.maximum(u, 0.0) / span), 0.0
        ).astype(np.float32)
        window = (1.0 / (1.0 + np.abs(u) / cfg.surrogate_width) ** 2).astype(np.float32)
        if cfg.surrogate == "window":
            susceptibility = window
        else:
            # d(value)/du where the emission is genuinely smooth. tanh(u/span)
            # has derivative sech^2(u/span) = 1 - tanh^2, and the emitted value
            # scales it by span, so d(value)/du = 1 - tanh^2(u/span).
            span = cfg.value_scale - cfg.value_base
            graded = np.where(fired, 1.0 - np.tanh(np.maximum(u, 0.0) / span) ** 2, 0.0)
            susceptibility = (
                graded if cfg.surrogate == "graded" else graded + cfg.surrogate_mix * window
            ).astype(np.float32)
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

    def publish(self, t: int) -> None:
        """Announce this column's last emission into the shared substrate.

        Split out of :meth:`step` so a multi-column driver can publish every
        column before any column reads. Combined with a minimum delay of one
        step, that makes the result independent of the order columns are
        stepped in -- the property that lets them live on separate machines.
        """
        self.transport.publish_slice(t, self.source_offset, self.out)

    # -----------------------------------------------------------------
    def step(self, t: int, external: np.ndarray | None = None) -> np.ndarray:
        """Advance one timestep and return this column's emitted values.

        Ordering matters: we publish *last* step's output before gathering, so
        every read goes through a delay of at least one step and no unit ever
        observes the present. That is what removes the synchronisation barrier.
        """
        cfg = self.cfg

        # 1. Publish the slice this column owns. External drive is published
        #    by the driver, since no column owns it. Publishing last step's
        #    output before gathering is what guarantees every read goes through
        #    a delay of at least one step.
        if external is not None and cfg.n_external:
            self.transport.publish_slice(t, 0, external)
            self.publish(t)

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
        self.b += self.gain_branch * (self.W * x).sum(axis=2)
        a, dphi, engaged = self._phi(self.b)

        # 4. Soma integration.
        drive = (self.G * a).sum(axis=1) + self.bias
        v_prev = self.v
        self.v = self.decay_soma * v_prev + self.gain_soma * drive

        # 5. Emission and soft reset. Subtracting the threshold rather than
        #    clearing to zero preserves what the integrator was holding, which
        #    matters for the long-constant neurons that carry working memory.
        #    The threshold a neuron actually uses is its homeostatic setpoint
        #    scaled by its current excitability bias; with allocation off the
        #    two are the same array and nothing changes.
        if cfg.engram:
            self.theta_eff = (self.theta * np.exp(-self.xi)).astype(np.float32)
        else:
            self.theta_eff = self.theta
        value, h = self._emit(self.v)
        fired = value > 0.0
        if cfg.stp:
            value = self._short_term_plasticity(value, fired)
        self.out = value
        self.v = (self.v - self.theta_eff * fired).astype(np.float32)

        # 6. Trace updates (all local: pre-activity, post-state, nothing else).
        self.pre *= self.decay_branch
        self.pre += self.gain_branch * x
        sens = (self.G * dphi)[:, :, None]  # d v / d b, per branch
        self.eps *= self.decay_soma[:, None, None]
        self.eps += self.gain_soma[:, None, None] * self.pre * sens
        self.elig *= self.decay_elig
        self.elig += self.gain_elig * h[:, None, None] * self.eps
        # Unfiltered per-step sensitivity, d(out)/dW before the reward-bridging
        # filter. Exposed so experiments/gradcheck.py can integrate it over a
        # run and compare against a finite difference.
        self.sensitivity = h[:, None, None] * self.eps

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
        if cfg.engram:
            # Excitability drifts on its own, slowly. This is the part that has
            # to happen *before* anything worth remembering does: allocation
            # works by the population already being unequal when the event
            # arrives, so the drift is not noise added to the mechanism, it is
            # the mechanism's input.
            self.xi = (
                self.decay_xi * self.xi
                + self.sigma_xi * self.rng.standard_normal(cfg.n_neurons)
            ).astype(np.float32)
            self.xi_slow = (
                self.decay_slow * self.xi_slow + (1.0 - self.decay_slow) * self.xi
            ).astype(np.float32)
            self.act_fast = (
                self.decay_fast * self.act_fast + (1.0 - self.decay_fast) * self.out
            ).astype(np.float32)
            # The baseline and variability that recruitment is scored against
            # are NOT accumulated here -- see _allocate. Advancing them every
            # timestep would score a neuron's activity at recruitment time
            # against the distribution of all timesteps, most of which are
            # nothing like it. A unit that reliably answers the go cue then
            # looks extraordinary at every single allocation, and the criterion
            # measures which neurons are phasic rather than which ones this
            # episode engaged. The readout failed in exactly this way, by
            # standardising an episode-level decision vector with per-timestep
            # variance; statistics have to be gathered over the objects being
            # scored, not over the substrate they are drawn from.

        if self.learning:
            if cfg.engram and cfg.excite_setpoint:
                # A per-neuron setpoint. Uniform-rate homeostasis is what
                # normally destroys allocation: it reads a transiently excitable
                # neuron as over-active and raises its threshold until the
                # excitability is gone. Referencing the target to the neuron's
                # *own* long-run excitability lets the difference stand while
                # still regulating the baseline.
                target = cfg.target_rate * np.exp(
                    cfg.excite_setpoint * (self.xi - self.xi_slow)
                )
            else:
                target = cfg.target_rate
            self.theta *= 1.0 + cfg.homeostatic_lr * (self.rate - target)
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

        # A modulator release is the salience gate: it says *now* is worth
        # remembering. Allocation reads only that -- not the modulator's sign,
        # not which class it points at -- so recruitment is unsupervised. That
        # matters, because the supervised path is the one that has repeatedly
        # failed to beat a frozen column, and this mechanism does not depend
        # on it being right.
        if cfg.engram:
            self._allocate()

        signal = (self.feedback @ m).astype(np.float32)  # (N,)

        # Normalise the update so `lr` means "this fraction of the weight scale
        # per update" rather than being hostage to trace magnitude. Chaining
        # unit-DC-gain filters leaves eligibility around 1e-2, and with a raw lr
        # the modulator moved weights by 0.1% while homeostatic scaling moved
        # them by 21% -- the three-factor rule was, measurably, decorative.
        # See `elig_norm` for why the scale is population-wide by default.
        if cfg.elig_norm == "neuron":
            rms = np.sqrt(np.mean(self.elig**2, axis=(1, 2))).astype(np.float32)
        elif cfg.elig_norm == "column":
            rms = np.full(
                self.cfg.n_neurons, np.sqrt(np.mean(self.elig**2)), dtype=np.float32
            )
        else:
            rms = np.ones(self.cfg.n_neurons, dtype=np.float32)
        d = self.decay_elig_rms
        self.n_updates += 1
        self.elig_rms = (d * self.elig_rms + (1.0 - d) * rms).astype(np.float32)
        corrected = self.elig_rms / (1.0 - d**self.n_updates)
        scale = np.maximum(corrected, 1e-8)[:, None, None]
        if cfg.elig_mode == "sign":
            involved = np.abs(self.elig) > cfg.elig_gate * scale
            drive = np.sign(self.elig) * involved
        else:
            drive = self.elig / scale
        self.W += cfg.lr * signal[:, None, None] * drive
        np.clip(self.W, 0.0, cfg.weight_max, out=self.W)

        if cfg.lr_tau > 0.0:
            grad = cfg.lr_tau * signal * self.elig_tau
            lam = np.clip(self.decay_soma + grad * self.decay_soma * (1.0 - self.decay_soma), 0.5, 0.9995)
            self.decay_soma = lam.astype(np.float32)
            self.gain_soma = (1.0 - self.decay_soma).astype(np.float32)
            self.tau_soma = (-cfg.dt / np.log(self.decay_soma)).astype(np.float32)

    def _allocate(self) -> None:
        """Recruit an engram, then make the winners refractory.

        Three things happen, in the order biology does them:

        1.  **Competition.** A neuron recruits itself if it is currently far
            more active than it usually is. The comparison is against its own
            baseline, so no ranking, sorting or population statistic is
            involved -- the criterion is dimensionless and strictly local.
            The per-neuron margin adapts to hold each unit's share of
            allocations near ``alloc_frac``, which is the local substitute for
            the lateral inhibition that enforces the competition in cortex.

        2.  **Binding.** Synapses that were driving a recruited neuron are
            strengthened. Only excitatory ones -- Hebbian LTP is a
            glutamatergic phenomenon, and boosting an inhibitory synapse that
            was active during recruitment would make the assembly *harder* to
            reactivate, which is the opposite of binding. The additive
            increment becomes competitive for free: synaptic scaling already
            holds each branch to a weight budget, so what one synapse gains it
            takes from its neighbours on the same branch.

        3.  **Refractoriness.** Being recruited costs excitability, so the next
            allocation lands on a different set. This is the step that makes
            memories separable instead of superimposed, and it is the one with
            no analogue anywhere in a conventional network.
        """
        cfg = self.cfg
        # Statistics advance once per allocation, so the window is measured in
        # allocations and z means "unusual for me, at moments like this one".
        d = self.decay_stats
        self.n_samples += 1
        self.act_slow = (d * self.act_slow + (1.0 - d) * self.act_fast).astype(np.float32)
        bias = 1.0 - d**self.n_samples
        dev = self.act_fast - self.act_slow / bias
        self.act_var = (d * self.act_var + (1.0 - d) * dev**2).astype(np.float32)
        z = dev / (np.sqrt(self.act_var / bias) + 1e-9)
        # Excitability adds directly to the competition rather than only tilting
        # the threshold. `xi_slow` is the reference for the same reason the
        # homeostatic setpoint uses it: what should win an allocation is a
        # neuron that is more excitable *than it usually is*, not one that
        # happens to sit high all the time.
        drive = z + cfg.excite_gain * (self.xi - self.xi_slow)
        tagged = drive > self.tag_margin
        # Additive, because the margin is now a z-score and can legitimately be
        # negative; a multiplicative step could never cross zero.
        self.tag_margin += (cfg.tag_lr * (tagged - cfg.alloc_frac)).astype(np.float32)
        np.clip(self.tag_margin, -20.0, 20.0, out=self.tag_margin)
        self.tag = tagged
        self.n_allocations += 1
        if not tagged.any():
            return

        self.xi = (self.xi - cfg.alloc_drop * tagged).astype(np.float32)
        if cfg.hebb_lr > 0.0:
            # `pre` carries the Dale sign; multiplying it back out recovers the
            # presynaptic activity itself, which is what Hebb's rule is about.
            excitatory = self.syn_sign > 0.0
            self.W += (
                cfg.hebb_lr * tagged[:, None, None] * (self.pre * self.syn_sign) * excitatory
            ).astype(np.float32)
            np.clip(self.W, 0.0, cfg.weight_max, out=self.W)

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
        # act_fast is dynamical state and goes; xi, its baseline and the
        # recruitment margin are intrinsic properties that persist across
        # episodes. They have to: an allocation refractory that reset every
        # episode would allocate every memory to the same neurons, which is
        # precisely the failure the mechanism exists to prevent.
        self.act_fast.fill(0.0)
        self.tag.fill(False)
        if not keep_homeostasis:
            self.rate.fill(self.cfg.target_rate)
            self.theta.fill(self.cfg.threshold_init)
            self.knee.fill(self.cfg.knee_init)
            self.engagement.fill(self.cfg.plateau_engagement)
            self.xi.fill(0.0)
            self.xi_slow.fill(0.0)
            self.act_slow.fill(0.0)
            self.act_var.fill(0.0)
            self.n_samples = 0
            self.tag_margin.fill(1.0)

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

    @property
    def excitability_spread(self) -> float:
        """Standard deviation of the excitability bias across the population.

        The quantity homeostasis would drive to zero if allowed to. Watching it
        is the cheapest way to tell whether allocation still has anything to
        allocate with.
        """
        return float(self.xi.std())

    @property
    def engram_size(self) -> float:
        """Fraction of the population recruited by the last allocation."""
        return float(self.tag.mean())
