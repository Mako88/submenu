"""Tests for the invariants the design actually depends on."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plexus import (
    Column,
    ColumnConfig,
    DelayedXOR,
    DistributedPlexus,
    EventBuffer,
    LocalTransport,
    Plexus,
)
from plexus.column import (
    NON_PLASTICITY_FLAGS,
    PLASTICITY_FLAGS,
    freeze_plasticity,
)
from plexus.readout import LinearReadout
from plexus.tasks import TemporalPatterns


# --------------------------------------------------------------------------
# Event buffer: timestamp indexing is what makes late delivery safe.
# --------------------------------------------------------------------------
def test_buffer_reads_by_emission_time():
    buf = EventBuffer(n_sources=4, depth=8)
    buf.write(10, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    got = buf.gather(13, np.array([0, 2]), np.array([3, 3]))
    assert np.allclose(got, [1.0, 3.0])


def test_buffer_returns_silence_for_unwritten_slots():
    buf = EventBuffer(n_sources=4, depth=8)
    buf.write(10, np.ones(4, dtype=np.float32))
    # Nothing was ever emitted at t=11, so a read of it must be zero rather
    # than an echo of whatever occupied the slot 8 steps earlier.
    assert np.allclose(buf.gather(12, np.array([0]), np.array([1])), [0.0])


def test_buffer_late_scatter_matches_dense_write():
    """A packet that arrives out of order must land in the right slot.

    This is the property that lets a distributed run compute the same thing as
    a local one: correctness depends on emission time, not arrival order.
    """
    dense = EventBuffer(4, 8)
    late = EventBuffer(4, 8)
    dense.write(5, np.array([0.0, 1.5, 0.0, 2.5], dtype=np.float32))
    # Same events, delivered in the wrong order and after later timesteps were
    # already being processed elsewhere.
    late.scatter(5, np.array([3]), np.array([2.5], dtype=np.float32))
    late.scatter(5, np.array([1]), np.array([1.5], dtype=np.float32))
    src, dly = np.arange(4), np.full(4, 2)
    assert np.allclose(dense.gather(7, src, dly), late.gather(7, src, dly))


# --------------------------------------------------------------------------
# Locality: the property the whole design rests on.
# --------------------------------------------------------------------------
def test_no_synapse_reads_the_present():
    """Every conduction delay must be >= 1 step.

    A delay of zero would mean a unit observing another unit's current state,
    which reintroduces exactly the synchronisation barrier we are avoiding.
    """
    cfg = ColumnConfig(n_neurons=32, n_external=8)
    transport = LocalTransport(8 + 32, cfg.delay_max + 2, cfg.modulator_dim)
    col = Column(cfg, transport)
    assert col.delay.min() >= 1


def test_delay_min_zero_is_rejected():
    with pytest.raises(ValueError):
        ColumnConfig(delay_min=0)


def test_transport_depth_must_exceed_max_delay():
    cfg = ColumnConfig(n_neurons=16, n_external=4, delay_max=10)
    transport = LocalTransport(4 + 16, depth=10, modulator_dim=2)
    with pytest.raises(ValueError):
        Column(cfg, transport)


# --------------------------------------------------------------------------
# Dynamics: the regime has to stay biologically sane on its own.
# --------------------------------------------------------------------------
def _settle(steps: int = 3000, **overrides):
    cfg = ColumnConfig(n_neurons=96, n_external=16, seed=0, **overrides)
    transport = LocalTransport(16 + 96, cfg.delay_max + 2, cfg.modulator_dim)
    col = Column(cfg, transport)
    rng = np.random.default_rng(0)
    for t in range(steps):
        ext = (rng.random(16) < 0.02).astype(np.float32)
        col.step(t, ext)
    return col


def test_homeostasis_holds_activity_sparse():
    """Firing rate must settle near target without any global normalisation."""
    col = _settle()
    assert 0.002 < col.sparsity < 0.12, col.sparsity


def test_no_runaway_excitation():
    """Membrane potential must stay on the scale the threshold defines.

    The bound is stated relative to theta rather than as an absolute, because
    an absolute one is unanchored: this asserted |v| < 50 while the settled
    operating range is 0.49, i.e. a hundredfold margin that no realistic
    failure could have crossed. Deliberately breaking homeostasis
    (target_rate=0.9) still produced |v| of 0.31 and the test still passed.
    A bound has to be tight enough that something can fail it.
    """
    col = _settle()
    assert np.all(np.isfinite(col.v))
    ceiling = 10.0 * float(np.median(col.theta))
    assert np.abs(col.v).max() < ceiling, (
        f"|v| reached {np.abs(col.v).max():.2f} against a threshold scale of "
        f"{np.median(col.theta):.2f}"
    )


def test_plateau_stays_engaged():
    """The dendritic nonlinearity must operate inside its useful band.

    Regression test for a real bug: with an absolute knee the branch potential
    sat two orders of magnitude below it and the nonlinearity never fired at
    all, silently reducing every neuron to a linear summer.

    The bounds used to be 0.01 to 0.5, which passed with knee homeostasis
    *entirely disabled* -- knee_lr=0 settles at 0.074 across five seeds
    [0.062, 0.084], comfortably inside them. A test guarding a
    silent-disconnection bug that survives its own disconnection is worth
    nothing, so the band is now tight enough to separate the two: adaptation
    on gives 0.152 [0.141, 0.170], off gives 0.074 [0.062, 0.084].

    The second assertion is the direct one. Engagement is a downstream
    quantity that the input distribution could push around on its own; the
    knee moving away from its initial value can only be the adaptation.
    """
    col = _settle()
    assert 0.10 < col.plateau_engagement < 0.25, col.plateau_engagement
    drift = float(np.abs(col.knee / ColumnConfig.knee_init - 1.0).mean())
    assert drift > 0.05, f"the knee never moved from its initial value ({drift:.3f})"


def test_membrane_gain_is_independent_of_time_constant():
    """Long time constants must buy memory, not gain.

    With a naive leaky integrator the DC gain is 1/(1-decay), so a 320ms
    neuron settles ~25x higher than a 12ms one under identical sustained
    drive. That is the exact defect this guards against, so the measurement is
    the steady-state level under constant input, with firing suppressed so the
    reset does not confound it. Transient *responsiveness* legitimately falls
    with tau; steady-state gain must not depend on it at all.
    """
    cfg = ColumnConfig(n_neurons=256, n_external=16, seed=1)
    transport = LocalTransport(16 + 256, cfg.delay_max + 2, cfg.modulator_dim)
    col = Column(cfg, transport)
    col.learning = False
    col.theta.fill(1e6)  # suppress firing; we want the raw integrator level
    steady = np.ones(16, dtype=np.float32)
    for t in range(4000):
        col.step(t, steady)
    r = np.corrcoef(col.tau_soma, np.abs(col.v))[0, 1]
    assert abs(r) < 0.2, f"tau/steady-state correlation {r:.3f} indicates a gain leak"


def test_dale_law_signs_are_preserved():
    """Learning may move synaptic magnitude but never flip a unit's sign."""
    task = DelayedXOR()
    m = Plexus(task.n_inputs, task.n_classes, column=ColumnConfig(n_neurons=48), seed=0)
    before = m.column.syn_sign.copy()
    rng = np.random.default_rng(0)
    for _ in range(3):
        m.run_episode(task.episode(rng), learn=True)
    assert np.array_equal(before, m.column.syn_sign)
    assert np.all(m.column.W >= 0.0)


def test_dale_signs_are_applied_to_the_input():
    """Preserving the signs is worthless if nothing multiplies by them.

    The test above checks only that `syn_sign` is not mutated and that W stays
    non-negative. Both hold perfectly well if the forward pass never applies
    the sign at all -- and a mutation that replaced `x * syn_sign` with
    `x * |syn_sign|`, deleting inhibition from the model entirely, was caught
    by nothing in the suite. E/I balance is a stated design property, so it
    needs an assertion that inhibition reaches the output.
    """
    cfg = ColumnConfig(n_neurons=32, n_external=8, seed=0, inhibitory_frac=0.4)

    def emissions(strip_inhibition):
        transport = LocalTransport(8 + 32, cfg.delay_max + 2, cfg.modulator_dim)
        col = Column(cfg, transport)
        col.learning = False
        if strip_inhibition:
            col.syn_sign = np.abs(col.syn_sign)
        rng = np.random.default_rng(3)
        return np.array(
            [col.step(t, (rng.random(8) < 0.2).astype(np.float32)) for t in range(400)]
        )

    mixed, excitatory_only = emissions(False), emissions(True)
    assert not np.allclose(mixed, excitatory_only), "the Dale sign never reached the input"
    # Homeostasis is off here (learning=False), so removing every inhibitory
    # synapse must leave the population strictly more active. With homeostasis
    # on, the thresholds would absorb the difference and hide the defect.
    assert excitatory_only.mean() > mixed.mean(), (
        "deleting all inhibition did not increase activity, so the sign is "
        "reaching the input as something other than a sign"
    )


def test_weights_stay_bounded():
    """Weights must stay finite and inside the clip under ordinary training."""
    task = DelayedXOR()
    cfg = ColumnConfig(n_neurons=48, lr=2e-2)
    m = Plexus(task.n_inputs, task.n_classes, column=cfg, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(6):
        m.run_episode(task.episode(rng), learn=True)
    assert np.all(np.isfinite(m.column.W))
    assert m.column.W.max() <= cfg.weight_max + 1e-5


def test_the_weight_clip_actually_binds():
    """The clip must be exercised, not merely satisfied by something else.

    The test above passes at settings where W tops out at 1.775 against a
    weight_max of 4.0 -- synaptic scaling holds it there, so the clip does no
    work and deleting it would change nothing. Raising weight_max to 1e9 left
    W at exactly 1.775, which is the proof: that assertion was vacuous.

    So this one puts the weights under real pressure -- scaling off, large
    learning rate, sustained one-signed modulator -- and asserts the clip is
    what stops them. Without it the update is unbounded.
    """
    cfg = ColumnConfig(n_neurons=16, n_external=4, lr=0.5, scaling_lr=0.0, weight_max=2.0)
    transport = LocalTransport(4 + 16, cfg.delay_max + 2, cfg.modulator_dim)
    col = Column(cfg, transport)
    rng = np.random.default_rng(0)
    col.feedback = np.ones_like(col.feedback)  # every neuron told "strengthen"
    for t in range(400):
        col.step(t, (rng.random(4) < 0.4).astype(np.float32))
        transport.broadcast(t, np.ones(cfg.modulator_dim, dtype=np.float32))
        col.apply_modulator(t)
    assert np.isclose(col.W.max(), cfg.weight_max), (
        f"W topped out at {col.W.max():.3f}, so the clip never bound and this "
        "test is not measuring it"
    )
    assert np.all(np.isfinite(col.W))


# --------------------------------------------------------------------------
# Short-term plasticity: the mechanism that carries memory across a delay.
# --------------------------------------------------------------------------
def test_rested_terminal_has_unit_efficacy():
    """Enabling STP must not silently rescale the whole network."""
    cfg = ColumnConfig(n_neurons=8, n_external=4)
    transport = LocalTransport(4 + 8, cfg.delay_max + 2, cfg.modulator_dim)
    col = Column(cfg, transport)
    value = np.ones(8, dtype=np.float32)
    fired = np.ones(8, dtype=bool)
    out = col._short_term_plasticity(value, fired)
    assert np.allclose(out, 1.0, atol=1e-5), out


def test_facilitation_outlasts_the_burst():
    """A transient burst must leave the terminal primed long after it ends.

    This is the whole point of the mechanism: it is what lets a cue be held
    through a delay in which the neuron is silent. Before STP existed, XOR
    decodability from the frozen column at answer time was 0.51 (chance).
    """
    cfg = ColumnConfig(n_neurons=4, n_external=4)
    transport = LocalTransport(4 + 4, cfg.delay_max + 2, cfg.modulator_dim)
    col = Column(cfg, transport)
    baseline = col.u.copy()
    burst, silent = np.ones(4, dtype=bool), np.zeros(4, dtype=bool)
    for _ in range(25):
        col._short_term_plasticity(np.ones(4, dtype=np.float32), burst)
    primed = col.u.copy()
    assert np.all(primed > baseline * 1.2), (baseline, primed)
    for _ in range(250):
        col._short_term_plasticity(np.zeros(4, dtype=np.float32), silent)
    assert np.all(col.u > baseline * 1.05), "facilitation decayed away too fast"


def test_depression_limits_sustained_release():
    """Resources must deplete under sustained firing, bounding runaway drive."""
    cfg = ColumnConfig(n_neurons=4, n_external=4)
    transport = LocalTransport(4 + 4, cfg.delay_max + 2, cfg.modulator_dim)
    col = Column(cfg, transport)
    fired = np.ones(4, dtype=bool)
    first = col._short_term_plasticity(np.ones(4, dtype=np.float32), fired)
    for _ in range(30):
        out = col._short_term_plasticity(np.ones(4, dtype=np.float32), fired)
    assert np.all(out < first), "sustained firing should deplete resources"
    assert np.all(col.res >= 0.0)


def test_stp_state_resets_between_episodes():
    task = DelayedXOR()
    m = Plexus(task.n_inputs, task.n_classes, column=ColumnConfig(n_neurons=32), seed=0)
    m.run_episode(task.episode(np.random.default_rng(0)), learn=False)
    m.column.reset_state()
    assert np.allclose(m.column.u, m.column.cfg.stp_u)
    assert np.allclose(m.column.res, 1.0)


# --------------------------------------------------------------------------
# Learning mechanics.
# --------------------------------------------------------------------------
def test_readout_standardisation_reaches_unit_scale():
    """Standardised features must actually have ~unit scale, on tiny inputs.

    Column traces are small, so an EMA variance seeded at 1.0 stays dominated
    by its own initialisation for thousands of episodes. That produced z with
    a standard deviation of 0.003 and pinned the decoder at chance on data an
    offline decoder read at 0.85 -- with nothing visibly wrong anywhere. The
    input scale here is deliberately tiny to catch exactly that.
    """
    r = LinearReadout(n_inputs=32, n_outputs=2, seed=0)
    rng = np.random.default_rng(0)
    zs = [r.decide(rng.normal(0.0, 1e-3, size=32).astype(np.float32))[1] for _ in range(300)]
    spread = float(np.array(zs[-100:]).std())
    assert 0.4 < spread < 2.5, f"standardised feature spread {spread:.4f} is not unit-scale"


def test_training_metric_does_not_leak_the_label():
    """Votes must be counted before any update on the same episode.

    Regression test for a metric leak that reported 1.000 training accuracy on
    a model whose held-out accuracy was 0.508. The readout was updating during
    the response window, fitting the current episode's label within a few
    steps, after which every remaining vote was trivially correct.
    """
    task = DelayedXOR()
    m = Plexus(task.n_inputs, task.n_classes, column=ColumnConfig(n_neurons=48), seed=0)

    updates: list[int] = []
    original = m.readout.update

    def spy(*args, **kwargs):
        updates.append(m._t)
        return original(*args, **kwargs)

    m.readout.update = spy
    ep = task.episode(np.random.default_rng(0))
    start = m._t
    m.run_episode(ep, learn=True)

    first_response = int(np.argmax(ep.response))
    last_vote = start + first_response + m.answer_steps - 1
    assert updates, "expected the readout to train at all"
    assert min(updates) > last_vote, "readout updated while votes were still being counted"



def test_output_depends_on_synaptic_weights():
    """Perturbing W must change what the column emits.

    Regression test for the worst bug in this model's history: the branch
    integration summed its inputs *unweighted*, so W was read by synaptic
    scaling and by the learning rule but never used to compute anything. The
    network ran on implicit unit weights, learning was a no-op with no visible
    symptom, and a plastic column and a frozen one produced bit-identical
    states while their weight matrices differed by 25%.
    """
    cfg = ColumnConfig(n_neurons=32, n_external=8, seed=0)

    def emissions(scale):
        transport = LocalTransport(8 + 32, cfg.delay_max + 2, cfg.modulator_dim)
        col = Column(cfg, transport)
        col.learning = False
        col.W = col.W * scale
        rng = np.random.default_rng(3)
        return np.array([col.step(t, (rng.random(8) < 0.1).astype(np.float32)) for t in range(400)])

    assert not np.allclose(emissions(1.0), emissions(0.4))


def test_frozen_column_weights_do_not_move():
    """lr=0 must disable the modulator-driven rule entirely.

    Unsupervised synaptic scaling is disabled here too, since it moves weights
    on its own schedule and would otherwise mask the thing being tested. The
    reservoir control in experiments/ablation.py deliberately leaves scaling
    on -- it isolates the three-factor rule, not all plasticity.
    """
    task = DelayedXOR()
    cfg = ColumnConfig(n_neurons=48, lr=0.0, scaling_lr=0.0)
    m = Plexus(task.n_inputs, task.n_classes, column=cfg, seed=0)
    before = m.column.W.copy()
    rng = np.random.default_rng(0)
    for _ in range(3):
        m.run_episode(task.episode(rng), learn=True)
    assert np.array_equal(before, m.column.W)


def test_learning_changes_weights_when_enabled():
    task = DelayedXOR()
    m = Plexus(task.n_inputs, task.n_classes, column=ColumnConfig(n_neurons=48), seed=0)
    before = m.column.W.copy()
    rng = np.random.default_rng(0)
    for _ in range(3):
        m.run_episode(task.episode(rng), learn=True)
    assert not np.allclose(before, m.column.W)


def test_eval_mode_freezes_everything():
    task = DelayedXOR()
    m = Plexus(task.n_inputs, task.n_classes, column=ColumnConfig(n_neurons=48), seed=0)
    rng = np.random.default_rng(0)
    m.run_episode(task.episode(rng), learn=True)
    snapshot = (m.column.W.copy(), m.column.theta.copy(), m.readout.W.copy())
    for _ in range(3):
        m.run_episode(task.episode(rng), learn=False)
    assert np.array_equal(snapshot[0], m.column.W)
    assert np.array_equal(snapshot[1], m.column.theta)
    assert np.array_equal(snapshot[2], m.readout.W)


def test_modulator_lag_does_not_break_the_step_loop():
    """A stale modulator must still be consumed and applied."""
    task = DelayedXOR()
    m = Plexus(
        task.n_inputs, task.n_classes, column=ColumnConfig(n_neurons=48), modulator_lag=150, seed=0
    )
    before = m.column.W.copy()
    rng = np.random.default_rng(0)
    for _ in range(3):
        m.run_episode(task.episode(rng), learn=True)
    assert np.all(np.isfinite(m.column.W))
    assert not np.allclose(before, m.column.W)


# --------------------------------------------------------------------------
# Distribution: the invariants that make splitting across machines safe.
# --------------------------------------------------------------------------
def _distributed(peer_delay=30, seed=0, columns=3, neurons=16, **kw):
    task = DelayedXOR()
    return task, DistributedPlexus(
        task.n_inputs,
        task.n_classes,
        n_columns=columns,
        column=ColumnConfig(n_neurons=neurons, seed=seed, **kw),
        peer_delay=peer_delay,
        seed=seed,
    )


def test_columns_agree_on_which_units_inhibit():
    """Dale signs must be a property of the emitting neuron, not the reader.

    Regression test for a real distribution bug: signs were drawn from each
    column's own seed, so the same neuron could excite one column while
    inhibiting another. No biological network does that, and nothing
    downstream could detect it.
    """
    _task, m = _distributed()
    first = m.columns[0].source_sign
    for col in m.columns[1:]:
        assert np.array_equal(first, col.source_sign)


def test_each_column_publishes_only_what_it_owns():
    """A column must never write into another column's slice."""
    _task, m = _distributed()
    for i, col in enumerate(m.columns):
        assert col.source_offset == m.n_inputs + i * m.per_column
        others = [c for c in m.columns if c is not col]
        for other in others:
            lo, hi = other.source_offset, other.source_offset + m.per_column
            assert not (lo <= col.source_offset < hi)


def test_step_order_does_not_change_the_result():
    """The result must not depend on which column is stepped first.

    Every column publishes before any column reads, and no synapse has a delay
    below one step, so no column can observe another's present. If that holds,
    reversing the step order is invisible -- and if it did not hold, the model
    could not be run on separate machines at all.
    """
    task, forward = _distributed(seed=3)
    _task2, reverse = _distributed(seed=3)
    reverse.columns.reverse()  # step them in the opposite order

    rng_a, rng_b = np.random.default_rng(5), np.random.default_rng(5)
    for _ in range(3):
        forward.run_episode(task.episode(rng_a), learn=False)
        reverse.run_episode(task.episode(rng_b), learn=False)

    got = {c.source_offset: c.out for c in reverse.columns}
    for col in forward.columns:
        assert np.allclose(col.out, got[col.source_offset]), "step order changed the result"


def test_peer_synapses_carry_the_longer_delay():
    """Reaching another column must cost the peer delay, and only that."""
    _task, m = _distributed(peer_delay=40)
    col = m.columns[0]
    own_lo, own_hi = col.source_offset, col.source_offset + m.per_column
    is_peer = (col.src >= m.n_inputs) & ((col.src < own_lo) | (col.src >= own_hi))
    assert is_peer.any(), "expected some cross-column synapses"
    assert col.delay[is_peer].min() >= 40
    local = (col.src >= own_lo) & (col.src < own_hi)
    assert col.delay[local].max() <= col.cfg.delay_max


def test_distributed_model_runs_and_stays_sparse():
    task, m = _distributed(peer_delay=120, columns=3, neurons=16)
    rng = np.random.default_rng(0)
    for _ in range(3):
        loss, correct = m.run_episode(task.episode(rng), learn=True)
        assert np.isfinite(loss)
        assert isinstance(correct, bool)
    assert 0.0 < m.sparsity < 0.25
    assert all(np.all(np.isfinite(c.W)) for c in m.columns)


# --------------------------------------------------------------------------
# Tasks: the benchmarks must not be solvable without temporal structure.
# --------------------------------------------------------------------------
def test_xor_classes_are_balanced_in_total_input():
    """All four cue combinations must deliver the same total drive.

    If they did not, total activity would leak the label and the task would
    measure nothing.
    """
    task = DelayedXOR(noise_rate=0.0)
    rng = np.random.default_rng(0)
    sums = [float(task.episode(rng).inputs.sum()) for _ in range(80)]
    assert np.std(sums) / np.mean(sums) < 0.12


def test_temporal_patterns_are_invisible_to_a_rate_code():
    """Summing over time must destroy the label.

    Every class fires the same channels the same number of times; only the
    timing differs. A per-channel histogram should therefore be identical
    across classes, which is what makes this a genuine test of timing.
    """
    task = TemporalPatterns(n_classes=4, seed=0)
    hists = []
    for channels, _times in task.templates:
        h = np.bincount(channels, minlength=task.n_inputs)
        hists.append(h)
    for h in hists[1:]:
        assert np.array_equal(hists[0], h)


def test_temporal_patterns_differ_in_time():
    task = TemporalPatterns(n_classes=4, seed=0)
    (_c0, t0), (_c1, t1) = task.templates[0], task.templates[1]
    assert not np.array_equal(t0, t1)


def test_episode_shapes_are_consistent():
    for task in (DelayedXOR(), TemporalPatterns()):
        ep = task.episode(np.random.default_rng(0))
        assert ep.inputs.shape == (task.length, task.n_inputs)
        assert ep.response.shape == (task.length,)
        assert ep.response.any()
        assert 0 <= ep.label < task.n_classes


# --------------------------------------------------------------------------
# Audit tests: quantities that could look healthy while measuring nothing.
# Every bug found late in this project had that shape, so these check the
# link between a quantity and what it claims to represent, not just outputs.
# --------------------------------------------------------------------------
def test_eligibility_normaliser_tracks_actual_magnitude():
    """elig_rms must track the trace it normalises by.

    Regression test for a bug that mis-scaled every sweep. It was seeded at
    1e-3 with a 0.999 decay, which was fine while the modulator fired ~70x per
    episode; once one decision produced one release it advanced 70x more slowly
    and stayed dominated by its initialisation, tracking 4.6e-3 against an
    actual 2.4e-2. The effective learning rate ran ~5x high and drifted.
    """
    task = DelayedXOR()
    m = Plexus(task.n_inputs, task.n_classes, column=ColumnConfig(n_neurons=32), seed=0)
    rng = np.random.default_rng(0)
    for _ in range(40):
        m.run_episode(task.episode(rng), learn=True)
    col = m.column
    assert col.n_updates > 0, "normaliser never updated"
    tracked = col.elig_rms / (1.0 - float(col.decay_elig_rms) ** col.n_updates)
    actual = np.sqrt(np.mean(col.elig**2, axis=(1, 2)))
    ratio = float(np.median(actual) / np.median(np.maximum(tracked, 1e-12)))
    assert 0.3 < ratio < 3.0, f"normaliser off by {ratio:.1f}x from the true magnitude"


def test_event_value_variation_comes_from_stp_not_threshold_crossing():
    """Records where an event's value actually comes from.

    The design intent was that suprathreshold magnitude makes an event more
    informative than a spike. Measured, that component uses well under 1% of
    its available span: the soft reset (v -= theta) means v never climbs far
    past threshold, so u stays ~0 and the raw emission is nearly constant.

    The variation is real but it comes from short-term plasticity scaling the
    amplitude -- which is arguably the more biological mechanism, since real
    terminals modulate amplitude through release probability. This test pins
    both halves so that a future change to the emission is noticed rather than
    silently assumed to have been working all along.
    """
    cfg = ColumnConfig(n_neurons=64, n_external=16, seed=0)
    transport = LocalTransport(16 + 64, cfg.delay_max + 2, cfg.modulator_dim)
    col = Column(cfg, transport)
    span = cfg.value_scale - cfg.value_base

    raw, post = [], []
    original = col._short_term_plasticity

    def spy(value, fired):
        raw.append(value[fired].copy())
        out = original(value, fired)
        post.append(out[fired].copy())
        return out

    col._short_term_plasticity = spy
    rng = np.random.default_rng(0)
    for t in range(4000):
        col.step(t, (rng.random(16) < 0.05).astype(np.float32))

    r, p = np.concatenate(raw), np.concatenate(post)
    graded_fraction = float((np.median(r) - cfg.value_base) / span)
    assert graded_fraction < 0.05, (
        f"suprathreshold grading now uses {graded_fraction:.1%} of its span; "
        "if the emission was changed on purpose, update this test and the README claim"
    )
    assert float(p.std() / np.median(p)) > 0.15, "events carry no amplitude variation at all"


def test_learnable_time_constants_run_without_breaking_dynamics():
    """lr_tau is an advertised feature that is off by default and so untested.

    Code that never executes is exactly where the forward-pass bug lived, so
    this at least exercises the path and checks it leaves the column in a sane
    state rather than silently producing NaNs or absurd constants.
    """
    task = DelayedXOR()
    cfg = ColumnConfig(n_neurons=32, lr_tau=1e-3)
    m = Plexus(task.n_inputs, task.n_classes, column=cfg, seed=0)
    before = m.column.tau_soma.copy()
    rng = np.random.default_rng(0)
    for _ in range(8):
        m.run_episode(task.episode(rng), learn=True)
    tau = m.column.tau_soma
    assert np.all(np.isfinite(tau))
    assert np.all(tau > 0.0)
    assert not np.allclose(before, tau), "lr_tau > 0 but time constants never moved"
    assert np.all(np.isfinite(m.column.v)) and np.all(np.isfinite(m.column.W))


# --------------------------------------------------------------------------
# Salience-gated Hebbian binding. What survived sweep 014 -- the engram
# allocator around it (excitability drift, recruitment competition, allocation
# refractory) measured worse than binding alone on 19 of 20 seeds and was
# deleted, so the tests for it went with it. See
# experiments/sweeps/engram-014-is-allocation-necessary.txt.
# --------------------------------------------------------------------------
def _hebb_run(episodes=40, neurons=48, seed=0, **kw):
    task = DelayedXOR()
    cfg = ColumnConfig(n_neurons=neurons, lr=0.0, seed=seed, hebbian=True, **kw)
    m = Plexus(task.n_inputs, task.n_classes, column=cfg, seed=seed)
    before = m.column.W.copy()
    rng = np.random.default_rng(1)
    for _ in range(episodes):
        m.run_episode(task.episode(rng), learn=True)
    return m, before


def test_hebbian_off_leaves_the_column_untouched():
    """The default must reproduce the model without the mechanism exactly.

    New mechanisms default to off so previous measurements stay reproducible.
    A default that quietly changed the dynamics would invalidate fourteen
    sweeps of recorded results without any test failing.
    """
    task = DelayedXOR()
    outs = []
    for hebbian in (False, True):
        cfg = ColumnConfig(n_neurons=32, lr=0.0, seed=0, hebbian=hebbian)
        m = Plexus(task.n_inputs, task.n_classes, column=cfg, seed=0)
        rng = np.random.default_rng(4)
        for _ in range(3):
            m.run_episode(task.episode(rng), learn=True)
        outs.append(m.column.W.copy())
    assert not np.allclose(outs[0], outs[1]), "hebbian=True changed nothing at all"

    cfg = ColumnConfig(n_neurons=32, lr=0.0, seed=0)
    m = Plexus(task.n_inputs, task.n_classes, column=cfg, seed=0)
    rng = np.random.default_rng(4)
    for _ in range(3):
        m.run_episode(task.episode(rng), learn=True)
    assert np.array_equal(m.column.W, outs[0]), "the default path is no longer the old path"


def _bind_column(n=8, **kw):
    cfg = ColumnConfig(n_neurons=n, n_external=4, seed=0, hebbian=True,
                       scaling_lr=0.0, **kw)
    transport = LocalTransport(4 + n, cfg.delay_max + 2, cfg.modulator_dim)
    return Column(cfg, transport)


def test_binding_scales_with_how_active_the_neuron_was():
    """The postsynaptic factor must reach the weights.

    Driven directly rather than through a training run, because a pooled
    statistic cannot see this. Per-neuron weight change varies with `pre` as
    well, so "std/mean across neurons" stays large even when the activity
    factor is replaced by a constant -- the first version of this test passed
    its own mutation for exactly that reason. Holding `pre` identical across
    neurons and varying only activity isolates the factor under test.

    Graded binding is also what sweep 014 bought: recruiting a fixed minority
    and giving them a full-strength update measured 0.830 against 0.876, worse
    on 19 of 20 seeds at p = 0.0001.
    """
    col = _bind_column()
    col.pre[:] = 0.5
    col.act_slow[:] = 0.1
    col.n_samples = 500  # bias correction settled
    col.act_fast[:] = 0.1
    col.act_fast[0] = 0.4  # four times its own baseline
    before = col.W.copy()
    col._bind()
    moved = np.abs(col.W - before).sum(axis=(1, 2))
    assert moved[0] > 2.0 * moved[1:].mean(), (
        f"the four-times-more-active neuron bound {moved[0]:.4f} against "
        f"{moved[1:].mean():.4f} for the rest -- activity is not scaling the update"
    )


def test_binding_strengthens_only_excitatory_synapses():
    """Hebbian LTP is glutamatergic.

    Strengthening an inhibitory synapse that was active during binding would
    make the pattern harder to reactivate -- the opposite of what binding is
    for -- so the sign gate has to hold.

    Synaptic scaling is off here. It rescales every weight on a branch
    multiplicatively, inhibitory ones included, so with it on the inhibitory
    weights move for reasons that have nothing to do with binding and the sign
    gate becomes unobservable. Isolating the mechanism under test from the one
    that renormalises it is the point.
    """
    m, before = _hebb_run(scaling_lr=0.0)
    col = m.column
    delta = col.W - before
    inhibitory = col.syn_sign < 0.0
    assert np.allclose(delta[inhibitory], 0.0), "an inhibitory synapse was potentiated"
    assert delta[~inhibitory].sum() > 0.0, "no excitatory synapse was potentiated"


def test_binding_carries_which_synapses_were_driving_the_neuron():
    """The presynaptic factor must reach the weights.

    Binding proportional to postsynaptic activity alone would strengthen a
    neuron's whole fan-in uniformly, writing no structure at all -- and it is
    invisible to any statistic pooled across neurons, since the activity factor
    still varies between them. So this looks *within* one neuron, where only
    `pre` can differ.
    """
    col = _bind_column()
    col.act_slow[:] = 0.1
    col.act_fast[:] = 0.2
    col.n_samples = 500
    col.pre[:] = 0.1
    col.pre[:, 0, 0] = 1.0  # one synapse per neuron was doing the driving
    before = col.W.copy()
    col._bind()
    delta = col.W - before
    excitatory = col.syn_sign[:, 0, 0] > 0.0
    assert excitatory.any(), "no excitatory synapse in the probed position"
    assert (delta[excitatory, 0, 0] > 3.0 * delta[excitatory, 0, 1]).all(), (
        "the strongly-driving synapse gained no more than its neighbour, so "
        "`pre` is not reaching the update"
    )


def test_binding_rate_decays_when_asked_to():
    """hebb_decay must shrink the update by exactly its own factor per event.

    Sweep 017 found binding and readout adaptation competing on one timescale:
    freezing binding so the readout can converge is worth +0.063 (p = 0.0013)
    while the same mechanism running flat delivers +0.007 (p = 0.68). A decaying
    rate is the proposed fix, so a decay knob that silently did nothing would
    read as a null result about schedules rather than a disconnected parameter.

    Measured as a *ratio between two runs at the same event index*, not as
    flatness within one run. The first version asserted that a run with
    hebb_decay=1.0 produces a constant update, and it failed -- correctly. The
    baseline `act_slow` adapts toward the activity it is scored against, so the
    postsynaptic factor shrinks on its own even at a fixed rate. That is the
    mechanism behaving properly, and the test was wrong about it.

    Both runs here share identical activity, baseline and presynaptic
    trajectories, since none of those depend on hebb_decay. So the ratio at
    event k isolates the rate and must be exactly decay**k.
    """
    def bind_deltas(decay, events=8):
        col = _bind_column(hebb_decay=decay)
        col.act_slow[:] = 0.1
        col.n_samples = 0
        out = []
        for _ in range(events):
            col.act_fast[:] = 0.2
            col.pre[:] = 0.5
            before = col.W.copy()
            col._bind()
            out.append(float(np.abs(col.W - before).sum()))
        return np.array(out)

    decay = 0.8
    flat, decayed = bind_deltas(1.0), bind_deltas(decay)
    assert flat[0] > 0.0, "nothing bound at all"
    expected = decay ** np.arange(len(flat))
    # 2e-3 rather than something tighter because the weights are float32 and
    # the observed ratio lands at 0.8001 against 0.8. That is arithmetic, not
    # slack: the mutation this guards against removes the decay entirely, which
    # puts the ratio at 1.0 and misses by 250x the tolerance.
    assert np.allclose(decayed / flat, expected, rtol=2e-3), (
        f"rate ratio {np.round(decayed / flat, 4)} against expected "
        f"{np.round(expected, 4)}"
    )


def test_probing_does_not_perturb_training():
    """Measuring the column mid-training must leave the training untouched.

    experiments/trajectory.py interleaves probe episodes with training to find
    out *when* binding's gain arrives. A probe that nudged the thing it
    measures would produce a curve describing the probe rather than the
    mechanism -- and this project has already shipped one measurement that
    changed its subject, the training metric that updated the readout
    mid-decision and reported 1.000 accuracy on a model at chance.

    Bit-identical is the right bar here, not approximately equal: probing runs
    with learn=False, so binding, weight updates, homeostasis and the readout's
    own statistics are all gated off, and there is no mechanism by which a
    single bit should move.
    """
    task = DelayedXOR()

    def train(probe_every):
        cfg = ColumnConfig(n_neurons=32, lr=0.0, seed=0, hebbian=True)
        m = Plexus(task.n_inputs, task.n_classes, column=cfg, seed=0)
        rng = np.random.default_rng(1000)
        probe_rng = np.random.default_rng(7)
        for i in range(12):
            m.train(task, 1, rng=rng, report_every=10**9)
            if probe_every and (i + 1) % probe_every == 0:
                for _ in range(3):
                    m.run_episode(task.episode(probe_rng), learn=False)
        return m.column

    plain, probed = train(0), train(2)
    assert np.array_equal(plain.W, probed.W), "probing moved the weights"
    assert np.array_equal(plain.theta, probed.theta), "probing moved the thresholds"
    assert np.array_equal(plain.knee, probed.knee), "probing moved the dendritic knees"
    assert plain.n_samples == probed.n_samples, (
        f"probing caused {probed.n_samples - plain.n_samples} extra binding events"
    )


def test_evaluation_does_not_advance_the_scaling_schedule():
    """`_steps` counts learning steps only, and an odd number of them proves it.

    Separate from test_probing_does_not_perturb_training because that test
    *cannot* see this. `_steps` exists to schedule synaptic scaling every
    `scaling_every` steps, and DelayedXOR episodes are 400 steps against a
    scaling period of 20 -- so any whole number of probe episodes leaves the
    phase exactly where it was, and the leak is invisible through that door.
    It would appear the moment a task's episode length stopped dividing by 20.

    Stepping an odd number of times with learning off is the direct check.
    """
    cfg = ColumnConfig(n_neurons=16, n_external=4, seed=0)
    assert 7 % cfg.scaling_every != 0, "pick a step count that would shift the phase"
    transport = LocalTransport(4 + 16, cfg.delay_max + 2, cfg.modulator_dim)
    col = Column(cfg, transport)
    rng = np.random.default_rng(0)

    for t in range(40):
        col.step(t, (rng.random(4) < 0.2).astype(np.float32))
    learned = col._steps

    col.learning = False
    for t in range(40, 47):
        col.step(t, (rng.random(4) < 0.2).astype(np.float32))
    assert col._steps == learned, (
        f"{col._steps - learned} evaluation steps advanced the scaling schedule"
    )

    col.learning = True
    col.step(47, (rng.random(4) < 0.2).astype(np.float32))
    assert col._steps == learned + 1, "learning steps stopped being counted"


def test_lateral_inhibition_only_moves_inhibitory_synapses():
    """Anti-Hebbian decorrelation is an inhibitory mechanism.

    Moving an excitatory weight here would make co-active units drive each
    other *harder*, which is the opposite of the rule's purpose and would look
    like a decorrelation mechanism while doing correlation.
    """
    cfg = ColumnConfig(n_neurons=32, n_external=8, seed=0, lateral=True, scaling_lr=0.0)
    transport = LocalTransport(8 + 32, cfg.delay_max + 2, cfg.modulator_dim)
    col = Column(cfg, transport)
    before = col.W.copy()
    rng = np.random.default_rng(0)
    for t in range(600):
        col.step(t, (rng.random(8) < 0.2).astype(np.float32))
    delta = col.W - before
    inhibitory = col.syn_sign < 0.0
    assert np.allclose(delta[~inhibitory], 0.0), "an excitatory synapse was moved"
    assert np.abs(delta[inhibitory]).max() > 0.0, "no inhibitory synapse moved"


def test_lateral_inhibition_step_is_scaled_to_the_weights():
    """`lateral_lr` must mean a fraction of the weight scale, not of `pre`.

    Unnormalised, this rule was four orders of magnitude too weak to matter:
    `pre` sits around 0.016, so the raw product moved W by 2.2e-5 against a
    weight scale of 0.375 -- present in the code, absent from the dynamics.
    Exactly the defect the eligibility trace had, where the three-factor rule
    moved weights by 0.1% while synaptic scaling moved them by 21%.

    Asserting on the *size* of the change rather than its existence, because a
    connection test alone passes on 2.2e-5.
    """
    def moved(lr):
        cfg = ColumnConfig(n_neurons=32, n_external=8, seed=0, lateral=True,
                           lateral_lr=lr, scaling_lr=0.0)
        transport = LocalTransport(8 + 32, cfg.delay_max + 2, cfg.modulator_dim)
        col = Column(cfg, transport)
        before = col.W.copy()
        rng = np.random.default_rng(0)
        for t in range(600):
            col.step(t, (rng.random(8) < 0.2).astype(np.float32))
        inhibitory = col.syn_sign < 0.0
        return float(np.abs(col.W - before)[inhibitory].mean() / before[inhibitory].mean())

    rel = moved(ColumnConfig.lateral_lr)
    assert rel > 0.01, (
        f"the default rate moves inhibitory weights by {rel:.2%} of their own "
        "scale, which is too small to change the dynamics"
    )
    # And it must track the rate, not sit at some floor set by something else.
    assert moved(ColumnConfig.lateral_lr * 4) > 2.0 * rel


def test_reported_sparsity_is_measured_and_not_its_own_initialisation():
    """A reporting property must not return the seed of the EMA behind it.

    `rate` is seeded at exactly `target_rate` — deliberately, so the controller
    starts with zero error and there is no startup transient. That makes the
    naive property return "perfectly on target" before a single sample, which is
    the most reassuring value available and completely uninformative.

    It cost a measurement. Sweep 022's trajectory probe reported sparsity 0.0300
    at checkpoint 0 for every condition including ones whose column was silent;
    counted directly, the untrained column runs at 0.0022 — a fifteenth of
    target. The sparsity column existed specifically to catch dead columns and
    was reporting the one number that hides them.

    The bound is tight on purpose: `raw_rate` reads 0.0254 at this point, so a
    test admitting anything up to target would pass on the broken version.
    """
    cfg = ColumnConfig(n_neurons=64, n_external=16, seed=0)
    transport = LocalTransport(16 + 64, cfg.delay_max + 2, cfg.modulator_dim)
    col = Column(cfg, transport)

    assert np.isnan(col.sparsity), "reported a rate before observing any step"
    assert np.isnan(col.plateau_engagement), "reported engagement before any step"

    rng = np.random.default_rng(0)
    for t in range(50):
        col.step(t, (rng.random(16) < 0.02).astype(np.float32))

    assert col.sparsity < 0.2 * cfg.target_rate, (
        f"reported {col.sparsity:.4f} for a column that has barely fired; the "
        f"controller's raw EMA reads {col.raw_rate:.4f} and must not leak here"
    )
    assert col.raw_rate > 0.5 * cfg.target_rate, (
        "the controller's own state should still be seed-dominated here — if "
        "not, this test is no longer separating the two and proves nothing"
    )


def test_every_ema_either_debiases_or_is_named_as_controller_state():
    """Third instance of one bug, so this asserts the class and not the case.

    Twice before: `elig_rms` kept an EMA seeded far from its true value and ran
    the effective learning rate ~5x high for thousands of episodes, and the
    readout standardised with a variance that had not converged. Both were fixed
    by bias-correcting the individual quantity. `rate` and `engagement` then
    repeated it in the reporting path.

    So this enumerates the EMAs instead of trusting the next one to be noticed.
    Every decaying accumulator must either carry a sample counter that lets its
    seed be divided out, or be reachable only through a name that says it is a
    controller's internal state.
    """
    cfg = ColumnConfig(n_neurons=32, n_external=8, seed=0, hebbian=True)
    transport = LocalTransport(8 + 32, cfg.delay_max + 2, cfg.modulator_dim)
    col = Column(cfg, transport)

    # (attribute, the counter that permits debiasing)
    emas = {
        "rate": "_obs_steps",
        "engagement": "_obs_steps",
        "act_slow": "n_samples",
        "elig_rms": "n_updates",
    }
    for name, counter in emas.items():
        assert hasattr(col, name), f"{name} is gone; update this test deliberately"
        assert hasattr(col, counter), (
            f"{name} has no sample counter ({counter}), so its seed cannot be "
            "divided out and anything reading it gets the initialisation back"
        )

    # And the raw controller state stays reachable, under a name that says so —
    # the fix must not hide what the homeostatic rules actually read.
    assert col.raw_rate == float(col.rate.mean())
    assert col.raw_engagement == float(col.engagement.mean())


def test_a_frozen_column_is_identical_across_readout_rules():
    """The README pairs two whole sweeps on this and nothing tested it.

    Sweeps 006 and 010 differ only in the readout rule, and the +0.032 for RLS
    is quoted as a *paired* comparison on the grounds that "a frozen column
    ignores the modulator entirely, so it is bit-identical across the two". That
    is the load-bearing assumption behind the first statistically significant
    positive result in the project.

    It is worth pinning now precisely because sweep 024 showed a frozen column
    is not an unchanging one. The distinction this asserts is the right one: a
    frozen column moves when its *input distribution* changes, and the readout
    rule does not change the input distribution — nothing reaches the column
    from the readout except the modulator, which `lr=0` ignores.
    """
    task = DelayedXOR()
    states, readouts = {}, {}
    for rule in ("delta", "rls"):
        model = Plexus(
            task.n_inputs, task.n_classes,
            column=ColumnConfig(n_neurons=32, lr=0.0, seed=0),
            readout_rule=rule, seed=0,
        )
        model.train(task, 4, rng=np.random.default_rng(1000), report_every=10**9)
        states[rule] = (model.column.W.copy(), model.column.theta.copy(),
                        model.column.knee.copy())
        readouts[rule] = model.readout.W.copy()

    # Non-vacuity: if the two rules produced the same readout the column being
    # identical would prove nothing, because there would be nothing to differ.
    assert not np.allclose(readouts["delta"], readouts["rls"]), (
        "the two readout rules learned the same weights, so this test is "
        "comparing one configuration with itself"
    )
    for name, a, b in zip(("W", "theta", "knee"), states["delta"], states["rls"]):
        assert np.array_equal(a, b), (
            f"the frozen column's {name} differs between readout rules, so the "
            "cross-sweep pairing behind the RLS result is not valid"
        )


def test_a_frozen_column_still_changes_when_the_input_changes():
    """"Frozen" means the learning rules are off, not that nothing moves.

    This project has conflated the two twice. Sweep 019 found that a frozen
    column is not an untrained one — homeostasis alone takes decodability from
    0.582 to 0.779. Sweep 024's experiment then shipped a docstring asserting a
    frozen column cannot forget, and lost 0.167 of task A at the first seed.

    With `lr=0`, no binding and no lateral inhibition, threshold homeostasis,
    knee adaptation and synaptic scaling all keep running. Two input
    distributions that are statistically identical in aggregate still give each
    neuron a different drive through its own fixed wiring, so its threshold
    follows — which is why AUDIT's rule that "a frozen measurement is unaffected
    by a learning-rule bug" is about *learning-rule* bugs specifically and does
    not generalise to "frozen results are stable".
    """
    cfg = ColumnConfig(n_neurons=48, n_external=16, seed=0, lr=0.0)
    transport = LocalTransport(16 + 48, cfg.delay_max + 2, cfg.modulator_dim)
    col = Column(cfg, transport)
    rng = np.random.default_rng(0)

    # Settle on one input distribution, then switch to another with the same
    # marginal rate but different per-channel structure.
    for t in range(1500):
        col.step(t, (rng.random(16) < 0.05).astype(np.float32))
    theta_a, knee_a, w_a = col.theta.copy(), col.knee.copy(), col.W.copy()

    biased = np.r_[np.full(8, 0.12), np.zeros(8)]
    for t in range(1500, 3000):
        col.step(t, (rng.random(16) < biased).astype(np.float32))

    assert not np.allclose(theta_a, col.theta), (
        "thresholds did not move when the input distribution changed, so the "
        "forgetting mechanism sweep 024 measured cannot be threshold drift"
    )
    moved = float(np.abs(col.theta / theta_a - 1.0).mean())
    assert moved > 0.05, f"thresholds moved only {moved:.3f} — too little to forget with"
    assert not np.allclose(knee_a, col.knee), "the knee is inert under an input change"
    assert not np.allclose(w_a, col.W), "synaptic scaling did not move W"


def test_channel_permutation_is_a_relabelling_and_not_a_harder_task():
    """A task stream needs variants that differ in mapping, not in difficulty.

    Continual learning asks whether learning B costs what was learned on A. If
    B is simply harder, forgetting and difficulty are not separable in the
    result and the benchmark answers neither question. A channel permutation is
    the strongest guarantee available: total input energy per timestep is
    *identical*, and the multiset of per-channel totals is identical, so the two
    tasks cannot differ in any quantity that ignores which channel is which.

    The third assertion is the one that stops this being vacuous — the mapping
    must actually differ, or the two tasks are the same task and retention is
    trivially perfect.
    """
    a = DelayedXOR(channel_seed=1)
    b = DelayedXOR(channel_seed=2)
    xa = a.episode(np.random.default_rng(7)).inputs
    xb = b.episode(np.random.default_rng(7)).inputs

    assert np.allclose(xa.sum(1), xb.sum(1)), "per-timestep energy differs"
    assert np.allclose(np.sort(xa.sum(0)), np.sort(xb.sum(0))), (
        "the multiset of per-channel totals differs, so one task carries more "
        "drive on some channel than the other"
    )
    assert not np.allclose(xa.sum(0), xb.sum(0)), (
        "the two variants map cues to the same channels, so they are one task"
    )


def test_channel_permutation_is_off_by_default():
    """Every recorded sweep ran without it, and must keep meaning what it did.

    A permutation applied by default would silently change the task underneath
    every number in `experiments/sweeps/`, and nothing in a result file points
    back at the task that produced it.
    """
    plain = DelayedXOR()
    assert plain.channel_perm is None
    rng_a, rng_b = np.random.default_rng(3), np.random.default_rng(3)
    assert np.array_equal(
        plain.episode(rng_a).inputs, DelayedXOR().episode(rng_b).inputs
    )


def _grown(n_new=8, warmup=400, seed=0):
    cfg = ColumnConfig(n_neurons=48, n_external=16, seed=seed)
    transport = LocalTransport(16 + 48, cfg.delay_max + 2, cfg.modulator_dim)
    col = Column(cfg, transport)
    rng = np.random.default_rng(seed)
    for t in range(warmup):
        col.step(t, (rng.random(16) < 0.15).astype(np.float32))
    before = col.src.copy()
    new_ids = col.add_inputs(n_new)
    return col, before, new_ids, warmup


def test_added_inputs_reach_the_output():
    """Growth that does not change what the column computes is not growth.

    The whole point of `add_inputs` is that a channel added at runtime becomes
    part of the computation. Driving the new channels alone must produce
    activity, and this is the assertion that fails if `add_inputs` rewires
    metadata without the forward pass ever reading it -- the same shape of bug
    as `W` being written by the learning rule and used to compute nothing.
    """
    col, _, _, warmup = _grown()
    n = col.n_inputs

    def drive(mask):
        col.reset_state()
        rng, total = np.random.default_rng(7), 0.0
        for t in range(warmup, warmup + 300):
            e = (rng.random(n) < 0.15).astype(np.float32) * mask
            total += float(col.step(t, e).sum())
        return total

    new_only = drive(np.r_[np.zeros(16), np.ones(n - 16)].astype(np.float32))
    old_only = drive(np.r_[np.ones(16), np.zeros(n - 16)].astype(np.float32))
    silent = drive(np.zeros(n, dtype=np.float32))
    assert silent == 0.0, "the column fired with no input at all"
    assert new_only > 0.0, "the appended channels drive nothing"
    assert old_only > 0.0, "growth silenced the original channels"


def test_adding_inputs_appends_and_never_renumbers():
    """No existing source may change index when the input space grows.

    This is the locality rule, not tidiness. Sources are addressed by global id,
    so inserting channels into the external block would shift every neuron --
    and in a distributed run, every column would have to agree on that shift at
    the same instant. Appending past the neurons costs nothing and lets each
    column rewire whenever it likes.

    Asserted as: every synapse whose source changed now points into the newly
    added range, and nothing points somewhere else that merely happens to work.
    """
    col, before, new_ids, _ = _grown()
    changed = col.src != before
    assert changed.any(), "add_inputs rewired nothing"
    assert np.isin(col.src[changed], new_ids).all(), (
        "a synapse was moved to a source that is not one of the new channels, "
        "so existing ids were renumbered"
    )
    # Every branch must reach the new channels, or some neurons are blind to
    # them while the population average looks connected.
    reaches = np.isin(col.src, new_ids).any(axis=2)
    assert reaches.all(), f"{(~reaches).sum()} branches got no new-channel synapse"


def test_adding_inputs_rebuilds_the_gather_table():
    """The precomputed index encodes both source ids and buffer width.

    `_flat` is built once at construction under the comment "sources and delays
    never change", which `add_inputs` makes false. A stale table does not raise
    -- it silently reads the wrong sources at the wrong offsets, which is the
    exact class of failure that had `W` disconnected for most of this project's
    life.

    Compared against the checked `gather` path, which recomputes from `src` and
    `delay` every step and so cannot go stale.
    """
    fast, _, _, warmup = _grown()
    slow, _, _, _ = _grown()
    slow._fast = False  # force the recomputed path

    rng_a, rng_b = np.random.default_rng(3), np.random.default_rng(3)
    n = fast.n_inputs
    for t in range(warmup, warmup + 200):
        a = fast.step(t, (rng_a.random(n) < 0.15).astype(np.float32))
        b = slow.step(t, (rng_b.random(n) < 0.15).astype(np.float32))
        assert np.array_equal(a, b), f"gather paths diverged at t={t}"


def test_growing_the_event_buffer_preserves_what_is_in_flight():
    """Growth mid-episode must not drop events already emitted.

    Conduction delays are up to 24 steps, so at any moment the buffer holds
    events that have been sent and not yet read. Reallocating without copying
    would silently delete them -- one lost timestep of history across the whole
    population, appearing as a transient the model has no way to attribute.
    """
    buf = EventBuffer(n_sources=4, depth=8)
    buf.write(10, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    buf.grow(6)
    assert np.allclose(buf.gather(13, np.array([0, 2]), np.array([3, 3])), [1.0, 3.0])
    assert np.allclose(buf.gather(13, np.array([4, 5]), np.array([3, 3])), [0.0, 0.0])
    with pytest.raises(ValueError):
        buf.grow(2)


def test_freeze_plasticity_accounts_for_every_mechanism_flag():
    """Adding a mechanism must force a decision about what "frozen" means.

    The sweep 017 catch-up condition cleared `hebbian` by hand, which was
    correct for exactly as long as binding was the only rule writing to W.
    Sweep 020 added lateral inhibition, and every catch-up condition run after
    it would have reported "the readout converged against a static column"
    while the column kept changing underneath it -- the same class of silent
    invalidation as the forward pass that ignored W.

    So this is deliberately structural rather than a list to keep updated: a new
    boolean on ColumnConfig fails here until it is registered in one of the two
    tuples. Enumerating the fields is what makes forgetting impossible.
    """
    from dataclasses import fields

    booleans = {
        f.name for f in fields(ColumnConfig)
        if isinstance(getattr(ColumnConfig, f.name, None), bool)
    }
    accounted = set(PLASTICITY_FLAGS) | set(NON_PLASTICITY_FLAGS)
    assert booleans == accounted, (
        f"unclassified boolean config flags: {sorted(booleans - accounted)}. "
        "Add each to PLASTICITY_FLAGS if it switches on a rule that writes to "
        "W, or to NON_PLASTICITY_FLAGS if it gates dynamics instead."
    )


def test_freezing_plasticity_is_the_same_as_never_enabling_it():
    """Freezing must stop every rule, not the one that was on someone's mind.

    Asserted bit-identically against a column configured without the mechanisms
    from the start, so a rule that keeps writing at a reduced rate fails here
    too. `lr=0` throughout, so the only thing this can be measuring is the two
    unsupervised rules -- and both are on in the condition being frozen.
    """
    def run(freeze: bool):
        cfg = ColumnConfig(n_neurons=32, n_external=8, seed=0, lr=0.0,
                           hebbian=not freeze, lateral=not freeze)
        if freeze:
            # A config that had them on, then frozen, must match one that never
            # did. Set them, then clear them through the helper.
            cfg.hebbian = cfg.lateral = True
            freeze_plasticity(cfg)
        transport = LocalTransport(8 + 32, cfg.delay_max + 2, cfg.modulator_dim)
        col = Column(cfg, transport)
        rng = np.random.default_rng(0)
        for t in range(300):
            col.step(t, (rng.random(8) < 0.2).astype(np.float32))
            if t % 50 == 0:
                transport.broadcast(t, np.ones(cfg.modulator_dim, dtype=np.float32))
                col.apply_modulator(t)
        return col.W.copy()

    frozen = run(freeze=True)
    never = ColumnConfig(n_neurons=32, n_external=8, seed=0, lr=0.0)
    transport = LocalTransport(8 + 32, never.delay_max + 2, never.modulator_dim)
    col = Column(never, transport)
    rng = np.random.default_rng(0)
    for t in range(300):
        col.step(t, (rng.random(8) < 0.2).astype(np.float32))
        if t % 50 == 0:
            transport.broadcast(t, np.ones(never.modulator_dim, dtype=np.float32))
            col.apply_modulator(t)

    assert np.array_equal(frozen, col.W), "a plasticity rule survived the freeze"


def test_binding_statistics_advance_once_per_event():
    """The baseline binding is scored against must match what is scored.

    Accumulating it every timestep compares a neuron's activity at binding time
    against the distribution of all timesteps, most of which are nothing like
    it, so the update tracks which units are phasic rather than what the
    episode engaged. The readout failed in precisely this way once already.
    """
    task = DelayedXOR()
    cfg = ColumnConfig(n_neurons=32, lr=0.0, seed=0, hebbian=True)
    m = Plexus(task.n_inputs, task.n_classes, column=cfg, seed=0)
    rng = np.random.default_rng(0)
    steps = 0
    for _ in range(3):
        ep = task.episode(rng)
        m.run_episode(ep, learn=True)
        steps += ep.inputs.shape[0]
    assert m.column.n_samples == 3, f"{m.column.n_samples} updates for 3 episodes"
    assert m.column.n_samples < steps, "statistics are advancing per timestep"


# --------------------------------------------------------------------------
# Connection tests for every remaining parameter that could be silently
# disconnected. The rule these serve: a mechanism is not tested by running
# without raising, it is tested by perturbing its input and seeing the output
# move. W was read by two subsystems and used by none for the whole first half
# of this project's history.
# --------------------------------------------------------------------------
def _emissions(steps=300, seed=3, n_ext=8, **cfg_kw):
    cfg = ColumnConfig(n_neurons=32, n_external=n_ext, seed=0, **cfg_kw)
    transport = LocalTransport(n_ext + 32, max(cfg.delay_max, 1) + 2, cfg.modulator_dim)
    col = Column(cfg, transport)
    col.learning = False
    rng = np.random.default_rng(seed)
    return np.array(
        [col.step(t, (rng.random(n_ext) < 0.1).astype(np.float32)) for t in range(steps)]
    )


def _weights_after(episodes=6, seed=0, **cfg_kw):
    task = DelayedXOR()
    cfg = ColumnConfig(n_neurons=32, lr=4e-3, seed=seed, **cfg_kw)
    m = Plexus(task.n_inputs, task.n_classes, column=cfg, seed=seed)
    rng = np.random.default_rng(7)
    for _ in range(episodes):
        m.run_episode(task.episode(rng), learn=True)
    return m.column.W.copy()


def test_output_depends_on_branch_gains():
    """G must reach the output, exactly as W must.

    The soma sums branches through G. Nothing else reads it, so if the sum
    dropped the factor the model would run identically and no existing test
    would notice -- which is precisely the shape of the forward-pass bug.
    """
    cfg = ColumnConfig(n_neurons=32, n_external=8, seed=0)

    def run(scale):
        transport = LocalTransport(8 + 32, cfg.delay_max + 2, cfg.modulator_dim)
        col = Column(cfg, transport)
        col.learning = False
        col.G = col.G * scale
        rng = np.random.default_rng(3)
        return np.array(
            [col.step(t, (rng.random(8) < 0.1).astype(np.float32)) for t in range(300)]
        )

    assert not np.allclose(run(1.0), run(0.5)), "branch gains never reached the soma"


def test_somatic_bias_is_inert():
    """`bias` is allocated, added to the drive, and never written by anything.

    Recorded rather than removed, because a reader can reasonably assume a term
    in the soma equation does something. It does not: no rule updates it, so it
    contributes a constant zero. If it is ever wired up, this test should fail
    and be replaced by one that checks it reaches the output.
    """
    task = DelayedXOR()
    m = Plexus(task.n_inputs, task.n_classes, column=ColumnConfig(n_neurons=32), seed=0)
    rng = np.random.default_rng(0)
    for _ in range(4):
        m.run_episode(task.episode(rng), learn=True)
    assert np.array_equal(m.column.bias, np.zeros_like(m.column.bias))


def test_plateau_nonlinearity_reaches_the_output():
    """The dendritic plateau is the model's main departure from a summing unit.

    It has already been disconnected once in a way no test caught: the knee sat
    an order of magnitude above the branch potentials, so phi was linear
    everywhere the column actually operated and the whole nonlinearity was
    decorative.
    """
    assert not np.allclose(_emissions(plateau=0.0), _emissions(plateau=1.2)), (
        "removing the plateau changed nothing -- the nonlinearity is not engaged"
    )


def test_conduction_delays_reach_the_output():
    """Delays are the parameter distribution actually costs, so they must bite."""
    near = _emissions(delay_min=1, delay_max=2)
    far = _emissions(delay_min=20, delay_max=24)
    assert not np.allclose(near, far), "conduction delays do not affect the dynamics"


@pytest.mark.parametrize("mode", ["window", "graded", "hybrid"])
def test_every_surrogate_mode_changes_the_update(mode):
    """Each surrogate must produce a different weight trajectory.

    Three named options that all computed the same thing would be three ways of
    running one experiment while believing they were three.
    """
    baseline = _weights_after(surrogate="window")
    other = _weights_after(surrogate=mode)
    if mode == "window":
        assert np.array_equal(baseline, other)
    else:
        assert not np.allclose(baseline, other), f"surrogate={mode} matches window exactly"


def test_eligibility_modes_change_the_update():
    """sign-only updates must actually differ from magnitude updates."""
    assert not np.allclose(
        _weights_after(elig_mode="magnitude"), _weights_after(elig_mode="sign")
    )


@pytest.mark.parametrize("norm", ["neuron", "none"])
def test_eligibility_normalisers_change_the_update(norm):
    """The normaliser choice is documented as consequential; verify that it is.

    It is also the knob whose miscalibration ran the effective learning rate
    ~5x high for thousands of episodes, so a silent no-op here would be
    expensive.
    """
    assert not np.allclose(_weights_after(elig_norm="column"), _weights_after(elig_norm=norm))


def test_dfa_feedback_leaves_the_columns_own_projection_alone():
    """The two feedback modes must genuinely differ in what crosses the wire.

    `symmetric` broadcasts the readout matrix and overwrites each neuron's
    projection; `dfa` sends only the error and the column keeps the random
    projection it was born with. If feedback_matrix() returned the same thing
    either way, the cheaper mode would be a fiction.
    """
    task = DelayedXOR()
    models = {}
    for mode in ("symmetric", "dfa"):
        cfg = ColumnConfig(n_neurons=32, seed=0)
        m = Plexus(task.n_inputs, task.n_classes, column=cfg, feedback_mode=mode, seed=0)
        models[mode] = (m, m.column.feedback.copy())
        m.run_episode(task.episode(np.random.default_rng(0)), learn=True)

    sym, sym_born = models["symmetric"]
    dfa, dfa_born = models["dfa"]
    assert dfa.readout.feedback_matrix() is None
    assert np.array_equal(dfa.column.feedback, dfa_born), "dfa overwrote the column's projection"
    assert not np.allclose(sym.column.feedback, sym_born), "symmetric never sent the matrix"


@pytest.mark.parametrize("rule", ["delta", "rls", "rls_block", "rls_diag"])
def test_every_readout_rule_reduces_loss(rule):
    """All four rules are selectable; all four must actually learn.

    rls_diag reads plausibly and measured at chance (0.500) on the end-to-end
    task, which is the kind of result that could mean 'this variant is weak' or
    'this variant is broken'. A rule that cannot fit trivially separable data
    is broken, and that is worth being able to tell apart.
    """
    rng = np.random.default_rng(0)
    n_in = 24
    W_true = rng.normal(size=n_in)
    readout = LinearReadout(n_in, 2, lr=0.5, rule=rule, rls_block=6, seed=0)
    losses = []
    for i in range(400):
        x = rng.normal(size=n_in).astype(np.float32)
        target = int(x @ W_true > 0)
        logits, z = readout.decide(x)
        err, loss = readout.error(logits, target)
        losses.append(loss)
        readout.update(err, z, target)
    early, late = float(np.mean(losses[:100])), float(np.mean(losses[-100:]))
    assert late < early, f"rule={rule} did not reduce loss ({early:.3f} -> {late:.3f})"


def test_synaptic_scaling_converges_to_the_branch_budget():
    """Scaling advertises a per-branch L1 target; check it is actually reached."""
    cfg = ColumnConfig(n_neurons=32, n_external=8, branch_budget=6.0, scaling_lr=5e-2)
    transport = LocalTransport(8 + 32, cfg.delay_max + 2, cfg.modulator_dim)
    col = Column(cfg, transport)
    rng = np.random.default_rng(1)
    for t in range(2000):
        col.step(t, (rng.random(8) < 0.1).astype(np.float32))
    norms = np.abs(col.W).sum(axis=2)
    assert np.allclose(norms, cfg.branch_budget, rtol=0.05), (
        f"branch L1 norms span {norms.min():.2f}-{norms.max():.2f}, budget {cfg.branch_budget}"
    )


def test_inhibitory_fraction_matches_the_config():
    """E/I balance is a stated design parameter, so it should be the stated value."""
    cfg = ColumnConfig(n_neurons=100, n_external=8, inhibitory_frac=0.2)
    transport = LocalTransport(8 + 100, cfg.delay_max + 2, cfg.modulator_dim)
    col = Column(cfg, transport)
    units = col.source_sign[cfg.n_external :]
    assert np.isclose((units < 0).mean(), 0.2)


def test_peer_fraction_matches_the_config():
    """`peer_frac` sets how much traffic crosses the network; verify the wiring.

    This is the parameter the whole distribution argument rests on -- bandwidth
    per machine scales with it directly -- so it should be the number it says
    it is, not merely nonzero.
    """
    model = DistributedPlexus(
        n_inputs=8, n_classes=2, n_columns=4,
        column=ColumnConfig(n_neurons=16, peer_frac=0.5, fanin_recurrent_frac=1.0),
        peer_delay=5, seed=0,
    )
    col = model.columns[1]
    lo, hi = col.source_offset, col.source_offset + col.cfg.n_neurons
    recurrent = col.src >= 8
    remote = recurrent & ((col.src < lo) | (col.src >= hi))
    share = remote.sum() / recurrent.sum()
    assert 0.3 < share < 0.6, f"{share:.2f} of recurrent synapses are remote, expected ~0.5"


def test_modulator_actually_arrives_late_when_lagged():
    """modulator_lag must delay delivery, not merely be accepted as an argument.

    Latency tolerance is the project's headline claim and this is the knob that
    tests it. A lag that was silently ignored would make every latency result a
    measurement of zero delay.
    """
    transport = LocalTransport(n_sources=4, depth=8, modulator_dim=2, modulator_lag=3)
    transport.broadcast(10, np.array([1.0, -1.0], dtype=np.float32))
    assert not np.any(transport.modulator(10)), "modulator arrived on the step it was sent"
    assert not np.any(transport.modulator(12)), "modulator arrived before its lag elapsed"
    assert np.allclose(transport.modulator(13), [1.0, -1.0]), "modulator never arrived"


def test_frozen_column_is_unaffected_by_modulator_lag():
    """A column that never reads the modulator must not notice its delay.

    `apply_modulator` returns immediately when learning is off, so at lr=0 the
    lag is unobservable in principle. It was observable in practice, and not
    through any leak: the episode's drain tail is as long as the lag, and
    homeostasis keeps adapting through those extra silent steps. A 200-step lag
    added 24,000 silent steps over 120 episodes and moved a *frozen* column's
    threshold from 0.295 to 0.231, its sparsity from 0.0289 to 0.0189 and its
    plateau engagement from 0.111 to 0.0926 -- worth -0.062 end-to-end accuracy
    on 17/20 paired seeds at p = 0.0011.

    That made every lag comparison a comparison of two operating points as well
    as two lags. Holding `drain_steps` constant separates them, and this asserts
    the separation is exact rather than merely smaller.
    """
    task = DelayedXOR()

    def train(lag):
        cfg = ColumnConfig(n_neurons=24, lr=0.0, seed=0)
        m = Plexus(
            task.n_inputs, task.n_classes, column=cfg,
            modulator_lag=lag, drain_steps=200, seed=0,
        )
        rng = np.random.default_rng(1000)
        for _ in range(12):
            m.run_episode(task.episode(rng), learn=True)
        return m.column

    a, b = train(0), train(200)
    assert np.array_equal(a.theta, b.theta), "thresholds diverged"
    assert np.array_equal(a.knee, b.knee), "dendritic knees diverged"
    assert np.array_equal(a.W, b.W), "weights diverged"
    assert a._steps == b._steps, f"{a._steps} steps vs {b._steps}"


def test_drain_shorter_than_the_lag_is_rejected():
    """Silently discarding every learning signal must not be reachable."""
    task = DelayedXOR()
    with pytest.raises(ValueError, match="discarding"):
        Plexus(task.n_inputs, task.n_classes, modulator_lag=200, drain_steps=50, seed=0)


def test_frozen_column_is_unaffected_by_modulator_lag():
    """A column that never reads the modulator must not notice its delay.

    `apply_modulator` returns immediately when learning is off, so at lr=0 the
    lag is unobservable in principle. It was observable in practice, and not
    through any leak: the episode's drain tail is as long as the lag, and
    homeostasis keeps adapting through those extra silent steps. A 200-step lag
    added 24,000 silent steps over 120 episodes and moved a *frozen* column's
    threshold from 0.295 to 0.231, its sparsity from 0.0289 to 0.0189 and its
    plateau engagement from 0.111 to 0.0926 -- worth -0.062 end-to-end accuracy
    on 17/20 paired seeds at p = 0.0011.

    That made every lag comparison a comparison of two operating points as well
    as two lags. Holding `drain_steps` constant separates them, and this asserts
    the separation is exact rather than merely smaller.
    """
    task = DelayedXOR()

    def train(lag):
        cfg = ColumnConfig(n_neurons=24, lr=0.0, seed=0)
        m = Plexus(
            task.n_inputs, task.n_classes, column=cfg,
            modulator_lag=lag, drain_steps=200, seed=0,
        )
        rng = np.random.default_rng(1000)
        for _ in range(12):
            m.run_episode(task.episode(rng), learn=True)
        return m.column

    a, b = train(0), train(200)
    assert np.array_equal(a.theta, b.theta), "thresholds diverged"
    assert np.array_equal(a.knee, b.knee), "dendritic knees diverged"
    assert np.array_equal(a.W, b.W), "weights diverged"
    assert a._steps == b._steps, f"{a._steps} steps vs {b._steps}"


def test_drain_shorter_than_the_lag_is_rejected():
    """Silently discarding every learning signal must not be reachable."""
    task = DelayedXOR()
    with pytest.raises(ValueError, match="discarding"):
        Plexus(task.n_inputs, task.n_classes, modulator_lag=200, drain_steps=50, seed=0)
