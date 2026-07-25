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
# Engram allocation. Every test here exists because the mechanism failed the
# property it checks during development -- the whole point of building the
# diagnostics before believing the mechanism.
# --------------------------------------------------------------------------
def _engram_model(neurons=64, episodes=0, seed=0, **kw):
    task = DelayedXOR()
    cfg = ColumnConfig(n_neurons=neurons, lr=0.0, seed=seed, engram=True, **kw)
    model = Plexus(task.n_inputs, task.n_classes, column=cfg, seed=seed)
    rng = np.random.default_rng(1000 + seed)
    for _ in range(episodes):
        model.run_episode(task.episode(rng), learn=True)
    return task, model, rng


def test_engram_off_leaves_the_column_untouched():
    """The default must be bit-identical to the model without the mechanism.

    New mechanisms default to off so that every previous measurement stays
    reproducible; a default that quietly changed the dynamics would invalidate
    eleven sweeps of recorded results without any test failing.
    """
    task = DelayedXOR()
    outs = []
    for engram in (False, True):
        cfg = ColumnConfig(n_neurons=32, lr=0.0, seed=0, engram=engram)
        m = Plexus(task.n_inputs, task.n_classes, column=cfg, seed=0)
        rng = np.random.default_rng(4)
        for _ in range(3):
            m.run_episode(task.episode(rng), learn=True)
        outs.append(m.column.W.copy())
    assert not np.allclose(outs[0], outs[1]), "engram=True changed nothing at all"

    cfg = ColumnConfig(n_neurons=32, lr=0.0, seed=0)
    m = Plexus(task.n_inputs, task.n_classes, column=cfg, seed=0)
    rng = np.random.default_rng(4)
    for _ in range(3):
        m.run_episode(task.episode(rng), learn=True)
    assert np.array_equal(m.column.W, outs[0]), "the default path is no longer the old path"


def test_excitability_reaches_the_threshold_when_that_path_is_enabled():
    """excite_threshold > 0 must actually change what the column emits.

    The disconnected-quantity check, in the form that caught the forward-pass
    bug: perturb the input, assert the output moves. The path is off by default
    -- see the test below for why -- and code that never executes is exactly
    where that bug lived, so it gets exercised explicitly.
    """
    task = DelayedXOR()

    def emissions(bump):
        cfg = ColumnConfig(
            n_neurons=32, lr=0.0, seed=0, engram=True, excite_drift=0.0,
            excite_threshold=1.0,
        )
        m = Plexus(task.n_inputs, task.n_classes, column=cfg, seed=0)
        m.column.xi[:16] = bump
        rng = np.random.default_rng(5)
        m.run_episode(task.episode(rng), learn=False)
        return m.column.rate.copy()

    quiet, excited = emissions(0.0), emissions(1.0)
    assert not np.allclose(quiet, excited), "excitability never reached the threshold"
    assert excited[:16].mean() > quiet[:16].mean(), "raising xi did not raise activity"


def test_default_keeps_excitability_out_of_the_threshold():
    """Excitability must not tilt the threshold by default, and here is why.

    Driving a neuron harder depletes its short-term synaptic resources, and the
    emitted value is scaled by what remains. So a neuron made more excitable
    fires more often but reports *less* activity per event, and the activity
    z-score the recruitment competition reads goes the wrong way: measured
    correlation between the excitability bias and that z-score was -0.41. The
    two ingredients of allocation were cancelling each other, and recruitment
    anti-predicted excitability at -0.22. Routing excitability only into the
    competition flips that to +0.20.

    Biology has no reason to separate the two paths. We do, and the reason is a
    property of our emission model rather than of neurons.
    """
    assert ColumnConfig.excite_threshold == 0.0
    task = DelayedXOR()
    cfg = ColumnConfig(n_neurons=32, lr=0.0, seed=0, engram=True)
    m = Plexus(task.n_inputs, task.n_classes, column=cfg, seed=0)
    m.column.xi[:16] = 2.0
    m.run_episode(task.episode(np.random.default_rng(5)), learn=False)
    assert np.array_equal(m.column.theta_eff, m.column.theta), (
        "excitability is still reaching the threshold at the default setting"
    )


def test_recruitment_is_biased_by_excitability():
    """Regression test for a criterion blind to its own driving variable.

    The first version recruited on ``act_fast > margin * act_slow``. That ratio
    is self-normalising: a neuron made more excitable raises its own baseline
    along with its activity, so the criterion cancelled exactly the effect it
    was meant to detect, and excitability correlated with recruitment at -0.40,
    i.e. backwards. Any future rewrite of the competition has to keep this.
    """
    task, model, rng = _engram_model(neurons=64, episodes=40)
    col = model.column
    col.xi.fill(0.0)
    col.xi_slow.fill(0.0)
    half = col.cfg.n_neurons // 2
    col.xi[:half] = 1.5

    tagged = np.zeros(col.cfg.n_neurons)
    trials = 12
    for _ in range(trials):
        col.xi[:half] = 1.5  # hold the bump against drift and allocation drops
        col.xi[half:] = 0.0
        model.run_episode(task.episode(rng), learn=True)
        tagged += col.tag
    assert tagged[:half].sum() > tagged[half:].sum(), (
        f"excitable half recruited {tagged[:half].sum():.0f} times vs "
        f"{tagged[half:].sum():.0f} for the rest -- recruitment ignores excitability"
    )


def test_allocation_recruits_a_minority():
    """An engram that is the whole population has allocated nothing."""
    task, model, rng = _engram_model(neurons=64, episodes=60, alloc_frac=0.15)
    sizes = []
    for _ in range(40):
        model.run_episode(task.episode(rng), learn=True)
        sizes.append(model.column.engram_size)
    mean = float(np.mean(sizes))
    assert 0.02 < mean < 0.45, f"recruited {mean:.1%} of the population per event"


def test_recruitment_lowers_excitability():
    """The allocation refractory has to be connected to the tag.

    This is the step with no analogue in a conventional network: being
    recruited must cost excitability, or successive memories pile onto the same
    neurons instead of being allocated to different ones.
    """
    task, model, rng = _engram_model(neurons=64, episodes=30)
    col = model.column
    before = col.xi.copy()
    model.run_episode(task.episode(rng), learn=True)
    tag = col.tag
    assert tag.any(), "nothing was recruited, so the test proves nothing"
    drop = before - col.xi
    assert drop[tag].mean() > drop[~tag].mean() + 0.1, (
        "recruited neurons did not lose excitability relative to the rest"
    )


def test_hebbian_binding_only_touches_recruited_neurons():
    """Binding must be gated by the tag, and must reach W.

    Checked on the *rows* of W rather than its mean: synaptic scaling holds
    each branch to a fixed L1 budget, so mean |W| is pinned to
    branch_budget / n_synapses no matter what the Hebbian term does. A test on
    the mean would pass identically with the binding removed.
    """
    task, model, rng = _engram_model(neurons=64, episodes=30, hebb_lr=0.05)
    col = model.column
    col.cfg.scaling_lr = 0.0  # isolate binding from the renormalisation
    before = col.W.copy()
    model.run_episode(task.episode(rng), learn=True)
    tag = col.tag
    assert tag.any() and not tag.all(), "need a partial engram for this test"
    moved = np.abs(col.W - before).sum(axis=(1, 2))
    assert moved[tag].mean() > 0.0, "binding never reached W"
    assert np.allclose(moved[~tag], 0.0), "unrecruited neurons had their weights changed"


def test_recruitment_statistics_advance_once_per_allocation():
    """The baseline recruitment is scored against must match what is scored.

    Accumulating it every timestep scores a neuron's activity at recruitment
    time against the distribution of all timesteps -- most of which are nothing
    like it -- so units that reliably answer the go cue look extraordinary at
    every allocation and the criterion measures phasicness rather than what the
    episode engaged. The readout failed in precisely this way once already.
    """
    task, model, rng = _engram_model(neurons=32, episodes=0)
    col = model.column
    steps = 0
    for _ in range(3):
        ep = task.episode(rng)
        model.run_episode(ep, learn=True)
        steps += ep.inputs.shape[0]
    assert col.n_samples == col.n_allocations, (
        f"{col.n_samples} statistics updates for {col.n_allocations} allocations"
    )
    assert col.n_samples < steps, "statistics are advancing per timestep, not per allocation"


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


def _bind_run(bind_mode, episodes=40, neurons=48, seed=0):
    task = DelayedXOR()
    cfg = ColumnConfig(n_neurons=neurons, lr=0.0, seed=seed, engram=True, bind_mode=bind_mode)
    m = Plexus(task.n_inputs, task.n_classes, column=cfg, seed=seed)
    before = m.column.W.copy()
    rng = np.random.default_rng(1)
    for _ in range(episodes):
        m.run_episode(task.episode(rng), learn=True)
    return m.column, before


def test_graded_binding_reaches_every_neuron():
    """"graded" must genuinely drop the competition, not just soften it.

    It is the ablation that could delete the entire engram mechanism, so it has
    to actually be the thing it claims: plain Hebbian LTP gated by salience,
    with no recruitment anywhere. If it quietly kept tagging, the ablation
    would be comparing the mechanism against itself.
    """
    col, before = _bind_run("graded")
    moved = np.abs(col.W - before).sum(axis=(1, 2)) > 0.0
    assert moved.all(), f"only {moved.mean():.0%} of neurons bound; this is still a competition"
    assert col.engram_size == 0.0, "graded mode reported an engram it did not allocate"

    col, before = _bind_run("tagged")
    moved = np.abs(col.W - before).sum(axis=(1, 2)) > 0.0
    assert moved.any(), "tagged mode bound nothing at all"


def test_binding_modes_apply_the_same_total_weight_change():
    """The two modes must differ in mechanism, not in learning rate.

    In tagged mode a fraction `alloc_frac` of neurons receive a unit-weighted
    update; graded mode scales its activity ratio by `alloc_frac` so the mean
    increment matches. Without that the ablation would vary two things at once
    and its outcome would say nothing about which one mattered -- the same
    defect that made sweep 013 uninterpretable.
    """
    tagged, t0 = _bind_run("tagged")
    graded, g0 = _bind_run("graded")
    dt = float(np.abs(tagged.W - t0).mean())
    dg = float(np.abs(graded.W - g0).mean())
    assert dt > 0 and dg > 0
    assert abs(dt - dg) / max(dt, dg) < 0.25, (
        f"tagged moved weights by {dt:.5f} and graded by {dg:.5f}; the modes are "
        "not scale-matched, so a difference between them would be confounded"
    )
