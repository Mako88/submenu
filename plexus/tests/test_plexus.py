"""Tests for the invariants the design actually depends on."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plexus import Column, ColumnConfig, DelayedXOR, EventBuffer, LocalTransport, Plexus
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
    col = _settle()
    assert np.all(np.isfinite(col.v))
    assert np.abs(col.v).max() < 50.0


def test_plateau_stays_engaged():
    """The dendritic nonlinearity must operate inside its useful band.

    Regression test for a real bug: with an absolute knee the branch potential
    sat two orders of magnitude below it and the nonlinearity never fired at
    all, silently reducing every neuron to a linear summer.
    """
    col = _settle()
    assert 0.01 < col.plateau_engagement < 0.5, col.plateau_engagement


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


def test_weights_stay_bounded():
    task = DelayedXOR()
    cfg = ColumnConfig(n_neurons=48, lr=2e-2)
    m = Plexus(task.n_inputs, task.n_classes, column=cfg, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(6):
        m.run_episode(task.episode(rng), learn=True)
    assert np.all(np.isfinite(m.column.W))
    assert m.column.W.max() <= cfg.weight_max + 1e-5


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

    def spy(err, state):
        updates.append(m._t)
        return original(err, state)

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
