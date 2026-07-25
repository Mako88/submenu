# What is outstanding

Ordered by what would change the most if it turned out differently, not by
effort. Each item says what it is, why it matters, and what would settle it —
because "improve the scheduler" is not a task and "does binding still pay when
the readout is not the bottleneck" is.

Nothing here is a claim. Where a number appears it points at the sweep that
measured it.

---

## 1. Does any of this survive a task that asks more?

**The most important open question, and the one nothing so far addresses.**

Delayed XOR has one episode, one answer, and no interference between memories.
That makes it a clean instrument and a narrow one: it cannot ask about
capacity, cannot ask about forgetting, and cannot ask whether structure
emerges. Every number in this repo is a number about that one task.

Three benchmarks would each open something currently invisible:

- **Capacity scaling.** Does the binding gain grow, hold or vanish with column
  size? Measured only at 96 neurons. If it vanishes at 1024, the mechanism is
  a small-column artefact and everything downstream of it changes.
- **Continual learning.** A stream of tasks rather than one. This is where the
  deleted engram allocator could earn its way back: the allocation refractory
  demonstrably produced engrams overlapping *less* than independent sampling
  (`engram-012`, rotate 0.815 against 1.661), and delayed XOR never asks for
  that. Sweep 014 refuted allocation *on this benchmark*, which is not the same
  as refuting it.
- **Emergent structure.** The original motivation and still entirely
  unmeasured. Needs a metric before it needs an experiment.

## 2. What is lateral inhibition actually doing?

Built to decorrelate. Sweep 020 measured it not decorrelating — mean
correlation moves −0.004 alone and not at all on top of binding, effective rank
moves *down* rather than up — while raising linear decodability by +0.028 alone
and +0.037 on top of binding (p = 0.0005). It is kept on the measurement, and
its explanation is open.

The decorrelation reasoning that put it here is refuted and recorded as such:
what works is **consolidation**. Binding raises correlation by +0.077 and halves
effective rank while gaining +0.074, on 20/20 and 0/20 seeds respectively. Sweep
011's finding stands — RLS gains by pooling correlations — but "therefore the
column needs decorrelating" does not follow from it.

Two things are open, in order:

- **Does it pay end to end?** Sweep 021, running. `latbind-on` against `rls-on`
  asks whether a local rule under the shipped delta readout reaches what a
  pooled correlation matrix reaches. Sweep 016 is the reason not to assume:
  binding's +0.074 offline arrived as +0.007. The prediction, recorded before
  the run, is that it does not, and names closing the direction as the
  consequence.
- **What is the mechanism?** Sparsity is flat (0.031 → 0.034), so it is not a
  gross activity change. Candidates worth a direct probe: per-unit dynamic
  range, which units respond at all rather than how they covary, or a second
  consolidating force by another route — which the effective-rank direction
  mildly favours. None is measured; none should be written down until one is.

## 3. Why does homeostatic settling alone buy +0.197?

Surfaced by sweep 019 and never asked about. A frozen column climbs from 0.582
to 0.779 decodability over its first fifty episodes with no learning rule of
any kind running — threshold and knee adaptation alone. That is more than twice
what salience-gated Hebbian binding adds on top, and nothing in this repo
explains it or has tried to.

If unsupervised homeostasis is doing most of the representational work, that is
worth knowing before more effort goes into rules layered above it.

## 4. Event-driven execution

The loop is clock-driven: every neuron updates every millisecond whether or not
anything reached it. At a 3% firing rate that is ~97% waste, and it is the
single largest lever available.

It is *exact* for this model rather than an approximation, for two specific
reasons: every filter is a pure exponential, so state can be caught up over a
gap in closed form (`b *= decay**Δt`); and decay is monotone downward, so a
neuron cannot cross threshold without input arriving. There is no "woke up and
fired spontaneously" case to miss.

Measured on this machine: ~64M synapse-updates/second/core, which is **~500
neurons in real time per core**. Event-driven should move that to ~25,000.

Watch for: homeostasis runs on a slow clock and would need catching up lazily
too, and `_steps` now counts learning steps only (see `column.py`), which an
event-driven rewrite must preserve rather than rediscover.

## 5. `Column.add_inputs()` — shipped, with one thing still open

In the library with four tests and three mutations. A settled 96-neuron column
grown from 16 to 24 channels re-settles sparsity 0.0344 → 0.0303 on its own,
and both old and new channels drive the output (153.9 and 59.8 against 0.0 with
everything silent).

Cheap for structural reasons worth keeping: there is no input-shaped weight
matrix (each synapse names its source by index), the readout reads neurons
rather than inputs so the output layer is untouched, homeostasis absorbs the
drive change locally, synaptic scaling makes new synapses compete rather than
add, and external inputs are always excitatory so Dale's law does not enter.

**Append, do not insert.** New ids go past the neurons, not at the end of the
external block — widening that block in place would shift every neuron's index,
which in a distributed run means every column agreeing on the shift
simultaneously. That is a global synchronisation event and violates the design
rule. `test_adding_inputs_appends_and_never_renumbers` asserts it, and
`EventBuffer.grow` refuses to shrink for the same reason.

Fan-in is fixed at `(N, B, S)`, so growth is **rewiring, not accretion**: a new
channel earns its way in by displacing an existing connection, which is also
what stops drive from growing without bound as inputs are added.

Still open, and the important part: **whether a *trained* model's accuracy
survives the growth.** Everything above was measured on a settled random
column, and homeostasis recovering is not the same as the learned function
surviving. That needs a sweep, not a probe.

## 6. The audit backlog

From `experiments/sweeps/AUDIT.md`. These are live decisions resting on
evidence that a later fix invalidated.

| item | why it matters |
|---|---|
| `modulator_lag` at 20 seeds, `drain_steps` pinned | **The architecture's headline claim.** That eligibility traces absorb a late modulator has never been measured — sweep 013 tried and was confounded by the drain tail. Currently an argument, not a result. |
| `PRETRAIN=0` | Rests solely on sweep 004, which ran before the eligibility-normaliser fix with the effective learning rate ~5× high. The reasoning behind pretraining was sound and it has never been tested under a correct rate. |
| `elig_gate=1.0` | Expressed as a multiple of the trace RMS, chosen when that RMS was tracking 4.6e-3 against an actual 2.4e-2. The number was picked against a scale that no longer exists. |
| `tau_eligibility=70` | Matched to a readout that decided on a window average. It now decides on the trace itself. The justification no longer describes the thing it was matched to. |
| `elig_norm="column"` | The docstring argues for it at length; nothing measured which of the three normalisers is better, only that they differ. |

Divergent defaults, where the library ships one thing and every experiment runs
another: `surrogate` is `"window"` (the version a fix was written *against*)
while experiments use `"graded"`; `lr` is 0.004 against a swept 0.005; and
`n_neurons=256` has never been measured at any point in this project.

`self.bias` is allocated, added to the somatic drive, and written by nothing.
A constant zero, recorded in a test rather than deleted so no reader assumes a
term in the soma equation is doing something.

## 7. Distribution, for real

`NetworkTransport` does not exist. Everything distributed has been measured
through `LocalTransport` with simulated delays, which is the right way to
develop it and not the same as having done it.

- **Wiring locality is the make-or-break number.** Per machine, 33M neurons at
  2% and 1 kHz emit ~6.7×10⁸ events/second. At 1% of synapses crossing the
  network that is ~53 MB/s, which home broadband carries; at 10% it is 530 MB/s,
  which it does not. `peer_frac` sets this directly and has never been swept
  against a bandwidth budget.
- **Batch events over the latency budget.** 150 ms of tolerance is measured and
  free, so ~100 ms of events can go in one packet. At 1 ms granularity, headers
  would swamp 4-byte payloads; batching amortises them over tens of thousands
  of events. Latency tolerance is not just survivable, it is what makes the
  packet economics work.
- **Memory is the binding constraint, not compute.** 16 bytes per synapse
  (`W` plus three traces). Cortex-scale is ~2 PB, so ~31,000 machines just to
  hold state. Trimming per-synapse state — fp16, or traces kept only for
  recently-active synapses — buys more than any compute optimisation.
- **Churn as an ordinary event.** A machine leaving should be a lesion, not a
  crash. This falls out of having no synchronisation barrier and has never been
  tested, because nothing has ever actually left.

## 8. The honest counter-argument

The model is numpy on CPU and largely memory-bandwidth-bound on the `(N,B,S)`
tensors. A single modern GPU would likely beat hundreds of CPU cores per
dollar. That pulls directly against the distribution thesis, and it deserves a
measurement rather than a preference — if a GPU wins by 50×, "no data centres"
needs to be argued on grounds other than cost.

---

## Housekeeping

- **Poll cadence from condition count.** Sweep wall clock per *seed*, measured:
  2 conditions → 3.5 min, 4 → 8, 10 → 11, and 12 → **15-17 min** once the
  trajectory probe was added. It is not a minute per condition — a condition
  that trains 300 episodes costs far more than one that evaluates. Estimate
  from what each condition *does*, not from how many there are.
- **Retire conditions once their sweep is written up.** Sweep 018's three
  schedule conditions and 019's trajectory probe kept running for two sweeps
  after their questions were answered, costing roughly six minutes of every
  seed to reproduce numbers already in the notes. Sweep 021 dropped them. The
  sweep notes are the record; CI is for open questions.
- **Workflow triggers are all sentinels now** (`experiments/run-*.txt`).
  Triggering on the sweep notes meant that *recording a result* re-ran the
  matrix that produced it — GitHub path filters cannot distinguish a file being
  added from one being edited, so the trigger has to live where results are
  never written. Cost ~11 minutes of CI per write-up before it was noticed.
- **`experiments/mutation.py` has caught its own staleness four times.** When a
  refactor moves a line a mutation targets, it reports "source moved" and
  fails rather than going quietly green. Keep that property in any rewrite.
