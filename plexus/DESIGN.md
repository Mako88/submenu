# plexus — design notes

## The premise

Brains are evolved, so they carry constraints that are not about computation at
all: axons attenuate, ion channels have fixed kinetics, neurotransmitters
diffuse, wiring costs volume and metabolism. Copying those faithfully is
expensive and buys nothing. The goal here is to keep what a neuron *computes*
and drop what a neuron merely had to *cope with*.

## The organising rule

Everything below follows from one constraint:

> **No operation may require globally synchronised state.**

Biology obeys it. That is *why* a brain tolerates hundreds of milliseconds of
delay, why it degrades gracefully as neurons die, and why it never needs a
barrier where every part waits for every other part.

Backpropagation violates it. The backward pass is a global barrier and it moves
tensors proportional to the parameter count. Latency-intolerance is the price of
exact gradients, and it is why deep networks need tightly-coupled hardware.

So latency tolerance is not a feature we add. It is what we get for free by
refusing to synchronise — and treating internet-distributability as a design
discipline keeps us honest, because it forbids quietly reintroducing a backward
pass.

## What we keep, and why

| Biological feature | Why it is real computation |
|---|---|
| Leaky temporal integration | Every unit gets a short working memory for free, with no recurrence machinery |
| Dendritic branch nonlinearity | A neuron is not a summer. Branches integrate locally and apply an NMDA-like plateau, making one neuron ≈ a 2-layer net that detects *coincident, clustered* input. High compute per message — and messages are the expensive resource |
| Three-factor plasticity | Credit assignment across time using only locally available quantities |
| Eligibility traces | Bridge the gap between an action and a late evaluation of it |
| Homeostasis | Stability without any global normalisation step |
| Sparsity + E/I balance | Sparsity is what makes the bandwidth bill survivable |
| Conduction delays | The brain *uses* delays for temporal coding; they are not lag to be minimised |

## Where we deviate, and why

### 1. Valued events instead of all-or-none spikes

The action potential is all-or-none because the axon is a lossy analog wire
needing regenerative signalling. We are on reliable digital transport, so we
keep the sparsity and asynchrony and drop the binarisation. Each event carries
a payload.

A rate code burns 10–50 spikes to convey one scalar. We convey it in one event.
Same event count, far more information per event.

An event's value is `value_base + graded component`. The baseline is not
cosmetic: with a soft threshold, neurons cross by a hair, so a purely graded
payload would be ≈0 for almost every event. With the baseline, an event is
never *less* informative than a binary spike and usually more.

### 2. Learnable, heterogeneous time constants

Biology's are pinned by channel kinetics. Ours are parameters, spread
log-uniformly over 12–320 ms. The spread alone is a large win: a population
with mixed constants holds working memory a homogeneous one cannot.

**This is where the first real bug lived.** A leaky integrator written
`v = decay·v + drive` has DC gain `1/(1-decay)`. With a 320 ms constant that is
a 320× amplification, so long-τ neurons were simply *louder*, not
longer-memoried — and the network saturated at 94% activity with every
threshold pinned at its clip ceiling. Every filter in the model is now written
with unit DC gain (`v = decay·v + (1-decay)·drive`). Guarded by
`test_membrane_gain_is_independent_of_time_constant`.

### 3. A routed vector modulator, not a chemical bath

Dopamine broadcasts one scalar to millions of synapses by diffusion — slow and
imprecise. We broadcast a small vector and let each neuron read it through its
own projection, so different neurons extract different credit from the same
signal. This is probably the largest single lever on credit-assignment quality,
and it costs bandwidth proportional to the *output* width, not the model size.

### 4. Scale-free dendritic plateau

**The second real bug.** The plateau nonlinearity initially had an absolute
knee. Branch potentials depend on fan-in, weight scale and input sparsity — they
settled around 0.067 while the knee sat at 1.0, so the nonlinearity never
engaged and every neuron silently degenerated into a linear summer.

The plateau is now defined relative to its own knee, and the knee adapts locally
to hold a target engagement rate — dendritic excitability homeostasis, the same
trick as the somatic threshold one level up. Guarded by
`test_plateau_stays_engaged`.

### 5. Dropped outright

Hodgkin–Huxley channel dynamics, vesicle chemistry, and the refractory period
*as biophysics* (a soft rate bound via homeostasis replaces it). These are
implementation details of a wet substrate.

## Why this distributes

- **Nothing waits.** A synapse reads history through a conduction delay of at
  least one step, never another unit's present state. Enforced by
  `test_no_synapse_reads_the_present`.
- **Events are addressed by emission time, not arrival time.** A late packet
  still lands in the correct slot of history, so a distributed run computes what
  a local run computes. Enforced by `test_buffer_late_scatter_matches_dense_write`.
- **The eligibility trace *is* the latency budget.** Traces span ~240 ms, so a
  modulator arriving hundreds of steps late still lands on the synapses that
  earned it. Biology evolved this to bridge delayed reward; we get delayed
  *packets* handled by the identical mechanism, with no extra machinery.
- **Bandwidth scales with activity, not parameters.** Sparse events out, a
  vector the width of the output back.

A machine 80 ms away is just a long axon.

### The caveat the episodic harness exposed

**The third real bug**, and an instructive one. With a 150-step modulator lag,
learning silently died — the delayed signal arrived *after* the episode ended
and was discarded at reset. A continuously running system has no such boundary;
`run_episode` now drains the in-flight modulator in a tail. Worth stating
plainly: latency tolerance requires the system to still be running when the
signal lands. Episode boundaries are the enemy of it, not network delay.

## The bug that invalidated everything before it

`W` was never applied in the forward pass. Branch integration read

```python
self.b += self.gain_branch * x.sum(axis=2)      # unweighted!
```

so synaptic weights were used by synaptic scaling and by the learning rule, but
never to compute anything. The network ran on implicit unit weights, and
learning was a no-op with no visible symptom — sparsity, plateau engagement,
thresholds and loss all looked healthy.

It surfaced only from a comparison that should have been impossible: a plastic
column and a frozen one produced **bit-identical** states while their weight
matrices differed by 25 %. Any measurement taken before this fix describes a
different model from the one in the repository.

The general lesson is worth stating: the dangerous bugs here are not the ones
that crash, they are the ones that leave a plausible-looking model that is
quietly not doing the thing it claims. Four separate bugs in this project all
produced the same innocuous symptom — a decoder at chance — which is why
`experiments/probe.py` earns its place. Asking *is the information present?*
separately from *is the rule extracting it?* localised every one of them.

## Honest limitations

- Local learning rules have historically plateaued below backprop on hard credit
  assignment. Unresolved here, and the reason the frozen-column control in
  `experiments/ablation.py` matters so much: a random recurrent population with
  heterogeneous time constants is a *reservoir*, and a trained linear readout on
  a reservoir solves many temporal tasks unaided. Without that control we would
  be admiring a number the plasticity rule had no part in producing.
- Latency tolerance does not buy bandwidth. The sparsity has to be real.
- The readout's RMS normalisation is the one place anything pools across
  neurons. It is cheap and low-dimensional, but it is a genuine exception to the
  locality rule and worth removing later.
- `symmetric` feedback requires broadcasting the readout matrix. Cheap for a
  small output space, but `dfa` is the strictly local option.
- Open-network participation means untrusted, heterogeneous, churning nodes.
  Start with a trusted cluster; treat that as much later work.

## Where the evidence actually stands

**Established.** The architecture holds information across a ~250-step delay,
and short-term synaptic plasticity is what makes that work: with STP disabled
the column is at chance (0.527 linear / 0.533 MLP), with it enabled the label
is recoverable at 0.864 ± 0.013 / 0.984 ± 0.006. That is a large,
reproducible architectural effect and it is the main positive result.

**Not established, and currently contradicted.** That the local three-factor
rule beats a frozen reservoir. Across three seeds it is neutral at a low rate
(0.856 ± 0.008 vs 0.864 ± 0.013 frozen) and clearly harmful at higher ones.
The most likely cause is bootstrapping: the column's credit signal is derived
from a readout that is itself weak, so the column follows a noisy teacher and
degrades a representation that was already good.

Worth being precise about what that does and does not impugn. The *locality*
claims — no synchronisation barrier, emission-time addressing, latency
tolerance via eligibility traces — are structural and hold regardless. What is
unproven is that this particular learning rule is worth running.

Both of those experiments have since been run, and both came back null: a task
with a +0.193 linear-to-MLP gap (+0.011, p = 0.45) and a readout pre-trained to
convergence before plasticity switches on (-0.003, p = 0.87).

A finite-difference check of the eligibility trace explains why. It carries
direction -- sign agreement 0.64 over 90 sampled synapses, one-sided p = 0.004
-- and no magnitude at all, correlation -0.045 against the measured change. The
update is lr * signal * elig, so each step combines a genuine directional
signal with magnitude noise of the same order. That is the shape of a rule that
neither helps nor destroys, and it accounts for every null so far without
blaming the benchmark or the teacher.

The next real work is therefore on the trace itself, not on anything feeding
it. The likely culprit is the surrogate `h`, which smooths the threshold to
keep gradients flowing but has no reason to be proportional to the true
sensitivity of a spike count to a weight. Candidates worth trying: scale the
surrogate by the neuron's distance from threshold rather than a fixed width;
accumulate sensitivity per emitted event instead of per timestep; or drop
magnitude entirely and use a sign-only update, which the check says is the only
part currently carrying information.

## Status

Single-column, single-process, transport-abstracted. Distribution is a transport
swap, not a rewrite — `NetworkTransport` implements the same three methods.
