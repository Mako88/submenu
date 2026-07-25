# plexus

A neural model built from what biological neurons *compute*, deliberately
discarding what they merely had to *cope with* — and organised around a single
rule that makes it distributable over ordinary internet latency.

See [DESIGN.md](DESIGN.md) for the full rationale.

## The rule

> **No operation may require globally synchronised state.**

Biology obeys it, which is why a brain tolerates hundreds of milliseconds of
delay and degrades gracefully as neurons die. Backpropagation violates it — the
backward pass is a global barrier moving tensors proportional to parameter
count, which is why deep nets need tightly-coupled hardware.

Latency tolerance is therefore not a feature to add. It is what falls out of
refusing to synchronise. A machine 80 ms away is just a long axon.

## What's here

```
plexus/
  events.py      valued events; timestamp-indexed ring buffer
  transport.py   the seam where a local run becomes a distributed one
  column.py      dendritic neurons, STP, three-factor local plasticity
  readout.py     linear decoder + broadcast modulator
  tasks.py       temporal benchmarks
  network.py     assembly and training loop
experiments/
  demo.py        train a column, report its settled dynamics
  baselines.py   what trivially simple models score (task sanity check)
  probe.py       is the information even present? memory vs. mixing
  ablation.py    does local plasticity beat a frozen reservoir?
```

## Quickstart

```bash
pip install numpy
python3 -m pytest plexus/tests/ -q
python3 experiments/demo.py --task xor
```

## Key deviations from biology

| Biology | Here | Why |
|---|---|---|
| All-or-none spikes | Sparse events carrying a **value** | The spike is binary because axons attenuate; digital transport doesn't. But see the correction below — the value turned out to be carried by synaptic efficacy, not by suprathreshold magnitude |
| Fixed time constants | Heterogeneous, learnable (12–320 ms) | Channel kinetics are a constraint, not a design |
| Diffuse chemical modulation | **Routed vector** modulator | Dopamine broadcasts one scalar by diffusion; we send a vector each neuron reads through its own projection |
| Delays as unavoidable lag | Delays as **first-class parameters** | Makes network latency a value the model already represents |
| Hodgkin–Huxley channels | Dropped | Implementation detail of wetware |

Kept because they're real computation: dendritic branch nonlinearities,
short-term synaptic plasticity, three-factor learning with eligibility traces,
homeostasis, sparsity, E/I balance.

## Results so far

**Benchmark sanity** (`experiments/baselines.py`) — a benchmark is only worth
reporting if simple things fail on it:

| Task | Rate code | Linear on 560 time-binned features | Chance |
|---|---|---|---|
| Delayed XOR | 0.494 | 0.486 | 0.500 |
| TemporalPatterns | 0.226 | **1.000** | 0.250 |

Delayed XOR defeats both. **TemporalPatterns is a weak benchmark** — it proves
timing matters but a linear model on time bins solves it outright, so it tests
nothing about the architecture. Reported here because it was one of the two
tasks originally built, and it should not be used to claim anything.

**Short-term plasticity is decisive** (`experiments/compare_step.py`) — XOR
decodability from the column state at answer time. Protocol: 96 neurons, 350
training episodes, 500 probe episodes, 70/30 split.

| | Linear decoder | MLP decoder |
|---|---|---|
| STP disabled | 0.527 | 0.533 |
| STP enabled | **0.864 ± 0.013** | **0.984 ± 0.006** |

Without facilitation the cues do not survive the ~250-step delay at all, so no
readout of any kind could work. With it, the cue is held in synaptic efficacy
rather than in ongoing spiking — activity-silent working memory, which is what
a 2 %-sparse network needs to bridge a delay.

**End to end, the model learns the task** (`experiments/e2e.py`). At 96 neurons
it reaches 0.717 ± 0.055 against 0.500 chance, with an offline decoder on the
same states reaching ~0.85. It previously sat at chance; the fix was aligning
the online decision vector with the one the probe validates (bug 8 below).

**Local plasticity does not help.** Eight 20-seed paired sweeps, each removing
one explanation:

| Sweep | What it changed | Δ (plastic − frozen) | p |
|---|---|---|---|
| 001 | 96 neurons, baseline | −0.003 | 0.79 |
| 002 | 24 neurons: label present but tangled (+0.193 linear→MLP gap) | +0.011 | 0.45 |
| 004 | readout pre-trained to convergence first | −0.003 | 0.87 |
| 006 | eligibility normaliser fixed (lr had run ~5× high) | +0.006 | 0.66 |
| 007 | sign-only updates, magnitude discarded | +0.003 | 0.77 |
| 008 | graded surrogate (gradient correlation −0.045 → +0.207) | −0.001 | 0.97 |
| 009 | measured the *representation* directly, bypassing the readout | −0.018 (linear) | 0.08 |
| 010 | RLS readout, so extraction is no longer the bottleneck | +0.006 | 0.60 |

Between them these exhaust the external explanations. Not the benchmark: it
fails with no headroom and with plenty. Not the teacher. Not the scaling:
correcting a 5×-inflated, drifting learning rate moves the number by 0.005. Not
the magnitude noise. Not the surrogate, where a genuine 5× improvement in
gradient fidelity moved accuracy by −0.001. Not the measuring instrument:
decoding the column directly shows no hidden gain, and trends slightly negative.
Not the readout, which now extracts essentially everything available.

Sweep 009 states the failure most sharply. A *frozen* column gives linear 0.639
against MLP 0.891 — the label is richly present and poorly linearly accessible,
a +0.25 gap that is precisely a learning rule's job. The rule does not close it,
and leaves the representation slightly worse than random wiring.

### What a working change looks like here, for contrast

The readout was switched from a delta rule to recursive least squares (FORCE,
Sussillo & Abbott 2009). Because a frozen column ignores the modulator entirely,
it is bit-identical across sweeps 006 and 010, which differ only in the readout
rule — so the two are paired by seed:

| Readout rule | Accuracy |
|---|---|
| delta | 0.592 ± 0.060 |
| **RLS** | **0.624 ± 0.057** |

**+0.032, better on 14/20 seeds, p = 0.0098.** The first statistically
significant positive result in the project, and it closed the online-vs-offline
extraction gap (0.693 against a ~0.69 ceiling).

The contrast is the point: when something in this model works, twenty seeds show
it plainly. The three-factor rule, measured in the very same sweep, gives +0.006
at p = 0.60.

RLS carries a real cost, documented rather than glossed: its (n+1)×(n+1) inverse
correlation matrix pools across the whole population, which is a genuine
exception to the locality rule. It stays inside the readout — the one component
that already sees every neuron — and never crosses the network, but it does not
scale to a large column.

How the first of these numbers moved is its own lesson:

| Seeds | Δ | |
|---|---|---|
| 3 | +0.059 | written up as a clear win |
| 5 | +0.042 | |
| 8 | +0.023 | p = 0.36 |
| **20** | **−0.003** | **p = 0.79** |

Nothing here should be believed from fewer than ~20 paired seeds, which is why
the sweeps run in CI — 20 in parallel take about two minutes of wall clock
against roughly an hour sequentially.

**Why it fails** (`experiments/gradcheck.py`). Everything rests on the claim
that `elig[n,b,s]` tracks how much neuron *n*'s output would change if that
synapse were strengthened, and nothing had tested it. A finite-difference check
over 90 sampled synapses, recurrence disabled:

| | |
|---|---|
| sign agreement | **0.64** (58/90, one-sided p = 0.004) |
| correlation | **−0.045** |
| median ratio (numeric / analytic) | +0.0023 |

The trace points the right way appreciably more often than chance and its
magnitudes carry no information at all. Sweep 007 shows that keeping only the
sign is not enough either, so the trace needs **rebuilding rather than
reinterpreting**. The suspect is the surrogate `h`: it smooths the threshold so
gradients keep flowing, but nothing makes it proportional to how sensitive a
spike count actually is to a weight. That is a design change, not a parameter,
and it is where the next real work is.

What the fixes *did* achieve is moving plasticity from **actively harmful**
(0.49–0.61 against 0.733 frozen) to **neutral**. The three changes that
mattered were: matching the eligibility window to the readout's filter, so
credit stops being smeared over 240 ms of activity that predates the cue;
releasing one modulator per decision instead of sustaining it for ~70 steps
against an increasingly contaminated trace; and normalising the update
population-wide rather than per neuron.

## Splitting it across machines

`DistributedPlexus` cuts the model into columns that share one event substrate.
Each column owns a slice of the source space, publishes only that slice, and
reaches its peers exclusively through a conduction delay. The per-step loop is:

```python
transport.begin(t)                    # open the timestep
transport.publish_slice(t, 0, ext)    # nobody's column owns the sensors
for col in columns: col.publish(t)    # each writes only what it owns
for col in columns: col.step(t)       # each reads history, never the present
```

Only the middle two lines would cross a network. Every column publishes before
any column reads, and every synapse carries a delay of at least one step, so no
column can observe another's current state — which means **the order columns
are stepped in cannot change the result**. That is asserted by
`test_step_order_does_not_change_the_result`, and it is the property that makes
the loop safe on separate machines rather than merely tidy about ownership.

Distribution costs the model exactly one parameter: `peer_delay`, the
conduction delay on synapses reaching a neuron another column owns. One step is
1 ms, so it reads directly as network latency — 2 is a rack, 30 a region, 150
intercontinental. Cortex runs 0.5–30 ms conduction delays natively, so the
lower half of that range is not a compromise; it is the regime the architecture
was designed around.

### Latency costs nothing

Four columns of 24 neurons, 4 seeds per point, frozen columns
(`.github/workflows/plexus-latency.yml`):

| peer delay | eval accuracy |
|---|---|
| 2 ms — same rack | 0.625 ± 0.079 |
| 10 ms | 0.650 ± 0.080 |
| 30 ms — same region | 0.673 ± 0.095 |
| 80 ms | 0.655 ± 0.056 |
| 150 ms — intercontinental | 0.673 ± 0.090 |

**A 75× increase in inter-column latency costs −0.048 accuracy** — that is,
nothing, and the sign is negative: the 150 ms condition scored slightly *higher*
than the 2 ms one. Seed-to-seed spread within a single condition (±0.06–0.10)
dwarfs any difference between conditions.

This is the architectural claim, measured rather than asserted. It holds because
delay is not lag the model is fighting — it is a parameter the model already
had. A peer 150 ms away is read the same way a peer 2 ms away is read: through
a conduction delay, addressed by emission time. Nothing waits, so nothing
degrades. (The slight upward trend is not significant here, but it is the
direction the design predicts: a wider spread of delays enriches the temporal
basis, which is why cortex uses delays for coding rather than minimising them.)

Two real bugs surfaced only once the model was actually split:

- The source space was sized for a column's own neurons, so any synapse
  reaching a peer indexed out of bounds.
- Dale signs were drawn from each column's own seed, so the same neuron could
  be excitatory as far as one column was concerned and inhibitory to another.
  Being excitatory is a property of the *emitting* neuron, not of whoever reads
  it. Signs now come from a shared `sign_seed`.

### Correction: where an event's value actually comes from

The design intent was that suprathreshold magnitude makes an event more
informative than a spike. Measured, that component uses **0.32 % of its
available span** — the soft reset (`v -= θ`) means the membrane never climbs far
past threshold, so `u ≈ 0` and the raw emission is nearly constant at
`value_base`. By that mechanism, events are binary in practice.

The variation is real and large — about **52 %** relative spread — but it comes
from short-term plasticity scaling the amplitude, not from the threshold
crossing. That is arguably the more biological answer, since real terminals
modulate amplitude through release probability rather than through how far past
threshold the soma got. It is not, however, what was claimed, and `value_scale`
is very nearly an inert parameter as a result.

Found by auditing every config field for whether it changes behaviour at all —
the generalised form of the test that caught the forward-pass bug. Pinned by
`test_event_value_variation_comes_from_stp_not_threshold_crossing`.

## Bugs worth knowing about

Each is now a regression test. Every one of them left a model that ran, trained,
and reported plausible numbers:

1. **`W` was never applied in the forward pass.** Branch integration summed its
   inputs *unweighted*, so synaptic weights were read by the learning rule and
   by synaptic scaling but never used to compute anything. Caught only because
   a plastic and a frozen column produced **bit-identical** states while their
   weight matrices differed by 25 %. Every measurement taken before this fix
   describes a network running on implicit unit weights.
2. **Leaky integrators with the wrong DC gain.** `v = decay·v + drive` has gain
   `1/(1-decay)`, so 320 ms neurons were 320× *louder* rather than
   longer-memoried. Network saturated at 94 % activity.
3. **An absolute plateau knee.** Branch potentials settled near 0.067 against a
   knee of 1.0, so the dendritic nonlinearity never engaged and every neuron
   degenerated into a linear summer.
4. **A metric that leaked the label.** The readout updated *during* the scoring
   window, fitting the current episode within a few steps. Reported 1.000
   training accuracy on a model whose held-out accuracy was 0.508.
5. **Eligibility traces too small to matter.** Chained unit-gain filters left
   eligibility near 1e-2; with a raw learning rate the modulator moved weights
   by 0.1 % while homeostatic scaling moved them by 21 %. The three-factor rule
   was measurably decorative. Fixed by per-neuron normalisation.
6. **Standardising by the wrong distribution.** The readout normalised using
   per-timestep variance but classified an episode-averaged vector, whose
   across-episode variance is ~100× smaller and unevenly so per dimension.
7. **EMA variance seeded at 1.0.** Column traces are tiny, so the leftover
   initialisation dominated the true variance for thousands of episodes,
   shrinking standardised features to std 0.003. Now bias-corrected.

8. **Deciding on a window average instead of the trace.** The online readout
   classified the answer-window mean while the probe validated the trace at the
   end of that window. The trace already applies ~60 ms of exponential
   weighting; averaging on top diluted a signal concentrated just after the go
   cue (0.69 vs 0.85 decodable). Aligning them took the end-to-end model from
   chance to 0.73 — the single largest improvement in the project.
9. **Eligibility window mismatched to the readout.** `tau_eligibility` was
   240 ms against a 60 ms readout filter, so credit was smeared over activity
   reaching back before cue B arrived. This was most of why plasticity was
   actively harmful.
10. **The modulator was sustained for ~70 steps.** One decision produced ~70
    weight updates with the same error, applied against an eligibility trace
    that kept absorbing post-decision activity — too strong and steadily more
    misdirected. Now one decision, one release.

Bugs 4–8 all produced the same symptom — a decoder stuck at chance on data that
was demonstrably separable — which is why `experiments/probe.py` exists. Being
able to ask "is the information even present?" separately from "is the learning
rule extracting it?" was worth more than any single fix.

### Which earlier fixes were themselves confounded

Bug 1 hid the forward pass for most of the project's life, so every tuning
decision before it was made on a network running with implicit unit weights.
Auditing those afterwards, most survive — the DC-gain fix, the scale-free
plateau knee, multiplicative homeostasis and the readout scaling fixes are all
mathematically independent of whether `W` is applied. Two did not:

- **Per-neuron eligibility normalisation** was introduced to fix a symptom bug 1
  created. It divides each neuron's update by *its own* eligibility magnitude,
  which destroys the natural weighting in which strongly engaged neurons receive
  larger updates, and amplifies noise from neurons that barely participated.
  Now `elig_norm="column"` by default; the old behaviour is still selectable.
- **`scaling_lr`** was set to `2e-2` while synaptic scaling was *dynamically
  inert*, since it only ever moved a `W` that nothing read. It is now a live
  force (~21 % weight movement) and is a genuine free parameter that has never
  been tuned against a working forward pass.

## Status

Single-column, single-process, transport-abstracted. Distribution is a
transport swap rather than a rewrite: `NetworkTransport` implements the same
three methods, and because events are addressed by *emission* time, a late
packet still lands in the correct slot of history.

**Established.** The architecture holds information across a ~250 ms delay and
solves delayed XOR end to end at 0.71–0.74 against 0.50 chance. Short-term
synaptic plasticity is what makes that work: without it the column is at chance,
with it the label is 0.86 decodable. That is the main positive result.

**Refuted.** That the local three-factor rule beats a frozen reservoir. Twenty
paired seeds put the difference at −0.003 with p = 0.79.

Two readings are worth separating. The *locality* claims — no synchronisation
barrier, emission-time addressing, latency tolerance through eligibility traces —
are structural and hold regardless of whether the learning rule helps. What is
refuted is that this particular rule is worth running on this task.

That was first blamed on the task leaving nothing to do, since a *random* column
already makes delayed XOR ~0.86 linearly decodable. Sweep 002 tested that
directly at 24 neurons, where the label is present but tangled (linear 0.689,
MLP 0.881), and the rule still gave +0.011 at p = 0.45. The excuse is spent:
the rule fails both with and without headroom, so the problem is the rule or
its credit signal, not the benchmark. First indications are that 3-cue parity is currently a
*memory* wall rather than a mixing one: with 96 neurons both linear and MLP
decoders sit at chance, and even at 192 neurons with shortened delays they
reach only ~0.57. Holding three cues looks to be past what this column's
short-term plasticity can carry, which is a capacity result worth having in its
own right.
