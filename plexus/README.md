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
| All-or-none spikes | Sparse events carrying a **value** | The spike is binary because axons attenuate. Digital transport doesn't. A rate code burns 10–50 spikes per scalar; we send one event |
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

**End to end, the model learns the task** (`experiments/e2e.py`). 96 neurons,
400 episodes, 150 held-out episodes, 8 paired seeds:

| Condition | Held-out accuracy |
|---|---|
| frozen reservoir | 0.717 ± 0.055 |
| plastic, `lr=5e-3` | 0.740 ± 0.044 |

Chance is 0.500 and an offline decoder on the same states reaches ~0.85, so the
online model captures most of what is there. It previously sat at chance; the
fix was aligning the online decision vector with the one the probe validates
(bug 8 below).

**Local plasticity does not help. This is settled, not pending.** 20 paired
seeds via `.github/workflows/plexus-experiment.yml`:

| Condition | Accuracy |
|---|---|
| frozen | 0.713 ± 0.078 |
| plastic | 0.710 ± 0.075 |

Mean paired difference **−0.003**, better on **9/20** seeds, exact permutation
**p = 0.79**. Zero, as precisely as this measurement can say so.

How that number moved is the lesson:

| Seeds | frozen | plastic | mean paired diff |
|---|---|---|---|
| 3 | 0.686 | 0.745 | +0.059 |
| 5 | 0.713 | 0.755 | +0.042 |
| 8 | 0.717 | 0.740 | +0.023 (p = 0.36) |
| **20** | **0.713** | **0.710** | **−0.003 (p = 0.79)** |

At three seeds this looked like a clear win and was written up as one. It
decayed monotonically as seeds accumulated and landed on exactly nothing.
Nothing here should be believed from fewer than ~20 paired seeds, which is why
the sweep runs in CI — 20 seeds in parallel take about two minutes of wall
clock, against roughly an hour of sequential local runs.

What the fixes *did* achieve is moving plasticity from **actively harmful**
(0.49–0.61 against 0.733 frozen) to **neutral**. The three changes that
mattered were: matching the eligibility window to the readout's filter, so
credit stops being smeared over 240 ms of activity that predates the cue;
releasing one modulator per decision instead of sustaining it for ~70 steps
against an increasingly contaminated trace; and normalising the update
population-wide rather than per neuron.

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

The most likely reason is that the task leaves nothing to do: a *random* column
already makes delayed XOR ~0.86 linearly decodable, so the rule is being asked
to improve a representation that is already near the ceiling. `DelayedParity`
(`n_cues=3`) is the intended follow-up — a frozen column gets ~0.57 there, so
there is real headroom. First indications are that 3-cue parity is currently a
*memory* wall rather than a mixing one: with 96 neurons both linear and MLP
decoders sit at chance, and even at 192 neurons with shortened delays they
reach only ~0.57. Holding three cues looks to be past what this column's
short-term plasticity can carry, which is a capacity result worth having in its
own right.
