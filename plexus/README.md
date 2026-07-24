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

**Local plasticity does not yet earn its place** — the negative result this
project exists to be honest about. Same protocol, 3 seeds, measuring how
linearly separable the column's own representation becomes:

| Condition | Linear | MLP |
|---|---|---|
| frozen (no three-factor rule) | **0.864 ± 0.013** | 0.984 ± 0.006 |
| plastic, `lr=2e-4` | 0.856 ± 0.008 | 0.980 ± 0.014 |
| plastic, `lr=1e-3` | 0.673 | 0.953 |
| plastic, `lr=4e-3` | 0.793 | 0.900 |

At best neutral, and clearly harmful as the rate rises. The likely cause is a
bootstrapping problem: the column's credit signal is `readout.Wᵀ @ (−err)`, and
the readout is itself weak, so the column faithfully follows a noisy teacher.

**The end-to-end model is still limited by readout sample efficiency.** The
representation is separable at ~0.86 offline, but the online single-pass delta
rule extracts it slowly — 0.44 → 0.55 → 0.58 over 300/600/900 episodes.
Learning, but far from what the same data supports offline.

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

Bugs 4–7 all produced the same symptom — a decoder stuck at chance on data that
was demonstrably separable — which is why `experiments/probe.py` exists. Being
able to ask "is the information even present?" separately from "is the learning
rule extracting it?" was worth more than any single fix.

## Status

Single-column, single-process, transport-abstracted. Distribution is a
transport swap rather than a rewrite: `NetworkTransport` implements the same
three methods, and because events are addressed by *emission* time, a late
packet still lands in the correct slot of history.

What is established: the architecture holds information across a delay, and
short-term plasticity is what makes that work. What is not: that the local
three-factor rule improves on a frozen reservoir. On the evidence so far it
does not.
