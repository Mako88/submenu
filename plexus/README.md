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

**Short-term plasticity** (`experiments/probe.py`) — XOR decodability from a
*frozen* column at answer time:

| | Linear decoder | MLP decoder |
|---|---|---|
| Before STP | 0.510 | 0.573 |
| With STP + go cue | **0.860** | **0.990** |

Before STP the cues simply did not survive the ~250-step delay, so no readout
of any kind could have worked. Facilitation holds the cue in synaptic efficacy
rather than in ongoing spiking — activity-silent working memory, which is what
a 2 %-sparse network needs to bridge a delay.

## Three bugs worth knowing about

Each is now a regression test, and each would have quietly hollowed out the
architecture while still "working":

1. **Leaky integrators with the wrong DC gain.** `v = decay·v + drive` has gain
   `1/(1-decay)`, so 320 ms neurons were 320× *louder* rather than
   longer-memoried. Network saturated at 94 % activity with every threshold
   pinned at its ceiling.
2. **An absolute plateau knee.** Branch potentials settled near 0.067 against a
   knee of 1.0, so the dendritic nonlinearity never engaged and every neuron
   degenerated into a linear summer.
3. **A metric that leaked the label.** The readout updated *during* the scoring
   window, fitting the current episode within a few steps. Reported 1.000
   training accuracy on a model whose held-out accuracy was 0.508 — pure
   chance. Answering and being told the answer are now separate phases.

## Status

Single-column, single-process, transport-abstracted. Distribution is a
transport swap rather than a rewrite: `NetworkTransport` implements the same
three methods, and because events are addressed by *emission* time, a late
packet still lands in the correct slot of history.

Not yet established: that local three-factor plasticity beats the frozen
reservoir. That is what `experiments/ablation.py` measures, and it is the
number that decides whether the learning rule is earning its place.
