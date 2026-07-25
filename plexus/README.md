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

## What actually accounts for the numbers

Read this before the detail below. Every figure here is measured and linked to
the sweep that measured it; the point of collecting them in one place is that
the individual results are easy to read as more encouraging than they are
together.

| contribution | worth | is it learning? |
|---|---|---|
| short-term plasticity holding the cue | **0.527 → 0.864** | no — synaptic state |
| homeostatic settling to an operating point | **+0.197** | no — a per-neuron fixed point |
| salience-gated Hebbian binding | +0.074 offline, +0.007 end to end | yes, unsupervised |
| lateral inhibition | +0.028 | yes, unsupervised |
| the designed three-factor learning rule | **−0.003** (p = 0.79) | it was supposed to be |

A frozen, randomly-wired column already reaches **0.802** linear decodability.
So the ordering is: the substrate does most of the work, self-regulation adds
the largest single increment, two unsupervised rules add refinements, and the
one mechanism designed to learn contributes nothing measurable.

Two later results sharpen that rather than softening it:

- **The operating point is not searched for, it is computable.** θ is a power
  law in each neuron's own membrane time constant, `θ = 5.927·τ^−0.748`, and
  using the formula is statistically indistinguishable from fifty episodes of
  homeostatic settling (p = 0.7672). The knee is a fixed quantile of its own
  branch's potential (r = 0.923). So the +0.197 is a fixed point these neurons
  have by construction, which the homeostatic loop finds by search only because
  nothing had told it the answer.
- **Three of the four headline departures in [DESIGN.md](DESIGN.md) had no
  measurement behind them.** The routed vector modulator is *inert* in every
  configuration that has produced a result — its content reaches the weights
  only through `W += lr · signal · drive`, and every sweep since 014 runs
  `lr = 0`. Heterogeneous time constants (sweep 036) and the dendritic plateau
  (sweep 037) are queued. The fourth was corrected earlier: an event's value
  turns out to be carried by synaptic efficacy, not by suprathreshold magnitude.

**What this does not say.** None of it means the architecture is wrong — the
locality properties are structural and hold regardless, and inference across
columns tolerates 150 ms for free. It means the *learning* story is currently
carried by two unsupervised rules worth +0.074 and +0.028, on one task, and that
the largest effects in the project belong to the substrate rather than to
anything that adapts to data.

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

This is *half* the architectural claim, measured rather than asserted, and the
half worth being careful about. What is measured here is that **conduction
delay between columns is free**, on frozen columns.

### A late modulator is not free, and the window is 12 steps

The other half — that a late *modulator* still lands on the synapses that
earned it, so learning survives a wide-area network — is the harder claim and
the one the design actually rests on. Sweep 026 ran it at twenty seeds on the
one mechanism that works, and **it does not hold.**

| modulator lag | binding's gain | as a fraction of lag 0 | |
|---|---|---|---|
| 0 | **+0.127** | 1.00 | 19/20 seeds, p = 0.0000 |
| 25 | +0.016 | 0.13 | p = 0.1610, null |
| 50 | +0.011 | 0.09 | p = 0.4184, null |
| 150 — intercontinental | +0.001 | 0.01 | p = 0.8849, null |
| 150, `tau_act_fast` 250 | +0.000 | 0.00 | p = 1.0000, null |

The cause is measured directly by `experiments/lagwindow.py` rather than
inferred from the decodability drop. `_bind` commits a **product of two traces**
with different time constants — `act_fast` (`tau_act_fast`, 50) and the branch
filter (`tau_branch`, 15) — and a product decays at the sum of the rates:

    1/tau_eff = 1/tau_act_fast + 1/tau_branch    →    tau_eff = 11.5 steps

Measured against three parameter settings, the product predicts 11.54 / 14.15 /
27.27 and the probe returns **11.87 / 15.25 / 28.68**. Either trace alone is
wrong by 3× to 17×.

So the window is dominated by the *shorter* constant, and the consequences are
specific:

- **`tau_act_fast` is not the lever.** Raising it 5× widens the window from 11.9
  steps to 15.3. Sweep 026 predicted that condition would restore most of the
  gain at lag 150; it restored +0.000.
- **`tau_branch` is the lever**, and it costs something. Raising it 15 → 60
  widens the window to 28.7 steps — but `tau_branch` is the branch filter in the
  *forward* path, so buying latency tolerance means changing what the column
  computes, not how it learns. Whether that column still decodes anything is
  unmeasured.
- **Nothing in this family reaches 150 steps.** Intercontinental credit
  assignment would need a trace the *rule* maintains, rather than one the
  forward path happens to leave lying around.

### Sweep 030 priced both routes, and inverted the prediction on each

**The forward-path route works.** With `tau_branch` at 60, binding still pays at
lags where the default pays nothing — measured against a reference at the *same*
`tau_branch`, 20 paired seeds:

| | gain over matched reference | | |
|---|---|---|---|
| lag 0 | +0.082 | 19/20 | p = 0.0000 |
| lag 25 | **+0.062** | 20/20 | p = 0.0000 |
| lag 50 | **+0.037** | 17/20 | p = 0.0000 |

At the default `tau_branch` the lag-25 comparison is null (+0.016, p = 0.1610).
So **regional latency — 25 to 50 steps, a rack to a metro area — is now measured
to work rather than hoped for.** Nothing reaches 150.

**The trace-the-rule-owns route is refuted.** `bind_tau_pre = 300` reaches a
154-step window in the direct probe, and it does not pay: its lag-150 gain is
+0.019 at **p = 0.0794, null**, and giving binding its own wide trace costs
**−0.099 at lag 0** on 0/20 seeds. A 154-step window scores activity from
essentially the whole episode, so binding strengthens whatever was active at
*any* point rather than what was active *then* — and that temporal specificity is
what made the mechanism work. The flag stays in the library, off by default; it
should not be described as a way to buy latency tolerance.

Together the two say something neither says alone: **a wider plasticity window
helps only when the forward path is wide enough to match it.**

### An accident worth its own sweep: `tau_branch` is worth more than anything else measured

The same sweep's *control* produced the largest single-parameter effect in this
project. With binding off entirely, widening the branch filter is worth:

| | decodability | vs `tau_branch` 15 | |
|---|---|---|---|
| `tau_branch` 15 | 0.620 | — | |
| `tau_branch` 60 | 0.838 | **+0.218** | 20/20, p = 0.0000 |
| `tau_branch` 120 | 0.918 | **+0.298** | 20/20, p = 0.0000 |

That is larger than homeostatic settling's +0.197 and larger than every learning
rule combined. It was found by accident, as the cost control for something else,
and the prediction had it going the *other way*.

**Sweep 034 removed the tail and four fifths of it went away.** At
`drain_steps 0`, against the standing 0.802:

| `tau_branch` | decodability | vs 15 | |
|---|---|---|---|
| 15 (default) | 0.802 | — | equals `off` exactly, p = 1.0000 |
| 30 | 0.852 | +0.050 | 17/20, p = 0.0002 |
| **60** | **0.858** | **+0.056** | 18/20, p = 0.0004 |
| 120 | 0.843 | +0.041 | 14/20, p = 0.0365 |
| 240 | 0.822 | +0.020 | 13/20, p = 0.2351 — null |

So of the +0.298, roughly **+0.056 is a real general improvement and +0.242 was
compensation for the 200-step drain tail** — a 15-step filter forgets across 200
silent steps and a 120-step one does not. Both link conditions reproduced sweep
030 exactly (0.620 and 0.918, same +0.298 at 20/20), so this re-scopes that
result rather than contradicting it.

The curve also turns over: the optimum is interior at 60, declining through 120
and 240, where under the tail it had been monotone increasing.

**`tau_branch = 60` is a known better setting that the defaults do not use, and
that is deliberate.** +0.056 at p = 0.0004 is comparable to salience-gated
binding's +0.074 and larger than lateral inhibition's +0.028 — from a
forward-path constant nobody ever tuned. But every standing number here was
measured at 15, so moving the default would invalidate all of them as a
comparison set. Re-baselining at 60 is a decision to take deliberately, with the
sweeps it would invalidate listed first.

One prediction was badly wrong and is worth the space. `tau_branch 240` was
included *because* it was predicted to fail — cues are 100 steps apart and a
240-step memory should not separate them — and it did not fail. The note had
pre-committed to checking the filter directly if that happened, so it was
checked: driving one impulse with recurrence silenced gives measured time
constants of 15.0, 60.0 and 240.0 against configured 15, 60 and 240. The filter
is exactly what it claims. What was wrong was the assumption that the branch
potential carries cue identity — **short-term plasticity does**, and it lives in
synaptic efficacy rather than in any filter, so smearing the filter costs
precision and not memory.

The three-factor rule's `tau_eligibility` is the mechanism the original claim
was about, and it is worth −0.003 at p = 0.79 with no lag at all — so it cannot
carry the claim either. **The distribution thesis holds for inference across
columns and does not currently hold for learning.** That is a narrower result
than this README asserted for most of the project's life, and it is the
measured one.

The measured half holds because delay is not lag the model is fighting — it is
a parameter the model already had. A peer 150 ms away is read the same way a peer 2 ms away is read: through
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
  been tuned against a working forward pass. Sweep 022 is the first evidence
  about what it buys and the answer is nothing detectable: disabling it gives
  0.780 decodability against 0.755, with identical sparsity, over 20 seeds. That
  is a curve comparison rather than a paired test, so it moves the parameter
  from "no evidence" to "evidence pointing at off" — not yet to a change.

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

**Salience-gated Hebbian binding: the first mechanism to improve both the
representation and the accuracy.** On a modulator release, every neuron
strengthens the excitatory synapses that were driving it, in proportion to its
own activity against its own baseline. Local — every quantity is a neuron
reading its own state, nothing pooled. Unsupervised — the modulator is read
only for its presence, never its sign or target, so it does not depend on the
error signal that failed eleven times.

| measurement | without | with | Δ | p |
|---|---|---|---|---|
| linear decodability (offline) | 0.802 | 0.876 | **+0.074** | 0.0000 |
| MLP decodability (offline) | 0.987 | 0.999 | +0.012 | 0.0001 |
| end to end, delta readout | 0.709 | 0.715 | +0.007 | 0.68 |
| end to end, RLS readout | 0.759 | 0.803 | **+0.043** | 0.0063 |
| end to end, binding frozen then readout allowed to converge (+150 episodes) | 0.707 | 0.770 | **+0.063** | 0.0013 |
| end to end, same 300-episode budget: decaying rate, or bind-then-stop | 0.709 | 0.715–0.731 | +0.006 … +0.023 | 0.10–0.56 |

20 paired seeds throughout, column `lr=0` so the three-factor rule contributes
nothing (`experiments/sweeps/engram-014`, `binding-015` … `binding-017`).

**The +0.074 carries a collection budget with it.** Every offline row above uses
500 collection episodes for the decoder. Sweep 028 re-measured the same column
at 1200 and got 0.853 → 0.913, a gain of **+0.060**. Neither number is wrong;
the difference is the part of the gain that comes from binding making the
representation *cheaper to fit* rather than better, which shrinks as the decoder
stops being starved. Quote the collection budget alongside the number, and do
not transfer it to a differently-sized probe.

### The gain holds across column size, and the task runs out before the column does

Sweep 028, 20 paired seeds, 1200 collection episodes, fan-in fixed at 8 × 16 so
each neuron's local computation is identical at every size:

| neurons | without | with | Δ | p |
|---|---|---|---|---|
| 48 | 0.734 | 0.812 | **+0.078** | 0.0001 |
| 96 | 0.853 | 0.913 | **+0.060** | 0.0000 |
| 192 | 0.897 | 0.963 | **+0.066** | 0.0000 |
| 384 | 0.973 | 0.993 | +0.020 | 0.0000 |
| 48, decoder starved to 384's samples/feature | 0.642 | 0.694 | +0.052 | 0.0684 — **null** |

The gain is flat across 48–192, which is what the architecture requires: fan-in
does not change with `N`, so a mechanism whose per-neuron operation is identical
at every size should have no size-dependent effect. A one-seed pilot had shown
the gain *rising* (+0.027 / +0.040 / +0.200); that was decoder starvation, and
the last row is the control that establishes it — starving the smallest column's
decoder did **not** inflate its gain.

The +0.020 at 384 is not a mechanism result. `off-384` reaches 0.973, so there
was no headroom for a larger gain to appear in. The same saturation shows up in
effective rank: 8× the neurons buys **15.8 → 20.5** dimensions with binding off
and **9.3 → 10.4** with it on. Neither is running out of column; both are running
out of things a delayed-XOR episode can be about. **The size question is answered
up to 192 and needs a harder task above it.**

One uncomfortable corollary: binding pins the representation to ~10 effective
dimensions almost regardless of column size. Sweep 020 read the 17.4 → 9.6
halving at 96 neurons as consolidation; across four sizes it looks less like a
halving and more like a *fixed ceiling*. That is good where the readout is
starved — which is every measurement here — and it is exactly what would stop
the mechanism scaling to a task needing more than ten dimensions. Untested.

**The third row is why the other rows needed explaining.** End to end with the
shipped delta readout, the +0.074 representation gain arrives as +0.007. It is
not lost — the readout cannot follow a representation that is *still changing
underneath it*. One line settles that: the frozen-binding condition scores
0.707 against 0.709 for the ordinary run, so the extra readout-only episodes
are worth nothing when there is no binding, and +0.063 when there is. The
readout does not need more time in general; it needs time against a column that
has stopped moving.

**And the gain is not free.** Sweep 018 tried to buy it inside the same
episode budget — a decaying binding rate, and binding for the first half then
stopping — and every condition was null. Lining up what each spends says why:

| | binding | idle | accuracy |
|---|---|---|---|
| flat | 300 | 0 | 0.715 |
| bind-then-stop | 150 | 150 | 0.731 |
| full + catch-up | 300 | 150 | 0.770 |

The idle period pays only on top of a *full* binding budget; taking binding
episodes away to fund it gives back nearly all the gain. So the honest headline
is a trade, not a free lunch: **binding buys about +0.06 end to end and costs
about 50% more training episodes to collect.**

That is also a real limitation stated as one: a system that must stop learning
before its readout can use what it learned has deferred continual learning
rather than solved it.

**And it is the readout that is slow, not the binding.** Sweep 019 tracked
decodability every 25 episodes rather than only at the end. The gap reaches
+0.097 by episode 150 and then goes flat for the remaining half of training —
the representation is finished at the halfway point. Yet `bind-then-stop`,
which stops binding exactly there and hands the readout 150 idle episodes,
delivers a third of what the full-budget catch-up delivers. The column had
learned the same thing in both; what differed was how long the readout had to
work against a settled representation.

So sweep 018's reading was wrong in an instructive way: binding was never the
bottleneck, which is why no binding schedule could fix it. The delta readout
needs roughly 300 episodes against a settled column. RLS is a faster readout
and delivers +0.043 with no schedule change at all — but needs a globally
pooled matrix that does not decompose (sweep 011).

One thing nobody had measured, visible only in the trajectory: the frozen
condition climbs from 0.582 to 0.779 over the first fifty episodes, on
homeostatic settling alone. **+0.197 of decodability from threshold and knee
adaptation with no learning rule of any kind** — more than twice what binding
adds.

### That +0.197 is an operating point, not a learned trajectory

Sweep 022 ablated the three quantities that adapt in that condition. The naive
version does not answer it: with threshold homeostasis off, sparsity falls from
0.031 to 0.0044 and the column is silent, so its flat curve is a fact about a
dead column. Sparsity is recorded beside decodability for exactly that reason.

The control that works separates *finding* an operating point from *keeping*
adapting: settle a twin — same seed, same weights, same wiring — copy its
threshold and knee across, then switch adaptation off. Twenty seeds, 100
episodes:

| condition | ep 0 | ep 100 | sparsity @100 | |
|---|---|---|---|---|
| everything adapting | 0.582 | 0.755 | 0.031 | |
| **preset, nothing adapting** | **0.752** | 0.752 | 0.023 | |
| preset, θ frozen only | 0.752 | 0.772 | 0.031 | no trend |
| no knee adaptation | 0.582 | 0.716 | 0.031 | |
| no synaptic scaling | 0.582 | **0.780** | 0.031 | |
| no threshold homeostasis | 0.582 | 0.582 | **0.0044** | *dead — uninterpretable* |

**Handing a column two per-neuron vectors gets 0.752 with nothing learning at
all** — +0.170 of the +0.197, 86% of the way to where full adaptation plateaus.
The fifty episodes are a *search* for an operating point, not an accumulation.
Ongoing adaptation on top of a correct one adds nothing measurable.

What the operating point mostly *is* became clear only after fixing a
measurement bug in the probe. The sparsity column read the firing-rate EMA,
which advances only while learning is on, so at checkpoint 0 it returned
`target_rate` unchanged from initialisation — 0.0300 for every condition,
looking like a measurement. Counted directly, **the untrained column runs at
0.0022, a fifteenth of target.** So the 0.582 baseline that every mechanism in
this project is measured against is a column that is barely firing, and
homeostasis's first job is lifting it to target. Presetting θ and knee supplies
that immediately.

**And it is better found from input with no task structure at all.** Sweep 023
settled the twin three ways — on the real task, on the task's marginals with
timing destroyed (each channel's time series permuted, so per-channel event
count and amplitude are preserved exactly), and on matched-rate noise:

| twin settled on | decodability | firing rate |
|---|---|---|
| **timing destroyed** | **0.803** | 0.027 |
| the real task | 0.752 | 0.022 |
| matched-rate noise | 0.691 | 0.017 |
| *(no preset, 100 episodes of settling)* | *0.755* | *0.030* |

Timing-destroyed beats the task-settled twin by **+0.051 (p = 0.0032)** and
beats a hundred episodes of ordinary settling by **+0.048 (p = 0.0049)**, on 20
paired seeds. Presetting *from the task* is null against ordinary settling
(p = 0.91) — so the entire gain comes from what the twin was shown, not from
presetting itself.

The obvious explanation is refuted by the sweep's own control column. If the
advantage were better rate calibration, the condition closest to the 0.03 target
would decode best. It does not: ordinary settling sits exactly on target and
decodes 0.048 *worse*. **The firing-rate ordering and the decodability ordering
do not match**, which rules out the confound in the strongest available form and
left what θ and the knee actually carry unexplained.

### θ is a power law in each neuron's own membrane time constant

Two sweeps closed that gap. Sweep 027 asked whether the operating point is a
*recipe* — a distribution of thresholds any neuron could draw from — or a
*fit* to each particular neuron. One permutation applied to both θ and the knee,
so every neuron receives a matched pair belonging to some other neuron:
distribution preserved exactly, ownership destroyed.

| | decodability | firing rate |
|---|---|---|
| operating point in place | 0.803 | 0.027 |
| same values, permuted between neurons | 0.673 | 0.060 |

**−0.130 on 0 of 20 seeds, p = 0.0000**, and the permuted column fires at double
its target rate. So it is a per-neuron fit, not a recipe — which raised the
obvious next question: a fit to *what*?

A direct probe answered it. θ is a power law in that neuron's own membrane time
constant, `θ = 5.927·τ^−0.748`, with log-log r = **−0.974** over six seeds —
90% of θ's variance from one static, local, construction-time property.

Sweep 029 then used the formula instead of settling, at 20 seeds:

| | decodability at episode 0 | after 100 episodes |
|---|---|---|
| ordinary settling | 0.582 | 0.755 |
| **θ from the formula, frozen** | **0.757** | 0.757 |
| θ from a settled twin, frozen | 0.763 | 0.763 |

The formula is **null against a settled twin's thresholds** (−0.006, 8/20 seeds,
p = 0.7672). A closed-form expression in each neuron's own τ is indistinguishable
from fifty episodes of homeostatic search.

**So the +0.197 — the largest single effect in this project, bigger than every
learning rule combined — is available at construction time from a one-line
function of a static local property.** It is not learning, and it is not
emergence in any sense that required the system to run. It is a fixed point the
neurons have by construction, which the homeostatic loop finds by search because
nothing had told it the answer.

Two honest limits. **The knee is not computable yet**: adding a settled twin's
knee to the formula reaches 0.786, statistically level with the full settled
operating point (p = 0.1997), but the knee still has to come from somewhere.
And **the exponent is fitted, not derived** — −0.748 comes from six seeds at 96
neurons on one task, so every claim here should be read as "a power law in τ
whose exponent must be refitted per configuration until something derives it".

### It is not criticality either

The obvious formalisation of "structure emerges from the system operating" is
self-organised criticality: a branching process turns one event into `m` further
events, and at `m = 1` a network holds and combines information over the longest
timescales available to it. Sweep 025 measured it and it is **not** what
homeostasis is producing.

| condition | branching excess | decodability |
|---|---|---|
| everything adapting | 0.003 | 0.755 |
| binding | 0.003 | 0.816 |
| binding + lateral | 0.004 | 0.837 |
| no knee adaptation | 0.001 | 0.716 |

Decodability moves **0.121** across these conditions; the branching excess moves
**0.003**, and every comparison of it is null. The sharpest case is `no-knee`,
which decodes reliably *worse* (−0.039, p = 0.0130) while its raw branching
ratio goes reliably *up* (+0.004, p = 0.0009) — the condition nearest critical
decodes worst.

**The control is what makes that legible, and it is the whole methodological
point.** Raw `m` reads 0.922–0.926 and three of four raw comparisons are
significant, which would have been written up as strong support. But silencing
the recurrent synapses — with homeostasis re-settled so activity matches — still
gives ~0.91. Roughly **99% of the apparent branching is membrane leakiness**, so
raw `m` is a statement about `tau_soma`. Reading it as criticality would have
produced a confident, significant, backwards conclusion.

So the emergence is real and it is simpler than a dynamical regime: an operating
point, two vectors long, reproducible by copying, better found from noise than
from the task, and now known to be neither criticality nor rate calibration.

Two things fell out that nobody was looking for. **Removing synaptic scaling
costs nothing** — 0.780 against 0.755, sparsity identical — which is the first
evidence about a live default (`scaling_lr = 2e-2`) that was chosen while the
mechanism was dynamically inert. And **settling is not monotone**: it overshoots
to 0.813 by episode 10 and relaxes to ~0.76. Sweep 019 checkpointed every 25
episodes and could not see it.

### What shape the representation is, and why decorrelating it was the wrong idea

Sweeps 011 and 019 both pointed at the same next step: decorrelate inside the
column, locally, so a cheap readout could extract what RLS extracts by pooling.
Sweep 020 built the mechanism — the Vogels–Sprekeler anti-Hebbian rule on the
inhibitory synapses that already exist — and, before tuning it, measured the
premise. Mean pairwise correlation and effective rank had never been recorded
for this model at all. Twenty paired seeds:

| condition | linear | MLP | mean \|corr\| | effective rank (of 96) |
|---|---|---|---|---|
| neither | 0.802 | 0.987 | 0.178 | 17.40 |
| lateral inhibition | 0.830 | 0.990 | 0.175 | 17.14 |
| binding | 0.876 | 0.999 | 0.255 | 9.64 |
| both | **0.913** | **1.000** | 0.256 | 9.38 |

**Binding raises correlation by +0.077 and nearly halves effective rank, while
raising linear decodability by +0.074.** All three at p = 0.0000, and on the two
shape metrics it is 20/20 and 0/20 seeds — there is no seed where it goes the
other way. The mechanism that works does not decorrelate. It **consolidates**:
it collapses the state onto fewer, more strongly co-varying directions, and
that is what makes it linearly readable.

So the reasoning behind the whole direction is refuted. Sweep 011's measurement
stands — RLS does gain by carrying a pooled correlation matrix, and that gain
does not decompose per column. The inference drawn from it, *therefore the
column needs decorrelating*, does not follow.

Lateral inhibition itself survived, by the branch its own prediction named as
the one that would save it: **+0.037 on top of binding, p = 0.0005**, taking
linear decodability past 0.9 for the first time. But it does not get to keep its
rationale. Alone it moves correlation by −0.004 and effective rank by −0.265 of
17.4 (null, p = 0.09); added to binding it moves correlation not at all
(p = 0.40) and effective rank *down* by 0.258 — the opposite of decorrelation.
**It improves the representation while moving neither decorrelation metric in
the direction decorrelation predicts.** What it is doing is unmeasured and
deliberately left that way in the code rather than filled in with a plausible
story.

### It pays end to end — through the readout it was built to replace

Sweep 021 asked whether any of that reaches the output. The point of lateral
inhibition was that a *local* rule could let the cheap shipped readout extract
what RLS extracts by pooling a correlation matrix across the whole population.
Twenty paired seeds:

| condition | readout | accuracy | vs | Δ | p |
|---|---|---|---|---|---|
| nothing on | delta | 0.709 | | | |
| lateral only | delta | 0.707 | nothing on | −0.001 | 0.90 |
| lateral + binding | delta | 0.731 | binding alone | +0.016 | 0.33 |
| lateral + binding | delta | 0.731 | **binding + RLS** | **−0.071** | **0.0012** |
| binding | RLS | 0.803 | | | |
| **lateral + binding** | **RLS** | **0.842** | binding + RLS | **+0.040** | **0.0284** |
| lateral + binding | delta, +150 catch-up | 0.775 | binding + catch-up | +0.005 | 0.60 |

Two things, and they pull opposite ways.

**The programme it was built for is closed.** Under the shipped delta readout
lateral inhibition is null — with or without a catch-up phase — and reliably
*worse* than RLS, on 5 of 20 seeds at p = 0.0012. Sweep 020 refuted the premise
and 021 refutes the payoff, so the direction stops rather than acquiring a rate
sweep. **RLS's pooled correlation matrix remains the one genuine exception to
the locality rule in this model, and nothing local has been shown to replace
it.**

**And it is the best result in the project.** Lateral + binding under RLS reaches
**0.842**, +0.040 over binding alone and +0.083 over the no-mechanism RLS
baseline. A mechanism built to make RLS unnecessary works only when RLS is
present.

The catch-up row is the informative failure. Line up how much of each offline
gain survives end to end:

| | catch-up | RLS | delta |
|---|---|---|---|
| binding (+0.074 offline) | 0.85 | 0.58 | 0.09 |
| lateral on top (+0.037 offline) | 0.14 | 1.08 | 0.43 (null) |

**The ordering inverts.** Binding's gain transfers best when the readout is
given a settled column; lateral's transfers essentially completely under RLS and
not at all under catch-up. So the delta readout's difficulty with lateral is not
*time* — 150 episodes against a static column collect nothing.

A hypothesis, labelled as one because nothing here measures it: what
distinguishes RLS from a delta rule given unlimited time is whitening. Sweep 020
measured lateral *not* reducing correlation, so if what it adds sits in
directions correlated with those already present, it would be recoverable by
whitening and invisible to gradient descent however long it runs. The test is a
principal-direction projection — if it holds, lateral's advantage lives in
low-variance directions and binding's in high-variance ones. Not run.

**The default configuration ships the delta readout, so 0.842 is not what the
model does today.** It is what it does with RLS, and RLS does not decompose
across columns.

The mechanism also arrived by way of one that was deleted. This began as an
engram allocator — excitability drift, a recruitment competition, an allocation
refractory (Han et al. 2007; Yiu et al. 2014; Cai et al. 2016). Sweep 012
measured the full apparatus at +0.029 over frozen, p = 0.0176, and that looked
like the result. Sweep 014 stripped it: the apparatus reads 0.830 against 0.876
for binding alone, worse on 19 of 20 seeds at p = 0.0001. The excitability half
cost 0.041; the competition contributed nothing (p = 0.52). All of it is gone
from the code and recorded in the sweeps.

Three lessons, cheaper to read than to rediscover:

- **A significant positive result is not evidence that the mechanism producing
  it is the right one.** 012's +0.029 was real, replicates at +0.028, and was
  less than half of what sat inside it in a component nobody had isolated.
- **Representation quality is not accuracy.** Sweep 009 established that in the
  direction that hid a loss; 016 is the same lesson in the direction that hid a
  gain, and 017 is what it took to tell them apart.
- **Run the control even when you expect it to be boring.** The
  frozen-binding-without-binding condition existed only for symmetry, and it
  turned out to carry the entire argument.

Two readings are worth separating. The *locality* claims — no synchronisation
barrier, emission-time addressing, tolerance of inter-column conduction delay —
are structural and hold regardless of whether the learning rule helps. What is
refuted is that this particular rule is worth running on this task.

One item on that list used to have to be moved off it: *latency tolerance
through eligibility traces* was an argument, not a measurement. Sweep 026 has
now measured it, on binding rather than on the three-factor rule, and the answer
is that the tolerance window is **12 steps** — see "A late modulator is not
free" above. The argument was not merely untested; it named the wrong time
constant, and the parameter it proposed to widen is the one that cannot help.

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
