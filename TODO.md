# What is outstanding

**Scan this table and stop, if that is all you have time for.** Every row is a
question in plain language, with where it stands. Detail is below.

| # | the question, plainly | where it stands |
|---|---|---|
| 1 | Does any of this survive a task that asks more than one puzzle? | **Forgetting: answered — nothing forgets, so there is nothing to fix.** Capacity: untouched. Emergent structure: criticality refuted, simpler answer standing |
| 2 | What is lateral inhibition actually doing? | Works (+0.037), best result in the project with it — but the reason it was built is refuted and the real mechanism is unknown |
| 3 | Is the "settling" the network does for free precomputable? | **Yes, and better from static than from the task.** Why, is open |
| 4 | Can we stop simulating every neuron every millisecond? | Untouched. ~97% of current work is wasted; would be ~50× faster |
| 5 | Can we add new senses to a running network? | Built and tested. Open: does a *trained* model survive it |
| 6 | Which old decisions rest on evidence a later fix destroyed? | Six items, tracked in AUDIT.md |
| 7 | Does it actually work spread across machines? | Never tried. All delays so far are simulated inside one process |
| 8 | Would a single GPU just beat this? | Honest counter-argument, unmeasured |

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
- **Continual learning.** Built and measured — sweep 024, and the answer is that
  **there is no forgetting here to study.** Two tasks that are the same
  computation over a permuted sensory mapping, 20 seeds: every condition loses
  between 0.015 and 0.030 of task A, and **all four retention comparisons are
  null** (p = 0.32 to 0.81). Binding does not retain better or worse than
  nothing.

  So the engram rematch does not trigger — for the opposite reason to the one
  the prediction named. Not because binding retained well, but because an
  undifferentiated Hebbian store shows no detectable interference at this scale.
  Allocation stays deleted, and the reason is now better than it was: sweep 014
  refuted it on a task that could not ask, and this one asked.

  What the benchmark does resolve is **acquisition**: off 0.269 → lateral 0.310
  → binding 0.325 → both 0.353, all at p ≤ 0.0007, plus binding beating off on
  20/20 seeds for task A itself. That is an independent replication of the
  standing results on a channel mapping no previous sweep used.

  **To make it bite** (re-run, do not stretch the conclusion): a longer stream
  than two tasks, a smaller column — 96 neurons for one delayed-XOR variant is
  not obviously capacity-limited — and more episodes per task so there is more
  to overwrite.

  One correction worth carrying: while building this I wrote that a frozen
  column "forgets 0.167", from one seed, into four files. At 20 seeds it is
  **0.015**. The mechanism is real and now tested — a frozen column does change
  when its input changes, since homeostasis and scaling keep running — but the
  magnitude was wrong by an order of magnitude. Third time a single seed has
  produced a claim twenty seeds corrected, and the first time one reached the
  record as a statement of fact rather than a prediction.
- **Emergent structure.** The original motivation, and it now has a metric —
  which is what this bullet was blocked on. Sweep 025.

  The framing that unblocked it: maybe learning here is not a mechanism to be
  designed but a property of the system running. The evidence for that is
  already on the board and was not being read that way. The designed
  three-factor rule is worth −0.003; homeostasis, which is a thermostat rather
  than a learning rule, is worth +0.197; and sweep 023 found that benefit comes
  out *better* from structureless input than from the task, so it is not
  acquiring the task at all.

  **Self-organised criticality** was the version of that idea with a number
  attached, and sweep 025 **refutes it**. The branching excess over a control
  with recurrence silenced spans 0.001 to 0.004 across every condition, and
  every comparison of it is null — while decodability over the same conditions
  moves 0.121, from 0.716 to 0.837. Forty times the range, none of it in `m`.

  The cleanest refutation is `no-knee`: disabling knee adaptation decodes
  **reliably worse** (−0.039, p = 0.0130) while its raw branching ratio goes
  **reliably up** (+0.004, p = 0.0009). The condition closest to critical is
  the one that decodes worst.

  The control is what makes that readable. Raw `m` sits at 0.922–0.926 and three
  of four raw comparisons are significant — without the control this would have
  been written up as strong support. It is membrane leakiness: the control reads
  ~0.91 on its own.

  **That closes the measurement, not the idea** — and the idea is in better
  shape than before. The emergent thing is an operating point, not a dynamical
  regime, which is a much simpler object: sweep 022 showed it can be handed over
  by copying two per-neuron vectors, and sweep 023 that it is better found from
  input with no task structure at all (+0.051, p = 0.0032).

  Two formalisations remain, both cheaper than this one:

  - **Structural.** Does the *wiring* self-organise? `Column.add_inputs` already
    does the mechanics of rewiring; nothing has asked whether rewiring on a local
    rule beats the fixed random graph.
  - **The simple answer.** Characterise what θ and the knee actually encode. Two
    explanations are now eliminated — it is not criticality (025) and not rate
    calibration (023's sparsity column) — and the quantity is two vectors long.
    This is the obvious next move.

## 2. What is lateral inhibition actually doing?

*In plain terms: a mechanism was added on the theory that the neurons were all
saying too-similar things and needed spreading out. It helps — it produced the
best result in the project. But it turns out it does not spread them out at all,
so nobody knows why it works. Keeping a mechanism whose explanation is wrong is
fine; pretending to know why is not.*

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

Sweep 021 settled the payoff question and closed the programme. Under the
shipped delta readout lateral is null (+0.016, p = 0.33) and reliably worse than
RLS (−0.071, 5/20 seeds, p = 0.0012). **RLS's pooled matrix remains the one
genuine locality exception and nothing local has replaced it.**

But it reaches **0.842 with RLS**, +0.040 over binding alone at p = 0.0284 — the
best end-to-end number in the project. A mechanism built to make RLS unnecessary
works only when RLS is present.

What is open is why, and it now has a sharp form. The transfer ratios invert:

| | catch-up | RLS | delta |
|---|---|---|---|
| binding (+0.074 offline) | 0.85 | 0.58 | 0.09 |
| lateral on top (+0.037 offline) | 0.14 | 1.08 | 0.43 (null) |

A catch-up phase collects 85% of binding's gain and 14% of lateral's, so the
delta readout's problem with lateral is not *time*. The hypothesis — untested,
and the reason this item stays — is that lateral's contribution sits in
directions correlated with those already present, recoverable by whitening and
invisible to gradient descent however long it runs. Sweep 020 measured lateral
not reducing correlation, which is consistent with that.

**The test:** project the column state onto its top-k principal directions and
measure how much of each mechanism's decodability advantage survives. If it
holds, lateral's gain lives in low-variance directions and binding's in
high-variance ones — and that would say what kind of readout is worth building
next, which is more useful than either accuracy number.

Sparsity is flat across all four conditions (0.031 → 0.034), so whatever lateral
does, it is not a gross activity change.

## 3. Is the settling operating point precomputable? (sweep 023)

Sweep 022 answered the original question. The +0.197 is **an operating point,
not a learned trajectory**: a column handed a settled twin's threshold and knee,
with nothing adapting and `lr=0`, reads 0.752 against 0.582 — +0.170 of the
+0.197, with no learning of any kind. Ongoing adaptation on top of a correct
operating point adds nothing measurable over 100 episodes.

What is left is whether it is **free**. The twin had fifty episodes of task
exposure to find those values — unsupervised, no labels, but derived from data.
Two conditions settle it:

- twin settled on the task, as measured (0.752)
- twin settled on **random input with matching rate and sparsity**

If the second matches, the operating point is precomputable from input
statistics alone, and fifty episodes of every future run come free. If it does
not, the settling is doing something task-specific and that is a more
interesting finding than the saving.

Keep it two conditions. It is a clean question and folding it into a larger
matrix is how it stops being one.

### Fell out of 022, both worth chasing

- **Synaptic scaling is not carrying anything.** Removing it gives 0.780
  against 0.755 with identical sparsity. `scaling_lr = 2e-2` was chosen while
  the mechanism was dynamically inert — it only ever moved a `W` the forward
  pass ignored — and AUDIT has flagged it as untuned ever since. This is the
  first evidence, and it points at *off*. Needs a paired test rather than a
  curve comparison before acting.
- **Settling is not monotone.** It overshoots to 0.813 by episode 10 and relaxes
  to ~0.76; `no-scaling` overshoots harder, to 0.834. Sweep 019 checkpointed
  every 25 episodes and missed it. "Converges to a plateau" is the natural
  description and it is wrong.

## 4. Event-driven execution

*In plain terms: right now every neuron is recalculated every millisecond, even
though only about 3% of them are doing anything. It is like polling every house
on a street each second to ask if anything happened. Switching to "tell me when
something happens" should be exact here rather than an approximation, and worth
roughly 50× — the single biggest speedup available.*

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

*In plain terms: several settings in this model were chosen based on
measurements that a later bug fix invalidated. The settings never got revisited.
This is the list of decisions currently resting on evidence that no longer
holds — including a couple where a good idea may have been discarded for a bad
reason.*

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

*In plain terms: the entire point of the project is running this spread across
ordinary machines over the internet. That has never actually been done — every
"distributed" measurement so far is one process pretending to have network
delays. The make-or-break number is how much traffic crosses between machines.*

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

*In plain terms: this might all be beaten by one graphics card. If a single GPU
outperforms hundreds of CPUs per dollar, then "no data centres" has to be
argued on grounds other than cost. Worth measuring rather than avoiding.*

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
- **Arm the wake-up in the same action that launches the sweep.** Sweep 025
  finished and sat unrecorded because the commit that triggered it did not also
  schedule a check-in — the scheduler was fine, every trigger in the account
  fired on time. The failure mode is silent: a finished run looks identical to a
  running one until someone asks. Launching and watching are one step, not two.
- **`experiments/mutation.py` has caught its own staleness four times.** When a
  refactor moves a line a mutation targets, it reports "source moved" and
  fails rather than going quietly green. Keep that property in any rewrite.
