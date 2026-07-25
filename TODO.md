# What is outstanding

**Scan this table and stop, if that is all you have time for.** Every row is a
question in plain language, with where it stands. Detail is below.

| # | the question, plainly | where it stands |
|---|---|---|
| 1 | Does any of this survive a task that asks more than one puzzle? | **Forgetting: answered — nothing forgets, so there is nothing to fix.** Capacity: untouched. Emergent structure: criticality refuted, simpler answer standing |
| 2 | What is lateral inhibition actually doing? | Works (+0.037), best result in the project with it — but the reason it was built is refuted and the real mechanism is unknown |
| 3 | Is the "settling" the network does for free precomputable? | **Yes — and for θ it is a closed form.** `θ = 5.927·τ^−0.748` matches a settled twin (p = 0.77). The knee is still open |
| 4 | Can we skip the work that isn't doing anything? | **Measured.** Not at neuron level (81% are active) — at connection level. Worth ~2× now, ~5× at scale. Demoted from "biggest lever" |
| 5 | Can we add new senses to a running network? | Built and tested. Open: does a *trained* model survive it |
| 6 | Which old decisions rest on evidence a later fix destroyed? | Six items, tracked in AUDIT.md |
| 7 | Does it actually work spread across machines? | Still never tried on real machines. But the property it depends on is now **measured**: delivery jitter below `delay_min` leaves a distributed run bit-identical, and above it does not |
| 8 | Would a single GPU just beat this? | **The premise was wrong.** Not bandwidth-bound — 17 % of DRAM peak at 96 neurons, working set fits in L2. It is overhead-bound, so the comparison cannot be made until the code is near *some* limit |
| 9 | Which claims in the record were never measured? | New. Heterogeneous time constants is the live one — **sweep 036 built and queued**. Three others measured today, two refuted |

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

- **Capacity scaling.** **Measured — sweep 028, and the gain holds.** +0.078 /
  +0.060 / +0.066 at 48 / 96 / 192 neurons, 20 paired seeds, all p ≤ 0.0001.
  Flat, which is what the architecture requires: fan-in is fixed at 8 × 16
  regardless of `N`, so a mechanism whose per-neuron operation does not change
  with size should have no size-dependent effect. The mechanistic prediction was
  written down in advance and beat a one-seed pilot that showed the gain rising
  to +0.200 — that rise was decoder starvation, and the `48-thin` control (same
  column, decoder starved to the largest column's samples-per-feature) came back
  **null at p = 0.0684**, which is what rules starvation out as the source.

  Two things it did not settle, both now the live half of this item:

  - **384 neurons is uninformative, because the task ran out.** `off-384` reaches
    0.973 and there is no headroom left for a gain to appear in (+0.020). The
    same ceiling shows in effective rank: 8× the neurons buys 15.8 → 20.5
    dimensions with binding off, 9.3 → 10.4 with it on. Re-running this matrix at
    1024 on delayed XOR would produce two numbers near 1.0 and no information.
    **The capacity question above 192 needs 3-cue parity or another task with
    more to say**, which is what the rest of this item is about.
  - **Binding may have a fixed dimensional ceiling.** `on` eff_rank is 9.3, 9.6,
    10.3, 10.4 across a factor of eight in column size. Sweep 020 read the
    halving at 96 neurons as consolidation; four sizes make it look like a
    ceiling near ten that merely happens to be half of 17.6 at `N = 96`. Good
    while the readout is starved, and exactly what would stop the mechanism
    scaling to a task needing more than ten dimensions. Whether the ceiling
    belongs to binding or to delayed XOR is unmeasured and is the first thing a
    harder task would answer.
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
  - **The simple answer.** Characterise what θ and the knee actually encode.
    Sweep 027 answered half of it: **the operating point is a per-neuron fit,
    not a recipe.** Permuting the values between neurons — same distribution,
    wrong owners — costs 0.130 on 0/20 seeds at p = 0.0000, and the permuted
    column fires at double target rate. So it cannot be sampled at
    initialisation, and 023's "precomputable" means *from any input*, not
    *without the column*. θ carries +0.181 of the +0.221; the knee adds +0.041.

    **Answered — θ is a power law in the neuron's own membrane time constant.**
    The probe correlated each neuron's settled θ against its own properties and
    τ carries it: `θ = 5.927·τ^−0.748`, log-log r = **−0.974** over six seeds,
    90% of θ's variance from one static construction-time property.

    Sweep 029 then used the formula instead of settling, at 20 seeds. `tau-init`
    reaches **0.757**, null against a settled twin's thresholds (−0.006, 8/20,
    p = 0.7672), where ordinary settling starts at 0.582 and needs 100 episodes
    to reach 0.755. **The +0.197 is available at construction time from a
    one-line function of a local static property**, so it is neither learning nor
    emergence — it is a fixed point the neurons have by construction that the
    homeostatic loop finds by search because nothing told it the answer.

    Two things this leaves open, and they are now the live half of this item:

    - **The knee is not computable.** `tau-init-knee` reaches 0.786, level with
      the full settled operating point (p = 0.1997), but its knee is copied from
      a settled twin. `preset-knee` alone sits at 0.566 with sparsity 0.002, so
      the two must be set together and only one has a formula. **Run the same
      probe for the knee**: correlate each branch's settled knee against its own
      properties. Cheap, and the obvious next thing.
    - **The exponent is fitted, not derived.** −0.748 came from six seeds at 96
      neurons on one task. Refit at 48 and 192 against the same references —
      `--theta-from-tau 1` makes this nearly free — and if the exponent moves
      with column size it is a property of this configuration rather than of the
      neuron model.

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

## 4. Sparse synapse updates (was: event-driven execution)

*In plain terms: it looked like most of the work was wasted on neurons doing
nothing, and that fixing it would be worth ~50×. Measuring it moved the target
twice. Most neurons are NOT idle — 81% receive input every step even though only
2.6% fire — so waking neurons selectively saves almost nothing. The real waste is
one level down: 97% of the individual connections carry nothing on a given step.
Skipping those is worth a measured 5× on that operation, which works out to about
2× on the whole thing at current size, and more as the network grows. Still worth
doing. Not the headline it was written as.*

**The framing this item carried was wrong, and measuring it moved the target.**
It said "every neuron updates every millisecond whether or not anything reached
it; at a 3% firing rate that is ~97% waste". That conflates *firing* with
*receiving input*. Measured over 12,000 steps of real episodes:

| level | receives an event each step | skippable |
|---|---|---|
| neurons | **81.2%** of 96 | 19% |
| branches | 33.2% of 768 | 67% |
| synapses | **2.99%** of 12,288 | **97%** |
| *(neurons that actually fire)* | *2.65%* | — |

Only 2.65% fire, but **81% receive input and so genuinely need their membrane
advanced**. A neuron-level scheduler — the thing this item described — would
skip 19% of the work and would not be worth the rewrite. Fully silent steps are
rarer still: 0.9%.

**The 97% is at the synapse level**, which is a different mechanism entirely:
not a scheduler that wakes neurons, but a reverse index from each emitting
source to the synapse slots that read it, scattering ~370 updates instead of
multiplying through a 12,288-entry tensor that is 97% zeros. Fan-out is ~110
synapses per source and ~3.4 sources emit per step, which is exactly the 367
measured — so the arithmetic closes.

**Measured, and the honest ceiling is far below what this item claimed.**

Branch integration, dense against sparse-scatter at the real 3% occupancy:

| | µs/step | |
|---|---|---|
| dense `(W*x).sum(axis=2)` | 28.8 | |
| sparse `np.add.at` | **5.6** | **5.15×**, results identical |
| sparse `np.bincount` | 6.5 | 4.46× |

So a 33× reduction in arithmetic buys **5×** in wall clock — the dense op is
cache-friendly and the scatter is not.

Then the share that op holds of a whole step, from `cProfile` over 4000 steps at
96 neurons (399 µs/step total):

| | µs | share |
|---|---|---|
| `step()`'s own inline `(N,B,S)` work — `pre`, `eps`, `elig`, `W*x` | 159 | 40% |
| transport `take` | 42 | 11% |
| ufunc `reduce` (the `axis=2` sums) | 38 | 10% |
| `_phi` (operates on `(N,B)`, not `(N,B,S)`) | 35 | 9% |
| short-term plasticity | 29 | 7% |
| `_emit` | 27 | 7% |

`(N,B,S)`-shaped work is ~60% of a step. Sparsifying all of it at the measured
5× gives **~1.9× overall at 96 neurons** — not 50×, and not the ~170× the
firing-rate arithmetic suggested.

**So this is not "the single largest lever available", and that line is now
removed.** It is a ~2× win at current size for a substantial rewrite.

What keeps it worth doing is scale, which is the actual deployment target. The
`(N,B,S)` share grows with column size as per-step Python overhead amortises —
already visible in the throughput table above — so at 1024+ neurons the dense
tensor dominates and the achievable factor approaches the full 5×.

Item 8's roofline measurement corroborates this from the other side and puts a
number on it: memory-system utilisation rises from 17 % at 96 neurons to 29 % at
1024, which is per-call overhead amortising rather than bandwidth saturating.
The two items agree that **overhead, not memory, is what this implementation is
currently spending its time on** — which is why a rewrite is worth more here than
a faster machine. The item is
re-scoped from "the big win" to "a 2–5× win that matters more the bigger the
column gets", and it should be sequenced accordingly.

It is *exact* for this model rather than an approximation, for two specific
reasons — and **both are now tested rather than argued**, because either being
false would kill the design after it was written:

- Every filter is a pure exponential, so state can be caught up over a gap in
  closed form (`b *= decay**Δt`). `test_filter_catchup_over_a_gap_equals_stepping_through_it`
  checks this against every real time constant in the model at gaps up to 500
  steps — the worst case a scheduler would skip, against the 320 ms membrane.
- Decay is monotone downward, so a neuron cannot cross threshold without input.
  Not self-evident: the soma sums through a plateau nonlinearity, so a positive
  resting drive would let potential rise with no input at all. (It once also
  added `self.bias`, since deleted as a constant zero — the guarantee is now
  structural rather than dependent on that term staying unwritten.) `test_a_silent_neuron_cannot_reach_threshold` cuts the input,
  waits for the delay lines to drain, and asserts zero emissions and a falling
  membrane.

**Measured, and the case is stronger than it was stated.** Four column sizes on
this container, 3000 steps each after warm-up:

| neurons | synapses | steps/s | Msyn-upd/s | neurons at 1× real time | silent |
|---|---|---|---|---|---|
| 48 | 6,144 | 3,639 | 22.4 | 175 | 99.5% |
| 96 | 12,288 | 2,567 | 31.5 | 246 | 99.4% |
| 192 | 24,576 | 1,905 | 46.8 | 366 | 99.4% |
| 384 | 49,152 | 1,217 | 59.8 | 467 | 99.5% |

Throughput rises with column size — the per-step Python overhead amortises over
a bigger tensor — so the earlier "~64M/s, ~500 neurons" figure was the *large*
end of the range, not a constant. Quote the row, not a single number.

**99.4% of neuron-updates are of units that did not fire**, against the ~97%
previously written here. At a 0.6% firing rate the theoretical ceiling for
event-driven is ~170×, though the real figure will be far lower once scheduling
overhead is paid.

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
| ~~`modulator_lag` at 20 seeds, `drain_steps` pinned~~ | **Measured — sweeps 026 and 030.** At the default the window is **12 steps** and binding is null from lag 25 on, set by the *product* of `act_fast` (τ 50) and the branch filter (τ 15) — `lagwindow.py` measures 11.87 / 15.25 / 28.68 against a product prediction of 11.54 / 14.15 / 27.27. Sweep 030 then bought it back: at `tau_branch 60` binding pays **+0.062 at lag 25 (20/20, p = 0.0000)** and **+0.037 at lag 50 (p = 0.0000)** against a matched reference. Regional latency works; 150 steps does not. Giving the rule its own trace (`bind_tau_pre 300`) is **refuted** — null at lag 150 (p = 0.0794) and −0.099 at lag 0. |
| `PRETRAIN=0` | Rests solely on sweep 004, which ran before the eligibility-normaliser fix with the effective learning rate ~5× high. The reasoning behind pretraining was sound and it has never been tested under a correct rate. |
| `elig_gate=1.0` | Expressed as a multiple of the trace RMS, chosen when that RMS was tracking 4.6e-3 against an actual 2.4e-2. The number was picked against a scale that no longer exists. |
| `tau_eligibility=70` | Matched to a readout that decided on a window average. It now decides on the trace itself. The justification no longer describes the thing it was matched to. |
| `elig_norm="column"` | The docstring argues for it at length; nothing measured which of the three normalisers is better, only that they differ. |

Divergent defaults, where the library ships one thing and every experiment runs
another: `surrogate` is `"window"` (the version a fix was written *against*)
while experiments use `"graded"`; `lr` is 0.004 against a swept 0.005; and
`n_neurons=256` has never been measured at any point in this project.

~~`self.bias` is allocated, added to the somatic drive, and written by nothing.~~
**Deleted.** It was a constant zero added to the soma on every step of every
run, at every configuration — the project's signature failure in its purest
form, a quantity that looks connected and is not. Removing it left `W`
bit-identical over five training episodes, which is both the proof it never did
anything and the reason nothing noticed.

It had been *recorded* in a test rather than removed, and that test passed for
the life of the project while the term sat there. So the fix is over the class:
`test_every_column_array_is_either_written_or_declared_constant` enumerates
every array on the column, drives a run with every mechanism enabled including
`lr_tau`, and requires each one to be either written by something or named in an
explicit `CONSTANT` list with its reason. A new dead quantity fails as
unclassified; a constant that becomes live fails too, so the list cannot rot in
either direction.

The check found exactly one dead quantity, and its own first version was wrong
in an instructive way: it listed the column's arrays once at construction, so an
array created *during* a run was invisible to it — and the mutation written to
verify it duly escaped. It now re-scans every step.

## 7. Distribution, for real

*In plain terms: the entire point of the project is running this spread across
ordinary machines over the internet. That has never actually been done — every
"distributed" measurement so far is one process pretending to have network
delays. The make-or-break number is how much traffic crosses between machines.*

`NetworkTransport` does not exist. Everything distributed has been measured
through `LocalTransport` with simulated delays, which is the right way to
develop it and not the same as having done it.

**One thing a real transport would have had to get right is now pinned rather
than argued.** `events.py` has always claimed that indexing by emission time
makes jitter benign — a late packet lands in the correct slot of history, so a
distributed run computes what a local one does. That was tested only at the
primitive level: one buffer, two out-of-order events. It is now tested end to
end. Delaying every published slice by a random 0–N steps, and applying
everything released on the same tick in shuffled order, leaves a three-column
run **bit-identical in its learned weights** for any lateness below `delay_min`
— and changes them past it.

So the tolerance is exactly `delay_min - 1` steps and not one more, because a
packet emitted at `t` is first read at `t + delay_min`. A real transport does
not need ordered delivery or bounded latency in general; it needs delivery
inside that window, which is a much weaker requirement and now a stated number
rather than a hope. It is also a **design lever nobody has used**: raising
`delay_min` buys jitter tolerance directly, and what that costs the column has
never been measured.

- **Wiring locality is the make-or-break number.** Per machine, 33M neurons at
  2% and 1 kHz emit ~6.7×10⁸ events/second. At 1% of synapses crossing the
  network that is ~53 MB/s, which home broadband carries; at 10% it is 530 MB/s,
  which it does not. `peer_frac` sets this directly and has never been swept
  against a bandwidth budget.
- **Batch events over the latency budget.** 150 ms of tolerance is measured and
  free *for conduction delay between columns*, so ~100 ms of events can go in
  one packet. At 1 ms granularity, headers would swamp 4-byte payloads;
  batching amortises them over tens of thousands of events. Latency tolerance is
  not just survivable, it is what makes the packet economics work.

  **This applies to inference traffic only.** Sweep 026 measured the modulator
  path separately and its window is 12 steps, not 150 — so a batching scheme
  that delays the salience broadcast by 100 ms deletes the one learning
  mechanism that works. The two paths have different budgets and a transport
  that treats them as one will silently take learning with it.
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

**"Largely memory-bandwidth-bound" was wrong, and `experiments/roofline.py`
measured it.** On this machine (4-core Xeon, 8 MiB L2, 260 MiB L3, **10.7 GB/s**
sustained streaming bandwidth measured on 512 MB arrays):

| neurons | µs/step | working set | achieved traffic | % of DRAM peak | steps/s | neurons at real time |
|---|---|---|---|---|---|---|
| 96 | 214 | 384 KB | 1.84 GB/s | **17.1 %** | 4684 | 450 |
| 256 | 373 | 1024 KB | 2.81 GB/s | 26.2 % | 2682 | 687 |
| 512 | 702 | 2048 KB | 2.99 GB/s | 27.8 % | 1424 | 729 |
| 1024 | 1332 | 4096 KB | 3.15 GB/s | **29.3 %** | 751 | 769 |

Two things follow, and the second is the one that matters.

**It is not bandwidth-bound.** At the 96 neurons every result in this repo was
measured at, the step uses 17 % of the memory system, and its entire working set
— all eight `(N,B,S)` arrays — is 384 KB against an 8 MiB L2. Even at 1024
neurons the working set is 4 MB and still fits, so the traffic figure above is
mostly served by cache and the true DRAM pressure is lower still.

**Efficiency rises with size, which is the signature of the opposite problem.**
17 % → 29 % as N goes 96 → 1024. A bandwidth-bound program saturates and then
degrades; one limited by per-call overhead amortises, which is independently
what item 4 observed from the other direction.

So the honest position on the GPU counter-argument is **not** "a GPU would win by
50×". It is that **the comparison cannot be made from these numbers at all**,
because the current implementation is nowhere near any hardware limit — it is
spending most of its time in numpy call overhead, and a naive GPU port would
have the same problem in a different language. The first several× is available
on this CPU by fusing the per-step operations, and only after that does the
question "CPU or accelerator" become a question about hardware rather than about
this code.

What *is* now measured, and is what the distribution argument actually needs:
**one core of this machine sustains roughly 450–770 neurons at biological real
time** (1 ms per step). That is the number to multiply by a machine count, and
it is a floor rather than a ceiling given the overhead finding.

(Recorded carefully because this exact quantity was once mis-derived here as
`steps/s × neurons`, which is not neurons-at-real-time and is larger by a factor
of a thousand.)

---

## 9. Claims in the record with no measurement behind them

*In plain terms: things the repo asserts as fact that nobody ever checked. Each
one is either true and worth a number, or false and quietly misleading whoever
reads it next.*

- **Heterogeneous membrane constants are "a large win".** `DESIGN.md` section 2
  and the `README.md` deviations table both say a population with mixed
  constants holds working memory a homogeneous one *cannot*. One of four
  headline departures from biology, stated in strong terms, and never measured.
  `test_membrane_gain_is_independent_of_time_constant` is sometimes read as
  covering it and does not — it guards the unit-DC-gain fix, which was a bug the
  spread *exposed*, not the benefit it claims.

  **Sweep 036 is built and queued**, with four homogeneous conditions rather
  than one so that a badly-chosen constant losing cannot be mistaken for the
  claim holding. Prediction recorded: the spread helps by +0.02 to +0.08 over
  the best homogeneous setting, so directionally right and substantially
  overstated — because STP, not the membrane, is what carries memory here
  (0.527 → 0.864), and it is identical in every condition.

- ~~"Largely memory-bandwidth-bound"~~ — measured and **refuted**, see item 8.
- ~~Emission-time indexing makes jitter benign~~ — measured, and the bound is
  exactly `delay_min - 1` steps. See item 7.
- ~~The fast gather path~~ — every recorded number used it and nothing checked
  it against the correct path. Now asserted bit-identical, with a mutation.

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

  > *Calibration, second instance and much worse.* Stacking sweeps 030, 034,
  > 035 and 036 into `plexus-binding.yml` without retiring anything took a seed
  > from 12 conditions to **48**. Sweeps 026 and 030 accounted for 17 of them —
  > the whole modulator-lag block, both `tau_branch` lag arms and all three tag
  > conditions — every one of which had its answer already written down. Cut
  > back to 31. The failure is easy to miss because each addition is individually
  > justified; nothing prompts the subtraction, so **retire in the same commit
  > that adds**, not later.
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
