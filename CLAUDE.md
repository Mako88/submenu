# Engineering standards for plexus

This project is trying to find out whether a locality-first neural model can
learn. That only means anything if the measurements are trustworthy, so the
standards below are about one thing: making the code deserve the conclusions
drawn from it.

They are written as commitments, not as warnings. Everyone building something
this fiddly ships a quantity that turns out to be disconnected, or a bound
looser than they meant, or a claim that outran its evidence — that is what
building at the edge of what you understand feels like from the inside. The
answer is not more care. It is a system that catches those things regardless of
how careful anyone was feeling that day, so that attention can go to the actual
problem.

Each rule carries a **calibration** note: the concrete number that shows where
the bar has to sit. Those are here because a standard without a number attached
drifts within a month, and because "wide enough to admit the broken case" is a
lot more useful when you can see how wide that turned out to be. They are
reference data, not a record of anyone's mistakes.

---

## The standard

**Every claim about how the system works is backed by a test or a
measurement.** Everything below is a way of making that true in a specific
situation.

The failure mode this project is built against is not the crash. It is the
*quantity that looks connected and is not* — a mechanism that runs, produces
plausible numbers, and is doing nothing. Those cost more than crashes, because
they invalidate every measurement taken while they were present, silently and
retroactively.

---

## Making claims

**1. State no behaviour that has not been measured.** Not in chat, not in a
docstring, not in a commit message, not in the README. If it has not been
measured, say so and say what would measure it.

**2. Measure the quantity the mechanism claims to change, not a downstream
proxy.** End-to-end accuracy cannot tell you which of six mechanisms is broken.
Prefer a direct probe — `experiments/engram.py`, `experiments/gradcheck.py`,
`experiments/probe.py` — and treat accuracy as a summary, not a diagnosis.

> *Calibration.* Ten multi-seed sweeps went into narrowing a single null result
> from the outcome side. The engram mechanism's three defects were each found in
> one run once the parts were measured directly.

**3. Multi-seed or it did not happen.** Use `experiments/paired_test.py` (exact
paired permutation) and the 20-seed CI matrix for anything that will be acted
on.

> *Calibration.* An effect of +0.059 at three seeds went to +0.023 at eight and
> to zero at twenty. Twice.

**4. Report negative results as results.** `experiments/sweeps/*.txt` is one
file per sweep recording question, prediction, and outcome — including the
outcomes that refuted the hypothesis. Write the prediction down *before* the
run so it cannot be retrofitted. A refutation that narrows the search is worth
more than an unmeasured success.

**5. Correct the record when a measurement refutes a written claim.** Falsified
claims get fixed, not softened.

> *Calibration.* The README described events as carrying graded payloads. An
> audit measured the graded component using 0.32% of its span, with the
> amplitude variation coming from short-term plasticity instead. The README now
> says that.

---

## Building mechanisms

**6. Ship the connection test with the mechanism.** Not a test that it runs
without raising — a test that its input reaches its output. Perturb the input,
assert the output moves. `test_output_depends_on_synaptic_weights` is the
template.

> *Calibration.* Branch integration once summed its inputs unweighted. `W` was
> read by synaptic scaling and by the learning rule and used to compute nothing,
> so a plastic column and a frozen one produced bit-identical states while their
> weight matrices differed by 25%.

**7. Distrust any criterion that normalises away its own input.** If a
mechanism divides by something that moves with the variable it is meant to
respond to, it is blind by construction. Ask this of every ratio, explicitly.

> *Calibration.* Engram recruitment used `act_fast > margin * act_slow`. Making
> a neuron more excitable raises its baseline along with its activity, so the
> criterion cancelled exactly what was supposed to drive it — excitability
> predicted recruitment at −0.40, backwards.

**8. Gather statistics over the objects being scored.** A distribution
collected at one granularity does not describe an object at another.

> *Calibration.* Three instances, same shape. The readout standardised an
> episode-level decision vector using per-timestep variance and sat at chance on
> states an offline decoder read at 0.85. `elig_rms` kept an EMA seeded far from
> its true value and ran the effective learning rate ~5× high for thousands of
> episodes. Engram recruitment scored a neuron's activity at allocation time
> against the distribution of all timesteps, measuring which units are phasic
> rather than what the episode engaged.

---

## Tests

**9. A test must fail when the thing it names is broken — verify that it
does.** Passing is not evidence. Break the mechanism deliberately and confirm
the test notices; `experiments/mutation.py` does this for the mechanisms it
knows about and runs in CI. Add a mutation there when you add a mechanism.

Watch particularly for an assertion on a quantity that something *else* pins.

> *Calibration.* Three tests survived their own mutation.
> `test_plateau_stays_engaged` passed with knee homeostasis fully disabled — its
> 0.01–0.5 bounds admitted the broken case at 0.074, while guarding the exact
> bug where the plateau was silently disconnected. `test_no_runaway_excitation`
> asserted `|v| < 50` against an operating range of 0.49.
> `test_weights_stay_bounded` asserted a clip that synaptic scaling kept W far
> below, so raising the limit to 1e9 changed nothing. A fourth mechanism,
> Dale's law, had no test that its sign was applied at all — inhibition could
> have been deleted from the model with the suite fully green.
>
> `test_hebbian_binding_only_touches_recruited_neurons` was nearly written
> against `W.mean()`, which synaptic scaling holds at `branch_budget /
> n_synapses` regardless of what the binding does.

**The same applies to an experimental condition.** Before running a sweep, ask
of each condition what outcome would *refute* the prediction attached to it. If
the predicted outcome is guaranteed by how the condition is built, it is not
evidence however it comes out — and it will read as confirmation.

> *Calibration.* Sweep 022's `preset-frozen` was predicted to be "flat
> throughout", with a climb naming the alternative hypothesis. Nothing in that
> condition can change: `lr=0`, all three adaptations disabled, fixed probe
> seed. The curve was identical to three decimals at all eleven checkpoints
> because it was the same column measured eleven times. What carried the finding
> was its *level* at the first checkpoint, 0.752 against 0.582 — a real
> comparison — and `theta-preset`, the condition that could have climbed and did
> not.

**10. A failing test is a claim about the production code until shown
otherwise.** Fix the code so the assertion holds. Widening a bound, deleting an
assertion or special-casing the input converts a caught bug into a silent one
and destroys the evidence that it existed.

Changing the test is right when the intended behaviour genuinely changed. Then
say what measurement or decision changed it, and **split rather than loosen** —
keep an assertion for the old path where it still applies, add one for the new.

> *Calibration.*
> `test_excitability_reaches_the_threshold_when_that_path_is_enabled` and
> `test_default_keeps_excitability_out_of_the_threshold` are that split: the
> default moved for a measured reason, so the original assertion was kept behind
> the flag that still enables it.

A test that passes while *vacuous* is the opposite problem — there the test is
what is wrong, and strengthening it is the fix. Rule 9 covers those.

---

## Keeping the record straight

**11. A bug fix is not finished when the tests pass.** It is finished when
`experiments/sweeps/AUDIT.md` records what the fix invalidated, and each
affected decision is marked re-validated, superseded, or pending. A fix does
not only correct the future; it removes the evidence under choices already
made, and those choices stay in force because nothing in a default value points
back at the run that chose it.

Sort before assuming the worst: **a measurement of the frozen column is
unaffected by a bug in the learning rule; a measurement of the plastic
condition is not.** And be equally careful in the other direction — a direction
abandoned because it "did not help" may have been tested through a broken
mechanism. Discarding a good idea on an invalid measurement is the most
expensive error available here.

**12. Put the reasoning where the reader will be standing.** Someone about to
change a threshold reads the test that guards it, not the sweep note. To keep
one rationale from drifting across four files:

| where | what belongs there |
|---|---|
| Code comment | Why this line is this way. Short, and only where it would otherwise read as arbitrary. |
| Test docstring | What breaks if this stops holding, with the concrete number from when it did. The history lives here. |
| `experiments/sweeps/*.txt` | Question, prediction made before the run, result. Never edited afterwards except to record the outcome. |
| `AUDIT.md` | What a later fix invalidated, and which decisions still rest on it. |

When a number appears in more than one, **the test docstring is canonical** —
it is the one under continuous execution.

---

## Adding to this document

**13. Write the standard, not the incident.** A rule earns its place by telling
someone who was not there what to do next time. So:

- **Lead with the commitment.** "Statistics are gathered over the objects being
  scored" — not "we once standardised with the wrong variance."
- **Attach the calibration, keep it subordinate.** The number is what makes a
  rule enforceable; put it in the aside, not the headline.
- **Prefer a rule that makes the mistake structurally impossible** over one
  that asks for more care. `experiments/mutation.py` is worth more than a rule
  saying "write good assertions", and rule 9 exists mainly to point at it. If a
  proposed rule cannot be turned into a check, say so plainly rather than
  pretending vigilance will hold.
- **Assume good faith and real constraints.** These standards exist because the
  work is genuinely hard, not because anyone was careless. A document that
  reads as a list of accusations gets defended against; one that reads as a bar
  worth clearing gets upheld.
- **Retire a rule when it stops paying.** A standard nobody applies is worse
  than no standard, because it makes the others look optional.

---

## Conventions

- The design rule the whole architecture serves: **no operation may require
  globally synchronised state.** A mechanism needing a population sort, a global
  mean, or a pooled matrix is a violation and gets flagged as one even when it
  improves the numbers.
- New mechanisms default to **off**, so existing results stay reproducible and
  the ablation is free.
- `python -m pytest plexus/tests/ -q` and `python experiments/mutation.py`
  before every commit.
