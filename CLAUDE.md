# Working agreement for this repo

## Every claim about how the system works must be backed by a test or a measurement

This is the project's central discipline, and it exists because it has been
earned. The worst bugs here were not crashes -- they were *quantities that
looked connected and were not*, and each one silently invalidated every
measurement taken before it was found:

- Synaptic weights were never applied in the forward pass. Plastic and frozen
  columns produced bit-identical states while their weights differed by 25%.
- The dendritic plateau knee sat an order of magnitude above the branch
  potentials, so the nonlinearity never engaged at all.
- The scoring metric updated the readout mid-decision and reported 1.000 train
  accuracy on a model at chance.
- The eligibility-magnitude normaliser was seeded so far from its true value
  that the effective learning rate ran ~5x high for thousands of episodes.
- The engram recruitment criterion was self-normalising per neuron, making it
  structurally blind to the excitability variable meant to drive it.

Every one of those passed a plausible-looking test suite and produced
plausible-looking numbers.

### What this means in practice

1. **Do not state how a mechanism behaves without having measured it.** Not in
   chat, not in a docstring, not in a commit message, not in the README. If it
   has not been measured, say so and say what would measure it.

2. **When adding a mechanism, add the test that would fail if it were
   disconnected.** Not a test that it runs without raising -- a test that its
   input actually reaches its output. Perturb the input, assert the output
   moves. `test_output_depends_on_synaptic_weights` is the template.

3. **Measure the quantity the mechanism claims to change, not a downstream
   proxy.** End-to-end accuracy cannot tell you which of six mechanisms is
   broken. Ten multi-seed sweeps were spent learning that. Prefer a direct
   probe (`experiments/engram.py`, `experiments/gradcheck.py`,
   `experiments/probe.py`) and treat accuracy as a summary, not a diagnosis.

4. **Beware criteria that normalise away their own input.** If a mechanism
   divides by something that moves with the variable it is supposed to respond
   to, it is blind by construction. Ask this explicitly of every ratio.

5. **Correct the record when a measurement refutes a written claim.** The
   README once said events carry graded payloads; an audit found the graded
   component uses 0.32% of its span and the variation comes from short-term
   plasticity instead. The README was changed. Falsified claims get fixed, not
   softened.

6. **Report negative results as results.** `experiments/sweeps/*.txt` is one
   file per sweep, each recording question, prediction, and outcome *including
   the ones that refuted the hypothesis*. A refutation that narrows the search
   is worth more than an unmeasured success.

7. **Multi-seed or it did not happen.** An effect measured at 3 seeds has
   repeatedly decayed to zero by 20. Use `experiments/paired_test.py` (exact
   paired permutation) and the 20-seed CI matrix for anything that will be
   acted on.

8. **When a bug is fixed, re-examine every decision that was taken before it.**
   A fix does not only correct the future; it retroactively removes the
   evidence for choices already made. Defaults, abandoned directions and
   "settled" questions all keep standing on their own long after the
   measurement under them has been invalidated, because nothing in the code
   points back at the run that justified them.

   So a bug fix is not finished when the tests pass. It is finished when
   `experiments/sweeps/AUDIT.md` has been updated with what the fix
   invalidated, and each affected decision has been marked re-validated,
   superseded, or pending.

   The distinction that does most of the work: **a measurement of the frozen
   column is unaffected by a bug in the learning rule; a measurement of the
   plastic condition is not.** Sort by that before assuming the worst.

   Be equally careful in the other direction. A direction abandoned because it
   "did not help" may have been tested through a broken mechanism, and the most
   expensive mistake available here is to permanently discard an idea on the
   strength of a measurement that was never valid.

## Conventions

- The design rule the whole architecture serves: **no operation may require
  globally synchronised state.** A new mechanism that needs a population sort,
  a global mean, or a pooled matrix is a rule violation and must be flagged as
  one even if it improves the numbers.
- New mechanisms default to **off**, so existing results stay reproducible and
  the ablation is free.
- `python -m pytest plexus/tests/test_plexus.py -q` before every commit.
