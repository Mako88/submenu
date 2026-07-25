# What each live decision rests on, and whether that evidence survived

Bugs in this project have a habit of invalidating measurements retroactively.
The forward-pass bug voided everything before it; the eligibility normaliser
"mis-scaled every sweep" up to that point; the surrogate fix changed the
quantity the learning rule was reading. Each fix silently removed the ground
under decisions that stayed in force anyway, because nothing in a default value
points back at the run that chose it.

This file is that pointer. Every entry names a decision still in force, the
measurement behind it, and whether any later fix invalidated that measurement.

## The fixes, in order

| # | commit | fix | what it invalidates |
|---|--------|-----|---------------------|
| 10 | `444f372` | forward pass ignored `W` entirely | **everything before it**, without exception |
| 24 | `15c143f` | `PRETRAIN` env default made sweep 003 test nothing | sweep 003 only |
| 27 | `ac364c0` | eligibility normaliser ran the effective lr ~5x high and drifting | every **plastic-condition** result before it |
| 30 | `14f28a8` | surrogate was the source of the trace's magnitude noise | plastic-condition results before it, weakly |
| 41 | `8298b48` | engram recruitment blind to excitability; wrong statistics timescale | engram probes before it (none published) |
| 55 | sweep 023 | the trajectory probe's sparsity column read `column.sparsity`, which is the firing-rate EMA and only advances while `learning` is True — so at checkpoint 0 it returned `target_rate` unchanged from initialisation | **sweep 022's episode-0 sparsity row only.** It reported 0.0300 for every condition; measured, the untrained column runs at **0.0022**. Rows from episode 10 onward were taken after training episodes, where the EMA does advance, and stand. The conclusion is unaffected and sharpened: the operating point is largely a firing rate, and homeostasis's first job is lifting a near-silent column to target |
| 54 | sweep 021 | catch-up froze only `hebbian`, so a second plasticity rule would have kept running through a condition whose whole claim is that the column stopped | **nothing** — every catch-up condition ever run had `lateral` off and `lr=0`. Recorded because the omission was structural, not because it leaked: `freeze_plasticity` now clears every registered flag and a test fails if a new boolean is left unclassified. |

**The sorting rule.** A frozen column (`lr=0`) never executes the learning
rule, so `ac364c0` and `14f28a8` cannot touch a frozen measurement. Anything
measured on the frozen column, on the task itself, or on the dynamics is
unaffected by those two. Only `444f372` voids frozen results as well, because
a column that ignores its own weights is a different column.

**And the limit of that rule, because it has been over-read twice.** It says a
frozen measurement is immune to *learning-rule* bugs. It does **not** say frozen
results are stable. A frozen column runs threshold homeostasis, knee adaptation
and synaptic scaling, so it moves whenever its input distribution moves — sweep
013 lost a whole run to this (a frozen column shifted −0.062, p = 0.0011, when
`modulator_lag` changed the length of the silent drain tail), sweep 019 found a
frozen column is not an untrained one (+0.197 from settling alone), and sweep
024 found a frozen column *forgets* — at 20 seeds, −0.015 of task A after
training on task B. Three instances of one confusion.

(The −0.167 first written here was a single seed and is corrected: 20 seeds put
it at −0.015. The mechanism stands, the magnitude did not. See sweep 024.)

The distinction that actually holds is now tested rather than argued:
`test_a_frozen_column_is_identical_across_readout_rules` pins the pairing behind
the RLS result — the readout rule does not change the column's input, so the
frozen column is bit-identical across sweeps 006 and 010 — and
`test_a_frozen_column_still_changes_when_the_input_changes` pins the other side.
Before a frozen measurement is called stable, ask whether the change under
comparison alters what the column *sees*.

## Decisions still in force

### Valid — evidence survived, or was re-established after the fix

| decision | where | rests on | why it stands |
|---|---|---|---|
| The settling operating point is precomputable, and better computed from structureless input | sweep 023 | 20 paired seeds | +0.051 over a task-settled twin (p = 0.0032) and +0.048 over 100 episodes of settling (p = 0.0049). Frozen column, `lr=0`, so no learning-path fix can reach it. Not explained: the firing-rate ordering does not match the decodability ordering, so it is not calibration |
| Binding and lateral inhibition replicate on a second channel mapping | sweep 024 | 20 paired seeds | Every prior number for these mechanisms came from one sensory mapping. On a permuted one: binding +0.069 on task A over nothing, 20/20 seeds, p = 0.0000; acquisition off 0.269 → both 0.353 |
| No mechanism forgets differently from any other | sweep 024 | 20 paired seeds | All four retention comparisons null (p = 0.32–0.81), every condition between −0.015 and −0.030. **At two tasks and 96 neurons there is no interference to study**, which is why the engram rematch did not trigger — and a narrower statement than "binding does not interfere" |
| `NEURONS=24` for plasticity sweeps | `plexus-experiment.yml` | sweep 002 | The *choice* of 24 came from a frozen-column headroom measurement (linear 0.689 vs MLP 0.881), which no learning-rule bug can touch. Its null result predates `ac364c0`, but sweeps 006/007/008 re-established the same null at the same size after both fixes. |
| `elig_mode="magnitude"` | `ColumnConfig` | sweeps 006, 007 | Ran after `ac364c0`; sweep 008 re-ran magnitude together with the fixed surrogate and found the same null. Post-both-fixes. |
| `readout_rule="delta"` | `network.py`, `distributed.py`, workflow | sweep 011 | Most recent decision in the project. Post-everything. |
| STP is decisive for memory (0.527 -> 0.864 decodable) | README, `probe.py` | frozen-column probe | Frozen measurement. Unaffected by `ac364c0`/`14f28a8`. |
| Unit DC gain, scale-free plateau knee, Dale's law, 20% inhibitory | `column.py` | direct unit tests | Tested by assertion, not by sweep. Re-checked on every run. |
| Rate-code and time-binned-linear baselines (0.494, 0.486) | `baselines.py` | task-only measurement | Involves no column at all. |
| Conduction-delay tolerance: 2ms -> 150ms costs -0.048 | README, `run-latency.txt` | latency sweep | `latency.py` defaults `--lr 0.0`, so this ran on **frozen** columns and neither learning-path fix can touch it. Valid -- but narrower than the claim it is used to support; see below. |
| Three-factor rule does not beat a frozen column | sweeps 006-010 | five post-fix nulls | The headline negative result was re-established after every relevant fix. This one is solid. |
| Salience-gated Hebbian binding beats a frozen column, 0.876 vs 0.802 offline | sweep 014 | 20 paired seeds | Sweep 015 reproduced both numbers exactly after the mechanism was rewritten, which confirms the refactor rather than replicating the effect -- same seeds, same deterministic computation, so it is one measurement reported twice. |
| Binding pays end to end, but only against a settled representation *and* only with extra episodes to pay for it | sweeps 017, 018 | 20 paired seeds each | +0.063 with binding frozen and the readout allowed to converge (p = 0.0013), +0.043 under an RLS readout (p = 0.0063), +0.007 under the shipped delta readout during training (p = 0.68). The last of those is the default configuration, so the headline gain is NOT what the shipped model does today. Sweep 018 showed the gain is unavailable inside a fixed 300-episode budget, and sweep 019 explained why: the representation is complete by episode 150, so binding was never the bottleneck -- the delta readout needs ~300 episodes against a settled column. Quote it as a trade, and attribute the cost to the readout rather than the mechanism. |
| The engram allocator is worse than binding alone | sweep 014 | 20 paired seeds, p = 0.0001 | Why the mechanism was deleted. Note what it does *not* say: delayed XOR asks nothing of memory separation, so this refutes allocation on this benchmark only. |
| Binding works by **consolidating**, not by decorrelating | sweep 020 | 20 paired seeds, p = 0.0000 on all three metrics | Correlation +0.077 (20/20 seeds), effective rank -7.763 of 17.4 (0/20 seeds), linear +0.074. Frozen column throughout, so no learning-path fix can reach it. This is the measurement that refuted the premise behind the whole lateral-inhibition direction. |
| Lateral inhibition pays end to end **under RLS only** | sweep 021 | 20 paired seeds | 0.842 against 0.803 for binding alone (+0.040, p = 0.0284), the best number in the project. Null under the shipped delta readout (+0.016, p = 0.33) and under a catch-up phase (+0.005, p = 0.60), and reliably worse than RLS there (−0.071, p = 0.0012). **The default ships delta, so 0.842 is not what the model does today.** Frozen column, `lr=0`, so no learning-path fix can reach it. |
| Local decorrelation cannot replace RLS's pooled matrix | sweeps 020, 021 | 20 paired seeds each | Premise refuted in 020 (lateral does not decorrelate), payoff refuted in 021 (it does not approach RLS under a local readout). The direction is closed, not parked. RLS remains the one genuine locality exception in the model. |
| Lateral inhibition helps the representation | sweep 020 | 20 paired seeds | +0.028 alone (p = 0.0297) and +0.037 on top of binding (p = 0.0005). Kept on this measurement and **not** on its rationale: the same sweep found it moves neither decorrelation metric in the predicted direction. What it is doing is unmeasured — do not write an explanation into the code until one exists. |

### Stale — decision still in force, evidence no longer valid

| decision | where | rests on | problem |
|---|---|---|---|
| `PRETRAIN=0` | `plexus-experiment.yml` | sweep 004 **only** | Sweep 004 ran before `ac364c0`, so "a converged teacher does not rescue the rule" was measured with the column's effective learning rate ~5x high and drifting. Never re-tested. This is the clearest case in the project: a live default whose sole evidence is a miscalibrated run. |
| `elig_norm="column"` rationale | `ColumnConfig` docstring | reasoning, not measurement | The docstring argues at length that `"neuron"` amplifies the noise of barely-participating units. That is an argument, not a result; the three normalisers were only measured against each other for *whether* they differ (a connection test), never for which is better. |
| `elig_gate=1.0` | `ColumnConfig` | chosen while `elig_rms` was miscalibrated | The gate is expressed as a multiple of the trace's own RMS. `ac364c0` changed what that RMS *is* -- it had been tracking 4.6e-3 against an actual 2.4e-2. So the number 1.0 was chosen against a scale that no longer exists. Exercised post-fix by sweep 007, never re-tuned. |
| `tau_eligibility=70.0` | `ColumnConfig`, workflow | matched to the readout's ~60ms filter | The match was made when the readout decided on a *window average*. It now decides on the trace itself, which weights time differently. The justification no longer describes the thing it was matched to. |

### Divergent — the library says one thing, every experiment does another

| what | library default | what actually runs | note |
|---|---|---|---|
| `surrogate` | `"window"` | `"graded"` everywhere | `14f28a8` identified `window` as the source of the trace's magnitude noise and `graded` as the fix. The fixed version never became the default. Sweep 008 found the two indistinguishable end-to-end (-0.001, p=0.97), so this changes no result -- but it means the default ships the version a fix was written against. |
| `lr` | `0.004` | `0.005` (`PLASTIC_LR`) | No recorded reason for the difference. |
| `n_neurons` | `256` | `24` or `96` | The default has never been measured at any point in this project. |

### Not to be quietly discarded

Directions dropped on evidence that later turned out to be invalid, listed so
that "we already tried that" cannot be said about them without a re-run:

- **Pretraining the readout before column plasticity.** Dropped on sweep 004,
  which is stale (above). The reasoning was good -- a weak readout is a noisy
  teacher and the column faithfully follows it -- and it has not been tested
  under a correctly scaled learning rate.
- **Sign-only eligibility updates at a tuned gate.** Sweep 007 tested
  `elig_mode="sign"` at `elig_gate=1.0`, a value chosen against the old RMS
  scale. The mode was tested; the gate was not.

## Claims broader than the measurement under them

Distinct from staleness, and easier to miss: the measurement is valid, it just
does not cover what it is cited for.

- **Latency tolerance.** Two separate claims travel under this name. One is
  that *conduction delay* between columns is free -- measured, frozen columns,
  valid, and honestly not very surprising, since a longer delay on a frozen
  reservoir is close to just a different random projection. The other is that
  *the eligibility traces absorb a late modulator*, so credit assignment
  survives a wide-area network. That is the hard half and the one the
  architecture actually rests on, and **it has never been run at 20 seeds.**
  `modulator_lag` appears as a condition in `ablation.py` and as a workflow
  input defaulting to 0; every recorded sweep ran it at 0. The README should
  not state the second claim on the strength of the first.

- **"No operation requires globally synchronised state."** True of the column
  by construction and tested. The readout's `decide()` standardises per
  neuron, which is local. But sweep 010 briefly shipped a globally pooled RLS
  matrix, and it took sweep 011 to notice and revert it. The rule holds today;
  it is not currently enforced by anything except attention.

## Pending re-validation

| item | how | status |
|---|---|---|
| What lateral inhibition is actually doing | principal-direction projection | Built to decorrelate; sweep 020 measured it not decorrelating while helping, and sweep 021 found its gain fully recoverable by RLS (ratio 1.08) and not at all by a catch-up phase (0.14) — the inverse of binding's ordering. The hypothesis is that it adds information in correlated directions, readable by whitening and not by time. **Unmeasured, and the mechanism is kept on its measurement rather than on any of this** |
| What theta and knee actually carry | new probe | **Answered in the wrong direction and now wide open.** Sweep 023: an operating point found on timing-destroyed input beats one found on the real task (+0.051, p = 0.0032) and beats 100 episodes of ordinary settling (+0.048, p = 0.0049). Presetting from the task is null against ordinary settling (p = 0.91), so the whole gain is *what the twin saw*. The obvious mechanism — better rate calibration — is refuted by the sweep's own sparsity column: `all` sits exactly on target and decodes 0.048 worse |
| `scaling_lr=2e-2` now has evidence against it | paired test | Sweep 022: removing synaptic scaling gives 0.780 against 0.755 with identical sparsity. Previously untuned-with-no-evidence (below); now untuned-with-evidence-pointing-at-off. Needs a paired test before acting — the curve comparison is not one |
| The residual 0.770-vs-0.876 gap after catch-up | offline probe on the catchup condition | not yet run |
| `modulator_lag` at 20 seeds -- the learning half of latency tolerance | `plexus-experiment.yml`, `modulator_lag=200` | **not yet run; highest value of the three** |
| `PRETRAIN` under a corrected learning rate | `plexus-experiment.yml`, `pretrain=400` | not yet run; held until the engram matrix clears CI |
| `elig_gate` against the corrected RMS scale | new sweep | not yet run |

## How to add an entry

Commit order finds candidates. Reading the code settles them. Never mark an
entry from the dates alone.

The latency result is the worked example of why. It predates two fixes, which
puts it squarely in the suspect pile on chronology; reading `latency.py` shows
`--lr` defaults to 0.0, so the sweep ran frozen and neither fix could reach it.
Chronology said stale, the code said valid, and the code was right. Then
reading it a second time surfaced something the date check would never have
found at all -- the measurement is sound but narrower than the claim built on
it, which turned out to be the more useful finding of the two.

So: date-sort to build the candidate list, then open the experiment and check
what it actually ran. Mark each entry valid, stale, superseded, or pending, and
name the line of code that decides it.
