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

**The sorting rule.** A frozen column (`lr=0`) never executes the learning
rule, so `ac364c0` and `14f28a8` cannot touch a frozen measurement. Anything
measured on the frozen column, on the task itself, or on the dynamics is
unaffected by those two. Only `444f372` voids frozen results as well, because
a column that ignores its own weights is a different column.

## Decisions still in force

### Valid — evidence survived, or was re-established after the fix

| decision | where | rests on | why it stands |
|---|---|---|---|
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
| Lateral inhibition for local decorrelation | new mechanism | **the live question.** Sweeps 011 and 019 both point here: RLS's gain needs global pooling, and 019 shows the readout is the bottleneck rather than the mechanism |
| Why homeostatic settling alone buys +0.197 decodability | new probe | Larger than anything the learning rules have produced, surfaced by sweep 019, and never asked about |
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
