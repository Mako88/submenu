"""Does a frozen column's state predict its own next input?

IN PLAIN TERMS
--------------
The plan is to change what the network is trying to do. Right now something
outside it tells it the right answer, and that message has to arrive within
about twelve milliseconds or the network cannot use it -- which is the single
biggest obstacle to running this thing spread across the internet.

The alternative is to have each neuron try to predict what it is about to
receive next. Nothing has to be sent, nothing can arrive late, and no labels are
needed -- which matters if this is ever going to run on other people's devices.
It is also, not coincidentally, the same thing a language model does.

But all of that only works if the network's state actually contains a hint of
what is coming. If it does not, there is nothing for a predicting rule to learn
from. **This probe is the cheap way to find that out before building anything.**

WHY THIS IS A GATE AND NOT A DETAIL
------------------------------------
The project has already spent eleven sweeps on learning rules measured against a
task a frozen column mostly solves before learning. That mistake was expensive
and is now item 11. This is the same mistake's shape, one step earlier: building
a predictive mechanism without first checking there is anything to predict.

WHAT IT MEASURES
----------------
For a **frozen** column (`lr = 0`, both mechanisms off), at each horizon `k`, a
decoder reads the column state at `t` and predicts something about `t + k`.
Three targets, because they ask three different questions and only the third is
the one the pivot needs:

    any     is *any* input channel carrying an event at t+k. Dominated by the
            Poisson background, so it is close to unpredictable by
            construction. Reported because it is the honest overall number,
            not because a null on it means anything.
    burst   is a cue or go burst on at t+k -- taken from the episode's burst
            schedule, not reconstructed from the input. This is the *timing*
            question: does the state know where it is in the episode.
    group   *which* of the five burst groups is on at t+k, over burst steps
            only. Chance is the majority-group rate, reported alongside.

**`group` is not yet the content question, and must not be quoted as one.**
Four of the five groups are cue groups carrying a random bit; the fifth is the
go cue, which is **identical in every episode**. A decoder that has learned
only "the go cue comes at a fixed time" scores well on this target without
predicting any content at all, and the go cue is the single commonest group
(45 steps against 25), so it dominates. Splitting `group` into go-versus-cue
(timing) and which-cue-of-the-pair restricted to non-go bursts (content) is
the next change to this file, and no conclusion about the pivot should be
drawn until it is made.

For each target: `linear` (logistic readout), `mlp` (small nonlinear decoder),
`gap = mlp - linear`, and two floors.

THE TWO FLOORS, BECAUSE ONE IS NOT ENOUGH
-------------------------------------------
    base       the majority-class rate on the test split. A decoder that
               predicts the commonest answer and nothing else scores this.
    shuffled   the same decoder trained against a time-shuffled target stream.
               This catches a decoder scoring off the marginal rate *and* any
               leakage through the train/test split.

`base` is here because the first version of this probe reported only
`shuffled`, and the `any` target ran at a base rate of 0.564 -- so a score of
0.549 would have read as "just above the 0.540 floor" when it is in fact
*worse than a constant*. Any number in this table is meaningless without the
`base` column beside it.

WHAT EACH OUTCOME MEANS
-------------------------
  linear high            The substrate predicts its input trivially and a
                         predictive rule adds nothing. The delayed-XOR trap
                         again, one level down.

  gap large, mlp well    The regime the pivot needs: present, not linearly
  above both floors      accessible, and therefore something a local rule could
                         plausibly make accessible.

  everything at the      The state carries no information about its own future.
  floors, `group`        **The pivot is dead as specified.**
  included

  `burst` above the      The state tracks episode *timing* but not *content*. A
  floors, `group` at     predictive rule would learn the clock and nothing
  them                   else. Weaker than it looks, and worth knowing before
                         building anything.

WHY THE BURST MASK IS NOT READ OFF THE INPUT
----------------------------------------------
Background noise lands on the cue channels deliberately, so that a cue must be
recognised rather than merely detected. The consequence for a diagnostic is
that "some cue-group channel is carrying something" is **not** a cue mask: at
the defaults it selects 0.490 of all timesteps against a true burst occupancy
of 0.208, so 57% of what it selects is background. The first version of this
probe used exactly that reconstruction, which made its cue-restricted condition
a measurement of the noise floor wearing a cue-shaped name.

The masks here come from `Episode.cue_active` / `Episode.cue_group`, recorded
from the burst schedule at generation time and never shown to the model.
`test_cue_active_marks_the_bursts_and_not_the_noise_on_the_same_channels`
holds that apart.

STATUS
------
Runs end to end. Timed and smoke-tested at `--quick` (1 seed, 15 episodes),
which is **shape only and not a result** -- the three-seed default has not been
run, and the `group` split described above has not been made. No number from
this file belongs in the record yet.

What the smoke run does establish, because they are properties of the probe
rather than of the model: the horizon-0 control is well above both floors on
`burst` and `group`, so the state *is* decodable and a null at k>0 would have
meant something; and the raw-spike version of this probe sat at the floor even
at horizon 0, which is why it now reads the filtered trace.

COST
----
About 7 minutes at the defaults on one core, measured by timing the pieces:
8s to settle a column, 1s per 15 episodes of trajectory, 0.3s per logistic fit
and 4.8s per MLP fit at 4200x96. The MLP fits are 95% of it, and there are
`3 seeds x 4 horizons x 3 targets x 2 (target + shuffled control)` of them.
Anything that needs to be faster should cut horizons, not rows.

    python3 experiments/predictability.py --quick
    python3 experiments/predictability.py --report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plexus import ColumnConfig, DelayedXOR, Plexus  # noqa: E402
from probe import logistic, logistic_score, mlp  # noqa: E402

OUT = Path(__file__).resolve().parent / "predictability_results.jsonl"


def trajectory(model, task, episodes: int, rng):
    """Column states and what was happening beside them, aligned in time.

    The column is observed and not disturbed: `learning` off throughout, so
    nothing adapts to the fact that it is being watched.

    `cue_active` and `cue_group` come from the episode's burst schedule rather
    than from its inputs -- see the module docstring for why reconstructing
    them from the input measures the noise floor instead.
    """
    col = model.column
    col.learning = False
    model.readout.learning = False
    states, inputs, active, group = [], [], [], []
    for _ in range(episodes):
        ep = task.episode(rng)
        col.reset_state()
        model.readout.reset()
        model.transport.reset()
        for k in range(ep.inputs.shape[0]):
            t = model._t
            model._t += 1
            # The *filtered* trace, not the raw spike vector. At the operating
            # sparsity of ~0.02 a single timestep of `col.step` output has 2 of
            # 96 units on, and a decoder over it cannot read the burst that is
            # happening *right now*, let alone a future one -- the first
            # version of this probe used the raw output and sat at the floor
            # even at horizon 0. `probe.collect` uses the trace for the same
            # reason; this is the column's state as the readout sees it.
            states.append(model.readout.observe(col.step(t, ep.inputs[k])).copy())
            inputs.append(ep.inputs[k].copy())
        active.append(ep.cue_active)
        group.append(ep.cue_group)
    return (np.array(states, dtype=np.float32),
            np.array(inputs, dtype=np.float32),
            np.concatenate(active), np.concatenate(group))


def shuffle_within(y, mask, seed: int):
    """Time-shuffle the target, permuting only the rows the mask will keep.

    Shuffling the whole array and masking afterwards is not the same control.
    For the `group` target the mask keeps burst steps only, roughly a fifth of
    the stream; a global shuffle would fill those rows with values drawn from
    the four fifths that are outside any burst, so the control would face a
    different class balance from the real target and its score would not be a
    floor for it. Permuting inside the mask holds the balance fixed and varies
    only the alignment to time, which is the one thing the control is for.
    """
    out = y.copy()
    rng = np.random.default_rng(seed)
    if mask is None:
        rng.shuffle(out)
        return out
    idx = np.flatnonzero(mask)
    out[idx] = out[rng.permutation(idx)]
    return out


def score(states, targets, mask=None, n_classes: int = 2, cap: int = 6000,
          seed: int = 0) -> tuple[float, float, float]:
    """Linear accuracy, MLP accuracy, and the majority-class floor.

    The third return value is not decoration. It is the score a decoder gets
    for ignoring its input entirely, and on the `any` target it sits at 0.564
    -- above what either decoder achieves. A linear/MLP pair reported without
    it reads as a weak positive when it is a negative.

    Subsampled to `cap` rows. A run of 60 episodes produces ~27,000 timesteps
    and this probe fits many decoders; uncapped it does not finish in a
    sensible time. Six thousand rows is far more than is needed to separate a
    real effect from the floors, and the cap is applied *before* the
    train/test split so both sides shrink together.

    Sampled without replacement rather than truncated: the trajectory is
    episode-ordered, so taking the first N rows would sample a handful of whole
    episodes instead of the run.
    """
    if mask is not None:
        states, targets = states[mask], targets[mask]
    if len(states) < 200 or targets.min() == targets.max():
        return float("nan"), float("nan"), float("nan")
    if len(states) > cap:
        idx = np.random.default_rng(seed).choice(len(states), cap, replace=False)
        idx.sort()          # keep time order, so the split stays a time split
        states, targets = states[idx], targets[idx]
    split = int(0.7 * len(states))
    Xtr, Xte = states[:split], states[split:]
    ytr, yte = targets[:split], targets[split:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    # The floor is read off the *test* split using the *training* majority, so
    # it is a number a real predictor could have achieved rather than an
    # oracle's.
    majority = np.bincount(ytr, minlength=n_classes).argmax()
    base = float((yte == majority).mean())
    lin = logistic_score(logistic(Xtr, ytr, n_classes=n_classes), Xte, yte)
    return float(lin), float(mlp(Xtr, ytr, Xte, yte, n_classes=n_classes)), base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 20, 50])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--neurons", type=int, default=96)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--settle", type=int, default=100)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.report:
        rows = [json.loads(x) for x in OUT.read_text().splitlines() if x.strip()]
        print(f"{'target':>8s}{'horizon':>8s}{'base':>8s}{'shuf':>8s}"
              f"{'linear':>9s}{'mlp':>8s}{'gap':>8s}{'over':>8s}{'seeds':>7s}")
        for target in ("any", "burst", "group"):
            for k in sorted({r["horizon"] for r in rows if r["target"] == target}):
                sel = [r for r in rows
                       if r["horizon"] == k and r["target"] == target]
                g = lambda f: np.nanmean([r[f] for r in sel])  # noqa: E731
                floor = max(g("base"), g("shuffled_mlp"))
                print(f"{target:>8s}{k:8d}{g('base'):8.3f}{g('shuffled_mlp'):8.3f}"
                      f"{g('linear'):9.3f}{g('mlp'):8.3f}"
                      f"{g('mlp') - g('linear'):8.3f}"
                      f"{g('mlp') - floor:+8.3f}{len(sel):7d}")
        print("\n`over` is mlp minus the HIGHER of the two floors -- the "
              "majority-class rate\nand the time-shuffled control. Only a "
              "positive `over` is predictive information.\n`gap` is the part "
              "of it a linear readout cannot reach, which is the part a\n"
              "representation-learning rule could have a job closing.")
        return

    seeds = 1 if args.quick else args.seeds
    episodes = 15 if args.quick else args.episodes
    task = DelayedXOR()

    for s in range(seeds):
        model = Plexus(
            task.n_inputs, task.n_classes,
            column=ColumnConfig(n_neurons=args.neurons, lr=0.0, seed=s),
            seed=s,
        )
        model.train(task, args.settle, rng=np.random.default_rng(1000 + s),
                    report_every=10**9)
        states, inputs, active, group = trajectory(
            model, task, episodes, np.random.default_rng(21 + s))

        # Three targets, weakest question first. See the module docstring.
        #   any    -- is anything arriving. Mostly Poisson, mostly unpredictable.
        #   burst  -- is a scheduled burst on. The timing question.
        #   group  -- which burst. The content question, and the only one whose
        #             answer would justify building a predictive rule.
        n_groups = 2 * task.n_cues + 1
        targets = {
            "any": ((inputs > 0).any(axis=1).astype(int), None, 2),
            "burst": (active.astype(int), None, 2),
            "group": (np.maximum(group, 0).astype(int), active, n_groups),
        }

        for name, (y, mask, n_classes) in targets.items():
            # Horizon 0 is a control and not a result: it asks whether the
            # state encodes what is happening *now*. Without it a null at k>0
            # is unreadable, because "the future is not predictable" and "this
            # state is not decodable by this probe" produce the same table. It
            # is prepended rather than left to the caller so it cannot be
            # omitted from a run whose numbers then get quoted.
            for k in [0] + list(args.horizons):
                end = len(states) if k == 0 else -k
                m = None if mask is None else mask[k:]
                lin, non, base = score(states[:end], y[k:], mask=m,
                                       n_classes=n_classes)
                _, sh, _ = score(states[:end], shuffle_within(y[k:], m, 777 + s),
                                 mask=m, n_classes=n_classes)
                row = dict(target=name, horizon=k, seed=s,
                           neurons=args.neurons, episodes=episodes,
                           linear=lin, mlp=non, base=base, shuffled_mlp=sh)
                with OUT.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(f"{name:>6s} k={k:3d} seed={s}  base {base:.3f}  "
                      f"shuf {sh:.3f}  linear {lin:.3f}  mlp {non:.3f}  "
                      f"gap {non - lin:+.3f}")


if __name__ == "__main__":
    main()
