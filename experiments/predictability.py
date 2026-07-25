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
For a **frozen** column (`lr = 0`, both mechanisms off), at each horizon `k`:

    linear   a logistic readout predicting the input at `t + k` from the column
             state at `t`
    mlp      the same, with a small nonlinear decoder
    gap      mlp - linear

and against a **shuffled control**: the identical decoder trained to predict a
*time-shuffled* input stream. That control is what makes the numbers mean
anything. The inputs here are mostly sparse Poisson noise with a fixed marginal
rate, so a decoder can score above zero purely by learning "channels are usually
silent" without predicting anything. The control measures exactly that floor.

**The target is predicted as a binary event per channel** -- did this channel
carry anything at `t + k` -- rather than as a magnitude. Magnitude is drawn
independently from a fixed range in this task, so it is unpredictable by
construction and including it would only dilute the signal.

WHAT EACH OUTCOME MEANS
-------------------------
  linear high            The substrate predicts its input trivially and a
                         predictive rule adds nothing. The delayed-XOR trap
                         again, one level down.

  gap large, mlp well    The regime the pivot needs: present, not linearly
  above the control      accessible, and therefore something a local rule could
                         plausibly make accessible.

  both at the control    The state carries no information about its own future.
                         **The pivot is dead as specified** -- but see below,
                         because the most likely cause is the benchmark rather
                         than the model.

THE CONFOUND THAT MATTERS MOST
--------------------------------
`DelayedParity` fills most timesteps with independent Poisson noise. That is
**unpredictable by construction**, so a null here may be a fact about the task
and not about the substrate. The cue bursts are the only predictable structure
in the stream, and they occupy a small fraction of it.

So the probe reports `during_cue` separately -- predictability restricted to
timesteps inside a cue burst, where there is something to predict. If the
overall number is at the control floor and `during_cue` is well above it, the
finding is "this task is mostly noise", not "this substrate cannot predict", and
the response is a task with temporal structure rather than abandoning the
direction.

STATUS: WRITTEN, NOT YET VERIFIED TO RUN
------------------------------------------
This has not completed a run. Two smoke attempts at reduced settings both hit a
115-second timeout, and subsampling to 6000 rows did not fix it, so the cost is
in the probe rather than in machine contention -- most likely the per-horizon
decoder count (six fits: linear and MLP, each for the raw target, the shuffled
control and the cue-only mask). **No number from this file should be quoted
until it has run end to end.** The next step is to time the pieces separately
rather than reduce settings blindly.

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
    """Column states and the input stream beside them, aligned in time.

    The column is observed and not disturbed: `learning` off throughout, so
    nothing adapts to the fact that it is being watched.
    """
    col = model.column
    col.learning = False
    states, inputs, in_cue = [], [], []
    for _ in range(episodes):
        ep = task.episode(rng)
        col.reset_state()
        model.transport.reset()
        # A timestep counts as "in a cue" when any cue-group channel carries
        # something. The go cue is included: it is predictable structure too.
        cue_span = task.group_size * (2 * task.n_cues + 1)
        for k in range(ep.inputs.shape[0]):
            t = model._t
            model._t += 1
            out = col.step(t, ep.inputs[k])
            states.append(out.copy())
            inputs.append(ep.inputs[k].copy())
            in_cue.append(bool((ep.inputs[k][:cue_span] > 0).any()))
    return (np.array(states, dtype=np.float32),
            np.array(inputs, dtype=np.float32),
            np.array(in_cue, dtype=bool))


def score(states, targets, mask=None, cap: int = 6000, seed: int = 0
          ) -> tuple[float, float]:
    """Linear and MLP accuracy predicting a binary target from column state.

    Subsampled to `cap` rows. A run of 60 episodes produces ~27,000 timesteps,
    and this probe fits six decoders per horizon across four horizons and three
    seeds -- 72 fits, each an MLP over 96 features. Uncapped that does not
    finish. Six thousand rows is far more than is needed to separate a real
    effect from the shuffled floor, and the cap is applied *before* the
    train/test split so both sides shrink together.

    Sampled without replacement rather than truncated: the trajectory is
    episode-ordered, so taking the first N rows would sample a handful of whole
    episodes instead of the run.
    """
    if mask is not None:
        states, targets = states[mask], targets[mask]
    if len(states) < 200 or targets.min() == targets.max():
        return float("nan"), float("nan")
    if len(states) > cap:
        idx = np.random.default_rng(seed).choice(len(states), cap, replace=False)
        idx.sort()          # keep time order, so the split stays a time split
        states, targets = states[idx], targets[idx]
    split = int(0.7 * len(states))
    Xtr, Xte = states[:split], states[split:]
    ytr, yte = targets[:split], targets[split:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    lin = logistic_score(logistic(Xtr, ytr), Xte, yte)
    return float(lin), float(mlp(Xtr, ytr, Xte, yte))


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
        print(f"{'horizon':>8s}{'linear':>9s}{'mlp':>9s}{'gap':>8s}"
              f"{'shuffled':>10s}{'lin-cue':>9s}{'mlp-cue':>9s}{'seeds':>7s}")
        for k in sorted({r["horizon"] for r in rows}):
            sel = [r for r in rows if r["horizon"] == k]
            g = lambda f: np.nanmean([r[f] for r in sel])  # noqa: E731
            print(f"{k:8d}{g('linear'):9.3f}{g('mlp'):9.3f}"
                  f"{g('mlp') - g('linear'):8.3f}{g('shuffled_mlp'):10.3f}"
                  f"{g('linear_cue'):9.3f}{g('mlp_cue'):9.3f}{len(sel):7d}")
        print("\n`shuffled` is the floor: the same decoder on a time-shuffled "
              "stream.\nA score at that level means no predictive information, "
              "however high it looks.")
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
        states, inputs, in_cue = trajectory(
            model, task, episodes, np.random.default_rng(21 + s))

        # Predict whether ANY channel carries an event at t+k. A per-channel
        # target would be dominated by the ~98% of channels that are silent at
        # any moment, so the score would report the marginal rate rather than
        # prediction. This asks the sharper question: is something coming?
        event = (inputs > 0).any(axis=1).astype(int)
        shuffled = event.copy()
        np.random.default_rng(777 + s).shuffle(shuffled)

        for k in args.horizons:
            lin, non = score(states[:-k], event[k:])
            _, sh = score(states[:-k], shuffled[k:])
            lin_c, non_c = score(states[:-k], event[k:], mask=in_cue[:-k])
            row = dict(horizon=k, seed=s, neurons=args.neurons,
                       episodes=episodes, linear=lin, mlp=non,
                       shuffled_mlp=sh, linear_cue=lin_c, mlp_cue=non_c)
            with OUT.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"k={k:3d} seed={s}  linear {lin:.3f}  mlp {non:.3f}  "
                  f"gap {non - lin:+.3f}  shuffled {sh:.3f}  "
                  f"cue-only {lin_c:.3f}/{non_c:.3f}")


if __name__ == "__main__":
    main()
