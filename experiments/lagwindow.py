"""How wide is binding's latency window, and which time constant sets it?

IN PLAIN TERMS
--------------
Binding is the one learning mechanism in this project that works. It fires when
a "this was worth remembering" signal arrives, and it strengthens whichever
synapses were busy just beforehand. Spread the model over a network and that
signal arrives late -- so the question the whole architecture rests on is how
late it can arrive before there is nothing left to strengthen.

Sweep 026 measured that end-to-end and found the window is much narrower than
the code's comments claim: the gain is already undetectable at a lag of 25
steps, and widening the trace the comments blame did not recover any of it.

This probe measures the window directly instead of inferring it from
decodability, and it separates the two traces the rule multiplies together, so
the answer names a parameter rather than a number.

WHAT IT MEASURES
----------------
`_bind` commits

    dW  ~  post x pre

with **two** traces on the right-hand side, decaying at different rates:

    post = act_fast / baseline     tau_act_fast = 50    the neuron's own activity
    pre  = the branch filter       tau_branch   = 15    the synapse's own input

Both have decayed by the time a late modulator lands, so the size of the update
falls as the *product* of the two decays:

    |dW|(lag) / |dW|(0)  ~  exp(-lag/50) * exp(-lag/15)  =  exp(-lag/tau_eff)

    1/tau_eff = 1/tau_act_fast + 1/tau_branch    ->    tau_eff = 11.5

That is the claim under test. It matters because the three candidate
explanations make numerically different predictions and only one of them is
actionable:

    act_fast alone   tau 50     retains 61% at lag 25   widen tau_act_fast
    pre alone        tau 15     retains 19% at lag 25   widen tau_branch
    the product      tau 11.5   retains 11% at lag 25   widen tau_branch, and
                                                        tau_act_fast cannot help

The third is why sweep 026's `on-lag150-slow` recovered nothing. A product is
dominated by its *shorter* constant, so raising `tau_act_fast` from 50 to 250
moves tau_eff only from 11.5 to 14.2 -- a 23% widening, bought at the cost of
scoring an activity window five times too wide.

    python3 experiments/lagwindow.py
    python3 experiments/lagwindow.py --tau-branch 60 --tau-act-fast 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plexus import ColumnConfig, DelayedXOR, Plexus  # noqa: E402


def _run(model, ep, lag: int) -> None:
    """Drive the column through one episode, then hold it silent for `lag`."""
    col = model.column
    col.reset_state()
    model.transport.reset()
    zero = np.zeros(ep.inputs.shape[1], dtype=np.float32)
    for k in range(ep.inputs.shape[0]):
        t = model._t
        model._t += 1
        col.step(t, ep.inputs[k])
    for _ in range(lag):
        t = model._t
        model._t += 1
        col.step(t, zero)


def binding_delta(model, task, ep, lag: int, warmup: int) -> float:
    """Total weight change binding commits when the modulator arrives `lag` late.

    The column is driven through one real episode, then held in silence for
    `lag` steps -- exactly what `network.run_episode`'s drain tail does -- and
    the binding rule is fired by hand at the end. `W` is snapshotted immediately
    before the call, so homeostatic scaling during the silent tail is excluded
    and the number is the binding update alone.

    **`warmup` is not a detail; without it this probe measures nothing.**
    `_bind` derives its baseline from `act_slow`, debiased by the sample count.
    On the *first* event `act_slow` has had exactly one update, so the debias
    returns `act_fast` itself and `post = act_fast / baseline` is identically
    1.0 whatever the activity was. Measured that way the probe reports
    `tau_eff = 15.4` -- precisely `tau_branch`, with the `act_fast` term
    contributing no decay at all, because it had been divided out by its own
    numerator. That is a criterion normalising away its own input (standards
    rule 7) sitting in the instrument rather than in the mechanism, and it would
    have been reported as a finding about binding.

    So the column is first given `warmup` binding events at lag 0, which is what
    a real run does hundreds of times, leaving `act_slow` an actual running
    average against which a decayed `act_fast` reads as small.
    """
    for _ in range(warmup):
        _run(model, task.episode(np.random.default_rng(7 + model.column.n_samples)), 0)
        model.column._bind()

    _run(model, ep, lag)
    before = model.column.W.copy()
    model.column._bind()
    return float(np.abs(model.column.W - before).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lags", type=int, nargs="+",
                    default=[0, 5, 10, 15, 25, 50, 100, 150])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--neurons", type=int, default=96)
    ap.add_argument("--tau-act-fast", type=float, default=50.0)
    ap.add_argument("--tau-branch", type=float, default=15.0)
    ap.add_argument("--bind-tau-pre", type=float, default=None,
                    help="give the binding rule its own presynaptic trace at "
                         "this constant, decoupling the window from the forward "
                         "path. The window is still a product, so tau_act_fast "
                         "must be raised alongside it.")
    ap.add_argument("--warmup", type=int, default=12,
                    help="binding events at lag 0 before measuring, so the "
                         "baseline is a running average rather than a copy of "
                         "the sample being scored. See binding_delta.")
    args = ap.parse_args()

    task = DelayedXOR()
    rows = np.zeros((args.seeds, len(args.lags)))
    for s in range(args.seeds):
        ep = task.episode(np.random.default_rng(100 + s))
        for j, lag in enumerate(args.lags):
            # A fresh column per lag: `_bind` advances `n_samples`, which sets
            # the binding rate and the baseline debias, so reusing one column
            # would make later lags cheaper for a reason unrelated to the lag.
            model = Plexus(
                task.n_inputs, task.n_classes,
                column=ColumnConfig(
                    n_neurons=args.neurons, lr=0.0, seed=s, hebbian=True,
                    tau_act_fast=args.tau_act_fast, tau_branch=args.tau_branch,
                    bind_tau_pre=args.bind_tau_pre,
                ),
                seed=s,
            )
            rows[s, j] = binding_delta(model, task, ep, lag, args.warmup)

    ref = rows[:, 0:1]
    retained = rows / np.maximum(ref, 1e-30)

    tau_pre = args.tau_branch if args.bind_tau_pre is None else args.bind_tau_pre
    tau_eff = 1.0 / (1.0 / args.tau_act_fast + 1.0 / tau_pre)
    print(f"tau_act_fast {args.tau_act_fast:g}   tau_pre {tau_pre:g}"
          f"   -> predicted tau_eff {tau_eff:.2f}")
    print(f"{'lag':>5s}{'|dW|':>12s}{'retained':>10s}"
          f"{'act_fast':>10s}{'pre':>8s}{'product':>9s}")
    for j, lag in enumerate(args.lags):
        r = retained[:, j].mean()
        print(f"{lag:5d}{rows[:, j].mean():12.5f}{r:10.4f}"
              f"{np.exp(-lag / args.tau_act_fast):10.4f}"
              f"{np.exp(-lag / tau_pre):8.4f}"
              f"{np.exp(-lag / tau_eff):9.4f}")

    # Fit the measured decay so the answer is a time constant rather than a
    # visual match against three columns.
    #
    # The tail fit is the one to read. `post` is clipped at 5.0, so for the
    # first ~15 steps the ratio sits on its ceiling and contributes no decay at
    # all; a fit spanning that flat region reports a window wider than the
    # mechanism has. Fitting from `tau_eff` onward -- by which point the clip
    # has released -- is what recovers the exponential the product predicts.
    lags = np.array(args.lags, dtype=np.float64)
    mean = retained.mean(0)
    keep = mean > 1e-6
    full = np.polyfit(lags[keep], np.log(mean[keep]), 1)[0]
    tail = keep & (lags >= tau_eff)
    print(f"\nmeasured tau_eff = {-1.0 / full:.2f} over all lags", end="")
    if tail.sum() >= 2:
        t = np.polyfit(lags[tail], np.log(mean[tail]), 1)[0]
        print(f", {-1.0 / t:.2f} over the tail (lag >= {tau_eff:.1f})")
    else:
        print("  [no tail points -- pass larger --lags for the usable fit]")
    print(f"predicted: act_fast alone {args.tau_act_fast:g}, "
          f"pre alone {tau_pre:g}, product {tau_eff:.2f}")


if __name__ == "__main__":
    main()
