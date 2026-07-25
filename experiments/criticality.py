"""Is the column self-organising toward criticality, and does that explain it?

The project's own numbers point here. The designed three-factor learning rule is
worth -0.003 (p = 0.79); homeostasis, which is not a learning rule at all but
each neuron holding its own activity near a setpoint, is worth +0.197. Sweep 023
then found that benefit is obtained *better* from structureless input than from
the task, so it is not acquiring the task -- it is finding a regime in which the
column becomes readable.

"A regime in which a network becomes readable" has a name and a measurement.
A branching process propagates one event into `m` further events on average.
Below 1 activity dies; above 1 it explodes; at `m = 1` the network holds and
combines information over the longest timescales it can. If homeostasis is
driving the column toward m = 1, then criticality is the mechanism behind the
+0.197 and the target to optimise is "stay critical" rather than "learn well".

WHAT MAKES THIS HARD TO MEASURE HONESTLY
-----------------------------------------
Avalanche-size power laws are the textbook signature and they are the wrong tool
here. An avalanche is a run of activity bounded by silence, and this column at
96 neurons and 3% sparsity emits ~2.9 events per millisecond, so silence is rare
and the whole recording is one avalanche. Reporting an exponent from that would
be a number computed from a definition that does not apply.

So the primary estimate is the multistep regression of Wilting & Priesemann
(2018), which reads `m` off how the activity autocorrelation decays and needs no
silent bins:

    r_k = Cov(n_t, n_t+k) / Var(n_t)  ~  A * m^k

and `m = exp(slope)` of `log r_k` against `k`.

**That estimator assumes the autocorrelation comes from propagation, and in this
model it does not only come from propagation.** Membrane time constants run to
320ms, so activity is autocorrelated whether or not events cause events. The
`--control` condition is what separates them: it zeroes the recurrent synapses,
leaving membrane dynamics and external drive untouched. Whatever `m` it reports
is the part that is *not* branching, and only the difference is evidence.

    python3 experiments/criticality.py --tag all --seed 0
    python3 experiments/criticality.py --report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plexus import ColumnConfig, DelayedXOR, Plexus, freeze_plasticity  # noqa: E402
from probe import collect, logistic, logistic_score  # noqa: E402

OUT = Path(__file__).resolve().parent / "criticality_results.jsonl"


def branching_ratio(counts: np.ndarray, kmax: int = 40) -> float:
    """Estimate the branching parameter from how activity autocorrelation decays.

    Wilting & Priesemann 2018. For a branching process the lag-k correlation of
    the activity falls as `A * m^k`, so `m` is the exponential of the slope of
    `log r_k` against `k`. Unlike avalanche segmentation this needs no silent
    bins, which is what makes it usable on a network that never goes quiet.

    Returns NaN rather than a number when the correlations are too small or too
    noisy to fit — the failure mode to avoid is returning a plausible `m` from a
    signal that carries none, which is how this project has been burned before.

    The fit stops at `r_k < FLOOR` rather than at the first non-positive value,
    and the difference is not cosmetic. Cutting on positivity keeps only the
    lags where sampling noise happened to land *above* zero, which is a
    selection effect: it flattens the tail, shallows the slope, and biases `m`
    upward. Measured against a process with a known m of 0.70, the
    positivity-cut version returned **0.7856** — an 8.5% overestimate, in the
    direction that makes any network look closer to critical than it is. Caught
    by `test_branching_estimator_recovers_a_known_branching_ratio`, which is why
    that test generates a process whose answer is known rather than asserting
    something about the column.
    """
    FLOOR = 0.02
    n = np.asarray(counts, dtype=np.float64)
    n = n - n.mean()
    var = float((n * n).mean())
    if var <= 0.0:
        return float("nan")
    ks, rs = [], []
    for k in range(1, kmax + 1):
        r = float((n[:-k] * n[k:]).mean()) / var
        if r < FLOOR:
            break
        ks.append(k)
        rs.append(r)
    if len(ks) < 5:
        return float("nan")
    slope = np.polyfit(np.array(ks, dtype=np.float64), np.log(rs), 1)[0]
    return float(np.exp(slope))


def decodability(model, task, episodes: int, seed: int) -> float:
    X, y = collect(model, task, episodes, np.random.default_rng(seed))
    split = int(0.7 * len(X))
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    return logistic_score(logistic((Xtr - mu) / sd, ytr), (Xte - mu) / sd, yte)


def activity(model, task, episodes: int, rng) -> np.ndarray:
    """Per-timestep event counts, with the column observed and not disturbed."""
    counts = []
    col = model.column
    col.learning = False
    for _ in range(episodes):
        ep = task.episode(rng)
        col.reset_state()
        model.transport.reset()
        for k in range(ep.inputs.shape[0]):
            t = model._t
            model._t += 1
            out = col.step(t, ep.inputs[k])
            counts.append(int((out > 0).sum()))
    return np.array(counts, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="all")
    ap.add_argument("--hebbian", type=int, default=0)
    ap.add_argument("--lateral", type=int, default=0)
    ap.add_argument("--homeostatic-lr", type=float, default=None)
    ap.add_argument("--knee-lr", type=float, default=None)
    ap.add_argument("--episodes", type=int, default=100,
                    help="settling episodes before measuring")
    ap.add_argument("--record", type=int, default=40,
                    help="episodes of activity recorded for the estimate")
    ap.add_argument("--probe", type=int, default=300)
    ap.add_argument("--neurons", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resettle", type=int, default=40,
                    help="episodes for the control to re-settle to matched "
                         "activity after its recurrence is silenced")
    ap.add_argument("--control", action="store_true",
                    help="zero the recurrent synapses, leaving membrane dynamics "
                         "and external drive. Whatever m this reports is the part "
                         "that is not branching.")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.report:
        rows = [json.loads(x) for x in OUT.read_text().splitlines() if x.strip()]
        cols = ["m", "m_control", "m_excess", "linear",
                "sparsity", "sparsity_control"]
        print(f"{'condition':16s}" + "".join(f"{c:>12s}" for c in cols) + "    n")
        for tag in sorted({r["tag"] for r in rows}):
            sel = [r for r in rows if r["tag"] == tag]
            cells = "".join(
                f"{np.nanmean([r.get(c, np.nan) for r in sel]):12.4f}" for c in cols
            )
            print(f"{tag:16s}{cells}{len(sel):5d}")
        return

    task = DelayedXOR()
    overrides = {
        k: v for k, v in (("homeostatic_lr", args.homeostatic_lr),
                          ("knee_lr", args.knee_lr)) if v is not None
    }
    model = Plexus(
        task.n_inputs, task.n_classes,
        column=ColumnConfig(
            n_neurons=args.neurons, lr=0.0, seed=args.seed,
            hebbian=bool(args.hebbian), lateral=bool(args.lateral), **overrides
        ),
        seed=args.seed,
    )
    rng = np.random.default_rng(1000 + args.seed)
    model.train(task, args.episodes, rng=rng, report_every=10**9)

    # Measure before the control mutilates the column, or the two numbers would
    # not describe the same object.
    linear = decodability(model, task, args.probe, 11 + args.seed)
    counts = activity(model, task, args.record, np.random.default_rng(21 + args.seed))
    m = branching_ratio(counts)
    sparsity = float((counts > 0).mean() * counts.mean() / args.neurons)

    # The control: same column, same membrane constants, same external drive,
    # recurrent synapses silenced. Any autocorrelation left is not propagation.
    #
    # Silencing recurrence also removes drive, so a naive control runs at a lower
    # firing rate and its `m` would differ for that reason as much as for the
    # missing propagation. So homeostasis is allowed to re-settle afterwards,
    # which returns activity to target while the recurrence stays dead --
    # synaptic scaling is multiplicative, so a zeroed weight is a permanently
    # zeroed weight. `sparsity_control` is recorded so the match can be checked
    # rather than assumed.
    freeze_plasticity(model.column.cfg)
    recurrent = model.column.src >= model.column.cfg.n_external
    model.column.W[recurrent] = 0.0
    model.column.learning = True
    model.train(task, args.resettle, rng=rng, report_every=10**9)
    ctrl = activity(model, task, args.record, np.random.default_rng(21 + args.seed))
    m_control = branching_ratio(ctrl)
    sparsity_control = float((ctrl > 0).mean() * ctrl.mean() / args.neurons)

    row = dict(tag=args.tag, seed=args.seed, hebbian=args.hebbian,
               lateral=args.lateral, episodes=args.episodes, neurons=args.neurons,
               m=m, m_control=m_control, m_excess=m - m_control,
               linear=linear, sparsity=sparsity,
               sparsity_control=sparsity_control, **overrides)
    with OUT.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"{args.tag} seed={args.seed}  m={m:.4f}  control={m_control:.4f}  "
          f"excess={m - m_control:+.4f}  linear={linear:.3f}  "
          f"sparsity {sparsity:.4f} vs control {sparsity_control:.4f}")


if __name__ == "__main__":
    main()
