"""Which benchmark leaves a learning rule something to do?

IN PLAIN TERMS
--------------
Every learning mechanism in this project has been tested on one puzzle, and an
untrained network already scores 0.802 out of 1.0 on it before it learns
anything at all. So even a flawless learning rule could only ever show a small
improvement, and every mechanism tried has been competing for the last fifth of
the problem. Eleven of them came back as "no measurable effect", and this may
simply be why.

This finds a better puzzle. Not a harder one -- a harder one can be *too* hard,
and we have that too: three cues instead of two puts every decoder at chance,
which leaves just as little to measure. What is wanted is the puzzle where the
information is clearly present in the network's state but a simple readout
cannot get at it. That gap is exactly the thing a learning rule is supposed to
close, so the size of the gap is the size of the job available.

WHAT IT MEASURES
----------------
For a **frozen, untrained** column in each configuration:

    linear   what a logistic readout gets from the column state
    mlp      what a small nonlinear decoder gets from the same states
    gap      mlp - linear

`gap` is the quantity of interest. It is a direct measurement of "the label is
present and not linearly accessible", which is the only thing a representation
learning rule can fix.

  gap near 0, linear high    the substrate already solved it. A perfect rule
                             wins almost nothing. **This is delayed XOR today:
                             linear 0.802, MLP 0.987.**
  gap near 0, both at chance too hard -- the information is not there at all, so
                             there is nothing to make accessible. **This is
                             3-cue parity at 96 neurons.**
  gap large                  present but tangled. **The regime a learning rule
                             has a job in**, and what this probe is looking for.

Sweep 002 already found one such point without pursuing it: 24 neurons on
delayed XOR gives linear 0.689 against MLP 0.881, a gap of +0.19.

WHY A FROZEN COLUMN
-------------------
The benchmark has to be chosen on a property of the *substrate*, not of any
mechanism, or the choice bakes in whichever rule happened to be on when it was
picked. Everything here runs `lr = 0` with binding and lateral inhibition off.

WHAT THIS DOES NOT DO
---------------------
It does not show that a large gap is *closable*. A gap means a linear readout
cannot reach the information; it does not promise that any local rule can make
it reachable. That is the next question and this probe cannot answer it -- what
it can do is stop us measuring mechanisms in a regime where success is
invisible.

Nor does it choose a task on its own. The output is a table; picking the
standard benchmark from it is a judgement call, and the criterion should be
recorded next to the choice.

    python3 experiments/headroom.py --quick
    python3 experiments/headroom.py --report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plexus import ColumnConfig, DelayedParity, Plexus  # noqa: E402
from probe import collect, logistic, logistic_score, mlp  # noqa: E402

OUT = Path(__file__).resolve().parent / "headroom_results.jsonl"


def measure(task, neurons: int, episodes: int, collect_n: int, seed: int) -> dict:
    """Linear and MLP decodability of a frozen column on this configuration."""
    model = Plexus(
        task.n_inputs, task.n_classes,
        column=ColumnConfig(n_neurons=neurons, lr=0.0, seed=seed),
        seed=seed,
    )
    # Settle only. No plasticity: the benchmark is chosen on a property of the
    # substrate, not of whichever mechanism happened to be enabled.
    model.train(task, episodes, rng=np.random.default_rng(1000 + seed),
                report_every=10**9)

    X, y = collect(model, task, collect_n, np.random.default_rng(11 + seed))
    split = int(0.7 * len(X))
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    lin = logistic_score(logistic(Xtr, ytr, n_classes=task.n_classes), Xte, yte)
    non = mlp(Xtr, ytr, Xte, yte)
    return dict(linear=float(lin), mlp=float(non), gap=float(non - lin),
                sparsity=float(model.column.sparsity))


# (label, task kwargs, neurons). Chosen to bracket the two known failure modes
# -- saturated at one end, at chance at the other -- rather than to search
# widely. A wide search is cheap to add once the shape is known.
GRID = [
    ("xor-96",            dict(n_cues=2), 96),   # the standing benchmark, 0.802
    ("xor-48",            dict(n_cues=2), 48),
    ("xor-24",            dict(n_cues=2), 24),   # sweep 002: gap +0.19
    ("xor-noisy-96",      dict(n_cues=2, noise_rate=0.08), 96),
    ("xor-weak-96",       dict(n_cues=2, cue_strength=0.4), 96),
    ("xor-distract-96",   dict(n_cues=2, n_distractor=32), 96),
    ("xor-far-96",        dict(n_cues=2, cue_spacing=200, response_gap=200), 96),
    ("parity3-96",        dict(n_cues=3), 96),   # known: both at chance
    ("parity3-192",       dict(n_cues=3), 192),
    ("parity3-close-192", dict(n_cues=3, cue_spacing=60, response_gap=60), 192),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--collect", type=int, default=400)
    ap.add_argument("--quick", action="store_true",
                    help="1 seed, fewer episodes -- shape only, not a result")
    ap.add_argument("--only", default=None, help="run one grid label")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.report:
        rows = [json.loads(x) for x in OUT.read_text().splitlines() if x.strip()]
        print(f"{'configuration':20s}{'linear':>9s}{'mlp':>9s}{'gap':>9s}"
              f"{'sparsity':>10s}{'seeds':>7s}")
        for label in sorted({r["label"] for r in rows}):
            sel = [r for r in rows if r["label"] == label]
            print(f"{label:20s}"
                  f"{np.mean([r['linear'] for r in sel]):9.3f}"
                  f"{np.mean([r['mlp'] for r in sel]):9.3f}"
                  f"{np.mean([r['gap'] for r in sel]):9.3f}"
                  f"{np.mean([r['sparsity'] for r in sel]):10.4f}"
                  f"{len(sel):7d}")
        print("\ngap = mlp - linear. Large gap and mlp well above chance is the "
              "regime\na learning rule has a job in. Chance is 0.500 for these "
              "tasks.")
        return

    seeds = 1 if args.quick else args.seeds
    episodes = 40 if args.quick else args.episodes
    collect_n = 200 if args.quick else args.collect

    for label, kw, neurons in GRID:
        if args.only and label != args.only:
            continue
        for s in range(seeds):
            task = DelayedParity(**kw)
            r = measure(task, neurons, episodes, collect_n, s)
            row = dict(label=label, neurons=neurons, seed=s,
                       episodes=episodes, collect=collect_n, **kw, **r)
            with OUT.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"{label:20s} seed={s}  linear {r['linear']:.3f}  "
                  f"mlp {r['mlp']:.3f}  gap {r['gap']:+.3f}  "
                  f"sparsity {r['sparsity']:.4f}")


if __name__ == "__main__":
    main()
