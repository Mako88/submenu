"""Does learning a second task cost what the column learned on the first?

Every number in this repo is a number about one task with one answer and no
interference between memories. Delayed XOR cannot ask about capacity, cannot ask
about forgetting, and is the reason sweep 014's refutation of engram allocation
is narrower than it reads: the allocator's whole purpose is to push successive
memories onto *different* neurons, and a benchmark with one memory never asks
for that.

This is the smallest benchmark that does. Two tasks that are the same
computation over a different sensory mapping -- delayed XOR with the input
channels permuted -- so they are identical in difficulty, identical in
statistics, and disjoint in what they ask the column's fixed wiring to do. If
task B were simply harder, forgetting and difficulty would not be separable in
the result.

    train on A  ->  measure A            (acquisition)
    train on B  ->  measure A and B      (retention, and did B learn at all)

The quantity is retention: A's decodability after B, against A's decodability
before B.

The first version of this docstring asserted that `off` cannot forget, "because
nothing about a frozen column changes". The mechanism half of that is wrong: a
column with `lr=0` and no binding still runs threshold homeostasis, knee
adaptation and synaptic scaling, and although A and B have *identical* marginal
statistics by construction, each neuron sees a different subset of channels
through its fixed wiring, so its drive distribution changes and its threshold
follows. Pinned by `test_a_frozen_column_still_changes_when_the_input_changes`.

The correction to the correction, which matters more. That first run showed
`off` losing 0.167, and 0.167 went into this docstring, TODO.md, AUDIT.md and a
commit message before it had been measured. **At 20 seeds it is 0.015.** The
mechanism is real; the magnitude was one seed and was wrong by an order of
magnitude.

So the honest statement is that every condition forgets a little (-0.015 to
-0.030) and none forgets differently from any other -- all four retention
comparisons null, sweep 024. What this benchmark resolves is acquisition, not
retention, and at this scale there is no interference here to study.

    python3 experiments/continual.py --tag on --hebbian 1 --seed 0
    python3 experiments/continual.py --report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plexus import ColumnConfig, DelayedXOR, Plexus  # noqa: E402
from probe import collect, logistic, logistic_score  # noqa: E402

OUT = Path(__file__).resolve().parent / "continual_results.jsonl"


def decodability(model, task, episodes: int, seed: int) -> float:
    """Linear decodability of the column state. Does not perturb training."""
    X, y = collect(model, task, episodes, np.random.default_rng(seed))
    split = int(0.7 * len(X))
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    return logistic_score(logistic((Xtr - mu) / sd, ytr), (Xte - mu) / sd, yte)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="on")
    ap.add_argument("--hebbian", type=int, default=1)
    ap.add_argument("--lateral", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=150,
                    help="training episodes per task")
    ap.add_argument("--probe", type=int, default=400)
    ap.add_argument("--neurons", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.report:
        rows = [json.loads(x) for x in OUT.read_text().splitlines() if x.strip()]
        cols = ["a_before", "a_after", "b_after", "retention", "acquisition"]
        print(f"{'condition':14s}" + "".join(f"{c:>12s}" for c in cols) + "   n")
        for tag in sorted({r["tag"] for r in rows}):
            sel = [r for r in rows if r["tag"] == tag]
            cells = "".join(f"{np.mean([r[c] for r in sel]):12.3f}" for c in cols)
            print(f"{tag:14s}{cells}{len(sel):5d}")
        return

    # Same underlying task, different sensory mapping. The channel seed is
    # deliberately independent of the model seed: every model seed must face the
    # same pair of tasks, or a hard permutation for one seed and an easy one for
    # another becomes seed variance nobody can see.
    task_a = DelayedXOR(channel_seed=1)
    task_b = DelayedXOR(channel_seed=2)

    model = Plexus(
        task_a.n_inputs, task_a.n_classes,
        column=ColumnConfig(
            n_neurons=args.neurons, lr=0.0, seed=args.seed,
            hebbian=bool(args.hebbian), lateral=bool(args.lateral),
        ),
        seed=args.seed,
    )
    rng = np.random.default_rng(1000 + args.seed)

    model.train(task_a, args.episodes, rng=rng, report_every=10**9)
    a_before = decodability(model, task_a, args.probe, 11 + args.seed)

    model.train(task_b, args.episodes, rng=rng, report_every=10**9)
    # Same probe seed as `a_before`, so the two numbers are the same episodes
    # scored against a column that changed in between. A fresh draw here would
    # put probe-sample noise into the retention figure, which is a difference of
    # two numbers and so carries both draws' noise.
    a_after = decodability(model, task_a, args.probe, 11 + args.seed)
    b_after = decodability(model, task_b, args.probe, 12 + args.seed)

    row = dict(
        tag=args.tag, seed=args.seed, hebbian=args.hebbian, lateral=args.lateral,
        episodes=args.episodes, neurons=args.neurons,
        a_before=a_before, a_after=a_after, b_after=b_after,
        # Reported as differences, not ratios. A ratio of two decodabilities
        # both near 0.5 is unstable and would read as catastrophic forgetting
        # from noise on a column that never learned A in the first place.
        retention=a_after - a_before,
        acquisition=b_after - 0.5,
    )
    with OUT.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"{args.tag} seed={args.seed}  A {a_before:.3f} -> {a_after:.3f}   "
          f"B {b_after:.3f}   retention {row['retention']:+.3f}")


if __name__ == "__main__":
    main()
