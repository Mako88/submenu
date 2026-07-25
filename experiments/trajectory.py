"""When during training does binding's representation gain actually arrive?

Sweep 018 left one question standing. Binding needs roughly its full 300
episodes -- halving the budget to fund a static period for the readout gives
back nearly all the gain -- and that is odd, because the baseline binding
scores against converges early, so late events should contribute progressively
less. Something is accumulating late that no diagnostic currently measures.

This measures it directly: train as normal, and every `--every` episodes pause
to fit a fresh linear decoder on held-out probe episodes. The result is a curve
rather than an endpoint, which is the only thing that can distinguish "binding
is slow" from "the readout is slow".

The probe must not disturb what it measures. Probe episodes run with
`learn=False`, which stops binding, weight updates, homeostasis and the
readout's own statistics; they draw from a separate RNG so the training stream
is untouched; and `test_probing_does_not_perturb_training` asserts that
training with probes interleaved leaves W, theta and the knee bit-identical to
training without. A measurement that changes its subject is worse than none,
and this project has already shipped one of those.

    python3 experiments/trajectory.py --tag on  --hebbian 1 --seed 0
    python3 experiments/trajectory.py --report
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

OUT = Path(__file__).resolve().parent / "trajectory_results.jsonl"


def decodability(model, task, episodes: int, seed: int) -> float:
    """Linear decodability of the column state, without touching training."""
    X, y = collect(model, task, episodes, np.random.default_rng(seed))
    split = int(0.7 * len(X))
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    return logistic_score(logistic((Xtr - mu) / sd, ytr), (Xte - mu) / sd, yte)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="on")
    ap.add_argument("--hebbian", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--probe", type=int, default=300)
    ap.add_argument("--neurons", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.report:
        rows = [json.loads(x) for x in OUT.read_text().splitlines() if x.strip()]
        tags = sorted({r["tag"] for r in rows})
        checkpoints = sorted({c for r in rows for c in map(int, r["curve"])})
        print(f"{'episode':>8s}" + "".join(f"{t:>12s}" for t in tags) + "        gap")
        for c in checkpoints:
            cells, vals = "", {}
            for t in tags:
                v = [r["curve"][str(c)] for r in rows if r["tag"] == t and str(c) in r["curve"]]
                vals[t] = float(np.mean(v)) if v else float("nan")
                cells += f"{vals[t]:12.3f}"
            gap = vals.get("on", float("nan")) - vals.get("off", float("nan"))
            print(f"{c:8d}{cells}{gap:+11.3f}")
        n = len({r["seed"] for r in rows})
        print(f"\n{n} seeds")
        return

    task = DelayedXOR()
    model = Plexus(
        task.n_inputs, task.n_classes,
        column=ColumnConfig(
            n_neurons=args.neurons, lr=0.0, seed=args.seed, hebbian=bool(args.hebbian)
        ),
        seed=args.seed,
    )
    rng = np.random.default_rng(1000 + args.seed)

    curve = {}
    done = 0
    # Checkpoint at 0 as well: it is the frozen column, and both conditions must
    # agree there or the probe is measuring something other than binding.
    curve[str(done)] = decodability(model, task, args.probe, 11 + args.seed)
    while done < args.episodes:
        step = min(args.every, args.episodes - done)
        model.train(task, step, rng=rng, report_every=10**9)
        done += step
        curve[str(done)] = decodability(model, task, args.probe, 11 + args.seed)

    row = dict(tag=args.tag, seed=args.seed, hebbian=args.hebbian,
               episodes=args.episodes, every=args.every, curve=curve)
    with OUT.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    pts = " ".join(f"{k}:{v:.3f}" for k, v in sorted(curve.items(), key=lambda kv: int(kv[0])))
    print(f"{args.tag} seed={args.seed}  {pts}")


if __name__ == "__main__":
    main()
