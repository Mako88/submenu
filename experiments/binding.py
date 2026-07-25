"""Does salience-gated Hebbian binding improve the representation?

Measures linear decodability of the column state, which is the cleanest read on
what a rule changes: it does not depend on the online readout's sample
efficiency at all, and sweep 009 established that the readout can hide a
column-level difference entirely.

This replaces experiments/engram.py, which measured an engram allocator --
excitability drift, a recruitment competition, an allocation refractory. That
mechanism was built, measured and deleted: sweep 014 put the full apparatus at
0.830 against 0.876 for binding alone, worse on 19 of 20 seeds at p = 0.0001.
The allocation diagnostics went with it, since there is no longer an allocation
to diagnose. They are recoverable from git history and from
experiments/sweeps/engram-012 and -014 if a benchmark ever asks for memory
separation, which delayed XOR does not.

    python3 experiments/binding.py --tag on  --hebbian 1 --seed 0
    python3 experiments/binding.py --tag off --hebbian 0 --seed 0
    python3 experiments/binding.py --report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plexus import ColumnConfig, DelayedXOR, Plexus  # noqa: E402
from probe import collect, logistic, logistic_score, mlp  # noqa: E402

OUT = Path(__file__).resolve().parent / "binding_results.jsonl"
FIELDS = ["linear", "mlp", "sparsity", "bound"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="on")
    ap.add_argument("--hebbian", type=int, default=1)
    ap.add_argument("--hebb-lr", type=float, default=ColumnConfig.hebb_lr)
    ap.add_argument("--bind-scale", type=float, default=ColumnConfig.bind_scale)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--collect", type=int, default=500)
    ap.add_argument("--neurons", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.report:
        rows = [json.loads(x) for x in OUT.read_text().splitlines() if x.strip()]
        head = "".join(f"{f:>10s}" for f in FIELDS)
        print(f"{'condition':16s}{head}   seeds")
        for tag in sorted({r["tag"] for r in rows}):
            sel = [r for r in rows if r["tag"] == tag]
            cells = "".join(f"{np.mean([r[f] for r in sel]):10.3f}" for f in FIELDS)
            print(f"{tag:16s}{cells}   {len(sel)}")
        return

    task = DelayedXOR()
    model = Plexus(
        task.n_inputs,
        task.n_classes,
        column=ColumnConfig(
            n_neurons=args.neurons,
            lr=args.lr,
            seed=args.seed,
            hebbian=bool(args.hebbian),
            hebb_lr=args.hebb_lr,
            bind_scale=args.bind_scale,
        ),
        seed=args.seed,
    )
    rng = np.random.default_rng(1000 + args.seed)
    model.train(task, args.episodes, rng=rng, report_every=10**9)

    X, y = collect(model, task, args.collect, np.random.default_rng(11 + args.seed))
    split = int(0.7 * len(X))
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    row = dict(
        tag=args.tag, seed=args.seed, episodes=args.episodes, neurons=args.neurons,
        linear=logistic_score(logistic(Xtr, ytr), Xte, yte),
        mlp=mlp(Xtr, ytr, Xte, yte),
        sparsity=model.column.sparsity,
        bound=model.column.bound_fraction,
    )
    with OUT.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"{args.tag} seed={args.seed}: " + "  ".join(f"{f} {row[f]:.3f}" for f in FIELDS))


if __name__ == "__main__":
    main()
