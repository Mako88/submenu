"""Run ONE condition of the plasticity comparison and append its result.

Split into single-condition invocations so each run finishes quickly and its
result is durable, rather than depending on one long process surviving.

    python3 experiments/compare_step.py --lr 0.004 --tag plastic
    python3 experiments/compare_step.py --lr 0.0    --tag frozen
    python3 experiments/compare_step.py --report

Measures how *linearly decodable* the column's representation is after
training, which is the cleanest read on what the three-factor rule does: it
does not depend on the online readout's sample efficiency at all.
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

OUT = Path(__file__).resolve().parent / "compare_results.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=ColumnConfig.lr)
    ap.add_argument("--tag", default="plastic")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--collect", type=int, default=600)
    ap.add_argument("--neurons", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-stp", action="store_true")
    ap.add_argument("--elig-norm", default="column", choices=["column","neuron","none"])
    ap.add_argument("--scaling-lr", type=float, default=ColumnConfig.scaling_lr)
    ap.add_argument("--pretrain", type=int, default=0,
                    help="episodes of readout-only training before column plasticity")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.report:
        rows = [json.loads(x) for x in OUT.read_text().splitlines() if x.strip()]
        tags = sorted({r["tag"] for r in rows})
        print(f"{'condition':10s} {'linear':>16s} {'MLP':>16s}   seeds")
        for tag in tags:
            sel = [r for r in rows if r["tag"] == tag]
            lin = [r["linear"] for r in sel]
            nl = [r["mlp"] for r in sel]
            print(
                f"{tag:10s} {np.mean(lin):.3f} +/- {np.std(lin):.3f}   "
                f"{np.mean(nl):.3f} +/- {np.std(nl):.3f}   {len(sel)}"
            )
        return

    task = DelayedXOR()
    model = Plexus(
        task.n_inputs,
        task.n_classes,
        column=ColumnConfig(n_neurons=args.neurons, lr=args.lr, seed=args.seed,
                            stp=not args.no_stp, elig_norm=args.elig_norm,
                            scaling_lr=args.scaling_lr),
        seed=args.seed,
    )
    rng = np.random.default_rng(1000 + args.seed)
    if args.pretrain:
        # Let the readout find a decent decoder before the column starts taking
        # its advice. The column's credit signal is readout.W^T @ (-err), so a
        # weak readout is a noisy teacher and the column faithfully follows it.
        saved, model.column.cfg.lr = model.column.cfg.lr, 0.0
        model.train(task, args.pretrain, rng=rng, report_every=10**9)
        model.column.cfg.lr = saved
    model.train(task, args.episodes, rng=rng, report_every=10**9)

    X, y = collect(model, task, args.collect, np.random.default_rng(11 + args.seed))
    split = int(0.7 * len(X))
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    lin = logistic_score(logistic(Xtr, ytr), Xte, yte)
    nl = mlp(Xtr, ytr, Xte, yte)

    row = dict(tag=args.tag, lr=args.lr, seed=args.seed, episodes=args.episodes,
               neurons=args.neurons, linear=lin, mlp=nl)
    with OUT.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"{args.tag} seed={args.seed}: linear {lin:.3f}  MLP {nl:.3f}")


if __name__ == "__main__":
    main()
