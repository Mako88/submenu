"""End-to-end accuracy for one (condition, seed), appended to a results file.

Each invocation does one run and records it, so a multi-seed comparison can be
accumulated across separate short runs instead of depending on one long process.

    python3 experiments/e2e.py --tag frozen  --lr 0
    python3 experiments/e2e.py --tag plastic --lr 0.005
    python3 experiments/e2e.py --report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plexus import ColumnConfig, DelayedXOR, Plexus  # noqa: E402

OUT = Path(__file__).resolve().parent / "e2e_results.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="plastic")
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--tau-elig", type=float, default=70.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--eval", type=int, default=150)
    ap.add_argument("--neurons", type=int, default=96)
    ap.add_argument("--readout-lr", type=float, default=0.5)
    ap.add_argument("--readout-rule", default="delta", choices=["delta","rls"])
    ap.add_argument("--modulator-lag", type=int, default=0)
    # Held constant across lag conditions on purpose: the drain tail is not
    # inert, so letting it track the lag confounds every lag comparison.
    ap.add_argument("--drain-steps", type=int, default=200)
    ap.add_argument("--elig-mode", default="magnitude", choices=["magnitude","sign"])
    ap.add_argument("--surrogate", default="window", choices=["window","graded","hybrid"])
    ap.add_argument("--pretrain", type=int, default=0,
                    help="episodes of readout-only training before column plasticity")
    ap.add_argument("--hebbian", type=int, default=0,
                    help="salience-gated Hebbian binding (sweep 014/015)")
    ap.add_argument("--feedback", default="symmetric")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.report:
        rows = [json.loads(x) for x in OUT.read_text().splitlines() if x.strip()]
        print(f"{'condition':14s} {'eval accuracy':>18s}   n   per-seed")
        for tag in sorted({r["tag"] for r in rows}):
            sel = sorted((r for r in rows if r["tag"] == tag), key=lambda r: r["seed"])
            a = [r["acc"] for r in sel]
            print(
                f"{tag:14s} {np.mean(a):.3f} +/- {np.std(a):.3f}   {len(a)}   "
                f"{[round(x, 3) for x in a]}"
            )
        return

    task = DelayedXOR()
    model = Plexus(
        task.n_inputs,
        task.n_classes,
        column=ColumnConfig(
            n_neurons=args.neurons, lr=args.lr, tau_eligibility=args.tau_elig,
            elig_mode=args.elig_mode, surrogate=args.surrogate, seed=args.seed,
            hebbian=bool(args.hebbian)
        ),
        readout_lr=args.readout_lr,
        readout_rule=args.readout_rule,
        feedback_mode=args.feedback,
        modulator_lag=args.modulator_lag,
        drain_steps=args.drain_steps,
        seed=args.seed,
    )
    rng = np.random.default_rng(1000 + args.seed)
    if args.pretrain:
        # Let the readout converge before the column starts taking its advice.
        # The column's credit signal is readout.W^T @ (-err), so a half-trained
        # decoder is a noisy teacher and the column follows it faithfully. Both
        # nulls so far are consistent with that, and this is the test.
        saved, model.column.cfg.lr = model.column.cfg.lr, 0.0
        model.train(task, args.pretrain, rng=rng, report_every=10**9)
        model.column.cfg.lr = saved
    model.train(task, args.episodes, rng=rng, report_every=10**9)
    acc = model.evaluate(task, args.eval, rng=np.random.default_rng(9999))

    with OUT.open("a") as fh:
        fh.write(json.dumps(dict(tag=args.tag, lr=args.lr, seed=args.seed, acc=acc,
                                 tau_elig=args.tau_elig, episodes=args.episodes,
                                 modulator_lag=args.modulator_lag,
                                 drain_steps=args.drain_steps,
                                 hebbian=args.hebbian,
                                 pretrain=args.pretrain, elig_mode=args.elig_mode,
                                 surrogate=args.surrogate,
                                 readout_rule=args.readout_rule)) + "\n")
    print(f"{args.tag} seed={args.seed}: eval {acc:.3f}")


if __name__ == "__main__":
    main()
