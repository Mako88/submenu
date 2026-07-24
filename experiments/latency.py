"""Does inter-column latency hurt? One (peer_delay, seed) per invocation.

This is the claim the architecture exists to support. A column reads its peers
only through a conduction delay, and events are addressed by emission time, so
moving a column further away should change *when* its events land and nothing
else. Cortex runs 0.5-30ms conduction delays natively; 150 steps is roughly an
intercontinental round trip.

If accuracy is flat across that range, latency tolerance is not a hope about
the design, it is a measured property of it.

    python3 experiments/latency.py --peer-delay 30 --seed 0
    python3 experiments/latency.py --report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plexus import ColumnConfig, DelayedXOR, DistributedPlexus  # noqa: E402

OUT = Path(__file__).resolve().parent / "latency_results.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--peer-delay", type=int, default=30)
    ap.add_argument("--columns", type=int, default=4)
    ap.add_argument("--neurons-per-column", type=int, default=24)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=350)
    ap.add_argument("--eval", type=int, default=150)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.report:
        rows = [json.loads(x) for x in OUT.read_text().splitlines() if x.strip()]
        print(f"{'peer delay':>12s} {'eval accuracy':>18s}   n   per-seed")
        for d in sorted({r["peer_delay"] for r in rows}):
            a = [r["acc"] for r in sorted((r for r in rows if r["peer_delay"] == d),
                                          key=lambda r: r["seed"])]
            print(f"{d:>10d}ms {np.mean(a):.3f} +/- {np.std(a):.3f}   {len(a)}   "
                  f"{[round(x, 3) for x in a]}")
        rows_by_d = {d: [r["acc"] for r in rows if r["peer_delay"] == d]
                     for d in {r["peer_delay"] for r in rows}}
        if len(rows_by_d) > 1:
            lo, hi = min(rows_by_d), max(rows_by_d)
            drop = np.mean(rows_by_d[lo]) - np.mean(rows_by_d[hi])
            print(f"\n{lo}ms -> {hi}ms costs {drop:+.3f} accuracy "
                  f"({hi / max(lo, 1):.0f}x the latency)")
        return

    task = DelayedXOR()
    model = DistributedPlexus(
        task.n_inputs,
        task.n_classes,
        n_columns=args.columns,
        column=ColumnConfig(n_neurons=args.neurons_per_column, lr=args.lr, seed=args.seed),
        peer_delay=args.peer_delay,
        seed=args.seed,
    )
    model.train(task, args.episodes, rng=np.random.default_rng(1000 + args.seed))
    acc = model.evaluate(task, args.eval, rng=np.random.default_rng(9999))

    with OUT.open("a") as fh:
        fh.write(json.dumps(dict(peer_delay=args.peer_delay, seed=args.seed, acc=acc,
                                 columns=args.columns, lr=args.lr,
                                 neurons=args.columns * args.neurons_per_column)) + "\n")
    print(f"peer_delay={args.peer_delay}ms seed={args.seed}: eval {acc:.3f}")


if __name__ == "__main__":
    main()
