"""Train a single column and report what its dynamics settled into.

Run:
    python3 experiments/demo.py --task xor
    python3 experiments/demo.py --task patterns --modulator-lag 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plexus import ColumnConfig, DelayedXOR, Plexus, TemporalPatterns  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="xor", choices=["xor", "patterns"])
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--neurons", type=int, default=192)
    ap.add_argument("--modulator-lag", type=int, default=0)
    ap.add_argument("--feedback", default="symmetric", choices=["symmetric", "dfa"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    task = DelayedXOR() if args.task == "xor" else TemporalPatterns(n_classes=4, seed=3)
    model = Plexus(
        task.n_inputs,
        task.n_classes,
        column=ColumnConfig(n_neurons=args.neurons, seed=args.seed),
        feedback_mode=args.feedback,
        modulator_lag=args.modulator_lag,
        seed=args.seed,
    )

    print(f"task={args.task}  neurons={args.neurons}  feedback={args.feedback}  "
          f"modulator_lag={args.modulator_lag}")
    print(f"chance = {1.0 / task.n_classes:.3f}\n")

    def report(r):
        print(
            f"ep {r.episode:5d}  loss {r.loss:.3f}  acc {r.accuracy:.3f}  "
            f"sparsity {r.sparsity:.4f}  mean|W| {r.mean_weight:.3f}",
            flush=True,
        )

    model.train(
        task,
        args.episodes,
        rng=np.random.default_rng(args.seed + 1000),
        report_every=max(args.episodes // 12, 1),
        on_report=report,
    )

    acc = model.evaluate(task, 200, rng=np.random.default_rng(9999))
    col = model.column
    print(f"\nheld-out accuracy   {acc:.3f}")
    print(f"sparsity            {col.sparsity:.4f}  (target {col.cfg.target_rate})")
    print(f"plateau engagement  {col.plateau_engagement:.4f}  (target {col.cfg.plateau_engagement})")
    print(
        f"threshold  min/med/max  {col.theta.min():.3f} / "
        f"{np.median(col.theta):.3f} / {col.theta.max():.3f}"
    )
    print(
        f"tau_soma   min/med/max  {col.tau_soma.min():.1f} / "
        f"{np.median(col.tau_soma):.1f} / {col.tau_soma.max():.1f} ms"
    )
    frac_inh = float((col.source_sign < 0).mean())
    print(f"inhibitory fraction {frac_inh:.3f}")


if __name__ == "__main__":
    main()
