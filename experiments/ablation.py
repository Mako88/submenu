"""Does the column's local plasticity actually do anything?

The failure mode this guards against is embarrassing and easy to miss: a
randomly wired recurrent population with heterogeneous time constants is a
reservoir, and a trained linear readout on a reservoir solves a lot of temporal
tasks by itself. If we do not run the frozen-column control, we can spend
weeks admiring a number that the plasticity rule had no part in producing.

Conditions:
    full      -- three-factor plasticity on, symmetric feedback
    frozen    -- column weights frozen; readout still trains (reservoir control)
    dfa       -- plasticity on, but credit arrives through a fixed random
                 projection instead of the readout weights
    lagged    -- plasticity on, modulator delayed by 200 steps, to check that
                 eligibility traces really do bridge a stale learning signal
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plexus import ColumnConfig, DelayedXOR, Plexus, TemporalPatterns  # noqa: E402

CONDITIONS = {
    "full": dict(lr=4e-3, feedback_mode="symmetric", modulator_lag=0),
    "frozen": dict(lr=0.0, feedback_mode="symmetric", modulator_lag=0),
    "dfa": dict(lr=4e-3, feedback_mode="dfa", modulator_lag=0),
    "lagged": dict(lr=4e-3, feedback_mode="symmetric", modulator_lag=200),
}


def build_task(name: str):
    if name == "xor":
        return DelayedXOR()
    if name == "patterns":
        return TemporalPatterns(n_classes=4, seed=3)
    raise ValueError(f"unknown task {name!r}")


def run(condition: str, task_name: str, episodes: int, seed: int, neurons: int, quiet: bool):
    spec = CONDITIONS[condition]
    task = build_task(task_name)
    cfg = ColumnConfig(n_neurons=neurons, lr=spec["lr"], seed=seed)
    model = Plexus(
        task.n_inputs,
        task.n_classes,
        column=cfg,
        feedback_mode=spec["feedback_mode"],
        modulator_lag=spec["modulator_lag"],
        seed=seed,
    )
    rng = np.random.default_rng(1000 + seed)
    t0 = time.time()

    def report(r):
        if not quiet:
            print(
                f"  [{condition}/s{seed}] ep {r.episode:5d}  loss {r.loss:.3f}  "
                f"acc {r.accuracy:.3f}  sparsity {r.sparsity:.4f}",
                flush=True,
            )

    model.train(task, episodes, rng=rng, report_every=max(episodes // 10, 1), on_report=report)
    acc = model.evaluate(task, 120, rng=np.random.default_rng(9999))
    return {
        "condition": condition,
        "seed": seed,
        "eval_accuracy": acc,
        "sparsity": model.column.sparsity,
        "plateau_engagement": model.column.plateau_engagement,
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="xor", choices=["xor", "patterns"])
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--neurons", type=int, default=192)
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    results = []
    for condition in args.conditions:
        for seed in range(args.seeds):
            res = run(condition, args.task, args.episodes, seed, args.neurons, args.quiet)
            print(
                f"{condition:8s} seed {seed}  eval_acc {res['eval_accuracy']:.3f}  "
                f"({res['seconds']}s)",
                flush=True,
            )
            results.append(res)

    print(f"\n=== {args.task}: {args.episodes} episodes, {args.seeds} seeds ===")
    chance = 1.0 / build_task(args.task).n_classes
    print(f"chance = {chance:.3f}")
    for condition in args.conditions:
        accs = [r["eval_accuracy"] for r in results if r["condition"] == condition]
        print(
            f"{condition:8s} eval acc {np.mean(accs):.3f} +/- {np.std(accs):.3f}   "
            f"{[round(a, 3) for a in accs]}"
        )

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
