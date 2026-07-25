"""Does allocation actually allocate?

The engram mechanism makes four claims, and end-to-end accuracy tests none of
them individually. Ten sweeps have already shown that a single downstream
number cannot tell you which part of a mechanism is working, so this measures
the parts:

    excite   -- correlation between a neuron's excitability bias and how active
                it is at allocation time. If this is ~0, the drift is decorative
                and the winners are decided by input alone. This is the quantity
                uniform-rate homeostasis destroys, and the reason the setpoint
                had to become per-neuron.

    alloc    -- fraction of the population recruited. Should sit near
                `alloc_frac`; a mechanism that recruits everybody has not
                allocated anything.

    rotate   -- overlap between the engrams of *consecutive* episodes, relative
                to what independent sampling would give. Below 1.0 means the
                allocation refractory is working: being recruited pushes the
                next memory somewhere else.

    separate -- overlap between engrams of episodes with the same cue pattern
                minus the overlap for different cue patterns. Above zero means
                recruitment is content-addressed, i.e. the engram is a
                representation of *what happened* rather than a rotating
                lottery. This is the claim that would make the mechanism useful,
                and the one most likely to be false.

`rotate` and `separate` pull against each other on purpose: pure refractoriness
gives rotation with no separation, pure content-addressing gives separation with
no rotation. A memory system needs both, so measuring only one would hide a
mechanism that had traded the other away.

    python3 experiments/engram.py --tag on --engram 1 --seed 0
    python3 experiments/engram.py --report
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

OUT = Path(__file__).resolve().parent / "engram_results.jsonl"
FIELDS = ["excite", "alloc", "rotate", "separate", "linear", "mlp", "sparsity"]


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = float((a | b).sum())
    return float((a & b).sum()) / union if union else 0.0


def run_and_record(model, task, episodes, rng):
    """Train, capturing the engram and the excitability that produced it.

    Everything here is read off an ordinary training run. There is no special
    measurement mode, because a mechanism that only shows up when you stop
    perturbing it is not the mechanism doing the work.
    """
    tags, patterns, corrs = [], [], []
    col = model.column
    for _ in range(episodes):
        ep = task.episode(rng)
        # Excitability is sampled *before* the episode: allocation is supposed
        # to be biased by who was excitable going in, not by who the episode
        # happened to excite. Sampling after would let the input take credit.
        # The regressor is xi relative to the neuron's own long-run level,
        # because that is the quantity the recruitment rule actually reads. Raw
        # xi is confounded by the mechanism's own output: being recruited drops
        # excitability, so a unit that keeps winning carries a chronically low
        # xi and the raw correlation reads negative even when the transient
        # bias is doing exactly its job.
        xi_before = (col.xi - col.xi_slow).copy()
        model.run_episode(ep, learn=True)
        tag = col.tag.copy()
        if tag.any() and not tag.all() and xi_before.std() > 1e-9:
            corrs.append(float(np.corrcoef(xi_before, tag.astype(np.float64))[0, 1]))
        tags.append(tag)
        patterns.append(ep.bits)
    return np.array(tags), patterns, corrs


def summarise(tags, patterns, corrs, tail=200):
    tags, patterns = tags[-tail:], patterns[-tail:]
    sizes = tags.mean(axis=1)
    n = len(tags)

    consecutive = [jaccard(tags[i], tags[i + 1]) for i in range(n - 1)]
    # Chance overlap for two independently drawn sets of the observed sizes.
    # Comparing raw overlap across conditions would be meaningless: a condition
    # that recruits more neurons overlaps more for free.
    pa, pb = sizes[:-1], sizes[1:]
    chance = np.where(
        (pa + pb - pa * pb) > 0, pa * pb / (pa + pb - pa * pb + 1e-12), 0.0
    )
    rotate = float(np.mean(consecutive) / (np.mean(chance) + 1e-12))

    same, diff = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (same if patterns[i] == patterns[j] else diff).append(jaccard(tags[i], tags[j]))
    separate = (float(np.mean(same)) - float(np.mean(diff))) if same and diff else 0.0

    return dict(
        excite=float(np.mean(corrs)) if corrs else 0.0,
        alloc=float(np.mean(sizes)),
        rotate=rotate,
        separate=separate,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="on")
    ap.add_argument("--engram", type=int, default=1)
    ap.add_argument("--excite-setpoint", type=float, default=1.0)
    ap.add_argument("--excite-threshold", type=float,
                    default=ColumnConfig.excite_threshold)
    ap.add_argument("--excite-gain", type=float, default=ColumnConfig.excite_gain)
    ap.add_argument("--hebb-lr", type=float, default=0.02)
    ap.add_argument("--alloc-drop", type=float, default=0.6)
    ap.add_argument("--excite-drift", type=float, default=0.35)
    ap.add_argument("--bind-mode", default="tagged", choices=["tagged", "graded"])
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--collect", type=int, default=600)
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
            engram=bool(args.engram),
            excite_setpoint=args.excite_setpoint,
            excite_threshold=args.excite_threshold,
            excite_gain=args.excite_gain,
            hebb_lr=args.hebb_lr,
            alloc_drop=args.alloc_drop,
            excite_drift=args.excite_drift,
            bind_mode=args.bind_mode,
        ),
        seed=args.seed,
    )
    rng = np.random.default_rng(1000 + args.seed)
    tags, patterns, corrs = run_and_record(model, task, args.episodes, rng)
    row = summarise(tags, patterns, corrs)

    X, y = collect(model, task, args.collect, np.random.default_rng(11 + args.seed))
    split = int(0.7 * len(X))
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    row["linear"] = logistic_score(logistic(Xtr, ytr), Xte, yte)
    row["mlp"] = mlp(Xtr, ytr, Xte, yte)
    row["sparsity"] = model.column.sparsity

    row.update(tag=args.tag, seed=args.seed, episodes=args.episodes, neurons=args.neurons)
    with OUT.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"{args.tag} seed={args.seed}: " + "  ".join(f"{f} {row[f]:.3f}" for f in FIELDS))


if __name__ == "__main__":
    main()
