"""Sanity baselines: what do trivially simple models score on these tasks?

A benchmark is only worth reporting if simple things fail on it. Two baselines,
both given *more* raw access to the data than the column ever gets:

    rate    -- logistic regression on the per-channel sum over time. Destroys
               all timing. Must be at chance, or the task leaks its label into
               total activity.
    linear  -- logistic regression on the full flattened (time x channel)
               array, coarsely binned. Sees every event at near-full temporal
               resolution but can only combine features linearly. Delayed XOR
               must defeat it, because XOR is not linearly separable in the cue
               indicators no matter how well you can see them.

If either baseline scores well, the task is broken and any result on it is
meaningless.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plexus import DelayedXOR, TemporalPatterns  # noqa: E402


def logistic(X, y, n_classes, epochs=300, lr=0.5, l2=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    X = np.hstack([X, np.ones((X.shape[0], 1))]).astype(np.float64)
    mu, sd = X[:, :-1].mean(0), X[:, :-1].std(0) + 1e-8
    X[:, :-1] = (X[:, :-1] - mu) / sd
    W = rng.normal(0, 0.01, size=(n_classes, X.shape[1]))
    Y = np.eye(n_classes)[y]
    for _ in range(epochs):
        z = X @ W.T
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        W -= lr * ((p - Y).T @ X / len(X) + l2 * W)
    return W, mu, sd


def score(W, mu, sd, X, y):
    X = np.hstack([(X - mu) / sd, np.ones((X.shape[0], 1))])
    return float((np.argmax(X @ W.T, axis=1) == y).mean())


def featurize(episodes, mode, bins=20):
    X = []
    for ep in episodes:
        if mode == "rate":
            X.append(ep.inputs.sum(axis=0))
        else:
            T = ep.inputs.shape[0]
            edges = np.linspace(0, T, bins + 1).astype(int)
            X.append(
                np.concatenate([ep.inputs[a:b].sum(axis=0) for a, b in zip(edges, edges[1:])])
            )
    return np.array(X, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="xor", choices=["xor", "patterns"])
    ap.add_argument("--n", type=int, default=1500)
    args = ap.parse_args()

    task = DelayedXOR() if args.task == "xor" else TemporalPatterns(n_classes=4, seed=3)
    rng = np.random.default_rng(0)
    train = [task.episode(rng) for _ in range(args.n)]
    test = [task.episode(rng) for _ in range(500)]
    ytr = np.array([e.label for e in train])
    yte = np.array([e.label for e in test])

    print(f"task={args.task}  chance={1.0 / task.n_classes:.3f}  "
          f"train={args.n} test={len(test)}\n")
    for mode in ("rate", "linear"):
        Xtr, Xte = featurize(train, mode), featurize(test, mode)
        W, mu, sd = logistic(Xtr, ytr, task.n_classes)
        print(
            f"{mode:7s} ({Xtr.shape[1]:5d} features)  "
            f"train {score(W, mu, sd, Xtr, ytr):.3f}   test {score(W, mu, sd, Xte, yte):.3f}"
        )


if __name__ == "__main__":
    main()
