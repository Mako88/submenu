"""What does the column actually still know when it is asked to answer?

Delayed XOR can fail in two quite different ways, and the fix is different for
each:

    1. The cues are gone. Nothing about A or B survives the delay, so no
       readout of any kind could work. That is a dynamics/memory problem.
    2. The cues survive but their conjunction is not linearly accessible. A and
       B are each decodable, XOR is not. That is a mixing problem -- the job
       the plasticity rule is supposed to do, since the readout is linear and
       XOR is provably not linearly separable in the cue indicators.

This probe distinguishes them by freezing the column, capturing its state at
answer time, and decoding A, B, and A xor B from it -- linearly, and then with
a small MLP. The MLP is the key comparison: if it recovers XOR from the same
state a linear model fails on, the information is present and merely tangled,
and the column's job is to untangle it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plexus import ColumnConfig, DelayedXOR, Plexus  # noqa: E402
from plexus.readout import softmax  # noqa: E402


def logistic(X, y, n_classes=2, epochs=400, lr=0.5, l2=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    Xb = np.hstack([X, np.ones((len(X), 1))])
    W = rng.normal(0, 0.01, size=(n_classes, Xb.shape[1]))
    Y = np.eye(n_classes)[y]
    for _ in range(epochs):
        p = softmax(Xb @ W.T)
        W -= lr * ((p - Y).T @ Xb / len(Xb) + l2 * W)
    return W


def logistic_score(W, X, y):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    return float((np.argmax(Xb @ W.T, axis=1) == y).mean())


def mlp(Xtr, ytr, Xte, yte, hidden=64, epochs=900, lr=0.05, seed=0):
    rng = np.random.default_rng(seed)
    d = Xtr.shape[1]
    W1 = rng.normal(0, np.sqrt(2.0 / d), size=(d, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, np.sqrt(2.0 / hidden), size=(hidden, 2))
    b2 = np.zeros(2)
    Y = np.eye(2)[ytr]
    for _ in range(epochs):
        h = np.maximum(Xtr @ W1 + b1, 0.0)
        p = softmax(h @ W2 + b2)
        g = (p - Y) / len(Xtr)
        gh = (g @ W2.T) * (h > 0)
        W2 -= lr * (h.T @ g)
        b2 -= lr * g.sum(0)
        W1 -= lr * (Xtr.T @ gh)
        b1 -= lr * gh.sum(0)
    h = np.maximum(Xte @ W1 + b1, 0.0)
    return float((np.argmax(h @ W2 + b2, axis=1) == yte).mean())


def collect(model, task, n, rng):
    """Run frozen episodes and grab the readout's view at answer time."""
    states, labels = [], []
    for _ in range(n):
        ep = task.episode(rng)
        model.column.learning = False
        model.readout.learning = False
        model.column.reset_state()
        model.readout.reset()
        model.transport.reset()
        answered = 0
        snap = None
        for k in range(ep.inputs.shape[0]):
            t = model._t
            model._t += 1
            act = model.column.step(t, ep.inputs[k])
            model.readout.observe(act)
            model.transport.broadcast(t, model._zero_mod)
            if ep.response[k]:
                answered += 1
                if answered == model.answer_steps:
                    snap = model.readout.normalized.copy()
        states.append(snap)
        labels.append(ep.label)
    return np.array(states, dtype=np.float64), np.array(labels)


def measure(model, task, n, seed=11):
    X, y = collect(model, task, n, np.random.default_rng(seed))
    split = int(0.7 * len(X))
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    return logistic_score(logistic(Xtr, ytr), Xte, yte), mlp(Xtr, ytr, Xte, yte)


def build(neurons, lr, seed):
    task = DelayedXOR()
    return task, Plexus(
        task.n_inputs,
        task.n_classes,
        column=ColumnConfig(n_neurons=neurons, lr=lr, seed=seed),
        seed=seed,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--neurons", type=int, default=192)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--compare",
        type=int,
        default=0,
        metavar="EPISODES",
        help="train plastic and frozen columns for EPISODES and compare how "
        "linearly decodable their representations become",
    )
    args = ap.parse_args()

    if args.compare:
        # This is the cleanest measurement of what the plasticity rule does,
        # because it does not depend on the online readout's sample efficiency
        # at all. It asks only one thing: after the same amount of experience,
        # is the column's own representation more linearly separable when the
        # three-factor rule was running than when it was frozen?
        print(f"training {args.compare} episodes per condition, then probing\n")
        for label, lr in (("plastic", ColumnConfig.lr), ("frozen ", 0.0)):
            task, model = build(args.neurons, lr, args.seed)
            rng = np.random.default_rng(1000 + args.seed)
            model.train(task, args.compare, rng=rng, report_every=10**9)
            lin, nonlin = measure(model, task, args.n)
            print(f"{label}  linear {lin:.3f}   MLP {nonlin:.3f}")
        print("\nA higher linear score for 'plastic' means the rule made the")
        print("representation easier to read -- which is its entire job.")
        return

    task, model = build(args.neurons, ColumnConfig.lr, args.seed)
    rng = np.random.default_rng(7)
    # Let homeostasis settle before probing, so we measure the operating regime
    # rather than the initial transient.
    for _ in range(args.warmup):
        model.run_episode(task.episode(rng), learn=True)

    lin, nonlin = measure(model, task, args.n)
    print(f"probing {args.neurons} neurons, {args.n} episodes, state at answer time\n")
    print(f"XOR decodable, linear : {lin:.3f}   (chance 0.500)")
    print(f"XOR decodable, MLP    : {nonlin:.3f}   (chance 0.500)")
    print()
    if nonlin > 0.65 and lin < 0.60:
        print("=> Cues survive but the conjunction is tangled: a MIXING problem.")
        print("   The column must form the conjunction; a linear readout cannot.")
    elif nonlin < 0.60:
        print("=> Neither decoder recovers XOR: the cues do not survive the delay.")
        print("   This is a MEMORY problem -- fix the dynamics before the learning rule.")
    else:
        print("=> XOR is already linearly decodable from the frozen column.")


if __name__ == "__main__":
    main()
