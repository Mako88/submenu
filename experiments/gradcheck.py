"""Does the eligibility trace actually approximate the gradient it stands for?

Everything in the learning rule rests on one claim: that ``elig[n,b,s]`` tracks
how much neuron n's filtered output would change if synapse (n,b,s) were made
slightly stronger. If that claim is false, no learning signal can help and no
amount of tuning will rescue the rule -- it would be descending a direction
unrelated to the one it believes it is descending.

Nothing so far has tested it. Two null sweeps are exactly what a broken
eligibility trace would produce, and the forward-pass bug earlier in this
project is a standing reminder that a quantity can look healthy while being
disconnected from what it is supposed to measure.

The check is a finite difference: nudge one weight, replay the identical input,
and see how much the neuron's filtered output actually moved. Recurrence is
disabled so the comparison is clean -- eligibility only ever captures the direct
path, so with feedback in the loop a mismatch would be ambiguous between "the
trace is wrong" and "the trace is a truncated approximation, as designed".

    python3 experiments/gradcheck.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plexus import Column, ColumnConfig, LocalTransport  # noqa: E402


def build(cfg: ColumnConfig) -> Column:
    transport = LocalTransport(cfg.n_external + cfg.n_neurons, cfg.delay_max + 2, cfg.modulator_dim)
    col = Column(cfg, transport)
    col.learning = False  # freeze homeostasis so replays are identical
    return col


def replay(cfg: ColumnConfig, drive: np.ndarray, W: np.ndarray, tau: float, state=None):
    """Run a fixed input sequence and return (filtered output, eligibility).

    ``state`` carries the settled thresholds and dendritic knees from a warm-up
    run. They must be identical across replays or the finite difference would
    measure homeostasis reacting rather than the weight change itself -- and
    with them left at their initial values the column never reaches threshold,
    so every replay returns silence and the check reports nothing at all.
    """
    col = build(cfg)
    col.W = W.copy()
    if state is not None:
        col.theta, col.knee = state[0].copy(), state[1].copy()
    # Target is the neuron's TOTAL emission over the run, not its filtered
    # value at one instant. At 2% sparsity an instantaneous filtered output is
    # a count of one or two spikes, so a finite difference on it measures
    # discrete spike-shuffling rather than a gradient. Integrating over the run
    # averages that out, and the matching analytic quantity is the unfiltered
    # sensitivity summed over the same window.
    total = np.zeros(cfg.n_neurons, dtype=np.float64)
    sens = np.zeros_like(col.W, dtype=np.float64)
    for t, ext in enumerate(drive):
        total += col.step(t, ext)
        sens += col.sensitivity
    return total, sens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neurons", type=int, default=24)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--samples", type=int, default=45)
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--surrogate", default="window",
                    choices=["window", "graded", "hybrid"])
    args = ap.parse_args()

    tau = 70.0
    cfg = ColumnConfig(
        n_neurons=args.neurons,
        n_external=16,
        n_branches=4,
        n_synapses=8,
        tau_eligibility=tau,
        surrogate=args.surrogate,
        fanin_recurrent_frac=0.0,  # feedforward: eligibility should be exact
        stp=False,  # STP is a separate nonlinearity, not part of dW
        seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)
    drive = (rng.random((args.steps, cfg.n_external)) < 0.05).astype(np.float32)
    # Warm up with homeostasis live so the column reaches its operating point,
    # then freeze what it settled on and reuse it for every replay.
    warm = build(cfg)
    warm.learning = True
    warm_rng = np.random.default_rng(args.seed + 7)
    for t in range(4000):
        warm.step(t, (warm_rng.random(cfg.n_external) < 0.05).astype(np.float32))
    state = (warm.theta.copy(), warm.knee.copy())
    print(f"warm-up sparsity {warm.sparsity:.4f}, median threshold {np.median(warm.theta):.3f}")
    base_W = warm.W.copy()

    base_trace, elig = replay(cfg, drive, base_W, tau, state)
    print(f"baseline total emission: mean {base_trace.mean():.3f}, max {base_trace.max():.3f}")

    # Sample synapses that carry some eligibility; a synapse whose trace is zero
    # tells us nothing either way.
    flat = np.abs(elig).ravel()
    candidates = np.argsort(flat)[-args.samples * 6 :]
    picks = rng.choice(candidates, size=min(args.samples, len(candidates)), replace=False)

    analytic, numeric = [], []
    for idx in picks:
        n, b, s = np.unravel_index(idx, elig.shape)
        W = base_W.copy()
        W[n, b, s] += args.eps
        moved, _ = replay(cfg, drive, W, tau, state)
        analytic.append(elig[n, b, s])
        numeric.append((moved[n] - base_trace[n]) / args.eps)

    a = np.array(analytic, dtype=np.float64)
    d = np.array(numeric, dtype=np.float64)
    nz = np.abs(d) > 1e-12
    print(f"sampled {len(a)} synapses; {int(nz.sum())} produced a measurable change\n")
    if nz.sum() < 5:
        print("=> Too few measurable changes to judge. Try a larger --eps.")
        return

    r = float(np.corrcoef(a[nz], d[nz])[0, 1])
    sign_match = float((np.sign(a[nz]) == np.sign(d[nz])).mean())
    scale = float(np.median(d[nz] / a[nz]))
    n = int(nz.sum())
    # Two-sided binomial tail for the sign test, so "better than chance" is a
    # claim rather than an impression.
    k = int(round(sign_match * n))
    from math import comb
    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2**n
    print(f"correlation(analytic, finite-difference) : {r:+.3f}")
    print(f"sign agreement                           : {sign_match:.2f}  "
          f"({k}/{n}, one-sided p = {tail:.4f})")
    print(f"median ratio (numeric / analytic)        : {scale:+.4f}")
    print()
    # Direction and magnitude are separate claims and here they disagree, so
    # report them separately rather than collapsing to a single verdict.
    directional = tail < 0.05 and sign_match > 0.5
    proportional = r > 0.3
    if directional and proportional:
        print("=> Trace tracks the gradient in both sign and magnitude.")
        print("   The rule descends what it thinks it does; look elsewhere.")
    elif directional:
        print("=> Trace carries DIRECTION but not MAGNITUDE. Updates point the")
        print("   right way more often than chance, while their sizes are")
        print("   uninformative -- so every step mixes real signal with noise")
        print("   of comparable scale. That is what a rule which neither helps")
        print("   nor destroys looks like, and it matches both null sweeps.")
    else:
        print("=> Trace does NOT track the gradient in sign or magnitude.")
        print("   No learning signal can help while this holds.")


if __name__ == "__main__":
    main()
