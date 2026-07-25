"""What does lateral inhibition actually do? It is not what it was built for.

IN PLAIN TERMS
--------------
One of the two mechanisms that helps in this project is a rule that makes
neurons which fire together inhibit each other. It was added for a specific
reason -- to spread the network's activity out so that different neurons carry
different information -- and that reason turned out to be wrong: sweep 020
measured it and the spreading does not happen.

The awkward part is that it still helps. It makes the network's state easier to
read by a useful margin, while leaving every property anyone has measured
looking exactly the same: same firing rate, same amount of redundancy between
neurons, same number of independent directions in the state.

So it is doing *something*, and nobody knows what. The README says as much and
says not to write an explanation into the code until one exists. This is the
probe that tries to produce one.

THE PUZZLE, IN NUMBERS
----------------------
Sweep 020, 20 paired seeds, frozen column:

                    off     lateral
    linear         0.802     0.830     +0.028, p = 0.0297
    correlation    0.178     0.175     unchanged
    eff_rank      17.404    17.139     unchanged
    sparsity       0.031     0.031     unchanged

A mechanism that improves linear readability while moving neither the
dimensionality nor the redundancy nor the firing rate has to be changing
*which directions carry the label*, not the shape of the representation. There
is not much else left.

WHAT THIS MEASURES
------------------
Three quantities the existing metrics cannot see, all of them about the
relationship between the state's variance and the label rather than about the
state alone:

  alignment    The label direction is `mu_1 - mu_0`, the difference between the
               two class means. Project it onto the top-k eigenvectors of the
               state's **correlation** matrix and report the fraction of its
               norm that survives. A linear decoder trained on finite data
               resolves leading directions easily and trailing ones badly, so
               rotating the label *into* the top components makes it readable
               without changing the spectrum's shape at all -- which is exactly
               the combination sweep 020 measured.

               Correlation rather than covariance, and it is not a detail. The
               features are standardised first, so the eigenbasis is of the
               correlation matrix; that is deliberate, because `probe.logistic`
               standardises its inputs too, and the whole point is to explain
               what *that* decoder sees. The first ground-truth check written
               for this probe placed the label along a high-*variance* feature
               axis, which standardisation erases, and the metric duly reported
               0.232 against a chance level of 0.323 -- below chance for a label
               sitting exactly where the test claimed to put it. The probe was
               right and the test was asking a different question.
               `test_alignment_probe_recovers_a_known_label_direction` is the
               corrected version and is what makes any number here trustworthy.

  fisher       Class separation in units of within-class spread:
               `|mu_1 - mu_0|^2 / tr(within-class covariance)`. This is
               readability with the decoder taken out of it, so it separates
               "the representation got better" from "the decoder found it
               easier".

  margin       Separation along the *best* direction, `w = S_w^-1 (mu_1 - mu_0)`
               with a ridge. The Fisher discriminant's own score.

Alignment is the hypothesis; the other two are there so that a null on
alignment still says something. If separability rises while alignment does not,
the mechanism is improving the representation rather than rotating it, and the
question moves to what the extra separation is made of.

WHY THESE AND NOT A DECODER
-----------------------------
`binding.py` already fits a decoder and reports the score. Fitting another one
would measure the same thing again. Every quantity here is computed from the
class-conditional means and covariances directly, so none of them depends on a
decoder's sample efficiency -- which sweep 009 established can hide a
column-level difference entirely, and sweep 028 established moves the headline
number by 0.023 on its own.

    python3 experiments/geometry.py --tag lateral --lateral 1 --seed 0
    python3 experiments/geometry.py --report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plexus import ColumnConfig, DelayedXOR, Plexus  # noqa: E402
from probe import collect  # noqa: E402

OUT = Path(__file__).resolve().parent / "geometry_results.jsonl"
FIELDS = ["align5", "align10", "align_rand", "fisher", "margin", "eff_rank"]


def geometry(X: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> dict:
    """Where the label sits relative to the state's own variance directions.

    Standardised per feature first, for the reason standards rule 8 exists: the
    neurons run at different rates, so an unstandardised covariance would be
    dominated by whichever units happen to be loudest and the principal
    components would describe firing rate rather than structure.
    """
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    a, b = X[y == 0], X[y == 1]
    delta = b.mean(0) - a.mean(0)
    dnorm = float(np.linalg.norm(delta))

    # Principal axes of the *pooled* state, which is what a decoder sees.
    C = (X.T @ X) / len(X)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]

    def captured(k: int) -> float:
        if dnorm <= 0.0:
            return float("nan")
        proj = evecs[:, :k].T @ delta
        return float(np.linalg.norm(proj) / dnorm)

    # The null this has to beat. A random direction in n dimensions already puts
    # sqrt(k/n) of its norm in any k-subspace, so an alignment of 0.4 means
    # nothing on its own at n = 96, k = 10 -- it is exactly chance. Reported
    # alongside rather than subtracted, so the comparison is visible.
    n = X.shape[1]
    align_rand = float(np.sqrt(10.0 / n))

    within = ((a - a.mean(0)).T @ (a - a.mean(0))
              + (b - b.mean(0)).T @ (b - b.mean(0))) / len(X)
    fisher = float(dnorm**2 / (np.trace(within) + 1e-12))
    w = np.linalg.solve(within + ridge * np.eye(n), delta)
    margin = float((delta @ w) / (np.sqrt(w @ within @ w) + 1e-12))

    ev = np.clip(evals, 0.0, None)
    eff_rank = float(ev.sum() ** 2 / max((ev**2).sum(), 1e-12))
    return dict(align5=captured(5), align10=captured(10), align_rand=align_rand,
                fisher=fisher, margin=margin, eff_rank=eff_rank)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="off")
    ap.add_argument("--hebbian", type=int, default=0)
    ap.add_argument("--lateral", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--collect", type=int, default=500)
    ap.add_argument("--neurons", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.report:
        rows = [json.loads(x) for x in OUT.read_text().splitlines() if x.strip()]
        head = "".join(f"{f:>12s}" for f in FIELDS)
        print(f"{'condition':14s}{head}   seeds")
        for tag in sorted({r["tag"] for r in rows}):
            sel = [r for r in rows if r["tag"] == tag]
            cells = "".join(f"{np.mean([r[f] for r in sel]):12.4f}" for f in FIELDS)
            print(f"{tag:14s}{cells}   {len(sel)}")
        return

    task = DelayedXOR()
    model = Plexus(
        task.n_inputs, task.n_classes,
        column=ColumnConfig(
            n_neurons=args.neurons, lr=args.lr, seed=args.seed,
            hebbian=bool(args.hebbian), lateral=bool(args.lateral),
        ),
        seed=args.seed,
    )
    model.train(task, args.episodes, rng=np.random.default_rng(1000 + args.seed),
                report_every=10**9)

    X, y = collect(model, task, args.collect, np.random.default_rng(11 + args.seed))
    row = dict(tag=args.tag, seed=args.seed, neurons=args.neurons,
               collect=args.collect, **geometry(X, y))
    with OUT.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"{args.tag} seed={args.seed}: "
          + "  ".join(f"{f} {row[f]:.4f}" for f in FIELDS))


if __name__ == "__main__":
    main()
