"""Paired comparison of the plastic and frozen conditions.

Paired, because both conditions share a seed, a training stream and an
evaluation set -- so the seed-to-seed spread (which is large here) cancels and
the test is far more sensitive than comparing two independent means.

Reports a permutation p-value rather than a t-test: eight-to-twenty paired
differences is not enough to lean on normality, and the sign-flip null is exact
under the only assumption that matters (that the labels 'plastic' and 'frozen'
are exchangeable within a seed if the rule does nothing).
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT = Path(__file__).resolve().parent / "e2e_results.jsonl"


def permutation_p(diffs: np.ndarray) -> float:
    """Exact two-sided sign-flip p-value for small samples."""
    n = len(diffs)
    observed = abs(diffs.mean())
    if n > 20:  # 2^21 sign patterns is where exhaustive stops being sensible
        rng = np.random.default_rng(0)
        signs = rng.choice([-1.0, 1.0], size=(20000, n))
        null = np.abs((signs * diffs).mean(axis=1))
        return float((null >= observed - 1e-12).mean())
    extreme = sum(
        abs(float(np.mean(np.array(flips) * diffs))) >= observed - 1e-12
        for flips in itertools.product([-1.0, 1.0], repeat=n)
    )
    return extreme / (2**n)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(DEFAULT))
    ap.add_argument("--metric", default="acc",
                    help="acc for end-to-end, linear/mlp for representation quality")
    # Which two conditions to pair. Defaults keep the plastic-vs-frozen
    # comparison the eleven recorded sweeps used, so those stay reproducible
    # with no arguments at all.
    ap.add_argument("--a", default="plastic", help="the condition under test")
    ap.add_argument("--b", default="frozen", help="the control it must beat")
    args = ap.parse_args()
    results = Path(args.file)
    if not results.exists():
        sys.exit(f"no results at {results}")
    rows = [json.loads(x) for x in results.read_text().splitlines() if x.strip()]
    for r in rows:
        r["acc"] = r[args.metric]
    frozen = {r["seed"]: r["acc"] for r in rows if r["tag"] == args.b}
    plastic = {r["seed"]: r["acc"] for r in rows if r["tag"] == args.a}
    seeds = sorted(set(frozen) & set(plastic))
    if not seeds:
        sys.exit("no seeds have both conditions")

    d = np.array([plastic[s] - frozen[s] for s in seeds])
    f = np.array([frozen[s] for s in seeds])
    p = np.array([plastic[s] for s in seeds])

    print(f"metric                : {args.metric}")
    print(f"seeds paired          : {len(seeds)}")
    print(f"{args.b:22s}: {f.mean():.3f} +/- {f.std():.3f}")
    print(f"{args.a:22s}: {p.mean():.3f} +/- {p.std():.3f}")
    print(f"mean paired difference: {d.mean():+.3f}")
    print(f"improved on           : {int((d > 0).sum())}/{len(d)} seeds")
    pval = permutation_p(d)
    print(f"permutation p (2-sided): {pval:.4f}")
    print()
    if pval < 0.05 and d.mean() > 0:
        print(f"=> {args.a} beats {args.b}.")
    elif pval < 0.05:
        print(f"=> {args.a} is reliably WORSE than {args.b}.")
    else:
        print(f"=> No detectable difference between {args.a} and {args.b}")
        print("   at this sample size.")


if __name__ == "__main__":
    main()
