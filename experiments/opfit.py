"""What are the settled operating-point values fitted to?

IN PLAIN TERMS
--------------
Every neuron here has two numbers that get tuned automatically while the network
runs: a threshold, and a "knee" that controls when its dendritic branches switch
into a boosted mode. Tuning them is the single biggest source of improvement in
this project -- bigger than every learning rule combined.

We already know the threshold half is not really being *learned*. It turns out
to be a simple formula in a property each neuron already has from the moment it
is built, so a hundred episodes of automatic tuning can be replaced by one line
of arithmetic.

This asks the same question about the knee. If the knee also has a formula, the
whole thing collapses to arithmetic and nothing has to be tuned at all. If it
does not, we learn what the knee is actually tracking, which is the next best
thing.

WHAT IS ALREADY KNOWN
---------------------
The threshold is a power law in the neuron's own membrane time constant:

    theta = 5.927 * tau^-0.748      log-log r = -0.974, 90% of the variance

Every other per-neuron property tested came back flat: excitatory fraction 0.04,
total incoming weight 0.03, mean delay -0.02, branch gain 0.08. Sweep 029 then
confirmed the formula end to end -- a column built with it reaches 0.757 against
0.763 for one that settled for fifty episodes, null at p = 0.7672.

Those numbers lived in a comment in `trajectory.py` and in no runnable file,
which is why this probe reproduces them rather than only extending them. A
number nothing re-derives is a number that quietly stops being true.

THE PREDICTION FOR THE KNEE (recorded before running)
--------------------------------------------------------
The knee is not per-neuron; it is per **branch**, one value for each of the
`n_branches` dendritic compartments of each neuron. And its adaptation rule says
what it is chasing outright:

    knee += knee_lr * knee * (engagement - plateau_engagement)

It rises when the branch's plateau engages too often and falls when it does not.
Engagement is the fraction of time the branch potential sits above the knee, so
**a settled knee is a fixed quantile of that branch's own potential
distribution** -- the (1 - 0.12) quantile, by construction.

That makes a sharp and falsifiable split:

  static structural properties      weak. Branch gain, total branch weight,
  (gain, weight, delays, sign mix)  excitatory fraction and mean delay all
                                    influence the potential, but only together
                                    and only through the input statistics.

  the branch potential's own scale  strong, and close to deterministic. If the
                                    knee is a quantile of `b`, then `log knee`
                                    against `log rms(b)` should be near-linear
                                    with r beyond -0.9 in magnitude.

**And the consequence matters more than the correlation.** The branch potential
scale is a property of the *running* column, not of its construction. So if the
prediction holds, the knee is *cheaper* than settling -- a short burst of input
gives you `rms(b)` and the knee follows in closed form -- but it is **not**
computable at construction the way theta is. That is a weaker result than the
theta one and should be reported as such rather than as "the operating point is
computable".

The outcome that would be better than predicted: a static property predicts the
knee well anyway, because the input statistics are homogeneous enough that
`rms(b)` is itself determined by the branch's weights. Then both halves are
construction-time and the whole operating point is arithmetic.

    python3 experiments/opfit.py --seeds 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plexus import ColumnConfig, DelayedXOR, Plexus  # noqa: E402


def logfit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Pearson r, exponent and coefficient of `y = a * x**b`, fitted in logs.

    Pairs with a non-positive value on either side are dropped rather than
    clipped: a clipped zero becomes a large negative log that dominates the fit
    and invents a correlation from the clip.
    """
    ok = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return float("nan"), float("nan"), float("nan")
    lx, ly = np.log(x[ok]), np.log(y[ok])
    r = float(np.corrcoef(lx, ly)[0, 1])
    b, loga = np.polyfit(lx, ly, 1)
    return r, float(b), float(np.exp(loga))


def observe(model, task, episodes: int, rng) -> np.ndarray:
    """RMS of each branch's potential, over a settled column left undisturbed."""
    col = model.column
    col.learning = False
    total = np.zeros_like(col.b, dtype=np.float64)
    n = 0
    for _ in range(episodes):
        ep = task.episode(rng)
        col.reset_state()
        model.transport.reset()
        for k in range(ep.inputs.shape[0]):
            t = model._t
            model._t += 1
            col.step(t, ep.inputs[k])
            total += col.b.astype(np.float64) ** 2
            n += 1
    return np.sqrt(total / max(n, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--neurons", type=int, default=96)
    ap.add_argument("--episodes", type=int, default=100, help="settling episodes")
    ap.add_argument("--observe", type=int, default=20)
    args = ap.parse_args()

    task = DelayedXOR()
    theta, knee = [], []
    per_neuron: dict[str, list] = {k: [] for k in
                                   ("tau", "weight", "exc_frac", "delay", "gain")}
    per_branch: dict[str, list] = {k: [] for k in
                                   ("rms_b", "gain", "weight", "exc_frac",
                                    "delay", "tau")}

    for s in range(args.seeds):
        model = Plexus(
            task.n_inputs, task.n_classes,
            column=ColumnConfig(n_neurons=args.neurons, lr=0.0, seed=s),
            seed=s,
        )
        model.train(task, args.episodes, rng=np.random.default_rng(1000 + s),
                    report_every=10**9)
        col = model.column
        rms_b = observe(model, task, args.observe, np.random.default_rng(21 + s))

        exc = (col.syn_sign > 0).astype(np.float64)
        wabs = np.abs(col.W).astype(np.float64)
        theta.append(col.theta.astype(np.float64))
        knee.append(col.knee.astype(np.float64).ravel())

        per_neuron["tau"].append(col.tau_soma.astype(np.float64))
        per_neuron["weight"].append(wabs.sum(axis=(1, 2)))
        per_neuron["exc_frac"].append(exc.mean(axis=(1, 2)))
        per_neuron["delay"].append(col.delay.astype(np.float64).mean(axis=(1, 2)))
        per_neuron["gain"].append(col.G.astype(np.float64).mean(axis=1))

        per_branch["rms_b"].append(rms_b.astype(np.float64).ravel())
        per_branch["gain"].append(col.G.astype(np.float64).ravel())
        per_branch["weight"].append(wabs.sum(axis=2).ravel())
        per_branch["exc_frac"].append(exc.mean(axis=2).ravel())
        per_branch["delay"].append(col.delay.astype(np.float64).mean(axis=2).ravel())
        # The owning neuron's tau, repeated across its branches -- so the thing
        # that predicts theta gets a fair chance at the knee too.
        per_branch["tau"].append(
            np.repeat(col.tau_soma.astype(np.float64), col.knee.shape[1])
        )

    theta = np.concatenate(theta)
    knee = np.concatenate(knee)

    print(f"{args.seeds} seeds, {args.neurons} neurons "
          f"-> {len(theta)} neurons, {len(knee)} branches\n")

    print("THETA, against per-neuron properties")
    print(f"  {'predictor':12s}{'r (log-log)':>14s}{'exponent':>12s}{'coef':>10s}")
    for name, vals in per_neuron.items():
        r, b, a = logfit(np.concatenate(vals), theta)
        print(f"  {name:12s}{r:14.3f}{b:12.3g}{a:10.3g}")

    print("\nKNEE, against per-branch properties")
    print(f"  {'predictor':12s}{'r (log-log)':>14s}{'exponent':>12s}{'coef':>10s}")
    for name, vals in per_branch.items():
        r, b, a = logfit(np.concatenate(vals), knee)
        print(f"  {name:12s}{r:14.3f}{b:12.3g}{a:10.3g}")

    # THE DIRECTION CONTROL, and the finding does not stand without it.
    #
    # The knee sets where the plateau engages, the plateau shapes the neuron's
    # output, and that output returns through recurrence as branch input. So a
    # correlation between the settled knee and the branch potential it was
    # measured against is circular by construction -- a large knee could be
    # *causing* a large potential rather than following it.
    #
    # `rms_b_frozen` is measured on a twin whose knee never moved: same seed,
    # same weights, `knee_lr = 0`, so its branch potentials cannot have been
    # shaped by knee adaptation. If that still predicts the *adapted* twin's
    # settled knee, the potential scale is upstream and the relation is real.
    frozen = []
    for s_i in range(args.seeds):
        twin = Plexus(
            task.n_inputs, task.n_classes,
            column=ColumnConfig(n_neurons=args.neurons, lr=0.0, seed=s_i,
                                knee_lr=0.0),
            seed=s_i,
        )
        twin.train(task, args.episodes, rng=np.random.default_rng(1000 + s_i),
                   report_every=10**9)
        frozen.append(
            observe(twin, task, args.observe,
                    np.random.default_rng(21 + s_i)).astype(np.float64).ravel()
        )
    #
    # Be clear about how strong this control is, because it is *nearly*
    # guaranteed by construction and a reader should not take it for more than
    # it is. `b` is the pre-nonlinearity potential -- a filtered weighted sum of
    # arriving input -- and the knee only enters downstream, in phi(b). So the
    # single path by which the knee could shape `b` is recurrence across
    # timesteps, and that is precisely what this measures. Measured: the knee
    # spans a 14x range across branches while `b` moves 1.3%.
    r_frozen, b_frozen, a_frozen = logfit(np.concatenate(frozen), knee)
    adapted_rms = np.concatenate(per_branch["rms_b"])
    frozen_rms = np.concatenate(frozen)
    drift = float(np.abs(adapted_rms - frozen_rms).max() / (frozen_rms.mean() + 1e-12))
    print(f"\n  {'rms_b (knee frozen)':22s}r = {r_frozen:.3f}   "
          f"exponent {b_frozen:.3f}   coef {a_frozen:.3f}")
    print(f"  knee spans {knee.min():.3f}..{knee.max():.3f} "
          f"({knee.max() / max(knee.min(), 1e-9):.0f}x) while freezing it moves "
          f"the branch potential by at most {drift:.1%}")

    # The prediction, stated as a number rather than left to the eye.
    r_rms, _, _ = logfit(np.concatenate(per_branch["rms_b"]), knee)
    r_static = max(
        abs(logfit(np.concatenate(per_branch[k]), knee)[0])
        for k in ("gain", "weight", "exc_frac", "delay", "tau")
    )
    print(f"\nbranch potential scale |r| = {abs(r_rms):.3f}, "
          f"best static property |r| = {r_static:.3f}")
    if abs(r_rms) > 0.9 and abs(r_rms) > 1.5 * r_static:
        if abs(r_frozen) > 0.8:
            print("=> the knee is a quantile of the branch's own potential, as "
                  "predicted, and a knee-frozen twin predicts it just as well "
                  "-- so the potential is upstream and this is not circular. "
                  "Cheaper than settling, NOT construction-time.")
        else:
            print(f"=> CIRCULAR. The correlation is {abs(r_rms):.3f} against the "
                  f"adapted column's own potentials but only {abs(r_frozen):.3f} "
                  "against a knee-frozen twin's, so the knee is shaping the "
                  "potential it appears to be following. Not a finding.")
    elif r_static > 0.9:
        print("=> a static property predicts the knee. Better than predicted: "
              "the whole operating point would be construction-time.")
    else:
        print("=> neither. The knee is fitted to something not tested here.")


if __name__ == "__main__":
    main()
