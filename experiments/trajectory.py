"""When during training does binding's representation gain actually arrive?

Sweep 018 left one question standing. Binding needs roughly its full 300
episodes -- halving the budget to fund a static period for the readout gives
back nearly all the gain -- and that is odd, because the baseline binding
scores against converges early, so late events should contribute progressively
less. Something is accumulating late that no diagnostic currently measures.

This measures it directly: train as normal, and every `--every` episodes pause
to fit a fresh linear decoder on held-out probe episodes. The result is a curve
rather than an endpoint, which is the only thing that can distinguish "binding
is slow" from "the readout is slow".

The probe must not disturb what it measures. Probe episodes run with
`learn=False`, which stops binding, weight updates, homeostasis and the
readout's own statistics; they draw from a separate RNG so the training stream
is untouched; and `test_probing_does_not_perturb_training` asserts that
training with probes interleaved leaves W, theta and the knee bit-identical to
training without. A measurement that changes its subject is worse than none,
and this project has already shipped one of those.

    python3 experiments/trajectory.py --tag on  --hebbian 1 --seed 0
    python3 experiments/trajectory.py --report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plexus import ColumnConfig, DelayedXOR, Plexus  # noqa: E402
from probe import collect, logistic, logistic_score  # noqa: E402

OUT = Path(__file__).resolve().parent / "trajectory_results.jsonl"

# Fitted once, over 6 seeds x 96 neurons of settled columns, and then frozen.
# Refitting per run would let the initialiser peek at the answer it is meant to
# predict, which is the whole point of the condition.
THETA_A, THETA_B = 5.927, -0.748
# Sweep 029 left the knee as the open half of the operating point, and
# `experiments/opfit.py` measured what it is fitted to. The knee's own
# adaptation rule raises it when its branch's plateau engages too often, so a
# settled knee is a fixed quantile of that branch's potential distribution --
# and it is, at log-log r = 0.923 over 6 seeds and 4608 branches, with an
# exponent of 1.05 (i.e. very nearly plain proportionality). Every static
# structural property was flat: excitatory fraction 0.20, branch gain 0.002,
# mean delay 0.002, the owning neuron's tau 0.005.
#
# The direction control is what makes it a finding rather than a circularity: a
# knee-frozen twin's potentials predict the adapted twin's settled knee equally
# well (r = 0.923), and the knee spans a 16x range across branches while
# freezing it moves the branch potential by at most 1.2%. So the potential is
# upstream.
#
# **This is weaker than the theta result and must be reported as such.** The
# branch potential scale is a property of the running column, so the knee is
# obtainable from a short observation pass rather than from a hundred episodes
# of closed-loop adaptation -- cheaper than settling, but NOT construction-time
# the way theta is.
KNEE_A, KNEE_B = 1.83, 1.05


class ShuffledInput:
    """The task's marginal input statistics with its timing destroyed.

    Sweep 023. Used only for the *twin* that finds an operating point, never for
    the column being measured. Each channel's time series is permuted
    independently, so per-channel event count and amplitude distribution are
    preserved exactly while cue timing, the cue-to-go relationship and
    cross-channel synchrony are gone.

    Exact preservation is the point. A Bernoulli stream at a matched rate would
    confound "homeostasis does not need timing" with "the rate was not matched
    closely enough"; shuffling makes the marginals identical by construction, so
    a difference can only be timing.
    """

    def __init__(self, task):
        self.task = task
        self.n_inputs = task.n_inputs
        self.n_classes = task.n_classes

    def episode(self, rng):
        ep = self.task.episode(rng)
        x = ep.inputs.copy()
        for c in range(x.shape[1]):
            rng.shuffle(x[:, c])
        # The label is meaningless once timing is gone, and it is never read:
        # the twin runs at lr=0 with no binding, so nothing it does depends on
        # being right. Kept only so the training loop has the shape it expects.
        return type(ep)(inputs=x, label=ep.label, response=ep.response, bits=None)


class NoiseInput:
    """Bernoulli events at the task's overall rate, and nothing else.

    The weakest input that still has the right amount of activity. If an
    operating point found on this matches one found on the task, homeostasis
    needs only "how much drive arrives", which is a number that could be
    supplied rather than discovered.
    """

    def __init__(self, task, rng):
        self.task = task
        self.n_inputs = task.n_inputs
        self.n_classes = task.n_classes
        probe = task.episode(rng)
        self.shape = probe.inputs.shape
        self.rate = float((probe.inputs > 0).mean())
        nz = probe.inputs[probe.inputs > 0]
        self.lo, self.hi = float(nz.min()), float(nz.max())
        self.response = probe.response
        self._episode_cls = type(probe)

    def episode(self, rng):
        x = np.zeros(self.shape, dtype=np.float32)
        mask = rng.random(self.shape) < self.rate
        x[mask] = rng.uniform(self.lo, self.hi, size=int(mask.sum())).astype(np.float32)
        return self._episode_cls(inputs=x, label=0, response=self.response, bits=None)


def decodability(model, task, episodes: int, seed: int) -> tuple[float, float]:
    """Linear decodability of the column state, and the firing rate it ran at.

    Sparsity is measured *here*, during the probe, rather than read off
    `column.sparsity`. That property returns the rate EMA, which only advances
    while `learning` is True -- so at checkpoint 0, before any training episode,
    it reports `target_rate` unchanged from initialisation. Reading it there
    gives 0.0300 for every condition including ones whose column is silent: a
    number that looks like a measurement, is not, and would hide exactly the
    dead-column case the sparsity column exists to catch.
    """
    fired = []
    hook = model.column.step

    def counting_step(t, external=None):
        out = hook(t, external)
        fired.append(float((out > 0).mean()))
        return out

    model.column.step = counting_step
    try:
        X, y = collect(model, task, episodes, np.random.default_rng(seed))
    finally:
        model.column.step = hook
    split = int(0.7 * len(X))
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    score = logistic_score(logistic((Xtr - mu) / sd, ytr), (Xte - mu) / sd, yte)
    return score, float(np.mean(fired))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="on")
    ap.add_argument("--hebbian", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--probe", type=int, default=300)
    ap.add_argument("--neurons", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    # Sweep 022. The frozen column climbs 0.582 -> 0.779 over its first fifty
    # episodes with no learning rule running at all, which is more than twice
    # what binding adds and has never been explained. Exactly three things adapt
    # in that condition, so each gets an off switch and the question becomes an
    # ablation rather than a hypothesis. `None` means "leave the default".
    ap.add_argument("--homeostatic-lr", type=float, default=None,
                    help="somatic threshold adaptation rate; 0 disables it")
    ap.add_argument("--knee-lr", type=float, default=None,
                    help="dendritic knee adaptation rate; 0 disables it")
    ap.add_argument("--scaling-lr", type=float, default=None,
                    help="synaptic scaling rate; 0 disables it")
    ap.add_argument("--lateral", type=int, default=0)
    # Disabling threshold homeostasis outright does not answer the question it
    # looks like it answers. Measured at one seed: sparsity collapses from 0.029
    # to 0.004 and decodability sits at its episode-0 value, so the column is
    # silent rather than unhelped, and "threshold adaptation contributes
    # nothing" read off that is a conclusion about a dead column.
    #
    # This separates the two things the mechanism does. Settle a twin column --
    # same seed, so the same weights and the same wiring -- for N episodes with
    # everything adapting, copy its threshold and knee onto this one, and only
    # then switch adaptation off. The operating point is preserved and what is
    # removed is the *ongoing* adaptation, which is the actual question.
    ap.add_argument("--preset-from", type=int, default=0,
                    help="settle a twin column for N episodes and adopt its "
                         "threshold and knee before measuring")
    # Sweep 023. What the twin must see to find the operating point: the task,
    # the task's marginals with timing destroyed, or matched-rate noise. The
    # column being measured always runs the real task.
    ap.add_argument("--preset-input", default="task",
                    choices=["task", "shuffled", "noise"])
    # Sweep 027. Sweeps 022-023 established that the settling gain is carried by
    # theta and knee, and eliminated two explanations for what they encode -- not
    # criticality (025), not rate calibration (023, where the condition sitting
    # exactly on target decodes worst of the three).
    #
    # These split the remaining possibilities. `--preset-what` asks which of the
    # two vectors carries it. `--preset-shuffle` permutes the values across
    # neurons, which preserves their distribution *exactly* while destroying
    # which neuron got which -- so it separates "the spread of thresholds
    # matters" from "each neuron needs its own".
    ap.add_argument("--preset-what", default="both",
                    choices=["both", "theta", "knee"])
    ap.add_argument("--preset-shuffle", type=int, default=0,
                    help="permute preset values across neurons: same "
                         "distribution, wrong owner")
    # Sweep 029. A probe after 027 found what theta is fitted to: the neuron's
    # own membrane time constant, and almost nothing else. Across 6 seeds and 576
    # neurons, log theta against log tau gives r = -0.974, and the power law
    # theta = 5.93 * tau^-0.748 explains 90% of the variance at a median error of
    # 10.6%. Every other local property was flat -- excitatory fraction 0.04,
    # total weight 0.03, mean delay -0.02, branch gain 0.08.
    #
    # That is computable at construction, from a quantity each neuron already
    # knows about itself, and it preserves the per-neuron assignment that sweep
    # 027 showed is what matters (permuting cost 0.130 at p = 0.0000). So it is
    # the version of "precomputable" that 027 did not refute.
    ap.add_argument("--theta-from-tau", type=int, default=0,
                    help="initialise theta as a power law in the neuron's own "
                         "membrane tau instead of settling for it")
    # Sweep 032. The knee half, from `opfit.py`: knee = 1.83 * rms(b)^1.05,
    # r = 0.923. Takes a number of observation episodes rather than a flag,
    # because the whole point is that it is cheap -- and how cheap is the
    # question. Default 0, so every existing condition runs exactly as before
    # (the sweep 026 lesson about a new flag's default).
    ap.add_argument("--knee-from-rms", type=int, default=0,
                    help="set each branch's knee from the RMS of its own "
                         "potential, measured over this many observation "
                         "episodes, instead of settling for it")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.report:
        rows = [json.loads(x) for x in OUT.read_text().splitlines() if x.strip()]
        tags = sorted({r["tag"] for r in rows})
        checkpoints = sorted({c for r in rows for c in map(int, r["curve"])})

        def table(field: str, fmt: str) -> None:
            print(f"{'episode':>8s}" + "".join(f"{t:>16s}" for t in tags))
            for c in checkpoints:
                cells = ""
                for t in tags:
                    v = [r[field][str(c)] for r in rows
                         if r["tag"] == t and str(c) in r.get(field, {})]
                    cells += format(float(np.mean(v)) if v else float("nan"), fmt)
                print(f"{c:8d}{cells}")

        print("linear decodability")
        table("curve", ">16.3f")
        # Always printed next to it, never on request: an ablation that silences
        # the column would otherwise read as a clean null on the table above.
        if any("sparsity" in r for r in rows):
            print("\nsparsity (firing fraction) -- a column at ~0 or ~1 is dead,")
            print("and its decodability says nothing about the mechanism")
            table("sparsity", ">16.4f")
        n = len({r["seed"] for r in rows})
        print(f"\n{n} seeds")
        return

    task = DelayedXOR()
    overrides = {
        name: value
        for name, value in (("homeostatic_lr", args.homeostatic_lr),
                            ("knee_lr", args.knee_lr),
                            ("scaling_lr", args.scaling_lr))
        if value is not None
    }
    def build(**extra):
        return Plexus(
            task.n_inputs, task.n_classes,
            column=ColumnConfig(
                n_neurons=args.neurons, lr=0.0, seed=args.seed,
                hebbian=bool(args.hebbian), lateral=bool(args.lateral),
                **{**overrides, **extra}
            ),
            seed=args.seed,
        )

    model = build()
    if args.theta_from_tau:
        col = model.column
        col.theta[:] = (THETA_A * col.tau_soma ** THETA_B).astype(np.float32)
    if args.knee_from_rms:
        # Observe, then set -- no closed loop, no twin, no adaptation. The pass
        # runs with `learning` off so nothing adapts while it is watched, and it
        # happens *after* theta-from-tau so the potentials being measured belong
        # to the column that will actually be used.
        col = model.column
        was = col.learning
        col.learning = False
        obs_rng = np.random.default_rng(6000 + args.seed)
        total = np.zeros_like(col.b, dtype=np.float64)
        n = 0
        for _ in range(args.knee_from_rms):
            ep = task.episode(obs_rng)
            col.reset_state()
            model.transport.reset()
            for k in range(ep.inputs.shape[0]):
                t = model._t
                model._t += 1
                col.step(t, ep.inputs[k])
                total += col.b.astype(np.float64) ** 2
                n += 1
        rms_b = np.sqrt(total / max(n, 1))
        col.knee[:] = np.clip(KNEE_A * rms_b ** KNEE_B, 1e-3, 50.0).astype(np.float32)
        col.reset_state()
        model.transport.reset()
        col.learning = was
    if args.preset_from:
        # The twin adapts with every default in force -- the overrides apply
        # only to the column being measured, or the preset would inherit the
        # very ablation it exists to compensate for.
        twin = build(homeostatic_lr=ColumnConfig.homeostatic_lr,
                     knee_lr=ColumnConfig.knee_lr,
                     scaling_lr=ColumnConfig.scaling_lr)
        twin_rng = np.random.default_rng(7000 + args.seed)
        twin_task = {
            "task": lambda: task,
            "shuffled": lambda: ShuffledInput(task),
            "noise": lambda: NoiseInput(task, np.random.default_rng(500 + args.seed)),
        }[args.preset_input]()
        twin.train(twin_task, args.preset_from, rng=twin_rng, report_every=10**9)
        # Assert the preset landed, in the run rather than in a unit test. The
        # whole condition's meaning depends on it: a preset that silently failed
        # would reproduce the `none` condition -- a dead column at its starting
        # value -- and the sweep would report "ongoing adaptation contributes
        # nothing" from a column that never had the operating point at all.
        #
        # The bar catches total failure, not insufficient settling: one episode
        # already moves theta by 7%, five by 45%, thirty by 63%. Anything under
        # 1% means the twin did not train.
        moved = float(np.abs(twin.column.theta / model.column.theta - 1.0).mean())
        if moved < 0.01:
            raise SystemExit(
                f"--preset-from {args.preset_from} left theta within {moved:.4f} "
                "of its initial value, so there is nothing to preset. The twin "
                "did not settle; the condition would be meaningless."
            )
        theta, knee = twin.column.theta.copy(), twin.column.knee.copy()
        if args.preset_shuffle:
            # One permutation applied to both, so a neuron receives a matched
            # (theta, knee) pair from some *other* neuron rather than two
            # unrelated ones. Mismatching them as well would confound "wrong
            # owner" with "internally inconsistent", and only the first is the
            # question.
            perm = np.random.default_rng(9000 + args.seed).permutation(len(theta))
            theta, knee = theta[perm], knee[perm]
        if args.preset_what in ("both", "theta"):
            model.column.theta[:] = theta
        if args.preset_what in ("both", "knee"):
            model.column.knee[:] = knee
    rng = np.random.default_rng(1000 + args.seed)

    curve, sparsity = {}, {}
    done = 0
    # Checkpoint at 0 as well: it is the frozen column, and both conditions must
    # agree there or the probe is measuring something other than binding.
    curve[str(done)], sparsity[str(done)] = decodability(
        model, task, args.probe, 11 + args.seed)
    # Recorded alongside, because the sweep 022 ablations can kill the column
    # rather than merely fail to help it. A silent or saturated column decodes
    # at chance for a reason that has nothing to do with the mechanism under
    # test, and "homeostasis contributes nothing" read off a dead column is a
    # conclusion drawn from a disconnected quantity.
    while done < args.episodes:
        step = min(args.every, args.episodes - done)
        model.train(task, step, rng=rng, report_every=10**9)
        done += step
        curve[str(done)], sparsity[str(done)] = decodability(
            model, task, args.probe, 11 + args.seed)

    # Flat copies of the last checkpoint, so `paired_test.py` can read them.
    # Without these the sweep can only report means across seeds, which is not
    # what this project counts as a result -- sweep 023 was written up with a
    # +0.051 difference and no p-value because of exactly that omission, and
    # the effect it inverted had looked the other way at one seed.
    row = dict(tag=args.tag, seed=args.seed, hebbian=args.hebbian,
               episodes=args.episodes, every=args.every, curve=curve,
               sparsity=sparsity, preset_from=args.preset_from,
               preset_input=args.preset_input, preset_what=args.preset_what,
               theta_from_tau=args.theta_from_tau,
               knee_from_rms=args.knee_from_rms,
               preset_shuffle=args.preset_shuffle,
               final=curve[str(done)], final_sparsity=sparsity[str(done)],
               **overrides)
    with OUT.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    pts = " ".join(f"{k}:{v:.3f}" for k, v in sorted(curve.items(), key=lambda kv: int(kv[0])))
    print(f"{args.tag} seed={args.seed}  {pts}")


if __name__ == "__main__":
    main()
