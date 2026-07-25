"""Does salience-gated Hebbian binding improve the representation?

Measures linear decodability of the column state, which is the cleanest read on
what a rule changes: it does not depend on the online readout's sample
efficiency at all, and sweep 009 established that the readout can hide a
column-level difference entirely.

This replaces experiments/engram.py, which measured an engram allocator --
excitability drift, a recruitment competition, an allocation refractory. That
mechanism was built, measured and deleted: sweep 014 put the full apparatus at
0.830 against 0.876 for binding alone, worse on 19 of 20 seeds at p = 0.0001.
The allocation diagnostics went with it, since there is no longer an allocation
to diagnose. They are recoverable from git history and from
experiments/sweeps/engram-012 and -014 if a benchmark ever asks for memory
separation, which delayed XOR does not.

    python3 experiments/binding.py --tag on  --hebbian 1 --seed 0
    python3 experiments/binding.py --tag off --hebbian 0 --seed 0
    python3 experiments/binding.py --report
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

OUT = Path(__file__).resolve().parent / "binding_results.jsonl"
FIELDS = ["linear", "mlp", "corr", "eff_rank", "sparsity"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="on")
    ap.add_argument("--hebbian", type=int, default=1)
    ap.add_argument("--lateral", type=int, default=0)
    ap.add_argument("--lateral-lr", type=float, default=ColumnConfig.lateral_lr)
    ap.add_argument("--hebb-lr", type=float, default=ColumnConfig.hebb_lr)
    ap.add_argument("--bind-scale", type=float, default=ColumnConfig.bind_scale)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--collect", type=int, default=500)
    ap.add_argument("--neurons", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    # Sweep 026. Binding reads the modulator only for its *presence*, and scores
    # `act_fast` -- a 50ms window on the neuron's own activity -- at the moment it
    # arrives. A late modulator therefore scores a window that has already
    # decayed, which makes this mechanism latency-sensitive by construction and
    # the right vehicle for the architecture's headline claim.
    ap.add_argument("--modulator-lag", type=int, default=0)
    # The window binding scores when the modulator lands. If latency tolerance
    # is set by this constant rather than by the architecture, raising it should
    # buy tolerance directly -- which is the condition that makes sweep 026
    # actionable rather than merely descriptive.
    ap.add_argument("--tau-act-fast", type=float, default=ColumnConfig.tau_act_fast)
    # And this is the constant that actually sets it. `_bind` commits a product
    # of two traces -- `act_fast` at `tau_act_fast` and `pre` at `tau_branch` --
    # so the window is `1/(1/50 + 1/15) = 11.5` steps and is dominated by the
    # shorter one. Measured directly by `experiments/lagwindow.py`: 11.87,
    # against 15.25 when `tau_act_fast` is raised five-fold and 28.68 when
    # `tau_branch` is raised four-fold.
    #
    # Exposed here because it is the *forward path's* filter, not a plasticity
    # setting, so raising it to buy latency tolerance changes what the column
    # computes. Whether the column still decodes anything is the question, and
    # it cannot be asked without this flag.
    ap.add_argument("--tau-branch", type=float, default=ColumnConfig.tau_branch)
    # The other route to a wider window: a presynaptic trace the *rule* owns,
    # so the forward path is untouched. `lagwindow.py` measures 154 steps at
    # `tau_act_fast = bind_tau_pre = 300` against a product prediction of 150,
    # with 40% of the update surviving lag 150. Raise `tau_act_fast` alongside
    # it -- the window is a product, so 300 on one side alone is still capped
    # near 50 by the other.
    ap.add_argument("--bind-tau-pre", type=float, default=None)
    # Sweep 035. `test_delivery_jitter_does_not_change_a_distributed_run`
    # established that delivery lateness below `delay_min` leaves a distributed
    # run bit-identical and lateness at or above it does not -- so `delay_min`
    # IS the jitter budget, exactly, and raising it buys tolerance directly.
    # What that costs the column has never been measured. `delay_max` is exposed
    # alongside it because raising the floor alone also narrows the *spread* of
    # delays, and the README claims the spread is what enriches the temporal
    # basis; without both flags the two cannot be told apart.
    # Sweep 036. DESIGN.md says of the log-uniform 12-320 ms spread: "The
    # spread alone is a large win: a population with mixed constants holds
    # working memory a homogeneous one cannot." Load-bearing, strongly worded,
    # and never measured -- `test_membrane_gain_is_independent_of_time_constant`
    # guards the DC-gain bug that spread exposed, not the benefit it claims.
    # Setting min == max makes the population homogeneous, which is the
    # comparison the claim needs and has never had.
    ap.add_argument("--tau-soma-min", type=float, default=ColumnConfig.tau_soma_min)
    ap.add_argument("--tau-soma-max", type=float, default=ColumnConfig.tau_soma_max)
    ap.add_argument("--delay-min", type=int, default=ColumnConfig.delay_min)
    ap.add_argument("--delay-max", type=int, default=ColumnConfig.delay_max)
    # Pinned, never left to track the lag. Sweep 013 lost a whole run to that:
    # the silent drain tail scaled with the lag, so a *frozen* column shifted
    # -0.062 (p = 0.0011) between lag settings and the comparison measured tail
    # length rather than latency.
    #
    # **Default 0, not 200.** Sweep 026 shipped this at 200 and every condition
    # in the file picked it up silently, because `off`, `on`, `lateral` and
    # `lateral+on` are invoked without the flag. The baseline moved from the
    # 0.802 that sweeps 015-025 all measured to 0.620 -- 200 extra silent steps
    # per episode shift the homeostatic operating point, and sweeps 022-023
    # established that operating point carries most of the representation
    # quality. A parameter added for one condition changed every condition.
    #
    # Same failure as sweep 003 (`PRETRAIN` default made it test nothing) and
    # sweep 013 (the drain tail itself). Third instance: **a new flag's default
    # is a silent change to every existing condition, and must reproduce what
    # they ran before.** Conditions that want the tail pass it explicitly.
    ap.add_argument("--drain-steps", type=int, default=0)
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
            hebbian=bool(args.hebbian),
            lateral=bool(args.lateral),
            lateral_lr=args.lateral_lr,
            hebb_lr=args.hebb_lr,
            bind_scale=args.bind_scale,
            tau_act_fast=args.tau_act_fast,
            tau_branch=args.tau_branch,
            bind_tau_pre=args.bind_tau_pre,
            delay_min=args.delay_min,
            delay_max=args.delay_max,
            tau_soma_min=args.tau_soma_min,
            tau_soma_max=args.tau_soma_max,
        ),
        modulator_lag=args.modulator_lag,
        drain_steps=args.drain_steps,
        seed=args.seed,
    )
    rng = np.random.default_rng(1000 + args.seed)
    model.train(task, args.episodes, rng=rng, report_every=10**9)

    X, y = collect(model, task, args.collect, np.random.default_rng(11 + args.seed))
    split = int(0.7 * len(X))
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    # What shape the representation has, not just how well it reads. Sweep 020
    # exists because these two disagree with the reason lateral inhibition was
    # proposed: the mechanism that works makes the state *more* correlated and
    # *lower* dimensional, not less.
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    C = (Xz.T @ Xz) / len(Xz)
    offdiag = np.abs(C[np.triu_indices(C.shape[0], 1)])
    ev = np.clip(np.linalg.eigvalsh(C)[::-1], 0.0, None)
    # Participation ratio: how many directions the variance actually occupies.
    eff_rank = float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-12))

    row = dict(
        tag=args.tag, seed=args.seed, episodes=args.episodes, neurons=args.neurons,
        modulator_lag=args.modulator_lag, drain_steps=args.drain_steps,
        tau_act_fast=args.tau_act_fast, tau_branch=args.tau_branch,
        bind_tau_pre=args.bind_tau_pre, collect=args.collect,
        delay_min=args.delay_min, delay_max=args.delay_max,
        tau_soma_min=args.tau_soma_min, tau_soma_max=args.tau_soma_max,
        linear=logistic_score(logistic(Xtr, ytr), Xte, yte),
        mlp=mlp(Xtr, ytr, Xte, yte),
        corr=float(offdiag.mean()),
        eff_rank=eff_rank,
        sparsity=model.column.sparsity,
    )
    with OUT.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"{args.tag} seed={args.seed}: " + "  ".join(f"{f} {row[f]:.3f}" for f in FIELDS))


if __name__ == "__main__":
    main()
