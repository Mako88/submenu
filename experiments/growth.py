"""Does a *trained* model survive having a new sense attached?

IN PLAIN TERMS
--------------
Suppose the network has learned to do something, and then you plug in a new
sensor -- another camera, another microphone, a channel that did not exist when
it was trained. Biological brains handle this. Most artificial ones do not: the
input layer is a fixed-size matrix, and widening it means rebuilding and
retraining.

This architecture is supposed to be different, because no weight matrix is
indexed by input -- every synapse names its source by number, so a new source is
just a new number. Adding one has already been shown to work on an *untrained*
column: it re-settles on its own and the new channels drive the output.

What nobody has measured is whether a model that has actually learned something
still knows it afterwards. There is a specific reason to doubt it: making room
for the new channels means **throwing away a quarter of the existing
connections**. If what the model learned lived in those connections, attaching a
sensor costs it.

QUESTION
--------
TODO item 5, the open half, quoted:

  "Still open, and the important part: **whether a *trained* model's accuracy
   survives the growth.** Everything above was measured on a settled random
   column, and homeostasis recovering is not the same as the learned function
   surviving."

`Column.add_inputs(n_new, rewire_frac)` moves `rewire_frac` of every branch's
synapses onto the new channels, because fan-in is fixed at `(N, B, S)` and
growth is rewiring rather than accretion. At the default that is **25% of every
neuron's fan-in reassigned**, and the weights on those synapses are re-drawn
from the initial distribution rather than inherited.

DECOMPOSING THE COST, WHICH IS WHY THERE ARE FOUR CONDITIONS
---------------------------------------------------------------
A drop after growth has two possible causes and they need different responses:

  losing synapses     25% of the learned fan-in is destroyed and re-drawn. This
                      is intrinsic to fixed fan-in and would be paid even if the
                      new sensor were perfect.

  gaining noise       the new channels carry input that is not task-relevant, so
                      they inject uninformative drive into the branches that now
                      read them.

`grow-silent` separates them without touching the library: the channels are
added and rewired to exactly as in `grow-noise`, but they are fed **zeros**, so
the new synapses contribute nothing and the only cost left is the destroyed
ones. The difference between the two conditions is the noise cost; the gap from
`nogrow` to `grow-silent` is the rewiring cost.

  nogrow        no growth, same episode budget      -- drift baseline
  grow-silent   +8 channels, always zero            -- rewiring cost alone
  grow-noise    +8 channels, carrying noise         -- rewiring + noise
  grow-noise-05 +8 channels, rewire_frac 0.05       -- does the cost track it?

`nogrow` is not decorative. The measurement after growth is taken later in
training than the one before it, and a frozen column drifts when anything about
its input changes (sweep 013 lost a whole run to exactly that). Without it, drift
and damage are the same number.

WHAT IS MEASURED
----------------
Linear decodability of the column state, before growth, immediately after, and
after a recovery period. Decodability rather than end-to-end accuracy because
the readout reads *neurons*, so growth does not touch it -- but sweep 009
established a readout can hide a column-level difference entirely, and it is the
column that is being damaged here.

    python3 experiments/growth.py --tag grow-noise --seed 0
    python3 experiments/growth.py --report
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

OUT = Path(__file__).resolve().parent / "growth_results.jsonl"
FIELDS = ["before", "after", "recovered", "cost", "residual"]


class Widened:
    """The same task, emitting `n_new` extra channels appended at the end.

    Appended, never inserted, so every existing channel keeps its index -- the
    same rule `add_inputs` obeys and for the same reason. `noise` decides
    whether the new channels carry anything: at zero they are a sensor that is
    attached and silent, which is what isolates the cost of the rewiring from
    the cost of the input.
    """

    def __init__(self, task, n_new: int, noise: float, rng: np.random.Generator):
        self.task = task
        self.n_new = n_new
        self.noise = noise
        self.rng = rng
        self.n_inputs = task.n_inputs + n_new
        self.n_classes = task.n_classes

    def episode(self, rng):
        ep = self.task.episode(rng)
        T = ep.inputs.shape[0]
        extra = np.zeros((T, self.n_new), dtype=np.float32)
        if self.noise > 0.0:
            mask = self.rng.random((T, self.n_new)) < self.noise
            extra[mask] = self.rng.uniform(
                0.4, 1.0, size=int(mask.sum())).astype(np.float32)
        ep.inputs = np.concatenate([ep.inputs, extra], axis=1)
        return ep


def decodability(model, task, episodes: int, seed: int) -> float:
    X, y = collect(model, task, episodes, np.random.default_rng(seed))
    split = int(0.7 * len(X))
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    return logistic_score(logistic((Xtr - mu) / sd, ytr), (Xte - mu) / sd, yte)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="grow-noise")
    ap.add_argument("--n-new", type=int, default=8)
    ap.add_argument("--rewire-frac", type=float, default=0.25)
    ap.add_argument("--noise", type=float, default=0.02,
                    help="event rate on the new channels; 0 attaches them silent")
    ap.add_argument("--grow", type=int, default=1)
    # The ceiling growth could aspire to. `grow-noise` changes the *task* as
    # well as the model -- eight extra distractor channels make delayed XOR
    # harder whether or not anything was damaged -- so its cost conflates the
    # two. A column trained from scratch on the widened task for the same total
    # budget has paid the task cost and none of the damage, which is what
    # separates them.
    ap.add_argument("--fresh", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--recover", type=int, default=100)
    ap.add_argument("--collect", type=int, default=500)
    ap.add_argument("--neurons", type=int, default=96)
    ap.add_argument("--hebbian", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.report:
        rows = [json.loads(x) for x in OUT.read_text().splitlines() if x.strip()]
        head = "".join(f"{f:>12s}" for f in FIELDS)
        print(f"{'condition':16s}{head}   seeds")
        for tag in sorted({r["tag"] for r in rows}):
            sel = [r for r in rows if r["tag"] == tag]
            cells = "".join(f"{np.mean([r[f] for r in sel]):12.4f}" for f in FIELDS)
            print(f"{tag:16s}{cells}   {len(sel)}")
        return

    base = DelayedXOR()
    if args.fresh:
        # No growth at all: a column built at the widened width and trained on
        # the widened task for the whole budget. `before` is undefined here and
        # is reported as the same number as `recovered` so `residual` reads 0 --
        # this condition is a reference level, not a before/after comparison.
        task = Widened(base, args.n_new, args.noise,
                       np.random.default_rng(6000 + args.seed))
        model = Plexus(
            task.n_inputs, task.n_classes,
            column=ColumnConfig(n_neurons=args.neurons, lr=0.0, seed=args.seed,
                                hebbian=bool(args.hebbian)),
            seed=args.seed,
        )
        rng = np.random.default_rng(1000 + args.seed)
        model.train(task, args.episodes + args.recover, rng=rng, report_every=10**9)
        level = decodability(model, task, args.collect, 13 + args.seed)
        row = dict(tag=args.tag, seed=args.seed, n_new=args.n_new,
                   rewire_frac=0.0, noise=args.noise, episodes=args.episodes,
                   recover=args.recover, collect=args.collect,
                   neurons=args.neurons, before=level, after=level,
                   recovered=level, cost=0.0, residual=0.0)
        with OUT.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"{args.tag} seed={args.seed}: level {level:.3f} "
              "(from-scratch reference, no growth)")
        return

    model = Plexus(
        base.n_inputs, base.n_classes,
        column=ColumnConfig(n_neurons=args.neurons, lr=0.0, seed=args.seed,
                            hebbian=bool(args.hebbian)),
        seed=args.seed,
    )
    rng = np.random.default_rng(1000 + args.seed)
    model.train(base, args.episodes, rng=rng, report_every=10**9)
    before = decodability(model, base, args.collect, 11 + args.seed)

    if args.grow:
        col = model.column
        grow_rng = np.random.default_rng(5000 + args.seed)
        # `add_inputs` widens the transport itself (`transport.grow` at the end
        # of its body), so nothing is done here first. An earlier version of
        # this file called `transport.grow` beforehand and carried a comment
        # explaining why that was necessary. It was not, and the comment was a
        # claim written without checking -- removing the call changed nothing.
        col.add_inputs(args.n_new, rewire_frac=args.rewire_frac, rng=grow_rng)
        task = Widened(base, args.n_new, args.noise,
                       np.random.default_rng(6000 + args.seed))
        model._zero_input = np.zeros(task.n_inputs, dtype=np.float32)

        # Assert the graft landed, in the run rather than in a unit test.
        #
        # If `transport.grow` had not widened the buffer, or the rewiring had
        # not taken, the appended channels would read as silence -- and this
        # experiment would report that growth costs nothing, which is both the
        # most reassuring outcome and a measurement of nothing at all. The unit
        # tests cover `Column.add_inputs`; what is new here is the transport
        # widening and the task emitting the extra columns, so those are what
        # get checked.
        new_ids = np.arange(model.transport.n_sources - args.n_new,
                            model.transport.n_sources)
        assert np.isin(col.src, new_ids).any(), (
            "no synapse points at an appended source: the rewiring did not take"
        )

        def _drive(active: bool) -> np.ndarray:
            col.reset_state()
            model.transport.reset()
            probe = np.random.default_rng(99)
            total = np.zeros(col.cfg.n_neurons, dtype=np.float64)
            for t in range(400):
                x = np.zeros(task.n_inputs, dtype=np.float32)
                if active:
                    x[base.n_inputs:] = (
                        probe.random(args.n_new) < 0.3).astype(np.float32)
                total += col.step(10**6 + t, x)
            return total

        assert not np.array_equal(_drive(True), _drive(False)), (
            "driving the appended channels changed nothing -- they are not "
            "reaching the column, so any cost measured here is not about growth"
        )
        col.reset_state()
        model.transport.reset()
    else:
        task = base

    after = decodability(model, task, args.collect, 12 + args.seed)
    model.train(task, args.recover, rng=rng, report_every=10**9)
    recovered = decodability(model, task, args.collect, 13 + args.seed)

    row = dict(tag=args.tag, seed=args.seed, n_new=args.n_new if args.grow else 0,
               rewire_frac=args.rewire_frac if args.grow else 0.0,
               noise=args.noise, episodes=args.episodes, recover=args.recover,
               collect=args.collect, neurons=args.neurons,
               before=before, after=after, recovered=recovered,
               cost=after - before, residual=recovered - before)
    with OUT.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"{args.tag} seed={args.seed}: before {before:.3f}  after {after:.3f}  "
          f"recovered {recovered:.3f}  cost {after - before:+.3f}  "
          f"residual {recovered - before:+.3f}")


if __name__ == "__main__":
    main()
