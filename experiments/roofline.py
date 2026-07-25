"""Is this model actually memory-bandwidth-bound? And what would a GPU buy?

IN PLAIN TERMS
--------------
The project's case for running on many ordinary machines instead of one big
accelerator rests partly on an assumption: that the work is limited by how fast
memory can be read, not by raw arithmetic. If that is true, a graphics card --
which mostly buys arithmetic and some memory speed -- wins by less than people
expect, and spreading over cheap machines looks reasonable. If it is false, the
counter-argument is stronger than the repo admits.

TODO item 8 states the assumption outright: "largely memory-bandwidth-bound on
the `(N,B,S)` tensors". Nothing has ever measured it. This does.

It also refuses to guess the other half. There is no GPU in this environment, so
no GPU number here is measured, and none is presented as though it were. What
can be done honestly is to establish which resource the model is actually
running out of, because that determines what a GPU could possibly help with.

WHAT IT MEASURES
----------------
  sustained bandwidth   A streaming kernel on arrays far larger than the last
                        level of cache, so it reaches DRAM. This machine's real
                        ceiling, not a spec sheet number.

  the working set       Every `(N, B, S)` array the step touches, in bytes,
                        against the measured cache sizes. An array that fits in
                        L2 is not costing DRAM bandwidth whatever the docstring
                        says.

  achieved traffic      Bytes the step must move divided by how long it takes.
                        Compared against sustained bandwidth, this says what
                        fraction of the memory system is actually in use.

  arithmetic intensity  FLOPs per byte. Low means bandwidth is the limit *if*
                        the data is in DRAM; it says nothing if the data is in
                        cache.

THE PREDICTION (recorded before running)
------------------------------------------
The model is **not** DRAM-bandwidth-bound at any size this project has ever run,
and the TODO claim is wrong as stated.

The reasoning is arithmetic and can be checked before any timing. One
`(N,B,S)` float32 array at the standard 96 neurons, 8 branches and 16 synapses
is `96*8*16*4` bytes = **48 KB**. The step touches roughly six of them (`W`,
`pre`, `eps`, `elig`, the gathered input, and a temporary), so the working set
is around **300 KB** against an L2 of 8 MiB. Even at 1024 neurons it is ~3 MB,
still inside L2, and this machine has a 260 MiB L3 behind that.

So the expected finding is that achieved traffic is a **small fraction** of
sustained DRAM bandwidth, and that what actually limits the step is per-call
overhead in numpy -- which is exactly what TODO item 4 already observed from the
other side, that throughput *rises* with column size as the per-step Python
overhead amortises over more work. A genuinely bandwidth-bound program does the
opposite.

If instead achieved traffic comes close to sustained bandwidth, the prediction
is wrong, the TODO claim is right, and the GPU counter-argument is stronger.

WHAT THIS CANNOT SAY
--------------------
Anything measured about a GPU. There is no GPU here. The most that can be
concluded is which resource is scarce, and therefore which resource a GPU would
have to be better at for the counter-argument to hold.

    python3 experiments/roofline.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plexus import ColumnConfig, DelayedXOR, Plexus  # noqa: E402


def sustained_bandwidth(mb: int = 512, repeats: int = 5) -> float:
    """Bytes/second this machine sustains on a streaming triad, out of cache.

    Arrays are sized well past the last level of cache so the number is DRAM
    and not a cache artefact. Best-of-N rather than mean: the interest is in
    what the machine *can* do, and a slow run only ever means interference.
    """
    n = (mb * 1024 * 1024) // 4
    a = np.ones(n, dtype=np.float32)
    b = np.ones(n, dtype=np.float32)
    c = np.empty(n, dtype=np.float32)
    best = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        np.add(a, b, out=c)
        dt = time.perf_counter() - t0
        # Two reads and one write of an n-element float32 array.
        best = max(best, (3 * n * 4) / dt)
    return best


def step_cost(neurons: int, steps: int = 400) -> tuple[float, int]:
    """Seconds per column step, and the bytes one step must move."""
    task = DelayedXOR()
    model = Plexus(
        task.n_inputs, task.n_classes,
        column=ColumnConfig(n_neurons=neurons, lr=0.004, seed=0, hebbian=True),
        seed=0,
    )
    col = model.column
    ep = task.episode(np.random.default_rng(0))
    col.reset_state()
    model.transport.reset()

    for k in range(50):  # warm up caches and any lazy allocation
        col.step(k, ep.inputs[k % ep.inputs.shape[0]])

    t0 = time.perf_counter()
    for k in range(steps):
        col.step(1000 + k, ep.inputs[k % ep.inputs.shape[0]])
    dt = (time.perf_counter() - t0) / steps

    # Every (N,B,S)-shaped array the step reads or writes. Counted from the
    # arrays that actually exist rather than from the source, so a new trace
    # cannot be silently left out of the accounting.
    per_array = neurons * col.cfg.n_branches * col.cfg.n_synapses * 4
    tensors = [n for n, v in vars(col).items()
               if isinstance(v, np.ndarray) and v.shape == col.W.shape]
    return dt, per_array * len(tensors), tensors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[96, 256, 512, 1024])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--stream-mb", type=int, default=512)
    args = ap.parse_args()

    bw = sustained_bandwidth(args.stream_mb)
    print(f"sustained streaming bandwidth: {bw / 1e9:.2f} GB/s "
          f"(triad on {args.stream_mb} MB arrays)\n")

    print(f"{'neurons':>8s}{'us/step':>10s}{'working set':>14s}"
          f"{'traffic':>12s}{'% of peak':>11s}{'steps/s':>10s}{'realtime N':>12s}")
    for n in args.sizes:
        dt, nbytes, tensors = step_cost(n, args.steps)
        traffic = nbytes / dt
        # Neurons this core could carry at biological real time. One step is
        # 1 ms, so a column of N neurons runs at real time when it sustains
        # 1000 steps/s; `realtime N` is N scaled by how far off that it is.
        # NOT steps/s x neurons, which is a different quantity entirely and was
        # once reported as this one.
        realtime = n * (1.0 / dt) / 1000.0
        print(f"{n:8d}{dt * 1e6:10.1f}{nbytes / 1024:11.0f} KB"
              f"{traffic / 1e9:10.2f} GB/s{100 * traffic / bw:10.1f}%"
              f"{1 / dt:10.0f}{realtime:12.0f}")

    print(f"\n(N,B,S) arrays counted per step: {len(tensors)} -> {sorted(tensors)}")
    print("\nIf the percentages are small, the step is not waiting on memory --\n"
          "it is waiting on per-call overhead, and the first speedups are on\n"
          "this CPU rather than on any accelerator.")


if __name__ == "__main__":
    main()
