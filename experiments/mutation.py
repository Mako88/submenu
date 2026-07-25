"""Break each mechanism on purpose and check the test that names it fails.

A passing suite proves nothing on its own. The question is whether any test
would go red if the thing it claims to guard stopped working, and the only way
to answer that is to break it and look.

Three tests in this repo failed that check when it was first run, all of them
green at the time:

    test_plateau_stays_engaged      bounds 0.01-0.5 admitted the broken case
                                    (knee_lr=0 settles at 0.074), while
                                    guarding the exact bug where the plateau
                                    was silently disconnected.
    test_no_runaway_excitation      asserted |v| < 50 where the operating range
                                    is 0.49. Breaking homeostasis outright
                                    still left |v| at 0.31.
    test_weights_stay_bounded       asserted W.max() <= weight_max at settings
                                    where synaptic scaling holds W at 1.775
                                    against a limit of 4.0. Raising the limit
                                    to 1e9 left W unchanged, so the clip was
                                    never doing anything the test could see.

Each mutation below patches one line of source, runs the tests that ought to
notice, and restores the file. Deliberately crude -- it edits text rather than
an AST -- because the point is to be obviously correct rather than general.

**Nothing else may run against this repo while this is running.** It edits
`plexus/column.py` in place, so any experiment launched during a mutation window
imports the broken module, produces plausible numbers from it, and writes them
to a results file that carries no record of the fact. That is the project's
signature failure -- a quantity that looks connected and is not -- with the
mutation harness as the cause. Finish or kill background runs first.

Being killed is survivable but not free: the `finally` that restores the source
does not run on SIGKILL, so the original is parked in `.mutation-backup` and the
next run restores from it. That path exists because a two-minute timeout once
left `column.py` carrying `self.bind_pre *= self.decay_branch`, and it was
committed.

    python3 experiments/mutation.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLUMN = ROOT / "plexus" / "column.py"
READOUT = ROOT / "plexus" / "readout.py"

# (file, description, find, replace, pytest -k expression that should now fail)
MUTATIONS = [
    (
        COLUMN,
        "plateau knee homeostasis disabled",
        "knee_lr: float = 3e-3",
        "knee_lr: float = 0.0",
        "plateau",
    ),
    (
        COLUMN,
        "somatic threshold homeostasis disabled",
        "homeostatic_lr: float = 1.5e-2",
        "homeostatic_lr: float = 0.0",
        "homeostasis or runaway",
    ),
    (
        COLUMN,
        "weight clip removed from the learning rule",
        "        np.clip(self.W, 0.0, cfg.weight_max, out=self.W)\n\n        if cfg.lr_tau",
        "        pass\n\n        if cfg.lr_tau",
        "clip or bounded",
    ),
    (
        COLUMN,
        "synaptic weights dropped from the forward pass (the original bug)",
        "self.b += self.gain_branch * (self.W * x).sum(axis=2)",
        "self.b += self.gain_branch * x.sum(axis=2)",
        "synaptic_weights or branch_gains",
    ),
    (
        COLUMN,
        "branch gains dropped from the soma",
        "drive = (self.G * a).sum(axis=1)",
        "drive = a.sum(axis=1)",
        "branch_gains",
    ),
    (
        COLUMN,
        "short-term plasticity made a no-op",
        "        return (value * release / self._release_ref).astype(np.float32)",
        "        return value.astype(np.float32)",
        "facilitation or depression or efficacy or stp",
    ),
    (
        COLUMN,
        "Dale sign dropped from the gathered input",
        "        x = x * self.syn_sign",
        "        x = x * np.abs(self.syn_sign)",
        "dale or inhibitory",
    ),
    (
        COLUMN,
        "Hebbian binding ignores how active the neuron was",
        "rate * post[:, None, None] * (bind_pre * self.syn_sign) * excitatory",
        "rate * cfg.bind_scale * (bind_pre * self.syn_sign) * excitatory",
        "binding_scales_with_how_active",
    ),
    (
        COLUMN,
        "Hebbian binding ignores which synapses were driving the neuron",
        "rate * post[:, None, None] * (bind_pre * self.syn_sign) * excitatory",
        "rate * post[:, None, None] * excitatory",
        "binding_carries_which_synapses",
    ),
    (
        COLUMN,
        "Hebbian binding potentiates inhibitory synapses too",
        "rate * post[:, None, None] * (bind_pre * self.syn_sign) * excitatory",
        "rate * post[:, None, None] * (bind_pre * self.syn_sign)",
        "excitatory_synapses",
    ),
    (
        COLUMN,
        "binding rate decay ignored",
        "rate = cfg.hebb_lr * cfg.hebb_decay ** (self.n_samples - 1)",
        "rate = cfg.hebb_lr",
        "binding_rate_decays",
    ),
    (
        COLUMN,
        "step counter advances during evaluation, shifting the scaling schedule",
        "            self._steps += 1\n\n        return self.out",
        "            pass\n\n        self._steps += 1\n        return self.out",
        "probing_does_not_perturb or advance_the_scaling_schedule",
    ),
    (
        COLUMN,
        "lateral inhibition potentiates excitatory synapses too",
        "self.W += (step * post * (pre_raw / scale) * inhibitory).astype(np.float32)",
        "self.W += (step * post * (pre_raw / scale)).astype(np.float32)",
        "lateral_inhibition_only_moves",
    ),
    (
        COLUMN,
        "lateral inhibition rate left unnormalised",
        "step = cfg.lateral_lr * (cfg.branch_budget / cfg.n_synapses)",
        "step = cfg.lateral_lr * 1e-4",
        "lateral_inhibition_step_is_scaled",
    ),
    (
        COLUMN,
        "freezing plasticity forgets a mechanism (the sweep 017 catch-up bug)",
        "    for flag in PLASTICITY_FLAGS:\n        setattr(cfg, flag, False)",
        "    cfg.hebbian = False",
        "freezing_plasticity_is_the_same",
    ),
    (
        COLUMN,
        "a new mechanism flag is added without classifying it",
        "    lateral: bool = False\n",
        "    lateral: bool = False\n    some_new_rule: bool = False\n",
        "freeze_plasticity_accounts_for_every",
    ),
    (
        COLUMN,
        "add_inputs leaves the precomputed gather table stale",
        "        if self._fast:\n            d = self.transport.depth\n            slot_tbl",
        "        if False:\n            d = self.transport.depth\n            slot_tbl",
        "adding_inputs_rebuilds",
    ),
    (
        COLUMN,
        "add_inputs rewires metadata but no synapse actually moves",
        "        self.src[idx[0], idx[1], slots] = picks",
        "        pass",
        "added_inputs_reach or adding_inputs_appends",
    ),
    (
        COLUMN,
        "the fast gather index is off by one source (fast path diverges silently)",
        "                (slots * transport.n_sources + self.src[None]).astype(np.int32).reshape(d, -1)",
        "                (slots * transport.n_sources + self.src[None] + 1).astype(np.int32).reshape(d, -1)",
        "fast_gather_path",
    ),
    (
        ROOT / "plexus" / "events.py",
        "a never-delivered packet reads as a stale echo instead of silence",
        "        return np.where(self._stamp[slots] == emitted_at, vals, 0.0)",
        "        return vals",
        "silence_for_unwritten or delivery_jitter",
    ),
    (
        ROOT / "plexus" / "events.py",
        "buffer growth drops events already in flight",
        "        buf[:, : self.n_sources] = self._buf",
        "        pass",
        "growing_the_event_buffer",
    ),
    (
        COLUMN,
        "reported sparsity returns the EMA seed instead of what was observed",
        "        return self._debias(self.rate, self.cfg.target_rate, float(self.decay_rate))",
        "        return float(self.rate.mean())",
        "reported_sparsity_is_measured",
    ),
    (
        COLUMN,
        "an EMA loses the sample counter that lets its seed be divided out",
        "            self._obs_steps += 1",
        "            pass",
        "reported_sparsity_is_measured or every_ema_either_debiases",
    ),
    (
        ROOT / "plexus" / "tasks.py",
        "task variants share a channel mapping, making the stream one task",
        "            else np.random.default_rng(channel_seed).permutation(self.n_inputs)",
        "            else np.random.default_rng(0).permutation(self.n_inputs)",
        "channel_permutation_is_a_relabelling",
    ),
    (
        ROOT / "plexus" / "tasks.py",
        "the channel permutation applies by default, changing every recorded sweep",
        "            None if channel_seed is None",
        "            None if False",
        "channel_permutation_is_off_by_default",
    ),
    (
        ROOT / "plexus" / "tasks.py",
        "the cue mask is widened past the burst, so it selects background noise",
        "            cue_active[t0 : t0 + dur] = True",
        "            cue_active[t0 : t0 + dur * 3] = True",
        "cue_active_marks_the_bursts",
    ),
    (
        ROOT / "experiments" / "criticality.py",
        "branching fit cuts on positivity, biasing m upward toward critical",
        "    FLOOR = 0.02",
        "    FLOOR = 1e-6",
        "branching_estimator_recovers",
    ),
    (
        ROOT / "experiments" / "criticality.py",
        "branching ratio read off the intercept instead of the decay slope",
        "    slope = np.polyfit(np.array(ks, dtype=np.float64), np.log(rs), 1)[0]",
        "    slope = np.polyfit(np.array(ks, dtype=np.float64), np.log(rs), 1)[1]",
        "branching_estimator_recovers",
    ),
    (
        READOUT,
        "readout standardisation reduced to centering",
        "z = ((state - mean) / (std + 1e-12)).astype(np.float32)",
        "z = (state - mean).astype(np.float32)",
        "standardisation",
    ),
    # The two traces binding multiplies together. Each mutation leaves the
    # *other* trace intact, so the rule still runs, still writes structured
    # weights, and still passes most of the binding suite -- what it loses is
    # one of the two constants that set how late a modulator may arrive, which
    # is the quantity sweep 026 got wrong by tuning the term that was not
    # limiting.
    (
        COLUMN,
        "presynaptic trace flattened in binding (latency window loses tau_branch)",
        "rate * post[:, None, None] * (bind_pre * self.syn_sign) * excitatory",
        "rate * post[:, None, None] * (np.ones_like(bind_pre) * self.syn_sign) "
        "* excitatory",
        "latency_window or synapses_were_driving",
    ),
    (
        COLUMN,
        "activity factor flattened in binding (latency window loses tau_act_fast)",
        "post = np.clip(self.act_fast / (baseline + 1e-9), 0.0, 5.0) * cfg.bind_scale",
        "post = np.ones_like(self.act_fast) * cfg.bind_scale",
        "latency_window or how_active",
    ),
    (
        COLUMN,
        "binding's own presynaptic trace decays at the forward path's constant",
        "            self.bind_pre *= self.decay_bind_pre",
        "            self.bind_pre *= self.decay_branch",
        "bind_tau_pre",
    ),
    (
        COLUMN,
        "a dead quantity is added to the forward path (the self.bias bug)",
        "        drive = (self.G * a).sum(axis=1)",
        "        self._dead = np.zeros_like(self.theta)\n"
        "        drive = (self.G * a).sum(axis=1) + self._dead",
        "written_or_declared_constant",
    ),
    (
        COLUMN,
        "binding reads the forward path's trace even when given its own",
        "bind_pre = self.pre if self.bind_pre is None else self.bind_pre",
        "bind_pre = self.pre",
        "bind_tau_pre",
    ),
]


def run(k: str) -> bool:
    """True if the selected tests failed, i.e. the mutation was caught."""
    # Bytecode caching makes this silently unreliable, and it took two runs
    # disagreeing to notice. CPython validates a .pyc against the source's size
    # and its mtime *truncated to whole seconds*. This script rewrites one path
    # several times per second, so a mutation whose source happens to match a
    # cached entry on both can be skipped entirely -- the subprocess then
    # imports the unmutated module, every test passes, and the mutation is
    # reported as ESCAPED. A checker that intermittently invents findings is
    # worse than no checker, so the caches go before every run.
    for cache in ROOT.rglob("__pycache__"):
        for pyc in cache.glob("*.pyc"):
            pyc.unlink()
    proc = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", "plexus/tests/test_plexus.py",
         "-q", "--no-header", "-p", "no:cacheprovider", "-k", k],
        capture_output=True, text=True, cwd=ROOT,
    )
    # Parenthesised deliberately: `A or B and C` binds as `A or (B and C)`,
    # which is not what this condition means.
    if "no tests ran" in proc.stdout or ("0 passed" in proc.stdout and "failed" not in proc.stdout):
        print(f"      (no test matched -k {k!r} -- that is itself the finding)")
        return False
    return proc.returncode != 0


# Where an in-flight mutation's original source is parked, so a run that is
# killed rather than finished can be undone by the next one.
#
# `try/finally` is not enough on its own. It restores the file when the process
# raises, and does nothing at all when the process is killed -- and this script
# gets killed: it was once invoked as a subprocess by `check_workflows.py`
# probing `--help`, hit a two-minute timeout, and left `column.py` carrying
# `self.bind_pre *= self.decay_branch`. That was committed. The test written for
# exactly that mutation failed on the committed tree, which is the system
# working, but nothing had to make it that far.
BACKUP = ROOT / ".mutation-backup"


def restore_any_interrupted_run() -> None:
    """Undo a mutation left behind by a run that was killed."""
    if not BACKUP.exists():
        return
    target, _, original = BACKUP.read_text().partition("\n")
    path = Path(target)
    if path.read_text() != original:
        path.write_text(original)
        print(f"  restored {path.name} from an interrupted run")
    BACKUP.unlink()


def main() -> None:
    restore_any_interrupted_run()
    escaped = []
    for path, name, find, repl, k in MUTATIONS:
        original = path.read_text()
        if find not in original:
            print(f"  SKIP    {name}: source moved, update the mutation")
            escaped.append(name + " (stale mutation)")
            continue
        try:
            BACKUP.write_text(f"{path}\n{original}")
            path.write_text(original.replace(find, repl, 1))
            caught = run(k)
        finally:
            path.write_text(original)
            BACKUP.unlink(missing_ok=True)
        print(f"  {'caught ' if caught else 'ESCAPED'} {name}")
        if not caught:
            escaped.append(name)

    print()
    if escaped:
        print(f"{len(escaped)}/{len(MUTATIONS)} mutations escaped. Each one is a mechanism")
        print("the suite would not notice the loss of:")
        for name in escaped:
            print(f"  - {name}")
        sys.exit(1)
    print(f"All {len(MUTATIONS)} mutations caught.")


if __name__ == "__main__":
    main()
