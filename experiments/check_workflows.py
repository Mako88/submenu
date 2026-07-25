"""Every flag a workflow passes must be a flag the experiment accepts.

IN PLAIN TERMS
--------------
The sweeps run on twenty machines at once and take about twenty minutes. If a
workflow passes an option that the script it calls does not understand, every
one of those machines dies on its first line and the whole run is wasted. You
find out twenty minutes later, and the fix is one word.

This checks all of them in about a second, before anything is launched.

WHY IT EXISTS
-------------
Three times now:

  - `plexus-binding.yml` once passed flags `binding.py` did not accept, and
    every seed died on its first line. The workflow's own "fail loudly if a
    condition produced nothing" guard exists because of it -- but that guard
    fires *after* the run, which is the expensive half.
  - Sweep 030 was written to pass `--bind-tau-pre` before `binding.py` had it.
    Caught by hand, one smoke test away from twenty wasted minutes.
  - The same shape as sweep 003's `PRETRAIN` and sweep 026's `--drain-steps`:
    the workflow and the script disagreeing about an option.

The guard-after-the-fact catches a run that produced nothing. This catches the
reason before it runs, which is the difference between a wasted matrix and a
typo.

HOW IT WORKS
------------
For every `python experiments/X.py ...` invocation in every workflow, resolve
the shell variables assigned in the same block (`common=`, `lag=`, `$tag`, and
so on), pull out every `--flag`, and check it against what `X.py --help` says it
accepts. Shell expansion is handled by substitution rather than by running
anything, so this is safe to run anywhere.

    python3 experiments/check_workflows.py
"""

from __future__ import annotations

import functools
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

# Substituted before parsing. These are GitHub's own expressions and shell
# constructs that carry no flags -- resolving them to a placeholder keeps the
# flag extraction honest without pretending to be a shell.
NOISE = [
    (re.compile(r"\$\{\{[^}]*\}\}"), "0"),
    (re.compile(r"\$\(\w+\)"), "0"),
]


@functools.lru_cache(maxsize=None)
def accepted_flags(script: Path) -> frozenset[str]:
    """What the script actually accepts, straight from its own parser.

    Cached: the workflows invoke the same handful of scripts about sixty times
    between them, and each `--help` pays for a numpy import. Uncached this took
    a minute, which is long enough that nobody would run it before committing --
    and a check nobody runs is worse than no check.
    """
    proc = subprocess.run(
        [sys.executable, script.name, "--help"],
        capture_output=True, text=True, cwd=script.parent,
    )
    if proc.returncode != 0:
        raise SystemExit(f"{script.name} --help failed:\n{proc.stderr}")
    return frozenset(re.findall(r"(--[a-z0-9][a-z0-9-]*)", proc.stdout))


def logical_lines(block: str) -> list[str]:
    """Join shell continuations, so a flag after a backslash is not lost."""
    out, buf = [], ""
    for raw in block.splitlines():
        line = raw.rstrip()
        if line.endswith("\\"):
            buf += line[:-1] + " "
            continue
        out.append(buf + line)
        buf = ""
    if buf:
        out.append(buf)
    return out


def check_block(block: str, where: str) -> list[str]:
    lines = logical_lines(block)
    for pattern, repl in NOISE:
        lines = [pattern.sub(repl, ln) for ln in lines]

    # Shell variables holding flag strings: common="--seed 0 --episodes 300".
    variables: dict[str, str] = {}
    for ln in lines:
        m = re.match(r'\s*(\w+)="([^"]*)"\s*$', ln)
        if m:
            variables[m.group(1)] = m.group(2)

    def expand(text: str) -> str:
        # Repeated because assignments nest: lag="--hebbian 1 $common".
        for _ in range(4):
            new = re.sub(r"\$(\w+)", lambda m: variables.get(m.group(1), ""), text)
            if new == text:
                break
            text = new
        return text

    problems = []
    for ln in lines:
        m = re.search(r"python3?\s+(experiments/(\w+)\.py)", ln)
        if not m:
            continue
        script = ROOT / m.group(1)
        if not script.exists():
            problems.append(f"{where}: no such script {m.group(1)}")
            continue
        used = set(re.findall(r"(--[a-z0-9][a-z0-9-]*)", expand(ln)))
        if not used:
            # Nothing to check, and probing anyway would be actively harmful:
            # `mutation.py` takes no arguments at all, so `--help` does not
            # short-circuit and the probe runs the entire mutation suite.
            continue
        unknown = used - accepted_flags(script)
        for flag in sorted(unknown):
            problems.append(f"{where}: {m.group(1)} does not accept {flag}")
    return problems


def main() -> None:
    problems, checked = [], 0
    for wf in sorted(WORKFLOWS.glob("*.yml")):
        text = wf.read_text()
        if "experiments/" not in text:
            continue
        checked += 1
        problems += check_block(text, wf.name)

    if problems:
        print(f"{len(problems)} workflow/script mismatches:")
        for p in problems:
            print(f"  - {p}")
        print("\nEach one kills every seed on its first line, twenty minutes "
              "before anyone finds out.")
        sys.exit(1)
    print(f"All experiment invocations in {checked} workflows accept the flags "
          "they are passed.")


if __name__ == "__main__":
    main()
