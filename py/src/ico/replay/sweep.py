"""Engine flags and the sweep file that lists the configs to try."""

from __future__ import annotations

import itertools
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

# Engine flags as a mapping, without the leading dashes. True is a bare
# switch (--enable-x), False its negation (--no-enable-x).
Flags = dict[str, Any]

# Changing any of these changes the numbers the model computes with, so the
# answers may get worse. Such a config is measured but cannot win until a
# quality check exists.
PRECISION_FLAGS = frozenset({"quantization", "dtype", "kv-cache-dtype", "model"})


class SweepError(ValueError):
    pass


def as_args(flags: Flags) -> list[str]:
    """Flags as a list of command-line arguments."""
    args = []
    for name, value in flags.items():
        if value is True:
            args.append(f"--{name}")
        elif value is False:
            args.append(f"--no-{name}")
        else:
            args += [f"--{name}", str(value)]
    return args


def render(flags: Flags) -> str:
    """Flags as they are typed on the engine's command line."""
    return " ".join(shlex.quote(arg) for arg in as_args(flags))


def parse_args(args: list[str]) -> tuple[list[str], Flags]:
    """Split an engine command line into its leading arguments and its flags.

    ``serve org/model --max-num-seqs 64 --enforce-eager`` gives
    ``["serve", "org/model"]`` and ``{"max-num-seqs": 64, "enforce-eager": True}``.
    Numbers come back as numbers, so they compare equal to the ones a sweep
    file names.
    """
    positional: list[str] = []
    flags: Flags = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if not arg.startswith("--"):
            if flags:
                raise SweepError(f"cannot tell the flags from the arguments in: {' '.join(args)}")
            positional.append(arg)
        else:
            name, has_value, value = arg[2:].partition("=")
            if has_value:
                flags[name] = _number(value)
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                flags[name] = _number(args[i + 1])
                i += 1
            elif name.startswith("no-"):
                flags[name[3:]] = False
            else:
                flags[name] = True
        i += 1
    return positional, flags


def _number(text: str) -> Any:
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def overlay(base: Flags, changes: Mapping[str, Any]) -> Flags:
    """base with changes laid over it. A change to None removes the flag."""
    merged = dict(base)
    for name, value in changes.items():
        if value is None:
            merged.pop(name, None)
        else:
            merged[name] = value
    return merged


@dataclass(frozen=True)
class Candidate:
    """One config to test: the replica's original flags with some of them changed."""

    id: str
    flags: Flags
    changes: Flags

    @property
    def quality_unverified(self) -> bool:
        return bool(PRECISION_FLAGS & self.changes.keys())


def baseline(original: Flags) -> Candidate:
    return Candidate(id="current", flags=dict(original), changes={})


def load(path: Path, original: Flags) -> list[Candidate]:
    """Read a sweep file.

    ``configs`` lists configs one by one, each a mapping of the flags to
    change. ``grid`` gives several values per flag and stands for every
    combination. Configs that change nothing, or repeat another, are dropped.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or not (raw.keys() <= {"configs", "grid"}):
        raise SweepError(f"{path}: expected the keys configs and grid, nothing else")

    changes: list[dict[str, Any]] = []
    for i, entry in enumerate(raw.get("configs") or []):
        if not isinstance(entry, dict) or not entry:
            raise SweepError(f"{path}: configs[{i}] must be a mapping of flag to value")
        changes.append(entry)
    grid = raw.get("grid") or {}
    if not isinstance(grid, dict) or not all(isinstance(v, list) and v for v in grid.values()):
        raise SweepError(f"{path}: grid must map each flag to a list of values")
    if grid:
        changes += [dict(zip(grid, combo)) for combo in itertools.product(*grid.values())]

    candidates: dict[str, Candidate] = {}
    for entry in changes:
        flags = overlay(original, entry)
        differing = {k: v for k, v in entry.items() if original.get(k) != v}
        if not differing:
            continue  # the replica already runs this
        name = ",".join(f"{k}={'off' if v is None else v}" for k, v in sorted(differing.items()))
        candidates.setdefault(name, Candidate(id=name, flags=flags, changes=differing))
    if not candidates:
        raise SweepError(f"{path}: no config differs from what the replica runs now")
    return list(candidates.values())
