"""Seed, hardware, software and parameters, stored next to every result."""

import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from importlib import metadata

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _read_proc(path: str, key: str) -> str | None:
    """Value of the first `key: value` line in a /proc file (Linux only)."""
    try:
        with open(path) as f:
            for line in f:
                if line.startswith(key):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_runinfo(params: dict, seed: int, packages: tuple[str, ...] = ()) -> dict:
    """`packages` are extra libraries whose versions matter for this run."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "params": params,
        "git_commit": _git("rev-parse", "HEAD"),
        # Uncommitted changes mean the commit alone does not reproduce the run.
        # results/ is excluded: a run rewrites its own output file, which
        # would otherwise mark every later run in the same session as dirty.
        "git_dirty": bool(_git("status", "--porcelain", "--", ".", ":(exclude)results")),
        "hardware": {
            "machine": platform.machine(),
            "cpu": _read_proc("/proc/cpuinfo", "model name") or platform.processor() or None,
            "logical_cores": os.cpu_count(),
            "ram": _read_proc("/proc/meminfo", "MemTotal"),
        },
        "software": {
            "os": platform.platform(),
            "python": platform.python_version(),
            **{pkg: metadata.version(pkg) for pkg in ("numpy", "pandas", "pyarrow", *packages)},
        },
    }
