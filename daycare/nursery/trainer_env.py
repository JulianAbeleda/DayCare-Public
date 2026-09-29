"""DayCare's train-only tinygrad.

The split:

    upstream tinygrad   everything -- too broad to depend on wholesale
    DayCare (this)      TRAIN only: a pinned upstream copy for autograd + optim
    arkey               SERVE only: inference, training stripped (daycare/substrate)

This module wires the vendored train tinygrad (autograd + nn.optim + nn.state) so
DayCare's consolidation code imports it explicitly, kept separate from the arkey
SERVE substrate in daycare/substrate/tinygrad_qwen3.py. Fetch the vendor dir with
`daycare/nursery/setup_trainer.sh` (pinned, git-ignored).
"""
from __future__ import annotations

import os
import sys

# The vendored train-only tinygrad (git-ignored; installed by setup_trainer.sh).
TRAIN_TINYGRAD_PATH = os.environ.get(
    "DAYCARE_TRAIN_TINYGRAD_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"),
)


def trainer_root() -> str | None:
    """The directory holding the trainer `tinygrad/` package, or None if there is none.

    DAYCARE_TRAIN_TINYGRAD_PATH (a tinygrad-arkey checkout, needed by the RL loop), else the
    vendor dir from setup_trainer.sh. A pip-installed tinygrad is deliberately not used: the
    tinygrad-arkey wheel omits the `extra/` tree its own modules import (docs/rl-training.md).
    """
    if os.path.isdir(os.path.join(TRAIN_TINYGRAD_PATH, "tinygrad")):
        return TRAIN_TINYGRAD_PATH
    return None


def trainer_available() -> bool:
    return trainer_root() is not None


def trainer_revision() -> str:
    """The trainer tinygrad's revision: git HEAD of a checkout, else the vendored version."""
    root = trainer_root()
    if root is None:
        raise RuntimeError("no trainer tinygrad (see use_train_tinygrad)")
    if os.path.exists(os.path.join(root, ".git")):
        import subprocess
        return subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"]).decode().strip()
    import glob
    found = sorted(glob.glob(os.path.join(root, "tinygrad-*.dist-info")))
    return os.path.basename(found[-1])[: -len(".dist-info")].replace("-", "==", 1) if found else "unknown"


def trainer_subprocess_env(env: dict | None = None) -> dict:
    """An environment for a child process that selects the same trainer tinygrad as this one."""
    env = dict(os.environ if env is None else env)
    root = trainer_root()
    if root is not None:
        env["DAYCARE_TRAIN_TINYGRAD_PATH"] = root
    return env


def use_train_tinygrad():
    """Put the trainer tinygrad on sys.path and return the module.

    Raises a clear error if there is none.
    """
    root = trainer_root()
    if root is None:
        raise RuntimeError(
            f"train tinygrad not found at {TRAIN_TINYGRAD_PATH}; set DAYCARE_TRAIN_TINYGRAD_PATH to a "
            "tinygrad-arkey checkout at the pinned commit, or run daycare/nursery/setup_trainer.sh "
            "(docs/rl-training.md)"
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    import tinygrad  # noqa: F401

    return tinygrad
