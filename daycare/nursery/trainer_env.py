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


def use_train_tinygrad():
    """Put the train-only tinygrad on sys.path and return the module.

    Raises a clear error if the vendor dir is missing (run setup_trainer.sh).
    """
    if not os.path.isdir(os.path.join(TRAIN_TINYGRAD_PATH, "tinygrad")):
        # Fall back to an installed tinygrad (`pip install "daycare[train]"` pins the tinygrad-arkey fork),
        # unless a path was set explicitly.
        if "DAYCARE_TRAIN_TINYGRAD_PATH" not in os.environ:
            try:
                import tinygrad  # noqa: F401
                return tinygrad
            except ImportError:
                pass
        raise RuntimeError(
            f"train tinygrad not found at {TRAIN_TINYGRAD_PATH}; "
            "run daycare/nursery/setup_trainer.sh, pip install \"daycare[train]\", or set DAYCARE_TRAIN_TINYGRAD_PATH"
        )
    if TRAIN_TINYGRAD_PATH not in sys.path:
        sys.path.insert(0, TRAIN_TINYGRAD_PATH)
    import tinygrad  # noqa: F401

    return tinygrad
