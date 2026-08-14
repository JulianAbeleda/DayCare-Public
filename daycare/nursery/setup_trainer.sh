#!/usr/bin/env bash
# Fetch DayCare's train-only tinygrad: a pinned upstream copy, vendored under
# daycare/nursery/vendor/ (git-ignored). This is the TRAIN side of the split;
# the SERVE side is the arkey fork used by daycare/substrate/tinygrad_qwen3.py.
#
# Pinned upstream provides autograd (gradient.py) + nn/optim.py + nn/state.py,
# which the arkey serve fork removed. No torch/transformers.
set -euo pipefail

PIN="tinygrad==0.13.0"
DIR="$(cd "$(dirname "$0")" && pwd)/vendor"

echo "installing $PIN (no deps) into $DIR ..."
python3 -m pip install --no-deps --upgrade --target "$DIR" "$PIN"
echo "train tinygrad ready: $DIR/tinygrad"
