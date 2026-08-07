#!/bin/zsh
# run_all_rendering.sh
# Runs all single-image rendering scripts sequentially, checking out each
# branch first. Not part of the repo -- run manually from apps/.

set -e  # stop if any step fails, so you notice rather than silently continuing

ORIGINAL_BRANCH=$(git branch --show-current)

echo "=== Wendland ==="
git checkout wendland
python wendland_rendering.py
python wendland_rendering_boxed.py

echo "=== Polynomial Kernel ==="
python polynomial_kernel_rendering.py

echo "=== Gaussian ==="
git checkout gaussian
python gaussian_rendering.py
python gaussian_rendering_boxed.py

echo "=== Shepard ==="
git checkout shepard
python shepard_rendering.py

echo "=== Restoring original branch: $ORIGINAL_BRANCH ==="
git checkout "$ORIGINAL_BRANCH"

echo "=== All done ==="