#!/bin/zsh
# run_all_n_vs_time.sh
# Runs all three n_vs_time benchmarks sequentially, checking out each branch first

set -e  # stop if any step fails

ORIGINAL_BRANCH=$(git branch --show-current)

echo "=== Wendland ==="
git checkout wendland
python n_vs_time_wendland.py
python n_vs_time_wendland_boxed.py

echo "=== Gaussian ==="
git checkout gaussian
python n_vs_time_gaussian.py
python n_vs_time_gaussian_boxed.py

echo "=== Shepard ==="
git checkout shepard
python n_vs_time_shepard.py

echo "=== Restoring original branch: $ORIGINAL_BRANCH ==="
git checkout "$ORIGINAL_BRANCH"

echo "=== All done ==="