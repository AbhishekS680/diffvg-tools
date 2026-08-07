#!/bin/zsh
# run_all_comparisons.sh
# Runs all three comparison pipelines across all 3 level hops, checking out
# each branch first, and scores each hop right after it finishes. Replaces
# the separate run_levels_*.py files -- the per-level loop and scoring now
# live directly in this script.
# Not part of the repo -- run manually from apps/.
set -e  # stop if any step fails, so you notice rather than silently continuing

ORIGINAL_BRANCH=$(git branch --show-current)

# 0 = original (sharpest), 3 = blurriest. Edit to match actual filenames.
LEVEL_0="imgs/level_0.png"
LEVEL_1="imgs/level_1.png"
LEVEL_2="imgs/level_2.png"
LEVEL_3="imgs/level_3.png"

run_hop() {
    local method=$1
    local script=$2
    local target=$3
    local degraded=$4
    local hop_label=$5
    local outdir="results/${hop_label}/$method"
    echo "--- $method: $hop_label ---"
    python "$script" --target "$target" --degraded "$degraded" --outdir "$outdir"
    python -c "
from score_results import score_path
ssim_val, lpips_val, passed = score_path('$outdir/final.png', '$target')
print(f'[$hop_label] $method: SSIM={ssim_val:.4f} LPIPS={lpips_val:.4f} PASS={passed}')
"
}

run_method() {
    local method=$1
    local script=$2
    run_hop "$method" "$script" "$LEVEL_2" "$LEVEL_3" "level_3_to_2"
    run_hop "$method" "$script" "$LEVEL_1" "$LEVEL_2" "level_2_to_1"
    run_hop "$method" "$script" "$LEVEL_0" "$LEVEL_1" "level_1_to_0"
}

echo "=== Wendland ==="
git checkout wendland
run_method wendland_boxed comparison_wendland_boxed.py

echo "=== Gaussian ==="
git checkout gaussian
run_method gaussian_boxed comparison_gaussian_boxed.py

echo "=== Shepard ==="
git checkout shepard
run_method shepard comparison_shepard.py

echo "=== Restoring original branch: $ORIGINAL_BRANCH ==="
git checkout "$ORIGINAL_BRANCH"

echo "=== All done ==="
echo "Run 'python score_results.py' now to see the full comparison table."