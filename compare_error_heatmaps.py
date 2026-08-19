# compare_error_heatmaps.py
# Cross-primitive difference map: shows exactly where reconstruction A
# beats reconstruction B and vice versa, pixel by pixel
#
# Usage:
#   python compare_error_heatmaps.py \
#       --target imgs/fruit_basket.png \
#       --recon-a results/wendland_rendering_boxed/final.png \
#       --recon-b results/gaussian_rendering_boxed/final.png \
#       --label-a Wendland \
#       --label-b Gaussian \
#       --outdir results/compare_wendland_vs_gaussian
import argparse
import os
import numpy as np
import skimage.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--target', required=True)
parser.add_argument('--recon-a', required=True)
parser.add_argument('--recon-b', required=True)
parser.add_argument('--label-a', default='A')
parser.add_argument('--label-b', default='B')
parser.add_argument('--outdir', default='results/error_diff')
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

def load_rgb(path):
    img = skimage.io.imread(path).astype(np.float32) / 255.0
    return img[:, :, :3]

target = load_rgb(args.target)
recon_a = load_rgb(args.recon_a)
recon_b = load_rgb(args.recon_b)

assert target.shape == recon_a.shape == recon_b.shape, \
    'target, recon-a, and recon-b must all be the same resolution'

# Per-pixel error: mean squared difference across RGB channels.
# Same metric convention as the error_heatmap.png produced by the
# individual *_rendering*.py scripts, so these numbers are directly
# comparable to those.
error_a = ((target - recon_a) ** 2).mean(axis=2)
error_b = ((target - recon_b) ** 2).mean(axis=2)

# Positive (red) = A has more error than B here (B wins this pixel).
# Negative (blue) = A has less error than B here (A wins this pixel).
diff = error_a - error_b

# Shared color scale for error_a/error_b so they're visually comparable
# to each other, not just to the diff map.
shared_vmax = max(error_a.max(), error_b.max())

# Symmetric scale for the diff map, centered at zero, so "no
# difference" is always the same color (white) regardless of which
# run has more error overall.
diff_abs_max = np.abs(diff).max()

print(f'{args.label_a} mean error: {error_a.mean():.6f}')
print(f'{args.label_b} mean error: {error_b.mean():.6f}')
print(f'{args.label_a} wins (lower error) on {(diff < 0).mean() * 100:.1f}% of pixels')
print(f'{args.label_b} wins (lower error) on {(diff > 0).mean() * 100:.1f}% of pixels')

with open(f'{args.outdir}/summary.txt', 'w') as f:
    f.write(f'{args.label_a} mean error: {error_a.mean():.6f}\n')
    f.write(f'{args.label_b} mean error: {error_b.mean():.6f}\n')
    f.write(f'{args.label_a} wins (lower error) on {(diff < 0).mean() * 100:.2f}% of pixels\n')
    f.write(f'{args.label_b} wins (lower error) on {(diff > 0).mean() * 100:.2f}% of pixels\n')

# -------------------------------------------------------------------
# Standalone diff map.
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(diff, cmap='RdBu', vmin=-diff_abs_max, vmax=diff_abs_max)
ax.axis('off')
ax.set_title(f'{args.label_a} error minus {args.label_b} error\n'
             f'(blue = {args.label_a} better, red = {args.label_b} better)')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.savefig(f'{args.outdir}/diff_map.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved diff_map.png')

# -------------------------------------------------------------------
# Full comparison grid: target | recon A | recon B | error A | error B | diff
# -------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

axes[0, 0].imshow(target)
axes[0, 0].set_title('Target')
axes[0, 0].axis('off')

axes[0, 1].imshow(recon_a)
axes[0, 1].set_title(f'{args.label_a} reconstruction')
axes[0, 1].axis('off')

axes[0, 2].imshow(recon_b)
axes[0, 2].set_title(f'{args.label_b} reconstruction')
axes[0, 2].axis('off')

im_a = axes[1, 0].imshow(error_a, cmap='inferno', vmin=0, vmax=shared_vmax)
axes[1, 0].set_title(f'{args.label_a} error (mean={error_a.mean():.5f})')
axes[1, 0].axis('off')
fig.colorbar(im_a, ax=axes[1, 0], fraction=0.046, pad=0.04)

im_b = axes[1, 1].imshow(error_b, cmap='inferno', vmin=0, vmax=shared_vmax)
axes[1, 1].set_title(f'{args.label_b} error (mean={error_b.mean():.5f})')
axes[1, 1].axis('off')
fig.colorbar(im_b, ax=axes[1, 1], fraction=0.046, pad=0.04)

im_diff = axes[1, 2].imshow(diff, cmap='RdBu', vmin=-diff_abs_max, vmax=diff_abs_max)
axes[1, 2].set_title(f'Difference ({args.label_a} - {args.label_b})')
axes[1, 2].axis('off')
fig.colorbar(im_diff, ax=axes[1, 2], fraction=0.046, pad=0.04)

plt.savefig(f'{args.outdir}/full_comparison.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved full_comparison.png')