# compare_error_heatmaps_multi.py
# Shared color-scale error heatmap comparison across N primitives at
# once (not just a pair). Without a shared scale, matplotlib
# auto-normalizes each heatmap to its own min/max, so the same color
# can mean very different actual error values between runs -- making
# "eyeball the grid" comparisons misleading. This forces every
# heatmap onto one common scale so brightness is directly comparable
# across all of them.
#
# Usage:
#   python compare_error_heatmaps_multi.py \
#       --target imgs/fruit_basket.png \
#       --recon Wendland:results/wendland_rendering_boxed/final.png \
#       --recon Gaussian:results/gaussian_rendering_boxed/final.png \
#       --recon Shepard:results/shepard_rendering/final.png \
#       --recon TriangleSoup:results/trianglesoup_rendering_boxed/final.png \
#       --outdir results/compare_all_primitives
import argparse
import os
import numpy as np
import skimage.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--target', required=True)
parser.add_argument('--recon', action='append', required=True,
                     help='Label:path pair, e.g. Wendland:results/.../final.png. Repeatable.')
parser.add_argument('--outdir', default='results/compare_all_primitives')
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

def load_rgb(path):
    img = skimage.io.imread(path).astype(np.float32) / 255.0
    return img[:, :, :3]

target = load_rgb(args.target)

labels = []
recons = []
for entry in args.recon:
    label, path = entry.split(':', 1)
    labels.append(label)
    recon = load_rgb(path)
    assert recon.shape == target.shape, \
        f'{label} reconstruction ({path}) is not the same resolution as target'
    recons.append(recon)

# Per-pixel error for each primitive: same metric convention as the
# individual *_rendering*.py scripts (mean squared diff across RGB),
# so these numbers are directly comparable to those.
error_maps = [((target - r) ** 2).mean(axis=2) for r in recons]
mean_errors = [e.mean() for e in error_maps]

# Shared scale across ALL heatmaps in this comparison -- this is the
# whole point. Every subplot uses the same vmin/vmax, so "brighter"
# always means "more error" by the same amount, regardless of which
# primitive's panel you're looking at.
shared_vmax = max(e.max() for e in error_maps)

print('Mean error by primitive:')
for label, err in zip(labels, mean_errors):
    print(f'  {label}: {err:.6f}')
best_idx = int(np.argmin(mean_errors))
print(f'Lowest mean error: {labels[best_idx]}')

with open(f'{args.outdir}/summary.txt', 'w') as f:
    f.write('Mean error by primitive (shared scale, vmax=%.6f):\n' % shared_vmax)
    for label, err in zip(labels, mean_errors):
        f.write(f'  {label}: {err:.6f}\n')
    f.write(f'Lowest mean error: {labels[best_idx]}\n')

# -------------------------------------------------------------------
# Grid of heatmaps, all sharing one color scale.
# -------------------------------------------------------------------
n = len(labels)
fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
if n == 1:
    axes = [axes]
for ax, label, err, mean_err in zip(axes, labels, error_maps, mean_errors):
    im = ax.imshow(err, cmap='inferno', vmin=0, vmax=shared_vmax)
    ax.set_title(f'{label}\nmean={mean_err:.5f}')
    ax.axis('off')
fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
plt.savefig(f'{args.outdir}/shared_scale_heatmaps.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved shared_scale_heatmaps.png')

# -------------------------------------------------------------------
# Bar chart of mean error per primitive -- the single-number summary
# for a report figure.
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(1.5 * n + 2, 5))
colors = plt.cm.tab10(np.linspace(0, 1, n))
bars = ax.bar(labels, mean_errors, color=colors)
ax.set_ylabel('Mean squared error')
ax.set_title('Mean reconstruction error by primitive')
for bar, err in zip(bars, mean_errors):
    ax.annotate(f'{err:.5f}', (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha='center', va='bottom', fontsize=9)
plt.savefig(f'{args.outdir}/mean_error_bar_chart.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved mean_error_bar_chart.png')

# -------------------------------------------------------------------
# Full grid: target, each reconstruction, each error map (shared scale).
# -------------------------------------------------------------------
fig, axes = plt.subplots(2, n + 1, figsize=(6 * (n + 1), 12))
axes[0, 0].imshow(target)
axes[0, 0].set_title('Target')
axes[0, 0].axis('off')
axes[1, 0].axis('off')
for i, (label, recon, err, mean_err) in enumerate(zip(labels, recons, error_maps, mean_errors)):
    axes[0, i + 1].imshow(recon)
    axes[0, i + 1].set_title(f'{label} reconstruction')
    axes[0, i + 1].axis('off')
    im = axes[1, i + 1].imshow(err, cmap='inferno', vmin=0, vmax=shared_vmax)
    axes[1, i + 1].set_title(f'{label} error (mean={mean_err:.5f})')
    axes[1, i + 1].axis('off')
fig.colorbar(im, ax=axes[1, :], fraction=0.015, pad=0.01)
plt.savefig(f'{args.outdir}/full_grid.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print('saved full_grid.png')