# compare_kernels.py
"""
Quick comparison figure: target image vs Shepard / Wendland / Gaussian outputs.
Usage: edit the paths below, then run:
    python compare_kernels.py
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# --- edit these paths ---
target_path   = "/Users/abhisheksinha/Desktop/NSERC/vector_graphics/Cat/level_0.jpg"
shepard_path  = "results/Cat Reconstruction/level_1_to_0/shepard/final.png"
wendland_path = "results/Cat Reconstruction/level_1_to_0/wendland/final.png"
gaussian_path = "results/Cat Reconstruction/level_1_to_0/gaussian/final.png"
out_path      = "results/kernel_comparison.png"
# -------------------------

images = [
    ("Target", target_path),
    ("Shepard IDW", shepard_path),
    ("Wendland C2", wendland_path),
    ("Gaussian RBF", gaussian_path),
]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (title, path) in zip(axes, images):
    img = mpimg.imread(path)
    ax.imshow(img)
    ax.set_title(title, fontsize=12)
    ax.axis("off")

plt.tight_layout()
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.show()