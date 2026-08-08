# score_results.py
# Run standalone after all branches have produced final.png outputs
import os
import numpy as np
import skimage.io
from skimage.metrics import structural_similarity as ssim
import lpips
import torch

# 0 = original (sharpest), 4 = blurriest
LEVEL_IMAGES = {
    0: 'imgs/level_0.png',
    1: 'imgs/level_1.png',
    2: 'imgs/level_2.png',
    3: 'imgs/level_3.png',
    4: 'imgs/level_4.png',
}

METHODS = ['wendland_boxed', 'gaussian_boxed', 'shepard']  # can remove/add methods

SSIM_THRESH = 0.9 # Above this is a pass
LPIPS_THRESH = 0.1 # Below this is a pass

_lpips_model = None

def get_lpips_model():
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net='alex')
    return _lpips_model

def to_lpips_tensor(img):
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    return t * 2 - 1  # lpips expects NCHW in [-1, 1]

def score(reconstructed_np, target_np, ssim_thresh=SSIM_THRESH, lpips_thresh=LPIPS_THRESH):
    """reconstructed_np, target_np: HxWx3 float arrays in [0,1]. Returns (ssim, lpips, passed)."""
    ssim_val = ssim(reconstructed_np, target_np, channel_axis=2, data_range=1.0)
    model = get_lpips_model()
    with torch.no_grad():
        lpips_val = model(to_lpips_tensor(reconstructed_np), to_lpips_tensor(target_np)).item()
    passed = (ssim_val > ssim_thresh) and (lpips_val < lpips_thresh)
    return ssim_val, lpips_val, passed

def score_path(final_path, target_path, ssim_thresh=SSIM_THRESH, lpips_thresh=LPIPS_THRESH):
    """Convenience wrapper: load two image files and score them."""
    final_np = skimage.io.imread(final_path).astype(np.float32)[:, :, :3] / 255.0
    target_np = skimage.io.imread(target_path).astype(np.float32)[:, :, :3] / 255.0
    return score(final_np, target_np, ssim_thresh, lpips_thresh)

def main():
    """Standalone sweep: baseline (degraded vs target, no reconstruction)
    followed by every level/method combo that has a final.png so far."""
    results_log = []

    print("--- Baseline (degraded vs target, no reconstruction) ---")
    for n in [4, 3, 2, 1]:
        n_minus_1 = n - 1
        target_path = LEVEL_IMAGES[n_minus_1]
        degraded_path = LEVEL_IMAGES[n]
        if not os.path.exists(target_path) or not os.path.exists(degraded_path):
            print(f"Missing image for level {n}->{n_minus_1}, skipping baseline")
            continue
        ssim_val, lpips_val, passed = score_path(degraded_path, target_path)
        results_log.append({
            'level': f'{n}->{n_minus_1}', 'method': 'baseline',
            'ssim': ssim_val, 'lpips': lpips_val, 'passed': passed,
        })
        print(f"[{n}->{n_minus_1}] baseline: SSIM={ssim_val:.4f} "
              f"LPIPS={lpips_val:.4f} PASS={passed}")

    print("--- Reconstructions ---")
    for n in [4, 3, 2, 1]:
        n_minus_1 = n - 1
        target_path = LEVEL_IMAGES[n_minus_1]
        if not os.path.exists(target_path):
            print(f"Missing target image: {target_path}, skipping level {n}->{n_minus_1}")
            continue
        for method in METHODS:
            final_path = f'results/level_{n}_to_{n_minus_1}/{method}/final.png'
            if not os.path.exists(final_path):
                print(f"[{n}->{n_minus_1}] {method}: no final.png found yet, skipping")
                continue
            ssim_val, lpips_val, passed = score_path(final_path, target_path)
            results_log.append({
                'level': f'{n}->{n_minus_1}', 'method': method,
                'ssim': ssim_val, 'lpips': lpips_val, 'passed': passed,
            })
            print(f"[{n}->{n_minus_1}] {method}: SSIM={ssim_val:.4f} "
                  f"LPIPS={lpips_val:.4f} PASS={passed}")

    os.makedirs('results', exist_ok=True)
    with open('results/level_summary.txt', 'w') as f:
        f.write("--- Baseline (degraded vs target, no reconstruction) ---\n")
        for r in results_log:
            if r['method'] == 'baseline':
                f.write(f"{r['level']} | {r['method']:9s} | SSIM={r['ssim']:.4f} "
                        f"LPIPS={r['lpips']:.4f} | PASS={r['passed']}\n")
        f.write("--- Reconstructions ---\n")
        for r in results_log:
            if r['method'] != 'baseline':
                f.write(f"{r['level']} | {r['method']:9s} | SSIM={r['ssim']:.4f} "
                        f"LPIPS={r['lpips']:.4f} | PASS={r['passed']}\n")
    print('saved results/level_summary.txt')

if __name__ == '__main__':
    main()