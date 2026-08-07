# plot_comparison.py
# Reads results/level_summary.txt (written by score_results.py) and produces
# grouped bar charts comparing SSIM and LPIPS across hops and methods.
# Run standalone, after score_results.py has been run at least once.
import re
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SUMMARY_PATH = 'results/Cat Reconstruction/level_summary.txt'
OUTDIR = 'results'

# Matches lines like:
# 3->2 | wendland  | SSIM=0.9866 LPIPS=0.0512 | PASS=True
LINE_RE = re.compile(
    r'^(?P<level>\d->\d)\s*\|\s*(?P<method>\S+)\s*\|\s*'
    r'SSIM=(?P<ssim>[\d.]+)\s*LPIPS=(?P<lpips>[\d.]+)\s*\|\s*PASS=(?P<passed>\w+)'
)

def parse_summary(path):
    """Returns dict: {method: {level: (ssim, lpips)}}, and ordered level/method lists."""
    data = {}
    levels_seen = []
    methods_seen = []
    with open(path) as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if not m:
                continue
            level = m.group('level')
            method = m.group('method')
            ssim = float(m.group('ssim'))
            lpips = float(m.group('lpips'))
            if level not in levels_seen:
                levels_seen.append(level)
            if method not in methods_seen:
                methods_seen.append(method)
            data.setdefault(method, {})[level] = (ssim, lpips)
    return data, levels_seen, methods_seen

def plot_metric(data, levels, methods, metric_idx, title, ylabel, ylim, outpath):
    n_methods = len(methods)
    n_levels = len(levels)
    x = np.arange(n_levels)
    group_width = 0.8
    width = group_width / n_methods
    bar_width = width * 0.82  # shrink each bar so a visible gap remains between them

    # Colors keyed by method name (lowercase). Anything not listed here falls
    # back to matplotlib's default color cycle.
    COLOR_MAP = {
        'baseline': '#2D3142',
        'wendland': '#3C896D',
        'gaussian': '#EF8354',
        'shepard':  '#0582CA',
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, method in enumerate(methods):
        values = [data[method].get(level, (None, None))[metric_idx] for level in levels]
        offset = (i - (n_methods - 1) / 2) * width
        color = COLOR_MAP.get(method.lower())
        ax.bar(x + offset, values, bar_width, label=method.capitalize(), color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_xlabel('Hop (blurrier -> sharper)')
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.legend()
    plt.savefig(outpath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'saved {outpath}')

def main():
    if not os.path.exists(SUMMARY_PATH):
        print(f'{SUMMARY_PATH} not found -- run score_results.py first')
        return
    data, levels, methods = parse_summary(SUMMARY_PATH)
    if not data:
        print('No matching lines found in level_summary.txt')
        return

    # Put baseline first if present, keep the rest in the order they appeared
    if 'baseline' in methods:
        methods = ['baseline'] + [m for m in methods if m != 'baseline']

    os.makedirs(OUTDIR, exist_ok=True)
    plot_metric(data, levels, methods, metric_idx=0,
                title='SSIM by hop (higher is better)', ylabel='SSIM',
                ylim=(0, 1), outpath=f'{OUTDIR}/comparison_ssim.png')
    plot_metric(data, levels, methods, metric_idx=1,
                title='LPIPS by hop (lower is better)', ylabel='LPIPS',
                ylim=(0, 1), outpath=f'{OUTDIR}/comparison_lpips.png')

if __name__ == '__main__':
    main()