"""Estimated shape images at the ranks chosen by AIC, BIC and the optimism selector.

    python plot_shape_estimates.py --model tucker

Rows: true signal, AIC, BIC, optimism selector; columns: shapes. Each shape shows the seed
listed in results/<model>/figure_seeds.json (fits from results/<model>/fits/). Panels give
the selected rank and that fit's test MSE. Writes figures/shape_estimates_3d_<model>.pdf.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SHAPES = ["square", "T", "cross", "disk", "triangle", "butterfly"]
ROWS = [("truth", "True signal"), ("aic", "AIC"), ("bic", "BIC"), ("sel_logn", "Optimism")]


def rank_text(r):
    return f"({r})" if "," in r else r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cp", "tucker"], required=True)
    model = p.parse_args().model
    res = ROOT / "results" / model
    seeds = json.loads((res / "figure_seeds.json").read_text())
    truth = np.load(res / "truth.npz")
    fits_all = pd.read_csv(res / "all_fits.csv", dtype={"rank": str, "true_rank": str})
    fig, axes = plt.subplots(len(ROWS), len(SHAPES), figsize=(1.75 * len(SHAPES) + 0.9, 1.95 * len(ROWS)))
    for j, shape in enumerate(SHAPES):
        info = seeds[shape]
        rep = info["rep"]
        fits = np.load(res / "fits" / f"{shape}_rep{rep}.npz")
        rows = fits_all[(fits_all["shape"] == shape) & (fits_all.rep == rep)].set_index("rank")
        for i, (key, label) in enumerate(ROWS):
            ax = axes[i, j]
            if key == "truth":
                ax.imshow(np.clip(truth[shape], 0, 1), interpolation="nearest")
                ax.set_title(shape.capitalize() if shape != "T" else "T", fontsize=11, pad=6)
            else:
                r = info[key]
                ax.imshow(np.clip(fits[r], 0, 1), interpolation="nearest")
                ax.set_title(f"rank {rank_text(r)}\ntest MSE {rows.loc[r, 'test_error']:.1f}",
                             fontsize=8.5, linespacing=1.15, pad=4)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(label, fontsize=11, labelpad=8)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / f"shape_estimates_3d_{model}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / f"shape_estimates_3d_{model}.pdf")


if __name__ == "__main__":
    main()
