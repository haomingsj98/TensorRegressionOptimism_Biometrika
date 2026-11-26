"""Test error, AIC, BIC and the optimism selector against Tucker rank (mean +- 1 SE over splits).

    python plot_utkface_criteria.py

Reads results/tucker/all_fits.csv. AIC and BIC are drawn as exp(AIC) = L_n e^{2d/n} and
exp(BIC) = L_n n^{d/n} (same minimizers). Large dots mark the minimizing ranks; the dotted
line marks the test-error optimal rank. Writes figures/criteria_tucker_single_exp_linear.pdf.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MEASURES = [("test_error", "Empirical test error", "tab:red"),
            ("aic", "AIC", "tab:purple"),
            ("bic", "BIC", "tab:green"),
            ("sel_logn", "Optimism selector", "tab:blue")]
INK2 = "#52514e"


def summarize(df, order, col):
    g = df.groupby("rank")[col]
    s = pd.DataFrame(dict(mean=g.mean(), sd=g.std(), n=g.count())).reindex(order)
    s["se"] = s["sd"] / np.sqrt(s["n"])
    return s


def main():
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    df = pd.read_csv(ROOT / "results" / "tucker" / "all_fits.csv", dtype={"rank": str})
    order = df.groupby("rank")["df"].first().sort_values().index.tolist()
    df = df.assign(aic=np.exp(df["aic"]), bic=np.exp(df["bic"]))
    stats = {col: summarize(df, order, col) for col, _, _ in MEASURES}
    x = np.arange(len(order))
    test_best = int(np.nanargmin(stats["test_error"]["mean"].values))
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.axvline(test_best, color=INK2, lw=1.0, ls=":", zorder=1)
    handles = []
    for col, name, color in MEASURES:
        s = stats[col]
        best = int(np.nanargmin(s["mean"].values))
        ax.errorbar(x, s["mean"], yerr=s["se"], color=color, marker="o", ms=5.5, lw=2,
                    capsize=3, elinewidth=1.2, zorder=3)
        ax.plot(x[best], s["mean"].iloc[best], "o", color=color, ms=11, mec="black",
                mew=1.0, zorder=4)
        handles.append(plt.Line2D([], [], color=color, marker="o", ms=5.5, lw=2, label=name))
        print(f"{name:22s} minimized at [{order[best]}]")
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax.set_xticks(x)
    ax.set_xticklabels([f"[{r}]" for r in order], rotation=45, ha="right")
    ax.set_xlabel("Tucker rank")
    ax.set_ylabel("Selection measure")
    handles += [plt.Line2D([], [], color="grey", marker="o", ms=11, mec="black", lw=0,
                           label="Minimizing rank"),
                plt.Line2D([], [], color=INK2, lw=1.0, ls=":", label="Test-error optimal rank")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, -0.01), handlelength=1.8, columnspacing=1.5)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "criteria_tucker_single_exp_linear.pdf", dpi=200, bbox_inches="tight")
    print("wrote", out / "criteria_tucker_single_exp_linear.pdf")


if __name__ == "__main__":
    main()
