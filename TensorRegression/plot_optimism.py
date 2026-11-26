"""Figures for the synthetic experiments (seed averages, error bars = 1 SE).

    python plot_optimism.py --results results/cp_ar [--noise-ratios 0.05]

AR design (--design ar runs):
  optimism_test_error_bars_1se.pdf   empirical vs. estimated optimism and test error
                                     (general plug-in form)
  selection_criteria_rawQ.pdf        optimism selector, AIC and BIC against rank
Identity design:
  optimism_estimators_bars_1se.pdf   empirical optimism vs. the identity and general forms

--noise-ratios 0.05 keeps one noise row and adds the suffix _noise5. Figures go to
figures/<results name>/, with plotted_values.csv. A degenerate fit (NaN estimate) is
dropped for its own (noise, n, rank, seed).
"""
import argparse
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullLocator

ROOT = Path(__file__).resolve().parent
KEYS = ["noise_ratio", "n_train", "rank"]

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
OPT, ERR = "#2a78d6", "#eb6834"
OPT_LIGHT, ERR_LIGHT = "#d6e6f9", "#fbe0d5"
N_COLORS = ["#b3d0f5", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 14,
    "xtick.labelsize": 12.5, "ytick.labelsize": 12.5, "legend.fontsize": 13.5,
    "figure.labelsize": 16,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42,
})
BAR_FONTS = {"axes.titlesize": 18, "axes.labelsize": 18, "xtick.labelsize": 15.5,
             "ytick.labelsize": 15.5, "legend.fontsize": 18, "figure.labelsize": 21}


def seed_means(df, cols, interval="se"):
    g = df.groupby(KEYS)[cols]
    spread = g.std(ddof=1) if interval == "sd" else g.std(ddof=1) / np.sqrt(g.count())
    return g.mean(), spread, g.count()


def noise_label(r):
    return rf"noise {100 * r:g}%"


def three_decimals(v, _pos=None):
    s = str(Decimal(repr(v)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
    return s.rstrip("0").rstrip(".")


def rank_ticks(ax, labels, tucker, tucker_size=10.5):
    ax.set_xticks(np.arange(len(labels)))
    if tucker:
        ax.set_xticklabels([f"({l})" for l in labels], rotation=45, ha="right",
                           rotation_mode="anchor", fontsize=tucker_size)
    else:
        ax.set_xticklabels(labels)


def finish(fig, axes, xlabel, ylabel, handles, margins=(0.5, 0.45, 0.5), **legend_kw):
    W, H = fig.get_size_inches()
    fig.align_ylabels(axes[:, 0])
    left, bottom, top = margins
    fig.tight_layout(rect=(left / W, bottom / H, 1, 1 - top / H), h_pad=0.8, w_pad=0.6)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), **legend_kw)
    fig.supxlabel(xlabel, y=0.01, va="bottom")
    fig.supylabel(ylabel, x=0.01, ha="left")


def plot_bars(m, h, out, labels, tucker, suffix):
    bars = [("emp_optimism", OPT, OPT, "-"), ("est_optimism_score", OPT_LIGHT, OPT, "--"),
            ("test_error", ERR, ERR, "-"), ("est_test_error_score", ERR_LIGHT, ERR, "--")]
    handles = [Patch(facecolor=OPT, edgecolor=OPT, label="Empirical optimism"),
               Patch(facecolor=OPT_LIGHT, edgecolor=OPT, linestyle="--", label="Estimated optimism"),
               Patch(facecolor=ERR, edgecolor=ERR, label="Empirical test error"),
               Patch(facecolor=ERR_LIGHT, edgecolor=ERR, linestyle="--", label="Estimated test error")]
    draw_bars(m, h, out, labels, tucker, bars, handles, "Error measure",
              f"optimism_test_error_bars_1se{suffix}", log_note=False, legend_size=22)


def plot_estimator_bars(m, h, out, labels, tucker, suffix):
    bars = [("emp_optimism", OPT, OPT, "-"), ("est_optimism", OPT_LIGHT, OPT, "--"),
            ("est_optimism_score", ERR_LIGHT, ERR, "--")]
    handles = [Patch(facecolor=OPT, edgecolor=OPT, label="Empirical optimism"),
               Patch(facecolor=OPT_LIGHT, edgecolor=OPT, linestyle="--",
                     label="Estimate, identity-design form"),
               Patch(facecolor=ERR_LIGHT, edgecolor=ERR, linestyle="--",
                     label="Estimate, general form")]
    draw_bars(m, h, out, labels, tucker, bars, handles, "Optimism",
              f"optimism_estimators_bars_1se{suffix}")


def draw_bars(m, h, out, labels, tucker, bars, handles, ylabel, stem, tucker_size=13.5,
              panel_h=3.4, margins=(0.7, 0.6, 0.65), log_note=True, legend_size=None):
    with plt.rc_context(BAR_FONTS):
        ratios, ns = m.index.levels[0], m.index.levels[1]
        ranks = np.arange(len(labels))
        single = len(ratios) == 1
        panel_w = 3.55 if single else 0.55 * len(labels) + 1.0
        fig_h = panel_h * len(ratios) + (1.5 if single else 0)
        fig, axes = plt.subplots(len(ratios), len(ns), figsize=(panel_w * len(ns), fig_h),
                                 sharex=True, sharey="row", squeeze=False)
        width, gap = 0.76 / len(bars), 0.012
        offsets = (np.arange(len(bars)) - (len(bars) - 1) / 2) * width
        for i, r in enumerate(ratios):
            lo_row = np.nanmin(m.loc[r][[c for c, *_ in bars]].to_numpy())
            bottom = 10 ** np.floor(np.log10(lo_row) - 0.15)
            for j, n in enumerate(ns):
                ax = axes[i, j]
                mm, hh = m.loc[(r, n)].reindex(labels), h.loc[(r, n)].reindex(labels)
                for (col, face, edge, ls), dx in zip(bars, offsets):
                    y, e = mm[col].to_numpy(), hh[col].to_numpy()
                    ax.bar(ranks + dx, y - bottom, width - gap, bottom=bottom, color=face,
                           edgecolor=edge, linestyle=ls, linewidth=1.0, zorder=2)
                    lower = np.minimum(e, (y - bottom) * 0.999)
                    ax.errorbar(ranks + dx, y, yerr=[lower, e], fmt="none", ecolor=INK,
                                elinewidth=0.7, capsize=1.5, capthick=0.7, zorder=3)
                ax.set_yscale("log")
                ax.set_ylim(bottom=bottom)
                rank_ticks(ax, labels, tucker, tucker_size)
                ax.grid(axis="x", visible=False)
                if i == 0:
                    ax.set_title(f"$n = {n}$", color=INK)
                if j == 0 and len(ratios) > 1:
                    ax.set_ylabel(noise_label(r))
        handles = handles + [Line2D([], [], color=INK, lw=0.9, marker="_", markersize=7,
                                    label="mean $\\pm$ SE")]
        finish(fig, axes, "Tucker rank" if tucker else "CP rank",
               "Error measure" if single else ylabel + (" (log scale)" if log_note else ""),
               handles, margins=margins, ncol=len(handles), handlelength=1.6, columnspacing=1.8,
               fontsize=None if single else legend_size)
        fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)


def plot_criteria(m, out, true_rank, labels, tucker, suffix, sel):
    ratios, ns = m.index.levels[0], m.index.levels[1]
    colors = N_COLORS[-len(ns):]
    crits = [(sel, "Optimism selector"), ("aic", "AIC"), ("bic", "BIC")]
    single = len(ratios) == 1
    fig, axes = plt.subplots(len(ratios), 3,
                             figsize=(10.5 if single else 12.5, 2.85 * len(ratios) + (1.0 if single else 0)),
                             sharex=True, squeeze=False)
    for i, r in enumerate(ratios):
        for j, (col, title) in enumerate(crits):
            ax = axes[i, j]
            ax.axvline(labels.index(true_rank), color=INK2, lw=0.8, ls=":", zorder=1)
            for k, n in enumerate(ns):
                s = m.loc[(r, n), col].reindex(labels)
                s.index = np.arange(len(labels))
                ax.plot(s.index, s.values, color=colors[k], marker="o", ms=5.5,
                        lw=1.6, mec="white", mew=0.6, zorder=3)
                best = s.idxmin()
                ax.plot(best, s[best], marker="o", ms=12, mfc="none", mec=colors[k],
                        mew=1.3, zorder=4)
            if col == sel:
                ax.set_yscale("log", base=2)
                ax.yaxis.set_major_locator(LogLocator(base=2, subs=(1.0,), numticks=30))
                ax.yaxis.set_major_formatter(FuncFormatter(three_decimals))
                ax.yaxis.set_minor_locator(NullLocator())
            else:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            rank_ticks(ax, labels, tucker)
            ax.grid(axis="x", visible=False)
            if i == 0:
                ax.set_title(title, color=INK)
            if j == 0 and len(ratios) > 1:
                ax.set_ylabel(noise_label(r))
    handles = [Line2D([], [], color=c, marker="o", ms=5.5, lw=1.6, mec="white", mew=0.6,
                      label=f"$n = {n}$") for c, n in zip(colors, ns)]
    handles += [Line2D([], [], ls="none", marker="o", ms=11, mfc="none", mec=INK2, mew=1.3,
                       label="minimizing rank"),
                Line2D([], [], color=INK2, lw=0.8, ls=":",
                       label="true rank" if single else f"true rank ({true_rank})")]
    compact = (dict(handlelength=1.2, handletextpad=0.35, columnspacing=0.7, fontsize=12)
               if single else dict(handlelength=1.8, columnspacing=1.4))
    finish(fig, axes, "Tucker rank" if tucker else "CP rank", "Selection metric value", handles,
           ncol=8, **compact)
    fig.savefig(out / f"selection_criteria_rawQ{suffix}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="e.g. results/cp_ar")
    p.add_argument("--noise-ratios", type=float, nargs="*", default=[])
    args = p.parse_args()
    res = ROOT / args.results
    out = ROOT / "figures" / res.name
    out.mkdir(parents=True, exist_ok=True)
    settings = json.loads((res / "settings.json").read_text())
    tucker = settings["model"] == "tucker"
    ar = settings["design"] == "ar"
    true_rank = settings["true_rank"] if tucker else int(settings["true_rank"])

    df = pd.read_csv(res / "all_fits.csv", dtype={"rank": str} if tucker else None)
    suffix = ""
    if args.noise_ratios:
        df = df[np.isclose(df.noise_ratio.to_numpy()[:, None], args.noise_ratios).any(axis=1)]
        suffix = "_noise" + "_".join(f"{100 * r:g}" for r in args.noise_ratios)
    labels = (df.sort_values("rank_order")["rank"].unique().tolist() if tucker
              else sorted(df["rank"].unique().tolist()))
    dropped = int(df.est_optimism.isna().sum())
    df = df.dropna(subset=["est_optimism"])
    cols = ["emp_optimism", "est_optimism", "test_error", "est_test_error", "aic", "bic",
            "sel_logn", "est_optimism_score", "est_test_error_score"]
    if ar:
        cols += ["sel_logn_sigma"]
    m, h, cnt = seed_means(df, cols)

    if ar:
        plot_bars(m, h, out, labels, tucker, suffix)
        plot_criteria(m, out, true_rank, labels, tucker, suffix, "sel_logn_sigma")
    else:
        plot_estimator_bars(m, h, out, labels, tucker, suffix)
    sd = seed_means(df, cols, "sd")[1]
    table = m.join(h.add_suffix("_se")).join(sd.add_suffix("_sd")).join(
        cnt[["emp_optimism"]].rename(columns={"emp_optimism": "seeds_used"}))
    table.to_csv(out / f"plotted_values{suffix}.csv")
    print(f"dropped {dropped} degenerate fits of {dropped + len(df)}; figures in {out}")


if __name__ == "__main__":
    main()
