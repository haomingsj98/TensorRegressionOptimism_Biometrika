"""Aggregate utkface_regression.py output.

    python summarize_utkface.py --output results/tucker

Collects <output>/raw/*.csv into all_fits.csv (or reads all_fits.csv if raw/ is absent) and
writes summary.csv (mean, SD and SE over splits per rank), rank_selection.csv (rank
minimizing each split-averaged criterion) and rank_selection_freq.csv (per-split selections
and agreement with each split's test-error-optimal rank).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIELDS = ["train_error", "test_error", "emp_optimism", "est_optimism", "est_test_error",
          "aic", "bic", "sel_logn"]
CRITERIA = ["test_error", "est_test_error", "aic", "bic", "sel_logn", "train_error"]
FLAGS = ["ok", "not_pd", "not_converged", "not_converged+not_pd", "degenerate", "fit_failed"]


def selection_frequencies(df, ranks):
    test_pick = df.loc[df.dropna(subset=["test_error"]).groupby("rep")["test_error"].idxmin(),
                       ["rep", "rank"]].set_index("rep")["rank"]
    rows = []
    for c in CRITERIA:
        d = df.dropna(subset=[c])
        picks = d.loc[d.groupby("rep")[c].idxmin(), ["rep", "rank"]].set_index("rep")["rank"]
        counts = picks.value_counts()
        row = dict(criterion=c, splits=len(picks))
        row.update({f"rank_{k}": int(counts.get(k, 0)) for k in ranks})
        row["prop_test_optimal"] = float((picks == test_pick.reindex(picks.index)).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    out = ROOT / p.parse_args().output
    files = sorted((out / "raw").glob("rep*.csv"))
    df = (pd.concat((pd.read_csv(f, dtype={"rank": str}) for f in files), ignore_index=True)
          if files else pd.read_csv(out / "all_fits.csv", dtype={"rank": str}))
    ranks = df.sort_values("rank_order")["rank"].unique().tolist()
    df["rank"] = pd.Categorical(df["rank"], ranks, ordered=True)
    df = df.sort_values(["rep", "rank"]).reset_index(drop=True)
    if files:
        df.to_csv(out / "all_fits.csv", index=False)

    rows = []
    for r, g in df.groupby("rank", observed=True):
        row = dict(rank=r, df=g["df"].iloc[0], splits=len(g))
        row.update({f"flag_{f}": int((g.flag == f).sum()) for f in FLAGS})
        for f in FIELDS:
            v = g[f].dropna()
            row[f + "_mean"] = v.mean()
            row[f + "_sd"] = v.std(ddof=1)
            row[f + "_se"] = v.std(ddof=1) / np.sqrt(len(v))
        paired = g.dropna(subset=["est_optimism"])
        row["opt_ratio_of_means"] = paired.est_optimism.mean() / paired.emp_optimism.mean()
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)

    means = summary.set_index("rank")
    sel = pd.Series({c: means[c + "_mean"].idxmin() for c in CRITERIA}, name="selected_rank")
    sel.rename_axis("criterion").reset_index().to_csv(out / "rank_selection.csv", index=False)
    selection_frequencies(df, ranks).to_csv(out / "rank_selection_freq.csv", index=False)
    print(f"{len(df)} fits -> {out}")
    print("Rank minimizing the split-averaged criterion:\n", sel.to_string())


if __name__ == "__main__":
    main()
