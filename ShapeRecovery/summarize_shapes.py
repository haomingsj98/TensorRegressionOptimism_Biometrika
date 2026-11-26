"""Aggregate shapes_regression.py output.

    python summarize_shapes.py --output results/tucker

Collects <output>/raw/*.csv into all_fits.csv (or reads all_fits.csv if raw/ is absent) and
writes summary.csv (mean, SD and SE over seeds per shape and rank), rank_selection.csv (rank
minimizing each seed-averaged criterion) and rank_selection_freq.csv (per-seed selections and
agreement with each seed's test-error-optimal rank). A NaN estimate is dropped only for its
own (shape, rank, seed).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SHAPES = ["square", "T", "cross", "disk", "triangle", "butterfly"]
KEYS = ["shape", "rank"]
FIELDS = ["train_error", "test_error", "emp_optimism", "est_optimism", "est_test_error",
          "est_optimism_score", "est_test_error_score", "aic", "bic", "sel_logn",
          "true_excess_risk"]
CRITERIA = ["test_error", "est_test_error", "aic", "bic", "sel_logn", "train_error"]
FLAGS = ["ok", "not_pd", "not_converged", "not_converged+not_pd", "degenerate", "fit_failed"]


def selection_frequencies(df, ranks):
    test_pick = df.loc[df.dropna(subset=["test_error"]).groupby(["shape", "rep"], observed=True)
                       ["test_error"].idxmin(), ["shape", "rep", "rank"]].set_index(["shape", "rep"])["rank"]
    rows = []
    for c in CRITERIA:
        d = df.dropna(subset=[c])
        picks = d.loc[d.groupby(["shape", "rep"], observed=True)[c].idxmin(),
                      ["shape", "rep", "rank", "true_rank"]]
        for s, g in picks.groupby("shape", sort=False, observed=True):
            counts = g["rank"].value_counts()
            row = dict(shape=s, criterion=c, seeds=len(g), true_rank=g["true_rank"].iloc[0])
            row.update({f"rank_{k}": int(counts.get(k, 0)) for k in ranks})
            row["prop_test_optimal"] = float(
                (g.set_index("rep")["rank"] == test_pick.loc[s].reindex(g["rep"]).values).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    out = ROOT / p.parse_args().output
    dtype = {"rank": str, "true_rank": str}
    files = sorted((out / "raw").glob("*_rep*.csv"))
    df = (pd.concat((pd.read_csv(f, dtype=dtype) for f in files), ignore_index=True) if files
          else pd.read_csv(out / "all_fits.csv", dtype=dtype))
    shapes = [s for s in SHAPES if s in set(df["shape"])]
    ranks = df.sort_values("rank_order")["rank"].unique().tolist()
    df["shape"] = pd.Categorical(df["shape"], shapes, ordered=True)
    df["rank"] = pd.Categorical(df["rank"], ranks, ordered=True)
    df = df.sort_values(["shape", "rep", "rank"]).reset_index(drop=True)
    if files:
        df.to_csv(out / "all_fits.csv", index=False)

    rows = []
    for (s, r), g in df.groupby(KEYS, observed=True):
        row = dict(shape=s, rank=r, true_rank=g["true_rank"].iloc[0], df=g["df"].iloc[0],
                   seeds=len(g))
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

    sel = summary.set_index(KEYS)[[c + "_mean" for c in CRITERIA]]
    sel = pd.DataFrame({c: sel[c + "_mean"].unstack("rank").idxmin(axis=1) for c in CRITERIA})
    sel.insert(0, "true_rank", summary.groupby("shape", observed=True)["true_rank"].first())
    sel.reset_index().to_csv(out / "rank_selection.csv", index=False)
    selection_frequencies(df, ranks).to_csv(out / "rank_selection_freq.csv", index=False)
    print(f"{len(df)} fits -> {out}")
    print("Rank minimizing the seed-averaged criterion:\n", sel.to_string())


if __name__ == "__main__":
    main()
