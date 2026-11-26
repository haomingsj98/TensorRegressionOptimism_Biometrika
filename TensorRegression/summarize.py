"""Aggregate cp_regression.py / tucker_regression.py output.

    python summarize.py --output results/cp_ar

Collects <output>/raw/*.csv into all_fits.csv (or reads all_fits.csv if raw/ is absent) and
writes summary.csv (mean, SD and SE over seeds per noise ratio, n and rank),
rank_selection.csv (rank minimizing each seed-averaged criterion) and
rank_selection_freq.csv (per-seed selections). A NaN estimate (degenerate fit) is dropped
only for its own (noise ratio, n, rank, seed).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
KEYS = ["noise_ratio", "n_train", "rank"]
FIELDS = ["train_error", "test_error", "emp_optimism", "est_optimism", "est_test_error",
          "est_optimism_score", "est_test_error_score", "aic", "bic", "sel_logn",
          "true_excess_risk"]
CRITERIA = ["test_error", "est_test_error", "est_test_error_score", "aic", "bic", "sel_logn",
            "train_error"]
AR_FIELDS = ["est_optimism_sigma", "est_test_error_sigma", "sel_logn_sigma"]
AR_CRITERIA = ["est_test_error_sigma", "sel_logn_sigma"]
FLAGS = ["ok", "not_pd", "not_converged", "not_converged+not_pd", "degenerate", "fit_failed"]


def selection_frequencies(df, true_rank, criteria):
    ranks = sorted(df["rank"].unique())
    rows = []
    for c in criteria:
        d = df.dropna(subset=[c])
        picks = d.loc[d.groupby(KEYS[:2] + ["rep"])[c].idxmin(), KEYS]
        for (r, n), g in picks.groupby(KEYS[:2]):
            counts = g["rank"].value_counts()
            row = dict(noise_ratio=r, n_train=n, criterion=c, seeds=len(g))
            row.update({f"rank{k}": int(counts.get(k, 0)) for k in ranks})
            row["prop_true_rank"] = counts.get(true_rank, 0) / len(g)
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    out = ROOT / p.parse_args().output
    settings = json.loads((out / "settings.json").read_text())
    tucker = settings["model"] == "tucker"
    dtype = {"rank": str} if tucker else None
    files = sorted((out / "raw").glob("noise*_n*_rep*.csv"))
    if files:
        df = pd.concat((pd.read_csv(f, dtype=dtype) for f in files), ignore_index=True)
        df = df.sort_values(KEYS[:2] + ["rep", "rank"]).reset_index(drop=True)
        df.to_csv(out / "all_fits.csv", index=False)
    else:
        df = pd.read_csv(out / "all_fits.csv", dtype=dtype)
    true_rank = settings["true_rank"] if tucker else int(settings["true_rank"])
    ar = settings["design"] == "ar"
    fields = FIELDS + (AR_FIELDS if ar else [])
    criteria = CRITERIA + (AR_CRITERIA if ar else [])

    rows = []
    for key, g in df.groupby(KEYS):
        row = dict(zip(KEYS, key), seeds=len(g))
        if tucker:
            row["df"] = g["df"].iloc[0]
        row.update({f"flag_{f}": int((g.flag == f).sum()) for f in FLAGS})
        for f in fields:
            v = g[f].dropna()
            row[f + "_mean"] = v.mean()
            row[f + "_sd"] = v.std(ddof=1)
            row[f + "_se"] = v.std(ddof=1) / np.sqrt(len(v))
        paired = g.dropna(subset=["est_optimism"])
        row["opt_ratio_of_means"] = paired.est_optimism.mean() / paired.emp_optimism.mean()
        for form in (["score", "sigma"] if ar else ["score"]):
            pf = g.dropna(subset=[f"est_optimism_{form}"])
            row[f"opt_ratio_{form}"] = pf[f"est_optimism_{form}"].mean() / pf.emp_optimism.mean()
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)

    sel = summary.set_index(KEYS)[[c + "_mean" for c in criteria]]
    sel = pd.DataFrame({c: sel[c + "_mean"].unstack("rank").idxmin(axis=1)
                        for c in criteria}).reset_index()
    if not tucker:
        sel[criteria] = sel[criteria].astype("Int64")
    sel.to_csv(out / "rank_selection.csv", index=False)
    selection_frequencies(df, true_rank, criteria).to_csv(out / "rank_selection_freq.csv",
                                                          index=False)
    print(f"{len(df)} fits -> {out}")
    print("Rank minimizing the seed-averaged criterion:\n", sel.to_string(index=False))


if __name__ == "__main__":
    main()
