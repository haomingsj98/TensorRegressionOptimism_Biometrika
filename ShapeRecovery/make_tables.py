"""Supplement tables for the shape experiments: mean and SE over seeds per shape and rank.

    python make_tables.py --model cp

Columns: training, test and estimated test error; empirical and estimated optimism
(identity plug-in form). Writes tables/shape_table_<model>.csv (numeric) and
tables/shape_table_<model>_formatted.csv ("mean ± SE", one decimal). A NaN estimate
(degenerate fit) is dropped only for its own (shape, rank, seed).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SHAPES = ["square", "T", "cross", "disk", "triangle", "butterfly"]
MEASURES = [("train_error", "Training error"), ("test_error", "Test error"),
            ("est_test_error", "Estimated test error"), ("emp_optimism", "Empirical optimism"),
            ("est_optimism", "Identity estimate")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cp", "tucker"], required=True)
    model = p.parse_args().model
    df = pd.read_csv(ROOT / "results" / model / "all_fits.csv", dtype={"rank": str})
    rows = []
    for (shape, d), g in df.groupby(["shape", "df"]):
        r = dict(shape=shape, rank=g["rank"].iloc[0], d=d)
        for col, _ in MEASURES:
            v = g[col].dropna()
            r[f"{col}_mean"], r[f"{col}_se"] = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
        rows.append(r)
    t = pd.DataFrame(rows)
    t["shape"] = pd.Categorical(t["shape"], SHAPES, ordered=True)
    t = t.sort_values(["shape", "d"]).drop(columns="d").reset_index(drop=True)
    out = ROOT / "tables"
    out.mkdir(exist_ok=True)
    t.to_csv(out / f"shape_table_{model}.csv", index=False)
    f = t[["shape", "rank"]].rename(columns={"shape": "Shape", "rank": "Rank"})
    if model == "tucker":
        f["Rank"] = "(" + f["Rank"] + ")"
    for col, name in MEASURES:
        f[name] = [f"{m:.1f} ± {s:.1f}" for m, s in zip(t[f"{col}_mean"], t[f"{col}_se"])]
    f.to_csv(out / f"shape_table_{model}_formatted.csv", index=False, encoding="utf-8-sig")
    print(f.to_string(index=False))


if __name__ == "__main__":
    main()
