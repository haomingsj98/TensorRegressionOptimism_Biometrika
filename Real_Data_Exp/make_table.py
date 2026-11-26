"""Supplement table for UTKFace Tucker regression: mean and SE over splits per rank.

    python make_table.py

Columns: training, test and estimated test error; empirical and estimated optimism (general
plug-in form), on the log(1 + age) scale. Writes tables/utkface_table_tucker.csv (numeric) and
tables/utkface_table_tucker_formatted.csv ("mean ± SE", three decimals).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MEASURES = [("train_error", "Training error"), ("test_error", "Test error"),
            ("est_test_error", "Estimated test error"), ("emp_optimism", "Empirical optimism"),
            ("est_optimism", "Estimated optimism")]


def main():
    df = pd.read_csv(ROOT / "results" / "tucker" / "all_fits.csv", dtype={"rank": str})
    rows = []
    for d, g in df.groupby("df"):
        r = dict(rank=g["rank"].iloc[0])
        for col, _ in MEASURES:
            v = g[col].dropna()
            r[f"{col}_mean"], r[f"{col}_se"] = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
        rows.append(r)
    t = pd.DataFrame(rows)
    out = ROOT / "tables"
    out.mkdir(exist_ok=True)
    t.to_csv(out / "utkface_table_tucker.csv", index=False)
    f = pd.DataFrame({"Rank": "(" + t["rank"] + ")"})
    for col, name in MEASURES:
        f[name] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(t[f"{col}_mean"], t[f"{col}_se"])]
    f.to_csv(out / "utkface_table_tucker_formatted.csv", index=False, encoding="utf-8-sig")
    print(f.to_string(index=False))


if __name__ == "__main__":
    main()
