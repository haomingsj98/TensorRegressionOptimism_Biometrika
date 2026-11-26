"""UTKFace age prediction with Tucker tensor regression on 64 x 64 x 3 images.

Data: data/utkface_64.npz from prepare_utkface.py, pixels scaled to [0, 1]; response
y = log(1 + age). Each seed draws one random image-level split (7000 train / 3000 test);
pixels and y are centred with the training means (no intercept). Every candidate Tucker rank
is fitted to the same split as in ../TensorRegression (best of --starts TensorLy fits,
Newton polish on the Tucker chart). Estimated optimism is the general plug-in form
(2/n) min{sum y_i^2, tr(A_hat^-1 Omega_hat)}, Omega_hat = (1/n) sum_i e_i^2 J'x_i x_i'J.

    python utkface_regression.py --jobs 8

One task per (split, rank), largest rank first. Output (<output>/): raw/rep{rep}_r{j}.csv,
fits/rep{rep}_r{j}.npz (B_hat and the test indices), settings.json. Aggregate with
summarize_utkface.py.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "TensorRegression"))
EIG_TOL = 1e-8
DEFAULT_RANKS = ["1,1,1", "2,2,1", "2,2,2", "3,3,2", "3,3,3", "4,4,3", "5,5,3", "6,6,3",
                 "7,7,3", "8,8,3", "9,9,3"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ranks", nargs="+", default=DEFAULT_RANKS)
    p.add_argument("--data", default=str(ROOT / "data" / "utkface_64.npz"))
    p.add_argument("--test-frac", type=float, default=0.3)
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--rep-start", type=int, default=0)
    p.add_argument("--base-seed", type=int, default=2015)
    p.add_argument("--starts", type=int, default=3)
    p.add_argument("--tol", type=float, default=1e-10)
    p.add_argument("--maxiter", type=int, default=2000)
    p.add_argument("--polish-steps", type=int, default=100)
    p.add_argument("--gtol", type=float, default=1e-6)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--threads", type=int, default=None, help="BLAS threads per worker")
    p.add_argument("--output", default="results/tucker")
    return p.parse_args(argv)


ARGS = parse_args() if __name__ == "__main__" else None
_threads = str(ARGS.threads or (1 if ARGS.jobs > 1 else 4)) if ARGS else "1"
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, _threads)

import numpy as np
import pandas as pd
import tensorly as tl

import cp_regression as cpx
import tucker_regression as tkx


def load_data(args):
    z = np.load(args.data)
    X = z["X"].astype(np.float64) / 255.0
    y = np.log1p(z["age"].astype(np.float64))
    return X, y


def split(n_total, rep, args):
    """Random image-level split; depends only on (base seed, rep)."""
    rng = np.random.default_rng(np.random.SeedSequence([args.base_seed, rep]))
    perm = rng.permutation(n_total)
    n_test = int(round(args.test_frac * n_total))
    return np.sort(perm[n_test:]), np.sort(perm[:n_test])


def general_optimism(geo, y):
    """(2/n) min{sum y_i^2, tr(A_hat^-1 Omega_hat)} on the positive part of A_hat."""
    n, cap = len(y), float(y @ y)
    trace = float("nan")
    if geo["full_rank"]:
        lam, vec = np.linalg.eigh(geo["Aw"])
        Vp, lp = vec[:, lam > EIG_TOL], lam[lam > EIG_TOL]
        SW = geo["Z"] @ geo["Wh"] * geo["e"][:, None]
        trace = float(np.sum((Vp.T @ (SW.T @ SW / n) @ Vp).diagonal() / lp))
    est = 2 * min(cap, trace) / n if np.isfinite(trace) else float("nan")
    return dict(est_optimism=est, trace_score=trace,
                cap_active=bool(np.isfinite(trace) and trace > cap))


def run_task(rep, j, args):
    stem = f"rep{rep}_r{j}"
    out_file = ROOT / args.output / "raw" / f"{stem}.csv"
    if out_file.exists():
        return str(out_file), 0.0, True
    t0 = time.perf_counter()
    tl.set_backend("numpy")
    Xall, yall = load_data(args)
    cpx.SHAPE = tkx.SHAPE = tuple(Xall.shape[1:])
    tr_idx, te_idx = split(len(yall), rep, args)
    mu_X, mu_y = Xall[tr_idx].mean(0), yall[tr_idx].mean()
    X = Xall[tr_idx] - mu_X
    y = yall[tr_idx] - mu_y
    Xte2 = (Xall[te_idx] - mu_X).reshape(len(te_idx), -1)
    yte = yall[te_idx] - mu_y
    del Xall
    n = len(y)
    X2 = X.reshape(n, -1)
    R = tkx.parse_rank(args.ranks[j])
    init_seeds = np.random.SeedSequence([args.base_seed, rep, 100 + j]).generate_state(args.starts)
    init_seeds = [int(s % (2**31 - 1)) for s in init_seeds]
    d = tkx.tucker_dim(R)
    base = dict(rep=rep, rank=tkx.rank_label(R), rank_order=j, df=d, n_train=n, n_test=len(te_idx))
    fits = {}
    fit, iters, spread, failed = tkx.fit_tucker(X, X2, y, R, init_seeds, args)
    if fit is None:
        row = dict(base, flag="fit_failed", failed_starts=failed, seconds=time.perf_counter() - t0)
    else:
        G, U = fit
        geo0 = tkx.tucker_geometry(G, U, X2, y)
        tl_train, tl_stat = geo0["train"], tkx.stationarity(geo0)
        steps = 0
        if geo0["full_rank"]:
            G, U, steps = tkx.newton_polish(G, U, X2, y, args.polish_steps, args.gtol)
        Bhat = tkx.tucker_full(G, U)
        fits[tkx.rank_label(R)] = Bhat.astype(np.float32)
        g = tkx.plugin_optimism(G, U, X2, y)
        for k in ("est_optimism", "est_optimism_pd", "est_optimism_score", "cap_active",
                  "trace", "trace_pd", "trace_score"):
            g.pop(k)
        est = general_optimism(tkx.tucker_geometry(G, U, X2, y), y)
        train = g.pop("train_error")
        test = float(np.mean((yte - Xte2 @ Bhat.ravel()) ** 2))
        with np.errstate(divide="ignore"):
            logtrain = np.log(train)
        row = dict(
            base, flag=cpx.fit_flag(g, args.gtol), d_ge_n=d >= n,
            train_error=train, test_error=test, emp_optimism=test - train,
            est_optimism=est["est_optimism"], est_test_error=train + est["est_optimism"],
            aic=logtrain + 2 * d / n, bic=logtrain + np.log(n) * d / n,
            sel_logn=train + np.log(n) / 2 * est["est_optimism"],
            trace_score=est["trace_score"], cap_active=est["cap_active"],
            naive_dim_optimism=2 * train * g["chart_dimension"] / n,
            converged=g["stationarity_norm"] < args.gtol, **g,
            tensorly_iterations=iters, tensorly_hit_maxiter=iters >= args.maxiter,
            tensorly_train_error=tl_train, tensorly_stationarity=tl_stat, polish_steps=steps,
            failed_starts=failed, restart_loss_spread=spread, seconds=time.perf_counter() - t0)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    (out_file.parent.parent / "fits").mkdir(exist_ok=True)
    np.savez_compressed(out_file.parent.parent / "fits" / f"{stem}.npz", test_idx=te_idx, **fits)
    tmp = out_file.with_suffix(".tmp")
    pd.DataFrame([row]).to_csv(tmp, index=False)
    os.replace(tmp, out_file)
    return str(out_file), time.perf_counter() - t0, False


def main(args):
    tl.set_backend("numpy")
    args.ranks = [tkx.rank_label(tkx.parse_rank(r)) for r in args.ranks]
    Xall, yall = load_data(args)
    shape = tuple(Xall.shape[1:])
    cpx.SHAPE = tkx.SHAPE = shape
    outdir = ROOT / args.output
    (outdir / "raw").mkdir(parents=True, exist_ok=True)
    tr_idx, te_idx = split(len(yall), args.rep_start, args)
    dfs = {r: tkx.tucker_dim(tkx.parse_rank(r)) for r in args.ranks}
    settings = dict(vars(args), model="tucker", shape=shape, n_images=len(yall),
                    n_train=len(tr_idx), n_test=len(te_idx), df=dfs, var_y=float(yall.var()),
                    tensorly=tl.__version__, numpy=np.__version__, python=sys.version.split()[0])
    settings["data"] = Path(args.data).name
    pd.Series({k: str(v) for k, v in settings.items()}).to_json(outdir / "settings.json", indent=2)
    print(f"shape {shape}, {len(tr_idx)} train / {len(te_idx)} test, ranks {args.ranks}", flush=True)
    order = sorted(range(len(args.ranks)), key=lambda j: -dfs[args.ranks[j]])
    tasks = [(rep, j) for j in order for rep in range(args.rep_start, args.rep_start + args.reps)]
    cpx.run_all(tasks, run_task, args, jobs=args.jobs)


if __name__ == "__main__":
    main(ARGS)
