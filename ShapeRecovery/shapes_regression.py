"""Shape recovery: CP or Tucker regression on 64 x 64 x 3 colour shape signals.

Signals: the six 64 x 64 binary shapes of Zhou, Li and Zhu (2013) (data/zhou_shapes_64.npz),
drawn as a yellow shape (1, 0.85, 0) on a blue background (0, 0.2, 0.8):
B = S o (fg - bg) + J o bg. Model y = <X, B> + eps with i.i.d. N(0, 1) covariates and noise
variance = --noise-ratio * ||S o (fg - bg)||_F^2. For every (shape, seed) each candidate rank
is fitted to the same data as in ../TensorRegression (best of --starts TensorLy fits, Newton
polish, plug-in optimism); Tucker candidates are (R, R, R).

    python shapes_regression.py --model cp --jobs 8
    python shapes_regression.py --model tucker --jobs 8

Output (<output>/): raw/{shape}_rep{rep}.csv, fits/{shape}_rep{rep}.npz (fitted B_hat per
rank), truth.npz, settings.json. Aggregate with summarize_shapes.py.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "TensorRegression"))
SHAPES = ["square", "T", "cross", "disk", "triangle", "butterfly"]
SHAPE = (64, 64, 3)
FG, BG = (1.0, 0.85, 0.0), (0.0, 0.2, 0.8)
DIM = 3                                             # part of the random-stream key


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["cp", "tucker"], required=True)
    p.add_argument("--shapes", nargs="+", default=SHAPES)
    p.add_argument("--ranks", nargs="+", default=["1", "2", "3"],
                   help="CP ranks, or Tucker ranks as R (meaning R,R,R) or 'a,b,c'")
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--test-size", type=int, default=1000)
    p.add_argument("--noise-ratio", type=float, default=0.05)
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--rep-start", type=int, default=0)
    p.add_argument("--base-seed", type=int, default=2013)
    p.add_argument("--starts", type=int, default=5)
    p.add_argument("--tol", type=float, default=1e-10)
    p.add_argument("--maxiter", type=int, default=2000)
    p.add_argument("--polish-steps", type=int, default=100)
    p.add_argument("--gtol", type=float, default=1e-6)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--threads", type=int, default=None, help="BLAS threads per worker")
    p.add_argument("--output", default=None, help="default results/<model>")
    args = p.parse_args(argv)
    args.output = args.output or f"results/{args.model}"
    return args


ARGS = parse_args() if __name__ == "__main__" else None
_threads = str(ARGS.threads or (1 if ARGS.jobs > 1 else 4)) if ARGS else "1"
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, _threads)

import numpy as np
import pandas as pd
import tensorly as tl

import cp_regression as cpx
import tucker_regression as tkx

cpx.SHAPE = tkx.SHAPE = SHAPE


def load_shapes():
    return {k: v.astype(np.float64)
            for k, v in np.load(ROOT / "data" / "zhou_shapes_64.npz").items()}


def make_signal(S):
    """Coefficient tensor and the energy of its shape part (the noise reference)."""
    fg, bg = np.array(FG), np.array(BG)
    B = S[..., None] * fg + (1 - S[..., None]) * bg
    return B, float(np.sum(S ** 2) * np.sum((fg - bg) ** 2))


def true_rank(S, model):
    r = int(np.linalg.matrix_rank(S))
    return str(r + 1) if model == "cp" else f"{r + 1},{r + 1},2"


def parse_rank(label, model):
    if model == "cp":
        return int(label)
    vals = tuple(int(v) for v in str(label).split(","))
    return vals * 3 if len(vals) == 1 else vals


def rank_label(R):
    return str(R) if isinstance(R, int) else ",".join(map(str, R))


def fit_candidate(R, X, X2, y, init_seeds, args):
    """(B_hat before and after the polish, plug-in dict, diagnostics) for one rank."""
    failed = 0
    if args.model == "cp":
        fs, iters, spread = cpx.fit_cp(X, X2, y, R, init_seeds, args)
        geo0 = cpx.cp_geometry(fs, X2, y)
        tl_train, tl_stat = geo0["train"], cpx.stationarity(geo0)
        B0 = cpx.cp_full(fs)
        fs, steps = cpx.newton_polish(fs, X2, y, args.polish_steps, args.gtol)
        Bhat = cpx.cp_full(fs)
        g = cpx.plugin_optimism(fs, X2, y)
        g.pop("hessian_eigs")
        df = R * (sum(SHAPE) - len(SHAPE) + 1)
    else:
        fit, iters, spread, failed = tkx.fit_tucker(X, X2, y, R, init_seeds, args)
        if fit is None:
            return None, None, dict(failed_starts=failed)
        G, U = fit
        geo0 = tkx.tucker_geometry(G, U, X2, y)
        tl_train, tl_stat = geo0["train"], tkx.stationarity(geo0)
        B0 = tkx.tucker_full(G, U)
        steps = 0
        if geo0["full_rank"]:
            G, U, steps = tkx.newton_polish(G, U, X2, y, args.polish_steps, args.gtol)
        Bhat = tkx.tucker_full(G, U)
        g = tkx.plugin_optimism(G, U, X2, y)
        df = tkx.tucker_dim(R)
    diag = dict(df=df, tensorly_iterations=iters, tensorly_hit_maxiter=iters >= args.maxiter,
                tensorly_train_error=tl_train, tensorly_stationarity=tl_stat, polish_steps=steps,
                failed_starts=failed, restart_loss_spread=spread)
    return (B0, Bhat), g, diag


def run_task(shape, rep, args):
    out_file = ROOT / args.output / "raw" / f"{shape}_rep{rep}.csv"
    if out_file.exists():
        return str(out_file), 0.0, True
    t0 = time.perf_counter()
    tl.set_backend("numpy")
    S = load_shapes()[shape]
    B, energy = make_signal(S)
    sigma = float(np.sqrt(args.noise_ratio * energy))
    k = SHAPES.index(shape)
    train_ss, test_ss = np.random.SeedSequence([args.base_seed, DIM, k, rep]).spawn(2)
    rng = np.random.default_rng(train_ss)
    X2 = rng.standard_normal((args.n_train, B.size))
    y = X2 @ B.ravel() + sigma * rng.standard_normal(args.n_train)
    rng = np.random.default_rng(test_ss)
    Xte2 = rng.standard_normal((args.test_size, B.size))
    yte = Xte2 @ B.ravel() + sigma * rng.standard_normal(args.test_size)
    X = X2.reshape(args.n_train, *SHAPE)
    rows, fits = [], {}
    for j, label in enumerate(args.ranks):
        R = parse_rank(label, args.model)
        tr = time.perf_counter()
        init_seeds = np.random.SeedSequence([args.base_seed, DIM, k, rep, 100 + j]).generate_state(args.starts)
        init_seeds = [int(s % (2**31 - 1)) for s in init_seeds]
        Bs, g, diag = fit_candidate(R, X, X2, y, init_seeds, args)
        base = dict(shape=shape, model=args.model, rep=rep, rank=rank_label(R), rank_order=j,
                    true_rank=true_rank(S, args.model), n_train=args.n_train,
                    n_test=args.test_size, noise_ratio=args.noise_ratio, noise_sd=sigma,
                    signal_norm2=float(np.sum(B ** 2)), shape_energy=energy)
        if Bs is None:
            rows.append(dict(base, flag="fit_failed", **diag, seconds=time.perf_counter() - tr))
            continue
        B0, Bhat = Bs
        fits[rank_label(R)] = Bhat.astype(np.float32)
        test = float(np.mean((yte - Xte2 @ Bhat.ravel()) ** 2))
        tl_test = float(np.mean((yte - Xte2 @ B0.ravel()) ** 2))
        train = g.pop("train_error")
        est = {k_: g.pop(k_) for k_ in ("est_optimism", "est_optimism_pd", "est_optimism_score")}
        df = diag["df"]
        with np.errstate(divide="ignore"):
            logtrain = np.log(train)
        rows.append(dict(
            base, flag=cpx.fit_flag(g, args.gtol), df=df, d_ge_n=df >= args.n_train,
            train_error=train, test_error=test, emp_optimism=test - train,
            est_optimism=est["est_optimism"], est_test_error=train + est["est_optimism"],
            aic=logtrain + 2 * df / args.n_train,
            bic=logtrain + np.log(args.n_train) * df / args.n_train,
            sel_logn=train + np.log(args.n_train) / 2 * est["est_optimism"],
            est_optimism_pd=est["est_optimism_pd"],
            est_test_error_pd=train + est["est_optimism_pd"],
            est_optimism_score=est["est_optimism_score"],
            est_test_error_score=train + est["est_optimism_score"],
            naive_dim_optimism=2 * train * g["chart_dimension"] / args.n_train,
            true_excess_risk=float(np.sum((Bhat - B) ** 2)),
            converged=g["stationarity_norm"] < args.gtol, **g,
            **{k_: v for k_, v in diag.items() if k_ != "df"},
            tensorly_test_error=tl_test, seconds=time.perf_counter() - tr))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    (out_file.parent.parent / "fits").mkdir(exist_ok=True)
    np.savez_compressed(out_file.parent.parent / "fits" / f"{shape}_rep{rep}.npz", **fits)
    tmp = out_file.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, out_file)
    return str(out_file), time.perf_counter() - t0, False


def main(args):
    tl.set_backend("numpy")
    args.ranks = [rank_label(parse_rank(r, args.model)) for r in args.ranks]
    outdir = ROOT / args.output
    (outdir / "raw").mkdir(parents=True, exist_ok=True)
    shapes = load_shapes()
    truth, info = {}, {}
    for name in args.shapes:
        B, energy = make_signal(shapes[name])
        truth[name] = B
        info[name] = dict(true_rank=true_rank(shapes[name], args.model),
                          noise_sd=float(np.sqrt(args.noise_ratio * energy)),
                          shape_energy=energy, signal_norm2=float(np.sum(B ** 2)))
    np.savez_compressed(outdir / "truth.npz", **truth)
    settings = dict(vars(args), shape=SHAPE, shapes_info=info, fg=FG, bg=BG,
                    tensorly=tl.__version__, numpy=np.__version__, python=sys.version.split()[0])
    pd.Series({k: str(v) for k, v in settings.items()}).to_json(outdir / "settings.json", indent=2)
    print(f"{args.model} ranks {args.ranks}; "
          + "; ".join(f"{k}: true rank {v['true_rank']}, noise SD {v['noise_sd']:.3f}"
                      for k, v in info.items()), flush=True)
    tasks = [(s, rep) for s in sorted(args.shapes, key=lambda s: -np.linalg.matrix_rank(shapes[s]))
             for rep in range(args.rep_start, args.rep_start + args.reps)]
    cpx.run_all(tasks, run_task, args, jobs=args.jobs)


if __name__ == "__main__":
    main(ARGS)
