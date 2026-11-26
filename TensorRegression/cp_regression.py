"""CP tensor regression: empirical vs. plug-in optimism across candidate ranks.

10 x 8 x 12 Gaussian covariates with identity covariance (--design identity) or AR(1)
covariance rho^|a-b| on vec(X) (--design ar, --rho 0.5); true CP rank 3 (orthonormal
factors, weights 1, 0.7, 0.5); noise variance = ratio * Var(<X, B>). For every
(noise ratio, n, seed) all candidate ranks are fitted to the same training sample
(TensorLy CPRegressor, best of --starts random starts, then a Newton polish on an
identified chart) and evaluated on an independent test sample.

    python cp_regression.py --design identity --jobs 8
    python cp_regression.py --design ar --jobs 8

Output: <output>/raw/noise{ratio}_n{n}_rep{rep}.csv (finished tasks are skipped),
truth.npy, settings.json. Aggregate with summarize.py.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

SHAPE = (10, 8, 12)
ROOT = Path(__file__).resolve().parent
EIG_TOL = 1e-8


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--design", choices=["identity", "ar"], default="identity")
    p.add_argument("--rho", type=float, default=0.5, help="AR(1) correlation (--design ar)")
    p.add_argument("--n-train", type=int, nargs="+", default=[400, 600, 800, 1200, 1600])
    p.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--rep-start", type=int, default=0)
    p.add_argument("--test-size", type=int, default=5000)
    p.add_argument("--true-rank", type=int, default=3)
    p.add_argument("--truth-seed", type=int, default=123)
    p.add_argument("--weights", type=float, nargs="+", default=[1.0, 0.7, 0.5])
    p.add_argument("--noise-ratios", type=float, nargs="+",
                   default=[0.01, 0.03, 0.05, 0.07, 0.10])
    p.add_argument("--base-seed", type=int, default=2026)
    p.add_argument("--starts", type=int, default=5)
    p.add_argument("--tol", type=float, default=1e-10)
    p.add_argument("--maxiter", type=int, default=2000)
    p.add_argument("--polish-steps", type=int, default=100)
    p.add_argument("--gtol", type=float, default=1e-6)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--threads", type=int, default=None, help="BLAS threads per worker")
    p.add_argument("--output", default=None, help="default results/cp_<design>")
    args = p.parse_args(argv)
    args.output = args.output or f"results/cp_{args.design}"
    return args


ARGS = parse_args() if __name__ == "__main__" else None
_threads = str(ARGS.threads or (1 if ARGS.jobs > 1 else 4)) if ARGS else "1"
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, _threads)

import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.cp_tensor import cp_to_tensor
from tensorly.random import random_cp
from tensorly.regression.cp_regression import CPRegressor

import ar_design as ard


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
def make_truth(rank, seed, weights):
    _, factors = random_cp(shape=SHAPE, rank=rank, full=False, orthogonal=True,
                           random_state=seed)
    return np.asarray(cp_to_tensor((np.asarray(weights, dtype=np.float64), factors)),
                      dtype=np.float64)


def simulate(n, B, sigma, rng, args):
    X = rng.standard_normal((n, *SHAPE))
    if args.design == "ar":
        X = ard.transform(X, args.rho)
    y = X.reshape(n, -1) @ B.ravel() + sigma * rng.standard_normal(n)
    return X, y


def task_rngs(base_seed, n, rep):
    train_ss, test_ss = np.random.SeedSequence([base_seed, n, rep]).spawn(2)
    return np.random.default_rng(train_ss), np.random.default_rng(test_ss)


def noise_sd(ratio, B, args):
    var = ard.signal_var(B, args.rho) if args.design == "ar" else float(np.sum(B ** 2))
    return float(np.sqrt(ratio * var))


# ----------------------------------------------------------------------------
# CP helpers
# ----------------------------------------------------------------------------
def cp_full(fs):
    return np.einsum("ir,jr,kr->ijk", *fs, optimize=True)


def cp_balance(fs):
    norms = [np.linalg.norm(u, axis=0) for u in fs]
    scale = np.cbrt(np.prod(norms, axis=0))
    return [u / np.maximum(v, 1e-300) * scale for u, v in zip(fs, norms)]


def cp_pack(fs):
    return np.concatenate([u.ravel() for u in fs])


def cp_unpack(theta, R):
    cuts = np.cumsum([0] + [s * R for s in SHAPE])
    return [theta[cuts[m]:cuts[m + 1]].reshape(SHAPE[m], R) for m in range(3)]


# ----------------------------------------------------------------------------
# Jacobian and Hessian on an identified chart
# ----------------------------------------------------------------------------
def cp_geometry(fs, X2, y):
    """Jacobian J, whitened Hessian A_hat and gradient on the chart orthogonal to the
    2R scaling directions. A_hat = Z'Z/n - D^2 <h(theta), C>, Z = X J, C = X'e/n."""
    U, V, W = fs
    (I, J_, K), R = SHAPE, U.shape[1]
    n = len(y)
    e = y - X2 @ cp_full(fs).ravel()
    C = (X2.T @ e / n).reshape(SHAPE)

    JU = np.einsum("ai,br,cr->abcir", np.eye(I), V, W).reshape(-1, I * R)
    JV = np.einsum("ar,bj,cr->abcjr", U, np.eye(J_), W).reshape(-1, J_ * R)
    JW = np.einsum("ar,br,ck->abckr", U, V, np.eye(K)).reshape(-1, K * R)
    Jac = np.hstack([JU, JV, JW])
    p = Jac.shape[1]

    eyeR = np.eye(R)
    sU, sV = slice(0, I * R), slice(I * R, (I + J_) * R)
    sW = slice((I + J_) * R, p)
    H = np.zeros((p, p))
    H[sU, sV] = np.einsum("ijc,cr,rs->irjs", C, W, eyeR).reshape(I * R, J_ * R)
    H[sU, sW] = np.einsum("ibk,br,rs->irks", C, V, eyeR).reshape(I * R, K * R)
    H[sV, sW] = np.einsum("ajk,ar,rs->jrks", C, U, eyeR).reshape(J_ * R, K * R)
    H = H + H.T

    N = np.zeros((p, 2 * R))
    for r in range(R):
        N[sU, 2 * r][r::R] = U[:, r]
        N[sV, 2 * r][r::R] = -V[:, r]
        N[sU, 2 * r + 1][r::R] = U[:, r]
        N[sW, 2 * r + 1][r::R] = -W[:, r]
    Q = np.linalg.qr(N, mode="complete")[0][:, 2 * R:]

    JQ = Jac @ Q
    ZQ = X2 @ JQ
    G = JQ.T @ JQ
    ge, gv = np.linalg.eigh(G)
    full_rank = bool((ge > ge[-1] * 1e-10).sum() == Q.shape[1])
    Wh = (gv / np.sqrt(np.maximum(ge, 1e-300))) @ gv.T
    Aw = Wh @ (ZQ.T @ ZQ / n - Q.T @ H @ Q) @ Wh
    return dict(e=e, train=float(e @ e / n), Q=Q, JQ=JQ, ZQ=ZQ, Wh=Wh,
                Aw=(Aw + Aw.T) / 2, grad=JQ.T @ C.ravel(),
                full_rank=full_rank, jacobian_condition=float(ge[-1] / max(ge[0], 1e-300)))


def stationarity(geo):
    return float(np.linalg.norm(geo["Wh"] @ geo["grad"]))


def newton_polish(fs, X2, y, max_steps, gtol):
    """Damped saddle-free Newton steps; only training-loss decreases are accepted."""
    R = fs[0].shape[1]
    fs = cp_balance(fs)
    start_loss = float(np.mean((y - X2 @ cp_full(fs).ravel()) ** 2))
    steps = 0
    for _ in range(max_steps):
        geo = cp_geometry(fs, X2, y)
        gw = geo["Wh"] @ geo["grad"]
        if not geo["full_rank"] or np.linalg.norm(gw) < gtol:
            break
        lam, vec = np.linalg.eigh(geo["Aw"])
        lam = np.maximum(np.abs(lam), 1e-3 * np.abs(lam).max())
        dz = geo["Wh"] @ (vec @ ((vec.T @ gw) / lam))
        Bnorm = np.linalg.norm(cp_full(fs))
        dz *= min(1.0, 0.1 * Bnorm / max(np.linalg.norm(geo["JQ"] @ dz), 1e-300))
        dtheta = geo["Q"] @ dz
        slope = float(geo["grad"] @ dz)
        theta, half = cp_pack(fs), geo["train"] / 2
        for k in range(30):
            t = 0.5 ** k
            trial = cp_unpack(theta + t * dtheta, R)
            e = y - X2 @ cp_full(trial).ravel()
            if 0.5 * np.mean(e ** 2) <= half - 1e-4 * t * slope:
                fs, steps = cp_balance(trial), steps + 1
                break
        else:
            break
    final_loss = float(np.mean((y - X2 @ cp_full(fs).ravel()) ** 2))
    assert final_loss <= start_loss * (1 + 1e-12)
    return fs, steps


def plugin_optimism(fs, X2, y):
    """Plug-in optimism at a fitted CP tensor.

    est_optimism        identity form (2/n) min{sum y_i^2, L_n tr(A_hat^-1 J'J)}
    est_optimism_score  general form  (2/n) tr(A_hat^-1 Omega_hat),
                        Omega_hat = (1/n) sum_i e_i^2 J'x_i x_i'J
    Eigenvalues of A_hat <= 1e-8 (in the J'J metric) are dropped; NaN if J is rank deficient.
    """
    n = len(y)
    geo = cp_geometry(fs, X2, y)
    train = geo["train"]
    nan = float("nan")
    out = dict(train_error=train, chart_dimension=geo["Q"].shape[1],
               jacobian_full_rank=geo["full_rank"],
               jacobian_condition=geo["jacobian_condition"],
               stationarity_norm=stationarity(geo), hessian_pd=False, n_negative_eigs=-1,
               hessian_min_eig=nan, hessian_min_pos_eig=nan, hessian_condition=nan,
               trace=nan, trace_pd=nan, trace_score=nan, hessian_eigs=None)
    if geo["full_rank"]:
        lam, vec = np.linalg.eigh(geo["Aw"])
        pos = lam > EIG_TOL
        ZW = geo["ZQ"] @ geo["Wh"] * geo["e"][:, None]
        Ow = ZW.T @ ZW / n
        Vp = vec[:, pos]
        out.update(hessian_eigs=lam, hessian_min_eig=float(lam[0]),
                   hessian_min_pos_eig=float(lam[pos].min()) if pos.any() else nan,
                   hessian_condition=float(lam[-1] / lam[0]),
                   n_negative_eigs=int((lam < -EIG_TOL).sum()),
                   hessian_pd=bool(pos.all()),
                   trace=float(np.sum(1 / lam[pos])),
                   trace_score=float(np.sum((Vp.T @ Ow @ Vp).diagonal() / lam[pos])))
        out["trace_pd"] = out["trace"] if out["hessian_pd"] else nan
    out["est_optimism"] = (2 * min(float(y @ y), train * out["trace"]) / n
                           if np.isfinite(out["trace"]) else nan)
    out["est_optimism_pd"] = 2 * train * out["trace_pd"] / n
    out["est_optimism_score"] = 2 * out["trace_score"] / n
    out["cap_active"] = bool(train * out["trace"] > float(y @ y))
    return out


# ----------------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------------
def to_numpy(x):
    return np.asarray(tl.to_numpy(x), dtype=np.float64)


def fit_cp(X, X2, y, rank, init_seeds, args):
    """CPRegressor from several random starts; keep the smallest training MSE."""
    best, losses = None, []
    for s in init_seeds:
        est = CPRegressor(weight_rank=rank, tol=args.tol, reg_W=0,
                          n_iter_max=args.maxiter, random_state=int(s), verbose=0)
        est.fit(X, y)
        w, fs = est.cp_weight_
        fs = [to_numpy(f) for f in fs]
        fs[0] = fs[0] * to_numpy(w)[None, :]
        loss = float(np.mean((y - X2 @ cp_full(fs).ravel()) ** 2))
        losses.append(loss)
        if np.isfinite(loss) and (best is None or loss < best[0]):
            best = (loss, fs, est.n_iterations_)
    if best is None:
        raise FloatingPointError(f"All CPRegressor starts failed at rank {rank}")
    loss, fs, iters = best
    return cp_balance(fs), iters, float(np.max(losses) - loss)


def fit_flag(g, gtol):
    if not g["jacobian_full_rank"]:
        return "degenerate"
    issues = [name for name, bad in (("not_converged", g["stationarity_norm"] >= gtol),
                                     ("not_pd", not g["hessian_pd"])) if bad]
    return "+".join(issues) or "ok"


def run_task(ratio, n, rep, args, B):
    out_file = ROOT / args.output / "raw" / f"noise{ratio:.2f}_n{n}_rep{rep}.csv"
    sigma = noise_sd(ratio, B, args)
    if out_file.exists():
        return str(out_file), 0.0, True
    t0 = time.perf_counter()
    tl.set_backend("numpy")
    train_rng, test_rng = task_rngs(args.base_seed, n, rep)
    X, y = simulate(n, B, sigma, train_rng, args)
    Xte, yte = simulate(args.test_size, B, sigma, test_rng, args)
    X2, Xte2 = X.reshape(n, -1), Xte.reshape(args.test_size, -1)
    rows = []
    for rank in args.ranks:
        tr = time.perf_counter()
        init_seeds = np.random.SeedSequence([args.base_seed, n, rep, rank]).generate_state(args.starts)
        init_seeds = [int(s % (2**31 - 1)) for s in init_seeds]
        fs, iters, spread = fit_cp(X, X2, y, rank, init_seeds, args)
        geo0 = cp_geometry(fs, X2, y)
        tl_train, tl_stat = geo0["train"], stationarity(geo0)
        tl_test = float(np.mean((yte - Xte2 @ cp_full(fs).ravel()) ** 2))
        fs, steps = newton_polish(fs, X2, y, args.polish_steps, args.gtol)
        Bhat = cp_full(fs)
        test = float(np.mean((yte - Xte2 @ Bhat.ravel()) ** 2))
        g = plugin_optimism(fs, X2, y)
        g.pop("hessian_eigs")
        train = g.pop("train_error")
        est = {k: g.pop(k) for k in ("est_optimism", "est_optimism_pd", "est_optimism_score")}
        if args.design == "ar":
            geo = cp_geometry(fs, X2, y)
            g.update(ard.sigma_columns(geo, geo["JQ"], SHAPE, args.rho, train, y, EIG_TOL))
        df_params = rank * (sum(SHAPE) - len(SHAPE) + 1)   # R(I + J + K - 3 + 1)
        rows.append(dict(
            noise_ratio=ratio, noise_sd=sigma, n_train=n, rep=rep, rank=rank,
            n_test=args.test_size, flag=fit_flag(g, args.gtol),
            train_error=train, test_error=test, emp_optimism=test - train,
            est_optimism=est["est_optimism"],
            est_test_error=train + est["est_optimism"],
            aic=np.log(train) + 2 * df_params / n,
            bic=np.log(train) + np.log(n) * df_params / n,
            sel_logn=train + np.log(n) / 2 * est["est_optimism"],
            est_optimism_pd=est["est_optimism_pd"],
            est_test_error_pd=train + est["est_optimism_pd"],
            est_optimism_score=est["est_optimism_score"],
            est_test_error_score=train + est["est_optimism_score"],
            naive_dim_optimism=2 * train * g["chart_dimension"] / n,
            true_excess_risk=float(np.sum((Bhat - B) ** 2)),
            converged=g["stationarity_norm"] < args.gtol, **g,
            tensorly_iterations=iters, tensorly_hit_maxiter=iters >= args.maxiter,
            tensorly_train_error=tl_train, tensorly_test_error=tl_test,
            tensorly_stationarity=tl_stat, polish_steps=steps,
            restart_loss_spread=spread, seconds=time.perf_counter() - tr))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, out_file)
    return str(out_file), time.perf_counter() - t0, False


def run_all(tasks, worker, *extra, jobs=1):
    begin, done = time.perf_counter(), 0

    def report(result):
        nonlocal done
        done += 1
        path, secs, skipped = result
        print(f"[{done}/{len(tasks)}] {Path(path).name} "
              f"{'(exists, skipped)' if skipped else f'{secs:.1f}s'} "
              f"elapsed {time.perf_counter() - begin:.0f}s", flush=True)

    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(worker, *t, *extra) for t in tasks]
            for f in as_completed(futures):
                report(f.result())
    else:
        for t in tasks:
            report(worker(*t, *extra))
    print(f"Finished {len(tasks)} tasks in {time.perf_counter() - begin:.1f}s", flush=True)


def main(args):
    tl.set_backend("numpy")
    B = make_truth(args.true_rank, args.truth_seed, args.weights)
    sds = {r: noise_sd(r, B, args) for r in args.noise_ratios}
    outdir = ROOT / args.output
    (outdir / "raw").mkdir(parents=True, exist_ok=True)
    np.save(outdir / "truth.npy", B)
    settings = dict(vars(args), model="cp", shape=SHAPE, noise_sd=sds,
                    truth_norm=float(np.linalg.norm(B)),
                    tensorly=tl.__version__, numpy=np.__version__, python=sys.version.split()[0])
    pd.Series({k: str(v) for k, v in settings.items()}).to_json(outdir / "settings.json", indent=2)
    print(f"||B||_F = {np.linalg.norm(B):.4f}; noise SD by ratio: "
          + ", ".join(f"{r:g}: {s:.4f}" for r, s in sds.items()), flush=True)
    tasks = [(r, n, rep) for n in sorted(args.n_train, reverse=True) for r in args.noise_ratios
             for rep in range(args.rep_start, args.rep_start + args.reps)]
    run_all(tasks, run_task, args, B, jobs=args.jobs)


if __name__ == "__main__":
    main(ARGS)
