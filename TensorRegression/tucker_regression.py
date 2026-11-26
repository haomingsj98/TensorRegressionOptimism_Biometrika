"""Tucker tensor regression: empirical vs. plug-in optimism across candidate ranks.

Same design, noise levels and random streams as cp_regression.py (a given (n, seed) has the
same covariates and standardized noise). True Tucker rank (3, 3, 3): B = G x_m U_m with
orthonormal U_m and a dense core whose mode unfoldings all have singular values
(1, 0.7, 0.5). Every candidate tuple is fitted with TensorLy TuckerRegressor (best of
--starts random starts), then polished on the Tucker chart
h(H, W_1, W_2, W_3) = H x_m (U_m + U_m,perp W_m) of dimension prod R_m + sum R_m (I_m - R_m).

    python tucker_regression.py --design identity --jobs 8
    python tucker_regression.py --design ar --jobs 8

Output as cp_regression.py; ranks are stored as strings such as "3,3,3".
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SHAPE = (10, 8, 12)
ROOT = Path(__file__).resolve().parent
EIG_TOL = 1e-8
DEFAULT_RANKS = ["1,1,1", "2,2,2", "2,3,3", "3,3,3", "4,3,3", "4,4,4", "5,5,5"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--design", choices=["identity", "ar"], default="identity")
    p.add_argument("--rho", type=float, default=0.5, help="AR(1) correlation (--design ar)")
    p.add_argument("--n-train", type=int, nargs="+", default=[400, 600, 800, 1200, 1600])
    p.add_argument("--ranks", nargs="+", default=DEFAULT_RANKS)
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--rep-start", type=int, default=0)
    p.add_argument("--test-size", type=int, default=5000)
    p.add_argument("--true-rank", default="3,3,3")
    p.add_argument("--truth-seed", type=int, default=123)
    p.add_argument("--mode-sv", type=float, nargs="+", default=[1.0, 0.7, 0.5])
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
    p.add_argument("--output", default=None, help="default results/tucker_<design>")
    args = p.parse_args(argv)
    args.output = args.output or f"results/tucker_{args.design}"
    return args


ARGS = parse_args() if __name__ == "__main__" else None
_threads = str(ARGS.threads or (1 if ARGS.jobs > 1 else 4)) if ARGS else "1"
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, _threads)

import time

import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.regression.tucker_regression import TuckerRegressor

import ar_design as ard
from cp_regression import fit_flag, run_all, simulate, task_rngs, noise_sd


def parse_rank(s):
    return tuple(int(v) for v in str(s).split(","))


def rank_label(R):
    return ",".join(map(str, R))


def tucker_dim(R):
    return int(np.prod(R) + sum(r * (I - r) for r, I in zip(R, SHAPE)))


# ----------------------------------------------------------------------------
# Tensor helpers (row-major vectorization)
# ----------------------------------------------------------------------------
def unfold(T, m):
    return np.moveaxis(T, m, 0).reshape(T.shape[m], -1)


def fold(M, m, shape):
    rest = [s for k, s in enumerate(shape) if k != m]
    return np.moveaxis(M.reshape(shape[m], *rest), 0, m)


def tucker_full(G, U):
    return np.einsum("pqr,ip,jq,kr->ijk", G, *U, optimize=True)


def hosvd(B, R):
    U = [np.linalg.svd(unfold(B, m), full_matrices=False)[0][:, :R[m]] for m in range(3)]
    return np.einsum("ijk,ip,jq,kr->pqr", B, *U, optimize=True), U


def complement(U):
    return np.linalg.qr(U, mode="complete")[0][:, U.shape[1]:]


def make_truth(R, seed, mode_sv, iters=10000):
    """Dense core with the given singular values in every mode unfolding (alternating
    projections); orthonormal factors from QR of Gaussian matrices."""
    rng = np.random.default_rng(seed)
    s = np.asarray(mode_sv, dtype=np.float64)
    G = rng.standard_normal(R)
    for _ in range(iters):
        for m in range(3):
            P, _, Qt = np.linalg.svd(unfold(G, m), full_matrices=False)
            G = fold(P @ np.diag(s) @ Qt, m, R)
        err = max(np.abs(np.linalg.svd(unfold(G, m), compute_uv=False) - s).max()
                  for m in range(3))
        if err < 1e-13:
            break
    else:
        raise RuntimeError(f"core spectra did not converge (max error {err:.2e})")
    U = [np.linalg.qr(rng.standard_normal((I, r)))[0] for I, r in zip(SHAPE, R)]
    return tucker_full(G, U)


# ----------------------------------------------------------------------------
# Jacobian and Hessian on the Tucker chart
# ----------------------------------------------------------------------------
def tucker_geometry(G, U, X2, y):
    """Coordinates: core H, then W_m, m = 1, 2, 3. A_hat = Z'Z/n - D^2 <h(theta), C>."""
    n = len(y)
    R = G.shape
    D = int(np.prod(SHAPE))
    Pp = [complement(u) for u in U]
    e = y - X2 @ tucker_full(G, U).ravel()
    C = (X2.T @ e / n).reshape(SHAPE)
    U0, U1, U2 = U
    P0, P1, P2 = Pp

    JH = np.einsum("ip,jq,kr->ijkpqr", U0, U1, U2).reshape(D, -1)
    T1 = np.einsum("pqr,jq,kr->pjk", G, U1, U2)
    T2 = np.einsum("pqr,ip,kr->iqk", G, U0, U2)
    T3 = np.einsum("pqr,ip,jq->ijr", G, U0, U1)
    JW1 = np.einsum("ia,pjk->ijkap", P0, T1).reshape(D, -1)
    JW2 = np.einsum("jb,iqk->ijkbq", P1, T2).reshape(D, -1)
    JW3 = np.einsum("kc,ijr->ijkcr", P2, T3).reshape(D, -1)
    Jac = np.hstack([JH, JW1, JW2, JW3])
    sizes = [JH.shape[1], JW1.shape[1], JW2.shape[1], JW3.shape[1]]
    cuts = np.cumsum([0] + sizes)
    blk = [slice(cuts[b], cuts[b + 1]) for b in range(4)]

    proj = lambda A, B_, Cm: np.einsum("ijk,ia,jb,kc->abc", C, A, B_, Cm, optimize=True)
    C1, C2, C3 = proj(P0, U1, U2), proj(U0, P1, U2), proj(U0, U1, P2)
    C12, C13, C23 = proj(P0, P1, U2), proj(P0, U1, P2), proj(U0, P1, P2)
    nH = sizes[0]
    H = np.zeros((cuts[-1], cuts[-1]))
    H[blk[0], blk[1]] = np.einsum("aqr,ps->pqras", C1, np.eye(R[0])).reshape(nH, -1)
    H[blk[0], blk[2]] = np.einsum("pbr,qt->pqrbt", C2, np.eye(R[1])).reshape(nH, -1)
    H[blk[0], blk[3]] = np.einsum("pqc,ru->pqrcu", C3, np.eye(R[2])).reshape(nH, -1)
    H[blk[1], blk[2]] = np.einsum("pqr,abr->apbq", G, C12).reshape(sizes[1], sizes[2])
    H[blk[1], blk[3]] = np.einsum("pqr,aqc->apcr", G, C13).reshape(sizes[1], sizes[3])
    H[blk[2], blk[3]] = np.einsum("pqr,pbc->bqcr", G, C23).reshape(sizes[2], sizes[3])
    H = H + H.T

    Z = X2 @ Jac
    Gram = Jac.T @ Jac
    ge, gv = np.linalg.eigh(Gram)
    full_rank = bool((ge > ge[-1] * 1e-10).sum() == len(ge))
    Wh = (gv / np.sqrt(np.maximum(ge, 1e-300))) @ gv.T
    Aw = Wh @ (Z.T @ Z / n - H) @ Wh
    return dict(e=e, train=float(e @ e / n), Jac=Jac, Z=Z, Wh=Wh, Pp=Pp, cuts=cuts,
                Aw=(Aw + Aw.T) / 2, grad=Jac.T @ C.ravel(), H=H,
                full_rank=full_rank, jacobian_condition=float(ge[-1] / max(ge[0], 1e-300)))


def stationarity(geo):
    return float(np.linalg.norm(geo["Wh"] @ geo["grad"]))


def chart_point(G, U, Pp, cuts, z):
    R = G.shape
    Hn = G + z[cuts[0]:cuts[1]].reshape(R)
    Un, Rs = [], []
    for m in range(3):
        Wm = z[cuts[m + 1]:cuts[m + 2]].reshape(SHAPE[m] - R[m], R[m])
        q, r = np.linalg.qr(U[m] + Pp[m] @ Wm)
        Un.append(q)
        Rs.append(r)
    return np.einsum("pqr,ap,bq,cr->abc", Hn, *Rs, optimize=True), Un


def newton_polish(G, U, X2, y, max_steps, gtol):
    start_loss = float(np.mean((y - X2 @ tucker_full(G, U).ravel()) ** 2))
    steps = 0
    for _ in range(max_steps):
        geo = tucker_geometry(G, U, X2, y)
        gw = geo["Wh"] @ geo["grad"]
        if not geo["full_rank"] or np.linalg.norm(gw) < gtol:
            break
        lam, vec = np.linalg.eigh(geo["Aw"])
        lam = np.maximum(np.abs(lam), 1e-3 * np.abs(lam).max())
        dz = geo["Wh"] @ (vec @ ((vec.T @ gw) / lam))
        Bnorm = np.linalg.norm(tucker_full(G, U))
        dz *= min(1.0, 0.1 * Bnorm / max(np.linalg.norm(geo["Jac"] @ dz), 1e-300))
        slope = float(geo["grad"] @ dz)
        half = geo["train"] / 2
        for k in range(30):
            t = 0.5 ** k
            Gt, Ut = chart_point(G, U, geo["Pp"], geo["cuts"], t * dz)
            e = y - X2 @ tucker_full(Gt, Ut).ravel()
            if 0.5 * np.mean(e ** 2) <= half - 1e-4 * t * slope:
                G, U, steps = Gt, Ut, steps + 1
                break
        else:
            break
    final_loss = float(np.mean((y - X2 @ tucker_full(G, U).ravel()) ** 2))
    assert final_loss <= start_loss * (1 + 1e-12)
    return G, U, steps


def plugin_optimism(G, U, X2, y):
    """Identity and general plug-in forms, as in cp_regression.plugin_optimism."""
    n = len(y)
    geo = tucker_geometry(G, U, X2, y)
    train = geo["train"]
    nan = float("nan")
    out = dict(train_error=train, chart_dimension=geo["Jac"].shape[1],
               jacobian_full_rank=geo["full_rank"],
               jacobian_condition=geo["jacobian_condition"],
               stationarity_norm=stationarity(geo), hessian_pd=False, n_negative_eigs=-1,
               hessian_min_eig=nan, hessian_min_pos_eig=nan, hessian_condition=nan,
               trace=nan, trace_pd=nan, trace_score=nan)
    if geo["full_rank"]:
        lam, vec = np.linalg.eigh(geo["Aw"])
        pos = lam > EIG_TOL
        ZW = geo["Z"] @ geo["Wh"] * geo["e"][:, None]
        Ow = ZW.T @ ZW / n
        Vp = vec[:, pos]
        out.update(hessian_min_eig=float(lam[0]),
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
def fit_tucker(X, X2, y, R, init_seeds, args):
    """TuckerRegressor from several random starts; keep the smallest training MSE."""
    best, losses, failed = None, [], 0
    for s in init_seeds:
        est = TuckerRegressor(weight_ranks=list(R), tol=args.tol, reg_W=0,
                              n_iter_max=args.maxiter, random_state=int(s), verbose=0)
        try:
            with np.errstate(all="ignore"):
                est.fit(X, y)
            B = np.asarray(est.weight_tensor_, dtype=np.float64)
            loss = float(np.mean((y - X2 @ B.ravel()) ** 2))
        except np.linalg.LinAlgError:
            failed += 1
            continue
        if not np.isfinite(loss):
            failed += 1
            continue
        losses.append(loss)
        if best is None or loss < best[0]:
            best = (loss, B, est.n_iterations_)
    if best is None:
        return None, 0, float("nan"), failed
    loss, B, iters = best
    G, U = hosvd(B, R)
    return (G, U), iters, float(np.max(losses) - loss), failed


def run_task(ratio, n, rep, args, B):
    out_file = ROOT / args.output / "raw" / f"noise{ratio:.2f}_n{n}_rep{rep}.csv"
    if out_file.exists():
        return str(out_file), 0.0, True
    t0 = time.perf_counter()
    tl.set_backend("numpy")
    sigma = noise_sd(ratio, B, args)
    train_rng, test_rng = task_rngs(args.base_seed, n, rep)
    X, y = simulate(n, B, sigma, train_rng, args)
    Xte, yte = simulate(args.test_size, B, sigma, test_rng, args)
    X2, Xte2 = X.reshape(n, -1), Xte.reshape(args.test_size, -1)
    rows = []
    for k, label in enumerate(args.ranks):
        R = parse_rank(label)
        tr = time.perf_counter()
        init_seeds = np.random.SeedSequence([args.base_seed, n, rep, 1000 + k]).generate_state(args.starts)
        init_seeds = [int(s % (2**31 - 1)) for s in init_seeds]
        fit, iters, spread, failed = fit_tucker(X, X2, y, R, init_seeds, args)
        d = tucker_dim(R)
        base = dict(noise_ratio=ratio, noise_sd=sigma, n_train=n, rep=rep, rank=rank_label(R),
                    rank_order=k, df=d, d_ge_n=d >= n, n_test=args.test_size)
        if fit is None:
            rows.append(dict(base, flag="fit_failed", failed_starts=failed,
                             seconds=time.perf_counter() - tr))
            continue
        G, U = fit
        geo0 = tucker_geometry(G, U, X2, y)
        tl_train, tl_stat = geo0["train"], stationarity(geo0)
        tl_test = float(np.mean((yte - Xte2 @ tucker_full(G, U).ravel()) ** 2))
        if geo0["full_rank"]:
            G, U, steps = newton_polish(G, U, X2, y, args.polish_steps, args.gtol)
        else:
            steps = 0
        Bhat = tucker_full(G, U)
        test = float(np.mean((yte - Xte2 @ Bhat.ravel()) ** 2))
        g = plugin_optimism(G, U, X2, y)
        train = g.pop("train_error")
        est = {k_: g.pop(k_) for k_ in ("est_optimism", "est_optimism_pd", "est_optimism_score")}
        if args.design == "ar":
            geo = tucker_geometry(G, U, X2, y)
            g.update(ard.sigma_columns(geo, geo["Jac"], SHAPE, args.rho, train, y, EIG_TOL))
        with np.errstate(divide="ignore"):
            logtrain = np.log(train)
        rows.append(dict(
            base, flag=fit_flag(g, args.gtol),
            train_error=train, test_error=test, emp_optimism=test - train,
            est_optimism=est["est_optimism"],
            est_test_error=train + est["est_optimism"],
            aic=logtrain + 2 * d / n,
            bic=logtrain + np.log(n) * d / n,
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
            tensorly_stationarity=tl_stat, polish_steps=steps, failed_starts=failed,
            restart_loss_spread=spread, seconds=time.perf_counter() - tr))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, out_file)
    return str(out_file), time.perf_counter() - t0, False


def main(args):
    tl.set_backend("numpy")
    args.ranks = [rank_label(parse_rank(r)) for r in args.ranks]
    args.true_rank = rank_label(parse_rank(args.true_rank))
    B = make_truth(parse_rank(args.true_rank), args.truth_seed, args.mode_sv)
    sds = {r: noise_sd(r, B, args) for r in args.noise_ratios}
    outdir = ROOT / args.output
    (outdir / "raw").mkdir(parents=True, exist_ok=True)
    np.save(outdir / "truth.npy", B)
    settings = dict(vars(args), model="tucker", shape=SHAPE, noise_sd=sds,
                    truth_norm=float(np.linalg.norm(B)),
                    df={r: tucker_dim(parse_rank(r)) for r in args.ranks},
                    tensorly=tl.__version__, numpy=np.__version__, python=sys.version.split()[0])
    pd.Series({k: str(v) for k, v in settings.items()}).to_json(outdir / "settings.json", indent=2)
    print(f"||B||_F = {np.linalg.norm(B):.4f}; noise SD by ratio: "
          + ", ".join(f"{r:g}: {s:.4f}" for r, s in sds.items()), flush=True)
    tasks = [(r, n, rep) for n in sorted(args.n_train, reverse=True) for r in args.noise_ratios
             for rep in range(args.rep_start, args.rep_start + args.reps)]
    run_all(tasks, run_task, args, B, jobs=args.jobs)


if __name__ == "__main__":
    main(ARGS)
