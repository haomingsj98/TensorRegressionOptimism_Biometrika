"""AR(1) covariance on the vectorized covariate: Cov(vec X)_{ab} = rho^|a - b|.

vec is the column-major vectorization (mode 1 fastest); the code stores tensors in
row-major order, so the matrices are permuted accordingly. X = L Z with L the Cholesky
factor and Z the identity-design draws.
"""
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def _matrices(shape, rho):
    D = int(np.prod(shape))
    f = np.arange(D).reshape(shape, order="F").ravel()
    idx = np.arange(D)
    S = rho ** np.abs(idx[:, None] - idx[None, :])
    return S[np.ix_(f, f)], np.linalg.cholesky(S)[np.ix_(f, f)]


def transform(Z, rho):
    L = _matrices(tuple(Z.shape[1:]), rho)[1]
    return (Z.reshape(len(Z), -1) @ L.T).reshape(Z.shape)


def sigma_apply(T, shape, rho):
    """Sigma vec(T) for an array of shape (*shape, ...)."""
    S = _matrices(tuple(shape), rho)[0]
    return (S @ T.reshape(S.shape[0], -1)).reshape(T.shape)


def signal_var(B, rho):
    return float(np.sum(B * sigma_apply(B, B.shape, rho)))      # Var(<X, B>)


def sigma_columns(geo, Jac, shape, rho, train, y, eig_tol):
    """Covariance-weighted plug-in 2 L_n tr(A_hat^-1 J' Sigma J) / n and its selector."""
    n, nan = len(y), float("nan")
    out = dict(trace_sigma=nan, est_optimism_sigma=nan, cap_active_sigma=False)
    if geo["full_rank"]:
        M = Jac.T @ sigma_apply(Jac.reshape(*shape, -1), shape, rho).reshape(Jac.shape)
        Mw = geo["Wh"] @ M @ geo["Wh"]
        lam, vec = np.linalg.eigh(geo["Aw"])
        pos = lam > eig_tol
        Vp = vec[:, pos]
        tr_s = float(np.sum((Vp.T @ Mw @ Vp).diagonal() / lam[pos]))
        out.update(trace_sigma=tr_s,
                   est_optimism_sigma=2 * min(float(y @ y), train * tr_s) / n,
                   cap_active_sigma=bool(train * tr_s > float(y @ y)))
    out["est_test_error_sigma"] = train + out["est_optimism_sigma"]
    out["sel_logn_sigma"] = train + np.log(n) / 2 * out["est_optimism_sigma"]
    return out
