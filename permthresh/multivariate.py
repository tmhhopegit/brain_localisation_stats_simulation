"""A multivariate lesion-deficit map: ridge regression on clusters of voxels, with lesion volume as a
covariate and a permutation max-statistic threshold (an addition; not in the original).

Why: a univariate map (PM3, or the volume-controlled partial correlation) asks, voxel by voxel,
"is damage here associated with the deficit?". Voxels on the same arterial branch are damaged
together, so every one of them is associated, and the map spreads along the vascular tree. The
multivariate question is "does damage here add anything once damage everywhere else is known?".

How:
1. Voxels are grouped into clusters of neighbouring voxels with similar lesion patterns
   (spatially constrained Ward clustering). Voxels with near-identical patterns carry the same
   information and cannot be told apart by any method, so they are tested together.
2. Impairment and each cluster's lesion load are residualised on lesion volume, and impairment is
   regressed on all clusters at once with a ridge penalty. The penalty strength is chosen by
   10-fold cross-validated prediction error, without reference to where the effect is.
3. Significance: the residualised impairment is permuted (Freedman & Lane) and the ridge fit is
   repeated. The fit is linear in the outcome, so all permutations are one matrix product. A
   cluster is significant if its coefficient exceeds the (1 - alpha) quantile of the largest
   coefficient over clusters in each permutation (family-wise, one-sided: damage -> impairment).

What to expect (see README): far fewer significant voxels outside the true region than either
univariate map, at the cost of finding a smaller part of it. A multivariate map says where damage
matters *beyond* everything else it co-occurs with, which is usually a core, not the whole region.
"""
from __future__ import annotations

import numpy as np

from .lpd import LPDResult

LAMBDA_GRID = np.logspace(0, 5, 21)


def lesion_clusters(lesions, coords, n_clusters: int, voxel_size: float | None = None) -> np.ndarray:
    """Cluster label per voxel: spatially contiguous groups of voxels with similar lesion patterns."""
    from scipy.spatial import cKDTree
    from sklearn.cluster import FeatureAgglomeration
    coords = np.asarray(coords, dtype=np.float64)
    if voxel_size is None:
        d, _ = cKDTree(coords).query(coords, k=2)
        voxel_size = float(np.median(d[:, 1]))
    pairs = cKDTree(coords).query_pairs(voxel_size * 1.01, output_type="ndarray")
    from scipy.sparse import coo_matrix
    n = len(coords)
    conn = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    conn = conn + conn.T
    n_clusters = int(min(n_clusters, n))
    agg = FeatureAgglomeration(n_clusters=n_clusters, connectivity=conn, linkage="ward")
    agg.fit(np.asarray(lesions, dtype=np.float32))
    return agg.labels_


def _design(lesions, labels, volume):
    L = np.asarray(lesions, dtype=np.float64)
    K = labels.max() + 1
    counts = np.bincount(labels, minlength=K)
    F = (L @ np.eye(K)[labels]) / counts                       # patients x clusters: fraction lesioned
    n = len(L)
    Z = np.column_stack([np.ones(n), volume])
    P = np.eye(n) - Z @ np.linalg.pinv(Z)                       # residual-maker for the covariates
    X = P @ F
    sd = X.std(0)
    sd[sd == 0] = 1.0
    return X / sd, P


def choose_lambda(X, y, k: int = 10, seed: int = 0, grid=LAMBDA_GRID) -> float:
    """Ridge penalty with the lowest 10-fold cross-validated squared prediction error."""
    rng = np.random.default_rng(seed)
    n = len(y)
    folds = rng.permutation(n) % k
    err = np.zeros(len(grid))
    for f in range(k):
        tr, te = folds != f, folds == f
        U, s, Vt = np.linalg.svd(X[tr], full_matrices=False)
        Uy = U.T @ y[tr]
        for i, lam in enumerate(grid):
            b = Vt.T @ (s / (s ** 2 + lam) * Uy)
            err[i] += ((y[te] - X[te] @ b) ** 2).sum()
    return float(grid[int(np.argmin(err))])


def permutation_ridge(lesions, impaired, labels, volume=None, n_perm: int = 1000, lam: float | None = None,
                      seed: int | None = 0, batch: int = 500) -> LPDResult:
    """Cluster-level ridge map with its permutation null, returned per voxel (every voxel carries its
    cluster's coefficient) so it can be thresholded like the univariate maps."""
    rng = np.random.default_rng(seed)
    impaired = np.asarray(impaired, bool)
    L = np.asarray(lesions, bool)
    n = len(impaired)
    if impaired.all() or not impaired.any():
        raise ValueError("need at least one impaired and one unimpaired patient")
    volume = L.sum(1).astype(np.float64) if volume is None else np.asarray(volume, dtype=np.float64)
    X, P = _design(L, labels, volume)
    y = P @ impaired.astype(np.float64)
    if lam is None:
        lam = choose_lambda(X, y, seed=int(rng.integers(2**31)))
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    A = (Vt.T * (s / (s ** 2 + lam))) @ U.T                     # beta = A @ y
    beta = A @ y
    null_max = np.empty(n_perm)
    null_arg = np.empty(n_perm, dtype=np.int64)
    exceed = np.zeros(len(beta), dtype=np.int64)
    done = 0
    while done < n_perm:
        b = min(batch, n_perm - done)
        Y = np.column_stack([y[rng.permutation(n)] for _ in range(b)])
        B = A @ Y                                               # clusters x b
        null_max[done:done + b] = B.max(0)
        null_arg[done:done + b] = B.argmax(0)
        exceed += (B >= beta[:, None] - 1e-12).sum(1)
        done += b
    first_voxel = np.array([np.flatnonzero(labels == k)[0] for k in range(len(beta))])
    res = LPDResult(beta[labels], null_max, first_voxel[null_arg], ((exceed + 1) / (n_perm + 1))[labels], n_perm)
    res.lam = lam
    res.cluster_beta = beta
    return res
