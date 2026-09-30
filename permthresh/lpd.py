"""Lesion-deficit maps with permutation thresholds.

The statistic is Rudrauf et al.'s proportional-difference ("PM3") map: for each voxel,

    PM3 = (impaired patients lesioned there / impaired patients)
        - (unimpaired patients lesioned there / unimpaired patients)

Significance comes from permuting the impairment labels (MakeLPDMap_Thresholded):

    family-wise   the largest PM3 anywhere in the brain is recorded for each permutation;
                  a voxel survives at level alpha if its PM3 exceeds the (1 - alpha)
                  quantile of those maxima
    voxel-wise    each voxel's p-value is the fraction of permutations with a PM3 at least
                  as large there; a voxel survives if p < alpha (uncorrected)

All permutations are computed at once as a matrix product, so 1000 permutations over tens
of thousands of voxels take a second or two.

statistic="partial_r" (an addition, not in the original) controls for lesion volume: the
correlation between lesion status and impairment after regressing lesion volume out of both,
with the volume-residualised impairment permuted (Freedman & Lane). PM3 ignores volume, and
patients with large lesions are both more often impaired and lesioned almost everywhere.

Voxel-wise p-values are (1 + permutations at least as large) / (1 + permutations); the original
counted strictly larger ones and did not add 1, which gives p = 0 at the top of the range.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def pm3_map(lesions: np.ndarray, impaired: np.ndarray) -> np.ndarray:
    """PM3 at each voxel. lesions: (patients, voxels) boolean; impaired: (patients,) boolean."""
    impaired = np.asarray(impaired, bool)
    L = np.asarray(lesions, dtype=np.float32)
    nd, nn = impaired.sum(), (~impaired).sum()
    if nd == 0 or nn == 0:
        raise ValueError("need at least one impaired and one unimpaired patient")
    return (impaired.astype(np.float32) @ L) / nd - ((~impaired).astype(np.float32) @ L) / nn


@dataclass
class LPDResult:
    map: np.ndarray               # PM3 per voxel
    null_max: np.ndarray          # largest PM3 over voxels, per permutation
    null_max_voxel: np.ndarray    # where that maximum was (first voxel reaching it)
    p_voxel: np.ndarray           # voxel-wise permutation p-values
    n_perm: int

    def fwe_threshold(self, alpha: float) -> float:
        """PM3 a voxel must exceed to survive family-wise correction at alpha."""
        return float(np.quantile(self.null_max, 1 - alpha, method="higher"))

    def fwe_mask(self, alpha: float) -> np.ndarray:
        return self.map > self.fwe_threshold(alpha)

    def voxel_mask(self, alpha: float) -> np.ndarray:
        return self.p_voxel < alpha

    def fwe_p(self) -> np.ndarray:
        """Family-wise corrected p-value per voxel."""
        s = np.sort(self.null_max)
        return (1 + len(s) - np.searchsorted(s, self.map, side="left")) / (len(s) + 1)


def _residualise(M, Z):
    beta, *_ = np.linalg.lstsq(Z, M, rcond=None)
    return M - Z @ beta


def permutation_lpd(lesions, impaired, n_perm: int = 1000, seed: int | None = 0, batch: int = 250,
                    statistic: str = "pm3", volume=None) -> LPDResult:
    """Lesion-deficit map with its permutation null (MakeLPDMap_Thresholded).

    statistic: 'pm3' (the original) or 'partial_r' (volume-controlled; `volume` defaults to the
    number of lesioned voxels given)."""
    rng = np.random.default_rng(seed)
    L = np.asarray(lesions, dtype=np.float32)
    impaired = np.asarray(impaired, bool)
    n = len(impaired)
    nd = int(impaired.sum())
    if nd == 0 or nd == n:
        raise ValueError("need at least one impaired and one unimpaired patient")
    if statistic == "pm3":
        obs = pm3_map(L, impaired)
        tot = L.sum(0, dtype=np.float32)[None, :]

        def batch_maps(P):                               # P: (patients, b) 0/1 relabellings
            imp = P.T @ L
            return imp / nd - (tot - imp) / (n - nd)

        def relabel(b):
            P = np.zeros((n, b), dtype=np.float32)
            for j in range(b):
                P[rng.permutation(n)[:nd], j] = 1
            return P
    elif statistic == "partial_r":
        vol = L.sum(1) if volume is None else np.asarray(volume, dtype=np.float64)
        Z = np.column_stack([np.ones(n), vol])
        Lr = _residualise(L.astype(np.float64), Z).astype(np.float32)
        norms = np.linalg.norm(Lr, axis=0)
        norms[norms == 0] = np.inf
        yr = _residualise(impaired.astype(np.float64)[:, None], Z).ravel()
        yr = (yr / np.linalg.norm(yr)).astype(np.float32)
        obs = (yr @ Lr) / norms

        def batch_maps(P):
            return (P.T @ Lr) / norms[None, :]

        def relabel(b):
            return np.column_stack([yr[rng.permutation(n)] for _ in range(b)])
    else:
        raise ValueError("statistic must be 'pm3' or 'partial_r'")
    null_max = np.empty(n_perm)
    null_pos = np.empty(n_perm, dtype=np.int64)
    exceed = np.zeros(L.shape[1], dtype=np.int64)
    done = 0
    while done < n_perm:
        b = min(batch, n_perm - done)
        maps = batch_maps(relabel(b))
        null_max[done:done + b] = maps.max(1)
        null_pos[done:done + b] = maps.argmax(1)
        exceed += (maps >= obs[None, :] - 1e-6).sum(0)
        done += b
    return LPDResult(obs, null_max, null_pos, (exceed + 1) / (n_perm + 1), n_perm)
