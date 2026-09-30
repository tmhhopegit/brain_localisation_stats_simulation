"""The simulation: pretend each region in turn is the one that matters, and see where the
lesion-deficit map puts the effect (RunPermThreshAnalysis.m).

For each region R, each lesion-load threshold t and each detection probability p:

    impaired = (patient's lesion covers >= t of R) and (with probability p, independently)

i.e. damage to R causes the deficit (p = 1), or causes it only sometimes (p < 1, the
"ProbThresholds" noise the original had commented out). A PM3 map is computed with
permutation thresholds, and for each alpha we count the significant voxels inside and outside
R, under family-wise and uncorrected voxel-wise thresholds.

Because the ground truth is known, every significant voxel outside R is a false positive
(for these purposes), and the distance from the significant voxels to R measures how far
the method displaces the effect, e.g. along the vascular tree (Mah et al., Brain 2014).

Preprocessing, as in the original: patients without a lesion are dropped; only voxels lesioned
in at least 10% of patients are tested; only regions with at least 100 testable voxels are
used as "critical" regions. Lesion loads are computed over the whole region, before the voxel
filter.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import warnings

from .lpd import permutation_lpd
from .multivariate import lesion_clusters, permutation_ridge


@dataclass
class Prepared:
    lesions: np.ndarray        # (patients, tested voxels) bool
    loads: np.ndarray          # (patients, regions) fraction of each region lesioned
    regions: np.ndarray        # (regions, tested voxels) bool
    region_ids: np.ndarray     # index of each kept region in the input atlas
    region_names: list
    coords: np.ndarray         # (tested voxels, 3) mm
    voxel_index: np.ndarray    # index of each tested voxel in the input voxel list
    region_centroids: np.ndarray
    volume: np.ndarray         # (patients,) whole-brain lesion volume in voxels
    clusters: np.ndarray | None = None   # cluster label per tested voxel (for the multivariate map)

    def get_clusters(self, n_clusters: int | None = None) -> np.ndarray:
        """Clusters for the multivariate map; default about one per four patients (at most 300)."""
        if self.clusters is None or n_clusters is not None:
            k = n_clusters or int(max(20, min(300, len(self.lesions) // 4)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)    # disconnected islands of tested voxels are fine
                self.clusters = lesion_clusters(self.lesions, self.coords, k)
        return self.clusters


def lesion_deficit_map(prep: "Prepared", impaired, statistic: str, n_perm: int, seed: int):
    """Any of the three maps, as an LPDResult: 'pm3', 'partial_r' or 'ridge'."""
    if statistic == "ridge":
        return permutation_ridge(prep.lesions, impaired, prep.get_clusters(), prep.volume, n_perm, seed=seed)
    return permutation_lpd(prep.lesions, impaired, n_perm, seed=seed, statistic=statistic, volume=prep.volume)


def prepare(lesions, regions, coords, region_names=None, min_lesion_fraction: float = 0.1,
            min_region_voxels: int = 100) -> Prepared:
    """lesions: (patients, voxels); regions: (regions, voxels) boolean; coords: (voxels, 3) in mm."""
    L = np.asarray(lesions, bool)
    R = np.asarray(regions, bool)
    coords = np.asarray(coords, dtype=np.float64)
    names = list(region_names) if region_names is not None else [f"region_{i}" for i in range(len(R))]
    size = R.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        loads = (L.astype(np.float32) @ R.T.astype(np.float32)) / size[None, :]
    has = L.any(1)
    L, loads = L[has], loads[has]
    volume = L.sum(1).astype(np.float64)
    keep_vox = L.sum(0) >= np.ceil(len(L) * min_lesion_fraction)
    L, R2, C = L[:, keep_vox], R[:, keep_vox], coords[keep_vox]
    keep_reg = R2.sum(1) >= min_region_voxels
    ids = np.flatnonzero(keep_reg)
    cent = np.array([coords[R[i]].mean(0) for i in ids])
    return Prepared(L, loads[:, keep_reg], R2[keep_reg], ids, [names[i] for i in ids], C, np.flatnonzero(keep_vox),
                    cent, volume)


def _distance_summary(mask, coords, region_mask, centroid):
    if not mask.any():
        return np.nan, np.nan
    sig_centroid = coords[mask].mean(0)
    # distance from each significant voxel to the nearest voxel of the region (0 inside)
    inside = region_mask[mask]
    if not (~inside).any():
        return float(np.linalg.norm(sig_centroid - centroid)), 0.0
    from scipy.spatial import cKDTree
    d_out, _ = cKDTree(coords[region_mask]).query(coords[mask][~inside])
    return float(np.linalg.norm(sig_centroid - centroid)), float(np.median(d_out))


def run(prep: Prepared, n_perm: int = 1000, load_thresholds=(0.5, 1.0), probabilities=(1.0,),
        alphas=(0.05, 0.01, 0.001), min_impaired: int = 5, seed: int = 0, regions=None,
        statistics=("pm3",), verbose: bool = False):
    """The simulation loop. Returns (results table, maxima_by_region, voxel_maxima):
    maxima_by_region[k, r] counts how many permutation maxima of analysis k fell in region r (the
    original's RegInds; the last column counts maxima outside every region), and voxel_maxima[stat]
    counts, per tested voxel, how often it held the maximum over all analyses.

    Analyses where fewer than `min_impaired` patients (or unimpaired patients) are left are
    skipped and reported with n_impaired only; the original divided by zero there."""
    rng = np.random.default_rng(seed)
    rows, maxima = [], []
    voxel_maxima = {st: np.zeros(prep.lesions.shape[1], dtype=np.int64) for st in statistics}
    order = np.argsort(prep.loads.__gt__(0).sum(0))           # least often lesioned first, as the original
    todo = order if regions is None else [r for r in order if r in set(regions)]
    region_size = prep.regions.sum(1)
    label = np.where(prep.regions.any(0), prep.regions.argmax(0), len(prep.regions))   # last bin = no region
    for r in todo:
        for t in load_thresholds:
            base = prep.loads[:, r] >= t
            for p in probabilities:
                impaired = base & (rng.random(len(base)) < p)
                info = {"region": prep.region_names[r], "region_index": int(r), "region_voxels": int(region_size[r]),
                        "patients_lesioned": int((prep.loads[:, r] > 0).sum()), "load_threshold": t,
                        "probability": p, "n_impaired": int(impaired.sum())}
                if impaired.sum() < min_impaired or (~impaired).sum() < min_impaired:
                    rows.append({**info, "skipped": True})
                    continue
                pseed = int(rng.integers(2**31))
                for stat in statistics:
                    res = lesion_deficit_map(prep, impaired, stat, n_perm, pseed)
                    maxima.append(np.bincount(label[res.null_max_voxel], minlength=len(prep.regions) + 1))
                    voxel_maxima[stat] += np.bincount(res.null_max_voxel, minlength=len(voxel_maxima[stat]))
                    peak = int(np.argmax(res.map))
                    for a in alphas:
                        row = dict(info, skipped=False, statistic=stat, alpha=a, n_perm=n_perm)
                        for kind, mask in (("fwe", res.fwe_mask(a)), ("voxel", res.voxel_mask(a))):
                            inn = int((mask & prep.regions[r]).sum())
                            out = int((mask & ~prep.regions[r]).sum())
                            cdist, med_out = _distance_summary(mask, prep.coords, prep.regions[r],
                                                               prep.region_centroids[r])
                            row.update({f"{kind}_in": inn, f"{kind}_out": out,
                                        f"{kind}_sensitivity": inn / region_size[r],
                                        f"{kind}_fdp": out / (inn + out) if inn + out else np.nan,
                                        f"{kind}_centroid_shift_mm": cdist,
                                        f"{kind}_median_outside_dist_mm": med_out})
                        fmask = res.fwe_mask(a)
                        per_region = (prep.regions & fmask[None, :]).sum(1)
                        row["top_region_correct"] = bool(fmask.any() and int(np.argmax(per_region)) == r)
                        row["regions_significant"] = int((per_region > 0).sum())
                        row["peak_in_region"] = bool(prep.regions[r, peak])
                        row["peak_distance_mm"] = float(np.linalg.norm(prep.coords[peak] - prep.region_centroids[r]))
                        row["fwe_threshold"] = res.fwe_threshold(a)
                        rows.append(row)
                if verbose:
                    print(f"{prep.region_names[r]:>6s} load>={t} p={p}: {info['n_impaired']} impaired", flush=True)
    return pd.DataFrame(rows), np.array(maxima), voxel_maxima


def null_calibration(prep: Prepared, n_datasets: int = 100, n_perm: int = 500, alpha: float = 0.05,
                     impaired_fraction: float = 0.3, statistic: str = "pm3", seed: int = 0) -> dict:
    """How often does the family-wise threshold find anything when impairment is unrelated to the
    lesions? It should be about alpha."""
    rng = np.random.default_rng(seed)
    n = len(prep.lesions)
    hits_fwe, hits_vox = 0, []
    for d in range(n_datasets):
        imp = rng.random(n) < impaired_fraction
        res = lesion_deficit_map(prep, imp, statistic, n_perm, int(rng.integers(2**31)))
        hits_fwe += bool(res.fwe_mask(alpha).any())
        hits_vox.append(res.voxel_mask(alpha).mean())
    return {"datasets": n_datasets, "alpha": alpha, "familywise_error_rate": hits_fwe / n_datasets,
            "voxelwise_false_positive_fraction": float(np.mean(hits_vox))}
