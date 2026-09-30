"""A simulated stroke population, with lesions shaped by the blood supply.

Real stroke lesions are not random blobs: they fill the territory downstream of a blocked
artery. That makes voxels co-occur in lesions along the vascular tree, and it is exactly what
can push a lesion-deficit map away from the truly critical region. This module imitates it:

- a brain (ellipsoid) on a 3 mm MNI-like grid;
- left-hemisphere territories for the anterior, middle and posterior cerebral arteries (ACA,
  MCA, PCA) and the deep lenticulostriate branches;
- in each territory, a branching tree built by recursively splitting the territory in two,
  down to small leaf territories;
- a stroke blocks one branch (proximal blocks are rarer and cause larger lesions), and the
  lesion is that branch's whole downstream territory, with a ragged edge (part of the
  territory survives);
- an anatomical atlas made independently of the vasculature: a k-means parcellation of the
  brain into regions of similar size, standing in for an atlas like AAL.

It is a caricature, not a model of real anatomy; the point is lesions whose shapes and
co-occurrence follow a tree, as real ones do.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter

SHAPE = (61, 73, 61)                           # MNI 3 mm grid
AFFINE = np.array([[-3.0, 0, 0, 90], [0, 3.0, 0, -126], [0, 0, 3.0, -72], [0, 0, 0, 1]])

ORIGIN = {"MCA": np.array([-24.0, 2.0, -12.0]), "PCA": np.array([-12.0, -28.0, -12.0]),
          "ACA": np.array([-6.0, 18.0, -8.0]), "deep": np.array([-22.0, 2.0, 0.0])}
TERRITORY_PROB = {"MCA": 0.70, "PCA": 0.13, "ACA": 0.10, "deep": 0.07}
DEPTH_PROB = np.array([0.04, 0.08, 0.14, 0.2, 0.2, 0.16, 0.11, 0.07])   # depth 0 = the main trunk


@dataclass
class Population:
    lesions: np.ndarray           # (patients, brain voxels) bool
    coords: np.ndarray            # (brain voxels, 3) mm
    regions: np.ndarray           # (regions, brain voxels) bool
    region_names: list
    brain_index: np.ndarray       # indices of the brain voxels in the full grid (Fortran order)
    territory: np.ndarray         # per brain voxel: 0 none/right, 1 MCA, 2 PCA, 3 ACA, 4 deep
    shape: tuple = SHAPE
    affine: np.ndarray = field(default_factory=lambda: AFFINE.copy())
    stroke_info: list | None = None

    def volume(self, flat):
        """A full 3D volume from a per-brain-voxel vector (for plotting / saving)."""
        out = np.zeros(int(np.prod(self.shape)), dtype=np.float32)
        out[self.brain_index] = flat
        return out.reshape(self.shape, order="F")


def _grid():
    i, j, k = np.meshgrid(*[np.arange(s) for s in SHAPE], indexing="ij")
    ijk = np.stack([i, j, k, np.ones_like(i)], -1).reshape(-1, 4, order="F")
    return (ijk @ AFFINE.T)[:, :3]


def _tree(points, idx, rng, depth=0, max_depth=8, min_size=12):
    """Recursive 2-means split of a territory: returns a nested dict of voxel index sets."""
    node = {"voxels": idx, "children": [], "depth": depth}
    if depth >= max_depth or len(idx) < 2 * min_size:
        return node
    P = points[idx]
    # split along a random direction through the territory, then refine with a few 2-means steps
    c = P[rng.choice(len(P), 2, replace=False)]
    for _ in range(8):
        lab = ((P[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)
        if lab.min() == lab.max():
            return node
        c = np.array([P[lab == g].mean(0) for g in (0, 1)])
    for g in (0, 1):
        node["children"].append(_tree(points, idx[lab == g], rng, depth + 1, max_depth, min_size))
    return node


def _walk(node, depth, points, origin, rng, scale):
    """Walk down the tree to `depth`, preferring branches nearer the artery's origin: emboli and
    thrombi lodge more often proximally, so territories near the trunk (e.g. the insula for the
    MCA) are lesioned more often, as in real lesion-overlap maps."""
    while node["depth"] < depth and node["children"]:
        d = np.array([np.linalg.norm(points[c["voxels"]].mean(0) - origin) for c in node["children"]])
        w = np.exp(-(d - d.min()) / scale)
        node = node["children"][rng.choice(len(w), p=w / w.sum())]
    return node


def simulate(n_patients: int = 500, n_regions: int = 64, proximal_bias: float = 20.0, seed: int = 0) -> Population:
    """proximal_bias: length scale (mm) of the preference for branches near each artery's origin;
    larger = more uniform lesion overlap."""
    rng = np.random.default_rng(seed)
    xyz = _grid()
    x, y, z = xyz.T
    brain = ((x / 68) ** 2 + ((y + 18) / 88) ** 2 + ((z - 12) / 62) ** 2 < 1) & (z > -48)
    brain_index = np.flatnonzero(brain)
    C = xyz[brain]
    x, y, z = C.T

    # territories (left hemisphere, x < 0 in MNI)
    left = x < -2
    deep = left & (((x + 22) / 14) ** 2 + ((y - 2) / 18) ** 2 + ((z - 4) / 14) ** 2 < 1)
    aca = left & ~deep & (x > -22) & (((z > 18) & (y > -55)) | ((y > 25) & (z > -5)))
    pca = left & ~deep & ~aca & ((y < -68) | ((z < -2) & (y < -38) & (x > -48)))
    mca = left & ~deep & ~aca & ~pca
    territory = np.zeros(len(C), dtype=int)
    for code, m in ((1, mca), (2, pca), (3, aca), (4, deep)):
        territory[m] = code
    trees = {name: _tree(C, np.flatnonzero(m), rng, max_depth=8 if name == "MCA" else 6)
             for name, m in (("MCA", mca), ("PCA", pca), ("ACA", aca), ("deep", deep))}

    # anatomical atlas: k-means parcellation of the whole brain, blind to the territories
    seeds = C[rng.choice(len(C), n_regions, replace=False)]
    for _ in range(15):
        lab = np.empty(len(C), dtype=int)
        for s in range(0, len(C), 20000):
            lab[s:s + 20000] = ((C[s:s + 20000, None, :] - seeds[None]) ** 2).sum(-1).argmin(1)
        seeds = np.array([C[lab == g].mean(0) if (lab == g).any() else seeds[g] for g in range(n_regions)])
    order = np.lexsort((seeds[:, 2], seeds[:, 1], seeds[:, 0] > 0))
    regions = np.array([lab == g for g in order])
    names = [("L" if seeds[g, 0] < 0 else "R") + f"{i + 1:02d}" for i, g in enumerate(order)]

    # patients
    terr_names = list(TERRITORY_PROB)
    terr_p = np.array(list(TERRITORY_PROB.values()))
    full_noise_shape = SHAPE
    lesions = np.zeros((n_patients, len(C)), dtype=bool)
    info = []
    for pt in range(n_patients):
        t = terr_names[rng.choice(len(terr_names), p=terr_p)]
        depth = int(rng.choice(len(DEPTH_PROB), p=DEPTH_PROB))
        if t != "MCA":
            depth = min(depth + 1, 6)             # smaller territories: fewer levels
        node = _walk(trees[t], depth, C, ORIGIN[t], rng, proximal_bias)
        vox = node["voxels"]
        # ragged edge: keep the part of the territory where a smooth random field is high
        noise = gaussian_filter(rng.normal(size=full_noise_shape), sigma=2.0).ravel(order="F")[brain_index][vox]
        keep_fraction = rng.uniform(0.55, 1.0)
        thr = np.quantile(noise, 1 - keep_fraction)
        lesions[pt, vox[noise >= thr]] = True
        info.append({"territory": t, "depth": int(node["depth"]), "voxels": int((noise >= thr).sum())})
    return Population(lesions, C, regions, names, brain_index, territory, stroke_info=info)
