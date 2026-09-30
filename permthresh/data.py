"""Loading real lesion maps and an atlas into the arrays the simulation needs."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import read_image


def load_real(lesion_paths, atlas_path, atlas_names=None, mask_path=None, labels=None):
    """Binary lesion images (all on one grid, e.g. MNI 2 mm) and a label atlas on the same grid.

    Returns (lesions (patients x voxels) bool, regions (regions x voxels) bool, coords (voxels x 3) mm,
    region names, voxel index into the grid). Voxels are those inside the mask (default: inside any
    atlas region)."""
    atlas, affine = read_image(atlas_path)
    atlas = np.rint(atlas).astype(int)
    shape = atlas.shape
    if mask_path is not None:
        mask = read_image(mask_path)[0] > 0
    else:
        mask = atlas > 0
    index = np.flatnonzero(mask.ravel(order="F"))
    labels = labels if labels is not None else [v for v in np.unique(atlas) if v != 0]
    flat_atlas = atlas.ravel(order="F")[index]
    regions = np.array([flat_atlas == v for v in labels])
    if atlas_names is None:
        names = [f"label_{v}" for v in labels]
    elif isinstance(atlas_names, (list, tuple)):
        names = list(atlas_names)
    else:
        import pandas as pd
        t = pd.read_csv(atlas_names, header=None, sep=None, engine="python") if not str(atlas_names).endswith(
            (".xls", ".xlsx")) else pd.read_excel(atlas_names, header=None)
        lookup = dict(zip(t.iloc[:, 0].astype(int), t.iloc[:, 1].astype(str)))
        names = [lookup.get(v, f"label_{v}") for v in labels]
    lesions = np.zeros((len(lesion_paths), len(index)), dtype=bool)
    for i, p in enumerate(lesion_paths):
        img, _ = read_image(p)
        if img.shape[:3] != shape:
            raise ValueError(f"{p}: shape {img.shape} differs from the atlas's {shape}")
        lesions[i] = img.ravel(order="F")[index] > 0
    ijk = np.array(np.unravel_index(index, shape, order="F")).T
    coords = np.c_[ijk, np.ones(len(ijk))] @ affine[:3].T
    return lesions, regions, coords, names, index, shape, affine


def lesion_files(folder, patterns=("*.nii", "*.nii.gz", "*.img")):
    folder = Path(folder)
    return sorted(p for pat in patterns for p in folder.glob(pat))
