import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLBACKEND", "Agg")

from permthresh.io import write_image  # noqa: E402
from permthresh.lesions import simulate  # noqa: E402
from permthresh.lpd import permutation_lpd, pm3_map  # noqa: E402
from permthresh.simulation import null_calibration, prepare, run  # noqa: E402


def matlab_lpd(lesions, impaired):
    """MakeLPDMap.m, line for line."""
    ks = lesions.T
    nd, nnd = impaired.sum(), (~impaired).sum()
    block = np.tile(impaired, (ks.shape[0], 1))
    return (ks & block).sum(1) / nd - (ks & ~block).sum(1) / nnd


def random_data(n=60, v=40, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, v)) < 0.3, rng.random(n) < 0.4


def test_pm3_matches_the_original_formula():
    L, imp = random_data()
    assert np.allclose(pm3_map(L, imp), matlab_lpd(L, imp), atol=1e-6)
    with pytest.raises(ValueError):
        pm3_map(L, np.zeros(len(imp), bool))


def test_permutation_p_values_match_a_direct_loop():
    L, imp = random_data(v=15)
    res = permutation_lpd(L, imp, n_perm=300, seed=4)
    rng = np.random.default_rng(4)
    n, nd = len(imp), imp.sum()
    maxes, exceed = [], np.zeros(L.shape[1])
    for _ in range(300):
        perm = np.zeros(n, bool)
        perm[rng.permutation(n)[:nd]] = True
        m = matlab_lpd(L, perm)
        maxes.append(m.max())
        exceed += m >= res.map - 1e-6
    assert np.allclose(np.sort(maxes), np.sort(res.null_max), atol=1e-6)
    assert np.allclose((exceed + 1) / 301, res.p_voxel)
    assert res.fwe_mask(0.05).sum() == (res.map > np.quantile(maxes, 0.95, method="higher")).sum()
    assert np.all(res.fwe_p() >= res.p_voxel - 1e-12)


def test_partial_r_is_the_volume_partial_correlation():
    L, imp = random_data(v=10)
    vol = L.sum(1).astype(float)
    res = permutation_lpd(L, imp, n_perm=20, statistic="partial_r", volume=vol)
    Z = np.c_[np.ones(len(vol)), vol]
    res_y = imp - Z @ np.linalg.lstsq(Z, imp.astype(float), rcond=None)[0]
    for v in (0, 3, 9):
        x = L[:, v].astype(float)
        rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
        assert np.isclose(res.map[v], np.corrcoef(rx, res_y)[0, 1], atol=1e-5)


def test_null_calibration_controls_family_wise_error():
    pop = simulate(120, seed=1)
    prep = prepare(pop.lesions, pop.regions, pop.coords, pop.region_names)
    for stat in ("pm3", "partial_r", "ridge"):
        cal = null_calibration(prep, n_datasets=40, n_perm=200, statistic=stat, seed=2)
        assert cal["familywise_error_rate"] <= 0.15


def test_prepare_filters_like_the_original():
    L = np.zeros((10, 300), bool)
    L[:5, :150] = True                                   # 5 patients lesioned in voxels 0-149
    L[5, 200] = True                                     # one patient with a single rare voxel
    regions = np.zeros((3, 300), bool)
    regions[0, :120] = True                              # 120 voxels, all testable
    regions[1, 120:200] = True                           # 30 testable voxels (< 100)
    regions[2, 200:300] = True                           # none testable
    coords = np.c_[np.arange(300.0), np.zeros(300), np.zeros(300)]
    prep = prepare(L, regions, coords)
    # 4 patients without lesions dropped; voxels lesioned in >= ceil(10% of 6) = 1 patient kept
    assert prep.lesions.shape == (6, 151)
    assert prep.region_names == ["region_0"]
    assert np.allclose(prep.loads[:5, 0], 1.0) and prep.loads[5, 0] == 0
    assert np.allclose(prep.volume, [150] * 5 + [1])


def test_the_simulation_finds_a_planted_region_and_skips_impossible_rules():
    pop = simulate(200, seed=3)
    prep = prepare(pop.lesions, pop.regions, pop.coords, pop.region_names)
    table, by_region, vox = run(prep, n_perm=200, load_thresholds=(0.5, 1.0), statistics=("pm3", "partial_r", "ridge"))
    done = table[~table.skipped]
    pm3 = done[(done.statistic == "pm3") & (done.alpha == 0.05)]
    assert (pm3[pm3.n_impaired >= 10].fwe_in > 0).all()          # found whenever >= 10 patients are impaired
    assert table[table.load_threshold == 1.0].skipped.all()       # nobody's lesion covers a whole region
    assert set(vox) == {"pm3", "partial_r", "ridge"} and by_region.shape[1] == len(prep.region_names) + 1
    assert {"fwe_fdp", "voxel_centroid_shift_mm", "peak_in_region", "top_region_correct"} <= set(table.columns)
    # the multivariate map puts far fewer significant voxels outside the true region than PM3
    f = done[done.alpha == 0.05]
    assert f[f.statistic == "ridge"].fwe_out.mean() < 0.2 * f[f.statistic == "pm3"].fwe_out.mean()


def test_simulated_population_is_left_hemisphere_and_reproducible():
    a, b = simulate(50, seed=7), simulate(50, seed=7)
    assert np.array_equal(a.lesions, b.lesions)
    assert not a.lesions[:, a.coords[:, 0] > 0].any()             # MNI x > 0 is right
    assert a.lesions.any(1).all()
    assert a.regions.sum(0).max() == 1                             # atlas regions do not overlap
    assert a.volume(a.lesions[0]).shape == a.shape


def test_real_data_loading_and_command_line(tmp_path):
    from permthresh.__main__ import main
    from permthresh.data import load_real
    pop = simulate(60, seed=5)
    (tmp_path / "lesions").mkdir()
    for i in range(60):
        write_image(tmp_path / "lesions" / f"p{i:02d}.nii", pop.volume(pop.lesions[i]), pop.affine)
    atlas = np.zeros(len(pop.brain_index))
    for r in range(len(pop.regions)):
        atlas[pop.regions[r]] = r + 1
    write_image(tmp_path / "atlas.nii", pop.volume(atlas), pop.affine)
    L, R, C, names, index, shape, affine = load_real(sorted((tmp_path / "lesions").glob("*.nii")),
                                                    tmp_path / "atlas.nii")
    assert np.array_equal(L, pop.lesions) and np.array_equal(R, pop.regions) and np.allclose(C, pop.coords)
    main(["real", "--lesions", str(tmp_path / "lesions"), "--atlas", str(tmp_path / "atlas.nii"), "--perms", "50",
          "--calibration", "5", "--thresholds", "0.5", "--probabilities", "1", "--out", str(tmp_path / "out")])
    for f in ("results.csv", "summary.csv", "null_calibration.csv", "run_info.json", "fig1_lesion_overlap.png",
              "fig3_where_significant.png", "fig4_permutation_maxima.png"):
        assert (tmp_path / "out" / f).exists(), f


def test_ridge_coefficients_and_clusters():
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    from permthresh.multivariate import _design, lesion_clusters, permutation_ridge
    pop = simulate(150, seed=6)
    prep = prepare(pop.lesions, pop.regions, pop.coords, pop.region_names)
    lab = lesion_clusters(prep.lesions, prep.coords, 40)
    assert lab.max() + 1 == 40
    from scipy.sparse import coo_matrix
    contiguous = 0
    for k in range(40):              # clusters are spatially contiguous, except where small islands of
        c = prep.coords[lab == k]    # tested voxels had to be joined to a neighbour
        pairs = cKDTree(c).query_pairs(3.01, output_type="ndarray")
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(c), len(c)))
        contiguous += connected_components(g, directed=False)[0] == 1
    assert contiguous >= 0.75 * 40
    y = prep.loads[:, 3] >= 0.25
    res = permutation_ridge(prep.lesions, y, lab, prep.volume, n_perm=100, lam=50.0)
    X, P = _design(prep.lesions, lab, prep.volume)
    beta = np.linalg.solve(X.T @ X + 50.0 * np.eye(X.shape[1]), X.T @ (P @ y.astype(float)))
    assert np.allclose(res.cluster_beta, beta, atol=1e-8)
    assert np.allclose(res.map, beta[lab])
    auto = permutation_ridge(prep.lesions, y, lab, prep.volume, n_perm=50)
    assert auto.lam > 0 and auto.fwe_mask(0.05).dtype == bool
