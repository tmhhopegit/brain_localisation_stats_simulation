"""Run the whole analysis and write tables, figures and a short summary to a folder."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .simulation import null_calibration, prepare, run


def summarise(table: pd.DataFrame) -> pd.DataFrame:
    """Mean over critical regions and impairment rules, per statistic x alpha x threshold type."""
    d = table[~table.skipped]
    rows = []
    for (stat, a, p), g in d.groupby(["statistic", "alpha", "probability"]):
        for kind in ("fwe", "voxel"):
            found = (g[f"{kind}_in"] + g[f"{kind}_out"]) > 0
            rows.append({"statistic": stat, "alpha": a, "probability": p, "threshold": kind,
                         "analyses": len(g), "anything_significant": float(found.mean()),
                         "region_detected": float((g[f"{kind}_in"] > 0).mean()),
                         "mean_sensitivity": float(g[f"{kind}_sensitivity"].mean()),
                         "mean_share_outside": float(g[f"{kind}_fdp"].mean()),
                         "median_centroid_shift_mm": float(g[f"{kind}_centroid_shift_mm"].median()),
                         "mean_voxels_outside": float(g[f"{kind}_out"].mean()),
                         "top_region_correct": float(g.top_region_correct.mean()) if kind == "fwe" else np.nan,
                         "peak_in_region": float(g.peak_in_region.mean())})
    return pd.DataFrame(rows)


def analyse(lesions, regions, coords, names, brain_index, shape, affine, out, n_perm: int = 1000,
            load_thresholds=(0.25, 0.5, 0.75), probabilities=(1.0, 0.5), alphas=(0.05, 0.01, 0.001),
            statistics=("pm3", "partial_r", "ridge"), calibration_datasets: int = 100, example_region=None,
            seed: int = 0, verbose: bool = True) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    prep = prepare(lesions, regions, coords, names)
    if verbose:
        print(f"{prep.lesions.shape[0]} patients with lesions; {prep.lesions.shape[1]} voxels tested; "
              f"{len(prep.region_names)} candidate critical regions", flush=True)
    table, by_region, vox_max = run(prep, n_perm, load_thresholds, probabilities, alphas, seed=seed,
                                    statistics=statistics, verbose=verbose)
    table.to_csv(out / "results.csv", index=False)
    summ = summarise(table)
    summ.to_csv(out / "summary.csv", index=False)
    cal = [dict(statistic=st, **null_calibration(prep, calibration_datasets, 500, 0.05, statistic=st, seed=seed))
           for st in statistics] if calibration_datasets else []
    pd.DataFrame(cal).to_csv(out / "null_calibration.csv", index=False)

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tested = prep.voxel_index
    plots.overlap_figure(np.asarray(lesions, bool)[np.asarray(lesions, bool).any(1)], brain_index, tested, shape,
                         affine, path=out / "fig1_lesion_overlap.png")
    plt.close("all")
    d = table[(~table.skipped) & (table.probability == 1.0) & (table.alpha == 0.05)]
    if example_region is None and len(d):
        # the region with the most impaired patients when "impaired" = at least half of it lesioned
        cand = d[(d.load_threshold == 0.5)]
        example_region = (cand if len(cand) else d).sort_values("n_impaired").region.iloc[-1]
    if example_region is not None:
        from .simulation import lesion_deficit_map
        r = prep.region_names.index(example_region)
        t = 0.5
        impaired = prep.loads[:, r] >= t
        masks = {st: lesion_deficit_map(prep, impaired, st, n_perm, seed).fwe_mask(0.05) for st in statistics}
        full_region = np.asarray(regions, bool)[prep.region_ids[r]]
        plots.example_figure(masks, full_region, brain_index, tested, shape, affine,
                             f"Example: region {example_region} is made critical (impaired = lesion covers at least "
                             f"half of it; {int(impaired.sum())} patients)", path=out / "fig2_example_maps.png")
        plt.close("all")
    plots.summary_figure(table, path=out / "fig3_where_significant.png")
    plt.close("all")
    plots.maxima_figure(prep.lesions.mean(0), vox_max, path=out / "fig4_permutation_maxima.png")
    plt.close("all")
    info = {"patients": int(prep.lesions.shape[0]), "voxels_tested": int(prep.lesions.shape[1]),
            "regions": prep.region_names, "n_perm": n_perm, "example_region": example_region,
            "calibration": cal}
    (out / "run_info.json").write_text(json.dumps(info, indent=2))
    return {"table": table, "summary": summ, "calibration": cal, "prepared": prep, "by_region": by_region,
            "voxel_maxima": vox_max, "info": info}
