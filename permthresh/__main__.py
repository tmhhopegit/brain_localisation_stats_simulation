"""Command line.

    python -m permthresh simulate [--patients 600] [--perms 1000] [--seed 0] --out results/
    python -m permthresh real --lesions FOLDER --atlas ATLAS.nii [--names NAMES.csv] [--mask MASK.nii]
                              [--perms 1000] --out results/

Both write results.csv (one row per critical region x impairment rule x statistic x alpha),
summary.csv, null_calibration.csv, run_info.json and four figures.
"""
from __future__ import annotations

import argparse


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m permthresh", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("simulate", "real"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--perms", type=int, default=1000)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--thresholds", type=float, nargs="+", default=[0.25, 0.5, 0.75],
                       help="lesion-load thresholds for 'impaired' (the original used 0.5 and 1)")
        p.add_argument("--probabilities", type=float, nargs="+", default=[1.0, 0.5])
        p.add_argument("--statistics", nargs="+", default=["pm3", "partial_r", "ridge"],
                       choices=["pm3", "partial_r", "ridge"])
        p.add_argument("--calibration", type=int, default=100, help="null datasets for the calibration check")
        if name == "simulate":
            p.add_argument("--patients", type=int, default=600)
        else:
            p.add_argument("--lesions", required=True, help="folder of binary lesion images")
            p.add_argument("--atlas", required=True)
            p.add_argument("--names")
            p.add_argument("--mask")
    a = ap.parse_args(argv)
    from .report import analyse
    common = dict(n_perm=a.perms, load_thresholds=tuple(a.thresholds), probabilities=tuple(a.probabilities),
                  statistics=tuple(a.statistics), calibration_datasets=a.calibration, seed=a.seed)
    if a.cmd == "simulate":
        from .lesions import simulate
        pop = simulate(a.patients, seed=a.seed)
        analyse(pop.lesions, pop.regions, pop.coords, pop.region_names, pop.brain_index, pop.shape, pop.affine,
                a.out, **common)
    else:
        from .data import lesion_files, load_real
        files = lesion_files(a.lesions)
        if not files:
            raise SystemExit(f"no .nii / .nii.gz / .img files in {a.lesions}")
        L, R, C, names, index, shape, affine = load_real(files, a.atlas, a.names, a.mask)
        analyse(L, R, C, names, index, shape, affine, a.out, **common)
    print(f"results written to {a.out}")


if __name__ == "__main__":
    main()
