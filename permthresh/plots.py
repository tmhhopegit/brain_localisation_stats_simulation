"""Figures for the simulation results (matplotlib, static PNGs)."""
from __future__ import annotations

import numpy as np

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
STAT_COLOUR = {"pm3": "#2a78d6", "partial_r": "#eb6834", "ridge": "#1baf7a"}   # categorical slots 1-3
STAT_LABEL = {"pm3": "PM3 (original)", "partial_r": "Volume-controlled", "ridge": "Multivariate (ridge)"}
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def to_volume(flat, index, shape):
    out = np.full(int(np.prod(shape)), np.nan, dtype=np.float32)
    out[index] = flat
    return out.reshape(shape, order="F")


def _slices(shape, affine, z_mm):
    return [int(round((z - affine[2, 3]) / affine[2, 2])) for z in z_mm]


def _brain_outline(ax, brain2d):
    ax.contour(brain2d.T, levels=[0.5], colors=[GRID], linewidths=0.8, origin="lower")


def overlap_figure(lesions, brain_index, tested_index, shape, affine, z_mm=(-6, 6, 18, 30, 42), path=None):
    """Lesion overlap (% of patients), with the tested voxels (lesioned in >= 10%) outlined."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cov = lesions.mean(0) * 100
    vol = to_volume(cov, brain_index, shape)
    brain = to_volume(np.ones(len(brain_index)), brain_index, shape)
    tested = np.zeros(int(np.prod(shape)))
    tested[brain_index[tested_index]] = 1
    tested = tested.reshape(shape, order="F")
    ks = _slices(shape, affine, z_mm)
    cmap = ListedColormap(BLUES)
    cmap.set_bad(SURFACE)
    fig, axes = plt.subplots(1, len(ks), figsize=(2.3 * len(ks), 3.0), facecolor=SURFACE)
    vmax = max(np.nanmax(cov), 1)
    for ax, k, z in zip(axes, ks, z_mm):
        v = vol[:, :, k].copy()
        v[v == 0] = np.nan
        im = ax.imshow(v.T, origin="lower", cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        _brain_outline(ax, np.nan_to_num(brain[:, :, k]))
        ax.contour(tested[:, :, k].T, levels=[0.5], colors=[INK], linewidths=0.9, origin="lower")
        ax.set_title(f"z = {z} mm", fontsize=9, color=INK2)
        ax.axis("off")
    axes[0].text(0.02, 0.02, "R", transform=axes[0].transAxes, color=INK2, fontsize=8)
    axes[0].text(0.95, 0.02, "L", transform=axes[0].transAxes, color=INK2, fontsize=8)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("% of patients lesioned", fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    fig.suptitle("Simulated lesion overlap; black outline = voxels tested (lesioned in at least 10% of patients)",
                 fontsize=10, color=INK, x=0.02, ha="left")
    if path:
        fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    return fig


def _project(vol3d, axis):
    return np.nanmax(np.nan_to_num(vol3d), axis=axis)


def example_figure(masks: dict, region_flat, brain_index, tested_index, shape, affine, title, path=None):
    """Glass-brain projections (axial, coronal, sagittal): every significant voxel shows.
    One row per statistic; significant voxels in the statistic's colour, tested voxels light grey,
    the true region outlined in ink."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    brain = to_volume(np.ones(len(brain_index)), brain_index, shape)
    region = to_volume(region_flat.astype(float), brain_index, shape)
    tested = np.zeros(len(brain_index))
    tested[tested_index] = 1
    tested = to_volume(tested, brain_index, shape)
    views = (("Axial (from above)", 2, lambda m: m.T), ("Coronal (from behind)", 1, lambda m: m.T),
             ("Sagittal (from the left)", 0, lambda m: m.T[:, ::-1]))
    fig, axes = plt.subplots(len(masks), 3, figsize=(10, 3.3 * len(masks)), facecolor=SURFACE, squeeze=False,
                             gridspec_kw={"width_ratios": [shape[0], shape[0], shape[1]]})
    for row, (stat, mask_tested) in enumerate(masks.items()):
        sig = np.zeros(len(brain_index))
        sig[tested_index[mask_tested]] = 1
        sig = to_volume(sig, brain_index, shape)
        for col, (name, axis, orient) in enumerate(views):
            ax = axes[row, col]
            b = orient(_project(brain, axis))
            t = orient(_project(tested, axis)).astype(float)
            t[t == 0] = np.nan
            sg = orient(_project(sig, axis)).astype(float)
            sg[sg == 0] = np.nan
            ax.imshow(t, origin="lower", cmap=ListedColormap([GRID]), interpolation="nearest")
            ax.imshow(sg, origin="lower", cmap=ListedColormap([STAT_COLOUR[stat]]), interpolation="nearest")
            ax.contour(b, levels=[0.5], colors=[INK2], linewidths=0.6, origin="lower")
            ax.contour(orient(_project(region, axis)), levels=[0.5], colors=[INK], linewidths=1.6, origin="lower")
            ax.axis("off")
            if row == 0:
                ax.set_title(name, fontsize=9, color=INK2)
        n = int(mask_tested.sum())
        axes[row, 0].text(-0.06, 0.5, f"{STAT_LABEL[stat]}\n{n} voxels significant", transform=axes[row, 0].transAxes,
                          rotation=90, va="center", ha="right", fontsize=9, color=INK)
    axes[0, 0].text(0.02, 0.02, "R", transform=axes[0, 0].transAxes, color=INK2, fontsize=8)
    axes[0, 0].text(0.94, 0.02, "L", transform=axes[0, 0].transAxes, color=INK2, fontsize=8)
    fig.suptitle(title, fontsize=10, color=INK, x=0.02, ha="left")
    fig.text(0.02, 0.0, "Black outline: the region that truly causes the deficit.  Grey: voxels tested.  "
             "Colour: significant after family-wise correction (p < .05), projected through the brain.",
             fontsize=8, color=INK2)
    if path:
        fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    return fig


def summary_figure(table, alpha=0.05, probability=1.0, path=None):
    """Per critical region and load threshold: share of significant voxels outside the true region,
    and how far the significant cluster's centre is from the region's centre. One panel per measure."""
    import matplotlib.pyplot as plt
    d = table[(~table.skipped) & (table.alpha == alpha) & (table.probability == probability)].copy()
    d["label"] = d.region + "  ≥" + (d.load_threshold * 100).astype(int).astype(str) + "%"
    order = (d[d.statistic == "pm3"].sort_values("patients_lesioned").label.tolist())
    ypos = {lab: i for i, lab in enumerate(order)}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 0.28 * len(order) + 1.6), facecolor=SURFACE, sharey=True)
    panels = (("fwe_fdp", "Share of significant voxels outside the true region", (0, 1.02)),
              ("fwe_centroid_shift_mm", "Distance from true region's centre to\nsignificant voxels' centre (mm)", None))
    stats_present = [st for st in STAT_COLOUR if st in set(d.statistic)]
    offs = dict(zip(stats_present, np.linspace(-0.22, 0.22, len(stats_present)) if len(stats_present) > 1 else [0]))
    for ax, (col, lab, lim) in zip(axes, panels):
        _style(ax)
        for stat in stats_present:
            s = d[d.statistic == stat]
            y = np.array([ypos[l] for l in s.label]) + offs[stat]
            vals = s[col].to_numpy()
            ok = np.isfinite(vals)
            ax.scatter(vals[ok], y[ok], s=34, color=STAT_COLOUR[stat], edgecolor=SURFACE, linewidth=1.5, zorder=3,
                       label=STAT_LABEL[stat])
            ax.scatter(np.zeros((~ok).sum()) if lim else np.zeros((~ok).sum()), y[~ok], s=34, marker="x",
                       color=STAT_COLOUR[stat], zorder=3, linewidth=1.2)
        ax.set_xlabel(lab, fontsize=8.5, color=INK2)
        if lim:
            ax.set_xlim(*lim)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(order, fontsize=7.5)
    axes[0].set_ylabel("True critical region, impairment rule (least → most often lesioned)", fontsize=8.5, color=INK2)
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(f"Where the significant voxels are (family-wise p < {alpha}; × = nothing significant)",
                 fontsize=10, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    return fig


def maxima_figure(coverage, max_counts: dict, path=None):
    """How over-represented voxels of each lesion frequency are among the permutation maxima
    (share of maxima / share of voxels; 1 = its fair share)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.6), facecolor=SURFACE)
    _style(ax)
    edges = np.quantile(coverage, np.linspace(0, 1, 9))
    edges[-1] += 1e-9
    idx = np.clip(np.searchsorted(edges, coverage, side="right") - 1, 0, len(edges) - 2)
    centres = np.array([coverage[idx == b].mean() * 100 for b in range(len(edges) - 1)])
    for stat, counts in max_counts.items():
        frac = counts / counts.sum()
        ratio = np.array([frac[idx == b].sum() / (idx == b).mean() for b in range(len(edges) - 1)])
        ax.plot(centres, ratio, color=STAT_COLOUR[stat], lw=2, marker="o", ms=8, mec=SURFACE, mew=2,
                label=STAT_LABEL[stat])
    ax.axhline(1, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.set_xlabel("% of patients lesioned at the voxel (eighths of the tested voxels)", fontsize=8.5, color=INK2)
    ax.set_ylabel("Share of permutation maxima\n÷ share of voxels", fontsize=8.5, color=INK2)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Which voxels set the family-wise threshold (1 = their fair share)", fontsize=10, color=INK,
                 x=0.02, ha="left")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    return fig
