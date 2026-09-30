# permthresh

This is an analysis of the effectiveness of voxel-wise lesion-deficit mapping, using simulated lesion-deficit data: i.e., a container to test how well various univariate ands multivariate statistical approaches can localise impairments to the region where damage 'really' causes that impairment.

The simulation makes each region in turn the "critical" one: a patient is impaired if their lesion covers at least a set fraction of it. It then counts significant voxels inside and outside that region under two thresholds:
- **family-wise:** a permutation max-statistic threshold, corrected across the whole brain;
- **voxel-wise:** uncorrected, voxel by voxel.

```
pip install numpy scipy pandas matplotlib
python -m permthresh simulate --out results                 # about 1 minute
python -m permthresh real --lesions FOLDER --atlas atlas.nii --names atlas_names.csv --out results_real
pytest tests                                                # 9 tests
```

## Layout

```
refactored/
  permthresh/
    lpd.py          PM3 map (MakeLPDMap), permutation null (MakeLPDMap_Thresholded), family-wise and
                    voxel-wise thresholds; plus a volume-controlled statistic (partial correlation)
    multivariate.py the better control: ridge regression on clusters of voxels, volume as a
                    covariate, penalty chosen by cross-validation, permutation max-statistic threshold
    simulation.py   prepare (the original's filters), run (the region-as-critical loop),
                    null_calibration (checks the thresholds under the null)
    lesions.py      a simulated stroke population whose lesions follow arterial trees, and an atlas
    data.py         load real lesion images and a label atlas
    report.py       run everything; write tables, figures, run_info.json
    plots.py        the figures
    io.py           NIfTI / Analyze reading and writing (copied from the ploras package)
    __main__.py     command line
  results/                  600 simulated patients (the main results below)
  results_150_patients/     the same with 150 patients
  tests/test_permthresh.py
```

Each results folder contains:
- `results.csv`: one row per critical region × impairment rule × statistic × alpha;
- `summary.csv`: averages over regions;
- `null_calibration.csv`;
- `run_info.json`;
- four figures.

## Results (simulated data)

These results come from a simulated population, built to share the key property of real strokes: lesions fill the territory downstream of a blocked artery. See `permthresh/lesions.py`.
- There are 600 patients with left-hemisphere strokes in MCA, PCA, ACA and deep territories.
- Each lesion is the territory of one branch of a branching arterial tree. Proximal occlusions are rarer and larger, and part of each territory survives.
- Median lesion volume is 15 cm³ (IQR 4–46), and peak lesion overlap is 22%. Both are in the range of real stroke cohorts.
- The atlas is an anatomical parcellation into 64 similar-sized regions, made without reference to the arteries (like AAL).
- 4,425 voxels (3 mm) were lesioned in at least 10% of patients and tested. 10 regions qualified as critical regions.
- Each region was run with 3 load thresholds × 2 detection probabilities; 23 of these analyses had at least 5 impaired patients.
- Every analysis used 1,000 permutations.

When impairment was random (100 null datasets), the family-wise threshold found anything in 1–3% of datasets for every method (it should be at most 5%). So the widespread significance below is a real effect of lesion anatomy, not a thresholding error.

### How the original analysis behaves

Family-wise p < .05, 600 patients, averaged over regions and impairment rules (p = 1):

| | PM3 (original) | volume-controlled |
|---|---|---|
| true region detected | 100% of analyses | 74% |
| share of the true region found | 99% | 59% |
| **share of significant voxels outside the true region** | **88%** | **72%** |
| significant voxels outside the true region (mean) | 3,840 of 4,425 tested | 1,593 |
| distance from true region's centre to the significant voxels' centre (median) | 22 mm | 20 mm |
| most significant voxel inside the true region | 61% | 65% |

1. **PM3 marks almost everything as significant.** Whichever region truly causes the deficit, nearly every tested voxel is significant, and 88% of significant voxels lie outside the true region. A patient only has half of a region destroyed if they had a large, proximal stroke. The impaired patients therefore have larger lesions everywhere, and PM3, a raw difference in lesion rates, picks that up. Even the single most significant voxel is outside the true region 39% of the time.
2. **Controlling for lesion volume helps, but not enough.** The share of significant voxels outside the true region falls from 88% to 72%. The rest comes from the arterial tree. Voxels on the same branch are lesioned together, so a region that lies downstream of others "drags" them along. That is the displacement described by Mah et al. (Brain, 2014), and volume control can't remove it.
3. **Family-wise vs voxel-wise thresholds.** For PM3, the corrected and uncorrected maps are almost identical (88% outside in both). The problem isn't the multiple-comparison correction: the statistic is confounded, so a stricter threshold doesn't help.
4. **Noisier deficits** (p = 0.5: only half of those with the damage are impaired) barely change PM3: still 88% outside. **Fewer patients (150)** give the same picture: PM3 89% outside, volume-controlled 69%.

## A better statistical control

Both univariate maps ask, voxel by voxel, "is damage here associated with the deficit?". Voxels on the same branch are damaged together, so all of them are associated. The question that separates them is conditional: "does damage here add anything once damage everywhere else, and lesion size, are known?". That needs a multivariate model, and the model has to be one whose errors can still be controlled.

### What works: `ridge` (in `multivariate.py`)

1. **Cluster the voxels.** Group the tested voxels into clusters of neighbours with similar lesion patterns (spatially constrained Ward clustering; by default about one cluster per four patients, at most 300). Voxels whose lesion patterns are near-identical carry the same information, so no method can tell them apart; they are tested together.
2. **Remove lesion volume.** Residualise impairment and each cluster's lesion load on lesion volume.
3. **Fit one model.** Regress impairment on all clusters at once with a ridge penalty. The penalty is chosen by 10-fold cross-validated prediction error, which never looks at where the effect is.
4. **Threshold by permutation.** Permute the residualised impairment and refit (Freedman–Lane). The fit is linear in the outcome, so 1,000 permutations are one matrix product. A cluster is significant if its coefficient beats the 95th percentile of the largest coefficient in each permutation: the same max-statistic logic as your original, applied to a conditional statistic.

Family-wise p < .05; 600 patients unless stated:

| | PM3 | volume-controlled | **multivariate ridge** |
|---|---|---|---|
| **region with the most significant voxels is the true one** | 13% | 43% | **65%** |
| same, among analyses where anything was significant | 13% | 56% | **83%** |
| **significant voxels outside the true region** (mean) | 3,840 | 1,593 | **23** |
| share of significant voxels outside the true region | 88% | 72% | **28%** |
| distance of significant voxels' centre from the truth (median) | 22 mm | 20 mm | **10 mm** |
| regions containing significant voxels (mean) | 10.0 | 6.0 | **2.3** |
| true region detected at all | 100% | 74% | 74% |
| share of the true region found | 99% | 59% | 16% |
| null family-wise error rate (should be ≤ 5%) | 2% | 3% | 1% |
| noisier deficits (p = 0.5): share outside / top region correct | 88% / 13% | 58% / 52% | 28% / 65% |
| 150 patients: share outside / top region correct | 89% / 14% | 69% / 43% | 49% / 52% |

**What it gains.**
- Significant voxels outside the true region drop from thousands to a few dozen.
- The region with the most significant voxels is the true one 1.5 times as often as with the volume-controlled map, and 5 times as often as with PM3.
- When the ridge map finds anything, it points at the right region 83% of the time (94% with the noisier deficits), and the significant cluster sits half as far from the truth.

**What it gives up.**
- It finds only a core of the true region: 16% of its voxels, against 99% for PM3.
- It finds nothing in about a quarter of analyses.

That trade is inherent in the conditional question. Once the rest of the region is in the model, each part of it adds little, so only the parts that carry unique information pass. Read the ridge map as "damage here matters, beyond everything it co-occurs with", not as the full extent of the critical area.

**At 150 patients** the advantage shrinks: 49% of significant voxels outside, and the true region top-ranked in 52% of analyses, against 43% for the volume-controlled map. With fewer patients, fewer lesion configurations separate the true region from its neighbours. Precision still beats both univariate maps by a wide margin (86 voxels outside, against 834 and 3,788).

**Tuning.**
- The cluster count trades precision against coverage. At 600 patients:

  | clusters | share of true region found | significant voxels outside |
  |---|---|---|
  | 150 | 17–23% | 25–40% |
  | 500 | 6–12% | 19–31% |

- Change it with `Prepared.get_clusters(n_clusters)`.
- The cross-validated penalty had a median of about 560 but ranged from 56 to 100,000 across analyses. Stronger shrinkage makes ridge more like the univariate maps, and weaker shrinkage makes it noisier.

### Caveats

- The lesions are simulated. Real vascular trees, collateral flow and white-matter damage are more varied, so the exact numbers won't transfer. But the core pattern of the results should be consistent, since the simulated lesions follow arterial branches, just as real ones do.
- The impairment rules are deterministic thresholds on region damage, even with p = 0.5. Real deficits are likely to be graded and noisier at least, and might well be just much more complex (e.g., driven by network disruption).
- Family-wise control by the permutation max-statistic is exact under the complete null. When an effect is present, it doesn't guarantee that other clusters are protected, for any of these methods. That is exactly why the share outside was measured directly.
- The ridge model is a linear probability model. It is used only to rank clusters, since the permutations provide the calibration.

**Figures** (in `results/`):
- `fig1_lesion_overlap.png`: the simulated lesion overlap and the tested voxels.
- `fig2_example_maps.png`: one region made critical (L23; 105 impaired patients). Glass-brain views of what each method calls significant: PM3 4,425 voxels, volume-controlled 4,249, ridge 16, all inside the true region.
- `fig3_where_significant.png`: for every critical region and impairment rule, the share of significant voxels outside the true region, and how far the significant cluster's centre is from the region's.
- `fig4_permutation_maxima.png`: which voxels set the family-wise threshold. The least often lesioned eighth of the tested voxels holds about 3 times its share of the permutation maxima for the univariate maps, and 1.6 times for ridge.
