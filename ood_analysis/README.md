# SC-MAPS out-of-distribution (OOD) analysis

**Drop this `ood_analysis/` directory unchanged into the existing SC-MAPS GitHub repository.** These are standalone Python scripts; you do not need to change, split or refactor the original training scripts. This folder is a **reconstructed, method-matched release implementation**, not a recovered byte-for-byte copy of the original OOD run.

## Contents and order

| Script | Purpose |
|---|---|
| `01_extract_ood_features_ARC.py` | Load the original **13-species, four-channel DenseNet121 concatenation** checkpoint, read the existing `.npy` crops, export probabilities and penultimate features for known reference/calibration/holdout and five unseen organisms. |
| `02_ood_scoring_and_population.py` | Standardise and PCA-reduce known-reference features; compare kNN, class-conditional Mahalanobis, entropy and MSP; calibrate on **known validation cells only**; evaluate held-out known and five unseen organisms; aggregate OOD scores over 1/5/10/20/50 cells with 200 repeats. |
| `03_plot_ood_figures.py` | Save OOD ROC curves, per-organism rejection, population rejection and known-population retention in PDF/PNG. |

## Inputs

**Known SC-MAPS dataset:** `Trillian_DATASET_with_holdout/` with:

```
Trillian_DATASET_with_holdout/
  splits/train.csv
  splits/val.csv
  splits/holdout_plate1.csv
  crops/.../*.npy   (actual crop locations follow the CSV)
```

**Additional unseen-organism dataset:** `SC_MAPS_NOVEL_SPECIES_SINGLE_CELL_DATASET/` with `cell_manifest.csv` and spatially matched 5-channel `.npy` crops. Class codes: `BSB`, `EF`, `EFM`, `HFLU` (historical `HELU` accepted), `LMON`.

Each CSV must have `species` and `crop_path` columns (or a supported alias). Both sets use **5 × 96 × 96** crops stored in **PhaseContrast, BF, DAPI, WGA, FM1** order. The model uses PhaseContrast, DAPI, WGA and FM1, with per-crop z-score normalisation **including the zero-padded background**. `sample_id` is preserved when present, otherwise constructed from acquisition metadata/filenames. All population results require correct acquisition-level sample IDs.

**Checkpoint (required):** the exact original four-channel, 13-species DenseNet121 concatenation `best_fusion_model.pt`. On Oxford ARC the previously used location is:

```
/data/phys-deep-learning-bacteria/phys3115/Bacteria-data-15 species/Trillian/DenseNet_Concat_weights/best_fusion_model.pt
```

**Do not** substitute a directly trained 8-class or SE-attention model, or a three-channel fusion checkpoint.

## Install

```bash
python -m pip install torch torchvision numpy pandas scipy scikit-learn matplotlib
```

Run from inside this folder on Oxford ARC, after both ZIP datasets have been extracted:

```bash
python 01_extract_ood_features_ARC.py \
  --checkpoint '/data/phys-deep-learning-bacteria/phys3115/Bacteria-data-15 species/Trillian/DenseNet_Concat_weights/best_fusion_model.pt' \
  --known-root /path/to/Trillian_DATASET_with_holdout \
  --unseen-root /path/to/SC_MAPS_NOVEL_SPECIES_SINGLE_CELL_DATASET \
  --outdir ood_outputs/input_arrays

python 02_ood_scoring_and_population.py \
  --input-dir ood_outputs/input_arrays \
  --output-dir ood_outputs/results \
  --pca-dim 50 --knn-neighbours 25 --retention 0.95 --repeats 200

python 03_plot_ood_figures.py --results ood_outputs/results
```

To process **all** available data instead of the default capped 2,000 cells per species in each cohort, append to Script 01:

```
--reference-per-species 0 --calibration-per-species 0 --known-test-per-species 0 --unseen-per-species 0
```

The default caps are intentional for a fast reproducibility check, **not** a claim that the published analyses used this exact subsampling. Large runs are GPU/CPU/RAM-intensive. Inputs are never changed, and each script can be rerun without retraining the classifier.

## Outputs

- `ood_outputs/input_arrays/{reference,calibration,known_test,unseen}.npz`: fixed model features and 13-species probabilities.
- `ood_outputs/input_arrays/provenance.json`: checkpoint SHA-256, class/channel order, sampling limits, manifests.
- `ood_outputs/results/ood_single_cell_metrics.csv`: AUROC, AUPRC, known retention and unseen rejection for all four methods.
- `ood_outputs/results/ood_per_organism_rejection.csv`: BSB, EF, EFM, HFLU, LMON rejection rates.
- `ood_outputs/results/ood_threshold_sensitivity.csv`: 90%, 95%, 99% calibration operating points.
- `ood_outputs/results/ood_cell_scores_*.csv`: per-cell scores and provenance.
- `ood_outputs/results/ood_population_metrics.csv`: 1/5/10/20/50-cell population rejection and known retention.
- `ood_outputs/results/figures/*.pdf` / `*.png`: four figures.

## Scientific and reproducibility notes

1. **No train/holdout mixing:** PCA, StandardScaler, nearest-neighbour reference set and class-conditional covariances use **known training cells only**; OOD thresholds are fitted to **known validation cells only**. Plate-1 strains are only evaluated, as are all five unseen organisms.
2. Features are taken from the **hidden representation immediately before the final species classification layer**, as in the original `SC_MAPS_CHRIS_REVIEW_ALL_IN_ONE_ARC_FIXED2.py` implementation.
3. OOD direction is high score = more unfamiliar. `kNN` = mean Euclidean distance to 25 known reference cells in PCA space. Mahalanobis = minimum class-specific shrinkage-covariance distance in the same space. MSP = 1 minus maximum softmax probability; entropy is normalised by `log(13)`.
4. For **population-level evaluation**, sampling is without replacement *within* each population and repeated 200 times per sample/size. Repeats are not independent biological replicates. **Population-size-specific cutoffs** are calibrated using known validation populations at each size rather than applying the single-cell cutoff blindly to averages. This detail should be reconciled with the original analysis settings before claiming exact reproduction of the published rates.
5. **Exact manuscript numbers are not guaranteed.** The current manuscript reports kNN AUROC 0.944, 77.8% unseen rejection and 94.3% known retention, with markedly greater rejection after population averaging. Reproducing those values may require the exact original reference, calibration and test sampling decisions, and the same saved checkpoint. The scripts do **not** insert these numbers in the outputs or plots.
6. Store `ood_outputs/` separately from GitHub where large feature arrays are generated; GitHub should contain the scripts and the download links for the 13-species and five-unseen-organism datasets, not the huge generated arrays.
