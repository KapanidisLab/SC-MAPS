# SC-MAPS — original Python scripts

**Single-Cell Multi-channel Analysis for antibiotic-Panel and Species classification**

These are the authors' original SC-MAPS analysis scripts, **organised by workflow**. Only their *folder locations* have changed. **The filenames and all lines of code are unchanged** from the supplied `Python.zip` (15 scripts included). There is no replacement framework, shared library, or new script required to run them.

## Repository layout

```text
SCMAPS_GitHub_Organised/
├── README.md
├── requirements.txt
├── .gitignore
├── training/
│   ├── species/           # Train 13-class single-channel and fusion models
│   └── ast_routing/       # Train direct 8-class AST-routing models
├── evaluation/            # Probability ensembles, species–routing comparisons, bootstrap
├── analysis/
│   ├── SC_MAPS_CHRIS_REVIEW_ALL_IN_ONE_ARC_FIXED2.py
│   ├── polymicrobial/     # Synthetic two-species populations
│   └── umap/              # Known and unseen-organism feature visualisations
└── supplementary/
    └── specialist/        # Specialist Enterobacterales experiments
```

**The scripts use the existing SC-MAPS single-cell `.npy` files.** Do not convert the image format or regenerate segmentation simply to use this repository.

## 1. Obtain the two datasets

Download the data **separately from ORA** here:

https://dx.doi.org/10.5287/ora-nz5e6m1bw


| Archive | Content | Where it is used |
|---|---|---|
| `Trillian_DATASET_with_holdout.zip` | The main 13-species SC-MAPS dataset, including per-cell five-channel `.npy` stacks, `cell_manifest.csv` and `splits/*.csv` | All training, independent-strain evaluation and polymicrobial analyses |
| `SC_MAPS_NOVEL_SPECIES_SINGLE_CELL_DATASET*.zip` | Five unseen species (`BSB`, `EF`, `EFM`, `HFLU`, `LMON`) | Unseen-organism forced predictions, feature extraction and UMAP analyses |


> Neither of the large dataset archives is included in this code ZIP. The five extra species must **not** be mixed with model-development data.

The main `.npy` stacks have five spatially matched channels, ordered **PhaseContrast, BF, DAPI, WGA, FM1-43**, and were prepared at **5 × 96 × 96** pixels. Use the supplied train, validation and `holdout_plate1.csv` files. Plate 1 is the independent-strain holdout. Do not randomly redistribute its individual cells into training.

One possible local arrangement is:

```text
SC-MAPS-project/
├── SCMAPS_GitHub_Organised/
│   └── training/, evaluation/, analysis/, supplementary/
└── data/
    ├── Trillian_DATASET_with_holdout.zip
    ├── SC_MAPS_NOVEL_SPECIES_SINGLE_CELL_DATASET.zip
    ├── Trillian_DATASET_with_holdout/          # if extracted by the user
    │   ├── cell_manifest.csv
    │   ├── splits/
    │   │   ├── train.csv
    │   │   ├── val.csv
    │   │   └── holdout_plate1.csv
    │   └── .../*.npy
    └── <unseen archive's actual extracted root>/
```

The *internal root folder* of each ZIP should be checked after download; avoid introducing accidental extra nesting when extracting. The ARC training scripts often handle ZIP copying/extraction themselves.

## 2. Python environment

Use the same compatible Python/PyTorch environment used for the original ARC experiments, if available. Otherwise, with Python 3.10+ and a PyTorch build suited to your machine:

```bash
python -m venv .venv
source .venv/bin/activate          # PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

GPU training uses considerable compute time and disk space; smaller CPU-only downstream analyses use previously saved prediction/feature files.

## 3. Configure the dataset paths

Most ARC training scripts read environment variables and/or have a short settings block near their beginning. They were originally written for the Oxford ARC `/data/.../Trillian` and `/scratch/...` filesystems. **If using a different machine, set the paths referenced by that individual script**. Many training scripts accept the following environment variables:

```bash
export PERSISTENT_DATA_PARENT="/your/data/directory"
export ZIP_PATH="/your/data/directory/Trillian_DATASET_with_holdout.zip"
export DATASET_FOLDER_NAME="Trillian_DATASET_with_holdout"
export SCRATCH_BASE="/your/large/scratch/directory"
```

Paths to already-trained channel checkpoints (`WEIGHTS_ROOT`) and fusion checkpoints must also be set for fusion training and inference. **The `.npy` dataset does not contain trained `.pt` weights** unless these are separately deposited. Either reproduce the earlier training steps or download the authors' model weights.

Some scripts use **hard-coded run folders** rather than environment variables: particularly the direct-panel bootstrap, specialist experiments, and parts of the post-hoc review pipeline. Open those scripts, find the existing `BASE`, `RUN_DIRS`, `DATA_ROOT`, `FUSION_CHECKPOINT_PATH` or similar config block, and edit only those paths as necessary. Keeping the original scripts unchanged here makes that requirement explicit.

## 4. Script order — what to run and why

The scripts below are **whole standalone programs**; these are not Python imports or modules to execute as one combined pipeline. Run only the analyses you need, working through their prerequisites.

### A. Training — species identification (13 classes)

| Order | Script | Purpose | Requires |
|---|---|---|---|
| A1 | `training/species/Train_SSRUN_multicondition_benchmark.py` | Trains/benchmarks individual channels and CNN architectures, with optional Optuna tuning | Main dataset ZIP and split CSVs |
| A2 | `training/species/Train_Trillian_SE_attention_feature_fusion_ARC.py` | Pretrained-branch DenseNet feature fusion with SE attention | A1 **DenseNet** single-channel checkpoints |
| A3 | `training/species/Train_Trillian_ResNet18_ResNet34_concat_and_SE_feature_fusion_ARC_final.py` | Trains concat and SE feature fusion for ResNet18/ResNet34 | A1 **ResNet** single-channel checkpoints |

For a complete five-channel single-channel benchmark, ensure all five channels are selected and that `MODEL_BENCHMARKS` includes `densenet121` (the A1 script's defaults are not necessarily all manuscript conditions). Example:

```bash
MODEL_BENCHMARKS="densenet121,resnet18,resnet34" \
CHANNEL_CONDITIONS="BF;DAPI;FM1;WGA;PHASECONTRAST" \
python -u training/species/Train_SSRUN_multicondition_benchmark.py
```

**Read A2's configuration before using it:** the supplied A2 file defaults to a *three-channel* DenseNet SE run (`DAPI`, `WGA`, `FM1`) rather than the four-channel reference used in the paper. Set `SELECTED_CHANNELS` and supply the exact matching weights for the experiment you intend to reproduce. **The original standalone 13-species DenseNet121 concatenation-fusion training program is not identifiable as a separate script in the uploaded archive.** Its trained checkpoint is required for several analyses below. The `ResNet18/34` trainer does not replace the missing DenseNet concat trainer.

### B. Training — direct AST-routing classification (eight groups)

| Order | Script | Purpose | Requires |
|---|---|---|---|
| B1 | `training/ast_routing/Train_SSRUN_direct_panel2_benchmark.py` | Train direct eight-group single-channel baselines | Main dataset ZIP and splits |
| B2 | `training/ast_routing/Train_Trillian_8class_concat_and_SE_feature_fusion_ARC.py` | Train direct eight-group DenseNet concat/SE feature fusion | B1 eight-class channel checkpoints (typically `8_class_weights/`) |

The direct eight-group models are trained from scratch on the eight-group targets; they are not simply the output of mapping species probabilities into groups.

### C. Evaluation — previously trained model outputs

| Order | Script | Purpose | Requires |
|---|---|---|---|
| C1 | `evaluation/SC_MAPS_three_model_probability_ensemble_ARC.py` | Evaluate channel and model probability ensembles | Holdout crops + single-channel `.pt` files |
| C2 | `evaluation/scmaps_arc_all_in_one_AUTOMODEL_FIXED_V2.py` | Compare four fusion models (species/panel × concat/SE); optional per-cell probabilities, aggregation and features | Matching holdout and four fusion checkpoints, or existing prediction CSVs |
| C3 | `evaluation/SCMAPS_direct_panel_bootstrap_ARC_CORRECTED.py` | Paired, sample-clustered bootstrap of species-pooled versus directly trained panel routing | Existing independent-holdout prediction files in configured run folders |

The C2 script supports `--inspect`, `--check-models`, `--compare-only`, and explicit checkpoint arguments. It should be run after training, not before.

Example inspection (no bulk model inference):

```bash
python -u evaluation/scmaps_arc_all_in_one_AUTOMODEL_FIXED_V2.py --inspect
```

### D. Downstream analyses — manuscript figures and supplementary analyses

| Order | Script | Purpose | Requires |
|---|---|---|---|
| D1 | `analysis/SC_MAPS_CHRIS_REVIEW_ALL_IN_ONE_ARC_FIXED2.py` | Main integrated post-hoc analysis: holdout performance, AST randomisation, bootstrap, selected mixtures, morphological figures and feature exports | Main dataset and 13-class DenseNet121 concat checkpoint; additional unseen input for unseen stages |
| D2 | `analysis/polymicrobial/SC_MAPS_polymicrobial_ALL_IN_ONE_ARC.py` | Full synthetic two-species mixing and constituent recovery analyses | Main holdout and trained 13-class model (or script-supported reused outputs) |
| D3 | `analysis/polymicrobial/SC_MAPS_200CELL_ONE_SCRIPT.py` | Separate old/new balanced 200-cell synthetic-mixture comparison | Original holdout *per-cell species probability CSV* (no GPU training) |
| D4 | `analysis/umap/SC_MAPS_FRESH_INDEPENDENT_UMAP_ARC.py` | Fresh known-only, joint, and unseen-only PCA/UMAP embeddings | Saved **penultimate feature** `.npz` files from known and unseen cells |
| D5 | `analysis/umap/SC_MAPS_UNSEEN_UMAP_EXPLORATORY_ARC.py` | Exploratory unseen-only UMAP, stability and PCA-space neighbours | Saved unseen penultimate-feature `.npz` file |

Once the required feature `.npz` files exist, inspect them before fitting UMAP:

```bash
python -u analysis/umap/SC_MAPS_FRESH_INDEPENDENT_UMAP_ARC.py --inspect-only
```

**UMAP scripts do not train neural networks or directly open microscopy `.npy` cell crops.** First run the model feature-extraction stage; then supply the saved vectors. The five unseen species must remain excluded from model development.

### E. Supplementary — specialist Enterobacterales experiments

| Script | Purpose | Requires |
|---|---|---|
| `supplementary/specialist/Train_secondary_confusion_specialist_from_fusion_ARC.py` | Original specialist experiment restricted to ENTC/KLAE/KLPN/SERM | Trained 13-class fusion checkpoint + main dataset |
| `supplementary/specialist/Train_secondary_pairwise_specialist_from_fusion_ARC_FIXED.py` | Corrected follow-up two-class KLAE/KLPN specialist variant | Trained fusion checkpoint + main dataset |

These are exploratory/supplementary; the original four-class specialist did **not** replace the selected species-trained fusion classifier in the paper.

## 5. Output locations and reproducibility cautions

- Training scripts save model checkpoints (`*.pt`), training histories, metrics and holdout predictions to the output directories configured **within each script**. Do not assume they will land next to the script in this repository.
- `analysis` and `evaluation` are **downstream** of the appropriate checkpoints or saved `*.csv`/`*.npz` outputs. Merely having both ZIP datasets does not eliminate the need to train or supply those checkpoints.
- Keep true `cell_id`, `sample_id`, strain and independent-holdout membership when comparing results; check column names before using saved CSVs from different model runs.

