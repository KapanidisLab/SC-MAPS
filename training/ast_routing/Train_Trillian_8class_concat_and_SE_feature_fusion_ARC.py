#!/usr/bin/env python3
"""
ARC dual feature-fusion training for the Trillian direct Panel-2 8-class task.

This script starts from the user's existing concat/SE feature-fusion workflow but
uses the directly trained 8-class single-channel checkpoints under
Trillian/8_class_weights. One invocation runs TWO independent experiments:

  1. concat       - projected channel features are concatenated directly
  2. se_attention - each projected branch is squeeze-and-excitation reweighted
                    before concatenation

The two child runs start from the same original pretrained single-channel
checkpoints. They therefore remain a clean comparison: the SE run never inherits
fine-tuned branch weights from the concat run.

The classification target is the Slide-11 / Panel-2 8-class mapping used by
Train_SSRUN_direct_panel2_benchmark.py:
  - Routine Enterobacterales: ECOL, KLPN
  - AmpC/caution Enterobacterales: KLAE, ENTC, SERM
  - Salmonella: SAENA
  - Pseudomonas: PSAR
  - Acinetobacter: ACAA
  - Stenotrophomonas: PSMA
  - S. aureus-like staphylococci: SAUR, SLUG
  - CoNS staphylococci: SEPI, STCP

Main behaviour
--------------
1. Parent process launches independent concat and SE-attention child runs.
2. Each child copies/extracts the prepared Trillian dataset to ARC scratch,
   reusing an already-complete scratch extraction when present.
3. Species labels in train/validation/Plate-1 holdout CSVs are converted to the
   exact direct 8-class Panel-2 labels.
4. Loads BF, DAPI, PHASECONTRAST, WGA and FM1 checkpoints from 8_class_weights.
5. Strictly verifies checkpoint class_to_idx, channel metadata, task_mode and
   panel definition before any fusion training.
6. Evaluates each individual channel and the mean-probability ensemble on the
   exact Plate-1 holdout.
7. Trains the fusion head with frozen branches, then optionally fine-tunes the
   final CNN block in every branch.
8. Saves checkpoints, histories, predictions, classification reports, confusion
   matrices and metrics to scratch, with persistent /data synchronisation.
9. After both child runs complete, writes a concat-vs-SE comparison summary to
   the persistent output parent.

Default invocation
------------------
python Train_Trillian_8class_concat_and_SE_feature_fusion_ARC.py

Important editable defaults
---------------------------
WEIGHTS_ROOT defaults to:
  /data/phys-deep-learning-bacteria/phys3115/Bacteria-data-15 species/Trillian/8_class_weights

SELECTED_CHANNELS defaults to all five Trillian channels. If your 8-class weight
folder does not contain one of them, remove that channel from SELECTED_CHANNELS
below; use the SAME channel list for concat and SE for a fair comparison.
"""
from __future__ import annotations

import copy
import gc
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import warnings
import zipfile
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

warnings.filterwarnings("once")

try:
    from IPython.display import display
except Exception:
    def display(value):
        print(value)


def env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {
        "1", "true", "yes", "y", "on"
    }


def _run_both_fusions_parent_if_requested() -> None:
    """Run concat and SE as separate child processes from one ARC invocation.

    A subprocess boundary is intentional: it guarantees that each fusion mode
    reloads the original 8-class single-channel checkpoints rather than
    inheriting branch weights fine-tuned by the previous fusion experiment.
    """
    if os.environ.get("FUSION_CHILD", "0").strip() == "1":
        return
    if not env_bool("RUN_BOTH_FUSIONS", True):
        return

    modes_text = os.environ.get("FUSION_MODES", "concat,se_attention")
    modes = [m.strip().lower() for m in modes_text.split(",") if m.strip()]
    allowed = {"concat", "se_attention"}
    invalid = [m for m in modes if m not in allowed]
    if invalid:
        raise ValueError(
            f"Invalid FUSION_MODES={invalid}. Allowed modes are {sorted(allowed)}."
        )
    if not modes:
        raise ValueError("FUSION_MODES did not contain any fusion modes.")

    script_path = Path(__file__).resolve()
    base_run_name = os.environ.get(
        "RUN_NAME",
        datetime.now().strftime("run_%Y%m%d_%H%M%S") + "_panel2_8class_fusion",
    )

    persistent_data_parent = Path(os.environ.get(
        "PERSISTENT_DATA_PARENT",
        "/data/phys-deep-learning-bacteria/phys3115/Bacteria-data-15 species/Trillian",
    )).expanduser()
    persistent_output_parent = Path(os.environ.get(
        "PERSISTENT_OUTPUT_PARENT",
        str(persistent_data_parent / "Trillian_8class_fusion_training_outputs_ARC"),
    )).expanduser()
    persistent_output_parent.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("PANEL-2 8-CLASS DUAL FUSION RUN")
    print("Script:", script_path)
    print("Modes:", modes)
    print("Base run name:", base_run_name)
    print("Persistent output parent:", persistent_output_parent)
    print("=" * 100)

    child_run_names = []
    for mode in modes:
        child_run_name = f"{base_run_name}_{mode}"
        child_run_names.append((mode, child_run_name))

        child_env = os.environ.copy()
        child_env["FUSION_CHILD"] = "1"
        child_env["RUN_BOTH_FUSIONS"] = "0"
        child_env["FUSION_MODE"] = mode
        child_env["RUN_NAME"] = child_run_name

        print("\n" + "#" * 100)
        print(f"STARTING CHILD FUSION RUN: {mode}")
        print("RUN_NAME:", child_run_name)
        print("#" * 100)

        subprocess.run(
            [sys.executable, str(script_path)],
            env=child_env,
            check=True,
        )

    # Cross-method summary: the complete detailed outputs remain in each child.
    comparison_rows = []
    for mode, child_run_name in child_run_names:
        summary_path = (
            persistent_output_parent
            / child_run_name
            / "final_run_summary.json"
        )
        if not summary_path.exists():
            comparison_rows.append({
                "fusion_mode": mode,
                "run_name": child_run_name,
                "summary_found": False,
            })
            continue

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = summary.get("feature_fusion_holdout_metrics", {}) or {}
        best_record = summary.get("best_validation_record", {}) or {}
        comparison_rows.append({
            "fusion_mode": mode,
            "run_name": child_run_name,
            "summary_found": True,
            "holdout_accuracy": metrics.get("accuracy"),
            "holdout_balanced_accuracy": metrics.get("balanced_accuracy"),
            "holdout_macro_f1": metrics.get("macro_f1"),
            "holdout_weighted_f1": metrics.get("weighted_f1"),
            "holdout_top2_accuracy": metrics.get("top2_accuracy"),
            "best_val_macro_f1": best_record.get("val_macro_f1"),
            "best_global_epoch": best_record.get("global_epoch"),
            "best_stage": best_record.get("stage"),
        })

    comparison_dir = persistent_output_parent / f"{base_run_name}_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    comparison_df = pd.DataFrame(comparison_rows)
    if (
        not comparison_df.empty
        and "holdout_macro_f1" in comparison_df.columns
    ):
        comparison_df = comparison_df.sort_values(
            "holdout_macro_f1",
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

    comparison_df.to_csv(
        comparison_dir / "concat_vs_se_holdout_comparison.csv",
        index=False,
    )
    (comparison_dir / "concat_vs_se_holdout_comparison.json").write_text(
        json.dumps(comparison_rows, indent=2),
        encoding="utf-8",
    )
    (comparison_dir / "child_run_names.json").write_text(
        json.dumps(
            {mode: run_name for mode, run_name in child_run_names},
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("BOTH FUSION RUNS COMPLETED")
    print("Cross-method comparison:")
    print(comparison_df.to_string(index=False))
    print("Comparison directory:", comparison_dir)
    print("=" * 100)
    raise SystemExit(0)


_run_both_fusions_parent_if_requested()


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
    def flush(self):
        for stream in self.streams:
            stream.flush()


def safe_copy_file(src: Path, dst: Path) -> Path:
    src, dst = Path(src), Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"Dataset ZIP not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        print(f"Scratch ZIP already present with matching size: {dst}")
        return dst
    print(f"Copying dataset ZIP to scratch:\n  {src}\n  -> {dst}")
    shutil.copy2(src, dst)
    return dst


def safe_extract_member(zf: zipfile.ZipFile, member: zipfile.ZipInfo, root: Path) -> None:
    root_resolved = Path(root).resolve()
    target = (root_resolved / member.filename).resolve()
    if not str(target).startswith(str(root_resolved)):
        raise RuntimeError(f"Unsafe ZIP member: {member.filename}")
    if member.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member, "r") as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)


def has_prepared_dataset(root: Path) -> bool:
    root = Path(root)
    return (root / "cell_manifest.csv").exists() and (root / "splits").is_dir()


def extract_dataset_zip(zip_path: Path, extract_root: Path, expected_root: Path, force: bool = False) -> Path:
    zip_path, extract_root, expected_root = map(Path, (zip_path, extract_root, expected_root))
    sentinel = expected_root / ".arc_extract_complete"
    if force and expected_root.exists():
        print(f"Removing previous scratch extraction: {expected_root}")
        shutil.rmtree(expected_root)
    if has_prepared_dataset(expected_root) and sentinel.exists():
        print(f"Using existing scratch dataset: {expected_root}")
        return expected_root
    if has_prepared_dataset(expected_root) and not force:
        sentinel.touch()
        return expected_root
    print(f"Extracting {zip_path} to {extract_root}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        for member in tqdm(members, desc="Extracting dataset", unit="file", mininterval=2.0):
            safe_extract_member(zf, member, extract_root)
    candidates = [expected_root] + [p.parent for p in extract_root.rglob("cell_manifest.csv")]
    for candidate in candidates:
        if has_prepared_dataset(candidate):
            (candidate / ".arc_extract_complete").touch()
            print(f"Resolved extracted dataset: {candidate}")
            return candidate
    raise FileNotFoundError("Extraction completed but prepared dataset root was not found.")


def sync_path(src: Path, dst: Path) -> None:
    src, dst = Path(src), Path(dst)
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        temporary = Path(str(dst) + ".tmp")
        shutil.copy2(src, temporary)
        os.replace(temporary, dst)
    elif shutil.which("rsync"):
        dst.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", "--delete", f"{src}/", f"{dst}/"], check=False)
    else:
        shutil.copytree(src, dst, dirs_exist_ok=True)


def sync_outputs_to_persistent(paths=None, label: str = "sync") -> None:
    if Path(RUN_DIR).resolve() == Path(PERSISTENT_RUN_DIR).resolve():
        return
    PERSISTENT_RUN_DIR.mkdir(parents=True, exist_ok=True)
    if paths is None:
        paths = [RUN_DIR]
    for src in paths:
        src = Path(src)
        try:
            relative = src.relative_to(RUN_DIR)
            dst = PERSISTENT_RUN_DIR / relative
        except ValueError:
            dst = PERSISTENT_RUN_DIR / src.name
        sync_path(src, dst)
    (PERSISTENT_RUN_DIR / "last_sync.json").write_text(json.dumps({
        "label": label,
        "time": datetime.now().isoformat(timespec="seconds"),
        "scratch_run_dir": str(RUN_DIR),
        "persistent_run_dir": str(PERSISTENT_RUN_DIR),
    }, indent=2))
    print(f"Synced outputs [{label}] -> {PERSISTENT_RUN_DIR}")


# -----------------------------------------------------------------------------
# ARC storage and dataset preparation
# -----------------------------------------------------------------------------
DATASET_FOLDER_NAME = os.environ.get("DATASET_FOLDER_NAME", "Trillian_DATASET_with_holdout")
PERSISTENT_DATA_PARENT = Path(os.environ.get(
    "PERSISTENT_DATA_PARENT",
    "/data/phys-deep-learning-bacteria/phys3115/Bacteria-data-15 species/Trillian",
)).expanduser()
ZIP_PATH = Path(os.environ.get(
    "ZIP_PATH", str(PERSISTENT_DATA_PARENT / f"{DATASET_FOLDER_NAME}.zip")
)).expanduser()
USER_NAME = os.environ.get("USER", "phys3115")
SCRATCH_BASE = Path(os.environ.get("SCRATCH_BASE", f"/scratch/{USER_NAME}")).expanduser()
SCRATCH_DATA_PARENT = Path(os.environ.get(
    "SCRATCH_DATA_PARENT", str(SCRATCH_BASE / "bacteria_data")
)).expanduser()
SCRATCH_OUTPUT_PARENT = Path(os.environ.get(
    "SCRATCH_OUTPUT_PARENT", str(SCRATCH_BASE / "bacteria_8class_fusion_outputs")
)).expanduser()
PERSISTENT_OUTPUT_PARENT = Path(os.environ.get(
    "PERSISTENT_OUTPUT_PARENT",
    str(PERSISTENT_DATA_PARENT / "Trillian_8class_fusion_training_outputs_ARC"),
)).expanduser()

SCRATCH_ZIP_PATH = SCRATCH_DATA_PARENT / ZIP_PATH.name
SCRATCH_DATA_ROOT = SCRATCH_DATA_PARENT / DATASET_FOLDER_NAME
USE_SCRATCH_DATA = env_bool("USE_SCRATCH_DATA", True)
COPY_ZIP_TO_SCRATCH = env_bool("COPY_ZIP_TO_SCRATCH", True)
FORCE_REEXTRACT = env_bool("FORCE_REEXTRACT", False)
SYNC_EVERY_EPOCH = env_bool("SYNC_EVERY_EPOCH", True)

SCRATCH_DATA_PARENT.mkdir(parents=True, exist_ok=True)
SCRATCH_OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)
PERSISTENT_OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)

if USE_SCRATCH_DATA:
    if COPY_ZIP_TO_SCRATCH:
        safe_copy_file(ZIP_PATH, SCRATCH_ZIP_PATH)
    DATA_ROOT = extract_dataset_zip(
        SCRATCH_ZIP_PATH, SCRATCH_DATA_PARENT, SCRATCH_DATA_ROOT, FORCE_REEXTRACT
    )
else:
    DATA_ROOT = Path(os.environ.get(
        "DATA_ROOT", str(PERSISTENT_DATA_PARENT / DATASET_FOLDER_NAME)
    )).expanduser()

# -----------------------------------------------------------------------------
# Direct Panel-2 8-class target definition
# -----------------------------------------------------------------------------
TASK_MODE = "panel"
PANEL_DEFINITION_NAME = "slide11_panel2_8class"
PANEL_TO_SPECIES = {
    "Routine Enterobacterales": ["ECOL", "KLPN"],
    "AmpC/caution Enterobacterales": ["KLAE", "ENTC", "SERM"],
    "Salmonella": ["SAENA"],
    "Pseudomonas": ["PSAR"],
    "Acinetobacter": ["ACAA"],
    "Stenotrophomonas": ["PSMA"],
    "S. aureus-like staphylococci": ["SAUR", "SLUG"],
    "CoNS staphylococci": ["SEPI", "STCP"],
}
SPECIES_TO_PANEL = {
    species: panel
    for panel, species_list in PANEL_TO_SPECIES.items()
    for species in species_list
}


def panel_label_from_species(species: Any) -> str:
    species_code = str(species).strip().upper()
    if species_code not in SPECIES_TO_PANEL:
        raise KeyError(
            f"Species {species_code!r} is not present in the Panel-2 8-class "
            f"mapping. Covered species: {sorted(SPECIES_TO_PANEL)}"
        )
    return SPECIES_TO_PANEL[species_code]


# -----------------------------------------------------------------------------
# Fusion configuration
# -----------------------------------------------------------------------------
DATA_SOURCE = "csv"
TRAIN_CSV = DATA_ROOT / "splits" / "train.csv"
VAL_CSV = DATA_ROOT / "splits" / "val.csv"
HOLDOUT_CSV = DATA_ROOT / "splits" / os.environ.get("HOLDOUT_CSV_NAME", "holdout_plate1.csv")
USE_MASKED_CROPS = False
CSV_PATH_COLUMN = "crop_path"
CSV_LABEL_COLUMN = "species"
PATH_REMAP = {}
TRAIN_ROOT = VAL_ROOT = HOLDOUT_ROOT = None
VALIDATION_FRACTION = 0.10
GROUP_COLUMN = None
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", 42))

CHANNEL_TO_INDEX = {
    "PHASECONTRAST": 0,
    "BF": 1,
    "DAPI": 2,
    "WGA": 3,
    "FM1": 4,
}
EXPECTED_CROP_CHANNELS = 5
ARRAY_LAYOUT = "CHW"
INPUT_SIZE = int(os.environ.get("INPUT_SIZE", 96))
SELECTED_CHANNELS = ["DAPI", "PHASECONTRAST", "WGA", "FM1"]
NORMALIZATION_BY_CHANNEL = {
    channel: {"mode": "per_crop_zscore"} for channel in CHANNEL_TO_INDEX
}
PERCENTILE_LOW = 1.0
PERCENTILE_HIGH = 99.0
NORMALIZATION_EPS = 1e-6

BRANCH_ARCHITECTURES = {channel: "densenet121" for channel in SELECTED_CHANNELS}
WEIGHTS_ROOT = Path(os.environ.get(
    "WEIGHTS_ROOT", str(PERSISTENT_DATA_PARENT / "8_class_weights")
)).expanduser()


def resolve_weight(channel: str) -> Path:
    normal = re.sub(r"[^A-Z0-9]", "", channel.upper())
    aliases = {
        "PHASECONTRAST": ["PHASECONTRAST", "PhaseContrast", "phasecontrast", "Phase_Contrast"],
        "BF": ["BF", "bf"],
        "DAPI": ["DAPI", "dapi"],
        "WGA": ["WGA", "wga"],
        "FM1": ["FM1", "fm1", "FM1-43", "FM1_43"],
    }[channel]

    # Preferred layout supplied by the user:
    #   8_class_weights/WGA/best_model.pt
    #   8_class_weights/DAPI/best_model.pt
    # etc.
    for folder_name in aliases:
        channel_root = WEIGHTS_ROOT / folder_name
        direct = channel_root / "best_model.pt"
        if direct.exists():
            return direct

        # Also tolerate one or more experiment subfolders inside the channel
        # folder. If several exist, use the most recently modified best_model.pt
        # and print the choice clearly in the run log.
        if channel_root.exists():
            nested = list(channel_root.rglob("best_model.pt"))
            if nested:
                nested.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                chosen = nested[0]
                print(
                    f"{channel}: no direct best_model.pt; selected newest nested "
                    f"checkpoint: {chosen}"
                )
                return chosen

    # Last-resort search by parent-folder name.
    for candidate in WEIGHTS_ROOT.rglob("best_model.pt"):
        folder_norm = re.sub(r"[^A-Z0-9]", "", candidate.parent.name.upper())
        if folder_norm == normal:
            return candidate

    raise FileNotFoundError(
        f"Could not find best_model.pt for {channel} below {WEIGHTS_ROOT}. "
        f"Expected e.g. {WEIGHTS_ROOT / channel / 'best_model.pt'}"
    )

WEIGHT_SOURCES = {channel: resolve_weight(channel) for channel in SELECTED_CHANNELS}
CHECKPOINT_CLASS_NAMES = {channel: None for channel in SELECTED_CHANNELS}
STRICT_CHECKPOINT_LOADING = True
VALIDATE_CHECKPOINT_METADATA = True
USE_CHECKPOINT_NORMALIZATION = True

FUSION_MODE = os.environ.get("FUSION_MODE", "se_attention").strip().lower()
SE_REDUCTION_RATIO = int(os.environ.get("SE_REDUCTION_RATIO", 4))
PROJECTION_DIM = int(os.environ.get("PROJECTION_DIM", 256))
FUSION_HIDDEN_DIM = int(os.environ.get("FUSION_HIDDEN_DIM", 512))
FUSION_DROPOUT = float(os.environ.get("FUSION_DROPOUT", 0.30))

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 2))
PIN_MEMORY = torch.cuda.is_available()
PERSISTENT_WORKERS = NUM_WORKERS > 0
PREFETCH_FACTOR = 2 if NUM_WORKERS > 0 else None
FROZEN_HEAD_EPOCHS = int(os.environ.get("FROZEN_HEAD_EPOCHS", 10))
FINETUNE_EPOCHS = int(os.environ.get("FINETUNE_EPOCHS", 30))
HEAD_LEARNING_RATE = float(os.environ.get("HEAD_LEARNING_RATE", 1e-3))
FINETUNE_HEAD_LEARNING_RATE = float(os.environ.get("FINETUNE_HEAD_LEARNING_RATE", 1e-4))
BACKBONE_LEARNING_RATE = float(os.environ.get("BACKBONE_LEARNING_RATE", 1e-5))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 1e-5))
UNFREEZE_POLICY = os.environ.get("UNFREEZE_POLICY", "last_block")
USE_CLASS_WEIGHTED_LOSS = env_bool("USE_CLASS_WEIGHTED_LOSS", True)
LABEL_SMOOTHING = float(os.environ.get("LABEL_SMOOTHING", 0.0))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", 1))
MAX_GRAD_NORM = float(os.environ.get("MAX_GRAD_NORM", 5.0))
USE_AMP = env_bool("USE_AMP", True)
EARLY_STOPPING_PATIENCE = int(os.environ.get("EARLY_STOPPING_PATIENCE", 8))
LR_SCHEDULER_PATIENCE = int(os.environ.get("LR_SCHEDULER_PATIENCE", 3))
LR_SCHEDULER_FACTOR = float(os.environ.get("LR_SCHEDULER_FACTOR", 0.3))
USE_AUGMENTATION = env_bool("USE_AUGMENTATION", True)
HORIZONTAL_FLIP_PROB = 0.5
VERTICAL_FLIP_PROB = 0.5
MAX_ROTATION_DEGREES = 12.0
SAVE_ALL_CLASS_PROBABILITIES = env_bool("SAVE_ALL_CLASS_PROBABILITIES", True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RUN_NAME = os.environ.get(
    "RUN_NAME",
    datetime.now().strftime("run_%Y%m%d_%H%M%S")
    + "_panel2_8class_" + FUSION_MODE,
)
OUTPUT_ROOT = SCRATCH_OUTPUT_PARENT
RUN_DIR = OUTPUT_ROOT / RUN_NAME
PERSISTENT_RUN_DIR = PERSISTENT_OUTPUT_PARENT / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=False)
PERSISTENT_RUN_DIR.mkdir(parents=True, exist_ok=True)

_log_handle = open(RUN_DIR / "run_log.txt", "a", buffering=1)
sys.stdout = Tee(sys.__stdout__, _log_handle)
sys.stderr = Tee(sys.__stderr__, _log_handle)

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("DATA_ROOT:", DATA_ROOT)
print("Task mode:", TASK_MODE)
print("Panel definition:", PANEL_DEFINITION_NAME)
print("Selected channels:", SELECTED_CHANNELS)
print("Fusion mode:", FUSION_MODE)
print("8-class weights root:", WEIGHTS_ROOT)
print("Panel mapping:")
for _panel_name, _species_list in PANEL_TO_SPECIES.items():
    print(f"  {_panel_name}: {', '.join(_species_list)}")
print("Resolved weights:")
for channel, path in WEIGHT_SOURCES.items():
    print(f"  {channel}: {path}")
print("Scratch run directory:", RUN_DIR)
print("Persistent run directory:", PERSISTENT_RUN_DIR)

# Save ARC path configuration immediately.
(RUN_DIR / "arc_path_config.json").write_text(json.dumps({
    "persistent_data_parent": str(PERSISTENT_DATA_PARENT),
    "zip_path": str(ZIP_PATH),
    "scratch_zip_path": str(SCRATCH_ZIP_PATH),
    "data_root": str(DATA_ROOT),
    "weights_root": str(WEIGHTS_ROOT),
    "scratch_run_dir": str(RUN_DIR),
    "persistent_run_dir": str(PERSISTENT_RUN_DIR),
    "sync_every_epoch": SYNC_EVERY_EPOCH,
}, indent=2))
sync_outputs_to_persistent([RUN_DIR / "arc_path_config.json", RUN_DIR / "run_log.txt"], "initial")

# ===== NOTEBOOK CODE CELL 2 =====

# %pip install -q torch torchvision pandas numpy scikit-learn matplotlib tqdm



# ===== NOTEBOOK CODE CELL 9 =====

SUPPORTED_ARCHITECTURES = {
    "densenet121",
    "resnet18",
    "resnet34",
    "convnext_tiny",
    "efficientnet_b0",
}

def validate_configuration() -> None:
    if DATA_SOURCE not in {"folders", "csv"}:
        raise ValueError("DATA_SOURCE must be 'folders' or 'csv'.")

    if USE_MASKED_CROPS:
        raise ValueError(
            "USE_MASKED_CROPS must be False for channel fusion. "
            "The mask_crops arrays are masks, not five-channel image crops."
        )

    if len(SELECTED_CHANNELS) < 2:
        raise ValueError("Select at least two channels for fusion.")

    missing_channel_indices = [c for c in SELECTED_CHANNELS if c not in CHANNEL_TO_INDEX]
    if missing_channel_indices:
        raise KeyError(f"Missing CHANNEL_TO_INDEX entries: {missing_channel_indices}")

    unset_channel_indices = [
        c for c in SELECTED_CHANNELS
        if CHANNEL_TO_INDEX.get(c) is None
    ]
    if unset_channel_indices:
        raise ValueError(
            "Set an integer channel index in CHANNEL_TO_INDEX for: "
            f"{unset_channel_indices}. For PhaseContrast, use the position of "
            "the PhaseContrast plane inside each .npy crop."
        )

    missing_architectures = [c for c in SELECTED_CHANNELS if c not in BRANCH_ARCHITECTURES]
    if missing_architectures:
        raise KeyError(f"Missing BRANCH_ARCHITECTURES entries: {missing_architectures}")

    invalid_architectures = {
        c: BRANCH_ARCHITECTURES[c]
        for c in SELECTED_CHANNELS
        if BRANCH_ARCHITECTURES[c] not in SUPPORTED_ARCHITECTURES
    }
    if invalid_architectures:
        raise ValueError(f"Unsupported architectures: {invalid_architectures}")

    missing_weights = [c for c in SELECTED_CHANNELS if c not in WEIGHT_SOURCES]
    if missing_weights:
        raise KeyError(f"Missing WEIGHT_SOURCES entries: {missing_weights}")

    if FUSION_MODE not in {"concat", "se_attention"}:
        raise ValueError("FUSION_MODE must be 'concat' or 'se_attention'.")

    if SE_REDUCTION_RATIO < 1:
        raise ValueError("SE_REDUCTION_RATIO must be at least 1.")

    if UNFREEZE_POLICY not in {"last_block", "all", "none"}:
        raise ValueError("UNFREEZE_POLICY must be 'last_block', 'all', or 'none'.")

validate_configuration()
print("Run directory:", RUN_DIR.resolve())



# ===== NOTEBOOK CODE CELL 11 =====

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Deterministic behaviour can reduce speed but improves repeatability.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(RANDOM_SEED)

def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted(json_safe(v) for v in value)
    return value

def collect_user_config() -> Dict[str, Any]:
    names = [
        "TASK_MODE", "PANEL_DEFINITION_NAME", "PANEL_TO_SPECIES", "SPECIES_TO_PANEL",
        "DATA_SOURCE", "DATA_ROOT", "TRAIN_ROOT", "VAL_ROOT", "HOLDOUT_ROOT",
        "TRAIN_CSV", "VAL_CSV", "HOLDOUT_CSV",
        "CSV_PATH_COLUMN", "CSV_LABEL_COLUMN", "USE_MASKED_CROPS", "PATH_REMAP",
        "VALIDATION_FRACTION", "GROUP_COLUMN", "RANDOM_SEED",
        "CHANNEL_TO_INDEX", "EXPECTED_CROP_CHANNELS", "SELECTED_CHANNELS", "ARRAY_LAYOUT", "INPUT_SIZE",
        "NORMALIZATION_BY_CHANNEL", "PERCENTILE_LOW", "PERCENTILE_HIGH",
        "NORMALIZATION_EPS", "BRANCH_ARCHITECTURES", "WEIGHT_SOURCES",
        "CHECKPOINT_CLASS_NAMES", "STRICT_CHECKPOINT_LOADING", "VALIDATE_CHECKPOINT_METADATA", "USE_CHECKPOINT_NORMALIZATION", "FUSION_MODE", "SE_REDUCTION_RATIO", "PROJECTION_DIM",
        "FUSION_HIDDEN_DIM", "FUSION_DROPOUT", "BATCH_SIZE", "NUM_WORKERS",
        "PIN_MEMORY", "PERSISTENT_WORKERS", "FROZEN_HEAD_EPOCHS",
        "FINETUNE_EPOCHS", "HEAD_LEARNING_RATE",
        "FINETUNE_HEAD_LEARNING_RATE", "BACKBONE_LEARNING_RATE",
        "WEIGHT_DECAY", "UNFREEZE_POLICY", "USE_CLASS_WEIGHTED_LOSS",
        "LABEL_SMOOTHING", "GRADIENT_ACCUMULATION_STEPS",
        "MAX_GRAD_NORM", "USE_AMP", "EARLY_STOPPING_PATIENCE",
        "LR_SCHEDULER_PATIENCE", "LR_SCHEDULER_FACTOR",
        "USE_AUGMENTATION", "HORIZONTAL_FLIP_PROB",
        "VERTICAL_FLIP_PROB", "MAX_ROTATION_DEGREES",
        "OUTPUT_ROOT", "RUN_NAME", "SAVE_ALL_CLASS_PROBABILITIES",
        "DEVICE",
    ]
    return {name: json_safe(globals()[name]) for name in names}

USER_CONFIG = collect_user_config()

ENVIRONMENT = {
    "created_at": datetime.now().isoformat(),
    "platform": platform.platform(),
    "python": sys.version,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit_learn": sklearn.__version__,
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
    "cudnn_version": torch.backends.cudnn.version(),
    "device": str(DEVICE),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}

(RUN_DIR / "run_config_initial.json").write_text(
    json.dumps(USER_CONFIG, indent=2), encoding="utf-8"
)
(RUN_DIR / "environment.json").write_text(
    json.dumps(json_safe(ENVIRONMENT), indent=2), encoding="utf-8"
)

print(json.dumps(ENVIRONMENT, indent=2))



# ===== NOTEBOOK CODE CELL 13 =====

PATH_COLUMN_CANDIDATES = [
    "crop_path", "crop_file", "crop", "image_path", "img_path",
    "npy_path", "path", "filepath", "file_path",
]
LABEL_COLUMN_CANDIDATES = [
    "species", "label", "species_id", "class", "class_name", "target",
]

def remap_path_string(path_value: Any) -> str:
    text = str(path_value)
    for old, new in PATH_REMAP.items():
        text = text.replace(str(old), str(new))
    return text

def resolve_csv_column(
    df: pd.DataFrame,
    explicit: Optional[str],
    candidates: Sequence[str],
    kind: str,
) -> str:
    if explicit is not None:
        if explicit not in df.columns:
            raise KeyError(
                f"Requested {kind} column '{explicit}' not found. "
                f"Columns: {list(df.columns)}"
            )
        return explicit

    lower_to_original = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]

    raise KeyError(
        f"Could not auto-detect {kind} column. "
        f"Available columns: {list(df.columns)}"
    )

def rebase_prepared_dataset_path(
    path_value: Any,
    row: Optional[pd.Series] = None,
    split_name: str = "",
) -> str:
    """Rebase old ARC/Colab/Windows paths onto the current DATA_ROOT."""
    original = remap_path_string(path_value)
    normalised = original.replace("\\", "/")

    if Path(original).exists():
        return str(Path(original))

    for marker in ("Trillian_DATASET_with_holdout", DATA_ROOT.name):
        if marker and marker in normalised:
            suffix = normalised.split(marker, 1)[1].lstrip("/")
            candidate = DATA_ROOT / Path(suffix)
            if candidate.exists():
                return str(candidate)

    basename = Path(normalised).name
    if row is not None and basename:
        partition = str(row.get("dataset_partition", "")).strip()
        if not partition or partition.lower() == "nan":
            partition = (
                "holdout_plate1"
                if "holdout" in split_name.lower()
                else "train_pool_plates_not_1"
            )
        candidate = DATA_ROOT / partition / "crops" / basename
        if candidate.exists():
            return str(candidate)

    return original

def load_csv_records(csv_path: Path, split_name: str) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    path_col = resolve_csv_column(
        df, CSV_PATH_COLUMN, PATH_COLUMN_CANDIDATES, "multichannel crop path"
    )
    label_col = resolve_csv_column(
        df, CSV_LABEL_COLUMN, LABEL_COLUMN_CANDIDATES, "label"
    )

    if "mask" in str(path_col).lower():
        raise ValueError(
            f"Detected path column '{path_col}', which appears to be a mask path. "
            "Set CSV_PATH_COLUMN='crop_path'."
        )

    out = df.copy()
    out["path"] = out.apply(
        lambda row: rebase_prepared_dataset_path(
            row[path_col], row=row, split_name=split_name
        ),
        axis=1,
    )
    out["species"] = out[label_col].astype(str).str.strip().str.upper()
    unmapped = sorted(set(out["species"]) - set(SPECIES_TO_PANEL))
    if unmapped:
        raise ValueError(
            f"{split_name}: Panel-2 mapping is incomplete for species {unmapped}."
        )
    out["label"] = out["species"].map(SPECIES_TO_PANEL)
    out["source_path_column"] = str(path_col)
    return out

def scan_class_folder(root: Path) -> pd.DataFrame:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    if root.name.lower() == "mask_crops":
        raise ValueError(
            "Folder mode points to mask_crops. Use the crops folder instead."
        )

    files = sorted(root.rglob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found below {root}")

    records = []
    for file_path in files:
        label = file_path.name.split("_", 1)[0]
        species = str(label).strip().upper()
        records.append({
            "path": str(file_path),
            "species": species,
            "label": panel_label_from_species(species),
        })
    return pd.DataFrame(records)

def validate_paths(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    out = df.copy()
    exists = out["path"].map(lambda p: Path(p).exists())
    missing_count = int((~exists).sum())
    if missing_count:
        examples = out.loc[~exists, "path"].head(5).tolist()
        raise FileNotFoundError(
            f"{split_name}: {missing_count:,} crop files are missing. "
            f"Examples: {examples}"
        )
    return out.reset_index(drop=True)

if DATA_SOURCE == "folders":
    if TRAIN_ROOT is None or HOLDOUT_ROOT is None:
        raise ValueError(
            "Folder mode requires TRAIN_ROOT and HOLDOUT_ROOT. "
            "CSV mode is recommended for this dataset."
        )
    train_df = scan_class_folder(TRAIN_ROOT)
    holdout_df = scan_class_folder(HOLDOUT_ROOT)
    val_df = scan_class_folder(VAL_ROOT) if VAL_ROOT is not None else None
else:
    train_df = load_csv_records(TRAIN_CSV, "train")
    holdout_df = load_csv_records(HOLDOUT_CSV, "holdout")
    val_df = (
        load_csv_records(VAL_CSV, "validation")
        if VAL_CSV is not None
        else None
    )

train_df = validate_paths(train_df, "train")
holdout_df = validate_paths(holdout_df, "holdout")
if val_df is not None:
    val_df = validate_paths(val_df, "validation")

if val_df is None:
    if (
        DATA_SOURCE == "csv"
        and GROUP_COLUMN is not None
        and GROUP_COLUMN in train_df.columns
    ):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=VALIDATION_FRACTION,
            random_state=RANDOM_SEED,
        )
        train_indices, val_indices = next(
            splitter.split(train_df, groups=train_df[GROUP_COLUMN])
        )
        val_df = train_df.iloc[val_indices].reset_index(drop=True)
        train_df = train_df.iloc[train_indices].reset_index(drop=True)
    else:
        train_df, val_df = train_test_split(
            train_df,
            test_size=VALIDATION_FRACTION,
            random_state=RANDOM_SEED,
            stratify=train_df["label"],
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

train_labels = set(train_df["label"])
val_labels = set(val_df["label"])
holdout_labels = set(holdout_df["label"])

if val_labels - train_labels:
    raise ValueError(
        f"Validation contains labels absent from training: "
        f"{sorted(val_labels - train_labels)}"
    )
if holdout_labels - train_labels:
    raise ValueError(
        f"Holdout contains labels absent from training: "
        f"{sorted(holdout_labels - train_labels)}"
    )

CLASS_NAMES = sorted(train_labels)
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

EXPECTED_PANEL_CLASS_NAMES = sorted(PANEL_TO_SPECIES)
if CLASS_NAMES != EXPECTED_PANEL_CLASS_NAMES:
    raise ValueError(
        "Current target labels do not match the expected direct Panel-2 "
        f"8-class labels. Current={CLASS_NAMES}; expected={EXPECTED_PANEL_CLASS_NAMES}"
    )
if NUM_CLASSES != 8:
    raise ValueError(f"Expected exactly 8 Panel-2 classes, found {NUM_CLASSES}.")

(RUN_DIR / "panel_definition.json").write_text(
    json.dumps({
        "task_mode": TASK_MODE,
        "panel_definition_name": PANEL_DEFINITION_NAME,
        "panel_to_species": PANEL_TO_SPECIES,
        "species_to_panel": SPECIES_TO_PANEL,
        "class_names_sorted": CLASS_NAMES,
        "class_to_index": CLASS_TO_INDEX,
    }, indent=2),
    encoding="utf-8",
)

for df in (train_df, val_df, holdout_df):
    df["target"] = df["label"].map(CLASS_TO_INDEX).astype(int)

split_summary = pd.DataFrame({
    "train": train_df["label"].value_counts(),
    "validation": val_df["label"].value_counts(),
    "holdout": holdout_df["label"].value_counts(),
}).fillna(0).astype(int).reindex(CLASS_NAMES)

display(split_summary)
print("Number of classes:", NUM_CLASSES)
print("Total train:", len(train_df))
print("Total validation:", len(val_df))
print("Total holdout:", len(holdout_df))
if "source_path_column" in train_df:
    print("Path column used:", train_df["source_path_column"].iloc[0])

train_df.to_csv(RUN_DIR / "resolved_train_samples.csv", index=False)
val_df.to_csv(RUN_DIR / "resolved_validation_samples.csv", index=False)
holdout_df.to_csv(RUN_DIR / "resolved_holdout_samples.csv", index=False)
split_summary.to_csv(RUN_DIR / "split_class_counts.csv")

species_split_summary = pd.DataFrame({
    "train": train_df["species"].value_counts(),
    "validation": val_df["species"].value_counts(),
    "holdout": holdout_df["species"].value_counts(),
}).fillna(0).astype(int).sort_index()
species_split_summary.to_csv(RUN_DIR / "split_species_counts.csv")

(RUN_DIR / "class_names.json").write_text(
    json.dumps(CLASS_NAMES, indent=2), encoding="utf-8"
)



# ===== NOTEBOOK CODE CELL 15 =====

def infer_chw(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array[None, ...]

    if array.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D crop, got shape {array.shape}")

    if ARRAY_LAYOUT == "CHW":
        return array
    if ARRAY_LAYOUT == "HWC":
        return np.moveaxis(array, -1, 0)
    if ARRAY_LAYOUT != "auto":
        raise ValueError("ARRAY_LAYOUT must be 'auto', 'CHW', or 'HWC'.")

    # Typical crops are Cx96x96 or 96x96xC.
    if array.shape[0] <= 16 and array.shape[1] > 16 and array.shape[2] > 16:
        return array
    if array.shape[-1] <= 16 and array.shape[0] > 16 and array.shape[1] > 16:
        return np.moveaxis(array, -1, 0)

    raise ValueError(
        f"Could not infer channel axis for shape {array.shape}. "
        "Set ARRAY_LAYOUT explicitly."
    )

def normalise_plane(plane: np.ndarray, channel_name: str) -> np.ndarray:
    plane = np.asarray(plane, dtype=np.float32)
    plane = np.nan_to_num(plane, nan=0.0, posinf=0.0, neginf=0.0)

    config = NORMALIZATION_BY_CHANNEL.get(
        channel_name, {"mode": "per_crop_zscore"}
    )
    mode = str(config.get("mode", "per_crop_zscore")).lower()

    if mode == "per_crop_zscore":
        mean = float(plane.mean())
        std = float(plane.std())
        return (plane - mean) / max(std, NORMALIZATION_EPS)

    if mode == "per_crop_percentile":
        low, high = np.percentile(plane, [PERCENTILE_LOW, PERCENTILE_HIGH])
        scale = max(float(high - low), NORMALIZATION_EPS)
        return np.clip((plane - low) / scale, 0.0, 1.0)

    if mode == "global_zscore":
        if "mean" not in config or "std" not in config:
            raise KeyError(
                f"{channel_name}: global_zscore requires 'mean' and 'std'."
            )
        mean = float(config["mean"])
        std = max(float(config["std"]), NORMALIZATION_EPS)
        return (plane - mean) / std

    if mode == "none":
        return plane

    raise ValueError(f"Unsupported normalisation mode '{mode}' for {channel_name}")

class JointMicroscopyAugment:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < HORIZONTAL_FLIP_PROB:
            x = torch.flip(x, dims=[2])
        if random.random() < VERTICAL_FLIP_PROB:
            x = torch.flip(x, dims=[1])
        if MAX_ROTATION_DEGREES > 0:
            angle = random.uniform(-MAX_ROTATION_DEGREES, MAX_ROTATION_DEGREES)
            x = TF.rotate(
                x,
                angle=angle,
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
        return x

class MultiChannelNpyDataset(Dataset):
    def __init__(
        self,
        records: pd.DataFrame,
        selected_channels: Sequence[str],
        augment: bool = False,
    ) -> None:
        self.records = records.reset_index(drop=True).copy()
        self.selected_channels = list(selected_channels)
        self.channel_indices = [CHANNEL_TO_INDEX[c] for c in self.selected_channels]
        self.augment = bool(augment)
        self.augmenter = JointMicroscopyAugment()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        row = self.records.iloc[index]
        path = str(row["path"])

        # mmap avoids loading the complete file into RAM before channel selection.
        raw = np.load(path, mmap_mode="r", allow_pickle=False)
        chw = infer_chw(raw)

        if chw.shape[0] == 1 and EXPECTED_CROP_CHANNELS > 1:
            raise ValueError(
                f"{path}: found shape {chw.shape}. This is almost certainly a "
                "mask_crop, not the five-channel image crop. Load the CSV "
                "crop_path column and keep USE_MASKED_CROPS=False."
            )

        if chw.shape[0] != EXPECTED_CROP_CHANNELS:
            raise ValueError(
                f"{path}: expected {EXPECTED_CROP_CHANNELS} channels in order "
                "PHASECONTRAST, BF, DAPI, WGA, FM1, but found "
                f"shape {chw.shape}."
            )

        max_required_index = max(self.channel_indices)
        if chw.shape[0] <= max_required_index:
            raise IndexError(
                f"{path}: array shape {chw.shape} has no channel index "
                f"{max_required_index}. Check CHANNEL_TO_INDEX and ARRAY_LAYOUT."
            )

        selected = []
        for channel_name, channel_index in zip(
            self.selected_channels, self.channel_indices
        ):
            plane = normalise_plane(chw[channel_index], channel_name)
            selected.append(torch.from_numpy(np.ascontiguousarray(plane)))

        x = torch.stack(selected, dim=0).float()

        if tuple(x.shape[-2:]) != (INPUT_SIZE, INPUT_SIZE):
            x = F.interpolate(
                x.unsqueeze(0),
                size=(INPUT_SIZE, INPUT_SIZE),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        if self.augment:
            x = self.augmenter(x)

        y = int(row["target"])
        return x, y, path



# ===== NOTEBOOK CODE CELL 17 =====

train_dataset = MultiChannelNpyDataset(
    train_df,
    selected_channels=SELECTED_CHANNELS,
    augment=USE_AUGMENTATION,
)
val_dataset = MultiChannelNpyDataset(
    val_df,
    selected_channels=SELECTED_CHANNELS,
    augment=False,
)
holdout_dataset = MultiChannelNpyDataset(
    holdout_df,
    selected_channels=SELECTED_CHANNELS,
    augment=False,
)

loader_kwargs = dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT_WORKERS,
)
if NUM_WORKERS > 0:
    loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    drop_last=False,
    **loader_kwargs,
)
val_loader = DataLoader(
    val_dataset,
    shuffle=False,
    drop_last=False,
    **loader_kwargs,
)
holdout_loader = DataLoader(
    holdout_dataset,
    shuffle=False,
    drop_last=False,
    **loader_kwargs,
)

sample_x, sample_y, sample_path = train_dataset[0]
print("Sample path:", sample_path)
print("Sample tensor shape:", tuple(sample_x.shape))
print("Sample label:", CLASS_NAMES[sample_y])
print("Per-channel mean:", sample_x.mean(dim=(1, 2)).tolist())
print("Per-channel std:", sample_x.std(dim=(1, 2)).tolist())



# ===== NOTEBOOK CODE CELL 19 =====

fig, axes = plt.subplots(1, len(SELECTED_CHANNELS), figsize=(4 * len(SELECTED_CHANNELS), 4))
if len(SELECTED_CHANNELS) == 1:
    axes = [axes]

for axis, channel_name, plane in zip(axes, SELECTED_CHANNELS, sample_x):
    axis.imshow(plane.numpy())
    axis.set_title(channel_name)
    axis.axis("off")

plt.suptitle(f"Label: {CLASS_NAMES[sample_y]}")
plt.tight_layout()



# ===== NOTEBOOK CODE CELL 21 =====

def clone_conv_with_input_channels(
    old_conv: nn.Conv2d,
    in_channels: int = 1,
) -> nn.Conv2d:
    """Recreate a convolution exactly as in the ARC training script."""
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
    )


def build_single_channel_classifier(
    architecture: str,
    num_classes: int,
    dropout: float = 0.0,
) -> nn.Module:
    """Mirror build_model_with_dropout() from the ARC training script."""
    architecture = architecture.lower()

    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        model.conv1 = clone_conv_with_input_channels(model.conv1, 1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if dropout > 0:
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )
        return model

    if architecture == "resnet34":
        model = models.resnet34(weights=None)
        model.conv1 = clone_conv_with_input_channels(model.conv1, 1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if dropout > 0:
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )
        return model

    if architecture == "densenet121":
        model = models.densenet121(weights=None)
        model.features.conv0 = clone_conv_with_input_channels(
            model.features.conv0, 1
        )
        model.classifier = nn.Linear(
            model.classifier.in_features, num_classes
        )
        if dropout > 0:
            in_features = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )
        return model

    if architecture == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        model.features[0][0] = clone_conv_with_input_channels(
            model.features[0][0], 1
        )
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features, num_classes
        )
        if dropout > 0:
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )
        return model

    # EfficientNet was not present in the supplied ARC script, but remains
    # available for checkpoints produced by a matching implementation.
    if architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.features[0][0] = clone_conv_with_input_channels(
            model.features[0][0], 1
        )
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features, num_classes
        )
        if dropout > 0:
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )
        return model

    raise ValueError(f"Unsupported architecture: {architecture}")


CHECKPOINT_EXTENSIONS = {".pt", ".pth", ".ckpt", ".bin"}


def checkpoint_priority(path: Path) -> Tuple[int, float]:
    name = path.name.lower()
    score = 0
    if name == "best_model.pt":
        score += 1000
    elif "best" in name:
        score += 100
    if "macro_f1" in name or "macrof1" in name:
        score += 30
    if "val_f1" in name or "valf1" in name:
        score += 20
    if "latest" in name:
        score -= 20
    return score, path.stat().st_mtime


def resolve_checkpoint(source: Path) -> Path:
    source = Path(source)
    if source.is_file():
        return source
    if not source.exists():
        raise FileNotFoundError(source)
    if not source.is_dir():
        raise ValueError(
            f"Weight source is neither a file nor folder: {source}"
        )

    candidates = [
        p for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in CHECKPOINT_EXTENSIONS
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint files found below {source}"
        )

    candidates.sort(key=checkpoint_priority, reverse=True)
    return candidates[0]


def load_arc_checkpoint(path: Path) -> Dict[str, Any]:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"{path} is not an ARC checkpoint dictionary."
        )
    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"{path} has no 'model_state_dict'. Point to best_model.pt "
            "from the individual-channel ARC training run."
        )
    return dict(checkpoint)


def normalisation_config_from_checkpoint(
    checkpoint: Mapping[str, Any],
    channel_name: str,
) -> Dict[str, Any]:
    mode = str(
        checkpoint.get(
            "normalization_mode",
            checkpoint.get("normalization_stats", {}).get(
                "normalization_mode", ""
            ),
        )
    ).lower()

    stats = checkpoint.get("normalization_stats", {}) or {}

    if mode == "per_crop_zscore":
        return {"mode": "per_crop_zscore"}

    if mode in {"percentile_minmax", "per_crop_percentile"}:
        return {"mode": "per_crop_percentile"}

    if mode == "train_global_zscore":
        means = stats.get("mean")
        stds = stats.get("std")
        if not means or not stds:
            raise ValueError(
                f"{channel_name}: checkpoint says train_global_zscore but "
                "does not contain normalization_stats mean/std."
            )
        return {
            "mode": "global_zscore",
            "mean": float(means[0]),
            "std": float(stds[0]),
        }

    if mode == "none":
        return {"mode": "none"}

    raise ValueError(
        f"{channel_name}: unsupported or missing checkpoint "
        f"normalization mode: {mode!r}"
    )


def validate_arc_checkpoint_metadata(
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    channel_name: str,
    configured_architecture: str,
) -> None:
    errors: List[str] = []

    checkpoint_architecture = str(
        checkpoint.get("model_name", "")
    ).lower()
    if checkpoint_architecture and checkpoint_architecture not in SUPPORTED_ARCHITECTURES:
        errors.append(
            f"checkpoint architecture {checkpoint_architecture!r} is unsupported"
        )

    checkpoint_task_mode = str(checkpoint.get("task_mode", "")).lower()
    if checkpoint_task_mode and checkpoint_task_mode != "panel":
        errors.append(
            f"task_mode is {checkpoint_task_mode!r}; expected 'panel'"
        )

    checkpoint_panel_name = checkpoint.get("panel_definition_name")
    if (
        checkpoint_panel_name is not None
        and str(checkpoint_panel_name) != PANEL_DEFINITION_NAME
    ):
        errors.append(
            f"panel_definition_name is {checkpoint_panel_name!r}; "
            f"expected {PANEL_DEFINITION_NAME!r}"
        )

    checkpoint_panel_to_species = checkpoint.get("panel_to_species")
    if checkpoint_panel_to_species is not None:
        canonical_checkpoint_panel = {
            str(panel): [str(s).upper() for s in species_list]
            for panel, species_list in checkpoint_panel_to_species.items()
        }
        if canonical_checkpoint_panel != PANEL_TO_SPECIES:
            errors.append(
                "panel_to_species in the checkpoint differs from the current "
                "Panel-2 definition"
            )

    checkpoint_channel_names = [
        str(x).upper()
        for x in checkpoint.get("selected_channel_names", [])
    ]
    if checkpoint_channel_names and checkpoint_channel_names != [channel_name.upper()]:
        errors.append(
            f"selected_channel_names is {checkpoint_channel_names}, "
            f"expected {[channel_name.upper()]}"
        )

    expected_index = CHANNEL_TO_INDEX[channel_name]
    checkpoint_indices = checkpoint.get("selected_channels")
    if checkpoint_indices is not None and list(checkpoint_indices) != [expected_index]:
        errors.append(
            f"selected_channels is {list(checkpoint_indices)}, "
            f"expected {[expected_index]}"
        )

    checkpoint_class_to_idx = checkpoint.get("class_to_idx")
    if checkpoint_class_to_idx is not None:
        checkpoint_class_to_idx = {
            str(k): int(v)
            for k, v in checkpoint_class_to_idx.items()
        }
        if checkpoint_class_to_idx != CLASS_TO_INDEX:
            errors.append(
                "class_to_idx does not exactly match the current split CSVs. "
                f"Checkpoint={checkpoint_class_to_idx}; current={CLASS_TO_INDEX}"
            )

    checkpoint_channel_map = checkpoint.get("channel_map")
    if checkpoint_channel_map is not None:
        checkpoint_channel_map = {
            str(k).upper(): int(v)
            for k, v in checkpoint_channel_map.items()
        }
        if checkpoint_channel_map != CHANNEL_TO_INDEX:
            errors.append(
                f"channel_map differs. Checkpoint={checkpoint_channel_map}; "
                f"current={CHANNEL_TO_INDEX}"
            )

    checkpoint_crop_order = checkpoint.get("crop_channel_order")
    expected_order = [
        name for name, _ in sorted(
            CHANNEL_TO_INDEX.items(), key=lambda item: item[1]
        )
    ]
    if checkpoint_crop_order is not None:
        checkpoint_crop_order = [
            str(x).upper() for x in checkpoint_crop_order
        ]
        if checkpoint_crop_order != expected_order:
            errors.append(
                f"crop_channel_order differs. "
                f"Checkpoint={checkpoint_crop_order}; current={expected_order}"
            )

    if bool(checkpoint.get("use_masked_crops", False)):
        errors.append(
            "checkpoint was trained with use_masked_crops=True, but this "
            "fusion notebook uses five-channel crop_path arrays"
        )

    if errors:
        joined = "\n  - ".join(errors)
        raise ValueError(
            f"Checkpoint metadata mismatch for {channel_name}:\n"
            f"  checkpoint: {checkpoint_path}\n"
            f"  - {joined}"
        )


def strict_load_arc_weights(
    model: nn.Module,
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
) -> Dict[str, Any]:
    state_dict = checkpoint["model_state_dict"]

    # DataParallel compatibility only. No shape skipping or partial loading.
    if state_dict and all(
        str(key).startswith("module.") for key in state_dict
    ):
        state_dict = {
            str(key)[len("module."):]: value
            for key, value in state_dict.items()
        }

    incompatible = model.load_state_dict(
        state_dict,
        strict=STRICT_CHECKPOINT_LOADING,
    )

    report = {
        "checkpoint_path": str(checkpoint_path),
        "strict": bool(STRICT_CHECKPOINT_LOADING),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "best_val_macro_f1": checkpoint.get("best_val_macro_f1"),
        "model_name": checkpoint.get("model_name"),
        "best_dropout": float(checkpoint.get("best_dropout", 0.0)),
        "selected_channel_names": checkpoint.get(
            "selected_channel_names"
        ),
        "selected_channels": checkpoint.get("selected_channels"),
        "normalization_mode": checkpoint.get("normalization_mode"),
        "class_to_idx": checkpoint.get("class_to_idx"),
    }

    if STRICT_CHECKPOINT_LOADING and (
        report["missing_keys"] or report["unexpected_keys"]
    ):
        raise RuntimeError(
            f"Strict checkpoint loading unexpectedly reported incompatible "
            f"keys for {checkpoint_path}: {report}"
        )

    return report



# ===== NOTEBOOK CODE CELL 23 =====

resolved_checkpoints: Dict[str, str] = {}
checkpoint_loading_reports: Dict[str, Dict[str, Any]] = {}
channel_classifiers: Dict[str, nn.Module] = {}
arc_checkpoints: Dict[str, Dict[str, Any]] = {}

for channel_name in SELECTED_CHANNELS:
    configured_architecture = BRANCH_ARCHITECTURES[channel_name]
    checkpoint_path = resolve_checkpoint(WEIGHT_SOURCES[channel_name])
    checkpoint = load_arc_checkpoint(checkpoint_path)

    if VALIDATE_CHECKPOINT_METADATA:
        validate_arc_checkpoint_metadata(
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            channel_name=channel_name,
            configured_architecture=configured_architecture,
        )

    checkpoint_architecture = str(
        checkpoint.get("model_name", configured_architecture)
    ).lower()
    if checkpoint_architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"{channel_name}: unsupported checkpoint architecture "
            f"{checkpoint_architecture!r}"
        )
    # The checkpoint metadata is authoritative. This also makes the later
    # fine-tuning policy target the correct architecture if channel checkpoints
    # were trained with different supported backbones.
    BRANCH_ARCHITECTURES[channel_name] = checkpoint_architecture
    dropout = float(checkpoint.get("best_dropout", 0.0))

    if USE_CHECKPOINT_NORMALIZATION:
        NORMALIZATION_BY_CHANNEL[channel_name] = (
            normalisation_config_from_checkpoint(
                checkpoint, channel_name
            )
        )

    model = build_single_channel_classifier(
        architecture=checkpoint_architecture,
        num_classes=NUM_CLASSES,
        dropout=dropout,
    )

    report = strict_load_arc_weights(
        model=model,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
    )

    resolved_checkpoints[channel_name] = str(checkpoint_path)
    checkpoint_loading_reports[channel_name] = report
    channel_classifiers[channel_name] = model
    arc_checkpoints[channel_name] = checkpoint

    print("=" * 80)
    print("Channel:", channel_name)
    print("Checkpoint:", checkpoint_path)
    print("Architecture:", checkpoint_architecture)
    print("Dropout:", dropout)
    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Checkpoint best val macro-F1:", checkpoint.get("best_val_macro_f1"))
    print("Normalization:", NORMALIZATION_BY_CHANNEL[channel_name])
    print("Strict load: PASSED")
    print("Missing keys:", report["missing_keys"])
    print("Unexpected keys:", report["unexpected_keys"])

(RUN_DIR / "resolved_checkpoints.json").write_text(
    json.dumps(resolved_checkpoints, indent=2), encoding="utf-8"
)
(RUN_DIR / "checkpoint_loading_reports.json").write_text(
    json.dumps(
        json_safe(checkpoint_loading_reports), indent=2
    ),
    encoding="utf-8",
)
(RUN_DIR / "effective_normalization_from_checkpoints.json").write_text(
    json.dumps(
        json_safe({
            channel: NORMALIZATION_BY_CHANNEL[channel]
            for channel in SELECTED_CHANNELS
        }),
        indent=2,
    ),
    encoding="utf-8",
)



# ===== NOTEBOOK CODE CELL 25 =====

for channel_name in SELECTED_CHANNELS:
    checkpoint = arc_checkpoints[channel_name]
    checkpoint_mapping = checkpoint.get("class_to_idx")

    if checkpoint_mapping is None:
        explicit_names = CHECKPOINT_CLASS_NAMES.get(channel_name)
        if explicit_names is None:
            raise ValueError(
                f"{channel_name}: checkpoint has no class_to_idx and no "
                "CHECKPOINT_CLASS_NAMES were supplied. Individual holdout "
                "predictions cannot be interpreted safely."
            )
        checkpoint_mapping = {
            str(name): index
            for index, name in enumerate(explicit_names)
        }

    checkpoint_mapping = {
        str(k): int(v) for k, v in checkpoint_mapping.items()
    }

    if checkpoint_mapping != CLASS_TO_INDEX:
        raise ValueError(
            f"{channel_name}: checkpoint class mapping differs from the "
            f"current dataset.\nCheckpoint: {checkpoint_mapping}\n"
            f"Current: {CLASS_TO_INDEX}"
        )

    print(
        f"{channel_name}: class order verified exactly for "
        f"{len(CLASS_NAMES)} Panel-2 classes."
    )



# ===== NOTEBOOK CODE CELL 27 =====

@torch.inference_mode()
def predict_single_channel(
    model: nn.Module,
    loader: DataLoader,
    channel_position: int,
    description: str,
) -> Dict[str, Any]:
    model = model.to(DEVICE)
    model.eval()

    all_targets: List[np.ndarray] = []
    all_probabilities: List[np.ndarray] = []
    all_paths: List[str] = []

    for x, y, paths in tqdm(loader, desc=description, leave=False):
        x_channel = x[:, channel_position:channel_position + 1].to(
            DEVICE, non_blocking=True
        )
        logits = model(x_channel)
        probabilities = torch.softmax(logits, dim=1)

        all_targets.append(y.numpy())
        all_probabilities.append(probabilities.cpu().numpy())
        all_paths.extend(list(paths))

    targets = np.concatenate(all_targets)
    probabilities = np.concatenate(all_probabilities)
    predictions = probabilities.argmax(axis=1)

    return {
        "targets": targets,
        "probabilities": probabilities,
        "predictions": predictions,
        "paths": all_paths,
    }

def calculate_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(targets, predictions)
        ),
        "macro_f1": float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(targets, predictions, average="weighted", zero_division=0)
        ),
    }

    if probabilities.shape[1] >= 2:
        metrics["top2_accuracy"] = float(
            top_k_accuracy_score(
                targets,
                probabilities,
                k=min(2, probabilities.shape[1]),
                labels=np.arange(probabilities.shape[1]),
            )
        )
    return metrics

def save_prediction_bundle(
    prefix: str,
    targets: np.ndarray,
    probabilities: np.ndarray,
    paths: Sequence[str],
) -> Dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)

    output = pd.DataFrame({
        "path": list(paths),
        "true_index": targets,
        "true_label": [CLASS_NAMES[i] for i in targets],
        "predicted_index": predictions,
        "predicted_label": [CLASS_NAMES[i] for i in predictions],
        "confidence": confidence,
        "correct": predictions == targets,
    })

    if SAVE_ALL_CLASS_PROBABILITIES:
        for class_index, class_name in enumerate(CLASS_NAMES):
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", class_name)
            output[f"prob_{safe_name}"] = probabilities[:, class_index]

    output.to_csv(RUN_DIR / f"{prefix}_predictions.csv", index=False)

    report_dict = classification_report(
        targets,
        predictions,
        labels=np.arange(NUM_CLASSES),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report_dict).T.to_csv(
        RUN_DIR / f"{prefix}_classification_report.csv"
    )
    (RUN_DIR / f"{prefix}_classification_report.json").write_text(
        json.dumps(json_safe(report_dict), indent=2),
        encoding="utf-8",
    )

    cm = confusion_matrix(
        targets,
        predictions,
        labels=np.arange(NUM_CLASSES),
    )
    pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(RUN_DIR / f"{prefix}_confusion_matrix_counts.csv")

    cm_normalised = confusion_matrix(
        targets,
        predictions,
        labels=np.arange(NUM_CLASSES),
        normalize="true",
    )
    pd.DataFrame(
        cm_normalised,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(RUN_DIR / f"{prefix}_confusion_matrix_normalised.csv")

    fig, ax = plt.subplots(figsize=(max(8, NUM_CLASSES * 0.7), max(7, NUM_CLASSES * 0.65)))
    image = ax.imshow(cm_normalised)
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(NUM_CLASSES),
        yticks=np.arange(NUM_CLASSES),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted Panel-2 class",
        ylabel="True Panel-2 class",
        title=f"{prefix}: normalised confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    fig.savefig(RUN_DIR / f"{prefix}_confusion_matrix.png", dpi=200)

    plt.close(fig)

    metrics = calculate_metrics(targets, probabilities)
    (RUN_DIR / f"{prefix}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics



# ===== NOTEBOOK CODE CELL 29 =====

individual_holdout_outputs: Dict[str, Dict[str, Any]] = {}
individual_holdout_metrics: Dict[str, Dict[str, float]] = {}

for channel_position, channel_name in enumerate(SELECTED_CHANNELS):
    model = channel_classifiers[channel_name]
    output = predict_single_channel(
        model=model,
        loader=holdout_loader,
        channel_position=channel_position,
        description=f"Holdout: {channel_name}",
    )
    individual_holdout_outputs[channel_name] = output

    prefix = f"holdout_individual_{channel_name}"
    metrics = save_prediction_bundle(
        prefix=prefix,
        targets=output["targets"],
        probabilities=output["probabilities"],
        paths=output["paths"],
    )
    individual_holdout_metrics[channel_name] = metrics

    print(channel_name, metrics)

    # Keep RAM copy of the model for fusion, but release GPU activations.
    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

individual_metrics_df = pd.DataFrame(individual_holdout_metrics).T
display(individual_metrics_df)
individual_metrics_df.to_csv(RUN_DIR / "holdout_individual_channel_metrics.csv")


# A very low result usually means the wrong checkpoint, class order,
# preprocessing, or holdout file. Stop before training fusion in that case.
for channel_name, metrics in individual_holdout_metrics.items():
    if metrics["accuracy"] < (1.0 / NUM_CLASSES) * 0.75:
        raise RuntimeError(
            f"{channel_name} holdout accuracy ({metrics['accuracy']:.4f}) is "
            "below 75% of random-chance accuracy. The original checkpoint "
            "has not been reproduced reliably, so fusion training has been "
            "stopped. Inspect the strict loading and metadata output above."
        )



# ===== NOTEBOOK CODE CELL 31 =====

reference_targets = individual_holdout_outputs[SELECTED_CHANNELS[0]]["targets"]
reference_paths = individual_holdout_outputs[SELECTED_CHANNELS[0]]["paths"]

for channel_name in SELECTED_CHANNELS[1:]:
    if not np.array_equal(
        reference_targets,
        individual_holdout_outputs[channel_name]["targets"],
    ):
        raise RuntimeError("Channel baseline target order does not match.")
    if reference_paths != individual_holdout_outputs[channel_name]["paths"]:
        raise RuntimeError("Channel baseline sample order does not match.")

mean_probability_ensemble = np.mean(
    [
        individual_holdout_outputs[channel_name]["probabilities"]
        for channel_name in SELECTED_CHANNELS
    ],
    axis=0,
)

ensemble_metrics = save_prediction_bundle(
    prefix="holdout_mean_probability_ensemble",
    targets=reference_targets,
    probabilities=mean_probability_ensemble,
    paths=reference_paths,
)
print("Mean-probability ensemble:", ensemble_metrics)



# ===== NOTEBOOK CODE CELL 33 =====

def final_linear_in_features(module: nn.Module) -> int:
    if isinstance(module, nn.Linear):
        return int(module.in_features)

    linear_layers = [
        child for child in module.modules()
        if isinstance(child, nn.Linear)
    ]
    if not linear_layers:
        raise TypeError(
            f"Could not find a Linear layer inside {module}"
        )
    return int(linear_layers[-1].in_features)


def remove_classifier(
    model: nn.Module,
    architecture: str,
) -> Tuple[nn.Module, int]:
    architecture = architecture.lower()

    if architecture in {"resnet18", "resnet34"}:
        feature_dim = final_linear_in_features(model.fc)
        model.fc = nn.Identity()
        return model, feature_dim

    if architecture == "densenet121":
        feature_dim = final_linear_in_features(model.classifier)
        model.classifier = nn.Identity()
        return model, feature_dim

    if architecture == "convnext_tiny":
        feature_dim = final_linear_in_features(model.classifier[-1])
        model.classifier[-1] = nn.Identity()
        return model, feature_dim

    if architecture == "efficientnet_b0":
        feature_dim = final_linear_in_features(model.classifier[-1])
        model.classifier[-1] = nn.Identity()
        return model, feature_dim

    raise ValueError(f"Unsupported architecture: {architecture}")


feature_branches: Dict[str, nn.Module] = {}
feature_dimensions: Dict[str, int] = {}

for channel_name in SELECTED_CHANNELS:
    checkpoint_architecture = str(
        arc_checkpoints[channel_name].get(
            "model_name",
            BRANCH_ARCHITECTURES[channel_name],
        )
    ).lower()

    branch, feature_dim = remove_classifier(
        channel_classifiers[channel_name],
        checkpoint_architecture,
    )
    feature_branches[channel_name] = branch
    feature_dimensions[channel_name] = feature_dim

    print(
        f"{channel_name}: {checkpoint_architecture} "
        f"-> {feature_dim} features"
    )



# ===== NOTEBOOK CODE CELL 35 =====

class SEFeatureBlock(nn.Module):
    """Squeeze-and-excitation block for an already pooled feature vector.

    Each projected branch has shape [batch, projection_dim]. The bottleneck MLP
    learns a sigmoid weight for every projected feature dimension. The mean of
    those feature-wise weights is also returned as an interpretable per-channel
    attention summary.
    """
    def __init__(self, feature_dim: int, reduction_ratio: int = 4) -> None:
        super().__init__()
        bottleneck_dim = max(feature_dim // reduction_ratio, 1)
        self.excitation = nn.Sequential(
            nn.Linear(feature_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_weights = self.excitation(features)
        reweighted = features * feature_weights
        channel_summary = feature_weights.mean(dim=1)
        return reweighted, channel_summary


class MultiBranchFeatureFusion(nn.Module):
    def __init__(
        self,
        branches: Mapping[str, nn.Module],
        feature_dimensions: Mapping[str, int],
        selected_channels: Sequence[str],
        num_classes: int,
        projection_dim: int,
        hidden_dim: int,
        dropout: float,
        fusion_mode: str = "concat",
        se_reduction_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.selected_channels = list(selected_channels)
        self.branches = nn.ModuleDict(
            {name: branches[name] for name in self.selected_channels}
        )
        self.fusion_mode = fusion_mode

        self.projections = nn.ModuleDict()
        for name in self.selected_channels:
            self.projections[name] = nn.Sequential(
                nn.Linear(feature_dimensions[name], projection_dim),
                nn.LayerNorm(projection_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        if fusion_mode == "concat":
            self.se_blocks = None
        elif fusion_mode == "se_attention":
            self.se_blocks = nn.ModuleDict({
                name: SEFeatureBlock(projection_dim, se_reduction_ratio)
                for name in self.selected_channels
            })
        else:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")

        # SE reweights each branch but preserves every branch vector, so the
        # attended vectors are concatenated exactly as in the concat baseline.
        fused_dim = projection_dim * len(self.selected_channels)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_gates: bool = False,
    ) -> Any:
        attended_features = []
        channel_attention_summaries = []

        for channel_position, channel_name in enumerate(self.selected_channels):
            channel_tensor = x[:, channel_position:channel_position + 1]
            features = self.branches[channel_name](channel_tensor)
            projected = self.projections[channel_name](features)

            if self.fusion_mode == "se_attention":
                projected, channel_summary = self.se_blocks[channel_name](projected)
                channel_attention_summaries.append(channel_summary)

            attended_features.append(projected)

        fused = torch.cat(attended_features, dim=1)
        logits = self.classifier(fused)

        if return_gates:
            if self.fusion_mode == "se_attention":
                summaries = torch.stack(channel_attention_summaries, dim=1)
                return logits, summaries
            return logits, None
        return logits

fusion_model = MultiBranchFeatureFusion(
    branches=feature_branches,
    feature_dimensions=feature_dimensions,
    selected_channels=SELECTED_CHANNELS,
    num_classes=NUM_CLASSES,
    projection_dim=PROJECTION_DIM,
    hidden_dim=FUSION_HIDDEN_DIM,
    dropout=FUSION_DROPOUT,
    fusion_mode=FUSION_MODE,
    se_reduction_ratio=SE_REDUCTION_RATIO,
)

def count_parameters(model: nn.Module) -> Dict[str, int]:
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

print(fusion_model)
print("Parameter counts:", count_parameters(fusion_model))



# ===== NOTEBOOK CODE CELL 37 =====

def set_all_branch_trainable(model: MultiBranchFeatureFusion, value: bool) -> None:
    for parameter in model.branches.parameters():
        parameter.requires_grad = value

def freeze_branches(model: MultiBranchFeatureFusion) -> None:
    set_all_branch_trainable(model, False)

def unfreeze_last_block_for_branch(
    branch: nn.Module,
    architecture: str,
) -> None:
    architecture = architecture.lower()

    if architecture in {"resnet18", "resnet34"}:
        modules = [branch.layer4]
    elif architecture == "densenet121":
        modules = [branch.features.denseblock4, branch.features.norm5]
    elif architecture == "convnext_tiny":
        modules = [branch.features[-1]]
    elif architecture == "efficientnet_b0":
        modules = [branch.features[-2], branch.features[-1]]
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True

def apply_unfreeze_policy(
    model: MultiBranchFeatureFusion,
    policy: str,
) -> None:
    freeze_branches(model)

    if policy == "none":
        return

    if policy == "all":
        set_all_branch_trainable(model, True)
        return

    if policy == "last_block":
        for channel_name in SELECTED_CHANNELS:
            unfreeze_last_block_for_branch(
                model.branches[channel_name],
                BRANCH_ARCHITECTURES[channel_name],
            )
        return

    raise ValueError(f"Unsupported unfreeze policy: {policy}")

freeze_branches(fusion_model)
print("Frozen-head stage parameters:", count_parameters(fusion_model))



# ===== NOTEBOOK CODE CELL 39 =====

class_counts = (
    train_df["target"]
    .value_counts()
    .reindex(np.arange(NUM_CLASSES), fill_value=0)
    .sort_index()
    .to_numpy(dtype=np.float64)
)

if np.any(class_counts == 0):
    raise ValueError("At least one class has zero training samples.")

class_weights = class_counts.sum() / (NUM_CLASSES * class_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights if USE_CLASS_WEIGHTED_LOSS else None,
    label_smoothing=LABEL_SMOOTHING,
)

def autocast_context():
    if DEVICE.type == "cuda" and USE_AMP:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()

def make_grad_scaler():
    enabled = DEVICE.type == "cuda" and USE_AMP
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)

def build_optimizer(
    model: MultiBranchFeatureFusion,
    head_lr: float,
    backbone_lr: Optional[float],
) -> torch.optim.Optimizer:
    head_parameters = [
        p for p in list(model.projections.parameters())
        + list(model.classifier.parameters())
        + ([] if model.se_blocks is None else list(model.se_blocks.parameters()))
        if p.requires_grad
    ]
    branch_parameters = [
        p for p in model.branches.parameters() if p.requires_grad
    ]

    parameter_groups = [
        {"params": head_parameters, "lr": head_lr},
    ]
    if branch_parameters:
        if backbone_lr is None:
            raise ValueError("backbone_lr must be supplied when branch parameters are trainable.")
        parameter_groups.append(
            {"params": branch_parameters, "lr": backbone_lr}
        )

    return torch.optim.AdamW(
        parameter_groups,
        weight_decay=WEIGHT_DECAY,
    )

def run_training_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_samples = 0
    all_targets: List[np.ndarray] = []
    all_predictions: List[np.ndarray] = []

    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(loader, desc="Train", leave=False)
    for batch_index, (x, y, _) in enumerate(progress):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        with autocast_context():
            logits = model(x)
            loss = criterion(logits, y)
            scaled_loss = loss / GRADIENT_ACCUMULATION_STEPS

        scaler.scale(scaled_loss).backward()

        should_step = (
            (batch_index + 1) % GRADIENT_ACCUMULATION_STEPS == 0
            or (batch_index + 1) == len(loader)
        )
        if should_step:
            scaler.unscale_(optimizer)
            if MAX_GRAD_NORM is not None:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    MAX_GRAD_NORM,
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_size_actual = y.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size_actual
        total_samples += batch_size_actual

        predictions = logits.detach().argmax(dim=1)
        all_targets.append(y.detach().cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())

        progress.set_postfix(loss=float(loss.detach().cpu()))

    targets = np.concatenate(all_targets)
    predictions = np.concatenate(all_predictions)

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        ),
    }

@torch.inference_mode()
def run_validation_epoch(
    model: nn.Module,
    loader: DataLoader,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_targets: List[np.ndarray] = []
    all_predictions: List[np.ndarray] = []

    for x, y, _ in tqdm(loader, desc="Validation", leave=False):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        with autocast_context():
            logits = model(x)
            loss = criterion(logits, y)

        batch_size_actual = y.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size_actual
        total_samples += batch_size_actual

        predictions = logits.argmax(dim=1)
        all_targets.append(y.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())

    targets = np.concatenate(all_targets)
    predictions = np.concatenate(all_predictions)

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        ),
    }

BEST_CHECKPOINT_PATH = RUN_DIR / "best_fusion_model.pt"
training_history: List[Dict[str, Any]] = []
best_val_macro_f1 = -float("inf")
best_epoch_record: Optional[Dict[str, Any]] = None

def save_fusion_checkpoint(
    model: MultiBranchFeatureFusion,
    epoch_record: Dict[str, Any],
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "selected_channels": SELECTED_CHANNELS,
        "channel_to_index": CHANNEL_TO_INDEX,
        "branch_architectures": BRANCH_ARCHITECTURES,
        "feature_dimensions": feature_dimensions,
        "resolved_checkpoints": resolved_checkpoints,
        "fusion_mode": FUSION_MODE,
        "se_reduction_ratio": SE_REDUCTION_RATIO,
        "projection_dim": PROJECTION_DIM,
        "fusion_hidden_dim": FUSION_HIDDEN_DIM,
        "fusion_dropout": FUSION_DROPOUT,
        "normalization_by_channel": NORMALIZATION_BY_CHANNEL,
        "input_size": INPUT_SIZE,
        "best_epoch_record": epoch_record,
        "run_config": USER_CONFIG,
    }
    torch.save(payload, BEST_CHECKPOINT_PATH)

def train_stage(
    model: MultiBranchFeatureFusion,
    stage_name: str,
    epochs: int,
    head_lr: float,
    backbone_lr: Optional[float],
) -> None:
    global best_val_macro_f1, best_epoch_record, training_history

    if epochs <= 0:
        print(f"Skipping {stage_name}: epochs={epochs}")
        return

    optimizer = build_optimizer(
        model,
        head_lr=head_lr,
        backbone_lr=backbone_lr,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
    )
    scaler = make_grad_scaler()

    stage_best = -float("inf")
    stage_epochs_without_improvement = 0

    for stage_epoch in range(1, epochs + 1):
        start_time = time.time()

        train_metrics = run_training_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
        )
        val_metrics = run_validation_epoch(
            model=model,
            loader=val_loader,
        )

        scheduler.step(val_metrics["macro_f1"])
        elapsed = time.time() - start_time

        epoch_record = {
            "stage": stage_name,
            "stage_epoch": stage_epoch,
            "global_epoch": len(training_history) + 1,
            "elapsed_seconds": elapsed,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "learning_rates": [
                group["lr"] for group in optimizer.param_groups
            ],
            "trainable_parameters": count_parameters(model)["trainable"],
        }
        training_history.append(epoch_record)

        pd.DataFrame(training_history).to_csv(
            RUN_DIR / "training_history.csv", index=False
        )
        (RUN_DIR / "training_history.json").write_text(
            json.dumps(json_safe(training_history), indent=2),
            encoding="utf-8",
        )

        if SYNC_EVERY_EPOCH:
            sync_outputs_to_persistent([
                RUN_DIR / "training_history.csv",
                RUN_DIR / "training_history.json",
                RUN_DIR / "best_fusion_model.pt",
                RUN_DIR / "run_log.txt",
                RUN_DIR / "run_config_initial.json",
                RUN_DIR / "environment.json",
            ], label=f"{stage_name}_epoch_{stage_epoch:03d}")

        print(
            f"[{stage_name}] epoch {stage_epoch:03d}/{epochs:03d} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"train F1 {train_metrics['macro_f1']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | "
            f"val F1 {val_metrics['macro_f1']:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch_record = copy.deepcopy(epoch_record)
            save_fusion_checkpoint(model, epoch_record)
            print(f"  Saved new overall best model: val macro-F1={best_val_macro_f1:.4f}")
            if SYNC_EVERY_EPOCH:
                sync_outputs_to_persistent(
                    [BEST_CHECKPOINT_PATH, RUN_DIR / "run_log.txt"],
                    label=f"{stage_name}_epoch_{stage_epoch:03d}_new_best",
                )

        if val_metrics["macro_f1"] > stage_best:
            stage_best = val_metrics["macro_f1"]
            stage_epochs_without_improvement = 0
        else:
            stage_epochs_without_improvement += 1

        if stage_epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping {stage_name} after "
                f"{stage_epochs_without_improvement} epochs without improvement."
            )
            break



# ===== NOTEBOOK CODE CELL 41 =====

fusion_model = fusion_model.to(DEVICE)
freeze_branches(fusion_model)

print("Stage 1 parameter counts:", count_parameters(fusion_model))

train_stage(
    model=fusion_model,
    stage_name="frozen_fusion_head",
    epochs=FROZEN_HEAD_EPOCHS,
    head_lr=HEAD_LEARNING_RATE,
    backbone_lr=None,
)



# ===== NOTEBOOK CODE CELL 43 =====

if FINETUNE_EPOCHS > 0 and UNFREEZE_POLICY != "none":
    stage1_checkpoint = torch.load(
        BEST_CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )
    fusion_model.load_state_dict(stage1_checkpoint["model_state_dict"])

    apply_unfreeze_policy(fusion_model, UNFREEZE_POLICY)
    print("Stage 2 parameter counts:", count_parameters(fusion_model))

    train_stage(
        model=fusion_model,
        stage_name=f"finetune_{UNFREEZE_POLICY}",
        epochs=FINETUNE_EPOCHS,
        head_lr=FINETUNE_HEAD_LEARNING_RATE,
        backbone_lr=BACKBONE_LEARNING_RATE,
    )
else:
    print("Joint fine-tuning skipped.")



# ===== NOTEBOOK CODE CELL 45 =====

history_df = pd.DataFrame(training_history)
display(history_df.tail())

if not history_df.empty:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history_df["global_epoch"], history_df["train_macro_f1"], label="Train macro-F1")
    ax.plot(history_df["global_epoch"], history_df["val_macro_f1"], label="Validation macro-F1")
    ax.set_xlabel("Global epoch")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Fusion training history")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RUN_DIR / "training_macro_f1.png", dpi=200)

    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history_df["global_epoch"], history_df["train_loss"], label="Train loss")
    ax.plot(history_df["global_epoch"], history_df["val_loss"], label="Validation loss")
    ax.set_xlabel("Global epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Fusion training loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RUN_DIR / "training_loss.png", dpi=200)

    plt.close(fig)



# ===== NOTEBOOK CODE CELL 47 =====

best_checkpoint = torch.load(
    BEST_CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False,
)
fusion_model.load_state_dict(best_checkpoint["model_state_dict"])
fusion_model = fusion_model.to(DEVICE)
fusion_model.eval()

print("Best validation record:")
print(json.dumps(json_safe(best_checkpoint["best_epoch_record"]), indent=2))



# ===== NOTEBOOK CODE CELL 48 =====

@torch.inference_mode()
def predict_fusion(
    model: MultiBranchFeatureFusion,
    loader: DataLoader,
) -> Dict[str, Any]:
    model.eval()

    all_targets: List[np.ndarray] = []
    all_probabilities: List[np.ndarray] = []
    all_paths: List[str] = []
    all_gates: List[np.ndarray] = []

    for x, y, paths in tqdm(loader, desc="Fusion holdout"):
        x = x.to(DEVICE, non_blocking=True)

        with autocast_context():
            logits, gates = model(x, return_gates=True)
            probabilities = torch.softmax(logits, dim=1)

        all_targets.append(y.numpy())
        all_probabilities.append(probabilities.cpu().numpy())
        all_paths.extend(list(paths))
        if gates is not None:
            all_gates.append(gates.cpu().numpy())

    result = {
        "targets": np.concatenate(all_targets),
        "probabilities": np.concatenate(all_probabilities),
        "paths": all_paths,
        "gates": np.concatenate(all_gates) if all_gates else None,
    }
    result["predictions"] = result["probabilities"].argmax(axis=1)
    return result

fusion_holdout_output = predict_fusion(
    model=fusion_model,
    loader=holdout_loader,
)

fusion_holdout_metrics = save_prediction_bundle(
    prefix="holdout_feature_fusion",
    targets=fusion_holdout_output["targets"],
    probabilities=fusion_holdout_output["probabilities"],
    paths=fusion_holdout_output["paths"],
)

print("Feature-fusion holdout metrics:")
print(json.dumps(fusion_holdout_metrics, indent=2))



# ===== NOTEBOOK CODE CELL 50 =====

if fusion_holdout_output["gates"] is not None:
    gate_df = pd.DataFrame(
        fusion_holdout_output["gates"],
        columns=[f"gate_{c}" for c in SELECTED_CHANNELS],
    )
    gate_df.insert(0, "path", fusion_holdout_output["paths"])
    gate_df.insert(
        1,
        "true_label",
        [CLASS_NAMES[i] for i in fusion_holdout_output["targets"]],
    )
    gate_df.to_csv(RUN_DIR / "holdout_se_channel_attention.csv", index=False)

    mean_gates = gate_df[[f"gate_{c}" for c in SELECTED_CHANNELS]].mean()
    display(mean_gates.to_frame("mean_se_attention"))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(SELECTED_CHANNELS, mean_gates.values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean SE attention")
    ax.set_title("Average SE attention by microscopy channel on holdout")
    fig.tight_layout()
    fig.savefig(RUN_DIR / "holdout_mean_se_channel_attention.png", dpi=200)

    plt.close(fig)
else:
    print("No SE attention values to analyse because FUSION_MODE='concat'.")



# ===== NOTEBOOK CODE CELL 52 =====

comparison_rows = []

for channel_name, metrics in individual_holdout_metrics.items():
    comparison_rows.append({
        "method": f"Individual: {channel_name}",
        **metrics,
    })

comparison_rows.append({
    "method": "Mean probability ensemble",
    **ensemble_metrics,
})
comparison_rows.append({
    "method": f"Feature fusion: {FUSION_MODE}",
    **fusion_holdout_metrics,
})

comparison_df = (
    pd.DataFrame(comparison_rows)
    .sort_values("macro_f1", ascending=False)
    .reset_index(drop=True)
)
display(comparison_df)

comparison_df.to_csv(
    RUN_DIR / "holdout_method_comparison.csv",
    index=False,
)
(RUN_DIR / "holdout_method_comparison.json").write_text(
    json.dumps(
        json_safe(comparison_df.to_dict(orient="records")),
        indent=2,
    ),
    encoding="utf-8",
)

fig, ax = plt.subplots(figsize=(max(8, len(comparison_df) * 1.7), 5))
ax.bar(comparison_df["method"], comparison_df["macro_f1"])
ax.set_ylim(0, 1)
ax.set_ylabel("Holdout macro-F1")
ax.set_title("Individual channels versus channel fusion")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
fig.tight_layout()
fig.savefig(RUN_DIR / "holdout_macro_f1_comparison.png", dpi=200)

plt.close(fig)

best_individual_f1 = max(
    metrics["macro_f1"] for metrics in individual_holdout_metrics.values()
)
fusion_gain = fusion_holdout_metrics["macro_f1"] - best_individual_f1
ensemble_gain = ensemble_metrics["macro_f1"] - best_individual_f1

print(f"Best individual-channel macro-F1: {best_individual_f1:.4f}")
print(f"Mean-probability ensemble gain: {ensemble_gain:+.4f}")
print(f"Feature-fusion gain: {fusion_gain:+.4f}")



# ===== NOTEBOOK CODE CELL 54 =====

FINAL_SUMMARY = {
    "completed_at": datetime.now().isoformat(),
    "run_directory": str(RUN_DIR.resolve()),
    "task_mode": TASK_MODE,
    "panel_definition_name": PANEL_DEFINITION_NAME,
    "panel_to_species": PANEL_TO_SPECIES,
    "class_names": CLASS_NAMES,
    "number_of_classes": NUM_CLASSES,
    "selected_channels": SELECTED_CHANNELS,
    "branch_architectures": BRANCH_ARCHITECTURES,
    "resolved_checkpoints": resolved_checkpoints,
    "feature_dimensions": feature_dimensions,
    "fusion_mode": FUSION_MODE,
        "se_reduction_ratio": SE_REDUCTION_RATIO,
    "parameter_counts": count_parameters(fusion_model),
    "best_validation_record": best_epoch_record,
    "individual_holdout_metrics": individual_holdout_metrics,
    "mean_probability_ensemble_holdout_metrics": ensemble_metrics,
    "feature_fusion_holdout_metrics": fusion_holdout_metrics,
    "best_individual_macro_f1": best_individual_f1,
    "ensemble_macro_f1_gain_over_best_individual": ensemble_gain,
    "fusion_macro_f1_gain_over_best_individual": fusion_gain,
    "best_checkpoint": str(BEST_CHECKPOINT_PATH.resolve()),
}

(RUN_DIR / "final_run_summary.json").write_text(
    json.dumps(json_safe(FINAL_SUMMARY), indent=2),
    encoding="utf-8",
)

# Update the config after class names and resolved files are known.
final_config = {
    **USER_CONFIG,
    "class_names": CLASS_NAMES,
    "class_to_index": CLASS_TO_INDEX,
    "resolved_checkpoints": resolved_checkpoints,
    "feature_dimensions": feature_dimensions,
}
(RUN_DIR / "run_config_final.json").write_text(
    json.dumps(json_safe(final_config), indent=2),
    encoding="utf-8",
)

print(json.dumps(json_safe(FINAL_SUMMARY), indent=2))
print("\nAll outputs saved to:")
print(RUN_DIR.resolve())



# Final persistent sync and output inventory.
inventory = []
for file_path in sorted(RUN_DIR.rglob("*")):
    if file_path.is_file():
        inventory.append({
            "relative_path": str(file_path.relative_to(RUN_DIR)),
            "size_bytes": int(file_path.stat().st_size),
            "modified_time": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(timespec="seconds"),
        })
pd.DataFrame(inventory).to_csv(RUN_DIR / "output_file_inventory.csv", index=False)
sync_outputs_to_persistent([RUN_DIR], label="final_full_sync")
print(f"PANEL-2 8-CLASS {FUSION_MODE} FUSION RUN COMPLETE")
print("Scratch outputs:", RUN_DIR)
print("Persistent outputs:", PERSISTENT_RUN_DIR)
