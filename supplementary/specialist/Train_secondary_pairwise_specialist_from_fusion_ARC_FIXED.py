#!/usr/bin/env python3
"""
ARC secondary specialist classifier for an existing feature-fusion model.

EDITING AND RUNNING
-------------------
All settings that you are expected to change are grouped in:

    0. USER-EDITABLE SETTINGS

Paste the full path to the trained fusion checkpoint into FUSION_CHECKPOINT_PATH,
edit any dataset, scratch, output or training values in the same block, and run:

    python Train_secondary_pairwise_specialist_from_fusion_ARC_FIXED.py

No environment variables are required.

The script loads the prepared train, validation and independent Plate 1 holdout
splits, freezes the existing fusion model, trains a separate specialist head for
the classes listed in SPECIALIST_CLASSES, tunes routing on validation data only, and reports final
specialist and full 13-class cascade performance on the untouched holdout set.
The original fusion checkpoint is never overwritten.
"""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import warnings
import zipfile
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

warnings.filterwarnings("once")


# =============================================================================
# 0. USER-EDITABLE SETTINGS
# =============================================================================
# Edit this block only. The rest of the script uses these values directly.

# -----------------------------------------------------------------------------
# A. EXISTING FUSION CHECKPOINT
# -----------------------------------------------------------------------------
# Paste the full path to your trained fusion checkpoint below.
# You may provide either the exact .pt/.pth file or its containing folder.
FUSION_CHECKPOINT_PATH = Path(
    "/data/phys-deep-learning-bacteria/phys3115/Bacteria-data-15 species/Trillian/Single_channel_weights/Fusion/best_fusion_model.pt"
).expanduser()

# -----------------------------------------------------------------------------
# B. DATASET LOCATIONS
# -----------------------------------------------------------------------------
DATASET_FOLDER_NAME = "Trillian_DATASET_with_holdout"

PERSISTENT_DATA_PARENT = Path(
    "/data/phys-deep-learning-bacteria/phys3115/Bacteria-data-15 species/Trillian"
).expanduser()

ZIP_PATH = PERSISTENT_DATA_PARENT / f"{DATASET_FOLDER_NAME}.zip"
PERSISTENT_EXTRACTED_DATA_ROOT = PERSISTENT_DATA_PARENT / DATASET_FOLDER_NAME

# Used only when USE_SCRATCH_DATA is False.
DIRECT_DATA_ROOT = PERSISTENT_EXTRACTED_DATA_ROOT

HOLDOUT_CSV_NAME = "holdout_plate1.csv"
DATASET_ROOT_MARKER = DATASET_FOLDER_NAME

# -----------------------------------------------------------------------------
# C. ARC SCRATCH AND OUTPUT LOCATIONS
# -----------------------------------------------------------------------------
ARC_USERNAME = os.environ.get("USER", "phys3115")
SCRATCH_BASE = Path("/scratch") / ARC_USERNAME
SCRATCH_DATA_PARENT = SCRATCH_BASE / "bacteria_data"
SCRATCH_OUTPUT_PARENT = SCRATCH_BASE / "bacteria_specialist_outputs"

PERSISTENT_OUTPUT_PARENT = (
    PERSISTENT_DATA_PARENT / "Trillian_secondary_specialist_outputs_ARC"
)

USE_SCRATCH_DATA = True
COPY_ZIP_TO_SCRATCH = True
EXTRACT_ZIP_TO_SCRATCH = True
FORCE_REEXTRACT = False
FALLBACK_TO_PERSISTENT_EXTRACTED = True

USE_SCRATCH_OUTPUT = True
SYNC_EVERY_EPOCH = True

# -----------------------------------------------------------------------------
# D. SPECIALIST CLASSES
# -----------------------------------------------------------------------------
# Example pairwise specialist:
SPECIALIST_CLASSES = ["KLAE", "KLPN"]

# The run folder name is generated automatically from the selected classes.
SPECIALIST_TAG = "-".join(SPECIALIST_CLASSES)
RUN_NAME = (
    datetime.now().strftime("run_%Y%m%d_%H%M%S")
    + f"_{SPECIALIST_TAG}_specialist"
)

# -----------------------------------------------------------------------------
# E. FEATURE EXTRACTION AND DEVICE SETTINGS
# -----------------------------------------------------------------------------
SEED = 42
FORCE_CPU = False
USE_AMP = True
NUM_WORKERS = 2
EXTRACTION_BATCH_SIZE = 128
FEATURE_BATCH_SIZE = 4096

# -----------------------------------------------------------------------------
# F. SPECIALIST-HEAD TRAINING SETTINGS
# -----------------------------------------------------------------------------
SPECIALIST_EPOCHS = 60
SPECIALIST_LR = 3e-4
SPECIALIST_WEIGHT_DECAY = 1e-5
SPECIALIST_HIDDEN_DIM = 128
SPECIALIST_DROPOUT = 0.25

EARLY_STOPPING_PATIENCE = 10
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_FACTOR = 0.5

USE_CLASS_WEIGHTED_LOSS = True
USE_WEIGHTED_SAMPLER = False
LABEL_SMOOTHING = 0.0

# -----------------------------------------------------------------------------
# G. CONSERVATIVE VALIDATION-TUNED ROUTING SETTINGS
# -----------------------------------------------------------------------------
# The first specialist run harmed Plate 1 performance because routing was too
# permissive. This version only allows a specialist intervention when both the
# base model and specialist are uncertain/certain in the required directions,
# and it blends probabilities rather than fully replacing the base prediction.
AUTO_TUNE_ROUTING = True
ROUTING_OBJECTIVE = "balanced_accuracy"

# Strict candidate grid. The previous 1.00 margin/top1 values were effectively
# non-restrictive and have deliberately been removed.
ROUTING_GROUP_PROB_GRID = [0.80, 0.90, 0.95]
ROUTING_MARGIN_GRID = [0.05, 0.10, 0.15]
ROUTING_TOP1_CAP_GRID = [0.50, 0.60, 0.70]
SPECIALIST_CONFIDENCE_GRID = [0.70, 0.80, 0.90]
SPECIALIST_MARGIN_GRID = [0.10, 0.20]
SPECIALIST_BLEND_ALPHA_GRID = [0.15, 0.25, 0.35]

# Safety gate: if the cascade does not make a meaningful validation improvement,
# the script automatically selects no routing, so holdout predictions remain
# identical to the original fusion model.
MIN_VALIDATION_OBJECTIVE_GAIN = 0.0010   # 0.10 percentage points
MIN_VALIDATION_ACCURACY_GAIN = 0.0005    # 0.05 percentage points
MIN_VALIDATION_NET_RESCUES = 10
MAX_VALIDATION_DAMAGE_RESCUE_RATIO = 0.75

# Used only when AUTO_TUNE_ROUTING is False.
FIXED_GROUP_PROB_THRESHOLD = 0.90
FIXED_MARGIN_THRESHOLD = 0.10
FIXED_TOP1_CAP = 0.60
FIXED_SPECIALIST_CONFIDENCE = 0.80
FIXED_SPECIALIST_MARGIN = 0.15
FIXED_BLEND_ALPHA = 0.25


# =============================================================================
# 1. SMALL HELPERS
# =============================================================================

def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted(json_safe(v) for v in value)
    return value


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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text))
    return text.strip("_")


def torch_load_checkpoint(path: Path, map_location: Any = "cpu") -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


# =============================================================================
# 2. ARC STORAGE AND DATASET PREPARATION
# =============================================================================
# Derived paths; edit their parent values only in section 0.

SCRATCH_ZIP_PATH = SCRATCH_DATA_PARENT / ZIP_PATH.name
SCRATCH_DATA_ROOT = SCRATCH_DATA_PARENT / DATASET_FOLDER_NAME
RUN_DIR = (
    SCRATCH_OUTPUT_PARENT / RUN_NAME
    if USE_SCRATCH_OUTPUT
    else PERSISTENT_OUTPUT_PARENT / RUN_NAME
)
PERSISTENT_RUN_DIR = PERSISTENT_OUTPUT_PARENT / RUN_NAME

for directory in [SCRATCH_DATA_PARENT, SCRATCH_OUTPUT_PARENT, PERSISTENT_OUTPUT_PARENT, RUN_DIR, PERSISTENT_RUN_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def has_prepared_dataset(root: Path) -> bool:
    root = Path(root)
    return (root / "cell_manifest.csv").exists() and (root / "splits").is_dir()


def safe_copy_file(src: Path, dst: Path) -> Path:
    src, dst = Path(src), Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"Source file does not exist: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        print(f"ZIP already present on scratch with matching size: {dst}")
        return dst
    print(f"Copying ZIP to scratch:\n  from: {src}\n  to:   {dst}")
    shutil.copy2(src, dst)
    print(f"Copied ZIP size: {dst.stat().st_size / 1e9:.2f} GB")
    return dst


def safe_extract_member(zf: zipfile.ZipFile, member: zipfile.ZipInfo, extract_root: Path) -> None:
    extract_root_resolved = Path(extract_root).resolve()
    target = (extract_root_resolved / member.filename).resolve()
    if not str(target).startswith(str(extract_root_resolved)):
        raise RuntimeError(f"Unsafe ZIP member path detected: {member.filename}")
    if member.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member, "r") as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)


def extract_zip(zip_path: Path, extract_root: Path, expected_root: Path, force: bool = False) -> Path:
    zip_path, extract_root, expected_root = map(Path, (zip_path, extract_root, expected_root))
    sentinel = expected_root / ".arc_extract_complete"

    if force and expected_root.exists():
        print(f"FORCE_REEXTRACT=1; deleting previous scratch dataset: {expected_root}")
        shutil.rmtree(expected_root)

    if has_prepared_dataset(expected_root) and sentinel.exists():
        print(f"Scratch dataset already extracted and marked complete: {expected_root}")
        return expected_root

    if has_prepared_dataset(expected_root) and not force:
        print(f"Scratch dataset appears complete; adding sentinel: {expected_root}")
        sentinel.touch()
        return expected_root

    if not zip_path.exists():
        raise FileNotFoundError(f"Cannot extract missing ZIP: {zip_path}")

    print(f"Extracting ZIP:\n  {zip_path}\n  -> {extract_root}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        iterator = tqdm(members, desc="Extracting dataset", unit="file", mininterval=2.0) if tqdm else members
        for i, member in enumerate(iterator, start=1):
            safe_extract_member(zf, member, extract_root)
            if tqdm is None and (i == 1 or i % 1000 == 0 or i == len(members)):
                print(f"Extracted {i:,}/{len(members):,} entries")

    candidates = [expected_root] + [p.parent for p in extract_root.rglob("cell_manifest.csv")]
    for candidate in candidates:
        if has_prepared_dataset(candidate):
            (candidate / ".arc_extract_complete").touch()
            print(f"Resolved dataset root: {candidate}")
            return candidate

    raise FileNotFoundError("Extraction completed, but a prepared dataset root was not found.")


def resolve_dataset_root(root: Path) -> Path:
    current = Path(root)
    for _ in range(10):
        if has_prepared_dataset(current):
            return current
        subdirs = [p for p in current.iterdir() if p.is_dir()] if current.exists() else []
        if len(subdirs) != 1:
            break
        current = subdirs[0]
    return current


if USE_SCRATCH_DATA:
    if COPY_ZIP_TO_SCRATCH and ZIP_PATH.exists():
        safe_copy_file(ZIP_PATH, SCRATCH_ZIP_PATH)
    elif COPY_ZIP_TO_SCRATCH:
        print(f"WARNING: ZIP_PATH does not exist: {ZIP_PATH}")

    if EXTRACT_ZIP_TO_SCRATCH and SCRATCH_ZIP_PATH.exists():
        DATA_ROOT = extract_zip(SCRATCH_ZIP_PATH, SCRATCH_DATA_PARENT, SCRATCH_DATA_ROOT, FORCE_REEXTRACT)
    elif has_prepared_dataset(SCRATCH_DATA_ROOT):
        DATA_ROOT = SCRATCH_DATA_ROOT
    elif FALLBACK_TO_PERSISTENT_EXTRACTED and has_prepared_dataset(PERSISTENT_EXTRACTED_DATA_ROOT):
        print("WARNING: using persistent extracted dataset because scratch data are unavailable.")
        DATA_ROOT = PERSISTENT_EXTRACTED_DATA_ROOT
    else:
        raise FileNotFoundError(
            "Could not prepare DATA_ROOT. Supply ZIP_PATH or PERSISTENT_EXTRACTED_DATA_ROOT.\n"
            f"ZIP_PATH={ZIP_PATH}\nSCRATCH_DATA_ROOT={SCRATCH_DATA_ROOT}\n"
            f"PERSISTENT_EXTRACTED_DATA_ROOT={PERSISTENT_EXTRACTED_DATA_ROOT}"
        )
else:
    DATA_ROOT = Path(DIRECT_DATA_ROOT).expanduser()

DATA_ROOT = resolve_dataset_root(DATA_ROOT)


# Start logging after dataset preparation so large extraction progress still appears in terminal.
_log_handle = open(RUN_DIR / "run_log.txt", "a", buffering=1)
sys.stdout = Tee(sys.__stdout__, _log_handle)
sys.stderr = Tee(sys.__stderr__, _log_handle)


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
    if RUN_DIR.resolve() == PERSISTENT_RUN_DIR.resolve():
        return
    PERSISTENT_RUN_DIR.mkdir(parents=True, exist_ok=True)
    paths = [RUN_DIR] if paths is None else paths
    for src in paths:
        src = Path(src)
        if not src.exists():
            continue
        try:
            relative = src.relative_to(RUN_DIR)
            dst = PERSISTENT_RUN_DIR / relative
        except ValueError:
            dst = PERSISTENT_RUN_DIR / src.name
        sync_path(src, dst)
    (PERSISTENT_RUN_DIR / "last_sync.json").write_text(json.dumps({
        "label": label,
        "time": datetime.now().isoformat(timespec="seconds"),
        "active_run_dir": str(RUN_DIR),
        "persistent_run_dir": str(PERSISTENT_RUN_DIR),
    }, indent=2))
    print(f"Synced outputs [{label}] -> {PERSISTENT_RUN_DIR}")


# =============================================================================
# 3. CONFIGURATION VALIDATION AND CHECKPOINT RESOLUTION
# =============================================================================

DEVICE = torch.device(
    "cpu" if FORCE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")
)

raw_specialist_classes = [str(name).strip().upper() for name in SPECIALIST_CLASSES]
CLASS_ALIASES = {"KLAR": "KLAE"}
SPECIALIST_CLASSES = [CLASS_ALIASES.get(name, name) for name in raw_specialist_classes]
SPECIALIST_CLASSES = list(dict.fromkeys(SPECIALIST_CLASSES))

if len(SPECIALIST_CLASSES) < 2:
    raise ValueError("SPECIALIST_CLASSES must contain at least two classes.")
if "KLAR" in raw_specialist_classes:
    print("Interpreting SPECIALIST_CLASSES entry 'KLAR' as dataset class 'KLAE'.")

FUSION_CHECKPOINT_PATH = Path(FUSION_CHECKPOINT_PATH).expanduser()
if FUSION_CHECKPOINT_PATH.is_dir():
    preferred_names = [
        "best_fusion_model.pt",
        "best_model.pt",
        "latest_fusion_model.pt",
    ]
    resolved_checkpoint = None
    for checkpoint_name in preferred_names:
        candidate = FUSION_CHECKPOINT_PATH / checkpoint_name
        if candidate.exists():
            resolved_checkpoint = candidate
            break

    if resolved_checkpoint is None:
        checkpoint_candidates = sorted(
            list(FUSION_CHECKPOINT_PATH.glob("*.pt"))
            + list(FUSION_CHECKPOINT_PATH.glob("*.pth"))
        )
        if len(checkpoint_candidates) == 1:
            resolved_checkpoint = checkpoint_candidates[0]
        elif len(checkpoint_candidates) > 1:
            raise FileNotFoundError(
                f"Multiple checkpoints were found in {FUSION_CHECKPOINT_PATH}. "
                "Set FUSION_CHECKPOINT_PATH to the exact checkpoint file."
            )

    if resolved_checkpoint is not None:
        FUSION_CHECKPOINT_PATH = resolved_checkpoint

if not FUSION_CHECKPOINT_PATH.exists() or not FUSION_CHECKPOINT_PATH.is_file():
    raise FileNotFoundError(
        "Fusion checkpoint not found. Edit FUSION_CHECKPOINT_PATH in section "
        f"'0. USER-EDITABLE SETTINGS'. Current value: {FUSION_CHECKPOINT_PATH}"
    )

if ROUTING_OBJECTIVE not in {
    "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"
}:
    raise ValueError(
        "ROUTING_OBJECTIVE must be one of: accuracy, balanced_accuracy, "
        "macro_f1, weighted_f1."
    )

seed_everything(SEED)

print("\n" + "=" * 88)
print("SECONDARY CONFUSION SPECIALIST TRAINING")
print("=" * 88)
print("Python:", sys.version.replace("\n", " "))
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("DATA_ROOT:", DATA_ROOT)
print("Fusion checkpoint:", FUSION_CHECKPOINT_PATH)
print("Specialist classes:", SPECIALIST_CLASSES)
print("Active output:", RUN_DIR)
print("Persistent output:", PERSISTENT_RUN_DIR)


# =============================================================================
# 4. LOAD AND REPAIR SPLIT CSVs
# =============================================================================

CELL_MANIFEST_CSV = DATA_ROOT / "cell_manifest.csv"
TRAIN_CSV = DATA_ROOT / "splits" / "train.csv"
VAL_CSV = DATA_ROOT / "splits" / "val.csv"
HOLDOUT_CSV = DATA_ROOT / "splits" / HOLDOUT_CSV_NAME

if not HOLDOUT_CSV.exists():
    candidates = sorted((DATA_ROOT / "splits").glob("holdout*.csv"))
    if candidates:
        print(f"WARNING: {HOLDOUT_CSV_NAME} not found; using {candidates[0].name}")
        HOLDOUT_CSV = candidates[0]

for path in [CELL_MANIFEST_CSV, TRAIN_CSV, VAL_CSV, HOLDOUT_CSV]:
    if not path.exists():
        raise FileNotFoundError(f"Required dataset file is missing: {path}")

cell_manifest_df = pd.read_csv(CELL_MANIFEST_CSV)
train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
holdout_df = pd.read_csv(HOLDOUT_CSV)


def find_first_existing_column(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    for column in candidates:
        if column in df.columns:
            return column
    if required:
        raise ValueError(f"Could not find any of {list(candidates)}. Available: {df.columns.tolist()}")
    return None


SPECIES_COL = find_first_existing_column(train_df, ["species", "label", "class_name", "class", "target"])
CROP_PATH_COL = find_first_existing_column(train_df, ["crop_path", "crop_file", "crop", "image_path", "img_path", "npy_path"])

for df in [cell_manifest_df, train_df, val_df, holdout_df]:
    if SPECIES_COL != "species" and SPECIES_COL in df.columns:
        df["species"] = df[SPECIES_COL]
    if CROP_PATH_COL != "crop_path" and CROP_PATH_COL in df.columns:
        df["crop_path"] = df[CROP_PATH_COL]
    df["species"] = df["species"].astype(str).str.upper()


def rebase_moved_dataset_path(path_value: Any, data_root: Path = DATA_ROOT) -> Any:
    if pd.isna(path_value):
        return path_value
    original = str(path_value)
    normalised = original.replace("\\", "/")
    if normalised.startswith("/") and Path(normalised).exists():
        return normalised

    markers: List[str] = []
    for marker in [DATASET_ROOT_MARKER, DATASET_FOLDER_NAME, DATA_ROOT.name]:
        if marker and marker not in markers:
            markers.append(marker)

    for marker in markers:
        if marker in normalised:
            suffix = normalised.split(marker, 1)[1].lstrip("/")
            return str(Path(data_root) / suffix)

    root_normalised = str(data_root).replace("\\", "/")
    if root_normalised in normalised:
        suffix = normalised.split(root_normalised, 1)[1].lstrip("/")
        return str(Path(data_root) / suffix)

    return original


def split_fallback_path(row: pd.Series, split_name: str) -> str:
    candidate = str(row.get("crop_path", ""))
    if candidate.startswith("/") and Path(candidate).exists():
        return candidate
    basename = Path(candidate.replace("\\", "/")).name
    partition = str(row.get("dataset_partition", "")).strip()
    if not partition or partition.lower() == "nan":
        partition = "holdout_plate1" if "holdout" in split_name.lower() else "train_pool_plates_not_1"
    return str(DATA_ROOT / partition / "crops" / basename)


def repair_df_paths(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    repaired = df.copy()
    repaired["crop_path_original"] = repaired["crop_path"].astype(str)
    repaired["crop_path"] = repaired["crop_path"].map(lambda x: rebase_moved_dataset_path(x, DATA_ROOT))
    exists = repaired["crop_path"].map(lambda x: Path(str(x)).exists())
    if not bool(exists.all()):
        repaired.loc[~exists, "crop_path"] = repaired.loc[~exists].apply(
            lambda row: split_fallback_path(row, split_name), axis=1
        )
    repaired["crop_exists"] = repaired["crop_path"].map(lambda x: Path(str(x)).exists())
    n_ok = int(repaired["crop_exists"].sum())
    print(f"{split_name}: crop paths found = {n_ok:,}/{len(repaired):,} ({repaired['crop_exists'].mean()*100:.2f}%)")
    if n_ok != len(repaired):
        repaired.loc[~repaired["crop_exists"], ["crop_path_original", "crop_path"]].head(50).to_csv(
            RUN_DIR / f"missing_paths_preview_{split_name}.csv", index=False
        )
    return repaired


train_df = repair_df_paths(train_df, "train")
val_df = repair_df_paths(val_df, "val")
holdout_df = repair_df_paths(holdout_df, "holdout")

if not train_df["crop_exists"].all():
    raise FileNotFoundError("Some training crop paths remain unresolved.")
if not val_df["crop_exists"].all():
    raise FileNotFoundError("Some validation crop paths remain unresolved.")
if not holdout_df["crop_exists"].all():
    raise FileNotFoundError("Some holdout crop paths remain unresolved.")

repaired_dir = RUN_DIR / "repaired_split_csvs"
repaired_dir.mkdir(exist_ok=True)
train_df.to_csv(repaired_dir / "train_repaired.csv", index=False)
val_df.to_csv(repaired_dir / "val_repaired.csv", index=False)
holdout_df.to_csv(repaired_dir / "holdout_repaired.csv", index=False)

if "plate" in train_df.columns and "plate1" in HOLDOUT_CSV.name.lower():
    train_plates = pd.to_numeric(train_df["plate"], errors="coerce")
    val_plates = pd.to_numeric(val_df["plate"], errors="coerce")
    holdout_plates = pd.to_numeric(holdout_df["plate"], errors="coerce")
    assert not (train_plates == 1).any(), "Plate 1 leaked into training."
    assert not (val_plates == 1).any(), "Plate 1 leaked into validation."
    assert (holdout_plates == 1).all(), "Holdout contains non-Plate-1 rows."
    print("Plate 1 leakage checks passed.")

split_counts = pd.DataFrame({
    "train": train_df["species"].value_counts(),
    "val": val_df["species"].value_counts(),
    "holdout": holdout_df["species"].value_counts(),
}).fillna(0).astype(int).sort_index()
split_counts.to_csv(RUN_DIR / "species_counts_by_split.csv")
print("\nSpecies counts by split:\n", split_counts.to_string())


# =============================================================================
# 5. RECONSTRUCT AND LOAD THE FUSION MODEL
# =============================================================================

CHANNEL_TO_INDEX_DEFAULT = {
    "PHASECONTRAST": 0,
    "BF": 1,
    "DAPI": 2,
    "WGA": 3,
    "FM1": 4,
}


def clone_conv_with_input_channels(old_conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
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


def build_feature_branch(architecture: str) -> Tuple[nn.Module, int]:
    architecture = architecture.lower()

    if architecture == "densenet121":
        model = models.densenet121(weights=None)
        model.features.conv0 = clone_conv_with_input_channels(model.features.conv0, 1)
        feature_dim = int(model.classifier.in_features)
        model.classifier = nn.Identity()
        return model, feature_dim

    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        model.conv1 = clone_conv_with_input_channels(model.conv1, 1)
        feature_dim = int(model.fc.in_features)
        model.fc = nn.Identity()
        return model, feature_dim

    if architecture == "resnet34":
        model = models.resnet34(weights=None)
        model.conv1 = clone_conv_with_input_channels(model.conv1, 1)
        feature_dim = int(model.fc.in_features)
        model.fc = nn.Identity()
        return model, feature_dim

    if architecture == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        model.features[0][0] = clone_conv_with_input_channels(model.features[0][0], 1)
        feature_dim = int(model.classifier[-1].in_features)
        model.classifier[-1] = nn.Identity()
        return model, feature_dim

    if architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.features[0][0] = clone_conv_with_input_channels(model.features[0][0], 1)
        feature_dim = int(model.classifier[-1].in_features)
        model.classifier[-1] = nn.Identity()
        return model, feature_dim

    raise ValueError(f"Unsupported branch architecture: {architecture}")


class SEFeatureBlock(nn.Module):
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
        return features * feature_weights, feature_weights.mean(dim=1)


class MultiBranchFeatureFusion(nn.Module):
    """Reconstructs Concat, gated-attention, or SE-attention fusion checkpoints."""

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
        self.fusion_mode = str(fusion_mode)
        self.branches = nn.ModuleDict({name: branches[name] for name in self.selected_channels})
        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(int(feature_dimensions[name]), projection_dim),
                nn.LayerNorm(projection_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for name in self.selected_channels
        })

        if self.fusion_mode == "concat":
            self.se_blocks = None
            self.gate_network = None
            fused_dim = projection_dim * len(self.selected_channels)
        elif self.fusion_mode == "se_attention":
            self.se_blocks = nn.ModuleDict({
                name: SEFeatureBlock(projection_dim, se_reduction_ratio)
                for name in self.selected_channels
            })
            self.gate_network = None
            fused_dim = projection_dim * len(self.selected_channels)
        elif self.fusion_mode == "gated_attention":
            self.se_blocks = None
            self.gate_network = nn.Sequential(
                nn.Linear(projection_dim * len(self.selected_channels), hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, len(self.selected_channels)),
            )
            fused_dim = projection_dim
        else:
            raise ValueError(f"Unsupported fusion mode: {self.fusion_mode}")

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def fuse_projected(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        projected_features: List[torch.Tensor] = []
        channel_summaries: List[torch.Tensor] = []

        for position, channel_name in enumerate(self.selected_channels):
            channel_tensor = x[:, position:position + 1]
            branch_features = self.branches[channel_name](channel_tensor)
            projected = self.projections[channel_name](branch_features)

            if self.fusion_mode == "se_attention":
                projected, summary = self.se_blocks[channel_name](projected)
                channel_summaries.append(summary)

            projected_features.append(projected)

        if self.fusion_mode in {"concat", "se_attention"}:
            fused = torch.cat(projected_features, dim=1)
            gates = torch.stack(channel_summaries, dim=1) if channel_summaries else None
            return fused, gates

        concatenated = torch.cat(projected_features, dim=1)
        gate_logits = self.gate_network(concatenated)
        gates = torch.softmax(gate_logits, dim=1)
        stacked = torch.stack(projected_features, dim=1)
        fused = (stacked * gates.unsqueeze(-1)).sum(dim=1)
        return fused, gates

    def forward_with_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        fused, gates = self.fuse_projected(x)
        # The existing classifier is Linear -> LayerNorm -> GELU -> Dropout -> Linear.
        # The specialist uses the hidden vector immediately before the final 13-class layer.
        hidden = self.classifier[:-1](fused)
        logits = self.classifier[-1](hidden)
        return logits, hidden, gates

    def forward(self, x: torch.Tensor, return_gates: bool = False) -> Any:
        logits, _, gates = self.forward_with_features(x)
        if return_gates:
            return logits, gates
        return logits


def derive_class_names(checkpoint: Mapping[str, Any]) -> List[str]:
    if checkpoint.get("class_names") is not None:
        return [str(x).upper() for x in checkpoint["class_names"]]
    mapping = checkpoint.get("class_to_idx")
    if mapping is not None:
        ordered = sorted(((int(index), str(name).upper()) for name, index in mapping.items()))
        return [name for _, name in ordered]
    raise KeyError("Fusion checkpoint has neither class_names nor class_to_idx metadata.")


def build_fusion_model_from_checkpoint(checkpoint: Mapping[str, Any]) -> Tuple[MultiBranchFeatureFusion, Dict[str, Any]]:
    required = [
        "model_state_dict", "selected_channels", "branch_architectures",
        "feature_dimensions", "projection_dim", "fusion_hidden_dim", "fusion_dropout",
    ]
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise KeyError(
            "The supplied file does not look like a complete fusion checkpoint. "
            f"Missing metadata keys: {missing}. Use best_fusion_model.pt rather than a raw state_dict."
        )

    class_names = derive_class_names(checkpoint)
    selected_channels = [str(x).upper() for x in checkpoint["selected_channels"]]
    branch_architectures = {str(k).upper(): str(v).lower() for k, v in checkpoint["branch_architectures"].items()}
    feature_dimensions = {str(k).upper(): int(v) for k, v in checkpoint["feature_dimensions"].items()}
    fusion_mode = str(checkpoint.get("fusion_mode", "concat"))

    branches: Dict[str, nn.Module] = {}
    reconstructed_dimensions: Dict[str, int] = {}
    for channel in selected_channels:
        if channel not in branch_architectures:
            raise KeyError(f"Checkpoint is missing branch architecture for {channel}")
        branch, feature_dim = build_feature_branch(branch_architectures[channel])
        branches[channel] = branch
        reconstructed_dimensions[channel] = feature_dim
        expected = feature_dimensions[channel]
        if feature_dim != expected:
            raise ValueError(
                f"{channel}: reconstructed feature dimension {feature_dim} != checkpoint {expected}"
            )

    model = MultiBranchFeatureFusion(
        branches=branches,
        feature_dimensions=feature_dimensions,
        selected_channels=selected_channels,
        num_classes=len(class_names),
        projection_dim=int(checkpoint["projection_dim"]),
        hidden_dim=int(checkpoint["fusion_hidden_dim"]),
        dropout=float(checkpoint["fusion_dropout"]),
        fusion_mode=fusion_mode,
        se_reduction_ratio=int(checkpoint.get("se_reduction_ratio", 4)),
    )

    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Strict checkpoint loading failed. Missing={incompatible.missing_keys}; "
            f"unexpected={incompatible.unexpected_keys}"
        )

    metadata = {
        "class_names": class_names,
        "selected_channels": selected_channels,
        "branch_architectures": branch_architectures,
        "feature_dimensions": feature_dimensions,
        "fusion_mode": fusion_mode,
        "projection_dim": int(checkpoint["projection_dim"]),
        "fusion_hidden_dim": int(checkpoint["fusion_hidden_dim"]),
        "fusion_dropout": float(checkpoint["fusion_dropout"]),
        "se_reduction_ratio": int(checkpoint.get("se_reduction_ratio", 4)),
        "normalization_by_channel": {
            str(k).upper(): v
            for k, v in checkpoint.get("normalization_by_channel", {}).items()
        },
        "channel_to_index": {
            str(k).upper(): int(v)
            for k, v in checkpoint.get("channel_to_index", CHANNEL_TO_INDEX_DEFAULT).items()
        },
        "input_size": int(checkpoint.get("input_size", 96)),
        "best_epoch_record": checkpoint.get("best_epoch_record"),
    }
    return model, metadata


fusion_checkpoint = torch_load_checkpoint(FUSION_CHECKPOINT_PATH, map_location="cpu")
fusion_model, fusion_metadata = build_fusion_model_from_checkpoint(fusion_checkpoint)
fusion_model = fusion_model.to(DEVICE)
fusion_model.eval()
for parameter in fusion_model.parameters():
    parameter.requires_grad = False

CLASS_NAMES = fusion_metadata["class_names"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
SELECTED_CHANNELS = fusion_metadata["selected_channels"]
CHANNEL_TO_INDEX = fusion_metadata["channel_to_index"]
NORMALIZATION_BY_CHANNEL = fusion_metadata["normalization_by_channel"]
INPUT_SIZE = fusion_metadata["input_size"]
BASE_NUM_CLASSES = len(CLASS_NAMES)

missing_specialist = [name for name in SPECIALIST_CLASSES if name not in CLASS_TO_INDEX]
if missing_specialist:
    raise ValueError(
        f"Specialist classes absent from checkpoint class order: {missing_specialist}. "
        f"Checkpoint classes: {CLASS_NAMES}"
    )

SPECIALIST_TO_INDEX = {name: i for i, name in enumerate(SPECIALIST_CLASSES)}
SPECIALIST_BASE_INDICES = np.array([CLASS_TO_INDEX[name] for name in SPECIALIST_CLASSES], dtype=np.int64)

# Confirm the dataset contains exactly the same classes needed by the checkpoint.
for split_name, df in [("train", train_df), ("val", val_df), ("holdout", holdout_df)]:
    unseen = sorted(set(df["species"].unique()) - set(CLASS_NAMES))
    if unseen:
        raise ValueError(f"{split_name} contains classes absent from fusion checkpoint: {unseen}")

example_crop = np.load(train_df.iloc[0]["crop_path"])
if example_crop.ndim != 3:
    raise ValueError(f"Expected CHW crop array, found shape {example_crop.shape}")
for channel in SELECTED_CHANNELS:
    if channel not in CHANNEL_TO_INDEX:
        raise KeyError(f"No channel index available for {channel}")
    if CHANNEL_TO_INDEX[channel] >= example_crop.shape[0]:
        raise IndexError(
            f"Crop has {example_crop.shape[0]} channels, but {channel} uses index {CHANNEL_TO_INDEX[channel]}"
        )

print("\nFusion model loaded strictly.")
print("Fusion mode:", fusion_metadata["fusion_mode"])
print("Selected channels:", SELECTED_CHANNELS)
print("Base classes:", CLASS_NAMES)
print("Penultimate hidden dimension:", fusion_metadata["fusion_hidden_dim"])

(RUN_DIR / "fusion_checkpoint_metadata.json").write_text(
    json.dumps(json_safe({
        **fusion_metadata,
        "fusion_checkpoint_path": str(FUSION_CHECKPOINT_PATH),
    }), indent=2)
)


# =============================================================================
# 6. DATASET AND CHECKPOINT-MATCHED NORMALISATION
# =============================================================================


def scalar_from_config(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    return float(array[0]) if array.size else float(default)


def normalise_one_channel(image: np.ndarray, config: Any) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    if isinstance(config, str):
        mode = config
        cfg: Mapping[str, Any] = {}
    elif isinstance(config, Mapping):
        cfg = config
        mode = str(cfg.get("mode", cfg.get("normalization_mode", "per_crop_zscore")))
    else:
        cfg = {}
        mode = "per_crop_zscore"

    mode = mode.strip().lower()
    eps = scalar_from_config(cfg.get("eps", cfg.get("epsilon", 1e-6)), 1e-6)

    if mode in {"per_crop_zscore", "per_image_zscore", "zscore"}:
        return (image - float(image.mean())) / (float(image.std()) + eps)

    if mode in {"train_global_zscore", "global_zscore"}:
        mean = scalar_from_config(cfg.get("mean", cfg.get("train_mean", 0.0)), 0.0)
        std = scalar_from_config(cfg.get("std", cfg.get("train_std", 1.0)), 1.0)
        return (image - mean) / (std + eps)

    if mode in {"percentile_minmax", "per_crop_percentile", "percentile"}:
        low = scalar_from_config(cfg.get("percentile_low", cfg.get("low", 1.0)), 1.0)
        high = scalar_from_config(cfg.get("percentile_high", cfg.get("high", 99.0)), 99.0)
        lo, hi = np.percentile(image, [low, high])
        if hi <= lo:
            hi = lo + 1.0
        return np.clip((image - lo) / (hi - lo), 0.0, 1.0)

    if mode in {"none", "identity", "raw"}:
        return image

    raise ValueError(f"Unsupported checkpoint normalisation mode: {mode}")


class FusionCropDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True).copy()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        full_crop = np.load(row["crop_path"]).astype(np.float32)
        channels: List[np.ndarray] = []
        for channel in SELECTED_CHANNELS:
            image = full_crop[CHANNEL_TO_INDEX[channel]]
            image = normalise_one_channel(image, NORMALIZATION_BY_CHANNEL.get(channel, {"mode": "per_crop_zscore"}))
            channels.append(image)
        crop = np.stack(channels, axis=0).astype(np.float32)
        if crop.shape[-2:] != (INPUT_SIZE, INPUT_SIZE):
            tensor = torch.from_numpy(crop).unsqueeze(0)
            tensor = F.interpolate(tensor, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)
            crop = tensor.squeeze(0).numpy()
        x = torch.from_numpy(crop).float()
        y = torch.tensor(CLASS_TO_INDEX[str(row["species"])], dtype=torch.long)
        return x, y, str(row["crop_path"])


def make_image_loader(df: pd.DataFrame, batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        FusionCropDataset(df),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )


train_specialist_df = train_df[train_df["species"].isin(SPECIALIST_CLASSES)].copy()
val_specialist_df = val_df[val_df["species"].isin(SPECIALIST_CLASSES)].copy()
holdout_specialist_df = holdout_df[holdout_df["species"].isin(SPECIALIST_CLASSES)].copy()

for name, df in [
    ("specialist train", train_specialist_df),
    ("specialist val", val_specialist_df),
    ("specialist holdout", holdout_specialist_df),
]:
    if len(df) == 0:
        raise ValueError(f"{name} is empty")

specialist_counts = pd.DataFrame({
    "train": train_specialist_df["species"].value_counts(),
    "val": val_specialist_df["species"].value_counts(),
    "holdout": holdout_specialist_df["species"].value_counts(),
}).reindex(SPECIALIST_CLASSES).fillna(0).astype(int)
specialist_counts.to_csv(RUN_DIR / "specialist_class_counts.csv")
print("\nSpecialist class counts:\n", specialist_counts.to_string())


# =============================================================================
# 7. EXTRACT FROZEN FUSION FEATURES AND BASE PROBABILITIES
# =============================================================================


def autocast_context():
    if DEVICE.type == "cuda" and USE_AMP:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


@torch.inference_mode()
def extract_fusion_bundle(loader: DataLoader, description: str) -> Dict[str, Any]:
    fusion_model.eval()
    all_hidden: List[np.ndarray] = []
    all_probabilities: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_paths: List[str] = []

    iterator = tqdm(loader, desc=description, dynamic_ncols=True) if tqdm else loader
    for batch_index, (x, y, paths) in enumerate(iterator, start=1):
        x = x.to(DEVICE, non_blocking=True)
        with autocast_context():
            logits, hidden, _ = fusion_model.forward_with_features(x)
            probabilities = torch.softmax(logits, dim=1)
        all_hidden.append(hidden.detach().float().cpu().numpy())
        all_probabilities.append(probabilities.detach().float().cpu().numpy())
        all_targets.append(y.numpy())
        all_paths.extend(list(paths))
        if tqdm is None and batch_index % 100 == 0:
            print(f"{description}: processed {batch_index} batches")

    return {
        "hidden": np.concatenate(all_hidden, axis=0).astype(np.float32, copy=False),
        "base_probabilities": np.concatenate(all_probabilities, axis=0).astype(np.float32, copy=False),
        "targets": np.concatenate(all_targets, axis=0).astype(np.int64, copy=False),
        "paths": all_paths,
    }


print("\nExtracting frozen fusion features. This is the CNN-heavy part of the run.")
train_bundle = extract_fusion_bundle(
    make_image_loader(train_specialist_df, EXTRACTION_BATCH_SIZE),
    "Extract specialist train features",
)
val_bundle = extract_fusion_bundle(
    make_image_loader(val_df, EXTRACTION_BATCH_SIZE),
    "Extract full validation features",
)
holdout_bundle = extract_fusion_bundle(
    make_image_loader(holdout_df, EXTRACTION_BATCH_SIZE),
    "Extract full holdout features",
)

print("Feature shapes:")
print("  train specialist:", train_bundle["hidden"].shape)
print("  validation full:  ", val_bundle["hidden"].shape)
print("  holdout full:     ", holdout_bundle["hidden"].shape)


# =============================================================================
# 8. SPECIALIST HEAD TRAINING
# =============================================================================


def full_targets_to_specialist(targets: np.ndarray) -> np.ndarray:
    lookup = {CLASS_TO_INDEX[name]: SPECIALIST_TO_INDEX[name] for name in SPECIALIST_CLASSES}
    output = np.full(len(targets), -1, dtype=np.int64)
    for base_index, specialist_index in lookup.items():
        output[targets == base_index] = specialist_index
    return output


class FeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.features[index]).float(),
            torch.tensor(int(self.labels[index]), dtype=torch.long),
        )


class SpecialistHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


train_specialist_targets = full_targets_to_specialist(train_bundle["targets"])
if np.any(train_specialist_targets < 0):
    raise RuntimeError("Non-specialist labels entered specialist training bundle.")

val_specialist_mask = np.isin(val_bundle["targets"], SPECIALIST_BASE_INDICES)
holdout_specialist_mask = np.isin(holdout_bundle["targets"], SPECIALIST_BASE_INDICES)
val_specialist_targets = full_targets_to_specialist(val_bundle["targets"][val_specialist_mask])
holdout_specialist_targets = full_targets_to_specialist(holdout_bundle["targets"][holdout_specialist_mask])

train_feature_ds = FeatureDataset(train_bundle["hidden"], train_specialist_targets)
val_feature_ds = FeatureDataset(val_bundle["hidden"][val_specialist_mask], val_specialist_targets)
holdout_feature_ds = FeatureDataset(holdout_bundle["hidden"][holdout_specialist_mask], holdout_specialist_targets)

sampler = None
if USE_WEIGHTED_SAMPLER:
    class_counts_array = np.bincount(train_specialist_targets, minlength=len(SPECIALIST_CLASSES))
    sample_weights = np.array([1.0 / max(class_counts_array[y], 1) for y in train_specialist_targets], dtype=np.float32)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_feature_loader = DataLoader(
    train_feature_ds,
    batch_size=FEATURE_BATCH_SIZE,
    shuffle=sampler is None,
    sampler=sampler,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)
val_feature_loader = DataLoader(
    val_feature_ds,
    batch_size=FEATURE_BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)
holdout_feature_loader = DataLoader(
    holdout_feature_ds,
    batch_size=FEATURE_BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

specialist_model = SpecialistHead(
    input_dim=train_bundle["hidden"].shape[1],
    hidden_dim=SPECIALIST_HIDDEN_DIM,
    num_classes=len(SPECIALIST_CLASSES),
    dropout=SPECIALIST_DROPOUT,
).to(DEVICE)

train_counts_array = np.bincount(train_specialist_targets, minlength=len(SPECIALIST_CLASSES)).astype(np.float32)
class_weights = train_counts_array.sum() / (len(SPECIALIST_CLASSES) * np.maximum(train_counts_array, 1))
class_weights = class_weights / class_weights.mean()
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights_tensor if USE_CLASS_WEIGHTED_LOSS else None,
    label_smoothing=LABEL_SMOOTHING,
)
optimizer = torch.optim.AdamW(
    specialist_model.parameters(), lr=SPECIALIST_LR, weight_decay=SPECIALIST_WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE
)
scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and DEVICE.type == "cuda"))


def feature_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_n = 0
    targets: List[np.ndarray] = []
    predictions: List[np.ndarray] = []

    for features, labels in loader:
        features = features.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with autocast_context():
                logits = model(features)
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        n = labels.size(0)
        total_loss += float(loss.detach().cpu()) * n
        total_n += n
        targets.append(labels.detach().cpu().numpy())
        predictions.append(logits.argmax(dim=1).detach().cpu().numpy())

    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    return {
        "loss": total_loss / max(total_n, 1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


BEST_SPECIALIST_PATH = RUN_DIR / "best_specialist_head.pt"
LATEST_SPECIALIST_PATH = RUN_DIR / "latest_specialist_head.pt"
history: List[Dict[str, Any]] = []
best_val_macro_f1 = -float("inf")
best_epoch = -1
epochs_without_improvement = 0

print("\nTraining specialist head. Fusion model remains frozen and is not modified.")
print("Specialist parameter count:", sum(p.numel() for p in specialist_model.parameters()))
print("Class weights:", {name: float(class_weights[i]) for i, name in enumerate(SPECIALIST_CLASSES)})

for epoch in range(1, SPECIALIST_EPOCHS + 1):
    start = time.time()
    train_metrics = feature_epoch(specialist_model, train_feature_loader, optimizer)
    val_metrics = feature_epoch(specialist_model, val_feature_loader, None)
    scheduler.step(val_metrics["macro_f1"])

    record = {
        "epoch": epoch,
        "elapsed_seconds": time.time() - start,
        "lr": optimizer.param_groups[0]["lr"],
        **{f"train_{k}": v for k, v in train_metrics.items()},
        **{f"val_{k}": v for k, v in val_metrics.items()},
    }
    history.append(record)
    pd.DataFrame(history).to_csv(RUN_DIR / "specialist_training_history.csv", index=False)

    checkpoint_payload = {
        "epoch": epoch,
        "specialist_state_dict": specialist_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "specialist_classes": SPECIALIST_CLASSES,
        "specialist_to_idx": SPECIALIST_TO_INDEX,
        "base_class_names": CLASS_NAMES,
        "base_class_to_idx": CLASS_TO_INDEX,
        "base_fusion_checkpoint_path": str(FUSION_CHECKPOINT_PATH),
        "base_fusion_mode": fusion_metadata["fusion_mode"],
        "base_selected_channels": SELECTED_CHANNELS,
        "feature_source": "penultimate fusion classifier hidden vector",
        "input_feature_dim": int(train_bundle["hidden"].shape[1]),
        "specialist_hidden_dim": SPECIALIST_HIDDEN_DIM,
        "specialist_dropout": SPECIALIST_DROPOUT,
        "validation_metrics": val_metrics,
        "config": {
            "lr": SPECIALIST_LR,
            "weight_decay": SPECIALIST_WEIGHT_DECAY,
            "use_class_weighted_loss": USE_CLASS_WEIGHTED_LOSS,
            "use_weighted_sampler": USE_WEIGHTED_SAMPLER,
            "label_smoothing": LABEL_SMOOTHING,
        },
    }
    torch.save(checkpoint_payload, LATEST_SPECIALIST_PATH)

    improved = val_metrics["macro_f1"] > best_val_macro_f1
    if improved:
        best_val_macro_f1 = val_metrics["macro_f1"]
        best_epoch = epoch
        epochs_without_improvement = 0
        checkpoint_payload["best_val_macro_f1"] = best_val_macro_f1
        torch.save(checkpoint_payload, BEST_SPECIALIST_PATH)
    else:
        epochs_without_improvement += 1

    print(
        f"Epoch {epoch:03d}/{SPECIALIST_EPOCHS} | "
        f"train loss={train_metrics['loss']:.4f} f1={train_metrics['macro_f1']:.4f} | "
        f"val loss={val_metrics['loss']:.4f} f1={val_metrics['macro_f1']:.4f} "
        f"bal_acc={val_metrics['balanced_accuracy']:.4f} | "
        f"lr={optimizer.param_groups[0]['lr']:.2e} | {record['elapsed_seconds']:.1f}s"
        + (" | NEW BEST" if improved else "")
    )

    if SYNC_EVERY_EPOCH:
        sync_outputs_to_persistent([
            RUN_DIR / "specialist_training_history.csv",
            BEST_SPECIALIST_PATH,
            LATEST_SPECIALIST_PATH,
            RUN_DIR / "run_log.txt",
        ], label=f"specialist_epoch_{epoch:03d}")

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print(f"Early stopping at epoch {epoch} after {EARLY_STOPPING_PATIENCE} unimproved epochs.")
        break

print("Best specialist epoch:", best_epoch)
print("Best specialist validation macro-F1:", best_val_macro_f1)

best_specialist_checkpoint = torch_load_checkpoint(BEST_SPECIALIST_PATH, map_location=DEVICE)
specialist_model.load_state_dict(best_specialist_checkpoint["specialist_state_dict"], strict=True)
specialist_model.eval()

history_df = pd.DataFrame(history)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(history_df["epoch"], history_df["train_macro_f1"], label="Train macro F1")
ax.plot(history_df["epoch"], history_df["val_macro_f1"], label="Validation macro F1")
ax.set_xlabel("Epoch")
ax.set_ylabel("Macro F1")
ax.set_title("Secondary specialist training")
ax.legend()
fig.tight_layout()
fig.savefig(RUN_DIR / "specialist_training_macro_f1.png", dpi=220)
plt.close(fig)


# =============================================================================
# 9. PREDICTION, REPORTING AND ROUTING
# =============================================================================


@torch.inference_mode()
def predict_specialist_probabilities(features: np.ndarray) -> np.ndarray:
    dataset = FeatureDataset(features, np.zeros(len(features), dtype=np.int64))
    loader = DataLoader(dataset, batch_size=FEATURE_BATCH_SIZE, shuffle=False, num_workers=0)
    probabilities: List[np.ndarray] = []
    specialist_model.eval()
    for batch_features, _ in loader:
        batch_features = batch_features.to(DEVICE, non_blocking=True)
        with autocast_context():
            logits = specialist_model(batch_features)
            probs = torch.softmax(logits, dim=1)
        probabilities.append(probs.detach().float().cpu().numpy())
    return np.concatenate(probabilities, axis=0).astype(np.float32, copy=False)


val_specialist_probabilities_all = predict_specialist_probabilities(val_bundle["hidden"])
holdout_specialist_probabilities_all = predict_specialist_probabilities(holdout_bundle["hidden"])


def calculate_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    """Calculate metrics for binary or multiclass probability arrays.

    sklearn's top_k_accuracy_score treats binary 2-column scores differently
    across versions and can raise an exception. Computing top-k membership
    directly is version-independent. For a 2-class specialist, top-2 accuracy
    is necessarily 1.0 and is retained only for output-format consistency.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities)

    if probabilities.ndim != 2:
        raise ValueError(
            f"Expected probabilities with shape (N, C), got {probabilities.shape}."
        )
    if len(y_true) != len(probabilities):
        raise ValueError(
            f"y_true length ({len(y_true)}) does not match probability rows "
            f"({len(probabilities)})."
        )
    if probabilities.shape[1] < 2:
        raise ValueError(
            f"At least two output classes are required, got {probabilities.shape[1]}."
        )

    y_pred = probabilities.argmax(axis=1)
    output = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    k = min(2, probabilities.shape[1])
    top_k_indices = np.argpartition(probabilities, -k, axis=1)[:, -k:]
    output["top2_accuracy"] = float(
        np.mean(np.any(top_k_indices == y_true[:, None], axis=1))
    )
    return output


def save_classification_bundle(
    prefix: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
    paths: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    class_names = list(class_names)
    labels = np.arange(len(class_names))
    y_pred = probabilities.argmax(axis=1)
    metrics = calculate_metrics(y_true, probabilities)

    (RUN_DIR / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2))
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names,
        zero_division=0, output_dict=True
    )
    pd.DataFrame(report).T.to_csv(RUN_DIR / f"{prefix}_classification_report.csv")

    cm_raw = confusion_matrix(y_true, y_pred, labels=labels)
    cm_pct = confusion_matrix(y_true, y_pred, labels=labels, normalize="true") * 100.0
    pd.DataFrame(cm_raw, index=class_names, columns=class_names).to_csv(
        RUN_DIR / f"{prefix}_confusion_matrix_raw.csv"
    )
    pd.DataFrame(cm_pct, index=class_names, columns=class_names).to_csv(
        RUN_DIR / f"{prefix}_confusion_matrix_row_percent.csv"
    )

    fig, ax = plt.subplots(
        figsize=(max(7, len(class_names) * 0.65), max(6, len(class_names) * 0.60))
    )
    image = ax.imshow(cm_pct)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Percentage (%)")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{prefix}: row-normalised confusion matrix")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm_pct[i, j] >= 1:
                ax.text(j, i, f"{cm_pct[i, j]:.0f}", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(RUN_DIR / f"{prefix}_confusion_matrix_row_percent.png", dpi=220)
    plt.close(fig)

    prediction_df = pd.DataFrame({
        "true_index": y_true,
        "true_class": [class_names[i] for i in y_true],
        "predicted_index": y_pred,
        "predicted_class": [class_names[i] for i in y_pred],
        "confidence": probabilities.max(axis=1),
    })
    if paths is not None:
        prediction_df.insert(0, "path", list(paths))
    for index, class_name in enumerate(class_names):
        prediction_df[f"prob_{class_name}"] = probabilities[:, index]
    prediction_df.to_csv(RUN_DIR / f"{prefix}_predictions.csv", index=False)
    return metrics


# Original base performance on full validation and holdout.
base_val_metrics = save_classification_bundle(
    "base_fusion_validation_full",
    val_bundle["targets"], val_bundle["base_probabilities"], CLASS_NAMES, val_bundle["paths"]
)
base_holdout_metrics = save_classification_bundle(
    "base_fusion_holdout_full",
    holdout_bundle["targets"], holdout_bundle["base_probabilities"], CLASS_NAMES, holdout_bundle["paths"]
)

# Specialist-only performance on the four relevant classes.
val_specialist_only_metrics = save_classification_bundle(
    "specialist_validation_subset",
    val_specialist_targets,
    val_specialist_probabilities_all[val_specialist_mask],
    SPECIALIST_CLASSES,
    np.asarray(val_bundle["paths"], dtype=object)[val_specialist_mask].tolist(),
)
holdout_specialist_only_metrics = save_classification_bundle(
    "specialist_holdout_subset",
    holdout_specialist_targets,
    holdout_specialist_probabilities_all[holdout_specialist_mask],
    SPECIALIST_CLASSES,
    np.asarray(holdout_bundle["paths"], dtype=object)[holdout_specialist_mask].tolist(),
)


def combine_base_and_specialist(
    base_probabilities: np.ndarray,
    specialist_probabilities: np.ndarray,
    route_mask: np.ndarray,
    blend_alpha: float,
) -> np.ndarray:
    """Conservatively blend specialist and base probabilities inside the group.

    blend_alpha=0 keeps the original fusion probabilities.
    blend_alpha=1 gives the specialist full control inside the specialist group.
    """
    combined = base_probabilities.copy()
    if not np.any(route_mask) or blend_alpha <= 0:
        return combined

    group_mass = base_probabilities[:, SPECIALIST_BASE_INDICES].sum(axis=1)
    base_group = base_probabilities[:, SPECIALIST_BASE_INDICES]
    base_group_conditional = base_group / np.maximum(group_mass[:, None], 1e-12)

    blended_conditional = (
        (1.0 - blend_alpha) * base_group_conditional
        + blend_alpha * specialist_probabilities
    )
    blended_conditional /= np.maximum(
        blended_conditional.sum(axis=1, keepdims=True), 1e-12
    )
    blended_group = blended_conditional * group_mass[:, None]

    combined[np.ix_(route_mask, SPECIALIST_BASE_INDICES)] = blended_group[route_mask]
    combined /= np.maximum(combined.sum(axis=1, keepdims=True), 1e-12)
    return combined


def routing_mask(
    base_probabilities: np.ndarray,
    specialist_probabilities: np.ndarray,
    group_prob_threshold: float,
    margin_threshold: float,
    top1_cap: float,
    specialist_confidence_threshold: float,
    specialist_margin_threshold: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    order = np.argsort(-base_probabilities, axis=1)
    top1_idx = order[:, 0]
    top2_idx = order[:, 1]
    rows = np.arange(len(base_probabilities))
    top1_prob = base_probabilities[rows, top1_idx]
    top2_prob = base_probabilities[rows, top2_idx]
    margin = top1_prob - top2_prob
    group_prob = base_probabilities[:, SPECIALIST_BASE_INDICES].sum(axis=1)
    top1_in_group = np.isin(top1_idx, SPECIALIST_BASE_INDICES)
    top2_in_group = np.isin(top2_idx, SPECIALIST_BASE_INDICES)

    specialist_order = np.argsort(-specialist_probabilities, axis=1)
    specialist_top1_local = specialist_order[:, 0]
    specialist_top2_local = specialist_order[:, 1]
    specialist_top1_prob = specialist_probabilities[rows, specialist_top1_local]
    specialist_top2_prob = specialist_probabilities[rows, specialist_top2_local]
    specialist_margin = specialist_top1_prob - specialist_top2_prob

    mask = (
        top1_in_group
        & top2_in_group
        & (group_prob >= group_prob_threshold)
        & (margin <= margin_threshold)
        & (top1_prob <= top1_cap)
        & (specialist_top1_prob >= specialist_confidence_threshold)
        & (specialist_margin >= specialist_margin_threshold)
    )
    diagnostics = {
        "top1_idx": top1_idx,
        "top2_idx": top2_idx,
        "top1_prob": top1_prob,
        "top2_prob": top2_prob,
        "margin": margin,
        "group_prob": group_prob,
        "specialist_top1_local": specialist_top1_local,
        "specialist_top2_local": specialist_top2_local,
        "specialist_top1_prob": specialist_top1_prob,
        "specialist_top2_prob": specialist_top2_prob,
        "specialist_margin": specialist_margin,
    }
    return mask, diagnostics


def routing_result_row(
    y_true: np.ndarray,
    base_probabilities: np.ndarray,
    specialist_probabilities: np.ndarray,
    group_prob_threshold: float,
    margin_threshold: float,
    top1_cap: float,
    specialist_confidence_threshold: float,
    specialist_margin_threshold: float,
    blend_alpha: float,
) -> Dict[str, Any]:
    mask, _ = routing_mask(
        base_probabilities,
        specialist_probabilities,
        group_prob_threshold,
        margin_threshold,
        top1_cap,
        specialist_confidence_threshold,
        specialist_margin_threshold,
    )
    combined = combine_base_and_specialist(
        base_probabilities, specialist_probabilities, mask, blend_alpha
    )
    base_pred = base_probabilities.argmax(axis=1)
    final_pred = combined.argmax(axis=1)
    metrics = calculate_metrics(y_true, combined)
    rescued = int(np.sum((base_pred != y_true) & (final_pred == y_true)))
    damaged = int(np.sum((base_pred == y_true) & (final_pred != y_true)))
    changed = int(np.sum(base_pred != final_pred))
    return {
        "group_prob_threshold": float(group_prob_threshold),
        "margin_threshold": float(margin_threshold),
        "top1_cap": float(top1_cap),
        "specialist_confidence_threshold": float(specialist_confidence_threshold),
        "specialist_margin_threshold": float(specialist_margin_threshold),
        "blend_alpha": float(blend_alpha),
        "n_routed": int(mask.sum()),
        "routed_fraction": float(mask.mean()),
        "n_changed": changed,
        "rescued_errors": rescued,
        "damaged_correct": damaged,
        "net_rescues": rescued - damaged,
        **metrics,
    }


if ROUTING_OBJECTIVE not in {"accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"}:
    raise ValueError(
        "ROUTING_OBJECTIVE must be one of accuracy, balanced_accuracy, macro_f1, weighted_f1"
    )

routing_rows: List[Dict[str, Any]] = []
# Explicit no-routing baseline. It is selected automatically if no candidate
# passes the safety requirements below.
baseline_routing = routing_result_row(
    val_bundle["targets"],
    val_bundle["base_probabilities"],
    val_specialist_probabilities_all,
    2.0, -1.0, -1.0, 2.0, 2.0, 0.0,
)
routing_rows.append(baseline_routing)

if AUTO_TUNE_ROUTING:
    for group_threshold in ROUTING_GROUP_PROB_GRID:
        for margin_threshold in ROUTING_MARGIN_GRID:
            for top1_cap in ROUTING_TOP1_CAP_GRID:
                for specialist_confidence in SPECIALIST_CONFIDENCE_GRID:
                    for specialist_margin in SPECIALIST_MARGIN_GRID:
                        for blend_alpha in SPECIALIST_BLEND_ALPHA_GRID:
                            routing_rows.append(routing_result_row(
                                val_bundle["targets"],
                                val_bundle["base_probabilities"],
                                val_specialist_probabilities_all,
                                group_threshold,
                                margin_threshold,
                                top1_cap,
                                specialist_confidence,
                                specialist_margin,
                                blend_alpha,
                            ))

    routing_df = pd.DataFrame(routing_rows)
    routing_df["objective_gain"] = (
        routing_df[ROUTING_OBJECTIVE] - baseline_routing[ROUTING_OBJECTIVE]
    )
    routing_df["accuracy_gain"] = (
        routing_df["accuracy"] - baseline_routing["accuracy"]
    )
    routing_df["balanced_accuracy_gain"] = (
        routing_df["balanced_accuracy"] - baseline_routing["balanced_accuracy"]
    )
    damage_limit = np.maximum(
        routing_df["rescued_errors"] * MAX_VALIDATION_DAMAGE_RESCUE_RATIO, 0
    )
    routing_df["passes_safety_gate"] = (
        (routing_df["objective_gain"] >= MIN_VALIDATION_OBJECTIVE_GAIN)
        & (routing_df["accuracy_gain"] >= MIN_VALIDATION_ACCURACY_GAIN)
        & (routing_df["net_rescues"] >= MIN_VALIDATION_NET_RESCUES)
        & (routing_df["damaged_correct"] <= damage_limit)
    )

    safe_candidates = routing_df[routing_df["passes_safety_gate"]].copy()
    if len(safe_candidates) == 0:
        print(
            "WARNING: no routing candidate passed the validation safety gate. "
            "The final cascade will use NO ROUTING and will therefore match the "
            "original fusion predictions."
        )
        best_routing = routing_df.iloc[0].to_dict()
    else:
        safe_candidates = safe_candidates.sort_values(
            [ROUTING_OBJECTIVE, "accuracy", "macro_f1", "net_rescues", "n_routed"],
            ascending=[False, False, False, False, True],
        ).reset_index(drop=True)
        best_routing = safe_candidates.iloc[0].to_dict()

    routing_df = routing_df.sort_values(
        ["passes_safety_gate", ROUTING_OBJECTIVE, "accuracy", "macro_f1", "net_rescues"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    routing_df.to_csv(RUN_DIR / "validation_routing_grid_search.csv", index=False)
else:
    best_routing = routing_result_row(
        val_bundle["targets"],
        val_bundle["base_probabilities"],
        val_specialist_probabilities_all,
        FIXED_GROUP_PROB_THRESHOLD,
        FIXED_MARGIN_THRESHOLD,
        FIXED_TOP1_CAP,
        FIXED_SPECIALIST_CONFIDENCE,
        FIXED_SPECIALIST_MARGIN,
        FIXED_BLEND_ALPHA,
    )
    pd.DataFrame([best_routing]).to_csv(
        RUN_DIR / "validation_fixed_routing_result.csv", index=False
    )

best_routing_config = {
    "auto_tuned_on_validation": AUTO_TUNE_ROUTING,
    "objective": ROUTING_OBJECTIVE,
    "group_prob_threshold": float(best_routing["group_prob_threshold"]),
    "margin_threshold": float(best_routing["margin_threshold"]),
    "top1_cap": float(best_routing["top1_cap"]),
    "specialist_confidence_threshold": float(
        best_routing["specialist_confidence_threshold"]
    ),
    "specialist_margin_threshold": float(best_routing["specialist_margin_threshold"]),
    "blend_alpha": float(best_routing["blend_alpha"]),
    "routing_enabled": bool(best_routing["n_routed"] > 0 and best_routing["blend_alpha"] > 0),
    "validation_result": json_safe(best_routing),
}
(RUN_DIR / "best_routing_config.json").write_text(
    json.dumps(best_routing_config, indent=2)
)
print("\nBest validation routing configuration:")
print(json.dumps(best_routing_config, indent=2))


def evaluate_cascade(
    split_name: str,
    y_true: np.ndarray,
    base_probabilities: np.ndarray,
    specialist_probabilities: np.ndarray,
    paths: Sequence[str],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    mask, diagnostics = routing_mask(
        base_probabilities,
        specialist_probabilities,
        best_routing_config["group_prob_threshold"],
        best_routing_config["margin_threshold"],
        best_routing_config["top1_cap"],
        best_routing_config["specialist_confidence_threshold"],
        best_routing_config["specialist_margin_threshold"],
    )
    combined = combine_base_and_specialist(
        base_probabilities,
        specialist_probabilities,
        mask,
        best_routing_config["blend_alpha"],
    )
    base_pred = base_probabilities.argmax(axis=1)
    final_pred = combined.argmax(axis=1)
    specialist_pred_local = specialist_probabilities.argmax(axis=1)
    specialist_pred_base = SPECIALIST_BASE_INDICES[specialist_pred_local]

    metrics = save_classification_bundle(
        f"cascade_{split_name}_full", y_true, combined, CLASS_NAMES, paths
    )
    routing_summary = {
        "split": split_name,
        "n_total": int(len(y_true)),
        "n_routed": int(mask.sum()),
        "routed_fraction": float(mask.mean()),
        "n_predictions_changed": int(np.sum(base_pred != final_pred)),
        "rescued_errors": int(np.sum((base_pred != y_true) & (final_pred == y_true))),
        "damaged_correct": int(np.sum((base_pred == y_true) & (final_pred != y_true))),
        "net_rescues": int(
            np.sum((base_pred != y_true) & (final_pred == y_true))
            - np.sum((base_pred == y_true) & (final_pred != y_true))
        ),
        "base_metrics": calculate_metrics(y_true, base_probabilities),
        "cascade_metrics": metrics,
    }

    prediction_df = pd.DataFrame({
        "path": list(paths),
        "true_class": [CLASS_NAMES[i] for i in y_true],
        "base_predicted_class": [CLASS_NAMES[i] for i in base_pred],
        "base_top1_probability": diagnostics["top1_prob"],
        "base_top2_class": [CLASS_NAMES[i] for i in diagnostics["top2_idx"]],
        "base_top2_probability": diagnostics["top2_prob"],
        "base_top1_top2_margin": diagnostics["margin"],
        "specialist_group_probability": diagnostics["group_prob"],
        "specialist_top1_probability": diagnostics["specialist_top1_prob"],
        "specialist_top2_probability": diagnostics["specialist_top2_prob"],
        "specialist_top1_top2_margin": diagnostics["specialist_margin"],
        "specialist_blend_alpha": best_routing_config["blend_alpha"],
        "routed_to_specialist": mask,
        "specialist_predicted_class": [CLASS_NAMES[i] for i in specialist_pred_base],
        "final_predicted_class": [CLASS_NAMES[i] for i in final_pred],
        "base_correct": base_pred == y_true,
        "final_correct": final_pred == y_true,
        "rescued": (base_pred != y_true) & (final_pred == y_true),
        "damaged": (base_pred == y_true) & (final_pred != y_true),
    })
    prediction_df.to_csv(RUN_DIR / f"cascade_{split_name}_routing_predictions.csv", index=False)
    (RUN_DIR / f"cascade_{split_name}_routing_summary.json").write_text(
        json.dumps(json_safe(routing_summary), indent=2)
    )
    return metrics, routing_summary


cascade_val_metrics, cascade_val_routing = evaluate_cascade(
    "validation", val_bundle["targets"], val_bundle["base_probabilities"],
    val_specialist_probabilities_all, val_bundle["paths"]
)
cascade_holdout_metrics, cascade_holdout_routing = evaluate_cascade(
    "holdout", holdout_bundle["targets"], holdout_bundle["base_probabilities"],
    holdout_specialist_probabilities_all, holdout_bundle["paths"]
)


# =============================================================================
# 10. FINAL SUMMARY AND OUTPUT SYNC
# =============================================================================

comparison_rows = [
    {"split": "validation", "method": "Original fusion", **base_val_metrics},
    {"split": "validation", "method": "Fusion + specialist cascade", **cascade_val_metrics},
    {"split": "holdout", "method": "Original fusion", **base_holdout_metrics},
    {"split": "holdout", "method": "Fusion + specialist cascade", **cascade_holdout_metrics},
    {"split": "validation specialist subset", "method": "Specialist only", **val_specialist_only_metrics},
    {"split": "holdout specialist subset", "method": "Specialist only", **holdout_specialist_only_metrics},
]
comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(RUN_DIR / "final_method_comparison.csv", index=False)
print("\nFinal method comparison:\n", comparison_df.to_string(index=False))

summary = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "data_root": str(DATA_ROOT),
    "holdout_csv": str(HOLDOUT_CSV),
    "fusion_checkpoint_path": str(FUSION_CHECKPOINT_PATH),
    "fusion_mode": fusion_metadata["fusion_mode"],
    "selected_channels": SELECTED_CHANNELS,
    "base_class_names": CLASS_NAMES,
    "specialist_classes": SPECIALIST_CLASSES,
    "specialist_weights_path": str(BEST_SPECIALIST_PATH),
    "specialist_best_epoch": int(best_epoch),
    "specialist_best_val_macro_f1": float(best_val_macro_f1),
    "n_train_specialist": int(len(train_specialist_df)),
    "n_val_full": int(len(val_df)),
    "n_holdout_full": int(len(holdout_df)),
    "n_val_specialist": int(val_specialist_mask.sum()),
    "n_holdout_specialist": int(holdout_specialist_mask.sum()),
    "base_validation_metrics": base_val_metrics,
    "base_holdout_metrics": base_holdout_metrics,
    "specialist_validation_subset_metrics": val_specialist_only_metrics,
    "specialist_holdout_subset_metrics": holdout_specialist_only_metrics,
    "cascade_validation_metrics": cascade_val_metrics,
    "cascade_holdout_metrics": cascade_holdout_metrics,
    "cascade_validation_routing": cascade_val_routing,
    "cascade_holdout_routing": cascade_holdout_routing,
    "routing_config": best_routing_config,
    "output_dir": str(RUN_DIR),
    "persistent_output_dir": str(PERSISTENT_RUN_DIR),
}
(RUN_DIR / "final_summary.json").write_text(json.dumps(json_safe(summary), indent=2))
pd.DataFrame([{
    "fusion_mode": fusion_metadata["fusion_mode"],
    "specialist_classes": "+".join(SPECIALIST_CLASSES),
    "specialist_best_epoch": best_epoch,
    "specialist_holdout_accuracy": holdout_specialist_only_metrics["accuracy"],
    "specialist_holdout_balanced_accuracy": holdout_specialist_only_metrics["balanced_accuracy"],
    "specialist_holdout_macro_f1": holdout_specialist_only_metrics["macro_f1"],
    "base_holdout_accuracy": base_holdout_metrics["accuracy"],
    "base_holdout_balanced_accuracy": base_holdout_metrics["balanced_accuracy"],
    "base_holdout_macro_f1": base_holdout_metrics["macro_f1"],
    "cascade_holdout_accuracy": cascade_holdout_metrics["accuracy"],
    "cascade_holdout_balanced_accuracy": cascade_holdout_metrics["balanced_accuracy"],
    "cascade_holdout_macro_f1": cascade_holdout_metrics["macro_f1"],
    "holdout_routed": cascade_holdout_routing["n_routed"],
    "holdout_rescued_errors": cascade_holdout_routing["rescued_errors"],
    "holdout_damaged_correct": cascade_holdout_routing["damaged_correct"],
    "holdout_net_rescues": cascade_holdout_routing["net_rescues"],
}]).to_csv(RUN_DIR / "final_summary.csv", index=False)

run_config = {
    "seed": SEED,
    "device": str(DEVICE),
    "use_amp": USE_AMP,
    "num_workers": NUM_WORKERS,
    "extraction_batch_size": EXTRACTION_BATCH_SIZE,
    "feature_batch_size": FEATURE_BATCH_SIZE,
    "specialist_epochs": SPECIALIST_EPOCHS,
    "specialist_lr": SPECIALIST_LR,
    "specialist_weight_decay": SPECIALIST_WEIGHT_DECAY,
    "specialist_hidden_dim": SPECIALIST_HIDDEN_DIM,
    "specialist_dropout": SPECIALIST_DROPOUT,
    "use_class_weighted_loss": USE_CLASS_WEIGHTED_LOSS,
    "use_weighted_sampler": USE_WEIGHTED_SAMPLER,
    "label_smoothing": LABEL_SMOOTHING,
    "auto_tune_routing": AUTO_TUNE_ROUTING,
    "routing_objective": ROUTING_OBJECTIVE,
    "routing_group_prob_grid": ROUTING_GROUP_PROB_GRID,
    "routing_margin_grid": ROUTING_MARGIN_GRID,
    "routing_top1_cap_grid": ROUTING_TOP1_CAP_GRID,
    "specialist_confidence_grid": SPECIALIST_CONFIDENCE_GRID,
    "specialist_margin_grid": SPECIALIST_MARGIN_GRID,
    "specialist_blend_alpha_grid": SPECIALIST_BLEND_ALPHA_GRID,
    "min_validation_objective_gain": MIN_VALIDATION_OBJECTIVE_GAIN,
    "min_validation_accuracy_gain": MIN_VALIDATION_ACCURACY_GAIN,
    "min_validation_net_rescues": MIN_VALIDATION_NET_RESCUES,
    "max_validation_damage_rescue_ratio": MAX_VALIDATION_DAMAGE_RESCUE_RATIO,
    "specialist_classes": SPECIALIST_CLASSES,
    "fusion_checkpoint_path": str(FUSION_CHECKPOINT_PATH),
}
(RUN_DIR / "run_config.json").write_text(json.dumps(run_config, indent=2))

# Output inventory.
inventory = []
for path in sorted(RUN_DIR.rglob("*")):
    if path.is_file():
        inventory.append({
            "relative_path": str(path.relative_to(RUN_DIR)),
            "size_bytes": int(path.stat().st_size),
            "modified_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        })
pd.DataFrame(inventory).to_csv(RUN_DIR / "output_file_inventory.csv", index=False)

sync_outputs_to_persistent([RUN_DIR], label="final_full_sync")

print("\n" + "=" * 88)
print("SPECIALIST TRAINING AND HOLDOUT TEST COMPLETE")
print("=" * 88)
print("Original fusion checkpoint was not modified:", FUSION_CHECKPOINT_PATH)
print("Separate specialist weights:", BEST_SPECIALIST_PATH)
print("Original fusion holdout metrics:", json.dumps(base_holdout_metrics, indent=2))
print("Specialist-only holdout metrics:", json.dumps(holdout_specialist_only_metrics, indent=2))
print("Full cascade holdout metrics:", json.dumps(cascade_holdout_metrics, indent=2))
print("Holdout routing summary:", json.dumps(json_safe(cascade_holdout_routing), indent=2))
print("Active output folder:", RUN_DIR)
print("Persistent output folder:", PERSISTENT_RUN_DIR)
