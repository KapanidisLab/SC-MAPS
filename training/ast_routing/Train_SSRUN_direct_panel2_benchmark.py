#!/usr/bin/env python3
"""
Multi-condition ARC GPU training script for Trillian bacteria crops.

Built from Train_SSRUN.py, preserving the original ARC behaviour:
  - optional ZIP copy from persistent /data to ARC /scratch
  - ZIP extraction to scratch with safety checks
  - path repair for moved Windows/Colab/ARC datasets
  - train/validation/holdout loading from prepared split CSVs
  - checkpoint/history/config/report saving
  - per-epoch sync from scratch output to persistent /data
  - optional Optuna search
  - validation and holdout evaluation after training

New behaviour:
  - run multiple channel conditions one after another
  - run multiple model backbones one after another for benchmarking
  - optionally run Optuna once per model, then reuse those hyperparameters
    for all subsequent channel-condition trainings for that model
  - TASK_MODE=panel (default) trains DIRECTLY on the Slide-11 Panel 2 labels
    instead of the original 13 species labels
  - TASK_MODE=species switches back to the original species task for an
    apples-to-apples control using the same code
  - arbitrary numbers of model backbones can be selected with MODEL_BENCHMARKS
  - reports top-2 accuracy in addition to the original metrics

Typical ARC command:

  PERSISTENT_DATA_PARENT="/data/phys-deep-learning-bacteria/phys3115/Bacteria-data-15 species/Trillian" \
  DATASET_FOLDER_NAME="Trillian_DATASET_with_holdout" \
  ZIP_PATH="/data/phys-deep-learning-bacteria/phys3115/Bacteria-data-15 species/Trillian/Trillian_DATASET_with_holdout.zip" \
  RUN_OPTUNA=1 N_TRIALS=10 OPTUNA_EPOCHS=6 \
  TASK_MODE=panel \
  RUN_NAME="panel2_direct_benchmark" \
  MODEL_BENCHMARKS="resnet18,resnet34,densenet121" \
  CHANNEL_CONDITIONS="DAPI;WGA;BF;PHASECONTRAST;FM1" \
  RUN_OPTUNA=1 N_TRIALS=12 OPTUNA_EPOCHS=6 \
  EPOCHS=100 BATCH_SIZE=128 NUM_WORKERS=4 \
  python Train_SSRUN_direct_panel2_benchmark.py
"""

# =========================
# 1. DEPENDENCIES / IMPORTS
# =========================
import os
import sys
import json
import math
import random
import shutil
import time
import zipfile
import subprocess
from datetime import datetime
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from IPython.display import display
except Exception:
    def display(x):
        print(x)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models

import albumentations as A
import optuna

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

print("Python:", sys.version.replace("\n", " "))
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# =========================
# 2. SMALL HELPERS
# =========================
def env_bool(name, default):
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_list(name, default_list, upper=True):
    value = os.environ.get(name, "")
    if not value.strip():
        return default_list
    out = [x.strip() for x in value.split(",") if x.strip()]
    return [x.upper() for x in out] if upper else out


def parse_channel_conditions(value, default_conditions):
    """Parse semicolon-separated channel conditions.

    Example:
      CHANNEL_CONDITIONS="DAPI,FM1;DAPI,WGA,FM1;PHASECONTRAST;DAPI,PHASECONTRAST"
    """
    if not value or not value.strip():
        return default_conditions
    conditions = []
    for block in value.split(";"):
        chans = [x.strip().upper() for x in block.split(",") if x.strip()]
        if chans:
            conditions.append(chans)
    if not conditions:
        raise ValueError("CHANNEL_CONDITIONS was provided but no valid conditions were parsed.")
    return conditions


def safe_name(parts):
    if isinstance(parts, str):
        text = parts
    else:
        text = "_".join(str(p) for p in parts)
    for bad in ["/", "\\", " ", ":", ";", ",", "(", ")", "[", "]"]:
        text = text.replace(bad, "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


# =========================
# 3. ARC PATHS / DATASET PREP
# =========================
USE_GOOGLE_DRIVE = False
print("Google Drive mount disabled. Running on ARC GPU node with scratch + persistent sync.")

DATASET_FOLDER_NAME = os.environ.get("DATASET_FOLDER_NAME", "Trillian_DATASET_with_holdout")

PERSISTENT_DATA_PARENT = Path(os.environ.get(
    "PERSISTENT_DATA_PARENT",
    "/data/phys-deep-learning-bacteria/phys3115/Bacteria-data-15 species/Trillian"
)).expanduser()

ZIP_PATH = Path(os.environ.get(
    "ZIP_PATH",
    str(PERSISTENT_DATA_PARENT / f"{DATASET_FOLDER_NAME}.zip")
)).expanduser()

PERSISTENT_EXTRACTED_DATA_ROOT = Path(os.environ.get(
    "PERSISTENT_EXTRACTED_DATA_ROOT",
    str(PERSISTENT_DATA_PARENT / DATASET_FOLDER_NAME)
)).expanduser()

PERSISTENT_OUTPUT_PARENT = Path(os.environ.get(
    "PERSISTENT_OUTPUT_PARENT",
    str(PERSISTENT_DATA_PARENT / "Trillian_multicondition_training_outputs_ARC")
)).expanduser()


def get_arc_scratch_base() -> Path:
    """Find an ARC scratch base. Avoid /tmp for large datasets."""
    user = os.environ.get("USER", "arc_user")
    scratch_env = os.environ.get("SCRATCH", "").strip()
    if scratch_env:
        return Path(scratch_env).expanduser() / user
    if Path("/scratch").exists():
        return Path("/scratch") / user
    return Path("/data/phys-deep-learning-bacteria") / user / "scratch"


SCRATCH_BASE = Path(os.environ.get("SCRATCH_BASE", str(get_arc_scratch_base()))).expanduser()
SCRATCH_DATA_PARENT = Path(os.environ.get("SCRATCH_DATA_PARENT", str(SCRATCH_BASE / "bacteria_data"))).expanduser()
SCRATCH_OUTPUT_PARENT = Path(os.environ.get("SCRATCH_OUTPUT_PARENT", str(SCRATCH_BASE / "bacteria_training_outputs"))).expanduser()
SCRATCH_ZIP_PATH = SCRATCH_DATA_PARENT / ZIP_PATH.name
SCRATCH_EXTRACT_ROOT = SCRATCH_DATA_PARENT
SCRATCH_DATA_ROOT = SCRATCH_EXTRACT_ROOT / DATASET_FOLDER_NAME

USE_SCRATCH_DATA = env_bool("USE_SCRATCH_DATA", True)
COPY_ZIP_TO_SCRATCH = env_bool("COPY_ZIP_TO_SCRATCH", True)
EXTRACT_ZIP_TO_SCRATCH = env_bool("EXTRACT_ZIP_TO_SCRATCH", True)
FORCE_REEXTRACT = env_bool("FORCE_REEXTRACT", False)
FALLBACK_TO_PERSISTENT_EXTRACTED = env_bool("FALLBACK_TO_PERSISTENT_EXTRACTED", True)
USE_SCRATCH_OUTPUT = env_bool("USE_SCRATCH_OUTPUT", True)
SYNC_EVERY_EPOCH = env_bool("SYNC_EVERY_EPOCH", True)
SAVE_FIXED_CSVS = env_bool("SAVE_FIXED_CSVS", False)

RUN_NAME = os.environ.get("RUN_NAME", datetime.now().strftime("run_%Y%m%d_%H%M%S_multicondition"))
OUTPUT_DIR = (SCRATCH_OUTPUT_PARENT / RUN_NAME) if USE_SCRATCH_OUTPUT else (PERSISTENT_OUTPUT_PARENT / RUN_NAME)
PERSISTENT_OUTPUT_DIR = PERSISTENT_OUTPUT_PARENT / RUN_NAME

for p in [SCRATCH_DATA_PARENT, SCRATCH_OUTPUT_PARENT, OUTPUT_DIR, PERSISTENT_OUTPUT_DIR, PERSISTENT_OUTPUT_PARENT]:
    p.mkdir(parents=True, exist_ok=True)


def has_prepared_dataset(root: Path) -> bool:
    root = Path(root)
    return (root / "cell_manifest.csv").exists() and (root / "splits").is_dir()


def safe_copy_file(src: Path, dst: Path):
    src = Path(src)
    dst = Path(dst)
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


def safe_extract_member(zf: zipfile.ZipFile, member: zipfile.ZipInfo, extract_root: Path):
    extract_root = Path(extract_root).resolve()
    target_path = (extract_root / member.filename).resolve()
    if not str(target_path).startswith(str(extract_root)):
        raise RuntimeError(f"Unsafe ZIP member path detected: {member.filename}")
    if member.is_dir():
        target_path.mkdir(parents=True, exist_ok=True)
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member, "r") as src, open(target_path, "wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    perm = member.external_attr >> 16
    if perm:
        try:
            target_path.chmod(perm)
        except Exception:
            pass


def extract_zip(zip_path: Path, extract_root: Path, dataset_root: Path, force=False):
    zip_path = Path(zip_path)
    extract_root = Path(extract_root)
    dataset_root = Path(dataset_root)
    sentinel = dataset_root / ".arc_extract_complete"

    if force and dataset_root.exists():
        print(f"FORCE_REEXTRACT=1, removing previous scratch dataset: {dataset_root}")
        shutil.rmtree(dataset_root)

    if has_prepared_dataset(dataset_root) and sentinel.exists():
        print(f"Scratch dataset already extracted and marked complete: {dataset_root}")
        return dataset_root

    if has_prepared_dataset(dataset_root) and not force:
        print(f"Scratch dataset already looks prepared, adding completion marker: {dataset_root}")
        sentinel.touch()
        return dataset_root

    if not zip_path.exists():
        raise FileNotFoundError(f"Cannot extract because ZIP does not exist: {zip_path}")

    print(f"Extracting ZIP on scratch with progress:\n  zip:  {zip_path}\n  into: {extract_root}")
    extract_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        total_files = len(members)
        total_gb = sum(m.file_size for m in members) / 1e9
        print(f"ZIP contains {total_files:,} entries; uncompressed size ≈ {total_gb:.2f} GB")
        iterator = tqdm(members, total=total_files, desc="Extracting dataset", unit="file", dynamic_ncols=True, mininterval=2.0) if tqdm else members
        for i, member in enumerate(iterator, start=1):
            safe_extract_member(zf, member, extract_root)
            if tqdm is None and (i == 1 or i % 1000 == 0 or i == total_files):
                print(f"Extracted {i:,}/{total_files:,} ZIP entries")

    if not has_prepared_dataset(dataset_root):
        candidates = [p for p in extract_root.rglob("cell_manifest.csv")]
        if candidates:
            dataset_root = candidates[0].parent
            print(f"Detected dataset root after extraction: {dataset_root}")

    if not has_prepared_dataset(dataset_root):
        raise FileNotFoundError(
            "Extraction finished, but could not find a prepared dataset root containing "
            f"cell_manifest.csv and splits/. Checked: {dataset_root}"
        )

    (dataset_root / ".arc_extract_complete").touch()
    print(f"Scratch extraction complete: {dataset_root}")
    return dataset_root


def resolve_dataset_root(root: Path) -> Path:
    root = Path(root)
    current = root
    for _ in range(10):
        if (current / "cell_manifest.csv").exists() and (current / "splits").exists():
            return current
        subdirs = [p for p in current.iterdir() if p.is_dir()] if current.exists() else []
        if len(subdirs) != 1:
            break
        current = subdirs[0]
    return current


if USE_SCRATCH_DATA:
    if COPY_ZIP_TO_SCRATCH and ZIP_PATH.exists():
        safe_copy_file(ZIP_PATH, SCRATCH_ZIP_PATH)
    elif not ZIP_PATH.exists():
        print(f"WARNING: ZIP_PATH does not exist: {ZIP_PATH}")

    if EXTRACT_ZIP_TO_SCRATCH and SCRATCH_ZIP_PATH.exists():
        DATA_ROOT = extract_zip(SCRATCH_ZIP_PATH, SCRATCH_EXTRACT_ROOT, SCRATCH_DATA_ROOT, force=FORCE_REEXTRACT)
    elif has_prepared_dataset(SCRATCH_DATA_ROOT):
        DATA_ROOT = SCRATCH_DATA_ROOT
    elif FALLBACK_TO_PERSISTENT_EXTRACTED and has_prepared_dataset(PERSISTENT_EXTRACTED_DATA_ROOT):
        print("WARNING: using persistent extracted DATA_ROOT because scratch extraction was unavailable.")
        DATA_ROOT = PERSISTENT_EXTRACTED_DATA_ROOT
    else:
        raise FileNotFoundError(
            "Could not prepare DATA_ROOT. Provide ZIP_PATH or PERSISTENT_EXTRACTED_DATA_ROOT.\n"
            f"ZIP_PATH={ZIP_PATH}\nSCRATCH_ZIP_PATH={SCRATCH_ZIP_PATH}\n"
            f"PERSISTENT_EXTRACTED_DATA_ROOT={PERSISTENT_EXTRACTED_DATA_ROOT}"
        )
else:
    DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(PERSISTENT_EXTRACTED_DATA_ROOT))).expanduser()

DATA_ROOT = resolve_dataset_root(DATA_ROOT)

# Start tee logging after DATA_ROOT prep so the log captures all training details.
_log_file_handle = open(OUTPUT_DIR / "run_log.txt", "a", buffering=1)
sys.stdout = Tee(sys.__stdout__, _log_file_handle)
sys.stderr = Tee(sys.__stderr__, _log_file_handle)

print("\nLogging started:", datetime.now().isoformat(timespec="seconds"))
print("Active outputs will be saved to:", OUTPUT_DIR)
print("Persistent backup location:", PERSISTENT_OUTPUT_DIR)


def sync_path_to_persistent(src: Path, dst: Path):
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        tmp_dst = dst.with_suffix(dst.suffix + ".tmp") if dst.suffix else Path(str(dst) + ".tmp")
        shutil.copy2(src, tmp_dst)
        os.replace(tmp_dst, dst)
    else:
        if shutil.which("rsync"):
            dst.mkdir(parents=True, exist_ok=True)
            subprocess.run(["rsync", "-a", "--delete", f"{src}/", f"{dst}/"], check=False)
        else:
            shutil.copytree(src, dst, dirs_exist_ok=True)


def sync_outputs_to_persistent(paths=None, label="sync"):
    if Path(OUTPUT_DIR).resolve() == Path(PERSISTENT_OUTPUT_DIR).resolve():
        return
    PERSISTENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if paths is None:
        paths = [OUTPUT_DIR]
    for src in paths:
        src = Path(src)
        try:
            rel = src.relative_to(OUTPUT_DIR)
            dst = PERSISTENT_OUTPUT_DIR / rel
        except ValueError:
            dst = PERSISTENT_OUTPUT_DIR / src.name
        sync_path_to_persistent(src, dst)
    stamp = {
        "label": label,
        "time": datetime.now().isoformat(timespec="seconds"),
        "active_output_dir": str(OUTPUT_DIR),
        "persistent_output_dir": str(PERSISTENT_OUTPUT_DIR),
    }
    with open(PERSISTENT_OUTPUT_DIR / "last_sync.json", "w") as f:
        json.dump(stamp, f, indent=2)
    print(f"Synced outputs to persistent /data backup [{label}]: {PERSISTENT_OUTPUT_DIR}")


# =========================
# 4. CONFIG
# =========================
SEED = int(os.environ.get("SEED", 42))
IMG_SIZE = int(os.environ.get("IMG_SIZE", 96))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 128))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 2))
EPOCHS = int(os.environ.get("EPOCHS", 100))
LR = float(os.environ.get("LR", 0.00012791028542324115))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 5.164354962656478e-06))
USE_PRETRAINED = env_bool("USE_PRETRAINED", False)
USE_WEIGHTED_LOSS = env_bool("USE_WEIGHTED_LOSS", False)
USE_WEIGHTED_SAMPLER = env_bool("USE_WEIGHTED_SAMPLER", False)
LABEL_SMOOTHING = float(os.environ.get("LABEL_SMOOTHING", 0.0))
EARLY_STOPPING_PATIENCE = int(os.environ.get("EARLY_STOPPING_PATIENCE", 10))
USE_AMP = env_bool("USE_AMP", True)
USE_AUGMENTATION = env_bool("USE_AUGMENTATION", True)
USE_MASKED_CROPS = env_bool("USE_MASKED_CROPS", False)
NORMALIZATION_MODE = os.environ.get("NORMALIZATION_MODE", "per_crop_zscore")
PERCENTILE_LOW = float(os.environ.get("PERCENTILE_LOW", 1))
PERCENTILE_HIGH = float(os.environ.get("PERCENTILE_HIGH", 99))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHANNEL_MAP = {
    "PHASECONTRAST": 0,
    "BF": 1,
    "DAPI": 2,
    "WGA": 3,
    "FM1": 4,
}
CROP_CHANNEL_ORDER = ["PHASECONTRAST", "BF", "DAPI", "WGA", "FM1"]

DEFAULT_MODEL_BENCHMARKS = ["densenet121","resnet18", "resnet34"]

MODEL_BENCHMARKS = env_list(
    "MODEL_BENCHMARKS",
    DEFAULT_MODEL_BENCHMARKS,
    upper=False
)

MODEL_BENCHMARKS = [m.strip() for m in MODEL_BENCHMARKS if m.strip()]
SUPPORTED_MODELS = {"densenet121","resnet18", "resnet34"}
unknown_models = [m for m in MODEL_BENCHMARKS if m not in SUPPORTED_MODELS]
if unknown_models:
    raise ValueError(f"Unknown MODEL_BENCHMARKS entries: {unknown_models}. Supported: {sorted(SUPPORTED_MODELS)}")

DEFAULT_CHANNEL_CONDITIONS = [
    ["DAPI"],
    ["WGA"],
    ["BF"],
    ["PHASECONTRAST"],
    ["FM1"],
]
CHANNEL_CONDITIONS = parse_channel_conditions(os.environ.get("CHANNEL_CONDITIONS", ""), DEFAULT_CHANNEL_CONDITIONS)
OPTUNA_CHANNEL_NAMES = env_list("OPTUNA_CHANNEL_NAMES", ["DAPI"], upper=True)

RUN_OPTUNA = env_bool("RUN_OPTUNA", env_bool("USE_OPTUNA", True))
OPTUNA_ONCE_PER_MODEL = env_bool("OPTUNA_ONCE_PER_MODEL", True)
REUSE_EXISTING_OPTUNA = env_bool("REUSE_EXISTING_OPTUNA", False)
N_TRIALS = int(os.environ.get("N_TRIALS", 12))
OPTUNA_EPOCHS = int(os.environ.get("OPTUNA_EPOCHS", 6))
_timeout_env = os.environ.get("OPTUNA_TIMEOUT", "").strip()
OPTUNA_TIMEOUT = int(_timeout_env) if _timeout_env else None
OPTUNA_METRIC = "macro_f1"
OPTUNA_BATCH_SIZES = [int(x) for x in env_list("OPTUNA_BATCH_SIZES", ["32", "64", "128"], upper=False)]

HOLDOUT_CSV_NAME = os.environ.get("HOLDOUT_CSV_NAME", "holdout_plate1.csv")
DATASET_ROOT_MARKER = os.environ.get("DATASET_ROOT_MARKER", DATASET_FOLDER_NAME)

SPECIES_INCLUDE = env_list("SPECIES_INCLUDE", [], upper=True)
SPECIES_EXCLUDE = env_list("SPECIES_EXCLUDE", [], upper=True)

# -------------------------------------------------------------------------
# Classification target
# -------------------------------------------------------------------------
# TASK_MODE=panel   -> direct 8-class training using Slide 11 / Panel 2
# TASK_MODE=species -> original 13-species training for a matched control
TASK_MODE ="panel"
if TASK_MODE not in {"panel", "species"}:
    raise ValueError("TASK_MODE must be either 'panel' or 'species'.")

TARGET_COL = "target_label"
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
SPECIES_TO_PANEL = {species: panel for panel, species_list in PANEL_TO_SPECIES.items() for species in species_list}

BASE_CONFIG = {
    "SEED": SEED,
    "IMG_SIZE": IMG_SIZE,
    "BATCH_SIZE_DEFAULT": BATCH_SIZE,
    "NUM_WORKERS": NUM_WORKERS,
    "EPOCHS": EPOCHS,
    "LR_DEFAULT": LR,
    "WEIGHT_DECAY_DEFAULT": WEIGHT_DECAY,
    "USE_PRETRAINED": USE_PRETRAINED,
    "USE_WEIGHTED_LOSS_DEFAULT": USE_WEIGHTED_LOSS,
    "USE_WEIGHTED_SAMPLER": USE_WEIGHTED_SAMPLER,
    "LABEL_SMOOTHING": LABEL_SMOOTHING,
    "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
    "USE_AMP": USE_AMP,
    "USE_AUGMENTATION": USE_AUGMENTATION,
    "USE_MASKED_CROPS": USE_MASKED_CROPS,
    "NORMALIZATION_MODE": NORMALIZATION_MODE,
    "PERCENTILE_LOW": PERCENTILE_LOW,
    "PERCENTILE_HIGH": PERCENTILE_HIGH,
    "DEVICE": DEVICE,
    "MODEL_BENCHMARKS": MODEL_BENCHMARKS,
    "CHANNEL_CONDITIONS": CHANNEL_CONDITIONS,
    "OPTUNA_CHANNEL_NAMES": OPTUNA_CHANNEL_NAMES,
    "RUN_OPTUNA": RUN_OPTUNA,
    "OPTUNA_ONCE_PER_MODEL": OPTUNA_ONCE_PER_MODEL,
    "N_TRIALS": N_TRIALS,
    "OPTUNA_EPOCHS": OPTUNA_EPOCHS,
    "OPTUNA_TIMEOUT": OPTUNA_TIMEOUT,
    "OPTUNA_BATCH_SIZES": OPTUNA_BATCH_SIZES,
    "SPECIES_INCLUDE": SPECIES_INCLUDE,
    "SPECIES_EXCLUDE": SPECIES_EXCLUDE,
    "TASK_MODE": TASK_MODE,
    "TARGET_COL": TARGET_COL,
    "PANEL_DEFINITION_NAME": PANEL_DEFINITION_NAME,
    "PANEL_TO_SPECIES": PANEL_TO_SPECIES,
}

path_config = {
    "DATA_ROOT": str(DATA_ROOT),
    "PERSISTENT_DATA_PARENT": str(PERSISTENT_DATA_PARENT),
    "ZIP_PATH": str(ZIP_PATH),
    "SCRATCH_BASE": str(SCRATCH_BASE),
    "SCRATCH_DATA_PARENT": str(SCRATCH_DATA_PARENT),
    "SCRATCH_ZIP_PATH": str(SCRATCH_ZIP_PATH),
    "SCRATCH_EXTRACT_ROOT": str(SCRATCH_EXTRACT_ROOT),
    "SCRATCH_DATA_ROOT": str(SCRATCH_DATA_ROOT),
    "PERSISTENT_OUTPUT_PARENT": str(PERSISTENT_OUTPUT_PARENT),
    "OUTPUT_DIR": str(OUTPUT_DIR),
    "PERSISTENT_OUTPUT_DIR": str(PERSISTENT_OUTPUT_DIR),
    "USE_SCRATCH_DATA": USE_SCRATCH_DATA,
    "USE_SCRATCH_OUTPUT": USE_SCRATCH_OUTPUT,
    "SYNC_EVERY_EPOCH": SYNC_EVERY_EPOCH,
    "SAVE_FIXED_CSVS": SAVE_FIXED_CSVS,
}

with open(OUTPUT_DIR / "path_config.json", "w") as f:
    json.dump(path_config, f, indent=2)
with open(OUTPUT_DIR / "benchmark_config.json", "w") as f:
    json.dump(BASE_CONFIG, f, indent=2)

print("\nARC storage configuration")
print(json.dumps(path_config, indent=2))
print("\nBenchmark configuration")
print(json.dumps(BASE_CONFIG, indent=2))
sync_outputs_to_persistent([OUTPUT_DIR / "path_config.json", OUTPUT_DIR / "benchmark_config.json", OUTPUT_DIR / "run_log.txt"], label="initial")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)


# =========================
# 5. LOAD AND REPAIR SPLIT CSVs
# =========================
CELL_MANIFEST_CSV = DATA_ROOT / "cell_manifest.csv"
TRAIN_CSV = DATA_ROOT / "splits" / "train.csv"
VAL_CSV = DATA_ROOT / "splits" / "val.csv"
HOLDOUT_CSV = DATA_ROOT / "splits" / HOLDOUT_CSV_NAME

if not HOLDOUT_CSV.exists():
    candidates = sorted((DATA_ROOT / "splits").glob("holdout*.csv"))
    if candidates:
        print(f"WARNING: {HOLDOUT_CSV_NAME} not found. Using first holdout-like CSV: {candidates[0].name}")
        HOLDOUT_CSV = candidates[0]

print("Resolved DATA_ROOT:", DATA_ROOT)
print("CELL_MANIFEST_CSV:", CELL_MANIFEST_CSV, "| exists:", CELL_MANIFEST_CSV.exists())
print("TRAIN_CSV:        ", TRAIN_CSV, "| exists:", TRAIN_CSV.exists())
print("VAL_CSV:          ", VAL_CSV, "| exists:", VAL_CSV.exists())
print("HOLDOUT_CSV:      ", HOLDOUT_CSV, "| exists:", HOLDOUT_CSV.exists())

assert CELL_MANIFEST_CSV.exists(), f"Missing cell_manifest.csv at {CELL_MANIFEST_CSV}"
assert TRAIN_CSV.exists(), f"Missing train.csv at {TRAIN_CSV}"
assert VAL_CSV.exists(), f"Missing val.csv at {VAL_CSV}"
assert HOLDOUT_CSV.exists(), f"Missing holdout CSV at {HOLDOUT_CSV}"

cell_manifest_df = pd.read_csv(CELL_MANIFEST_CSV)
train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
holdout_df = pd.read_csv(HOLDOUT_CSV)

print("Cell manifest:", cell_manifest_df.shape)
print("Train:        ", train_df.shape)
print("Val:          ", val_df.shape)
print("Holdout:      ", holdout_df.shape)
print("Columns:", train_df.columns.tolist())


def find_first_existing_column(df, candidates, required=True):
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(f"Could not find any of {candidates}. Available columns: {df.columns.tolist()}")
    return None


SPECIES_COL = find_first_existing_column(train_df, ["species", "label", "class_name", "class", "target"])
CROP_PATH_COL = find_first_existing_column(train_df, ["crop_path", "crop_file", "crop", "image_path", "img_path", "npy_path"])
MASK_PATH_COL = find_first_existing_column(train_df, ["mask_crop_path", "mask_path", "mask_file", "mask_npy_path"], required=False)
CELL_ID_COL = find_first_existing_column(train_df, ["cell_id", "id", "cell_name", "sample_id"], required=False)

for df in [cell_manifest_df, train_df, val_df, holdout_df]:
    if SPECIES_COL != "species" and SPECIES_COL in df.columns:
        df["species"] = df[SPECIES_COL]
    if CROP_PATH_COL != "crop_path" and CROP_PATH_COL in df.columns:
        df["crop_path"] = df[CROP_PATH_COL]
    if MASK_PATH_COL and MASK_PATH_COL != "mask_crop_path" and MASK_PATH_COL in df.columns:
        df["mask_crop_path"] = df[MASK_PATH_COL]
    if CELL_ID_COL and CELL_ID_COL != "cell_id" and CELL_ID_COL in df.columns:
        df["cell_id"] = df[CELL_ID_COL]

print("Using species column:", SPECIES_COL)
print("Using crop path column:", CROP_PATH_COL)
print("Using mask path column:", MASK_PATH_COL)


def rebase_moved_dataset_path(path_value, data_root: Path = DATA_ROOT):
    if pd.isna(path_value):
        return path_value
    s_original = str(path_value)
    s = s_original.replace("\\", "/")
    if s.startswith("/") and Path(s).exists():
        return s

    markers = []
    for marker in [DATASET_ROOT_MARKER, DATASET_FOLDER_NAME, DATA_ROOT.name]:
        if marker and marker not in markers:
            markers.append(marker)

    for marker in markers:
        if marker in s:
            suffix = s.split(marker, 1)[1].lstrip("/")
            return str(Path(data_root) / suffix)

    data_root_s = str(data_root).replace("\\", "/")
    if data_root_s in s:
        suffix = s.split(data_root_s, 1)[1].lstrip("/")
        return str(Path(data_root) / suffix)

    return s_original


def split_fallback_path(row, split_name, path_col="crop_path"):
    candidate = str(row.get(path_col, ""))
    if candidate.startswith("/") and Path(candidate).exists():
        return candidate
    basename = Path(candidate.replace("\\", "/")).name
    if not basename or basename == ".":
        return candidate
    partition = str(row.get("dataset_partition", "")).strip()
    if not partition or partition.lower() == "nan":
        partition = "holdout_plate1" if "holdout" in split_name.lower() else "train_pool_plates_not_1"
    folder = "mask_crops" if path_col == "mask_crop_path" else "crops"
    return str(DATA_ROOT / partition / folder / basename)


def repair_df_paths_fast(df, split_name):
    df = df.copy()
    df["crop_path_original"] = df["crop_path"].astype(str)
    df["crop_path"] = df["crop_path"].map(lambda x: rebase_moved_dataset_path(x, DATA_ROOT))
    exists_after_rebase = df["crop_path"].map(lambda x: Path(str(x)).exists())
    if not bool(exists_after_rebase.all()):
        df.loc[~exists_after_rebase, "crop_path"] = df.loc[~exists_after_rebase].apply(
            lambda r: split_fallback_path(r, split_name, "crop_path"), axis=1
        )

    if "mask_crop_path" in df.columns:
        df["mask_crop_path_original"] = df["mask_crop_path"].astype(str)
        df["mask_crop_path"] = df["mask_crop_path"].map(lambda x: rebase_moved_dataset_path(x, DATA_ROOT))
        mask_exists_after_rebase = df["mask_crop_path"].map(lambda x: Path(str(x)).exists())
        if not bool(mask_exists_after_rebase.all()):
            df.loc[~mask_exists_after_rebase, "mask_crop_path"] = df.loc[~mask_exists_after_rebase].apply(
                lambda r: split_fallback_path(r, split_name, "mask_crop_path"), axis=1
            )
        df["mask_crop_exists"] = df["mask_crop_path"].map(lambda x: Path(str(x)).exists())

    df["crop_exists"] = df["crop_path"].map(lambda x: Path(str(x)).exists())
    path_col_to_assert = "mask_crop_exists" if USE_MASKED_CROPS and "mask_crop_exists" in df.columns else "crop_exists"
    n_ok = int(df[path_col_to_assert].sum())
    n_total = int(len(df))
    pct = (n_ok / n_total * 100) if n_total else 0
    print(f"{split_name}: {path_col_to_assert} = {pct:.2f}% ({n_ok}/{n_total})")

    if n_ok < n_total:
        preview_cols = ["crop_path_original", "crop_path"]
        if "mask_crop_path_original" in df.columns:
            preview_cols += ["mask_crop_path_original", "mask_crop_path"]
        missing_preview = df.loc[~df[path_col_to_assert], preview_cols].head(20)
        missing_preview.to_csv(OUTPUT_DIR / f"missing_paths_preview_{split_name}.csv", index=False)
        print(f"Saved missing path preview: {OUTPUT_DIR / f'missing_paths_preview_{split_name}.csv'}")

    return df


print("Fast-repairing split CSV paths by rebasing old paths onto DATA_ROOT...")
train_df = repair_df_paths_fast(train_df, "train")
val_df = repair_df_paths_fast(val_df, "val")
holdout_df = repair_df_paths_fast(holdout_df, "holdout")

assert train_df["crop_exists"].all(), "Some train crop paths still do not exist."
assert val_df["crop_exists"].all(), "Some val crop paths still do not exist."
assert holdout_df["crop_exists"].all(), "Some holdout crop paths still do not exist."
if USE_MASKED_CROPS:
    if "mask_crop_path" not in train_df.columns:
        raise ValueError("USE_MASKED_CROPS=True but mask_crop_path is not present in the split CSVs.")
    assert train_df["mask_crop_exists"].all(), "Some train mask crop paths still do not exist."
    assert val_df["mask_crop_exists"].all(), "Some val mask crop paths still do not exist."
    assert holdout_df["mask_crop_exists"].all(), "Some holdout mask crop paths still do not exist."

repaired_splits_dir = OUTPUT_DIR / "repaired_split_csvs"
repaired_splits_dir.mkdir(parents=True, exist_ok=True)
train_df.to_csv(repaired_splits_dir / "train_repaired.csv", index=False)
val_df.to_csv(repaired_splits_dir / "val_repaired.csv", index=False)
holdout_df.to_csv(repaired_splits_dir / "holdout_repaired.csv", index=False)

if SAVE_FIXED_CSVS:
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    holdout_df.to_csv(HOLDOUT_CSV, index=False)
    print("Overwrote original split CSVs because SAVE_FIXED_CSVS=True.")

# Optional species filtering for subset training.
if SPECIES_INCLUDE:
    include_set = set(SPECIES_INCLUDE)
    train_df = train_df[train_df["species"].astype(str).str.upper().isin(include_set)].copy()
    val_df = val_df[val_df["species"].astype(str).str.upper().isin(include_set)].copy()
    holdout_df = holdout_df[holdout_df["species"].astype(str).str.upper().isin(include_set)].copy()
    print("Applied SPECIES_INCLUDE:", SPECIES_INCLUDE)
if SPECIES_EXCLUDE:
    exclude_set = set(SPECIES_EXCLUDE)
    train_df = train_df[~train_df["species"].astype(str).str.upper().isin(exclude_set)].copy()
    val_df = val_df[~val_df["species"].astype(str).str.upper().isin(exclude_set)].copy()
    holdout_df = holdout_df[~holdout_df["species"].astype(str).str.upper().isin(exclude_set)].copy()
    print("Applied SPECIES_EXCLUDE:", SPECIES_EXCLUDE)

if len(train_df) == 0 or len(val_df) == 0 or len(holdout_df) == 0:
    raise ValueError(f"Empty split after filtering: train={len(train_df)}, val={len(val_df)}, holdout={len(holdout_df)}")

# Normalize species codes and construct the direct-training target label.
for df in [train_df, val_df, holdout_df]:
    df["species"] = df["species"].astype(str).str.upper().str.strip()

if TASK_MODE == "panel":
    observed_species = set(pd.concat([train_df["species"], val_df["species"], holdout_df["species"]]).unique())
    unmapped = sorted(observed_species - set(SPECIES_TO_PANEL))
    if unmapped:
        raise ValueError(
            f"Panel mapping is incomplete. These species are present but not mapped: {unmapped}. "
            f"Update PANEL_TO_SPECIES before training."
        )
    for df in [train_df, val_df, holdout_df]:
        df[TARGET_COL] = df["species"].map(SPECIES_TO_PANEL)
else:
    for df in [train_df, val_df, holdout_df]:
        df[TARGET_COL] = df["species"]

print(f"Classification task: {TASK_MODE}")
print(f"Target column: {TARGET_COL}")
if TASK_MODE == "panel":
    print(f"Panel definition: {PANEL_DEFINITION_NAME}")
    for panel, species_list in PANEL_TO_SPECIES.items():
        print(f"  {panel}: {', '.join(species_list)}")

# Save labelled split copies without altering the source dataset CSVs.
labelled_splits_dir = OUTPUT_DIR / "labelled_split_csvs"
labelled_splits_dir.mkdir(parents=True, exist_ok=True)
train_df.to_csv(labelled_splits_dir / "train_with_targets.csv", index=False)
val_df.to_csv(labelled_splits_dir / "val_with_targets.csv", index=False)
holdout_df.to_csv(labelled_splits_dir / "holdout_with_targets.csv", index=False)
with open(OUTPUT_DIR / "panel_definition.json", "w") as f:
    json.dump({
        "task_mode": TASK_MODE,
        "panel_definition_name": PANEL_DEFINITION_NAME,
        "panel_to_species": PANEL_TO_SPECIES,
        "species_to_panel": SPECIES_TO_PANEL,
    }, f, indent=2)

path_repair_summary = pd.DataFrame([
    {"split": "train", "n": len(train_df), "n_existing": int(train_df["crop_exists"].sum()), "pct_existing": float(train_df["crop_exists"].mean() * 100)},
    {"split": "val", "n": len(val_df), "n_existing": int(val_df["crop_exists"].sum()), "pct_existing": float(val_df["crop_exists"].mean() * 100)},
    {"split": "holdout", "n": len(holdout_df), "n_existing": int(holdout_df["crop_exists"].sum()), "pct_existing": float(holdout_df["crop_exists"].mean() * 100)},
])
path_repair_summary.to_csv(OUTPUT_DIR / "path_repair_summary.csv", index=False)
print(path_repair_summary.to_string(index=False))

train_counts = train_df["species"].value_counts().sort_index().rename("n_train").to_frame()
val_counts = val_df["species"].value_counts().sort_index().rename("n_val").to_frame()
holdout_counts = holdout_df["species"].value_counts().sort_index().rename("n_holdout").to_frame()
counts_summary = train_counts.join(val_counts, how="outer").join(holdout_counts, how="outer").fillna(0).astype(int)
counts_summary.to_csv(OUTPUT_DIR / "species_counts_by_split.csv")
print("Saved species count summary:", OUTPUT_DIR / "species_counts_by_split.csv")
print(counts_summary.to_string())

target_train_counts = train_df[TARGET_COL].value_counts().sort_index().rename("n_train").to_frame()
target_val_counts = val_df[TARGET_COL].value_counts().sort_index().rename("n_val").to_frame()
target_holdout_counts = holdout_df[TARGET_COL].value_counts().sort_index().rename("n_holdout").to_frame()
target_counts_summary = target_train_counts.join(target_val_counts, how="outer").join(target_holdout_counts, how="outer").fillna(0).astype(int)
target_counts_summary.to_csv(OUTPUT_DIR / "target_class_counts_by_split.csv")
print("Saved target-class count summary:", OUTPUT_DIR / "target_class_counts_by_split.csv")
print(target_counts_summary.to_string())

path_col_for_load = "mask_crop_path" if USE_MASKED_CROPS else "crop_path"
example_crop = np.load(train_df.iloc[0][path_col_for_load])
print("Example crop path:", train_df.iloc[0][path_col_for_load])
print("Example crop shape:", example_crop.shape, "dtype:", example_crop.dtype)
assert example_crop.ndim == 3, "Expected crop shape (C, H, W)."
assert example_crop.shape[0] == 5, "Expected 5 channels: PHASECONTRAST, BF, DAPI, WGA, FM1."

crop_info = {
    "example_crop_path": str(train_df.iloc[0][path_col_for_load]),
    "example_crop_shape": list(example_crop.shape),
    "example_crop_dtype": str(example_crop.dtype),
    "expected_channel_order": CROP_CHANNEL_ORDER,
    "use_masked_crops": USE_MASKED_CROPS,
}
with open(OUTPUT_DIR / "crop_sanity_check.json", "w") as f:
    json.dump(crop_info, f, indent=2)

if "plate" in train_df.columns:
    print("Plate distribution after filtering:")
    print("Train plates:\n", train_df["plate"].value_counts().sort_index().to_string())
    print("Val plates:\n", val_df["plate"].value_counts().sort_index().to_string())
    print("Holdout plates:\n", holdout_df["plate"].value_counts().sort_index().to_string())
    # Keep original Plate 1 leakage checks only when the holdout CSV appears to be Plate 1.
    if "plate1" in HOLDOUT_CSV.name.lower():
        assert not (pd.to_numeric(train_df["plate"], errors="coerce") == 1).any(), "Plate 1 leaked into train."
        assert not (pd.to_numeric(val_df["plate"], errors="coerce") == 1).any(), "Plate 1 leaked into val."
        assert (pd.to_numeric(holdout_df["plate"], errors="coerce") == 1).all(), "Holdout contains non-Plate1 rows."
        print("Plate 1 leakage checks passed.")

all_classes = sorted(train_df[TARGET_COL].astype(str).unique())
class_to_idx = {c: i for i, c in enumerate(all_classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}
for name, df in [("val", val_df), ("holdout", holdout_df)]:
    unseen = sorted(set(df[TARGET_COL].astype(str).unique()) - set(all_classes))
    if unseen:
        raise ValueError(f"{name} contains target classes unseen in train: {unseen}")
print(f"Number of classes ({TASK_MODE}):", len(all_classes))
print(all_classes)
with open(OUTPUT_DIR / "class_to_idx.json", "w") as f:
    json.dump(class_to_idx, f, indent=2)

sync_outputs_to_persistent([
    OUTPUT_DIR / "path_repair_summary.csv",
    OUTPUT_DIR / "species_counts_by_split.csv",
    OUTPUT_DIR / "target_class_counts_by_split.csv",
    OUTPUT_DIR / "panel_definition.json",
    OUTPUT_DIR / "crop_sanity_check.json",
    OUTPUT_DIR / "class_to_idx.json",
    repaired_splits_dir,
    labelled_splits_dir,
    OUTPUT_DIR / "run_log.txt",
], label="after_data_checks")


# =========================
# 6. DATASET, NORMALIZATION, MODEL
# =========================
train_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.04, scale_limit=0.05, rotate_limit=12, border_mode=0, value=0, p=0.5),
])


def validate_channel_names(channel_names):
    unknown = [c for c in channel_names if c not in CHANNEL_MAP]
    if unknown:
        raise ValueError(f"Unknown channel names: {unknown}. Valid channels: {list(CHANNEL_MAP)}")
    return [CHANNEL_MAP[c] for c in channel_names]


def compute_train_channel_stats(df, selected_channels, max_samples=5000):
    if len(df) > max_samples:
        sample_df = df.sample(n=max_samples, random_state=SEED)
    else:
        sample_df = df
    sums = np.zeros(len(selected_channels), dtype=np.float64)
    sq_sums = np.zeros(len(selected_channels), dtype=np.float64)
    counts = np.zeros(len(selected_channels), dtype=np.float64)
    for path in sample_df[path_col_for_load]:
        crop = np.load(path).astype(np.float32)[selected_channels]
        flat = crop.reshape(crop.shape[0], -1)
        sums += flat.sum(axis=1)
        sq_sums += (flat ** 2).sum(axis=1)
        counts += flat.shape[1]
    means = sums / np.maximum(counts, 1)
    vars_ = sq_sums / np.maximum(counts, 1) - means ** 2
    stds = np.sqrt(np.maximum(vars_, 1e-8))
    return means.astype(np.float32), stds.astype(np.float32)


class BacteriaCropDataset(Dataset):
    def __init__(self, df, class_to_idx, selected_channels, img_size=96, augment=False,
                 normalization_mode="train_global_zscore", train_mean=None, train_std=None,
                 percentile_low=1, percentile_high=99, path_col="crop_path"):
        self.df = df.reset_index(drop=True).copy()
        self.class_to_idx = class_to_idx
        self.selected_channels = selected_channels
        self.img_size = img_size
        self.augment = augment
        self.normalization_mode = normalization_mode
        self.train_mean = np.asarray(train_mean if train_mean is not None else 0, dtype=np.float32).reshape(-1, 1, 1)
        self.train_std = np.asarray(train_std if train_std is not None else 1, dtype=np.float32).reshape(-1, 1, 1)
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.path_col = path_col

    def __len__(self):
        return len(self.df)

    def _normalize(self, x):
        x = x.astype(np.float32)
        if self.normalization_mode == "train_global_zscore":
            return (x - self.train_mean) / (self.train_std + 1e-6)
        elif self.normalization_mode == "per_crop_zscore":
            mean = x.reshape(x.shape[0], -1).mean(axis=1).reshape(-1, 1, 1)
            std = x.reshape(x.shape[0], -1).std(axis=1).reshape(-1, 1, 1)
            return (x - mean) / (std + 1e-6)
        elif self.normalization_mode == "percentile_minmax":
            out = np.zeros_like(x, dtype=np.float32)
            for i in range(x.shape[0]):
                lo, hi = np.percentile(x[i], [self.percentile_low, self.percentile_high])
                if hi <= lo:
                    hi = lo + 1.0
                out[i] = np.clip((x[i] - lo) / (hi - lo), 0, 1)
            return out
        elif self.normalization_mode == "none":
            return x
        else:
            raise ValueError(f"Unknown normalization mode: {self.normalization_mode}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        crop = np.load(row[self.path_col]).astype(np.float32)
        crop = crop[self.selected_channels]
        if crop.shape[-2:] != (self.img_size, self.img_size):
            t = torch.from_numpy(crop).unsqueeze(0)
            t = F.interpolate(t, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
            crop = t.squeeze(0).numpy()
        if self.augment and USE_AUGMENTATION:
            hwc = np.transpose(crop, (1, 2, 0))
            hwc = train_aug(image=hwc)["image"]
            crop = np.transpose(hwc, (2, 0, 1))
        crop = self._normalize(crop)
        x = torch.from_numpy(crop).float()
        y = torch.tensor(self.class_to_idx[str(row[TARGET_COL])], dtype=torch.long)
        return x, y


def make_weighted_sampler(df):
    labels = df[TARGET_COL].astype(str).map(class_to_idx).values
    counts = Counter(labels)
    sample_weights = np.array([1.0 / counts[y] for y in labels], dtype=np.float32)
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def build_datasets_and_loaders(selected_channels, batch_size, train_mean, train_std):
    train_ds = BacteriaCropDataset(train_df, class_to_idx, selected_channels, IMG_SIZE, augment=True,
                                   normalization_mode=NORMALIZATION_MODE, train_mean=train_mean, train_std=train_std,
                                   percentile_low=PERCENTILE_LOW, percentile_high=PERCENTILE_HIGH,
                                   path_col=path_col_for_load)
    val_ds = BacteriaCropDataset(val_df, class_to_idx, selected_channels, IMG_SIZE, augment=False,
                                 normalization_mode=NORMALIZATION_MODE, train_mean=train_mean, train_std=train_std,
                                 percentile_low=PERCENTILE_LOW, percentile_high=PERCENTILE_HIGH,
                                 path_col=path_col_for_load)
    holdout_ds = BacteriaCropDataset(holdout_df, class_to_idx, selected_channels, IMG_SIZE, augment=False,
                                     normalization_mode=NORMALIZATION_MODE, train_mean=train_mean, train_std=train_std,
                                     percentile_low=PERCENTILE_LOW, percentile_high=PERCENTILE_HIGH,
                                     path_col=path_col_for_load)
    sampler = make_weighted_sampler(train_df) if USE_WEIGHTED_SAMPLER else None
    shuffle = sampler is None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    holdout_loader = DataLoader(holdout_ds, batch_size=batch_size, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)
    return train_ds, val_ds, holdout_ds, train_loader, val_loader, holdout_loader


def get_norm_stats_for_condition(channel_names, selected_channels):
    if NORMALIZATION_MODE == "train_global_zscore":
        train_mean, train_std = compute_train_channel_stats(train_df, selected_channels)
    else:
        train_mean = np.zeros(len(selected_channels), dtype=np.float32)
        train_std = np.ones(len(selected_channels), dtype=np.float32)
    norm_stats = {
        "normalization_mode": NORMALIZATION_MODE,
        "selected_channel_names": channel_names,
        "selected_channels": selected_channels,
        "mean": train_mean.tolist(),
        "std": train_std.tolist(),
    }
    return train_mean, train_std, norm_stats


def get_weights_arg(pretrained):
    return "DEFAULT" if pretrained else None


def _adapt_input_conv(old: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Create an input convolution with a new channel count while preserving
    pretrained information when possible.

    The original script replaced the first convolution with a randomly
    initialized layer even when USE_PRETRAINED=True. This helper keeps the
    original weights for 3-channel input, averages RGB filters for 1 channel,
    and repeats/scales the mean filter for other channel counts.
    """
    if in_channels == old.in_channels:
        return old

    new_conv = nn.Conv2d(
        in_channels, old.out_channels, kernel_size=old.kernel_size,
        stride=old.stride, padding=old.padding, dilation=old.dilation,
        groups=old.groups, bias=(old.bias is not None), padding_mode=old.padding_mode,
    )
    with torch.no_grad():
        mean_weight = old.weight.mean(dim=1, keepdim=True)
        if in_channels == 1:
            new_weight = mean_weight
        else:
            new_weight = mean_weight.repeat(1, in_channels, 1, 1)
            # Keep the overall activation scale close to the original RGB stem.
            new_weight *= old.in_channels / float(in_channels)
        new_conv.weight.copy_(new_weight)
        if old.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(old.bias)
    return new_conv


def replace_first_conv_resnet(model, in_channels):
    model.conv1 = _adapt_input_conv(model.conv1, in_channels)
    return model


def replace_first_conv_densenet(model, in_channels):
    model.features.conv0 = _adapt_input_conv(model.features.conv0, in_channels)
    return model


def replace_first_conv_convnext(model, in_channels):
    model.features[0][0] = _adapt_input_conv(model.features[0][0], in_channels)
    return model


def build_model(model_name, in_channels, num_classes, pretrained=False):
    weights = get_weights_arg(pretrained)
    if model_name == "resnet18":
        model = models.resnet18(weights=weights)
        model = replace_first_conv_resnet(model, in_channels)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet34":
        model = models.resnet34(weights=weights)
        model = replace_first_conv_resnet(model, in_channels)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "densenet121":
        model = models.densenet121(weights=weights)
        model = replace_first_conv_densenet(model, in_channels)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=weights)
        model = replace_first_conv_convnext(model, in_channels)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def build_model_with_dropout(model_name, in_channels, num_classes, pretrained=False, dropout=0.0):
    model = build_model(model_name, in_channels, num_classes, pretrained=pretrained)
    if dropout <= 0:
        return model
    if model_name.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    elif model_name == "densenet121":
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    elif model_name == "convnext_tiny":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return model


def make_criterion(use_weighted_loss=True):
    if not use_weighted_loss:
        return nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    train_labels = train_df[TARGET_COL].astype(str).map(class_to_idx).values
    counts = np.bincount(train_labels, minlength=len(all_classes)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()
    weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    print("Class weights:", {idx_to_class[i]: float(weights[i].detach().cpu()) for i in range(len(all_classes))})
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)


def run_one_epoch(model, loader, criterion, optimizer=None, scaler=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_n = 0
    all_true = []
    all_pred = []
    all_top2_correct = []

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast("cuda", enabled=(USE_AMP and DEVICE == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_n += bs
        pred = logits.argmax(dim=1)
        topk = min(2, logits.shape[1])
        top2_idx = torch.topk(logits, k=topk, dim=1).indices
        top2_correct = (top2_idx == y.unsqueeze(1)).any(dim=1)
        all_true.extend(y.detach().cpu().numpy().tolist())
        all_pred.extend(pred.detach().cpu().numpy().tolist())
        all_top2_correct.extend(top2_correct.detach().cpu().numpy().astype(np.float32).tolist())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    return {
        "loss": total_loss / max(total_n, 1),
        "acc": accuracy_score(all_true, all_pred),
        "macro_f1": f1_score(all_true, all_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(all_true, all_pred, average="weighted", zero_division=0),
        "bal_acc": balanced_accuracy_score(all_true, all_pred),
        "top2_acc": float(np.mean(all_top2_correct)) if all_top2_correct else float("nan"),
        "y_true": all_true,
        "y_pred": all_pred,
    }


def save_visual_sanity_check(channel_names, selected_channels, out_dir):
    def robust01(x, p1=1, p2=99):
        x = np.asarray(x).astype(np.float32)
        lo, hi = np.percentile(x, [p1, p2])
        if hi <= lo:
            hi = lo + 1.0
        return np.clip((x - lo) / (hi - lo), 0, 1)

    n_show = min(4, len(train_df))
    sample_rows = train_df.sample(n=n_show, random_state=SEED)
    n_cols = len(selected_channels)
    fig, axes = plt.subplots(n_show, n_cols, figsize=(max(4, 3*n_cols), 3*n_show))
    if n_show == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_show == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    for r, (_, row) in enumerate(sample_rows.iterrows()):
        crop = np.load(row[path_col_for_load])
        for c, ch_idx in enumerate(selected_channels):
            axes[r, c].imshow(robust01(crop[ch_idx]), cmap="gray")
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(channel_names[c])
        display_label = str(row[TARGET_COL])
        if TASK_MODE == "panel":
            display_label = f"{row['species']} -> {display_label}"
        axes[r, 0].set_ylabel(display_label, rotation=0, labelpad=55, va="center")

    plt.tight_layout()
    png = out_dir / "sanity_check_selected_channels.png"
    plt.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    sample_rows.to_csv(out_dir / "sanity_check_sample_rows.csv", index=False)
    print("Saved visual sanity check:", png)


# =========================
# 7. OPTUNA AND TRAINING ROUTINES
# =========================
def run_optuna_for_model(model_name, model_dir, optuna_channel_names=None):
    """Run Optuna for a model.

    If optuna_channel_names is omitted, OPTUNA_CHANNEL_NAMES is used. When
    OPTUNA_ONCE_PER_MODEL=0, main() passes each channel condition here so each
    condition can receive its own hyperparameter search.
    """
    channel_names_for_optuna = optuna_channel_names or OPTUNA_CHANNEL_NAMES
    optuna_dir = model_dir / "optuna" / safe_name(channel_names_for_optuna)
    optuna_dir.mkdir(parents=True, exist_ok=True)
    best_params_path = optuna_dir / "optuna_best_params.json"

    if REUSE_EXISTING_OPTUNA and best_params_path.exists():
        print(f"Reusing existing Optuna params for {model_name}: {best_params_path}")
        with open(best_params_path, "r") as f:
            return json.load(f)

    selected_channels = validate_channel_names(channel_names_for_optuna)
    train_mean, train_std, norm_stats = get_norm_stats_for_condition(channel_names_for_optuna, selected_channels)
    with open(optuna_dir / "normalization_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)

    def objective(trial):
        seed_everything(SEED)
        trial_lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        trial_weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        trial_dropout = trial.suggest_float("dropout", 0.0, 0.5)
        trial_batch_size = trial.suggest_categorical("batch_size", OPTUNA_BATCH_SIZES)
        trial_use_weighted_loss = trial.suggest_categorical("use_weighted_loss", [True, False])

        _, _, _, trial_train_loader, trial_val_loader, _ = build_datasets_and_loaders(
            selected_channels, trial_batch_size, train_mean, train_std
        )
        trial_model = build_model_with_dropout(
            model_name=model_name,
            in_channels=len(selected_channels),
            num_classes=len(all_classes),
            pretrained=USE_PRETRAINED,
            dropout=trial_dropout,
        ).to(DEVICE)
        trial_criterion = make_criterion(trial_use_weighted_loss)
        trial_optimizer = torch.optim.AdamW(trial_model.parameters(), lr=trial_lr, weight_decay=trial_weight_decay)
        trial_scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and DEVICE == "cuda"))

        best_val_f1 = -1.0
        for epoch in range(OPTUNA_EPOCHS):
            _ = run_one_epoch(trial_model, trial_train_loader, trial_criterion, optimizer=trial_optimizer, scaler=trial_scaler)
            val_metrics = run_one_epoch(trial_model, trial_val_loader, trial_criterion, optimizer=None, scaler=trial_scaler)
            best_val_f1 = max(best_val_f1, val_metrics["macro_f1"])
            trial.report(best_val_f1, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return best_val_f1

    print(f"\nStarting Optuna for model={model_name}, channels={channel_names_for_optuna}")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, timeout=OPTUNA_TIMEOUT)

    print("Best trial value:", study.best_trial.value)
    print("Best params:", study.best_trial.params)

    optuna_df = study.trials_dataframe()
    optuna_df.to_csv(optuna_dir / "optuna_trials.csv", index=False)
    best_params = dict(study.best_trial.params)
    best_params["model_name"] = model_name
    best_params["optuna_channel_names"] = channel_names_for_optuna
    best_params["best_trial_value"] = float(study.best_trial.value)
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    sync_outputs_to_persistent([optuna_dir, OUTPUT_DIR / "run_log.txt"], label=f"optuna_{model_name}")
    return best_params


def default_params_for_model(model_name):
    return {
        "model_name": model_name,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "dropout": 0.0,
        "batch_size": BATCH_SIZE,
        "use_weighted_loss": USE_WEIGHTED_LOSS,
        "best_trial_value": None,
        "optuna_channel_names": None,
    }


def save_classification_outputs(metrics, split_name, experiment_dir):
    y_true = metrics["y_true"]
    y_pred = metrics["y_pred"]
    labels = list(range(len(all_classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100

    raw_csv = experiment_dir / f"confusion_matrix_{split_name}_raw.csv"
    pct_csv = experiment_dir / f"confusion_matrix_{split_name}_row_percent.csv"
    report_csv = experiment_dir / f"classification_report_{split_name}.csv"

    pd.DataFrame(cm, index=all_classes, columns=all_classes).to_csv(raw_csv)
    pd.DataFrame(cm_pct, index=all_classes, columns=all_classes).to_csv(pct_csv)

    report_dict = classification_report(y_true, y_pred, labels=labels, target_names=all_classes, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report_dict).T
    report_df.to_csv(report_csv)
    display(report_df)

    fig, ax = plt.subplots(figsize=(max(8, len(all_classes) * 0.55), max(7, len(all_classes) * 0.55)))
    im = ax.imshow(cm_pct)
    ax.set_xticks(np.arange(len(all_classes)))
    ax.set_yticks(np.arange(len(all_classes)))
    ax.set_xticklabels(all_classes, rotation=45, ha="right")
    ax.set_yticklabels(all_classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{split_name}: row-normalized confusion matrix (%)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(len(all_classes)):
        for j in range(len(all_classes)):
            val = cm_pct[i, j]
            if val >= 1:
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7)
    plt.tight_layout()
    png = experiment_dir / f"confusion_matrix_{split_name}_row_percent.png"
    plt.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {split_name} outputs:")
    print("  ", raw_csv)
    print("  ", pct_csv)
    print("  ", report_csv)
    print("  ", png)


def train_one_condition(model_name, channel_names, best_params, model_dir):
    seed_everything(SEED)
    selected_channels = validate_channel_names(channel_names)
    condition_name = safe_name(channel_names)
    experiment_name = f"{TASK_MODE}_{model_name}_{condition_name}_{NORMALIZATION_MODE}"
    experiment_dir = model_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    persistent_experiment_dir = PERSISTENT_OUTPUT_DIR / model_name / experiment_name
    persistent_experiment_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"Training condition: model={model_name} | channels={channel_names}")
    print("Active experiment directory:", experiment_dir)
    print("Persistent experiment backup:", persistent_experiment_dir)

    train_mean, train_std, norm_stats = get_norm_stats_for_condition(channel_names, selected_channels)
    batch_size = int(best_params.get("batch_size", BATCH_SIZE))
    lr = float(best_params.get("lr", LR))
    weight_decay = float(best_params.get("weight_decay", WEIGHT_DECAY))
    dropout = float(best_params.get("dropout", 0.0))
    use_weighted_loss = bool(best_params.get("use_weighted_loss", USE_WEIGHTED_LOSS))

    train_ds, val_ds, holdout_ds, train_loader, val_loader, holdout_loader = build_datasets_and_loaders(
        selected_channels, batch_size, train_mean, train_std
    )

    xb, yb = next(iter(train_loader))
    print("Batch x:", xb.shape, xb.dtype)
    print("Batch y:", yb.shape, yb[:10])

    save_visual_sanity_check(channel_names, selected_channels, experiment_dir)

    model = build_model_with_dropout(model_name, len(selected_channels), len(all_classes), USE_PRETRAINED, dropout).to(DEVICE)
    criterion = make_criterion(use_weighted_loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and DEVICE == "cuda"))

    effective_config = dict(BASE_CONFIG)
    effective_config.update({
        "MODEL_NAME": model_name,
        "TASK_MODE": TASK_MODE,
        "TARGET_COL": TARGET_COL,
        "PANEL_DEFINITION_NAME": PANEL_DEFINITION_NAME,
        "PANEL_TO_SPECIES": PANEL_TO_SPECIES,
        "SELECTED_CHANNEL_NAMES": channel_names,
        "SELECTED_CHANNELS": selected_channels,
        "BATCH_SIZE": batch_size,
        "LR": lr,
        "WEIGHT_DECAY": weight_decay,
        "USE_WEIGHTED_LOSS": use_weighted_loss,
        "BEST_DROPOUT": dropout,
        "experiment_name": experiment_name,
        "experiment_dir": str(experiment_dir),
        "persistent_experiment_dir": str(persistent_experiment_dir),
        "reused_optuna_params": best_params,
    })

    with open(experiment_dir / "training_config_effective.json", "w") as f:
        json.dump(effective_config, f, indent=2)
    with open(experiment_dir / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)
    with open(experiment_dir / "normalization_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)

    sync_outputs_to_persistent([experiment_dir, OUTPUT_DIR / "run_log.txt"], label=f"before_training_{experiment_name}")

    history = []
    best_val_f1 = -1.0
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        train_metrics = run_one_epoch(model, train_loader, criterion, optimizer=optimizer, scaler=scaler)
        val_metrics = run_one_epoch(model, val_loader, criterion, optimizer=None, scaler=scaler)
        scheduler.step(val_metrics["macro_f1"])

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_weighted_f1": train_metrics["weighted_f1"],
            "train_bal_acc": train_metrics["bal_acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_bal_acc": val_metrics["bal_acc"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.time() - epoch_start,
        }
        history.append(row)
        history_df = pd.DataFrame(history)
        history_df.to_csv(experiment_dir / "training_history.csv", index=False)

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train_loss={row['train_loss']:.4f} | train_f1={row['train_macro_f1']:.4f} | "
            f"val_loss={row['val_loss']:.4f} | val_f1={row['val_macro_f1']:.4f} | "
            f"lr={row['lr']:.2e} | {row['epoch_seconds']:.1f}s"
        )

        latest_ckpt_path = experiment_dir / "latest_model.pt"
        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
            "selected_channel_names": channel_names,
            "selected_channels": selected_channels,
            "channel_map": CHANNEL_MAP,
            "crop_channel_order": CROP_CHANNEL_ORDER,
            "model_name": model_name,
            "task_mode": TASK_MODE,
            "target_col": TARGET_COL,
            "panel_definition_name": PANEL_DEFINITION_NAME,
            "panel_to_species": PANEL_TO_SPECIES,
            "normalization_mode": NORMALIZATION_MODE,
            "normalization_stats": norm_stats,
            "current_val_macro_f1": float(val_metrics["macro_f1"]),
            "best_val_macro_f1_so_far": float(max(best_val_f1, val_metrics["macro_f1"])),
            "best_dropout": dropout,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "use_masked_crops": USE_MASKED_CROPS,
            "path_col_for_load": path_col_for_load,
        }
        torch.save(ckpt_payload, latest_ckpt_path)

        best_updated = False
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_no_improve = 0
            best_ckpt_path = experiment_dir / "best_model.pt"
            ckpt_payload["best_val_macro_f1"] = float(best_val_f1)
            torch.save(ckpt_payload, best_ckpt_path)
            best_updated = True
            print(f"  New best model saved: epoch={epoch}, val_macro_f1={best_val_f1:.4f}")
        else:
            epochs_no_improve += 1

        if SYNC_EVERY_EPOCH:
            epoch_sync_paths = [
                experiment_dir / "training_history.csv",
                experiment_dir / "latest_model.pt",
                experiment_dir / "best_model.pt",
                experiment_dir / "training_config_effective.json",
                experiment_dir / "class_to_idx.json",
                experiment_dir / "normalization_stats.json",
                OUTPUT_DIR / "run_log.txt",
                OUTPUT_DIR / "path_config.json",
                OUTPUT_DIR / "benchmark_config.json",
            ]
            sync_outputs_to_persistent(epoch_sync_paths, label=f"{experiment_name}_epoch_{epoch:03d}")

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(experiment_dir / "training_history.csv", index=False)
    sync_outputs_to_persistent([
        experiment_dir / "training_history.csv",
        experiment_dir / "latest_model.pt",
        experiment_dir / "best_model.pt",
        OUTPUT_DIR / "run_log.txt",
    ], label=f"training_finished_{experiment_name}")

    print("Best epoch:", best_epoch)
    print("Best val macro F1:", best_val_f1)
    display(history_df.tail())

    # Plot training curves.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history_df["epoch"], history_df["train_macro_f1"], label="train macro F1")
    ax.plot(history_df["epoch"], history_df["val_macro_f1"], label="val macro F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro F1")
    ax.set_title("Training curves")
    ax.legend()
    plt.tight_layout()
    plt.savefig(experiment_dir / "training_curve_macro_f1.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history_df["epoch"], history_df["train_loss"], label="train loss")
    ax.plot(history_df["epoch"], history_df["val_loss"], label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss curves")
    ax.legend()
    plt.tight_layout()
    plt.savefig(experiment_dir / "training_curve_loss.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Load best checkpoint and evaluate val + holdout.
    ckpt = torch.load(experiment_dir / "best_model.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    print("Loaded best model from epoch:", ckpt["epoch"])
    print("Best val macro F1:", ckpt["best_val_macro_f1"])

    val_metrics = run_one_epoch(model, val_loader, criterion, optimizer=None, scaler=scaler)
    holdout_metrics = run_one_epoch(model, holdout_loader, criterion, optimizer=None, scaler=scaler)

    print("Validation metrics")
    print({k: val_metrics[k] for k in ["loss", "acc", "macro_f1", "weighted_f1", "bal_acc", "top2_acc"]})
    print("\nHoldout metrics")
    print({k: holdout_metrics[k] for k in ["loss", "acc", "macro_f1", "weighted_f1", "bal_acc", "top2_acc"]})

    save_classification_outputs(val_metrics, "val", experiment_dir)
    save_classification_outputs(holdout_metrics, "holdout", experiment_dir)

    summary = {
        "task_mode": TASK_MODE,
        "target_col": TARGET_COL,
        "panel_definition_name": PANEL_DEFINITION_NAME if TASK_MODE == "panel" else None,
        "model_name": model_name,
        "selected_channel_names": channel_names,
        "selected_channels": selected_channels,
        "channel_condition": "+".join(channel_names),
        "crop_channel_order": CROP_CHANNEL_ORDER,
        "normalization_mode": NORMALIZATION_MODE,
        "num_classes": len(all_classes),
        "classes": all_classes,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_holdout": int(len(holdout_df)),
        "best_epoch": int(ckpt["epoch"]),
        "best_val_macro_f1": float(ckpt["best_val_macro_f1"]),
        "val_acc": float(val_metrics["acc"]),
        "val_macro_f1": float(val_metrics["macro_f1"]),
        "val_weighted_f1": float(val_metrics["weighted_f1"]),
        "val_bal_acc": float(val_metrics["bal_acc"]),
        "val_top2_acc": float(val_metrics["top2_acc"]),
        "holdout_acc": float(holdout_metrics["acc"]),
        "holdout_macro_f1": float(holdout_metrics["macro_f1"]),
        "holdout_weighted_f1": float(holdout_metrics["weighted_f1"]),
        "holdout_bal_acc": float(holdout_metrics["bal_acc"]),
        "holdout_top2_acc": float(holdout_metrics["top2_acc"]),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "best_dropout": float(dropout),
        "use_weighted_loss": bool(use_weighted_loss),
        "run_optuna": bool(RUN_OPTUNA),
        "optuna_trials": int(N_TRIALS if RUN_OPTUNA else 0),
        "optuna_channel_names": best_params.get("optuna_channel_names"),
        "optuna_best_trial_value": best_params.get("best_trial_value"),
        "use_masked_crops": bool(USE_MASKED_CROPS),
        "data_root": str(DATA_ROOT),
        "output_dir": str(OUTPUT_DIR),
        "persistent_output_dir": str(PERSISTENT_OUTPUT_DIR),
        "experiment_dir": str(experiment_dir),
        "persistent_experiment_dir": str(persistent_experiment_dir),
    }

    summary_df = pd.DataFrame([summary])
    display(summary_df)
    summary_df.to_csv(experiment_dir / "experiment_summary.csv", index=False)
    with open(experiment_dir / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    sync_outputs_to_persistent([experiment_dir, OUTPUT_DIR / "run_log.txt"], label=f"summary_and_metrics_{experiment_name}")
    return summary


# =========================
# 8. MAIN BENCHMARK LOOP
# =========================
def main():
    benchmark_summaries = []
    optuna_params_by_model = {}

    for model_name in MODEL_BENCHMARKS:
        model_dir = OUTPUT_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        print("\n" + "#" * 100)
        print(f"MODEL BENCHMARK: {model_name} | TASK_MODE={TASK_MODE} | N_CLASSES={len(all_classes)}")
        print("#" * 100)

        if RUN_OPTUNA and OPTUNA_ONCE_PER_MODEL:
            shared_best_params = run_optuna_for_model(model_name, model_dir)
        elif RUN_OPTUNA and not OPTUNA_ONCE_PER_MODEL:
            shared_best_params = None
        else:
            shared_best_params = default_params_for_model(model_name)

        if shared_best_params is not None:
            optuna_params_by_model[model_name] = shared_best_params
            with open(model_dir / "reused_hyperparameters_for_conditions.json", "w") as f:
                json.dump(shared_best_params, f, indent=2)
            sync_outputs_to_persistent([model_dir / "reused_hyperparameters_for_conditions.json"], label=f"params_{model_name}")
        else:
            optuna_params_by_model[model_name] = {}

        for channel_names in CHANNEL_CONDITIONS:
            if RUN_OPTUNA and not OPTUNA_ONCE_PER_MODEL:
                best_params = run_optuna_for_model(model_name, model_dir, optuna_channel_names=channel_names)
                optuna_params_by_model[model_name]["+".join(channel_names)] = best_params
            else:
                best_params = shared_best_params
            summary = train_one_condition(model_name, channel_names, best_params, model_dir)
            benchmark_summaries.append(summary)
            benchmark_df = pd.DataFrame(benchmark_summaries)
            benchmark_df.to_csv(OUTPUT_DIR / "benchmark_summary.csv", index=False)
            sync_outputs_to_persistent([OUTPUT_DIR / "benchmark_summary.csv", OUTPUT_DIR / "run_log.txt"], label="benchmark_summary_incremental")

    benchmark_df = pd.DataFrame(benchmark_summaries)
    benchmark_csv = OUTPUT_DIR / "benchmark_summary.csv"
    benchmark_json = OUTPUT_DIR / "benchmark_summary.json"
    benchmark_df.to_csv(benchmark_csv, index=False)
    with open(benchmark_json, "w") as f:
        json.dump(benchmark_summaries, f, indent=2)

    with open(OUTPUT_DIR / "optuna_params_by_model.json", "w") as f:
        json.dump(optuna_params_by_model, f, indent=2)

    inventory_rows = []
    for p in sorted(OUTPUT_DIR.rglob("*")):
        if p.is_file():
            inventory_rows.append({
                "relative_path": str(p.relative_to(OUTPUT_DIR)),
                "size_bytes": int(p.stat().st_size),
                "modified_time": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
            })
    inventory_df = pd.DataFrame(inventory_rows)
    inventory_csv = OUTPUT_DIR / "output_file_inventory.csv"
    inventory_df.to_csv(inventory_csv, index=False)

    sync_outputs_to_persistent([OUTPUT_DIR], label="final_full_sync")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("Benchmark summary:", benchmark_csv)
    print("Output inventory:", inventory_csv)
    print("Main active output folder:", OUTPUT_DIR)
    print("Persistent output folder:", PERSISTENT_OUTPUT_DIR)
    print("\nTop-level output files/folders:")
    for p in sorted(OUTPUT_DIR.iterdir()):
        print(" -", p.name)

    print("\nKey files to check from terminal:")
    for p in [
        OUTPUT_DIR / "run_log.txt",
        OUTPUT_DIR / "species_counts_by_split.csv",
        OUTPUT_DIR / "target_class_counts_by_split.csv",
        OUTPUT_DIR / "panel_definition.json",
        OUTPUT_DIR / "benchmark_summary.csv",
        OUTPUT_DIR / "optuna_params_by_model.json",
        OUTPUT_DIR / "output_file_inventory.csv",
    ]:
        print(" -", p, "| exists:", p.exists())


if __name__ == "__main__":
    main()
