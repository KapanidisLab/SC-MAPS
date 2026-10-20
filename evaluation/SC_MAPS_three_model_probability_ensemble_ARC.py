#!/usr/bin/env python3
"""
SC-MAPS three-model probability ensemble evaluation for ARC.

Evaluates single-channel DenseNet121, ResNet18 and ResNet34 checkpoints
on a locked holdout set, creates within-architecture mean-probability
ensembles for selectable channel combinations, and optionally averages
those architecture-level ensembles across all three architectures.

Outputs:
  - individual model/channel metrics
  - architecture probability-ensemble metrics
  - cross-architecture ensemble metrics
  - classification reports
  - raw and row-normalised confusion matrices (CSV + PNG)
  - per-cell predictions
  - optional full probability arrays
"""


# ==============================================================================
# 1. Imports
# ==============================================================================


from __future__ import annotations

import gc
import json
import math
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm.auto import tqdm

warnings.filterwarnings("once")

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ==============================================================================
# 2. User configuration
# ==============================================================================


# ============================================================
# EDIT THESE PATHS
# ============================================================

DATA_ROOT = Path(
     r"/scratch/phys3115/bacteria_data/Trillian_DATASET_with_holdout"
)

# Locked Plate-1 holdout.
HOLDOUT_CSV = DATA_ROOT / "splits" / "holdout_plate1.csv"

# Point each architecture to the TOP-LEVEL folder containing its single-channel weights.
MODEL_WEIGHT_ROOTS = {
    "densenet121": Path(
        r"/data/phys-deep-learning-bacteria/phys3115/Bacteria-data-15 species/Trillian/Single_channel_weights"
    ),
    "resnet18": Path(
        r"/data/phys-deep-learning-bacteria/phys3115/Bacteria-data-15 species/Trillian/Resnet_weights/Resnet18"
    ),
    "resnet34": Path(
        r"/data/phys-deep-learning-bacteria/phys3115/Bacteria-data-15 species/Trillian/Resnet_weights/Resnet34"
    ),
}

# The architectures to evaluate. You can remove/add supported models here.
MODELS_TO_RUN = ["densenet121", "resnet18", "resnet34"]

# ============================================================
# SELECT CHANNELS HERE
# ============================================================

CHANNEL_SETS = {
    # Main example: same four channels used in the earlier probability ensemble.
    "DAPI_WGA_FM1_PHASE": ["DAPI", "WGA", "FM1", "PHASECONTRAST"],

    # Add more combinations if useful; they reuse the same cached probabilities.
    "FLUORESCENCE_ONLY": ["DAPI", "WGA", "FM1"],
    "DAPI_FM1": ["DAPI", "FM1"],
    "ALL_5": ["PHASECONTRAST", "BF", "DAPI", "WGA", "FM1"],
}

# Exact Trillian five-channel crop order.
CHANNEL_TO_INDEX = {
    "PHASECONTRAST": 0,
    "BF": 1,
    "DAPI": 2,
    "WGA": 3,
    "FM1": 4,
}

# Common folder aliases used by your existing weight directories.
CHANNEL_FOLDER_ALIASES = {
    "PHASECONTRAST": ["PHASECONTRAST", "PHASE_CONTRAST", "PHASE", "PC"],
    "BF": ["BF", "BRIGHTFIELD", "BRIGHT_FIELD"],
    "DAPI": ["DAPI"],
    "WGA": ["WGA", "WGA647", "WGA_647"],
    "FM1": ["FM1", "FM1-43", "FM1_43", "FM143"],
}

# ============================================================
# HOLDOUT CSV / ARRAY SETTINGS
# ============================================================

# Leave as None for auto-detection.
CSV_PATH_COLUMN = None
CSV_LABEL_COLUMN = None

# If paths stored in the CSV point to an old machine/location, add replacements here.
PATH_REMAP = {
    # Example:
    # r"/scratch/phys3115/bacteria_data/Trillian_DATASET_with_holdout": str(DATA_ROOT),
}

EXPECTED_CROP_CHANNELS = 5
INPUT_SIZE = 96

# ============================================================
# EVALUATION SETTINGS
# ============================================================

BATCH_SIZE = 512
NUM_WORKERS = 4
USE_AMP = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Save full N x C probability matrices for ensemble outputs.
# They are useful for later aggregation analyses but can be large.
SAVE_ENSEMBLE_PROBABILITIES = True

# Individual model/channel probability matrices are larger in total.
SAVE_INDIVIDUAL_PROBABILITIES = False

OUTPUT_DIR = Path(
    r"/data/phys-deep-learning-bacteria/phys3115/Bacteria-data-15 species/Trillian/probability_ensemble_results"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional fallback only if a checkpoint does not contain class metadata.
# Normally leave as None so class order comes directly from the saved checkpoints.
MANUAL_CLASS_NAMES = None
# Example:
# MANUAL_CLASS_NAMES = ["ACAA", "ECOL", "ENTC", "KLAE", "KLPN", "PSAR", "PSMA",
#                       "SAENA", "SAUR", "SEPI", "SERM", "SLUG", "STCP"]

print("Device:", DEVICE)
print("Models:", MODELS_TO_RUN)
print("Channel sets:", CHANNEL_SETS)
print("Output:", OUTPUT_DIR)


# ==============================================================================
# 3. Helpers: channels, checkpoint discovery and model reconstruction
# ==============================================================================


CHECKPOINT_EXTENSIONS = {".pt", ".pth", ".ckpt", ".bin"}

def canonical_token(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())

def canonical_channel(name: str) -> str:
    token = canonical_token(name)
    for channel, aliases in CHANNEL_FOLDER_ALIASES.items():
        if token == canonical_token(channel):
            return channel
        if any(token == canonical_token(alias) for alias in aliases):
            return channel
    raise ValueError(
        f"Unknown channel {name!r}. Valid canonical channels: {list(CHANNEL_TO_INDEX)}"
    )

# Canonicalise user channel sets now.
CHANNEL_SETS = {
    name: [canonical_channel(ch) for ch in channels]
    for name, channels in CHANNEL_SETS.items()
}
for name, channels in CHANNEL_SETS.items():
    if not channels:
        raise ValueError(f"Channel set {name!r} is empty.")
    if len(set(channels)) != len(channels):
        raise ValueError(f"Channel set {name!r} contains duplicates: {channels}")

REQUIRED_CHANNELS = []
for channels in CHANNEL_SETS.values():
    for ch in channels:
        if ch not in REQUIRED_CHANNELS:
            REQUIRED_CHANNELS.append(ch)

print("Required channels:", REQUIRED_CHANNELS)


def checkpoint_priority(path: Path) -> Tuple[int, float]:
    name = path.name.lower()
    score = 0
    if name == "best_model.pt":
        score += 1000
    elif "best_model" in name:
        score += 800
    elif "best" in name:
        score += 100

    if "macro_f1" in name or "macrof1" in name:
        score += 30
    if "val_f1" in name or "valf1" in name:
        score += 20
    if "latest" in name or "last" in name:
        score -= 20

    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return score, mtime


def resolve_checkpoint(source: Path) -> Path:
    source = Path(source).expanduser()

    if source.is_file():
        return source

    if not source.exists():
        raise FileNotFoundError(f"Weight source does not exist: {source}")

    candidates = [
        p for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in CHECKPOINT_EXTENSIONS
    ]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found below: {source}")

    candidates.sort(key=checkpoint_priority, reverse=True)
    return candidates[0]


def find_channel_folder(root: Path, channel: str) -> Path:
    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Model weight root does not exist: {root}")

    wanted = {canonical_token(channel)}
    wanted.update(canonical_token(x) for x in CHANNEL_FOLDER_ALIASES[channel])

    # Prefer immediate child folders.
    immediate = [
        p for p in root.iterdir()
        if p.is_dir() and canonical_token(p.name) in wanted
    ]
    if immediate:
        immediate.sort(key=lambda p: len(str(p)))
        return immediate[0]

    # Then search recursively for a folder whose name matches the channel alias.
    recursive = [
        p for p in root.rglob("*")
        if p.is_dir() and canonical_token(p.name) in wanted
    ]
    if recursive:
        recursive.sort(key=lambda p: (len(p.parts), len(str(p))))
        return recursive[0]

    # Final fallback: find checkpoints where one path component matches an alias.
    candidates = []
    for p in root.rglob("*"):
        if not (p.is_file() and p.suffix.lower() in CHECKPOINT_EXTENSIONS):
            continue
        parts = {canonical_token(part) for part in p.parts}
        if parts.intersection(wanted):
            candidates.append(p)

    if candidates:
        candidates.sort(key=checkpoint_priority, reverse=True)
        return candidates[0]

    raise FileNotFoundError(
        f"Could not find a folder/checkpoint for channel {channel} below {root}"
    )


def load_checkpoint(path: Path) -> Dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)

    # Normal ARC checkpoint.
    if isinstance(obj, Mapping) and "model_state_dict" in obj:
        return dict(obj)

    # Accept common alternative key.
    if isinstance(obj, Mapping) and "state_dict" in obj:
        out = dict(obj)
        out["model_state_dict"] = out["state_dict"]
        return out

    # Raw state_dict fallback.
    if isinstance(obj, Mapping) and obj and all(torch.is_tensor(v) for v in obj.values()):
        return {"model_state_dict": dict(obj)}

    raise TypeError(
        f"{path} is not a recognised checkpoint dictionary/state_dict."
    )


def strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if state_dict and all(str(k).startswith("module.") for k in state_dict):
        return {str(k)[7:]: v for k, v in state_dict.items()}
    return dict(state_dict)


def infer_dropout(checkpoint: Mapping[str, Any]) -> float:
    candidates = [
        checkpoint.get("best_dropout"),
        checkpoint.get("dropout"),
        (checkpoint.get("best_params") or {}).get("dropout")
            if isinstance(checkpoint.get("best_params"), Mapping) else None,
        (checkpoint.get("hyperparameters") or {}).get("dropout")
            if isinstance(checkpoint.get("hyperparameters"), Mapping) else None,
        (checkpoint.get("config") or {}).get("dropout")
            if isinstance(checkpoint.get("config"), Mapping) else None,
    ]
    for value in candidates:
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return 0.0


def derive_class_names(checkpoint: Mapping[str, Any]) -> Optional[List[str]]:
    for key in ("class_names", "classes", "all_classes"):
        value = checkpoint.get(key)
        if value is not None:
            return [str(x).upper() for x in list(value)]

    mapping = checkpoint.get("class_to_idx")
    if isinstance(mapping, Mapping) and mapping:
        ordered = sorted(
            ((int(index), str(name).upper()) for name, index in mapping.items()),
            key=lambda x: x[0],
        )
        return [name for _, name in ordered]

    idx_mapping = checkpoint.get("idx_to_class")
    if isinstance(idx_mapping, Mapping) and idx_mapping:
        ordered = sorted(
            ((int(index), str(name).upper()) for index, name in idx_mapping.items()),
            key=lambda x: x[0],
        )
        return [name for _, name in ordered]

    return None


def normalisation_config_from_checkpoint(
    checkpoint: Mapping[str, Any],
    channel_name: str,
) -> Dict[str, Any]:
    stats = checkpoint.get("normalization_stats", {}) or {}
    mode = str(
        checkpoint.get(
            "normalization_mode",
            stats.get("normalization_mode", "per_crop_zscore"),
        )
    ).lower()

    if mode in {"per_crop_zscore", "per_image_zscore", "zscore"}:
        return {"mode": "per_crop_zscore", "eps": 1e-6}

    if mode in {"percentile_minmax", "per_crop_percentile", "percentile"}:
        return {
            "mode": "per_crop_percentile",
            "percentile_low": float(checkpoint.get("percentile_low", stats.get("percentile_low", 1.0))),
            "percentile_high": float(checkpoint.get("percentile_high", stats.get("percentile_high", 99.0))),
            "eps": 1e-6,
        }

    if mode in {"train_global_zscore", "global_zscore"}:
        means = stats.get("mean", checkpoint.get("train_mean"))
        stds = stats.get("std", checkpoint.get("train_std"))

        def first_number(value):
            if isinstance(value, (list, tuple, np.ndarray)):
                return float(value[0])
            return float(value)

        if means is None or stds is None:
            raise ValueError(
                f"{channel_name}: checkpoint requests global z-score but mean/std are missing."
            )
        return {
            "mode": "global_zscore",
            "mean": first_number(means),
            "std": first_number(stds),
            "eps": 1e-6,
        }

    if mode in {"none", "identity", "raw"}:
        return {"mode": "none"}

    raise ValueError(
        f"{channel_name}: unsupported checkpoint normalisation mode {mode!r}"
    )


def clone_conv_with_input_channels(old: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups if old.groups == 1 else old.groups,
        bias=(old.bias is not None),
        padding_mode=old.padding_mode,
    )


def build_single_channel_classifier(
    architecture: str,
    num_classes: int,
    dropout: float = 0.0,
) -> nn.Module:
    architecture = architecture.lower()

    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        model.conv1 = clone_conv_with_input_channels(model.conv1, 1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if dropout > 0:
            in_features = model.fc.in_features
            model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        return model

    if architecture == "resnet34":
        model = models.resnet34(weights=None)
        model.conv1 = clone_conv_with_input_channels(model.conv1, 1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if dropout > 0:
            in_features = model.fc.in_features
            model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        return model

    if architecture == "densenet121":
        model = models.densenet121(weights=None)
        model.features.conv0 = clone_conv_with_input_channels(model.features.conv0, 1)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        if dropout > 0:
            in_features = model.classifier.in_features
            model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        return model

    if architecture == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        model.features[0][0] = clone_conv_with_input_channels(model.features[0][0], 1)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        if dropout > 0:
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(in_features, num_classes)
            )
        return model

    if architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.features[0][0] = clone_conv_with_input_channels(model.features[0][0], 1)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        if dropout > 0:
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(in_features, num_classes)
            )
        return model

    raise ValueError(f"Unsupported architecture: {architecture}")

def main():
    # ==============================================================================
    # 4. Load the holdout CSV and repair paths
    # ==============================================================================


    PATH_COLUMN_CANDIDATES = [
        "crop_path", "path", "file_path", "filepath", "npy_path", "image_path"
    ]
    LABEL_COLUMN_CANDIDATES = [
        "species", "label", "class", "target", "species_code"
    ]

    def detect_column(df: pd.DataFrame, explicit: Optional[str], candidates: Sequence[str], kind: str) -> str:
        if explicit is not None:
            if explicit not in df.columns:
                raise KeyError(f"Configured {kind} column {explicit!r} not found. Columns: {list(df.columns)}")
            return explicit

        lower_lookup = {str(c).lower(): c for c in df.columns}
        for candidate in candidates:
            if candidate.lower() in lower_lookup:
                return lower_lookup[candidate.lower()]

        raise KeyError(
            f"Could not auto-detect {kind} column. Set it manually. Columns: {list(df.columns)}"
        )


    def repair_crop_path(value: Any) -> str:
        raw = str(value).strip().strip('"').strip("'")
        if not raw:
            return raw

        # Manual replacements first.
        remapped = raw
        for old, new in PATH_REMAP.items():
            remapped = remapped.replace(str(old), str(new))
        candidate = Path(remapped).expanduser()
        if candidate.exists():
            return str(candidate)

        # Windows paths stored in CSVs can be handled by normalising separators.
        normalised = remapped.replace("\\", "/")

        # If the old path includes the dataset folder name, preserve only the suffix.
        dataset_name = DATA_ROOT.name
        marker = f"/{dataset_name}/"
        if marker.lower() in normalised.lower():
            idx = normalised.lower().index(marker.lower())
            suffix = normalised[idx + len(marker):]
            candidate = DATA_ROOT / Path(suffix)
            if candidate.exists():
                return str(candidate)

        # Relative path fallback.
        rel = Path(normalised)
        if not rel.is_absolute():
            candidate = DATA_ROOT / rel
            if candidate.exists():
                return str(candidate)

        # Return the best reconstructed candidate even if it does not yet exist;
        # the validation below will show examples.
        if marker.lower() in normalised.lower():
            idx = normalised.lower().index(marker.lower())
            suffix = normalised[idx + len(marker):]
            return str(DATA_ROOT / Path(suffix))

        return str(candidate)


    if not HOLDOUT_CSV.exists():
        raise FileNotFoundError(
            f"HOLDOUT_CSV does not exist: {HOLDOUT_CSV}\n"
            "Edit DATA_ROOT/HOLDOUT_CSV in the configuration cell."
        )

    holdout_raw = pd.read_csv(HOLDOUT_CSV)
    PATH_COL = detect_column(holdout_raw, CSV_PATH_COLUMN, PATH_COLUMN_CANDIDATES, "path")
    LABEL_COL = detect_column(holdout_raw, CSV_LABEL_COLUMN, LABEL_COLUMN_CANDIDATES, "label")

    holdout_df = holdout_raw.copy()
    holdout_df["path"] = holdout_df[PATH_COL].map(repair_crop_path)
    holdout_df["label_name"] = holdout_df[LABEL_COL].astype(str).str.upper().str.strip()

    missing_mask = ~holdout_df["path"].map(lambda x: Path(x).exists())
    missing_count = int(missing_mask.sum())

    print(f"Holdout rows: {len(holdout_df):,}")
    print("Path column:", PATH_COL)
    print("Label column:", LABEL_COL)
    print(f"Missing crop paths after repair: {missing_count:,}")

    if missing_count:
        print(holdout_df.loc[missing_mask, [PATH_COL, "path"]].head(10).to_string(index=False))
        raise FileNotFoundError(
            f"{missing_count:,} holdout crop paths could not be resolved. "
            "Add the old→new prefix to PATH_REMAP, then rerun this cell."
        )

    print(holdout_df[["label_name", "path"]].head().to_string(index=False))


    # ==============================================================================
    # 5. Discover checkpoints and verify class order / metadata
    # ==============================================================================


    checkpoint_registry: Dict[str, Dict[str, Dict[str, Any]]] = {}
    reference_class_names: Optional[List[str]] = (
        [str(x).upper() for x in MANUAL_CLASS_NAMES]
        if MANUAL_CLASS_NAMES is not None else None
    )

    registry_rows = []

    for architecture in MODELS_TO_RUN:
        if architecture not in MODEL_WEIGHT_ROOTS:
            raise KeyError(f"No weight root configured for {architecture}")

        root = MODEL_WEIGHT_ROOTS[architecture]
        checkpoint_registry[architecture] = {}

        for channel in REQUIRED_CHANNELS:
            source = find_channel_folder(root, channel)
            checkpoint_path = resolve_checkpoint(source)
            checkpoint = load_checkpoint(checkpoint_path)

            checkpoint_arch = str(checkpoint.get("model_name", architecture)).lower()
            if checkpoint_arch and checkpoint_arch != architecture.lower():
                raise ValueError(
                    f"{architecture}/{channel}: checkpoint metadata says model_name={checkpoint_arch!r}\n"
                    f"Checkpoint: {checkpoint_path}"
                )

            checkpoint_channels = [
                canonical_channel(x)
                for x in checkpoint.get("selected_channel_names", [])
            ]
            if checkpoint_channels and checkpoint_channels != [channel]:
                raise ValueError(
                    f"{architecture}/{channel}: checkpoint selected_channel_names={checkpoint_channels}, "
                    f"expected {[channel]}\nCheckpoint: {checkpoint_path}"
                )

            checkpoint_indices = checkpoint.get("selected_channels")
            if checkpoint_indices is not None:
                try:
                    indices = [int(x) for x in checkpoint_indices]
                    if indices != [CHANNEL_TO_INDEX[channel]]:
                        raise ValueError(
                            f"{architecture}/{channel}: checkpoint selected_channels={indices}, "
                            f"expected {[CHANNEL_TO_INDEX[channel]]}"
                        )
                except TypeError:
                    pass

            class_names = derive_class_names(checkpoint)
            if class_names is not None:
                if reference_class_names is None:
                    reference_class_names = class_names
                elif class_names != reference_class_names:
                    raise ValueError(
                        f"Class order mismatch in {architecture}/{channel}.\n"
                        f"Reference: {reference_class_names}\n"
                        f"Checkpoint: {class_names}\n"
                        f"File: {checkpoint_path}"
                    )

            dropout = infer_dropout(checkpoint)
            norm_cfg = normalisation_config_from_checkpoint(checkpoint, channel)

            checkpoint_registry[architecture][channel] = {
                "path": checkpoint_path,
                "checkpoint": checkpoint,
                "dropout": dropout,
                "normalisation": norm_cfg,
            }

            registry_rows.append({
                "architecture": architecture,
                "channel": channel,
                "checkpoint": str(checkpoint_path),
                "dropout": dropout,
                "normalisation": norm_cfg["mode"],
                "classes_in_checkpoint": None if class_names is None else len(class_names),
            })

    if reference_class_names is None:
        raise RuntimeError(
            "No class metadata was found in any checkpoint. "
            "Set MANUAL_CLASS_NAMES in the configuration cell."
        )

    CLASS_NAMES = list(reference_class_names)
    CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
    NUM_CLASSES = len(CLASS_NAMES)

    unknown_labels = sorted(set(holdout_df["label_name"]) - set(CLASS_NAMES))
    if unknown_labels:
        raise ValueError(
            f"Holdout labels not present in checkpoint class mapping: {unknown_labels}"
        )

    holdout_df["target"] = holdout_df["label_name"].map(CLASS_TO_INDEX).astype(int)

    registry_df = pd.DataFrame(registry_rows)
    print(registry_df.to_string(index=False))

    print("Class order:")
    print(CLASS_NAMES)
    print("Number of classes:", NUM_CLASSES)

    registry_df.to_csv(OUTPUT_DIR / "resolved_checkpoints.csv", index=False)


    # ==============================================================================
    # 6. Dataset and normalisation
    # ==============================================================================


    def infer_chw(array: np.ndarray) -> np.ndarray:
        array = np.asarray(array)

        if array.ndim == 2:
            return array[None, :, :]

        if array.ndim != 3:
            raise ValueError(f"Expected a 2D/3D crop, found shape {array.shape}")

        # Exact expected channel count gives the safest inference.
        if array.shape[0] == EXPECTED_CROP_CHANNELS:
            return array
        if array.shape[-1] == EXPECTED_CROP_CHANNELS:
            return np.moveaxis(array, -1, 0)

        # Common fallback for 1-channel arrays.
        if array.shape[0] == 1:
            return array
        if array.shape[-1] == 1:
            return np.moveaxis(array, -1, 0)

        raise ValueError(
            f"Could not infer channel axis for shape {array.shape}; "
            f"expected {EXPECTED_CROP_CHANNELS} channels."
        )


    def normalise_plane(plane: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
        image = np.asarray(plane, dtype=np.float32)
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        mode = str(cfg.get("mode", "per_crop_zscore")).lower()
        eps = float(cfg.get("eps", 1e-6))

        if mode == "per_crop_zscore":
            mean = float(image.mean())
            std = float(image.std())
            return (image - mean) / max(std, eps)

        if mode == "global_zscore":
            mean = float(cfg["mean"])
            std = max(float(cfg["std"]), eps)
            return (image - mean) / std

        if mode == "per_crop_percentile":
            low = float(cfg.get("percentile_low", 1.0))
            high = float(cfg.get("percentile_high", 99.0))
            lo, hi = np.percentile(image, [low, high])
            scale = max(float(hi - lo), eps)
            return np.clip((image - lo) / scale, 0.0, 1.0)

        if mode == "none":
            return image

        raise ValueError(f"Unsupported normalisation mode: {mode}")


    class SingleChannelHoldoutDataset(Dataset):
        def __init__(
            self,
            records: pd.DataFrame,
            channel: str,
            normalisation: Mapping[str, Any],
        ):
            self.records = records.reset_index(drop=True)
            self.channel = channel
            self.channel_index = CHANNEL_TO_INDEX[channel]
            self.normalisation = dict(normalisation)

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int):
            row = self.records.iloc[index]
            path = str(row["path"])

            raw = np.load(path, mmap_mode="r", allow_pickle=False)
            chw = infer_chw(raw)

            if chw.shape[0] == 1 and EXPECTED_CROP_CHANNELS > 1:
                raise ValueError(
                    f"{path}: found shape {chw.shape}; this looks like a mask crop, "
                    "not the five-channel image crop."
                )

            if chw.shape[0] != EXPECTED_CROP_CHANNELS:
                raise ValueError(
                    f"{path}: expected {EXPECTED_CROP_CHANNELS} channels, found {chw.shape}"
                )

            image = np.asarray(chw[self.channel_index], dtype=np.float32)
            image = normalise_plane(image, self.normalisation)

            x = torch.from_numpy(image[None, :, :]).float()

            if tuple(x.shape[-2:]) != (INPUT_SIZE, INPUT_SIZE):
                x = F.interpolate(
                    x.unsqueeze(0),
                    size=(INPUT_SIZE, INPUT_SIZE),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            y = int(row["target"])
            return x, y, path


    def make_holdout_loader(channel: str, normalisation: Mapping[str, Any]) -> DataLoader:
        ds = SingleChannelHoldoutDataset(
            holdout_df,
            channel=channel,
            normalisation=normalisation,
        )
        kwargs = dict(
            dataset=ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )
        if NUM_WORKERS > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 2
        return DataLoader(**kwargs)


    # ==============================================================================
    # 7. Prediction and metric helpers
    # ==============================================================================


    def predict_probabilities(
        model: nn.Module,
        loader: DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        model.eval()
        probs_all = []
        targets_all = []
        paths_all: List[str] = []

        amp_enabled = bool(USE_AMP and DEVICE == "cuda")

        with torch.inference_mode():
            for x, y, paths in tqdm(loader, desc="Holdout inference", leave=False):
                x = x.to(DEVICE, non_blocking=True)

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    logits = model(x)

                probs = torch.softmax(logits.float(), dim=1)

                probs_all.append(probs.cpu().numpy())
                targets_all.append(np.asarray(y, dtype=np.int64))
                paths_all.extend(list(paths))

        return (
            np.concatenate(probs_all, axis=0),
            np.concatenate(targets_all, axis=0),
            paths_all,
        )


    def calculate_metrics(targets: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
        pred = probabilities.argmax(axis=1)

        metrics = {
            "accuracy": float(accuracy_score(targets, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(targets, pred)),
            "macro_f1": float(f1_score(targets, pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(targets, pred, average="weighted", zero_division=0)),
        }

        if probabilities.shape[1] >= 2:
            try:
                metrics["top2_accuracy"] = float(
                    top_k_accuracy_score(
                        targets,
                        probabilities,
                        k=2,
                        labels=np.arange(probabilities.shape[1]),
                    )
                )
            except ValueError:
                top2 = np.argpartition(probabilities, -2, axis=1)[:, -2:]
                metrics["top2_accuracy"] = float(
                    np.mean([t in pair for t, pair in zip(targets, top2)])
                )
        else:
            metrics["top2_accuracy"] = 1.0

        return metrics


    def save_confusion_matrices(
        name: str,
        targets: np.ndarray,
        probabilities: np.ndarray,
        output_dir: Path,
    ):
        pred = probabilities.argmax(axis=1)
        cm = confusion_matrix(
            targets,
            pred,
            labels=np.arange(NUM_CLASSES),
        )

        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(
            cm,
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0,
        )

        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")

        pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
            output_dir / f"{safe}_confusion_matrix_counts.csv"
        )
        pd.DataFrame(cm_norm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
            output_dir / f"{safe}_confusion_matrix_row_normalised.csv"
        )

        for matrix, suffix, fmt, vmax in [
            (cm, "counts", "d", None),
            (cm_norm, "row_normalised", ".2f", 1.0),
        ]:
            fig, ax = plt.subplots(figsize=(11, 9))
            im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
            ax.set_xticks(np.arange(NUM_CLASSES))
            ax.set_yticks(np.arange(NUM_CLASSES))
            ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
            ax.set_yticklabels(CLASS_NAMES)
            ax.set_xlabel("Predicted class")
            ax.set_ylabel("True class")
            ax.set_title(f"{name} — {suffix.replace('_', ' ')}")

            # Annotate when the class count is manageable.
            if NUM_CLASSES <= 15:
                threshold = (matrix.max() / 2.0) if matrix.size else 0
                for i in range(NUM_CLASSES):
                    for j in range(NUM_CLASSES):
                        value = matrix[i, j]
                        text = format(value, fmt)
                        ax.text(
                            j, i, text,
                            ha="center", va="center",
                            color="white" if value > threshold else "black",
                            fontsize=7,
                        )

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(output_dir / f"{safe}_confusion_matrix_{suffix}.png", dpi=250)
            plt.close(fig)


    def save_prediction_bundle(
        name: str,
        targets: np.ndarray,
        probabilities: np.ndarray,
        paths: Sequence[str],
        output_dir: Path,
        save_full_probabilities: bool,
    ) -> Dict[str, float]:
        metrics = calculate_metrics(targets, probabilities)
        pred = probabilities.argmax(axis=1)

        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")

        report = classification_report(
            targets,
            pred,
            labels=np.arange(NUM_CLASSES),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        pd.DataFrame(report).T.to_csv(output_dir / f"{safe}_classification_report.csv")

        pred_df = pd.DataFrame({
            "path": list(paths),
            "true_index": targets,
            "true_class": [CLASS_NAMES[i] for i in targets],
            "pred_index": pred,
            "pred_class": [CLASS_NAMES[i] for i in pred],
            "max_probability": probabilities.max(axis=1),
        })
        pred_df.to_csv(output_dir / f"{safe}_predictions.csv", index=False)

        if save_full_probabilities:
            np.savez_compressed(
                output_dir / f"{safe}_probabilities.npz",
                targets=targets,
                probabilities=probabilities.astype(np.float32),
                paths=np.asarray(list(paths), dtype=str),
                class_names=np.asarray(CLASS_NAMES, dtype=str),
            )

        save_confusion_matrices(name, targets, probabilities, output_dir)

        with open(output_dir / f"{safe}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics


    # ==============================================================================
    # 8. Run every individual model × channel once
    # ==============================================================================


    individual_outputs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    individual_metric_rows = []

    reference_targets = None
    reference_paths = None

    for architecture in MODELS_TO_RUN:
        print("\n" + "=" * 90)
        print("ARCHITECTURE:", architecture)
        print("=" * 90)

        individual_outputs[architecture] = {}

        for channel in REQUIRED_CHANNELS:
            info = checkpoint_registry[architecture][channel]
            checkpoint = info["checkpoint"]
            checkpoint_path = info["path"]
            dropout = info["dropout"]
            norm_cfg = info["normalisation"]

            print(f"\n{architecture} / {channel}")
            print("Checkpoint:", checkpoint_path)
            print("Dropout:", dropout)
            print("Normalisation:", norm_cfg)

            model = build_single_channel_classifier(
                architecture=architecture,
                num_classes=NUM_CLASSES,
                dropout=dropout,
            )

            state_dict = strip_module_prefix(checkpoint["model_state_dict"])
            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                raise RuntimeError(
                    f"Strict checkpoint loading failed for {architecture}/{channel}.\n"
                    f"Checkpoint: {checkpoint_path}\n"
                    f"Likely causes: wrong architecture, wrong class count/order, or wrong dropout.\n\n{e}"
                ) from e

            model = model.to(DEVICE)
            loader = make_holdout_loader(channel, norm_cfg)

            probabilities, targets, paths = predict_probabilities(model, loader)

            if probabilities.shape != (len(holdout_df), NUM_CLASSES):
                raise RuntimeError(
                    f"{architecture}/{channel}: probability shape {probabilities.shape}, "
                    f"expected {(len(holdout_df), NUM_CLASSES)}"
                )

            if reference_targets is None:
                reference_targets = targets.copy()
                reference_paths = list(paths)
            else:
                if not np.array_equal(reference_targets, targets):
                    raise RuntimeError(
                        f"{architecture}/{channel}: target order does not match earlier predictions."
                    )
                if reference_paths != list(paths):
                    raise RuntimeError(
                        f"{architecture}/{channel}: sample path order does not match earlier predictions."
                    )

            metrics = calculate_metrics(targets, probabilities)
            print(metrics)

            individual_outputs[architecture][channel] = {
                "probabilities": probabilities,
                "targets": targets,
                "paths": list(paths),
                "metrics": metrics,
                "checkpoint": str(checkpoint_path),
            }

            individual_metric_rows.append({
                "level": "individual_channel",
                "architecture": architecture,
                "channel_set": channel,
                "n_channels": 1,
                **metrics,
                "checkpoint": str(checkpoint_path),
            })

            if SAVE_INDIVIDUAL_PROBABILITIES:
                indiv_dir = OUTPUT_DIR / "individual_model_channels"
                indiv_dir.mkdir(parents=True, exist_ok=True)
                save_prediction_bundle(
                    name=f"{architecture}_{channel}",
                    targets=targets,
                    probabilities=probabilities,
                    paths=paths,
                    output_dir=indiv_dir,
                    save_full_probabilities=True,
                )

            del model, loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    individual_metrics_df = pd.DataFrame(individual_metric_rows)
    individual_metrics_df.to_csv(OUTPUT_DIR / "individual_model_channel_metrics.csv", index=False)

    print(individual_metrics_df.sort_values(["architecture", "macro_f1"], ascending=[True, False]).reset_index(drop=True).to_string(index=False))


    # ==============================================================================
    # 9. Calculate the selected-channel probability ensembles for all three models
    # ==============================================================================


    ensemble_outputs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    ensemble_metric_rows = []

    ensemble_dir = OUTPUT_DIR / "architecture_probability_ensembles"
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    for set_name, channels in CHANNEL_SETS.items():
        print("\n" + "#" * 90)
        print("CHANNEL SET:", set_name, channels)
        print("#" * 90)

        ensemble_outputs[set_name] = {}

        for architecture in MODELS_TO_RUN:
            probability_stack = np.stack(
                [
                    individual_outputs[architecture][channel]["probabilities"]
                    for channel in channels
                ],
                axis=0,
            )
            mean_probabilities = probability_stack.mean(axis=0)

            name = f"{architecture}__{set_name}"
            metrics = save_prediction_bundle(
                name=name,
                targets=reference_targets,
                probabilities=mean_probabilities,
                paths=reference_paths,
                output_dir=ensemble_dir,
                save_full_probabilities=SAVE_ENSEMBLE_PROBABILITIES,
            )

            ensemble_outputs[set_name][architecture] = {
                "probabilities": mean_probabilities,
                "metrics": metrics,
                "channels": list(channels),
            }

            row = {
                "level": "within_architecture_probability_ensemble",
                "architecture": architecture,
                "channel_set": set_name,
                "channels": " + ".join(channels),
                "n_channels": len(channels),
                **metrics,
            }
            ensemble_metric_rows.append(row)

            print(f"{architecture:12s} | {set_name:25s} | {metrics}")

    architecture_ensemble_df = pd.DataFrame(ensemble_metric_rows)
    architecture_ensemble_df.to_csv(
        OUTPUT_DIR / "architecture_probability_ensemble_metrics.csv",
        index=False,
    )

    print(architecture_ensemble_df.sort_values(["channel_set", "macro_f1"], ascending=[True, False]).reset_index(drop=True).to_string(index=False))


    # ==============================================================================
    # 10. Optional ensemble across the three architectures
    # ==============================================================================


    cross_arch_rows = []
    cross_arch_outputs = {}

    cross_dir = OUTPUT_DIR / "cross_architecture_probability_ensembles"
    cross_dir.mkdir(parents=True, exist_ok=True)

    for set_name, channels in CHANNEL_SETS.items():
        architecture_stack = np.stack(
            [
                ensemble_outputs[set_name][architecture]["probabilities"]
                for architecture in MODELS_TO_RUN
            ],
            axis=0,
        )
        mean_probabilities = architecture_stack.mean(axis=0)

        name = f"ALL_MODELS__{set_name}"
        metrics = save_prediction_bundle(
            name=name,
            targets=reference_targets,
            probabilities=mean_probabilities,
            paths=reference_paths,
            output_dir=cross_dir,
            save_full_probabilities=SAVE_ENSEMBLE_PROBABILITIES,
        )

        cross_arch_outputs[set_name] = {
            "probabilities": mean_probabilities,
            "metrics": metrics,
        }

        cross_arch_rows.append({
            "level": "cross_architecture_probability_ensemble",
            "architecture": " + ".join(MODELS_TO_RUN),
            "channel_set": set_name,
            "channels": " + ".join(channels),
            "n_channels": len(channels),
            "n_architectures": len(MODELS_TO_RUN),
            **metrics,
        })

    cross_arch_df = pd.DataFrame(cross_arch_rows)
    cross_arch_df.to_csv(
        OUTPUT_DIR / "cross_architecture_probability_ensemble_metrics.csv",
        index=False,
    )

    print(cross_arch_df.to_string(index=False))


    # ==============================================================================
    # 11. Final comparison table
    # ==============================================================================


    all_rows = []

    if not individual_metrics_df.empty:
        all_rows.extend(individual_metrics_df.to_dict(orient="records"))

    if not architecture_ensemble_df.empty:
        all_rows.extend(architecture_ensemble_df.to_dict(orient="records"))

    if not cross_arch_df.empty:
        all_rows.extend(cross_arch_df.to_dict(orient="records"))

    comparison_df = pd.DataFrame(all_rows)

    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "top2_accuracy",
    ]

    for col in metric_cols:
        if col in comparison_df.columns:
            comparison_df[col] = pd.to_numeric(comparison_df[col], errors="coerce")

    comparison_df = comparison_df.sort_values(
        ["macro_f1", "accuracy"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    comparison_df.to_csv(
        OUTPUT_DIR / "ALL_probability_ensemble_results.csv",
        index=False,
    )

    display_cols = [
        c for c in [
            "level", "architecture", "channel_set", "channels",
            "accuracy", "balanced_accuracy", "macro_f1",
            "weighted_f1", "top2_accuracy"
        ]
        if c in comparison_df.columns
    ]
    print(comparison_df[display_cols].to_string(index=False))

    print("\nBest result by macro-F1:")
    print(comparison_df[display_cols].head(1).to_string(index=False))


    # ==============================================================================
    # 12. Plot architecture-level ensemble comparison
    # ==============================================================================


    if len(architecture_ensemble_df):
        plot_df = architecture_ensemble_df.copy()
        plot_df["label"] = plot_df["architecture"] + " | " + plot_df["channel_set"]

        fig, ax = plt.subplots(figsize=(max(9, len(plot_df) * 1.3), 5))
        ax.bar(plot_df["label"], plot_df["macro_f1"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Holdout macro-F1")
        ax.set_title("Probability ensemble performance by architecture")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "architecture_probability_ensemble_macro_f1.png", dpi=250)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(max(9, len(plot_df) * 1.3), 5))
        ax.bar(plot_df["label"], plot_df["accuracy"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Holdout accuracy")
        ax.set_title("Probability ensemble accuracy by architecture")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "architecture_probability_ensemble_accuracy.png", dpi=250)
        plt.close(fig)


    # ==============================================================================
    # 13. Save a reproducibility summary
    # ==============================================================================


    summary = {
        "holdout_csv": str(HOLDOUT_CSV),
        "n_holdout_cells": int(len(holdout_df)),
        "models_to_run": list(MODELS_TO_RUN),
        "model_weight_roots": {k: str(v) for k, v in MODEL_WEIGHT_ROOTS.items()},
        "channel_sets": CHANNEL_SETS,
        "channel_to_index": CHANNEL_TO_INDEX,
        "class_names": CLASS_NAMES,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "device": DEVICE,
        "resolved_checkpoints": registry_df.to_dict(orient="records"),
        "architecture_ensemble_results": architecture_ensemble_df.to_dict(orient="records"),
        "cross_architecture_results": cross_arch_df.to_dict(orient="records"),
    }

    with open(OUTPUT_DIR / "probability_ensemble_run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("Finished.")
    print("Results saved to:", OUTPUT_DIR)
    print("\nPrimary summary:")
    print("  ", OUTPUT_DIR / "architecture_probability_ensemble_metrics.csv")
    print("Overall summary:")
    print("  ", OUTPUT_DIR / "ALL_probability_ensemble_results.csv")


if __name__ == "__main__":
    main()
