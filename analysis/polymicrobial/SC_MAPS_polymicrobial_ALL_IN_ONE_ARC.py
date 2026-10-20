#!/usr/bin/env python3
"""
SC-MAPS in-silico polymicrobial stress test — all-in-one ARC script
==================================================================

Run on ARC with:

    python -u SC_MAPS_polymicrobial_ALL_IN_ONE_ARC.py

Dataset preparation deliberately follows the same ARC scratch workflow used by
Train_SSRUN_multicondition_benchmark.py:

  1. Check whether scratch already contains a prepared Trillian dataset
     (cell_manifest.csv + splits/).
  2. If it does, reuse it immediately. If .arc_extract_complete is absent,
     create the marker WITHOUT re-extracting.
  3. Only when the prepared dataset is genuinely absent, copy the persistent
     Trillian ZIP from /data to /scratch.
  4. Reuse an existing scratch ZIP when its byte size matches the source.
  5. Extract safely with progress and create .arc_extract_complete.
  6. Keep the scratch ZIP, matching the established workflow, so a later rebuild
     does not require another large /data -> /scratch copy.

After dataset preparation the script generates/reuses Plate-1 DenseNet121
four-channel concat probabilities and runs the complete two-species in-silico
polymicrobial stress test.

These are synthetic mixtures of cell predictions from separately prepared
monocultures and are not experimental validation of genuine polymicrobial
specimens.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import re
import sys
import zlib
import zipfile
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # essential for headless ARC nodes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import densenet121
from sklearn.metrics import roc_auc_score, roc_curve

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


# -----------------------------------------------------------------------------
# Fixed SC-MAPS class definitions
# -----------------------------------------------------------------------------

CLASS_ORDER = [
    "ACAA", "ECOL", "ENTC", "KLAE", "KLPN", "PSAR", "PSMA",
    "SAENA", "SAUR", "SEPI", "SERM", "SLUG", "STCP",
]

PANEL_MAP = {
    "ECOL": "Routine Enterobacterales",
    "KLPN": "Routine Enterobacterales",
    "KLAE": "AmpC/caution Enterobacterales",
    "ENTC": "AmpC/caution Enterobacterales",
    "SERM": "AmpC/caution Enterobacterales",
    "SAENA": "Salmonella",
    "PSAR": "Pseudomonas",
    "ACAA": "Acinetobacter",
    "PSMA": "Stenotrophomonas",
    "SAUR": "S. aureus-like",
    "SLUG": "S. aureus-like",
    "SEPI": "CoNS",
    "STCP": "CoNS",
}

PANELS = [
    "Routine Enterobacterales",
    "AmpC/caution Enterobacterales",
    "Salmonella",
    "Pseudomonas",
    "Acinetobacter",
    "Stenotrophomonas",
    "S. aureus-like",
    "CoNS",
]

GRAM_MAP = {
    "ACAA": "Gram-negative",
    "ECOL": "Gram-negative",
    "ENTC": "Gram-negative",
    "KLAE": "Gram-negative",
    "KLPN": "Gram-negative",
    "PSAR": "Gram-negative",
    "PSMA": "Gram-negative",
    "SAENA": "Gram-negative",
    "SERM": "Gram-negative",
    "SAUR": "Gram-positive",
    "SEPI": "Gram-positive",
    "SLUG": "Gram-positive",
    "STCP": "Gram-positive",
}

SPECIES = CLASS_ORDER.copy()
SPECIES_TO_IDX = {s: i for i, s in enumerate(SPECIES)}
PANEL_TO_IDX = {p: i for i, p in enumerate(PANELS)}

PANEL_MEMBERSHIP = np.zeros((len(SPECIES), len(PANELS)), dtype=np.float64)
for _sp, _panel in PANEL_MAP.items():
    PANEL_MEMBERSHIP[SPECIES_TO_IDX[_sp], PANEL_TO_IDX[_panel]] = 1.0



# -----------------------------------------------------------------------------
# HARD-CODED ARC CONFIGURATION
# -----------------------------------------------------------------------------
#
# This script is intentionally self-contained. Run it simply as:
#
#     python -u SC_MAPS_polymicrobial_ALL_IN_ONE_ARC.py
#
# No SLURM wrapper and no command-line arguments are required.
#

# Source archive on persistent ARC storage.
TRILLIAN_ZIP = Path(
    "/data/phys-deep-learning-bacteria/phys3115/"
    "Bacteria-data-15 species/Trillian/"
    "Trillian_DATASET_with_holdout.zip"
)

# Fast working location on ARC scratch.
SCRATCH_PARENT = Path(
    "/scratch/phys3115/bacteria_data"
)

DATA_ROOT = SCRATCH_PARENT / "Trillian_DATASET_with_holdout"

# The persistent archive is first copied here, then extracted locally in scratch.
SCRATCH_ZIP = SCRATCH_PARENT / "Trillian_DATASET_with_holdout.zip"

HOLDOUT_CSV = DATA_ROOT / "splits" / "holdout_plate1.csv"
EXTRACTION_COMPLETE_MARKER = DATA_ROOT / ".arc_extract_complete"

DENSENET_CONCAT_CHECKPOINT = Path(
    "/data/phys-deep-learning-bacteria/phys3115/"
    "Bacteria-data-15 species/Trillian/"
    "DenseNet_Concat_weights/best_fusion_model.pt"
)

OUTPUT_DIR = Path(
    "/data/phys-deep-learning-bacteria/phys3115/"
    "Bacteria-data-15 species/Trillian/"
    "Polymicrobial_analysis_ARC"
)

# The script will create this automatically from the holdout images + checkpoint.
PREDICTIONS_CSV = OUTPUT_DIR / "holdout_plate1_DenseNet121_concat_cell_probabilities.csv"

# Set True only if you deliberately want to recalculate all Plate-1 predictions.
FORCE_REINFER = False

# Inference settings
INFERENCE_BATCH_SIZE = 256
INFERENCE_NUM_WORKERS = 4
USE_AMP = True

# Canonical stored crop order in SC-MAPS:
# [PHASECONTRAST, BF, DAPI, WGA, FM1]
STORED_CHANNEL_INDEX = {
    "PHASECONTRAST": 0,
    "PHASE": 0,
    "BF": 1,
    "BRIGHTFIELD": 1,
    "DAPI": 2,
    "WGA": 3,
    "WGAAF647": 3,
    "FM1": 4,
    "FM143": 4,
}

# Polymicrobial simulation settings
CELL_COUNTS = [5, 10, 20, 50, 100, 200]
DOMINANT_FRACTIONS = [0.50, 0.75, 0.90, 0.95]
N_REPEATS = 200
MAX_SAMPLE_PAIRS_PER_SPECIES_PAIR = 10   # None = every available sample pair
N_BOOTSTRAP = 2000
RANDOM_SEED = 20260921
SAMPLING_WITH_REPLACEMENT = False
SIMULATE_BOTH_DIRECTIONS = True
SAVE_RAW_REPEAT_LEVEL = False
RESUME_FROM_PAIR_CHECKPOINTS = True
SKIP_PLOTS = False
LOG_LEVEL = "INFO"



# -----------------------------------------------------------------------------
# Dataset extraction to ARC scratch
# -----------------------------------------------------------------------------

def has_prepared_dataset(root: Path) -> bool:
    """Return True for the same prepared-dataset structure used by prior ARC scripts."""
    root = Path(root)
    return (root / "cell_manifest.csv").exists() and (root / "splits").is_dir()


def safe_copy_file(src: Path, dst: Path) -> Path:
    """
    Copy the ZIP from persistent /data to scratch only when needed.

    This deliberately matches the established Trillian ARC workflow: if the
    scratch ZIP already exists with the same byte size, reuse it immediately.
    """
    src = Path(src)
    dst = Path(dst)

    if not src.exists():
        raise FileNotFoundError(f"Source file does not exist: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        logging.info("ZIP already present on scratch with matching size: %s", dst)
        return dst

    if dst.exists():
        logging.warning("Removing mismatched/incomplete scratch ZIP: %s", dst)
        dst.unlink()

    logging.info("Copying ZIP to scratch:")
    logging.info("  from: %s", src)
    logging.info("  to:   %s", dst)
    shutil.copy2(src, dst)

    if dst.stat().st_size != src.stat().st_size:
        raise IOError(
            "ZIP copy size mismatch after copy.\n"
            f"Source bytes: {src.stat().st_size}\n"
            f"Scratch bytes: {dst.stat().st_size}"
        )

    logging.info("Copied ZIP size: %.2f GB", dst.stat().st_size / 1e9)
    return dst


def safe_extract_member(
    zf: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    extract_root: Path,
) -> None:
    """Safely extract one ZIP member, matching the prior ARC implementation."""
    extract_root = Path(extract_root).resolve()
    target_path = (extract_root / member.filename).resolve()

    if not str(target_path).startswith(str(extract_root)):
        raise RuntimeError(f"Unsafe ZIP member path detected: {member.filename}")

    if member.is_dir():
        target_path.mkdir(parents=True, exist_ok=True)
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)

    with zf.open(member, "r") as source, open(target_path, "wb") as destination:
        shutil.copyfileobj(source, destination, length=1024 * 1024)

    perm = member.external_attr >> 16
    if perm:
        try:
            target_path.chmod(perm)
        except Exception:
            pass


def extract_zip(
    zip_path: Path,
    extract_root: Path,
    dataset_root: Path,
    force: bool = False,
) -> Path:
    """
    Extract the Trillian ZIP using the same reuse logic as the benchmark script.

    Crucially, a dataset that already contains cell_manifest.csv + splits/ is
    accepted as prepared even when the sentinel is missing. In that case the
    sentinel is simply added and NO extraction is performed.
    """
    zip_path = Path(zip_path)
    extract_root = Path(extract_root)
    dataset_root = Path(dataset_root)
    sentinel = dataset_root / ".arc_extract_complete"

    if force and dataset_root.exists():
        logging.warning(
            "FORCE_REEXTRACT requested; removing previous scratch dataset: %s",
            dataset_root,
        )
        shutil.rmtree(dataset_root)

    if has_prepared_dataset(dataset_root) and sentinel.exists():
        logging.info(
            "Scratch dataset already extracted and marked complete: %s",
            dataset_root,
        )
        return dataset_root

    if has_prepared_dataset(dataset_root) and not force:
        logging.info(
            "Scratch dataset already looks prepared; adding completion marker "
            "without re-extracting: %s",
            dataset_root,
        )
        sentinel.touch()
        return dataset_root

    if not zip_path.exists():
        raise FileNotFoundError(f"Cannot extract because ZIP does not exist: {zip_path}")

    logging.info("Extracting ZIP on scratch with progress:")
    logging.info("  zip:  %s", zip_path)
    logging.info("  into: %s", extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        total_files = len(members)
        total_gb = sum(m.file_size for m in members) / 1e9
        logging.info(
            "ZIP contains %s entries; uncompressed size ≈ %.2f GB",
            f"{total_files:,}",
            total_gb,
        )

        iterator = (
            tqdm(
                members,
                total=total_files,
                desc="Extracting dataset",
                unit="file",
                dynamic_ncols=True,
                mininterval=2.0,
            )
            if tqdm is not None
            else members
        )

        for i, member in enumerate(iterator, start=1):
            safe_extract_member(zf, member, extract_root)
            if tqdm is None and (i == 1 or i % 1000 == 0 or i == total_files):
                logging.info(
                    "Extracted %s/%s ZIP entries",
                    f"{i:,}",
                    f"{total_files:,}",
                )

    if not has_prepared_dataset(dataset_root):
        candidates = list(extract_root.rglob("cell_manifest.csv"))
        if candidates:
            dataset_root = candidates[0].parent
            logging.info("Detected dataset root after extraction: %s", dataset_root)

    if not has_prepared_dataset(dataset_root):
        raise FileNotFoundError(
            "Extraction finished, but could not find a prepared dataset root "
            "containing cell_manifest.csv and splits/. Checked: "
            f"{dataset_root}"
        )

    (dataset_root / ".arc_extract_complete").touch()
    logging.info("Scratch extraction complete: %s", dataset_root)
    return dataset_root


def resolve_dataset_root(root: Path) -> Path:
    """Resolve an accidentally nested dataset root, as in the prior ARC script."""
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


def ensure_trillian_dataset() -> Path:
    """
    Prepare/reuse Trillian using the same logic as Train_SSRUN_multicondition_benchmark.

    Workflow:
      1. If DATA_ROOT already has cell_manifest.csv + splits/, use it immediately.
      2. If .arc_extract_complete is missing, touch it; do NOT re-extract.
      3. Only if the prepared dataset is absent, copy the ZIP /data -> /scratch.
      4. If a same-size scratch ZIP already exists, reuse it.
      5. Extract with per-file safety/progress.
      6. Resolve nested roots and verify the holdout split.

    The scratch ZIP is intentionally retained, matching the established ARC
    workflow, so a later rebuild does not require another large /data copy.
    """
    global DATA_ROOT, HOLDOUT_CSV, EXTRACTION_COMPLETE_MARKER

    SCRATCH_PARENT.mkdir(parents=True, exist_ok=True)

    # Fast path: a prepared dataset is enough; marker is optional.
    if has_prepared_dataset(DATA_ROOT):
        EXTRACTION_COMPLETE_MARKER = DATA_ROOT / ".arc_extract_complete"
        if not EXTRACTION_COMPLETE_MARKER.exists():
            logging.info(
                "Prepared Trillian dataset already exists in scratch; adding "
                ".arc_extract_complete marker and skipping extraction."
            )
            EXTRACTION_COMPLETE_MARKER.touch()
        else:
            logging.info("Prepared Trillian dataset already exists in scratch: %s", DATA_ROOT)

        DATA_ROOT = resolve_dataset_root(DATA_ROOT)
    else:
        # Dataset genuinely absent: now copy and extract.
        safe_copy_file(TRILLIAN_ZIP, SCRATCH_ZIP)
        DATA_ROOT = extract_zip(
            SCRATCH_ZIP,
            SCRATCH_PARENT,
            DATA_ROOT,
            force=False,
        )
        DATA_ROOT = resolve_dataset_root(DATA_ROOT)

    HOLDOUT_CSV = DATA_ROOT / "splits" / "holdout_plate1.csv"
    EXTRACTION_COMPLETE_MARKER = DATA_ROOT / ".arc_extract_complete"

    if not HOLDOUT_CSV.exists():
        candidates = sorted((DATA_ROOT / "splits").glob("holdout*.csv"))
        if candidates:
            HOLDOUT_CSV = candidates[0]
            logging.warning(
                "holdout_plate1.csv not found; using first holdout-like CSV: %s",
                HOLDOUT_CSV,
            )
        else:
            raise FileNotFoundError(f"No holdout CSV found in {DATA_ROOT / 'splits'}")

    logging.info("Resolved Trillian DATA_ROOT: %s", DATA_ROOT)
    logging.info("Resolved HOLDOUT_CSV: %s", HOLDOUT_CSV)
    return DATA_ROOT


# -----------------------------------------------------------------------------
# DenseNet121 four-branch concatenation inference
# -----------------------------------------------------------------------------

def _norm_channel_name(name: str) -> str:
    """Normalize checkpoint/channel labels to the SC-MAPS canonical names."""
    text = re.sub(r"[^A-Za-z0-9]+", "", str(name)).upper()
    aliases = {
        "PHASECONTRAST": "PHASECONTRAST",
        "PHASE": "PHASECONTRAST",
        "PHASECONTRASTIMAGE": "PHASECONTRAST",
        "BRIGHTFIELD": "BF",
        "BF": "BF",
        "DAPI": "DAPI",
        "WGA": "WGA",
        "WGAAF647": "WGA",
        "WGA647": "WGA",
        "FM1": "FM1",
        "FM143": "FM1",
        "FM1_43": "FM1",
    }
    return aliases.get(text, text)


def _state_dict_from_checkpoint(ckpt):
    """Extract and clean a state_dict from the saved training checkpoint."""
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                sd = ckpt[key]
                break
        else:
            # Raw state_dict checkpoints are also dictionaries.
            if all(torch.is_tensor(v) for v in ckpt.values()):
                sd = ckpt
            else:
                raise KeyError(
                    "Could not find model_state_dict/state_dict in checkpoint."
                )
    else:
        raise TypeError("Checkpoint is not a dictionary.")

    cleaned = {}
    for k, v in sd.items():
        nk = k[7:] if k.startswith("module.") else k
        cleaned[nk] = v
    return cleaned


def _checkpoint_class_order(ckpt, state_dict):
    """Recover the exact class order used by the trained model."""
    if isinstance(ckpt, dict):
        class_to_idx = ckpt.get("class_to_idx")
        if isinstance(class_to_idx, dict) and class_to_idx:
            return [
                k for k, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])
            ]

        for key in ("classes", "class_names", "species"):
            vals = ckpt.get(key)
            if isinstance(vals, (list, tuple)) and len(vals) == len(CLASS_ORDER):
                return list(vals)

    # Final fallback is the known SC-MAPS order.
    return CLASS_ORDER.copy()


def _infer_branch_names(ckpt, state_dict):
    """
    Recover the exact ModuleDict branch keys used by the checkpoint.
    The known model uses DAPI + PhaseContrast + WGA + FM1.
    """
    # Prefer explicit checkpoint metadata.
    if isinstance(ckpt, dict):
        for key in (
            "selected_channels", "selected_channel_names",
            "channel_names", "channels"
        ):
            vals = ckpt.get(key)
            if isinstance(vals, (list, tuple)) and len(vals) == 4:
                # Metadata may be canonical labels while state_dict uses a slightly
                # different spelling. We'll reconcile against the state_dict below.
                metadata_names = [str(x) for x in vals]
                break
        else:
            metadata_names = None
    else:
        metadata_names = None

    state_names = []
    for k in state_dict:
        m = re.match(r"^branches\.([^.]+)\.", k)
        if m:
            name = m.group(1)
            if name not in state_names:
                state_names.append(name)

    if len(state_names) != 4:
        raise RuntimeError(
            "Expected four fusion branches in checkpoint, but found "
            f"{state_names}. First keys: {list(state_dict)[:20]}"
        )

    if metadata_names:
        ordered = []
        for meta in metadata_names:
            target = _norm_channel_name(meta)
            matches = [s for s in state_names if _norm_channel_name(s) == target]
            if len(matches) == 1:
                ordered.append(matches[0])
        if len(ordered) == 4 and len(set(ordered)) == 4:
            return ordered

    # Fallback to the known biological channel order used for the concat model.
    wanted = ["DAPI", "PHASECONTRAST", "WGA", "FM1"]
    ordered = []
    for target in wanted:
        matches = [s for s in state_names if _norm_channel_name(s) == target]
        if len(matches) != 1:
            raise RuntimeError(
                f"Could not uniquely map checkpoint branches {state_names} "
                f"to required channel {target}."
            )
        ordered.append(matches[0])
    return ordered


def _infer_fusion_dimensions(state_dict, branch_names):
    """Infer projection and classifier dimensions directly from checkpoint weights."""
    first_branch = branch_names[0]

    proj_key = f"projections.{first_branch}.0.weight"
    if proj_key not in state_dict:
        candidates = [
            k for k in state_dict
            if k.startswith(f"projections.{first_branch}.")
            and k.endswith(".weight")
            and state_dict[k].ndim == 2
        ]
        if not candidates:
            raise RuntimeError(
                f"Could not infer projection layer for branch {first_branch}."
            )
        proj_key = candidates[0]

    projection_dim = int(state_dict[proj_key].shape[0])
    branch_feature_dim = int(state_dict[proj_key].shape[1])

    cls_linear_keys = [
        k for k in state_dict
        if k.startswith("classifier.")
        and k.endswith(".weight")
        and state_dict[k].ndim == 2
    ]
    cls_linear_keys = sorted(
        cls_linear_keys,
        key=lambda x: [int(t) if t.isdigit() else t
                       for t in re.split(r"(\d+)", x)]
    )

    if len(cls_linear_keys) < 2:
        raise RuntimeError(
            f"Expected at least two classifier Linear weights, found {cls_linear_keys}"
        )

    first_cls = state_dict[cls_linear_keys[0]]
    last_cls = state_dict[cls_linear_keys[-1]]

    classifier_hidden = int(first_cls.shape[0])
    classifier_input = int(first_cls.shape[1])
    n_classes = int(last_cls.shape[0])

    return {
        "branch_feature_dim": branch_feature_dim,
        "projection_dim": projection_dim,
        "classifier_input": classifier_input,
        "classifier_hidden": classifier_hidden,
        "n_classes": n_classes,
    }


class FourBranchDenseNet121Concat(nn.Module):
    """
    Checkpoint-matched SC-MAPS fusion architecture:

    1-channel DenseNet121 branch per modality
      -> 1024-D DenseNet representation
      -> Linear to 256-D + LayerNorm + GELU + Dropout
      -> concatenate four representations
      -> Linear(1024,512) + LayerNorm + GELU + Dropout
      -> Linear(512,13)
    """

    def __init__(
        self,
        branch_names,
        projection_dim,
        classifier_hidden,
        n_classes,
        dropout,
    ):
        super().__init__()

        self.branch_names = list(branch_names)

        self.branches = nn.ModuleDict()
        self.projections = nn.ModuleDict()

        for name in self.branch_names:
            backbone = densenet121(weights=None)

            # Each modality branch receives one channel.
            old = backbone.features.conv0
            backbone.features.conv0 = nn.Conv2d(
                1,
                old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=False,
            )

            # DenseNet forward now returns its 1024-D pooled feature vector.
            backbone.classifier = nn.Identity()
            self.branches[name] = backbone

            self.projections[name] = nn.Sequential(
                nn.Linear(1024, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        fused_dim = projection_dim * len(self.branch_names)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, classifier_hidden),
            nn.LayerNorm(classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, n_classes),
        )

    def forward(self, channel_tensor_dict):
        projected = []
        for name in self.branch_names:
            x = channel_tensor_dict[name]
            feat = self.branches[name](x)
            feat = self.projections[name](feat)
            projected.append(feat)

        fused = torch.cat(projected, dim=1)
        return self.classifier(fused)


def _load_concat_model(checkpoint_path: Path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"DenseNet concat checkpoint not found:\n{checkpoint_path}"
        )

    logging.info("Loading fusion checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _state_dict_from_checkpoint(ckpt)

    branch_names = _infer_branch_names(ckpt, state_dict)
    dims = _infer_fusion_dimensions(state_dict, branch_names)
    class_order = _checkpoint_class_order(ckpt, state_dict)

    if dims["n_classes"] != len(class_order):
        raise RuntimeError(
            f"Checkpoint classifier has {dims['n_classes']} outputs but "
            f"class metadata contains {len(class_order)} classes."
        )

    dropout = 0.30
    if isinstance(ckpt, dict):
        for key in ("best_dropout", "dropout", "fusion_dropout"):
            if key in ckpt:
                try:
                    dropout = float(ckpt[key])
                    break
                except Exception:
                    pass

    model = FourBranchDenseNet121Concat(
        branch_names=branch_names,
        projection_dim=dims["projection_dim"],
        classifier_hidden=dims["classifier_hidden"],
        n_classes=dims["n_classes"],
        dropout=dropout,
    )

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        logging.error("Strict checkpoint loading failed.")
        logging.error("Checkpoint branch names: %s", branch_names)
        logging.error("Inferred dimensions: %s", dims)
        raise RuntimeError(
            "The saved concat checkpoint did not exactly match the reconstructed "
            "four-branch DenseNet121 architecture. The original error was:\n"
            f"{exc}"
        ) from exc

    model.to(device)
    model.eval()

    logging.info("Checkpoint branches: %s", branch_names)
    logging.info("Checkpoint classes: %s", class_order)
    logging.info("Fusion dimensions: %s", dims)
    logging.info("Dropout metadata: %.4f", dropout)

    return model, branch_names, class_order


def _resolve_crop_path(value) -> Path:
    """
    Resolve crop paths in the holdout CSV against the known ARC Trillian root.

    This handles:
      - correct absolute ARC paths,
      - relative paths,
      - paths created on a different machine but containing the
        Trillian_DATASET_with_holdout marker.
    """
    p = Path(str(value))

    if p.exists():
        return p

    if not p.is_absolute():
        candidate = DATA_ROOT / p
        if candidate.exists():
            return candidate

    text = str(value).replace("\\", "/")
    marker = "Trillian_DATASET_with_holdout/"
    if marker in text:
        suffix = text.split(marker, 1)[1]
        candidate = DATA_ROOT / Path(suffix)
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve crop path from holdout CSV:\n{value}\n"
        f"Expected dataset root: {DATA_ROOT}"
    )


def _parse_sample_id_from_path(path_text: str, species: str) -> str:
    """
    Reproduce the Plate-1 imaging-series grouping used in the prior
    within-sample/aggregation analyses.

    Example:
      ACAA_ACAA_plate1_rep1_series001_T000_Z000_cell_0001.npy
      -> ACAA__plate1__rep1__series001
    """
    stem = Path(str(path_text)).name

    def grab(pattern, default):
        m = re.search(pattern, stem, flags=re.I)
        return m.group(1) if m else default

    plate = grab(r"plate[_-]?(\d+)", "1")
    rep = grab(r"rep[_-]?(\d+)", "1")
    series = grab(r"series[_-]?(\d+)", None)

    if series is None:
        # Preserve uniqueness if a legacy filename lacks "series".
        # Remove only the cell-specific tail where possible.
        base = re.sub(r"_cell[_-]?\d+.*$", "", Path(stem).stem, flags=re.I)
        return f"{species}__plate{plate}__rep{rep}__{base}"

    return f"{species}__plate{plate}__rep{rep}__series{int(series):03d}"


def _detect_holdout_columns(df):
    species_col = None
    for c in ("species", "true_species", "label", "target"):
        if c in df.columns:
            species_col = c
            break
    if species_col is None:
        raise ValueError(
            f"Could not find species column in {HOLDOUT_CSV}. "
            f"Columns: {list(df.columns)}"
        )

    crop_col = None
    for c in ("crop_path", "npy_path", "path", "file_path", "mask_crop_path"):
        if c in df.columns:
            crop_col = c
            break
    if crop_col is None:
        raise ValueError(
            f"Could not find crop-path column in {HOLDOUT_CSV}. "
            f"Columns: {list(df.columns)}"
        )

    return species_col, crop_col


class HoldoutFusionDataset(Dataset):
    def __init__(self, frame, crop_col, branch_names):
        self.frame = frame.reset_index(drop=True)
        self.crop_col = crop_col
        self.branch_names = list(branch_names)

        self.branch_crop_indices = {}
        for name in self.branch_names:
            norm = _norm_channel_name(name)
            if norm not in STORED_CHANNEL_INDEX:
                raise KeyError(
                    f"Unknown fusion branch/channel {name!r} -> {norm!r}. "
                    f"Known channels: {sorted(STORED_CHANNEL_INDEX)}"
                )
            self.branch_crop_indices[name] = STORED_CHANNEL_INDEX[norm]

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        p = _resolve_crop_path(row[self.crop_col])
        arr = np.load(p, allow_pickle=False)

        if arr.ndim != 3:
            raise ValueError(f"Expected 3-D crop array, got {arr.shape} for {p}")

        # Canonical storage is C,H,W. Accept H,W,C defensively.
        if arr.shape[0] == 5:
            chw = arr
        elif arr.shape[-1] == 5:
            chw = np.moveaxis(arr, -1, 0)
        else:
            raise ValueError(
                f"Expected five-channel crop, got shape {arr.shape} for {p}"
            )

        channels = {}
        for name in self.branch_names:
            ci = self.branch_crop_indices[name]
            x = chw[ci].astype(np.float32, copy=False)

            # Per-crop, per-channel z-score used in SC-MAPS training.
            mean = float(x.mean())
            std = float(x.std())
            if std < 1e-6:
                std = 1.0
            x = (x - mean) / std

            channels[name] = torch.from_numpy(x[None, ...].copy())

        return channels, idx


def _collate_holdout(batch):
    channel_names = list(batch[0][0].keys())
    stacked = {
        name: torch.stack([item[0][name] for item in batch], dim=0)
        for name in channel_names
    }
    indices = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return stacked, indices


def ensure_holdout_predictions() -> Path:
    """
    Generate Plate-1 per-cell species probabilities if they do not already exist.
    This makes the polymicrobial analysis genuinely one-script on ARC.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if PREDICTIONS_CSV.exists() and not FORCE_REINFER:
        logging.info(
            "Reusing existing Plate-1 probability file: %s", PREDICTIONS_CSV
        )
        return PREDICTIONS_CSV

    if not HOLDOUT_CSV.exists():
        raise FileNotFoundError(
            f"Known Plate-1 holdout CSV is missing:\n{HOLDOUT_CSV}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Inference device: %s", device)

    model, branch_names, model_class_order = _load_concat_model(
        DENSENET_CONCAT_CHECKPOINT, device
    )

    holdout = pd.read_csv(HOLDOUT_CSV)
    species_col, crop_col = _detect_holdout_columns(holdout)

    holdout = holdout[holdout[species_col].astype(str).isin(model_class_order)].copy()
    holdout = holdout.reset_index(drop=True)

    # Use an existing explicit sample column only if available; otherwise reproduce
    # the imaging-series grouping used in the previous aggregation/consistency work.
    explicit_sample_col = None
    for c in (
        "sample_id", "sample", "sample_name", "experimental_sample",
        "source_sample"
    ):
        if c in holdout.columns:
            explicit_sample_col = c
            break

    if explicit_sample_col is not None:
        sample_ids = holdout[explicit_sample_col].astype(str)
        logging.info("Using existing holdout sample column: %s", explicit_sample_col)
    else:
        sample_ids = [
            _parse_sample_id_from_path(path, str(sp))
            for path, sp in zip(holdout[crop_col], holdout[species_col])
        ]
        logging.info(
            "No explicit sample column found; parsed Plate/rep/series sample IDs "
            "from crop filenames."
        )

    dataset = HoldoutFusionDataset(holdout, crop_col, branch_names)
    loader = DataLoader(
        dataset,
        batch_size=INFERENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=INFERENCE_NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(INFERENCE_NUM_WORKERS > 0),
        collate_fn=_collate_holdout,
    )

    all_probs = np.empty(
        (len(dataset), len(model_class_order)),
        dtype=np.float32
    )

    logging.info(
        "Generating probabilities for %s Plate-1 cells...", f"{len(dataset):,}"
    )

    use_amp_now = USE_AMP and device.type == "cuda"

    with torch.inference_mode():
        for batch_i, (channel_dict, row_idx) in enumerate(loader, start=1):
            channel_dict = {
                name: tensor.to(device, non_blocking=True)
                for name, tensor in channel_dict.items()
            }

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp_now,
            ):
                logits = model(channel_dict)

            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            all_probs[row_idx.numpy()] = probs

            if batch_i % 100 == 0 or batch_i == len(loader):
                logging.info(
                    "Inference batch %d/%d (%s/%s cells)",
                    batch_i,
                    len(loader),
                    f"{min(batch_i * INFERENCE_BATCH_SIZE, len(dataset)):,}",
                    f"{len(dataset):,}",
                )

    out = pd.DataFrame({
        "true_species": holdout[species_col].astype(str).to_numpy(),
        "sample_id": np.asarray(sample_ids, dtype=str),
        "crop_path": holdout[crop_col].astype(str).to_numpy(),
    })

    for optional_col in ("cell_id", "plate", "replicate", "field_id"):
        if optional_col in holdout.columns:
            out[optional_col] = holdout[optional_col].to_numpy()

    for j, sp in enumerate(model_class_order):
        out[f"prob_{sp}"] = all_probs[:, j]

    out["pred_species"] = [
        model_class_order[i] for i in np.argmax(all_probs, axis=1)
    ]
    out["pred_confidence"] = np.max(all_probs, axis=1)

    out.to_csv(PREDICTIONS_CSV, index=False)

    logging.info(
        "Saved Plate-1 cell probabilities: %s", PREDICTIONS_CSV
    )
    logging.info(
        "Source sample/image-series units: %d", out["sample_id"].nunique()
    )

    return PREDICTIONS_CSV


def build_arc_config():
    """
    Build the analysis configuration expected by the simulation functions below.
    Nothing needs to be supplied at the command line.
    """
    return SimpleNamespace(
        predictions_csv=PREDICTIONS_CSV,
        output_dir=OUTPUT_DIR,
        true_species_col="true_species",
        sample_id_col="sample_id",
        prob_map_json=None,
        cell_counts=list(CELL_COUNTS),
        fractions=list(DOMINANT_FRACTIONS),
        n_repeats=int(N_REPEATS),
        max_sample_pairs=(
            0 if MAX_SAMPLE_PAIRS_PER_SPECIES_PAIR is None
            else int(MAX_SAMPLE_PAIRS_PER_SPECIES_PAIR)
        ),
        bootstrap=int(N_BOOTSTRAP),
        seed=int(RANDOM_SEED),
        sampling_with_replacement=bool(SAMPLING_WITH_REPLACEMENT),
        one_direction=not bool(SIMULATE_BOTH_DIRECTIONS),
        save_raw=bool(SAVE_RAW_REPEAT_LEVEL),
        resume=bool(RESUME_FROM_PAIR_CHECKPOINTS),
        skip_plots=bool(SKIP_PLOTS),
        quick_test=False,
        log_level=LOG_LEVEL,
    )


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def stable_seed(base_seed: int, label: str) -> int:
    """Deterministic seed independent of checkpoint/resume order."""
    crc = zlib.crc32(label.encode("utf-8")) & 0xFFFFFFFF
    return (int(base_seed) + crc) % (2**32 - 1)


def setup_logging(output_dir: Path, level: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "polymicrobial_analysis.log"
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)]
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def entropy_bits(p: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0)
    return -(p * np.log2(p)).sum(axis=axis)


def generalized_js_dispersion(prob_matrix: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(prob_matrix, dtype=float)
    x = np.clip(x, eps, 1.0)
    x = x / x.sum(axis=1, keepdims=True)
    m = np.clip(x.mean(axis=0), eps, 1.0)
    kl = (x * (np.log2(x) - np.log2(m))).sum(axis=1)
    return float(kl.mean())


def top_k_labels(probs: np.ndarray, labels: Sequence[str], k: int = 2) -> List[str]:
    idx = np.argsort(probs)[::-1][:k]
    return [labels[i] for i in idx]


def probs_to_panel_probs(species_probs: np.ndarray) -> np.ndarray:
    return np.asarray(species_probs, dtype=float) @ PANEL_MEMBERSHIP


def auto_detect_column(columns: Iterable[str], candidates: Sequence[str], label: str) -> str:
    cols = list(columns)
    lower_map = {str(c).lower(): c for c in cols}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    raise ValueError(
        f"Could not auto-detect {label}. Tried {list(candidates)}. "
        "Please edit the hard-coded configuration/column logic at the top of this script."
    )


def infer_probability_columns(
    df: pd.DataFrame,
    explicit_map: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    if explicit_map is not None:
        missing_species = [s for s in SPECIES if s not in explicit_map]
        if missing_species:
            raise ValueError(f"Probability map is missing species: {missing_species}")
        missing_cols = [explicit_map[s] for s in SPECIES if explicit_map[s] not in df.columns]
        if missing_cols:
            raise ValueError(f"Mapped probability columns are missing: {missing_cols}")
        return {s: explicit_map[s] for s in SPECIES}

    mapping: Dict[str, str] = {}
    for sp in SPECIES:
        candidates = [
            f"prob_{sp}", f"p_{sp}", f"{sp}_prob", f"{sp}_probability",
            f"probability_{sp}", sp,
        ]
        for c in candidates:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                mapping[sp] = c
                break

    if len(mapping) == len(SPECIES):
        return mapping

    indexed: List[Tuple[int, str]] = []
    for c in df.columns:
        m = re.fullmatch(r"(?:prob|p)[_\-]?(\d+)", str(c), flags=re.I)
        if m:
            indexed.append((int(m.group(1)), c))
    indexed = sorted(indexed)

    if len(indexed) == len(SPECIES):
        logging.warning(
            "Using indexed probability columns with CLASS_ORDER. Verify the model output "
            "class order is exactly: %s", CLASS_ORDER,
        )
        return {sp: col for sp, (_, col) in zip(CLASS_ORDER, indexed)}

    raise ValueError(
        "Could not infer all 13 probability columns. Use --prob-map-json to provide an "
        "explicit species->column mapping."
    )


def realised_counts(total_cells: int, dominant_fraction: float) -> Tuple[int, int, float]:
    n_dom = int(round(total_cells * dominant_fraction))
    n_dom = min(max(n_dom, 0), total_cells)
    n_min = total_cells - n_dom
    return n_dom, n_min, n_dom / total_cells


def bootstrap_mean_ci(
    values: Sequence[float], n_boot: int, seed: int
) -> Tuple[float, float, float, int]:
    x = pd.Series(values).dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan, 0
    if len(x) == 1:
        return float(x[0]), float(x[0]), float(x[0]), 1
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return (
        float(x.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
        len(x),
    )


def aggregate_with_bootstrap(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    metric_cols: Sequence[str],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for group_i, (keys, g) in enumerate(df.groupby(list(group_cols), dropna=False)):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for metric_i, metric in enumerate(metric_cols):
            local_seed = seed + group_i * 1009 + metric_i * 9173
            mean, lo, hi, n = bootstrap_mean_ci(g[metric], n_boot, local_seed)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
            row[f"{metric}_n_units"] = n
        rows.append(row)
    return pd.DataFrame(rows)


def savefig(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Simulation state
# -----------------------------------------------------------------------------

class AnalysisState:
    def __init__(
        self,
        work: pd.DataFrame,
        true_species_col: str,
        sample_id_col: str,
        prob_cols: Sequence[str],
        sampling_with_replacement: bool,
        max_sample_pairs: Optional[int],
    ) -> None:
        self.work = work
        self.true_species_col = true_species_col
        self.sample_id_col = sample_id_col
        self.prob_cols = list(prob_cols)
        self.sampling_with_replacement = sampling_with_replacement
        self.max_sample_pairs = max_sample_pairs

        self.sample_arrays: Dict[Tuple[str, str], np.ndarray] = {}
        self.species_samples: Dict[str, List[str]] = defaultdict(list)

        for (sp, sid), g in work.groupby([true_species_col, sample_id_col], sort=False):
            arr = g[self.prob_cols].to_numpy(dtype=np.float32)
            self.sample_arrays[(sp, sid)] = arr
            self.species_samples[sp].append(sid)

        for sp in SPECIES:
            self.species_samples[sp] = sorted(self.species_samples[sp])

    def draw_cells(self, arr: np.ndarray, n: int, rng: np.random.Generator) -> Optional[np.ndarray]:
        if n == 0:
            return np.empty((0, arr.shape[1]), dtype=arr.dtype)
        if (not self.sampling_with_replacement) and n > len(arr):
            return None
        idx = rng.choice(len(arr), size=n, replace=self.sampling_with_replacement)
        return arr[idx]

    def select_sample_pairs(
        self, sp_a: str, sp_b: str, rng: np.random.Generator
    ) -> List[Tuple[str, str]]:
        all_pairs = list(itertools.product(self.species_samples[sp_a], self.species_samples[sp_b]))
        if self.max_sample_pairs is None or len(all_pairs) <= self.max_sample_pairs:
            return all_pairs
        idx = rng.choice(len(all_pairs), size=self.max_sample_pairs, replace=False)
        return [all_pairs[i] for i in idx]


# -----------------------------------------------------------------------------
# Metrics and simulations
# -----------------------------------------------------------------------------

def metrics_for_cell_set(
    cell_probs: np.ndarray,
    true_species_set: Iterable[str],
    dominant_species: Optional[str] = None,
    minority_species: Optional[str] = None,
) -> Dict[str, object]:
    agg = cell_probs.mean(axis=0)
    agg = agg / agg.sum()

    top2 = top_k_labels(agg, SPECIES, k=2)
    top1 = top2[0]

    cell_votes_idx = np.argmax(cell_probs, axis=1)
    vote_counts = np.bincount(cell_votes_idx, minlength=len(SPECIES)).astype(float)
    vote_probs = vote_counts / vote_counts.sum()

    modal_vote_fraction = float(vote_probs.max())
    vote_entropy = float(entropy_bits(vote_probs))
    agg_entropy = float(entropy_bits(agg))
    js_disp = generalized_js_dispersion(cell_probs)

    true_species_set = set(true_species_set)
    true_species_mass = float(sum(agg[SPECIES_TO_IDX[s]] for s in true_species_set))

    panel_probs = probs_to_panel_probs(agg)
    panel_probs = panel_probs / panel_probs.sum()
    top2_panels = top_k_labels(panel_probs, PANELS, k=2)
    top1_panel = top2_panels[0]

    true_panels = {PANEL_MAP[s] for s in true_species_set}
    true_panel_mass = float(sum(panel_probs[PANEL_TO_IDX[p]] for p in true_panels))

    top1_is_true = top1 in true_species_set

    dominant_species_top1 = np.nan
    minority_species_top2 = np.nan
    dual_species_top2 = np.nan

    if dominant_species is not None:
        dominant_species_top1 = float(top1 == dominant_species)
    if minority_species is not None:
        minority_species_top2 = float(minority_species in top2)
    if len(true_species_set) == 2:
        dual_species_top2 = float(set(top2) == true_species_set)

    dominant_panel_top1 = np.nan
    minority_panel_top2 = np.nan
    dual_panel_top2 = np.nan
    same_panel_correct_top1 = np.nan

    if len(true_panels) == 1:
        only_panel = next(iter(true_panels))
        same_panel_correct_top1 = float(top1_panel == only_panel)
    else:
        if dominant_species is not None:
            dominant_panel_top1 = float(top1_panel == PANEL_MAP[dominant_species])
        if minority_species is not None:
            minority_panel_top2 = float(PANEL_MAP[minority_species] in top2_panels)
        dual_panel_top2 = float(set(top2_panels) == true_panels)

    return {
        "top1_species": top1,
        "top2_species_1": top2[0],
        "top2_species_2": top2[1],
        "top1_panel": top1_panel,
        "top2_panel_1": top2_panels[0],
        "top2_panel_2": top2_panels[1],
        "top1_is_true_species": float(top1_is_true),
        "dominant_species_top1": dominant_species_top1,
        "minority_species_top2": minority_species_top2,
        "dual_species_top2": dual_species_top2,
        "dominant_panel_top1": dominant_panel_top1,
        "minority_panel_top2": minority_panel_top2,
        "dual_panel_top2": dual_panel_top2,
        "same_panel_correct_top1": same_panel_correct_top1,
        "modal_vote_fraction": modal_vote_fraction,
        "vote_entropy_bits": vote_entropy,
        "aggregate_entropy_bits": agg_entropy,
        "js_dispersion_bits": js_disp,
        "true_species_probability_mass": true_species_mass,
        "true_panel_probability_mass": true_panel_mass,
    }


def summarise_repeat_metrics(repeat_df: pd.DataFrame, id_fields: Sequence[str]) -> Dict[str, object]:
    out: Dict[str, object] = {k: repeat_df.iloc[0][k] for k in id_fields}

    binary_cols = [
        "top1_is_true_species",
        "dominant_species_top1",
        "minority_species_top2",
        "dual_species_top2",
        "dominant_panel_top1",
        "minority_panel_top2",
        "dual_panel_top2",
        "same_panel_correct_top1",
    ]
    continuous_cols = [
        "modal_vote_fraction",
        "vote_entropy_bits",
        "aggregate_entropy_bits",
        "js_dispersion_bits",
        "true_species_probability_mass",
        "true_panel_probability_mass",
    ]

    for c in binary_cols:
        vals = pd.to_numeric(repeat_df[c], errors="coerce").dropna()
        out[c + "_rate"] = vals.mean() if len(vals) else np.nan

    for c in continuous_cols:
        vals = pd.to_numeric(repeat_df[c], errors="coerce").dropna().to_numpy()
        if len(vals):
            out[c + "_mean"] = float(np.mean(vals))
            out[c + "_p025"] = float(np.quantile(vals, 0.025))
            out[c + "_p975"] = float(np.quantile(vals, 0.975))
        else:
            out[c + "_mean"] = np.nan
            out[c + "_p025"] = np.nan
            out[c + "_p975"] = np.nan

    return out


def simulate_pure_sample(
    state: AnalysisState,
    sp: str,
    sid: str,
    total_cells: int,
    n_repeats: int,
    rng: np.random.Generator,
) -> Tuple[Optional[Dict[str, object]], Optional[pd.DataFrame]]:
    arr = state.sample_arrays[(sp, sid)]
    if (not state.sampling_with_replacement) and total_cells > len(arr):
        return None, None

    repeat_rows = []
    for rep in range(n_repeats):
        chosen = state.draw_cells(arr, total_cells, rng)
        if chosen is None:
            continue
        m = metrics_for_cell_set(chosen, {sp}, dominant_species=sp)
        m["repeat"] = rep
        repeat_rows.append(m)

    if not repeat_rows:
        return None, None

    rep_df = pd.DataFrame(repeat_rows)
    id_values = {
        "condition_type": "pure",
        "species_a": sp,
        "species_b": None,
        "dominant_species": sp,
        "minority_species": None,
        "sample_a": sid,
        "sample_b": None,
        "species_pair": sp,
        "sample_pair_id": f"{sp}:{sid}",
        "requested_dominant_fraction": 1.0,
        "realised_dominant_fraction": 1.0,
        "minority_fraction": 0.0,
        "n_dominant": total_cells,
        "n_minority": 0,
        "total_cells": total_cells,
        "same_panel_pair": True,
        "gram_pair_type": GRAM_MAP[sp],
    }
    for k, v in id_values.items():
        rep_df[k] = v

    summary = summarise_repeat_metrics(rep_df, list(id_values))
    return summary, rep_df


def simulate_mixed_sample_pair(
    state: AnalysisState,
    sp_a: str,
    sid_a: str,
    sp_b: str,
    sid_b: str,
    total_cells: int,
    dominant_fraction: float,
    dominant_is_a: bool,
    n_repeats: int,
    rng: np.random.Generator,
) -> Tuple[Optional[Dict[str, object]], Optional[pd.DataFrame]]:
    arr_a = state.sample_arrays[(sp_a, sid_a)]
    arr_b = state.sample_arrays[(sp_b, sid_b)]

    # 50:50 has no meaningful dominant/minority label.
    if np.isclose(dominant_fraction, 0.50):
        dominant_is_a = True

    n_dom, n_min, realised_dom = realised_counts(total_cells, dominant_fraction)
    if dominant_fraction < 1.0 and (n_dom == 0 or n_min == 0):
        return None, None

    if dominant_is_a:
        dom_sp, dom_arr = sp_a, arr_a
        min_sp, min_arr = sp_b, arr_b
    else:
        dom_sp, dom_arr = sp_b, arr_b
        min_sp, min_arr = sp_a, arr_a

    if not state.sampling_with_replacement:
        if n_dom > len(dom_arr) or n_min > len(min_arr):
            return None, None

    repeat_rows = []
    for rep in range(n_repeats):
        x_dom = state.draw_cells(dom_arr, n_dom, rng)
        x_min = state.draw_cells(min_arr, n_min, rng)
        if x_dom is None or x_min is None:
            continue
        chosen = np.vstack([x_dom, x_min])
        balanced = np.isclose(dominant_fraction, 0.50)
        m = metrics_for_cell_set(
            chosen,
            {sp_a, sp_b},
            dominant_species=None if balanced else dom_sp,
            minority_species=None if balanced else min_sp,
        )
        m["repeat"] = rep
        repeat_rows.append(m)

    if not repeat_rows:
        return None, None

    pair_sorted = tuple(sorted([sp_a, sp_b]))
    species_pair = f"{pair_sorted[0]}__{pair_sorted[1]}"
    sample_pair_id = f"{sp_a}:{sid_a}__{sp_b}:{sid_b}"
    same_panel = PANEL_MAP[sp_a] == PANEL_MAP[sp_b]
    same_gram = GRAM_MAP[sp_a] == GRAM_MAP[sp_b]
    gram_pair_type = f"same {GRAM_MAP[sp_a]}" if same_gram else "mixed Gram status"

    rep_df = pd.DataFrame(repeat_rows)
    id_values = {
        "condition_type": "mixed",
        "species_a": sp_a,
        "species_b": sp_b,
        "dominant_species": None if np.isclose(dominant_fraction, 0.50) else dom_sp,
        "minority_species": None if np.isclose(dominant_fraction, 0.50) else min_sp,
        "sample_a": sid_a,
        "sample_b": sid_b,
        "species_pair": species_pair,
        "sample_pair_id": sample_pair_id,
        "requested_dominant_fraction": dominant_fraction,
        "realised_dominant_fraction": realised_dom,
        "minority_fraction": 1.0 - realised_dom,
        "n_dominant": n_dom,
        "n_minority": n_min,
        "total_cells": total_cells,
        "same_panel_pair": same_panel,
        "gram_pair_type": gram_pair_type,
    }
    for k, v in id_values.items():
        rep_df[k] = v

    summary = summarise_repeat_metrics(rep_df, list(id_values))
    return summary, rep_df


# -----------------------------------------------------------------------------
# Input loading
# -----------------------------------------------------------------------------

def load_input(args: argparse.Namespace) -> Tuple[AnalysisState, str, str, Dict[str, str]]:
    if not args.predictions_csv.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {args.predictions_csv}")

    logging.info("Reading predictions: %s", args.predictions_csv)
    df = pd.read_csv(args.predictions_csv)
    logging.info("Loaded %s rows x %s columns", f"{len(df):,}", df.shape[1])

    true_species_col = args.true_species_col or auto_detect_column(
        df.columns,
        ["true_species", "species", "true_label", "target", "label", "y_true"],
        "true species column",
    )
    sample_id_col = args.sample_id_col or auto_detect_column(
        df.columns,
        ["sample_id", "sample", "source_sample", "sample_name", "field_id", "source_group", "group_id"],
        "sample ID column",
    )

    explicit_map = None
    if args.prob_map_json is not None:
        with open(args.prob_map_json, "r", encoding="utf-8") as f:
            explicit_map = json.load(f)

    prob_map = infer_probability_columns(df, explicit_map)
    prob_cols = [prob_map[s] for s in SPECIES]

    logging.info("True species column: %s", true_species_col)
    logging.info("Sample ID column: %s", sample_id_col)
    logging.info("Probability columns: %s", prob_map)
    if args.sample_id_col is None:
        logging.warning(
            "Sample ID was auto-detected as '%s'. Confirm this matches the biological/sample "
            "unit used in your original aggregation analysis; do not silently use field_id if "
            "the original analysis aggregated a higher-level sample/well.",
            sample_id_col,
        )

    work = df[[true_species_col, sample_id_col] + prob_cols].copy()
    work = work[work[true_species_col].isin(SPECIES)].copy()
    work = work.dropna(subset=[true_species_col, sample_id_col] + prob_cols)

    P = work[prob_cols].to_numpy(dtype=float)
    if np.any(P < 0):
        raise ValueError("Negative probability values detected.")
    row_sums = P.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ValueError("At least one row has probability sum <= 0.")
    logging.info(
        "Probability row sums before normalisation: median=%.6f min=%.6f max=%.6f",
        np.median(row_sums), row_sums.min(), row_sums.max(),
    )
    P /= row_sums[:, None]
    work.loc[:, prob_cols] = P
    work[true_species_col] = work[true_species_col].astype(str)
    work[sample_id_col] = work[sample_id_col].astype(str)

    sample_species_n = work.groupby(sample_id_col)[true_species_col].nunique()
    if (sample_species_n > 1).any():
        bad = sample_species_n[sample_species_n > 1].head(10).to_dict()
        raise ValueError(
            "Some source sample IDs contain >1 true species; source holdout samples must be "
            f"monocultures for this simulation. First examples: {bad}"
        )

    max_sample_pairs = None if args.max_sample_pairs == 0 else args.max_sample_pairs
    state = AnalysisState(
        work=work,
        true_species_col=true_species_col,
        sample_id_col=sample_id_col,
        prob_cols=prob_cols,
        sampling_with_replacement=args.sampling_with_replacement,
        max_sample_pairs=max_sample_pairs,
    )

    rows = []
    for sp in SPECIES:
        samples = state.species_samples[sp]
        counts = [len(state.sample_arrays[(sp, sid)]) for sid in samples]
        rows.append({
            "species": sp,
            "n_cells": int((work[true_species_col] == sp).sum()),
            "n_samples": len(samples),
            "min_cells_per_sample": min(counts) if counts else 0,
            "median_cells_per_sample": float(np.median(counts)) if counts else 0,
            "max_cells_per_sample": max(counts) if counts else 0,
        })
    sample_table = pd.DataFrame(rows)
    sample_table.to_csv(args.output_dir / "source_sample_counts.csv", index=False)
    logging.info("Source sample summary:\n%s", sample_table.to_string(index=False))

    return state, true_species_col, sample_id_col, prob_map


# -----------------------------------------------------------------------------
# Simulation runners with checkpoints
# -----------------------------------------------------------------------------

def run_pure_controls(
    state: AnalysisState,
    args: argparse.Namespace,
) -> pd.DataFrame:
    out_csv = args.output_dir / "pure_control_unit_summary.csv"
    if args.resume and out_csv.exists():
        logging.info("Reusing pure-control checkpoint: %s", out_csv)
        return pd.read_csv(out_csv)

    rows: List[Dict[str, object]] = []
    raw_dir = args.output_dir / "raw_pure"
    if args.save_raw:
        raw_dir.mkdir(exist_ok=True)

    total_jobs = sum(len(state.species_samples[s]) for s in SPECIES) * len(args.cell_counts)
    done = 0
    for sp in SPECIES:
        for sid in state.species_samples[sp]:
            for n_cells in args.cell_counts:
                rng = np.random.default_rng(
                    stable_seed(args.seed, f"pure|{sp}|{sid}|{n_cells}")
                )
                summary, raw = simulate_pure_sample(
                    state, sp, sid, n_cells, args.n_repeats, rng
                )
                done += 1
                if summary is not None:
                    rows.append(summary)
                    if args.save_raw and raw is not None:
                        safe_sid = re.sub(r"[^A-Za-z0-9_.-]+", "_", sid)
                        raw.to_csv(
                            raw_dir / f"pure_{sp}_{safe_sid}_{n_cells}cells.csv.gz",
                            index=False, compression="gzip",
                        )
                if done % 25 == 0 or done == total_jobs:
                    logging.info("Pure controls: %d/%d", done, total_jobs)

    pure_units = pd.DataFrame(rows)
    pure_units.to_csv(out_csv, index=False)
    logging.info("Saved %s pure sample x cell-count units", f"{len(pure_units):,}")
    return pure_units


def run_mixtures(
    state: AnalysisState,
    args: argparse.Namespace,
) -> pd.DataFrame:
    checkpoint_dir = args.output_dir / "pair_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    raw_dir = args.output_dir / "raw_mixtures"
    if args.save_raw:
        raw_dir.mkdir(exist_ok=True)

    species_pairs = list(itertools.combinations(SPECIES, 2))
    if len(species_pairs) != 78:
        raise RuntimeError("Expected 78 unordered species pairs.")

    pair_frames: List[pd.DataFrame] = []

    for pair_i, (sp_a, sp_b) in enumerate(species_pairs, start=1):
        pair_name = f"{sp_a}__{sp_b}"
        checkpoint = checkpoint_dir / f"{pair_name}.csv"

        if args.resume and checkpoint.exists():
            pair_df = pd.read_csv(checkpoint)
            pair_frames.append(pair_df)
            logging.info("[%d/78] Reused checkpoint %s (%d units)", pair_i, pair_name, len(pair_df))
            continue

        pair_rng = np.random.default_rng(stable_seed(args.seed, f"pair-select|{pair_name}"))
        selected_pairs = state.select_sample_pairs(sp_a, sp_b, pair_rng)
        rows: List[Dict[str, object]] = []

        for sample_pair_i, (sid_a, sid_b) in enumerate(selected_pairs):
            for frac in args.fractions:
                directions = [True]
                if (not np.isclose(frac, 0.50)) and (not args.one_direction):
                    directions = [True, False]

                for dominant_is_a in directions:
                    direction_label = "A" if dominant_is_a else "B"
                    for n_cells in args.cell_counts:
                        rng_label = (
                            f"mix|{sp_a}|{sid_a}|{sp_b}|{sid_b}|{frac}|"
                            f"{direction_label}|{n_cells}"
                        )
                        rng = np.random.default_rng(stable_seed(args.seed, rng_label))
                        summary, raw = simulate_mixed_sample_pair(
                            state=state,
                            sp_a=sp_a,
                            sid_a=sid_a,
                            sp_b=sp_b,
                            sid_b=sid_b,
                            total_cells=n_cells,
                            dominant_fraction=frac,
                            dominant_is_a=dominant_is_a,
                            n_repeats=args.n_repeats,
                            rng=rng,
                        )
                        if summary is not None:
                            rows.append(summary)
                            if args.save_raw and raw is not None:
                                raw_name = (
                                    f"{pair_name}_spair{sample_pair_i:03d}_"
                                    f"f{frac:.3f}_{direction_label}_{n_cells}cells.csv.gz"
                                )
                                raw.to_csv(raw_dir / raw_name, index=False, compression="gzip")

        pair_df = pd.DataFrame(rows)
        pair_df.to_csv(checkpoint, index=False)
        pair_frames.append(pair_df)
        logging.info(
            "[%d/78] Completed %s: %d source sample-pairs -> %d condition units",
            pair_i, pair_name, len(selected_pairs), len(pair_df),
        )

    mixed_units = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    mixed_units.to_csv(args.output_dir / "synthetic_mixture_unit_summary.csv", index=False)
    logging.info("Saved %s mixed sample-pair x condition units", f"{len(mixed_units):,}")
    return mixed_units


# -----------------------------------------------------------------------------
# Summaries and figures
# -----------------------------------------------------------------------------

def create_summaries_and_figures(
    pure_units: pd.DataFrame,
    mixed_units: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    if mixed_units.empty:
        raise RuntimeError("No synthetic mixture conditions were generated.")

    coverage = (
        mixed_units.groupby(["total_cells", "requested_dominant_fraction"], dropna=False)
        .agg(
            n_units=("sample_pair_id", "size"),
            n_species_pairs=("species_pair", "nunique"),
            n_sample_pairs=("sample_pair_id", "nunique"),
        )
        .reset_index()
    )
    coverage.to_csv(args.output_dir / "simulation_coverage.csv", index=False)
    logging.info("Simulation coverage:\n%s", coverage.to_string(index=False))

    mixed_metrics = [
        "top1_is_true_species_rate",
        "dominant_species_top1_rate",
        "minority_species_top2_rate",
        "dual_species_top2_rate",
        "dominant_panel_top1_rate",
        "minority_panel_top2_rate",
        "dual_panel_top2_rate",
        "same_panel_correct_top1_rate",
        "modal_vote_fraction_mean",
        "vote_entropy_bits_mean",
        "aggregate_entropy_bits_mean",
        "js_dispersion_bits_mean",
    ]
    global_summary = aggregate_with_bootstrap(
        mixed_units,
        ["total_cells", "requested_dominant_fraction", "realised_dominant_fraction"],
        mixed_metrics,
        args.bootstrap,
        args.seed,
    )
    global_summary.to_csv(args.output_dir / "global_mixture_summary_bootstrap.csv", index=False)

    asym = mixed_units[mixed_units["requested_dominant_fraction"] > 0.50].copy()
    plot_summary = aggregate_with_bootstrap(
        asym,
        ["total_cells", "minority_fraction"],
        ["dominant_species_top1_rate", "minority_species_top2_rate", "dual_species_top2_rate"],
        args.bootstrap,
        args.seed + 11,
    )
    plot_summary.to_csv(args.output_dir / "species_recovery_summary.csv", index=False)

    panel_strat = aggregate_with_bootstrap(
        mixed_units,
        ["same_panel_pair", "total_cells", "requested_dominant_fraction", "minority_fraction"],
        [
            "top1_is_true_species_rate",
            "dual_species_top2_rate",
            "same_panel_correct_top1_rate",
            "dual_panel_top2_rate",
            "minority_panel_top2_rate",
        ],
        args.bootstrap,
        args.seed + 23,
    )
    panel_strat.to_csv(args.output_dir / "same_vs_different_panel_summary.csv", index=False)

    gram_summary = aggregate_with_bootstrap(
        mixed_units,
        ["gram_pair_type", "total_cells", "requested_dominant_fraction", "minority_fraction"],
        ["dual_species_top2_rate", "minority_species_top2_rate", "js_dispersion_bits_mean"],
        args.bootstrap,
        args.seed + 37,
    )
    gram_summary.to_csv(args.output_dir / "gram_pair_stratified_summary.csv", index=False)

    # Mixture heterogeneity AUCs
    heterogeneity_features = {
        "modal_vote_fraction_mean": -1,
        "vote_entropy_bits_mean": +1,
        "aggregate_entropy_bits_mean": +1,
        "js_dispersion_bits_mean": +1,
    }
    auc_rows = []
    for n_cells in args.cell_counts:
        pure_n = pure_units[pure_units["total_cells"] == n_cells].copy()
        mixed_n = mixed_units[mixed_units["total_cells"] == n_cells].copy()
        for feat, direction in heterogeneity_features.items():
            a = pure_n[[feat]].dropna().copy()
            a["is_mixed"] = 0
            b = mixed_n[[feat]].dropna().copy()
            b["is_mixed"] = 1
            d = pd.concat([a, b], ignore_index=True)
            if d.empty or d["is_mixed"].nunique() < 2:
                continue
            score = direction * d[feat].to_numpy()
            auc_rows.append({
                "total_cells": n_cells,
                "feature": feat,
                "auc": roc_auc_score(d["is_mixed"], score),
                "n_pure_units": len(a),
                "n_mixed_units": len(b),
            })
    heterogeneity_auc = pd.DataFrame(auc_rows)
    heterogeneity_auc.to_csv(args.output_dir / "heterogeneity_mixture_detection_auc.csv", index=False)

    # Key manuscript table
    key_conditions = asym[
        asym["total_cells"].isin([10, 20, 50, 100])
        & asym["requested_dominant_fraction"].isin([0.75, 0.90, 0.95])
    ].copy()
    if not key_conditions.empty:
        key_table = aggregate_with_bootstrap(
            key_conditions,
            ["total_cells", "requested_dominant_fraction", "minority_fraction"],
            [
                "dominant_species_top1_rate",
                "minority_species_top2_rate",
                "dual_species_top2_rate",
                "dominant_panel_top1_rate",
                "minority_panel_top2_rate",
                "dual_panel_top2_rate",
            ],
            args.bootstrap,
            args.seed + 51,
        )
        for c in key_table.columns:
            if "_rate_" in c and (c.endswith("_mean") or c.endswith("_ci_low") or c.endswith("_ci_high")):
                key_table[c] *= 100.0
        key_table.to_csv(args.output_dir / "manuscript_key_conditions_table.csv", index=False)

    if args.skip_plots:
        logging.info("--skip-plots requested; summary tables complete.")
        return

    # 1. Minority species Top-2
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for n_cells in sorted(plot_summary["total_cells"].unique()):
        g = plot_summary[plot_summary["total_cells"] == n_cells].sort_values("minority_fraction")
        ax.plot(
            100 * g["minority_fraction"],
            100 * g["minority_species_top2_rate_mean"],
            marker="o", label=f"{n_cells} cells",
        )
    ax.set_xlabel("Minority population (%)")
    ax.set_ylabel("Minority species recovered in Top-2 (%)")
    ax.set_ylim(0, 100)
    ax.legend(title="Total cells", ncol=2)
    ax.set_title("Minority-species recovery from synthetic polymicrobial mixtures")
    savefig(fig, args.output_dir, "minority_species_top2_recovery")

    # 2. Dominant species Top-1
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for n_cells in sorted(plot_summary["total_cells"].unique()):
        g = plot_summary[plot_summary["total_cells"] == n_cells].sort_values("minority_fraction")
        ax.plot(
            100 * g["minority_fraction"],
            100 * g["dominant_species_top1_rate_mean"],
            marker="o", label=f"{n_cells} cells",
        )
    ax.set_xlabel("Minority population (%)")
    ax.set_ylabel("Dominant species recovered as Top-1 (%)")
    ax.set_ylim(0, 100)
    ax.legend(title="Total cells", ncol=2)
    ax.set_title("Dominant-species recovery under synthetic mixing")
    savefig(fig, args.output_dir, "dominant_species_top1_recovery")

    # 3. Dual species heatmap
    heat = (
        asym.groupby(["total_cells", "minority_fraction"])["dual_species_top2_rate"]
        .mean().unstack("minority_fraction").sort_index()
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    im = ax.imshow(100 * heat.to_numpy(), aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels([f"{100*x:.1f}%" for x in heat.columns])
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index.astype(int))
    ax.set_xlabel("Minority population")
    ax.set_ylabel("Total cells")
    ax.set_title("Both true species recovered as the Top-2 predictions")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = 100 * heat.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Dual-species recovery (%)")
    savefig(fig, args.output_dir, "dual_species_top2_heatmap")

    # 4. Different-panel dual-panel recovery
    diff_panel = panel_strat[
        (panel_strat["same_panel_pair"] == False)
        & (panel_strat["requested_dominant_fraction"] > 0.50)
    ].copy()
    if not diff_panel.empty:
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        for n_cells in sorted(diff_panel["total_cells"].unique()):
            g = diff_panel[diff_panel["total_cells"] == n_cells].sort_values("minority_fraction")
            ax.plot(
                100 * g["minority_fraction"],
                100 * g["dual_panel_top2_rate_mean"],
                marker="o", label=f"{n_cells} cells",
            )
        ax.set_xlabel("Minority population (%)")
        ax.set_ylabel("Both true AST-routing panels recovered in Top-2 (%)")
        ax.set_ylim(0, 100)
        ax.legend(title="Total cells", ncol=2)
        ax.set_title("Different-panel synthetic mixtures")
        savefig(fig, args.output_dir, "different_panel_dual_panel_recovery")

    # 5. Same-panel shared route Top-1
    same_panel = panel_strat[panel_strat["same_panel_pair"] == True].copy()
    if not same_panel.empty:
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        for n_cells in sorted(same_panel["total_cells"].unique()):
            g = same_panel[same_panel["total_cells"] == n_cells].sort_values("minority_fraction")
            ax.plot(
                100 * g["minority_fraction"],
                100 * g["same_panel_correct_top1_rate_mean"],
                marker="o", label=f"{n_cells} cells",
            )
        ax.set_xlabel("Minority population (%)")
        ax.set_ylabel("Shared AST-routing panel recovered as Top-1 (%)")
        ax.set_ylim(0, 100)
        ax.legend(title="Total cells", ncol=2)
        ax.set_title("Same-panel synthetic mixtures")
        savefig(fig, args.output_dir, "same_panel_top1_recovery")

    # 6. Heterogeneity AUC
    if not heterogeneity_auc.empty:
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        for feat in heterogeneity_auc["feature"].unique():
            g = heterogeneity_auc[heterogeneity_auc["feature"] == feat].sort_values("total_cells")
            ax.plot(g["total_cells"], g["auc"], marker="o", label=feat.replace("_mean", ""))
        ax.axhline(0.5, linestyle="--", linewidth=1)
        ax.set_xlabel("Total cells")
        ax.set_ylabel("AUC: synthetic mixed vs pure")
        ax.set_ylim(0.45, 1.0)
        ax.legend(fontsize=8)
        ax.set_title("Prediction heterogeneity as an indicator of synthetic mixing")
        savefig(fig, args.output_dir, "heterogeneity_auc_vs_cell_number")

    # 7. Pure vs mixed JS dispersion, choose 20 cells if available else closest count
    selected_n = 20 if 20 in args.cell_counts else sorted(args.cell_counts)[0]
    pure_h = pure_units[pure_units["total_cells"] == selected_n].copy()
    mixed_h = mixed_units[mixed_units["total_cells"] == selected_n].copy()
    feature = "js_dispersion_bits_mean"
    pure_vals = pure_h[feature].dropna().to_numpy()
    mixed_vals = mixed_h[feature].dropna().to_numpy()
    if len(pure_vals) and len(mixed_vals):
        fig, ax = plt.subplots(figsize=(6.2, 5.0))
        ax.boxplot([pure_vals, mixed_vals], labels=["Pure", "Synthetic mixed"], showfliers=False)
        ax.set_ylabel("Cell-probability dispersion (bits)")
        ax.set_title(f"Prediction heterogeneity using {selected_n} cells")
        savefig(fig, args.output_dir, "pure_vs_mixed_js_dispersion")

    # 8. Species-pair heatmap at 20 cells / 90:10 if present, else nearest available
    pair_n = 20 if 20 in args.cell_counts else sorted(args.cell_counts)[0]
    pair_frac = min(args.fractions, key=lambda x: abs(x - 0.90))
    pair_condition = mixed_units[
        (mixed_units["total_cells"] == pair_n)
        & np.isclose(mixed_units["requested_dominant_fraction"], pair_frac)
    ].copy()
    if not pair_condition.empty:
        pair_scores = (
            pair_condition.groupby(["species_a", "species_b"])["dual_species_top2_rate"]
            .mean().reset_index()
        )
        pair_scores.to_csv(
            args.output_dir / "species_pair_dual_recovery_selected_condition.csv", index=False
        )
        mat = pd.DataFrame(np.nan, index=SPECIES, columns=SPECIES, dtype=float)
        for _, r in pair_scores.iterrows():
            a, b = r["species_a"], r["species_b"]
            mat.loc[a, b] = r["dual_species_top2_rate"]
            mat.loc[b, a] = r["dual_species_top2_rate"]
        fig, ax = plt.subplots(figsize=(8.0, 7.2))
        im = ax.imshow(100 * mat.to_numpy(), vmin=0, vmax=100)
        ax.set_xticks(np.arange(len(SPECIES)))
        ax.set_xticklabels(SPECIES, rotation=90)
        ax.set_yticks(np.arange(len(SPECIES)))
        ax.set_yticklabels(SPECIES)
        ax.set_title(
            f"Dual-species recovery by pair\n{pair_n} cells, "
            f"{int(round(100*pair_frac))}:{int(round(100*(1-pair_frac)))}"
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Both true species in Top-2 (%)")
        savefig(fig, args.output_dir, "species_pair_dual_recovery_heatmap")

    # 9. Selected ROC curve using JS dispersion
    roc_n = selected_n
    pure_n = pure_units[pure_units["total_cells"] == roc_n][[feature]].dropna().copy()
    pure_n["is_mixed"] = 0
    mixed_n = mixed_units[mixed_units["total_cells"] == roc_n][[feature]].dropna().copy()
    mixed_n["is_mixed"] = 1
    roc_df = pd.concat([pure_n, mixed_n], ignore_index=True)
    if not roc_df.empty and roc_df["is_mixed"].nunique() == 2:
        y = roc_df["is_mixed"].to_numpy()
        score = roc_df[feature].to_numpy()
        fpr, tpr, _ = roc_curve(y, score)
        auc = roc_auc_score(y, score)
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_xlabel("False-positive rate")
        ax.set_ylabel("True-positive rate")
        ax.set_title(f"Synthetic mixture detection, {roc_n} cells")
        ax.legend()
        savefig(fig, args.output_dir, "selected_heterogeneity_roc")

    logging.info("Summary tables and figures completed.")



# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    args = build_arc_config()

    if args.n_repeats <= 0:
        raise ValueError("N_REPEATS must be > 0")
    if args.bootstrap <= 0:
        raise ValueError("N_BOOTSTRAP must be > 0")
    if args.max_sample_pairs < 0:
        raise ValueError("MAX_SAMPLE_PAIRS_PER_SPECIES_PAIR must be >= 0 or None")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output_dir, args.log_level)

    logging.info("=" * 79)
    logging.info("SC-MAPS POLYMICROBIAL ANALYSIS — ALL-IN-ONE ARC SCRIPT")
    logging.info("=" * 79)
    logging.info("Trillian source ZIP: %s", TRILLIAN_ZIP)
    logging.info("Trillian scratch root: %s", DATA_ROOT)
    logging.info("Temporary scratch ZIP: %s", SCRATCH_ZIP)
    logging.info("Extraction completion marker: %s", EXTRACTION_COMPLETE_MARKER)
    logging.info("Plate-1 holdout: %s", HOLDOUT_CSV)
    logging.info("DenseNet concat checkpoint: %s", DENSENET_CONCAT_CHECKPOINT)
    logging.info("Persistent output directory: %s", OUTPUT_DIR)
    logging.info("Cell counts: %s", args.cell_counts)
    logging.info("Dominant fractions: %s", args.fractions)
    logging.info("Repeats per condition: %d", args.n_repeats)
    logging.info(
        "Sample pairs/species-pair: %s",
        "ALL" if args.max_sample_pairs == 0 else args.max_sample_pairs,
    )
    logging.info("Both dominance directions: %s", not args.one_direction)
    logging.info("Resume pair checkpoints: %s", args.resume)

    # -----------------------------------------------------------------
    # Stage 0: ensure the Trillian ZIP has been extracted to ARC scratch.
    # If the dataset is already present and complete, this is skipped.
    # -----------------------------------------------------------------
    ensure_trillian_dataset()

    # -----------------------------------------------------------------
    # Stage 1: create/reuse Plate-1 cell probabilities directly from
    # the Trillian holdout + trained DenseNet121 concat model.
    # -----------------------------------------------------------------
    ensure_holdout_predictions()

    # -----------------------------------------------------------------
    # Stage 2: run pure controls and all 78 in-silico two-species tests.
    # -----------------------------------------------------------------
    state, true_species_col, sample_id_col, prob_map = load_input(args)

    pure_units = run_pure_controls(state, args)
    mixed_units = run_mixtures(state, args)

    # -----------------------------------------------------------------
    # Stage 3: create statistical summaries + publication figures.
    # -----------------------------------------------------------------
    create_summaries_and_figures(pure_units, mixed_units, args)

    manifest = {
        "trillian_zip": str(TRILLIAN_ZIP),
        "scratch_zip": str(SCRATCH_ZIP),
        "extraction_complete_marker": str(EXTRACTION_COMPLETE_MARKER),
        "data_root": str(DATA_ROOT),
        "holdout_csv": str(HOLDOUT_CSV),
        "checkpoint": str(DENSENET_CONCAT_CHECKPOINT),
        "predictions_csv": str(PREDICTIONS_CSV),
        "output_dir": str(args.output_dir.resolve()),
        "n_input_cells": int(len(state.work)),
        "n_source_samples": int(state.work[sample_id_col].nunique()),
        "species": ",".join(SPECIES),
        "cell_counts": ",".join(map(str, args.cell_counts)),
        "mixture_dominant_fractions": ",".join(map(str, args.fractions)),
        "n_repeats": args.n_repeats,
        "simulate_both_directions": not args.one_direction,
        "sampling_with_replacement": args.sampling_with_replacement,
        "max_sample_pairs_per_species_pair": (
            "ALL" if args.max_sample_pairs == 0 else args.max_sample_pairs
        ),
        "random_seed": args.seed,
        "bootstrap_repetitions": args.bootstrap,
        "save_raw_repeat_level": args.save_raw,
        "true_species_col": true_species_col,
        "sample_id_col": sample_id_col,
        "probability_columns": json.dumps(prob_map),
        "interpretation": (
            "In-silico stress test using cell predictions from separately prepared "
            "Plate-1 monocultures; not genuine polymicrobial validation."
        ),
    }
    pd.DataFrame([manifest]).to_csv(
        args.output_dir / "run_manifest.csv", index=False
    )

    logging.info("=" * 79)
    logging.info("DONE")
    logging.info("All results are in: %s", args.output_dir)
    logging.info("=" * 79)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
