#!/usr/bin/env python3
"""
SC-MAPS — Chris review analyses, all-in-one ARC script
=====================================================

Purpose
-------
Runs the post-hoc analyses requested during Chris's manuscript review in one job:

  0. Reuse / prepare the Trillian dataset on ARC scratch using the established
     safe ZIP workflow (IMPORTANT: the ZIP is extracted into SCRATCH_PARENT,
     not into DATA_ROOT, so the top-level dataset directory is not duplicated).
  1. Generate/reuse Plate-1 DenseNet121 concatenation per-cell probabilities.
     A balanced subset of penultimate-layer features is also extracted.
  2. Recalculate observed species and AST-routing metrics.
  3. Build group-size-matched random AST-routing nulls for:
       - top-1 species -> routing mapping accuracy
       - probability-pooled routing accuracy
       - top-2 routing accuracy
       - rescue of species errors within the correct routing group
  4. Cluster-bootstrap the observed metrics by holdout sample.
  5. Recreate the 200-cell, 50:50 synthetic-mixture analysis and test the
     observed routing recovery against group-size-matched random mappings.
  6. Produce PCA/UMAP feature-space plots and kNN neighbour summaries.
     Unseen-organism analysis is run automatically if an unseen manifest can
     be found, or if --unseen-manifest is supplied.
  7. Rebuild aligned Figure-1C/Figure-1D-style phenotype panels using the same
     feature list/order wherever the manifest contains the required columns.
  8. Summarise repeated-seed model results if prediction CSVs from multiple
     seeds are supplied via --seed-results-dir.

IMPORTANT ABOUT RANDOM-SEED RETRAINING
--------------------------------------
Repeated training from different random initialisations cannot be recreated
faithfully from the final checkpoint alone. This script therefore DOES NOT
pretend that dropout/bootstrapping is equivalent to retraining. Stage 8 will
analyse genuine repeated-seed outputs if they exist. To make this script launch
new seed trainings itself, the original training program and its exact CLI/API
are required; see the generated SEED_RETRAINING_REQUIRED.txt when no seed runs
are found.

Run on ARC
----------
Typical interactive/batch execution after loading your existing PyTorch env:

    python -u SC_MAPS_CHRIS_REVIEW_ALL_IN_ONE_ARC.py

Optional unseen manifest:

    python -u SC_MAPS_CHRIS_REVIEW_ALL_IN_ONE_ARC.py \
        --unseen-manifest /path/to/unseen_cell_manifest.csv

Optional directory containing genuine repeated-seed prediction CSVs:

    python -u SC_MAPS_CHRIS_REVIEW_ALL_IN_ONE_ARC.py \
        --seed-results-dir /path/to/seed_outputs

Dependencies: numpy, pandas, matplotlib, scikit-learn, torch, torchvision.
Optional for UMAP: umap-learn. If unavailable, all other stages still run.

The ZIP/scratch logic below is deliberately based on the previously used
SC_MAPS_polymicrobial_ALL_IN_ONE_ARC.py workflow:
  * prepared scratch dataset => reuse immediately;
  * missing sentinel on a prepared dataset => touch sentinel, DO NOT re-extract;
  * dataset genuinely absent => copy persistent ZIP to scratch;
  * same-size scratch ZIP => reuse;
  * extract into SCRATCH_PARENT (not DATA_ROOT);
  * resolve accidental single-directory nesting;
  * keep the scratch ZIP after extraction.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import time
import zipfile
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision.models import densenet121
except Exception as exc:
    raise RuntimeError(
        "PyTorch/torchvision could not be imported. Load the same ARC environment "
        "used for SC-MAPS training before running this script."
    ) from exc

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.neighbors import NearestNeighbors

try:
    import umap.umap_ as umap
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


# =============================================================================
# 0. FIXED SC-MAPS DEFINITIONS
# =============================================================================

SPECIES = [
    "ACAA", "ECOL", "ENTC", "KLAE", "KLPN", "PSAR", "PSMA",
    "SAENA", "SAUR", "SEPI", "SERM", "SLUG", "STCP",
]
SPECIES_TO_IDX = {s: i for i, s in enumerate(SPECIES)}

ROUTING_GROUPS = [
    "Routine Enterobacterales",
    "AmpC/caution Enterobacterales",
    "Salmonella",
    "Pseudomonas",
    "Acinetobacter",
    "Stenotrophomonas",
    "S. aureus-like",
    "CoNS",
]
ROUTING_TO_IDX = {g: i for i, g in enumerate(ROUTING_GROUPS)}

SPECIES_TO_GROUP = {
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

# Group sizes of the real routing scheme: 3,2,2,2,1,1,1,1.
# Random mappings preserve these sizes exactly.
NULL_GROUP_SLOTS = np.array(
    [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7], dtype=np.int16
)
assert len(NULL_GROUP_SLOTS) == len(SPECIES)

REAL_MAPPING_IDX = np.array(
    [ROUTING_TO_IDX[SPECIES_TO_GROUP[s]] for s in SPECIES], dtype=np.int16
)

UNSEEN_ALIASES = {
    "BSB": "BSB",
    "EF": "EF",
    "EFM": "EFM",
    "LMON": "LMON",
    "HFLU": "HFLU",
    "HELU": "HFLU",  # historical spelling in some SC-MAPS files
}

# Canonical stored crop order: PhaseContrast, BF, DAPI, WGA, FM1-43
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


# =============================================================================
# 1. ARC PATHS AND ANALYSIS SETTINGS
# =============================================================================

TRILLIAN_ZIP = Path(
    "/data/phys-deep-learning-bacteria/phys3115/"
    "Bacteria-data-15 species/Trillian/"
    "Trillian_DATASET_with_holdout.zip"
)
PERSISTENT_TRILLIAN_DIR = TRILLIAN_ZIP.parent

# Unseen-organism single-cell archive used by the prior OOD/unseen analysis.
# The persistent filename may carry a suffix/version, so it is discovered by glob.
NOVEL_ZIP_GLOB = "SC_MAPS_NOVEL_SPECIES_SINGLE_CELL_DATASET*.zip"
NOVEL_SCRATCH_PARENT = Path("/scratch/phys3115/sc_maps_figure4_ood")
NOVEL_DATA_ROOT: Optional[Path] = None

SCRATCH_PARENT = Path("/scratch/phys3115/bacteria_data")
DATA_ROOT = SCRATCH_PARENT / "Trillian_DATASET_with_holdout"
SCRATCH_ZIP = SCRATCH_PARENT / "Trillian_DATASET_with_holdout.zip"
HOLDOUT_CSV = DATA_ROOT / "splits" / "holdout_plate1.csv"
EXTRACTION_COMPLETE_MARKER = DATA_ROOT / ".arc_extract_complete"

DENSENET_CONCAT_CHECKPOINT = PERSISTENT_TRILLIAN_DIR / "DenseNet_Concat_weights" / "best_fusion_model.pt"

OUTPUT_DIR = PERSISTENT_TRILLIAN_DIR / "Chris_review_analysis_ARC"
PREDICTIONS_CSV = OUTPUT_DIR / "holdout_plate1_DenseNet121_concat_cell_probabilities.csv"
FEATURES_NPZ = OUTPUT_DIR / "holdout_balanced_penultimate_features.npz"
LEGACY_PREDICTIONS_CSV = (
    PERSISTENT_TRILLIAN_DIR / "Polymicrobial_analysis_ARC" /
    "holdout_plate1_DenseNet121_concat_cell_probabilities.csv"
)

RANDOM_SEED = 20261008
INFERENCE_BATCH_SIZE = 256
INFERENCE_NUM_WORKERS = 4
USE_AMP = True
FORCE_REINFER = False

# Known feature-space reference: at most this many cells per known species.
KNOWN_FEATURES_PER_SPECIES = 2000
UNSEEN_FEATURES_PER_SPECIES = 4000

# Random-routing nulls.
# Mapping/rescue are cheap and use all 10,000 mappings exactly from a 13x13 CM.
# Pooled/top2 require cell-level probability pooling; 2,000 is a practical ARC
# default and yields empirical p-value resolution ~5e-4.
N_NULL_TOPOLOGY = 10_000
N_NULL_POOLING = 2_000
NULL_GPU_BATCH = 16

# Sample-cluster bootstrap.
N_BOOTSTRAP = 5000

# 50:50 mixture recreation matching the established analysis.
MIXTURE_TOTAL_CELLS = 200
MIXTURE_REPEATS = 200
MIXTURE_MAX_SAMPLE_PAIRS = 10
N_NULL_MIXTURE = 2000
MIXTURE_NULL_GPU_BATCH = 16

# Figure 1 phenotype analysis.
FIG1_MAX_ROWS = None  # None = use all manifest rows


# =============================================================================
# 2. GENERAL UTILITIES
# =============================================================================

def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "chris_review_analysis.log"
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, mode="a")]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.info("Log: %s", log_path)


def stable_seed(base_seed: int, label: str) -> int:
    return (int(base_seed) + zlib.crc32(label.encode("utf-8"))) % (2**32 - 1)


def savefig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def normalise_probabilities(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=np.float32)
    if np.any(~np.isfinite(P)):
        raise ValueError("Non-finite probability values detected.")
    if np.any(P < -1e-7):
        raise ValueError("Negative probability values detected.")
    P = np.clip(P, 0.0, None)
    sums = P.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("Probability row with sum <= 0 detected.")
    return P / sums


def empirical_p_ge(null: np.ndarray, observed: float) -> float:
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    return float((np.sum(null >= observed) + 1) / (len(null) + 1))


def percentile_ci(x: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, alpha / 2)), float(np.quantile(x, 1 - alpha / 2))


def write_json(path: Path, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x))


def _norm_text(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


# =============================================================================
# 3. CORRECT ARC ZIP / SCRATCH LOGIC
# =============================================================================

def has_prepared_dataset(root: Path) -> bool:
    root = Path(root)
    return (root / "cell_manifest.csv").exists() and (root / "splits").is_dir()


def safe_copy_file(src: Path, dst: Path) -> Path:
    """Copy persistent ZIP to scratch only if a same-size scratch copy is absent."""
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"Source ZIP does not exist: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        logging.info("ZIP already present on scratch with matching size: %s", dst)
        return dst

    if dst.exists():
        logging.warning("Removing mismatched/incomplete scratch ZIP: %s", dst)
        dst.unlink()

    logging.info("Copying ZIP to scratch: %s -> %s", src, dst)
    shutil.copy2(src, dst)

    if dst.stat().st_size != src.stat().st_size:
        raise IOError(
            "ZIP copy size mismatch after copy.\n"
            f"Source bytes: {src.stat().st_size}\n"
            f"Scratch bytes: {dst.stat().st_size}"
        )
    logging.info("Copied ZIP size: %.2f GB", dst.stat().st_size / 1e9)
    return dst


def _safe_zip_target(extract_root: Path, member_name: str) -> Path:
    """Resolve a ZIP member and reject path traversal robustly."""
    root = extract_root.resolve()
    target = (root / member_name).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Unsafe ZIP member path detected: {member_name}") from exc
    return target


def safe_extract_member(zf: zipfile.ZipFile, member: zipfile.ZipInfo, extract_root: Path) -> None:
    target = _safe_zip_target(Path(extract_root), member.filename)
    if member.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member, "r") as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    perm = member.external_attr >> 16
    if perm:
        try:
            target.chmod(perm)
        except Exception:
            pass


def resolve_dataset_root(root: Path) -> Path:
    """Resolve an accidental single-directory nesting without inventing paths."""
    current = Path(root)
    for _ in range(10):
        if has_prepared_dataset(current):
            return current
        subdirs = [p for p in current.iterdir() if p.is_dir()] if current.exists() else []
        if len(subdirs) != 1:
            break
        current = subdirs[0]
    return current


def extract_zip(zip_path: Path, extract_root: Path, expected_dataset_root: Path, force: bool = False) -> Path:
    """
    Extract into SCRATCH_PARENT, NOT DATA_ROOT.

    If DATA_ROOT is already prepared but the sentinel is missing, touch the
    sentinel and do not extract again. This is the key behaviour that prevents
    the previous double/nested extraction problem.
    """
    zip_path = Path(zip_path)
    extract_root = Path(extract_root)
    dataset_root = Path(expected_dataset_root)
    sentinel = dataset_root / ".arc_extract_complete"

    if force and dataset_root.exists():
        logging.warning("Force re-extraction requested; removing %s", dataset_root)
        shutil.rmtree(dataset_root)

    if has_prepared_dataset(dataset_root) and sentinel.exists():
        logging.info("Scratch dataset already prepared and marked complete: %s", dataset_root)
        return dataset_root

    if has_prepared_dataset(dataset_root) and not force:
        logging.info("Prepared scratch dataset found without sentinel; touching marker, no extraction.")
        sentinel.touch()
        return dataset_root

    if not zip_path.exists():
        raise FileNotFoundError(f"Cannot extract missing ZIP: {zip_path}")

    # Opening the archive verifies the central directory before writing files.
    logging.info("Extracting ZIP into SCRATCH_PARENT: %s", extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        total_files = len(members)
        total_gb = sum(m.file_size for m in members) / 1e9
        logging.info("ZIP entries: %s; uncompressed size ~ %.2f GB", f"{total_files:,}", total_gb)
        iterator = tqdm(members, total=total_files, desc="Extracting Trillian", unit="file") if tqdm else members
        for i, member in enumerate(iterator, start=1):
            safe_extract_member(zf, member, extract_root)
            if tqdm is None and (i == 1 or i % 1000 == 0 or i == total_files):
                logging.info("Extracted %s/%s entries", f"{i:,}", f"{total_files:,}")

    dataset_root = resolve_dataset_root(dataset_root)
    if not has_prepared_dataset(dataset_root):
        # Last-resort detection after extraction. Prefer candidates whose parent
        # directory is named Trillian_DATASET_with_holdout.
        candidates = list(extract_root.rglob("cell_manifest.csv"))
        prepared = [p.parent for p in candidates if (p.parent / "splits").is_dir()]
        if prepared:
            preferred = [p for p in prepared if p.name == "Trillian_DATASET_with_holdout"]
            dataset_root = preferred[0] if preferred else prepared[0]
            logging.info("Detected prepared dataset root: %s", dataset_root)

    if not has_prepared_dataset(dataset_root):
        raise FileNotFoundError(
            "Extraction completed but no dataset root containing cell_manifest.csv + splits/ was found. "
            f"Expected near: {expected_dataset_root}"
        )

    (dataset_root / ".arc_extract_complete").touch()
    logging.info("Scratch extraction complete: %s", dataset_root)
    return dataset_root


def ensure_trillian_dataset() -> Path:
    global DATA_ROOT, HOLDOUT_CSV, EXTRACTION_COMPLETE_MARKER
    SCRATCH_PARENT.mkdir(parents=True, exist_ok=True)

    # Before deciding the dataset is absent, resolve any pre-existing accidental
    # single-directory nesting from an older extraction. This avoids copying or
    # extracting the huge ZIP again merely because DATA_ROOT itself is a wrapper.
    resolved_existing = resolve_dataset_root(DATA_ROOT)
    if has_prepared_dataset(resolved_existing):
        DATA_ROOT = resolved_existing

    if has_prepared_dataset(DATA_ROOT):
        EXTRACTION_COMPLETE_MARKER = DATA_ROOT / ".arc_extract_complete"
        if not EXTRACTION_COMPLETE_MARKER.exists():
            logging.info("Prepared dataset already exists; adding sentinel without re-extracting.")
            EXTRACTION_COMPLETE_MARKER.touch()
        else:
            logging.info("Prepared dataset already exists in scratch: %s", DATA_ROOT)
        DATA_ROOT = resolve_dataset_root(DATA_ROOT)
    else:
        safe_copy_file(TRILLIAN_ZIP, SCRATCH_ZIP)
        DATA_ROOT = extract_zip(SCRATCH_ZIP, SCRATCH_PARENT, DATA_ROOT, force=False)
        DATA_ROOT = resolve_dataset_root(DATA_ROOT)

    HOLDOUT_CSV = DATA_ROOT / "splits" / "holdout_plate1.csv"
    EXTRACTION_COMPLETE_MARKER = DATA_ROOT / ".arc_extract_complete"

    if not HOLDOUT_CSV.exists():
        candidates = sorted((DATA_ROOT / "splits").glob("holdout*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No holdout CSV found in {DATA_ROOT / 'splits'}")
        HOLDOUT_CSV = candidates[0]
        logging.warning("holdout_plate1.csv absent; using %s", HOLDOUT_CSV)

    logging.info("Resolved DATA_ROOT: %s", DATA_ROOT)
    logging.info("Resolved HOLDOUT_CSV: %s", HOLDOUT_CSV)
    logging.info("Scratch ZIP retained at: %s", SCRATCH_ZIP)
    return DATA_ROOT


def _find_prepared_novel_root(parent: Path) -> Optional[Path]:
    """Find an extracted unseen/novel dataset by its cell_manifest.csv."""
    parent = Path(parent)
    if not parent.exists():
        return None
    direct = parent / "cell_manifest.csv"
    if direct.exists():
        return parent
    candidates = sorted(parent.rglob("cell_manifest.csv"))
    if not candidates:
        return None
    # Prefer a manifest whose parent name contains novel/unseen/single_cell.
    preferred = [
        x.parent for x in candidates
        if any(tok in x.parent.name.lower() for tok in ("novel", "unseen", "single_cell", "single-cell"))
    ]
    return preferred[0] if preferred else candidates[0].parent


def ensure_novel_dataset() -> Optional[Path]:
    """
    Prepare/reuse the unseen-organism single-cell ZIP on its established scratch area.

    This deliberately mirrors the Trillian logic: reuse a prepared extraction first;
    otherwise select the newest matching persistent SC_MAPS_NOVEL_SPECIES_SINGLE_CELL_DATASET*.zip,
    reuse/copy a same-size scratch ZIP, extract into NOVEL_SCRATCH_PARENT (not into
    an assumed inner directory), locate cell_manifest.csv, and retain the scratch ZIP.
    """
    global NOVEL_DATA_ROOT
    NOVEL_SCRATCH_PARENT.mkdir(parents=True, exist_ok=True)

    existing = _find_prepared_novel_root(NOVEL_SCRATCH_PARENT)
    if existing is not None:
        sentinel = existing / ".arc_extract_complete"
        if not sentinel.exists():
            sentinel.touch()
        NOVEL_DATA_ROOT = existing
        logging.info("Prepared unseen/novel dataset already exists in scratch: %s", existing)
        return existing

    matches = sorted(
        PERSISTENT_TRILLIAN_DIR.glob(NOVEL_ZIP_GLOB),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        logging.warning(
            "No unseen/novel ZIP matching %s found in %s",
            NOVEL_ZIP_GLOB, PERSISTENT_TRILLIAN_DIR,
        )
        return None
    source_zip = matches[0]
    if len(matches) > 1:
        logging.warning(
            "Multiple unseen/novel ZIPs matched; using newest: %s. Matches: %s",
            source_zip, [str(x) for x in matches],
        )
    scratch_zip = NOVEL_SCRATCH_PARENT / source_zip.name
    safe_copy_file(source_zip, scratch_zip)

    logging.info("Extracting unseen/novel ZIP into scratch parent: %s", NOVEL_SCRATCH_PARENT)
    with zipfile.ZipFile(scratch_zip, "r") as zf:
        members = zf.infolist()
        iterator = tqdm(members, total=len(members), desc="Extracting unseen dataset", unit="file") if tqdm else members
        for member in iterator:
            safe_extract_member(zf, member, NOVEL_SCRATCH_PARENT)

    root = _find_prepared_novel_root(NOVEL_SCRATCH_PARENT)
    if root is None:
        raise FileNotFoundError(
            "Unseen/novel ZIP extracted, but no cell_manifest.csv was found under "
            f"{NOVEL_SCRATCH_PARENT}"
        )
    (root / ".arc_extract_complete").touch()
    NOVEL_DATA_ROOT = root
    logging.info("Resolved unseen/novel DATA_ROOT: %s", root)
    logging.info("Unseen scratch ZIP retained at: %s", scratch_zip)
    return root


# =============================================================================
# 4. CHECKPOINT-MATCHED DENSENET121 CONCAT MODEL
# =============================================================================

def _norm_channel_name(name: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "", str(name)).upper()
    aliases = {
        "PHASECONTRAST": "PHASECONTRAST", "PHASE": "PHASECONTRAST",
        "PHASECONTRASTIMAGE": "PHASECONTRAST", "BRIGHTFIELD": "BF", "BF": "BF",
        "DAPI": "DAPI", "WGA": "WGA", "WGAAF647": "WGA", "WGA647": "WGA",
        "FM1": "FM1", "FM143": "FM1",
    }
    return aliases.get(text, text)


def _state_dict_from_checkpoint(ckpt):
    if not isinstance(ckpt, dict):
        raise TypeError("Checkpoint is not a dictionary.")
    for key in ("model_state_dict", "state_dict", "model"):
        if key in ckpt and isinstance(ckpt[key], dict):
            sd = ckpt[key]
            break
    else:
        if all(torch.is_tensor(v) for v in ckpt.values()):
            sd = ckpt
        else:
            raise KeyError("Could not find model_state_dict/state_dict/model in checkpoint.")
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def _checkpoint_class_order(ckpt, state_dict):
    class_to_idx = ckpt.get("class_to_idx") if isinstance(ckpt, dict) else None
    if isinstance(class_to_idx, dict) and class_to_idx:
        return [k for k, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    if isinstance(ckpt, dict):
        for key in ("classes", "class_names", "species"):
            vals = ckpt.get(key)
            if isinstance(vals, (list, tuple)) and len(vals) == len(SPECIES):
                return list(vals)
    return SPECIES.copy()


def _infer_branch_names(ckpt, state_dict):
    metadata_names = None
    if isinstance(ckpt, dict):
        for key in ("selected_channels", "selected_channel_names", "channel_names", "channels"):
            vals = ckpt.get(key)
            if isinstance(vals, (list, tuple)) and len(vals) == 4:
                metadata_names = [str(x) for x in vals]
                break
    state_names = []
    for k in state_dict:
        m = re.match(r"^branches\.([^.]+)\.", k)
        if m and m.group(1) not in state_names:
            state_names.append(m.group(1))
    if len(state_names) != 4:
        raise RuntimeError(f"Expected 4 fusion branches in checkpoint, found: {state_names}")
    if metadata_names:
        ordered = []
        for meta in metadata_names:
            target = _norm_channel_name(meta)
            matches = [s for s in state_names if _norm_channel_name(s) == target]
            if len(matches) == 1:
                ordered.append(matches[0])
        if len(ordered) == 4 and len(set(ordered)) == 4:
            return ordered
    wanted = ["DAPI", "PHASECONTRAST", "WGA", "FM1"]
    ordered = []
    for target in wanted:
        matches = [s for s in state_names if _norm_channel_name(s) == target]
        if len(matches) != 1:
            raise RuntimeError(f"Could not uniquely map {state_names} to required channel {target}")
        ordered.append(matches[0])
    return ordered


def _infer_fusion_dimensions(state_dict, branch_names):
    first = branch_names[0]
    proj_key = f"projections.{first}.0.weight"
    if proj_key not in state_dict:
        candidates = [k for k in state_dict if k.startswith(f"projections.{first}.") and k.endswith(".weight") and state_dict[k].ndim == 2]
        if not candidates:
            raise RuntimeError(f"Could not infer projection layer for {first}")
        proj_key = candidates[0]
    projection_dim = int(state_dict[proj_key].shape[0])
    cls_keys = [k for k in state_dict if k.startswith("classifier.") and k.endswith(".weight") and state_dict[k].ndim == 2]
    cls_keys = sorted(cls_keys, key=lambda x: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", x)])
    if len(cls_keys) < 2:
        raise RuntimeError(f"Expected >=2 classifier linear weights, got {cls_keys}")
    first_cls, last_cls = state_dict[cls_keys[0]], state_dict[cls_keys[-1]]
    return {
        "projection_dim": int(state_dict[proj_key].shape[0]),
        "classifier_hidden": int(first_cls.shape[0]),
        "classifier_input": int(first_cls.shape[1]),
        "n_classes": int(last_cls.shape[0]),
    }


class FourBranchDenseNet121Concat(nn.Module):
    def __init__(self, branch_names, projection_dim, classifier_hidden, n_classes, dropout):
        super().__init__()
        self.branch_names = list(branch_names)
        self.branches = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        for name in self.branch_names:
            backbone = densenet121(weights=None)
            old = backbone.features.conv0
            backbone.features.conv0 = nn.Conv2d(
                1, old.out_channels, kernel_size=old.kernel_size,
                stride=old.stride, padding=old.padding, bias=False
            )
            backbone.classifier = nn.Identity()
            self.branches[name] = backbone
            self.projections[name] = nn.Sequential(
                nn.Linear(1024, projection_dim), nn.LayerNorm(projection_dim),
                nn.GELU(), nn.Dropout(dropout)
            )
        fused_dim = projection_dim * len(self.branch_names)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, classifier_hidden),
            nn.LayerNorm(classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, n_classes),
        )

    def _fused(self, channel_tensor_dict):
        projected = []
        for name in self.branch_names:
            feat = self.branches[name](channel_tensor_dict[name])
            projected.append(self.projections[name](feat))
        return torch.cat(projected, dim=1)

    def forward(self, channel_tensor_dict):
        fused = self._fused(channel_tensor_dict)
        return self.classifier(fused)

    def forward_with_penultimate(self, channel_tensor_dict):
        """Return logits and the hidden vector immediately before final Linear."""
        fused = self._fused(channel_tensor_dict)
        hidden = fused
        for layer in list(self.classifier.children())[:-1]:
            hidden = layer(hidden)
        logits = self.classifier[-1](hidden)
        return logits, hidden


def load_concat_model(checkpoint_path: Path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"DenseNet concat checkpoint not found: {checkpoint_path}")
    logging.info("Loading concat checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sd = _state_dict_from_checkpoint(ckpt)
    branches = _infer_branch_names(ckpt, sd)
    dims = _infer_fusion_dimensions(sd, branches)
    classes = _checkpoint_class_order(ckpt, sd)
    if dims["n_classes"] != len(classes):
        raise RuntimeError("Checkpoint output dimension does not match class metadata.")
    dropout = 0.30
    if isinstance(ckpt, dict):
        for key in ("best_dropout", "dropout", "fusion_dropout"):
            if key in ckpt:
                try:
                    dropout = float(ckpt[key]); break
                except Exception:
                    pass
    model = FourBranchDenseNet121Concat(
        branches, dims["projection_dim"], dims["classifier_hidden"], dims["n_classes"], dropout
    )
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    logging.info("Branches: %s", branches)
    logging.info("Classes: %s", classes)
    logging.info("Dimensions: %s", dims)
    return model, branches, classes, dims


# =============================================================================
# 5. CROP PATHS, SAMPLE IDs, DATASET
# =============================================================================

def resolve_crop_path(value, extra_bases: Optional[Sequence[Path]] = None) -> Path:
    """Resolve crop paths recorded on either Windows or ARC/Linux.

    The manifests were originally generated on Windows and can therefore contain
    absolute entries such as::

        D:\\Anuj\\...\\SC_MAPS_NOVEL_SPECIES_SINGLE_CELL_DATASET\\crops\\cell.npy

    Those paths are *not* meaningful on ARC.  We therefore never prepend an ARC
    base to the full ``D:/...`` string.  Instead we recover the dataset-relative
    suffix (preferentially from a known dataset-root marker, otherwise from
    ``crops/``) and test it beneath the actual extracted ARC roots.

    This function deliberately avoids ``rglob``/filesystem-wide searching because
    it is called once per cell by DataLoader workers and must stay cheap.
    """
    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "none"}:
        raise FileNotFoundError(f"Empty/invalid crop path: {value}")

    # 1) Native path first (works for already-correct Linux absolute/relative paths).
    native = Path(raw)
    if native.exists():
        return native

    # Normalise Windows separators before doing any suffix logic.  On POSIX,
    # Path('D:/...') is considered relative, so explicitly detect drive paths.
    text = raw.replace("\\", "/")
    is_windows_abs = bool(re.match(r"^[A-Za-z]:/", text))
    norm = Path(text)
    if norm.exists():
        return norm

    # Build an ordered, de-duplicated set of candidate ARC dataset roots.
    bases: List[Path] = [Path(DATA_ROOT)]
    if extra_bases:
        bases.extend(Path(x) for x in extra_bases if x is not None)
    if NOVEL_DATA_ROOT is not None:
        bases.append(Path(NOVEL_DATA_ROOT))

    dedup_bases: List[Path] = []
    seen = set()
    for b in bases:
        key = str(b)
        if key not in seen:
            dedup_bases.append(b)
            seen.add(key)
    bases = dedup_bases

    candidates: List[Path] = []

    # 2) Genuine relative paths from the manifest can be appended directly.
    # Do NOT do this for a Windows drive path such as D:/Anuj/... .
    if not is_windows_abs and not norm.is_absolute():
        for base in bases:
            candidates.append(base / norm)

    # 3) Recover suffixes after known dataset roots.  This is the key fix for
    # both the main Trillian manifest and the unseen-organism Windows manifest.
    marker_patterns = [
        r"(?:^|/)Trillian_DATASET_with_holdout/",
        r"(?:^|/)SC_MAPS_NOVEL_SPECIES_SINGLE_CELL_DATASET[^/]*/",
    ]
    for pat in marker_patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            suffix = text[m.end():]
            if suffix:
                for base in bases:
                    candidates.append(base / Path(suffix))

    # 4) Generic crop suffix fallback.  This covers manifests whose historical
    # root directory has a different name but whose extracted dataset still has
    # the canonical crops/<file>.npy layout.
    m = re.search(r"(?:^|/)(crops/.*)$", text, flags=re.I)
    if m:
        crop_suffix = m.group(1)
        for base in bases:
            candidates.append(base / Path(crop_suffix))

    # 5) Last cheap fallback: filename directly beneath <base>/crops.
    # Using slash-normalised text is important because Path(raw).name would
    # otherwise treat a Windows path as one giant filename on Linux.
    filename = text.rsplit("/", 1)[-1]
    if filename:
        for base in bases:
            candidates.append(base / "crops" / filename)

    seen_candidates = set()
    tried = []
    for q in candidates:
        q = Path(q)
        key = str(q)
        if key in seen_candidates:
            continue
        seen_candidates.add(key)
        tried.append(key)
        if q.exists():
            return q

    preview = "\n  - ".join(tried[:12])
    if len(tried) > 12:
        preview += f"\n  ... ({len(tried) - 12} more candidates)"
    raise FileNotFoundError(
        f"Could not resolve crop path: {value}\n"
        f"Tried ARC candidates:\n  - {preview if preview else '[none]'}"
    )


def parse_sample_id_from_path(path_text: str, species: str) -> str:
    stem = Path(str(path_text)).name
    def grab(pattern, default):
        m = re.search(pattern, stem, flags=re.I)
        return m.group(1) if m else default
    plate = grab(r"plate[_-]?(\d+)", "1")
    rep = grab(r"rep[_-]?(\d+)", "1")
    series = grab(r"series[_-]?(\d+)", None)
    if series is None:
        base = re.sub(r"_cell[_-]?\d+.*$", "", Path(stem).stem, flags=re.I)
        return f"{species}__plate{plate}__rep{rep}__{base}"
    return f"{species}__plate{plate}__rep{rep}__series{int(series):03d}"


def detect_species_crop_columns(df: pd.DataFrame) -> Tuple[str, str]:
    species_col = next((c for c in ("species", "true_species", "label", "target") if c in df.columns), None)
    crop_col = next((c for c in ("crop_path", "npy_path", "path", "file_path", "mask_crop_path") if c in df.columns), None)
    if species_col is None or crop_col is None:
        raise ValueError(f"Could not detect species/crop columns. Columns: {list(df.columns)}")
    return species_col, crop_col


class FusionCropDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, crop_col: str, branch_names: Sequence[str], extra_bases=None):
        self.frame = frame.reset_index(drop=True)
        self.crop_col = crop_col
        self.branch_names = list(branch_names)
        self.extra_bases = list(extra_bases or [])
        self.branch_crop_indices = {}
        for name in self.branch_names:
            norm = _norm_channel_name(name)
            if norm not in STORED_CHANNEL_INDEX:
                raise KeyError(f"Unknown channel {name!r} -> {norm!r}")
            self.branch_crop_indices[name] = STORED_CHANNEL_INDEX[norm]

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        p = resolve_crop_path(row[self.crop_col], self.extra_bases)
        arr = np.load(p, allow_pickle=False)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3-D crop, got {arr.shape}: {p}")
        if arr.shape[0] == 5:
            chw = arr
        elif arr.shape[-1] == 5:
            chw = np.moveaxis(arr, -1, 0)
        else:
            raise ValueError(f"Expected 5-channel crop, got {arr.shape}: {p}")
        channels = {}
        for name in self.branch_names:
            x = chw[self.branch_crop_indices[name]].astype(np.float32, copy=False)
            mean, std = float(x.mean()), float(x.std())
            if std < 1e-6:
                std = 1.0
            x = (x - mean) / std
            channels[name] = torch.from_numpy(x[None, ...].copy())
        return channels, idx


def collate_fusion(batch):
    names = list(batch[0][0].keys())
    channels = {name: torch.stack([item[0][name] for item in batch], dim=0) for name in names}
    idx = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return channels, idx


def balanced_feature_indices(labels: Sequence[str], max_per_class: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=str)
    rng = np.random.default_rng(seed)
    chosen = []
    for lab in sorted(np.unique(labels)):
        idx = np.flatnonzero(labels == lab)
        n = min(max_per_class, len(idx))
        if n:
            chosen.extend(rng.choice(idx, size=n, replace=False).tolist())
    return np.array(sorted(chosen), dtype=np.int64)


# =============================================================================
# 6. HOLDOUT INFERENCE + FEATURE EXTRACTION
# =============================================================================

def infer_dataframe(
    frame: pd.DataFrame,
    species_col: str,
    crop_col: str,
    predictions_csv: Path,
    features_npz: Path,
    feature_max_per_species: int,
    extra_bases: Optional[Sequence[Path]] = None,
    allow_reuse_predictions: bool = True,
) -> Tuple[Path, Path]:
    """Inference over a cell manifest; save all probabilities + balanced hidden features."""
    predictions_csv.parent.mkdir(parents=True, exist_ok=True)

    predictions_ok = predictions_csv.exists() and allow_reuse_predictions and not FORCE_REINFER
    features_ok = features_npz.exists() and not FORCE_REINFER
    if predictions_ok and features_ok:
        logging.info("Reusing predictions: %s", predictions_csv)
        logging.info("Reusing features: %s", features_npz)
        return predictions_csv, features_npz

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Inference device: %s", device)
    model, branches, class_order, dims = load_concat_model(DENSENET_CONCAT_CHECKPOINT, device)

    work = frame.copy().reset_index(drop=True)
    # Force NumPy Unicode dtype so feature-cache strings can be loaded safely
    # with allow_pickle=False. pandas .to_numpy() otherwise often returns dtype=object.
    labels_raw = np.asarray(work[species_col].astype(str).to_numpy(), dtype=np.str_)
    feature_idx = balanced_feature_indices(labels_raw, feature_max_per_species, RANDOM_SEED + 17)
    feature_pos = np.full(len(work), -1, dtype=np.int64)
    feature_pos[feature_idx] = np.arange(len(feature_idx), dtype=np.int64)

    dataset = FusionCropDataset(work, crop_col, branches, extra_bases=extra_bases)
    loader = DataLoader(
        dataset, batch_size=INFERENCE_BATCH_SIZE, shuffle=False,
        num_workers=INFERENCE_NUM_WORKERS, pin_memory=(device.type == "cuda"),
        persistent_workers=(INFERENCE_NUM_WORKERS > 0), collate_fn=collate_fusion,
    )

    all_probs = np.empty((len(work), len(class_order)), dtype=np.float32)
    feat_store = np.empty((len(feature_idx), dims["classifier_hidden"]), dtype=np.float32)
    use_amp_now = USE_AMP and device.type == "cuda"

    with torch.inference_mode():
        for batch_i, (channel_dict, row_idx) in enumerate(loader, start=1):
            channel_dict = {k: v.to(device, non_blocking=True) for k, v in channel_dict.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp_now):
                logits, hidden = model.forward_with_penultimate(channel_dict)
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            rows_np = row_idx.numpy()
            all_probs[rows_np] = probs

            positions = feature_pos[rows_np]
            take = positions >= 0
            if np.any(take):
                hidden_np = hidden.float().cpu().numpy()
                feat_store[positions[take]] = hidden_np[take]

            if batch_i % 100 == 0 or batch_i == len(loader):
                logging.info("Inference %d/%d batches", batch_i, len(loader))

    # Sample IDs: preserve explicit sample column when present, otherwise parse.
    explicit_sample = next((c for c in ("sample_id", "sample", "sample_name", "experimental_sample", "source_sample") if c in work.columns), None)
    if explicit_sample:
        sample_ids = work[explicit_sample].astype(str).to_numpy()
    else:
        sample_ids = np.array([
            parse_sample_id_from_path(p, str(s)) for p, s in zip(work[crop_col], work[species_col])
        ], dtype=str)

    out = pd.DataFrame({
        "true_species": work[species_col].astype(str).to_numpy(),
        "sample_id": sample_ids,
        "crop_path": work[crop_col].astype(str).to_numpy(),
    })
    for optional in ("cell_id", "plate", "replicate", "field_id", "series"):
        if optional in work.columns:
            out[optional] = work[optional].to_numpy()
    for j, sp in enumerate(class_order):
        out[f"prob_{sp}"] = all_probs[:, j]
    out["pred_species"] = np.array(class_order, dtype=object)[all_probs.argmax(axis=1)]
    out["pred_confidence"] = all_probs.max(axis=1)
    out.to_csv(predictions_csv, index=False)

    np.savez_compressed(
        features_npz,
        features=feat_store,
        labels=np.asarray(labels_raw[feature_idx], dtype=np.str_),
        row_indices=feature_idx,
        crop_paths=np.asarray(work[crop_col].astype(str).to_numpy()[feature_idx], dtype=np.str_),
    )
    logging.info("Saved predictions: %s", predictions_csv)
    logging.info("Saved balanced penultimate features: %s (%s cells)", features_npz, f"{len(feature_idx):,}")
    return predictions_csv, features_npz


def ensure_holdout_predictions_and_features() -> Tuple[Path, Path]:
    # Reuse a previously generated probability CSV without moving/deleting it.
    if not PREDICTIONS_CSV.exists() and LEGACY_PREDICTIONS_CSV.exists() and not FORCE_REINFER:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logging.info("Copying existing validated holdout prediction CSV into Chris-analysis output directory.")
        shutil.copy2(LEGACY_PREDICTIONS_CSV, PREDICTIONS_CSV)

    holdout = pd.read_csv(HOLDOUT_CSV)
    species_col, crop_col = detect_species_crop_columns(holdout)
    holdout = holdout[holdout[species_col].astype(str).isin(SPECIES)].copy().reset_index(drop=True)
    return infer_dataframe(
        holdout, species_col, crop_col, PREDICTIONS_CSV, FEATURES_NPZ,
        feature_max_per_species=KNOWN_FEATURES_PER_SPECIES,
        extra_bases=[HOLDOUT_CSV.parent, DATA_ROOT],
        allow_reuse_predictions=True,
    )


def load_holdout_predictions() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(PREDICTIONS_CSV)
    prob_cols = [f"prob_{s}" for s in SPECIES]
    missing = [c for c in prob_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing probability columns: {missing}")
    df = df[df["true_species"].isin(SPECIES)].copy().reset_index(drop=True)
    P = normalise_probabilities(df[prob_cols].to_numpy())
    true_idx = np.array([SPECIES_TO_IDX[s] for s in df["true_species"]], dtype=np.int16)
    pred_idx = P.argmax(axis=1).astype(np.int16)
    sample_ids = df["sample_id"].astype(str).to_numpy()
    return df, P, true_idx, pred_idx, sample_ids


# =============================================================================
# 7. ROUTING METRICS
# =============================================================================

def mapping_matrix(mapping_idx: np.ndarray) -> np.ndarray:
    M = np.zeros((len(SPECIES), 8), dtype=np.float32)
    M[np.arange(len(SPECIES)), mapping_idx.astype(int)] = 1.0
    return M


def calculate_observed_metrics(P, true_idx, pred_idx, mapping_idx=REAL_MAPPING_IDX) -> Dict[str, float]:
    true_group = mapping_idx[true_idx]
    top1_group_from_species = mapping_idx[pred_idx]
    Pg = P @ mapping_matrix(mapping_idx)
    pred_group = Pg.argmax(axis=1)
    top2 = np.argpartition(Pg, kth=Pg.shape[1]-2, axis=1)[:, -2:]
    top2_ok = np.any(top2 == true_group[:, None], axis=1)
    species_wrong = pred_idx != true_idx
    rescue = np.mean(top1_group_from_species[species_wrong] == true_group[species_wrong])

    # Macro-F1 for the pooled routing prediction.
    routing_macro_f1 = f1_score(true_group, pred_group, labels=np.arange(8), average="macro", zero_division=0)
    return {
        "species_accuracy": float(np.mean(pred_idx == true_idx)),
        "routing_top1_species_mapping_accuracy": float(np.mean(top1_group_from_species == true_group)),
        "routing_probability_pooling_accuracy": float(np.mean(pred_group == true_group)),
        "routing_probability_pooling_macro_f1": float(routing_macro_f1),
        "routing_top2_accuracy": float(np.mean(top2_ok)),
        "species_error_rescue_fraction": float(rescue),
        "n_cells": int(len(P)),
        "n_species_errors": int(species_wrong.sum()),
        "n_species_errors_rescued": int(np.sum((top1_group_from_species == true_group) & species_wrong)),
    }


def generate_null_mappings(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((n, len(SPECIES)), dtype=np.int16)
    for i in range(n):
        out[i] = rng.permutation(NULL_GROUP_SLOTS)
    return out


def topology_null_from_confusion(true_idx, pred_idx, mappings: np.ndarray) -> pd.DataFrame:
    C = confusion_matrix(true_idx, pred_idx, labels=np.arange(len(SPECIES))).astype(np.int64)
    total = C.sum()
    errors = total - np.trace(C)
    rows = []
    for i, m in enumerate(mappings):
        same = m[:, None] == m[None, :]
        mapped_correct = int(C[same].sum())
        rescued = int((C * same).sum() - np.trace(C))
        rows.append((i, mapped_correct / total, rescued / errors if errors else np.nan))
    return pd.DataFrame(rows, columns=["permutation", "mapping_accuracy", "rescue_fraction"])


def pooled_null_gpu(P, true_idx, mappings: np.ndarray, batch_size: int) -> pd.DataFrame:
    """Evaluate probability-pooling and top-2 nulls in mapping batches."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Pooling-null device: %s", device)
    P_t = torch.from_numpy(P.astype(np.float32, copy=False)).to(device)
    true_t = torch.from_numpy(true_idx.astype(np.int64, copy=False)).to(device)
    results = []

    for start in range(0, len(mappings), batch_size):
        maps_np = mappings[start:start+batch_size].astype(np.int64)
        B = len(maps_np)
        maps = torch.from_numpy(maps_np).to(device)
        # [B,13,8] one-hot species->pseudo-group, flattened for one GEMM.
        M = torch.zeros((B, len(SPECIES), 8), device=device, dtype=torch.float32)
        species_axis = torch.arange(len(SPECIES), device=device).view(1, -1).expand(B, -1)
        batch_axis = torch.arange(B, device=device).view(-1, 1).expand(-1, len(SPECIES))
        M[batch_axis, species_axis, maps] = 1.0
        Mflat = M.permute(1, 0, 2).reshape(len(SPECIES), B * 8)
        G = (P_t @ Mflat).reshape(len(P), B, 8)
        true_g = maps[:, true_t].T  # [N,B]
        pred_g = G.argmax(dim=2)
        acc = (pred_g == true_g).float().mean(dim=0)
        top2 = G.topk(k=2, dim=2).indices
        top2_ok = (top2 == true_g.unsqueeze(-1)).any(dim=2).float().mean(dim=0)
        acc_np = acc.detach().cpu().numpy()
        top2_np = top2_ok.detach().cpu().numpy()
        for j in range(B):
            results.append((start+j, float(acc_np[j]), float(top2_np[j])))
        del M, Mflat, G, true_g, pred_g, top2, top2_ok
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if start == 0 or (start // batch_size) % 10 == 0:
            logging.info("Pooled null: %d/%d mappings", min(start+B, len(mappings)), len(mappings))

    return pd.DataFrame(results, columns=["permutation", "pooling_accuracy", "top2_accuracy"])


def run_routing_nulls(P, true_idx, pred_idx, observed: Dict[str, float]) -> pd.DataFrame:
    logging.info("Stage 3: group-size-matched routing nulls")
    mappings = generate_null_mappings(N_NULL_TOPOLOGY, RANDOM_SEED + 101)
    topo = topology_null_from_confusion(true_idx, pred_idx, mappings)
    pooled_maps = mappings[:N_NULL_POOLING]
    pooled = pooled_null_gpu(P, true_idx, pooled_maps, NULL_GPU_BATCH)
    null_df = topo.merge(pooled, on="permutation", how="left")
    null_df.to_csv(OUTPUT_DIR / "routing_group_size_matched_nulls.csv", index=False)

    summary_rows = []
    metric_pairs = [
        ("mapping_accuracy", "routing_top1_species_mapping_accuracy"),
        ("pooling_accuracy", "routing_probability_pooling_accuracy"),
        ("top2_accuracy", "routing_top2_accuracy"),
        ("rescue_fraction", "species_error_rescue_fraction"),
    ]
    for null_col, obs_key in metric_pairs:
        vals = null_df[null_col].dropna().to_numpy(dtype=float)
        obs = observed[obs_key]
        summary_rows.append({
            "metric": obs_key,
            "observed": obs,
            "null_n": len(vals),
            "null_mean": float(vals.mean()),
            "null_sd": float(vals.std(ddof=1)),
            "null_p025": float(np.quantile(vals, 0.025)),
            "null_p975": float(np.quantile(vals, 0.975)),
            "empirical_p_ge": empirical_p_ge(vals, obs),
            "percentile_of_observed": float(100 * np.mean(vals < obs)),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "routing_null_summary.csv", index=False)

    # Plot four independent panels.
    for null_col, obs_key in metric_pairs:
        vals = null_df[null_col].dropna().to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        ax.hist(vals, bins=40)
        ax.axvline(observed[obs_key], linestyle="--", linewidth=2, label=f"Observed = {observed[obs_key]:.3f}")
        ax.set_xlabel(obs_key.replace("_", " "))
        ax.set_ylabel("Random groupings")
        ax.legend()
        ax.set_title("Group-size-matched random routing null")
        savefig(fig, OUTPUT_DIR, f"null_{null_col}")

    return summary


# =============================================================================
# 8. SAMPLE-CLUSTER BOOTSTRAP
# =============================================================================

def run_cluster_bootstrap(df, P, true_idx, pred_idx, sample_ids, observed) -> pd.DataFrame:
    mapping = REAL_MAPPING_IDX
    true_g = mapping[true_idx]
    mapped_top1 = mapping[pred_idx]
    Pg = P @ mapping_matrix(mapping)
    pooled_pred = Pg.argmax(axis=1)
    top2 = np.argpartition(Pg, kth=6, axis=1)[:, -2:]
    top2_ok = np.any(top2 == true_g[:, None], axis=1)
    species_correct = pred_idx == true_idx
    species_wrong = ~species_correct
    rescue_event = species_wrong & (mapped_top1 == true_g)

    temp = pd.DataFrame({
        "sample_id": sample_ids,
        "n": 1,
        "species_correct": species_correct.astype(int),
        "mapping_correct": (mapped_top1 == true_g).astype(int),
        "pooling_correct": (pooled_pred == true_g).astype(int),
        "top2_correct": top2_ok.astype(int),
        "species_wrong": species_wrong.astype(int),
        "rescued": rescue_event.astype(int),
    })
    agg = temp.groupby("sample_id", sort=False).sum(numeric_only=True).reset_index()
    sample_table_path = OUTPUT_DIR / "bootstrap_sample_sufficient_statistics.csv"
    agg.to_csv(sample_table_path, index=False)

    A = agg[["n", "species_correct", "mapping_correct", "pooling_correct", "top2_correct", "species_wrong", "rescued"]].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(RANDOM_SEED + 202)
    B = np.empty((N_BOOTSTRAP, 5), dtype=np.float64)
    n_samples = len(A)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n_samples, size=n_samples)
        s = A[idx].sum(axis=0)
        B[b, 0] = s[1] / s[0]
        B[b, 1] = s[2] / s[0]
        B[b, 2] = s[3] / s[0]
        B[b, 3] = s[4] / s[0]
        B[b, 4] = s[6] / s[5] if s[5] else np.nan

    cols = [
        "species_accuracy",
        "routing_top1_species_mapping_accuracy",
        "routing_probability_pooling_accuracy",
        "routing_top2_accuracy",
        "species_error_rescue_fraction",
    ]
    boot_df = pd.DataFrame(B, columns=cols)
    boot_df.to_csv(OUTPUT_DIR / "sample_cluster_bootstrap_replicates.csv.gz", index=False, compression="gzip")
    rows = []
    for c in cols:
        lo, hi = percentile_ci(boot_df[c].to_numpy())
        rows.append({"metric": c, "observed": observed[c], "bootstrap_mean": boot_df[c].mean(), "ci95_low": lo, "ci95_high": hi, "n_samples": n_samples, "n_bootstrap": N_BOOTSTRAP})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "sample_cluster_bootstrap_summary.csv", index=False)
    return summary


# =============================================================================
# 9. 50:50 SYNTHETIC MIXTURE RECREATION + RANDOM-GROUP NULL
# =============================================================================

@dataclass
class SampleProbState:
    arrays: Dict[Tuple[str, str], np.ndarray]
    species_samples: Dict[str, List[str]]


def build_sample_prob_state(df: pd.DataFrame, P: np.ndarray) -> SampleProbState:
    arrays = {}
    species_samples = defaultdict(list)
    tmp = df[["true_species", "sample_id"]].copy()
    tmp["_row"] = np.arange(len(tmp))
    for (sp, sid), g in tmp.groupby(["true_species", "sample_id"], sort=False):
        arr = P[g["_row"].to_numpy(dtype=int)]
        arrays[(sp, str(sid))] = arr
        species_samples[sp].append(str(sid))
    for sp in SPECIES:
        species_samples[sp] = sorted(species_samples[sp])
    return SampleProbState(arrays, dict(species_samples))


def select_sample_pairs(state: SampleProbState, sp_a: str, sp_b: str, seed: int) -> List[Tuple[str, str]]:
    all_pairs = list(itertools.product(state.species_samples[sp_a], state.species_samples[sp_b]))
    if len(all_pairs) <= MIXTURE_MAX_SAMPLE_PAIRS:
        return all_pairs
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_pairs), size=MIXTURE_MAX_SAMPLE_PAIRS, replace=False)
    return [all_pairs[i] for i in idx]


def ensure_balanced_mixture_aggregates(df, P) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    cache = OUTPUT_DIR / "balanced_200cell_mixture_aggregate_probabilities.npz"
    meta_csv = OUTPUT_DIR / "balanced_200cell_mixture_metadata.csv"
    if cache.exists() and meta_csv.exists():
        z = np.load(cache)
        meta = pd.read_csv(meta_csv)
        logging.info("Reusing cached balanced mixture aggregate probabilities: %s", cache)
        return z["agg_probs"], z["species_a_idx"], z["species_b_idx"], meta

    state = build_sample_prob_state(df, P)
    agg_probs = []
    a_idx = []
    b_idx = []
    meta_rows = []
    half = MIXTURE_TOTAL_CELLS // 2
    if MIXTURE_TOTAL_CELLS % 2:
        raise ValueError("MIXTURE_TOTAL_CELLS must be even for 50:50 mixtures.")

    pairs = list(itertools.combinations(SPECIES, 2))
    for pair_i, (sp_a, sp_b) in enumerate(pairs, start=1):
        pair_name = f"{sp_a}__{sp_b}"
        pair_seed = stable_seed(RANDOM_SEED, f"pair-select|{pair_name}")
        sample_pairs = select_sample_pairs(state, sp_a, sp_b, pair_seed)
        used_pairs = 0
        for sample_pair_i, (sid_a, sid_b) in enumerate(sample_pairs):
            arr_a = state.arrays[(sp_a, sid_a)]
            arr_b = state.arrays[(sp_b, sid_b)]
            if len(arr_a) < half or len(arr_b) < half:
                continue
            used_pairs += 1
            rng_label = f"mix|{sp_a}|{sid_a}|{sp_b}|{sid_b}|0.5|A|{MIXTURE_TOTAL_CELLS}"
            rng = np.random.default_rng(stable_seed(RANDOM_SEED, rng_label))
            for rep in range(MIXTURE_REPEATS):
                ia = rng.choice(len(arr_a), size=half, replace=False)
                ib = rng.choice(len(arr_b), size=half, replace=False)
                agg = (arr_a[ia].sum(axis=0) + arr_b[ib].sum(axis=0)) / MIXTURE_TOTAL_CELLS
                agg = agg / agg.sum()
                agg_probs.append(agg.astype(np.float32))
                a_idx.append(SPECIES_TO_IDX[sp_a])
                b_idx.append(SPECIES_TO_IDX[sp_b])
                meta_rows.append({
                    "species_a": sp_a, "species_b": sp_b,
                    "sample_a": sid_a, "sample_b": sid_b,
                    "sample_pair_index": sample_pair_i, "repeat": rep,
                })
        logging.info("Mixture pair %d/78 %s: %d eligible source sample-pairs", pair_i, pair_name, used_pairs)

    A = np.vstack(agg_probs).astype(np.float32)
    ai = np.asarray(a_idx, dtype=np.int16)
    bi = np.asarray(b_idx, dtype=np.int16)
    meta = pd.DataFrame(meta_rows)
    np.savez_compressed(cache, agg_probs=A, species_a_idx=ai, species_b_idx=bi)
    meta.to_csv(meta_csv, index=False)
    logging.info("Saved %s balanced mixture repeats", f"{len(A):,}")
    return A, ai, bi, meta


def mixture_observed_metrics(A, ai, bi) -> Dict[str, float]:
    # Exact species top-2 recovery.
    top2_species = np.argpartition(A, kth=A.shape[1]-2, axis=1)[:, -2:]
    dual_species = (
        ((top2_species[:, 0] == ai) & (top2_species[:, 1] == bi)) |
        ((top2_species[:, 0] == bi) & (top2_species[:, 1] == ai))
    )

    ga, gb = REAL_MAPPING_IDX[ai], REAL_MAPPING_IDX[bi]
    same = ga == gb
    Pg = A @ mapping_matrix(REAL_MAPPING_IDX)
    pred_top1_g = Pg.argmax(axis=1)
    top2_g = np.argpartition(Pg, kth=Pg.shape[1]-2, axis=1)[:, -2:]
    dual_group = (
        ((top2_g[:, 0] == ga) & (top2_g[:, 1] == gb)) |
        ((top2_g[:, 0] == gb) & (top2_g[:, 1] == ga))
    )
    out = {
        "different_group_n_repeats": int((~same).sum()),
        "same_group_n_repeats": int(same.sum()),
        "different_group_dual_species_top2": float(dual_species[~same].mean()),
        "different_group_dual_routing_top2": float(dual_group[~same].mean()),
        "same_group_dual_species_top2": float(dual_species[same].mean()),
        "same_group_shared_routing_top1": float((pred_top1_g[same] == ga[same]).mean()),
    }
    return out


def mixture_null_gpu(A, ai, bi, mappings: np.ndarray, batch_size: int) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Mixture-null device: %s", device)
    P_t = torch.from_numpy(A.astype(np.float32, copy=False)).to(device)
    ai_t = torch.from_numpy(ai.astype(np.int64, copy=False)).to(device)
    bi_t = torch.from_numpy(bi.astype(np.int64, copy=False)).to(device)
    out = np.empty(len(mappings), dtype=np.float64)

    for start in range(0, len(mappings), batch_size):
        maps_np = mappings[start:start+batch_size].astype(np.int64)
        B = len(maps_np)
        maps = torch.from_numpy(maps_np).to(device)
        M = torch.zeros((B, len(SPECIES), 8), device=device, dtype=torch.float32)
        species_axis = torch.arange(len(SPECIES), device=device).view(1, -1).expand(B, -1)
        batch_axis = torch.arange(B, device=device).view(-1, 1).expand(-1, len(SPECIES))
        M[batch_axis, species_axis, maps] = 1.0
        Mflat = M.permute(1, 0, 2).reshape(len(SPECIES), B*8)
        G = (P_t @ Mflat).reshape(len(A), B, 8)
        ga = maps[:, ai_t].T
        gb = maps[:, bi_t].T
        diff = ga != gb
        top2 = G.topk(k=2, dim=2).indices
        hit_a = (top2 == ga.unsqueeze(-1)).any(dim=2)
        hit_b = (top2 == gb.unsqueeze(-1)).any(dim=2)
        success = hit_a & hit_b & diff
        numer = success.sum(dim=0).float()
        denom = diff.sum(dim=0).float()
        rates = torch.where(denom > 0, numer/denom, torch.nan)
        out[start:start+B] = rates.detach().cpu().numpy()
        del M, Mflat, G, ga, gb, diff, top2, hit_a, hit_b, success
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if start == 0 or (start // batch_size) % 10 == 0:
            logging.info("Mixture null: %d/%d mappings", min(start+B, len(mappings)), len(mappings))
    return out


def run_mixture_null(df, P) -> Dict[str, float]:
    logging.info("Stage 5: balanced-mixture random-routing null")
    A, ai, bi, meta = ensure_balanced_mixture_aggregates(df, P)
    obs = mixture_observed_metrics(A, ai, bi)
    write_json(OUTPUT_DIR / "balanced_mixture_observed_metrics.json", obs)

    maps = generate_null_mappings(N_NULL_MIXTURE, RANDOM_SEED + 303)
    null = mixture_null_gpu(A, ai, bi, maps, MIXTURE_NULL_GPU_BATCH)
    pd.DataFrame({"permutation": np.arange(len(null)), "dual_routing_top2_accuracy": null}).to_csv(
        OUTPUT_DIR / "balanced_mixture_group_size_matched_null.csv", index=False
    )
    summary = {
        **obs,
        "random_group_null_n": int(np.isfinite(null).sum()),
        "random_group_null_mean": float(np.nanmean(null)),
        "random_group_null_sd": float(np.nanstd(null, ddof=1)),
        "random_group_null_p025": float(np.nanquantile(null, 0.025)),
        "random_group_null_p975": float(np.nanquantile(null, 0.975)),
        "random_group_empirical_p_ge_observed": empirical_p_ge(null, obs["different_group_dual_routing_top2"]),
    }
    write_json(OUTPUT_DIR / "balanced_mixture_null_summary.json", summary)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.hist(null[np.isfinite(null)], bins=40)
    ax.axvline(obs["different_group_dual_routing_top2"], linestyle="--", linewidth=2,
               label=f"Observed = {obs['different_group_dual_routing_top2']:.3f}")
    ax.set_xlabel("Both pseudo-routing groups recovered in Top-2")
    ax.set_ylabel("Random groupings")
    ax.legend()
    ax.set_title("50:50, 200-cell mixtures: group-size-matched null")
    savefig(fig, OUTPUT_DIR, "balanced_mixture_routing_null")
    return summary


# =============================================================================
# 10. UMAP / PCA / KNN FEATURE ANALYSIS
# =============================================================================

def discover_unseen_manifest(user_path: Optional[str]) -> Optional[Path]:
    if user_path:
        p = Path(user_path)
        if not p.exists():
            raise FileNotFoundError(f"--unseen-manifest does not exist: {p}")
        return p

    # First use the established unseen/novel ZIP workflow. This makes the
    # default invocation genuinely one-command when that archive is present.
    novel_root = ensure_novel_dataset()
    if novel_root is not None:
        p = novel_root / "cell_manifest.csv"
        if p.exists():
            logging.info("Using unseen/novel cell manifest: %s", p)
            return p

    candidates = [
        PERSISTENT_TRILLIAN_DIR / "unseen_manifest.csv",
        PERSISTENT_TRILLIAN_DIR / "unseen_cells.csv",
        PERSISTENT_TRILLIAN_DIR / "Unseen_species" / "cell_manifest.csv",
        PERSISTENT_TRILLIAN_DIR / "Unseen_species" / "unseen_manifest.csv",
        PERSISTENT_TRILLIAN_DIR / "Unseen_species_analysis" / "unseen_manifest.csv",
        DATA_ROOT / "unseen_manifest.csv",
        DATA_ROOT / "splits" / "unseen.csv",
    ]
    for p in candidates:
        if p.exists():
            logging.info("Auto-detected unseen manifest: %s", p)
            return p
    return None


def prepare_unseen_manifest(path: Path) -> Tuple[pd.DataFrame, str, str]:
    df = pd.read_csv(path)
    species_col, crop_col = detect_species_crop_columns(df)
    normalized = []
    for x in df[species_col].astype(str):
        key = re.sub(r"[^A-Za-z0-9]+", "", x).upper()
        normalized.append(UNSEEN_ALIASES.get(key, key))
    df = df.copy()
    df[species_col] = normalized
    df = df[df[species_col].isin(sorted(set(UNSEEN_ALIASES.values())))].copy().reset_index(drop=True)
    if df.empty:
        raise ValueError(f"Unseen manifest contains no recognized unseen codes: {path}")
    return df, species_col, crop_col


def run_unseen_inference(unseen_manifest: Optional[Path]) -> Tuple[Optional[Path], Optional[Path]]:
    if unseen_manifest is None:
        logging.warning("No unseen manifest found; unseen UMAP/kNN stage will be skipped.")
        return None, None
    unseen_df, sp_col, crop_col = prepare_unseen_manifest(unseen_manifest)
    unseen_bases = [unseen_manifest.parent, PERSISTENT_TRILLIAN_DIR, DATA_ROOT] + ([NOVEL_DATA_ROOT] if NOVEL_DATA_ROOT else [])

    # Cheap preflight: resolve one crop from each unseen species before spawning
    # DataLoader workers.  This makes Windows->ARC path problems fail immediately
    # with a clear message rather than several minutes into GPU inference.
    for unseen_code, grp in unseen_df.groupby(sp_col, sort=True):
        example_value = grp.iloc[0][crop_col]
        resolved = resolve_crop_path(example_value, unseen_bases)
        logging.info("Unseen path preflight %s: %s -> %s", unseen_code, example_value, resolved)

    pred_path = OUTPUT_DIR / "unseen_DenseNet121_concat_cell_probabilities.csv"
    feat_path = OUTPUT_DIR / "unseen_balanced_penultimate_features.npz"
    return infer_dataframe(
        unseen_df, sp_col, crop_col, pred_path, feat_path,
        feature_max_per_species=UNSEEN_FEATURES_PER_SPECIES,
        extra_bases=unseen_bases,
        allow_reuse_predictions=True,
    )


def load_feature_arrays_compat(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load features + labels from a feature NPZ.

    New caches are written with Unicode string arrays and therefore load with
    allow_pickle=False. Some caches produced by the previous version of this
    script used pandas-derived dtype=object for labels/crop_paths. Those files
    are our own locally generated, trusted caches, so we support them via a
    narrowly scoped compatibility fallback to allow_pickle=True.

    This means an ARC rerun can reuse the already-computed 20,000-cell unseen
    feature cache instead of repeating GPU inference merely to change string
    dtype metadata.
    """
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as z:
            X = z["features"].astype(np.float32)
            y = np.asarray(z["labels"]).astype(str)
        return X, y
    except ValueError as exc:
        if "Object arrays cannot be loaded when allow_pickle=False" not in str(exc):
            raise
        logging.warning(
            "Legacy feature cache contains object-typed strings; loading trusted "
            "self-generated cache with allow_pickle=True: %s", path
        )
        with np.load(path, allow_pickle=True) as z:
            X = z["features"].astype(np.float32)
            y = np.asarray(z["labels"]).astype(str)
        return X, y


def run_feature_space_analysis(known_features_npz: Path, unseen_features_npz: Optional[Path]) -> None:
    logging.info("Stage 6: PCA/UMAP/kNN feature-space analysis")
    Xk, yk = load_feature_arrays_compat(known_features_npz)

    pca = PCA(n_components=min(50, Xk.shape[1], len(Xk)-1), random_state=RANDOM_SEED)
    Xk_pca = pca.fit_transform(Xk)
    pd.DataFrame({
        "pc": np.arange(1, len(pca.explained_variance_ratio_)+1),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
    }).to_csv(OUTPUT_DIR / "feature_PCA_variance.csv", index=False)

    Xu_pca = None
    yu = None
    if unseen_features_npz is not None and unseen_features_npz.exists():
        Xu, yu = load_feature_arrays_compat(unseen_features_npz)
        Xu_pca = pca.transform(Xu)

        # Quantitative kNN in 50-D PCA space, not distances on a 2-D embedding.
        knn = NearestNeighbors(n_neighbors=min(25, len(Xk)), metric="cosine")
        knn.fit(Xk_pca)
        distances, indices = knn.kneighbors(Xu_pca)
        neighbor_labels = yk[indices]
        rows = []
        for unseen in sorted(np.unique(yu)):
            mask = yu == unseen
            labs = neighbor_labels[mask].reshape(-1)
            vals, counts = np.unique(labs, return_counts=True)
            total = counts.sum()
            for lab, count in sorted(zip(vals, counts), key=lambda x: -x[1]):
                rows.append({
                    "unseen_species": unseen,
                    "known_neighbor_species": lab,
                    "neighbor_count": int(count),
                    "neighbor_fraction": float(count/total),
                    "k": int(neighbor_labels.shape[1]),
                    "n_unseen_cells": int(mask.sum()),
                })
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / "unseen_knn_in_PCA50_space.csv", index=False)

    if not HAVE_UMAP:
        msg = (
            "umap-learn is not installed in this ARC Python environment. All other analyses completed.\n"
            "Install/load umap-learn in the same environment, then rerun the script; cached inference and nulls will be reused.\n"
            "Typical command in an environment with pip access: python -m pip install --user umap-learn\n"
        )
        (OUTPUT_DIR / "UMAP_SKIPPED.txt").write_text(msg, encoding="utf-8")
        logging.warning(msg.strip())
        return

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.2, metric="cosine", random_state=RANDOM_SEED)
    Uk = reducer.fit_transform(Xk_pca)
    known_out = pd.DataFrame({"UMAP1": Uk[:,0], "UMAP2": Uk[:,1], "species": yk})
    known_out.to_csv(OUTPUT_DIR / "known_species_UMAP_coordinates.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    for sp in SPECIES:
        m = yk == sp
        if m.any():
            ax.scatter(Uk[m,0], Uk[m,1], s=4, alpha=0.35, label=sp)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title("Known holdout species — UMAP of PCA-reduced penultimate features")
    ax.legend(markerscale=3, fontsize=7, ncol=2)
    savefig(fig, OUTPUT_DIR, "known_species_UMAP")

    if Xu_pca is not None:
        Uu = reducer.transform(Xu_pca)
        unseen_out = pd.DataFrame({"UMAP1": Uu[:,0], "UMAP2": Uu[:,1], "species": yu})
        unseen_out.to_csv(OUTPUT_DIR / "unseen_species_UMAP_coordinates.csv", index=False)

        fig, ax = plt.subplots(figsize=(8.4, 7.2))
        ax.scatter(Uk[:,0], Uk[:,1], s=3, alpha=0.12, label="Known SC-MAPS cells")
        markers = ["x", "+", "1", "2", "3", "4", "*"]
        for i, sp in enumerate(sorted(np.unique(yu))):
            m = yu == sp
            ax.scatter(Uu[m,0], Uu[m,1], s=10, alpha=0.65, marker=markers[i % len(markers)], label=sp)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.set_title("Unseen organisms projected into known-species UMAP reference")
        ax.legend(fontsize=8, ncol=2)
        savefig(fig, OUTPUT_DIR, "known_plus_unseen_UMAP")


# =============================================================================
# 11. FIGURE 1C/1D ALIGNED PHENOTYPE FEATURES
# =============================================================================

FEATURE_ALIASES = {
    "Cell area": ["area", "cellarea", "maskarea", "regionarea"],
    "Major axis length": ["majoraxislength", "axismajorlength", "majoraxis"],
    "Minor axis length": ["minoraxislength", "axisminorlength", "minoraxis"],
    "Max. Feret diameter": ["feretdiametermax", "maxferetdiameter", "feretmax", "maximumferetdiameter"],
    "Aspect ratio": ["aspectratio", "majorminorratio"],
    "Eccentricity": ["eccentricity"],
    "Solidity": ["solidity"],
    "Compactness": ["compactness"],
    "Phase contrast": ["phasecontrastcontrast", "phasecontrast", "phasecontrastmeancontrast", "pccontrast"],
    "Bright field": ["brightfieldcontrast", "bfcontrast", "brightfield"],
    "DAPI": ["dapicontrast", "dapimeancontrast", "dapi"],
    "WGA-AF647": ["wgacontrast", "wgaaf647contrast", "wgaf647contrast", "wgaaf647"],
    "FM1-43": ["fm1contrast", "fm143contrast", "fm1meancontrast", "fm143"],
}


def find_best_column(columns: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    norm = {_norm_text(c): c for c in columns}
    # Exact normalized alias first.
    for a in aliases:
        na = _norm_text(a)
        if na in norm:
            return norm[na]
    # Then containment, preferring columns containing 'contrast' when alias says contrast.
    candidates = []
    for nc, original in norm.items():
        for a in aliases:
            na = _norm_text(a)
            if na and (na in nc or nc in na):
                score = abs(len(nc) - len(na))
                if "contrast" in na and "contrast" not in nc:
                    score += 100
                candidates.append((score, original))
    return sorted(candidates)[0][1] if candidates else None


def detect_id_column(columns, names) -> Optional[str]:
    norm = {_norm_text(c): c for c in columns}
    for n in names:
        if _norm_text(n) in norm:
            return norm[_norm_text(n)]
    return None


def nested_ss_decomposition(df: pd.DataFrame, species_col: str, plate_col: str, feature: str) -> Dict[str, float]:
    xdf = df[[species_col, plate_col, feature]].dropna().copy()
    xdf[feature] = pd.to_numeric(xdf[feature], errors="coerce")
    xdf = xdf.dropna()
    if len(xdf) < 2:
        return {"species_pct": np.nan, "strain_prep_pct": np.nan, "cell_pct": np.nan}
    grand = xdf[feature].mean()
    ss_total = float(((xdf[feature] - grand)**2).sum())
    if ss_total <= 0:
        return {"species_pct": 0.0, "strain_prep_pct": 0.0, "cell_pct": 0.0}

    species_stats = xdf.groupby(species_col)[feature].agg(["count", "mean"])
    ss_species = float((species_stats["count"] * (species_stats["mean"] - grand)**2).sum())

    sp_plate = xdf.groupby([species_col, plate_col])[feature].agg(["count", "mean"]).reset_index()
    sp_mean_map = species_stats["mean"].to_dict()
    ss_plate = float(sum(row["count"] * (row["mean"] - sp_mean_map[row[species_col]])**2 for _, row in sp_plate.iterrows()))

    plate_means = xdf.groupby([species_col, plate_col])[feature].transform("mean")
    ss_cell = float(((xdf[feature] - plate_means)**2).sum())
    return {
        "species_pct": 100 * ss_species/ss_total,
        "strain_prep_pct": 100 * ss_plate/ss_total,
        "cell_pct": 100 * ss_cell/ss_total,
    }


def run_figure1_phenotype_alignment() -> None:
    logging.info("Stage 7: Figure-1 phenotype alignment")
    manifest = DATA_ROOT / "cell_manifest.csv"
    if not manifest.exists():
        logging.warning("Manifest missing; skipping Figure-1 phenotype rebuild: %s", manifest)
        return
    header = pd.read_csv(manifest, nrows=0)
    cols = list(header.columns)
    species_col = detect_id_column(cols, ["species", "true_species"])
    plate_col = detect_id_column(cols, ["plate", "plate_id", "biological_plate", "strain_plate"])
    if species_col is None or plate_col is None:
        logging.warning("Could not detect species/plate columns; skipping Figure-1 phenotype rebuild.")
        return

    resolved = {}
    for display, aliases in FEATURE_ALIASES.items():
        c = find_best_column(cols, aliases)
        if c is not None:
            resolved[display] = c
    pd.DataFrame([{"display_feature": k, "manifest_column": v} for k,v in resolved.items()]).to_csv(
        OUTPUT_DIR / "figure1_resolved_feature_columns.csv", index=False
    )
    if len(resolved) < 5:
        logging.warning("Only %d phenotype columns resolved; skipping plots. See resolved-column CSV.", len(resolved))
        return

    usecols = [species_col, plate_col] + list(dict.fromkeys(resolved.values()))
    feat = pd.read_csv(manifest, usecols=usecols, nrows=FIG1_MAX_ROWS)
    feat = feat[feat[species_col].astype(str).isin(SPECIES)].copy()

    # Derive aspect ratio only if no direct column resolved and major/minor exist.
    if "Aspect ratio" not in resolved and "Major axis length" in resolved and "Minor axis length" in resolved:
        feat["__derived_aspect_ratio"] = pd.to_numeric(feat[resolved["Major axis length"]], errors="coerce") / pd.to_numeric(feat[resolved["Minor axis length"]], errors="coerce").replace(0, np.nan)
        resolved["Aspect ratio"] = "__derived_aspect_ratio"

    display_order = [k for k in FEATURE_ALIASES.keys() if k in resolved]
    # Figure 1C: species medians, z-standardized across species for each feature.
    med = pd.DataFrame(index=SPECIES)
    for display in display_order:
        med[display] = feat.assign(_x=pd.to_numeric(feat[resolved[display]], errors="coerce")).groupby(species_col)["_x"].median().reindex(SPECIES)
    z = (med - med.mean(axis=0)) / med.std(axis=0, ddof=0).replace(0, np.nan)
    z.to_csv(OUTPUT_DIR / "figure1C_species_median_zscores.csv")

    fig, ax = plt.subplots(figsize=(8.6, max(5.5, 0.42*len(display_order))))
    im = ax.imshow(z.T.to_numpy(), aspect="auto")
    ax.set_xticks(np.arange(len(SPECIES))); ax.set_xticklabels(SPECIES, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(display_order))); ax.set_yticklabels(display_order)
    ax.set_xlabel("Species"); ax.set_title("Species-level phenotypic diversity")
    cbar = fig.colorbar(im, ax=ax); cbar.set_label("Standardised species median")
    savefig(fig, OUTPUT_DIR, "figure1C_aligned_species_medians")

    # Figure 1D: same feature order, nested descriptive SS decomposition.
    var_rows = []
    for display in display_order:
        d = nested_ss_decomposition(feat, species_col, plate_col, resolved[display])
        var_rows.append({"feature": display, **d})
    var_df = pd.DataFrame(var_rows)
    var_df.to_csv(OUTPUT_DIR / "figure1D_nested_variance_decomposition.csv", index=False)

    y = np.arange(len(var_df))
    fig, ax = plt.subplots(figsize=(8.6, max(5.5, 0.42*len(var_df))))
    left1 = var_df["species_pct"].to_numpy()
    left2 = left1 + var_df["strain_prep_pct"].to_numpy()
    ax.barh(y, var_df["species_pct"], label="Species")
    ax.barh(y, var_df["strain_prep_pct"], left=left1, label="Strain/preparation within species")
    ax.barh(y, var_df["cell_pct"], left=left2, label="Cell level")
    ax.set_yticks(y); ax.set_yticklabels(var_df["feature"])
    ax.invert_yaxis(); ax.set_xlim(0, 100)
    ax.set_xlabel("Proportion of total variance (%)")
    ax.set_title("Variance partitioning of the same phenotypic features")
    ax.legend(fontsize=8)
    savefig(fig, OUTPUT_DIR, "figure1D_aligned_variance_partition")


# =============================================================================
# 12. GENUINE REPEATED-SEED RESULT SUMMARISER
# =============================================================================

def detect_prob_columns_generic(df: pd.DataFrame, expected_labels: Sequence[str]) -> Optional[List[str]]:
    cols = []
    for lab in expected_labels:
        candidates = [f"prob_{lab}", f"p_{lab}", lab]
        c = next((x for x in candidates if x in df.columns), None)
        if c is None:
            return None
        cols.append(c)
    return cols


def summarize_seed_results(seed_dir: Optional[str]) -> None:
    logging.info("Stage 8: genuine repeated-seed result audit")
    out_txt = OUTPUT_DIR / "SEED_RETRAINING_REQUIRED.txt"
    if not seed_dir:
        out_txt.write_text(
            "Chris's random-initialisation question requires genuinely retrained models.\n\n"
            "This post-hoc script cannot reconstruct the original training stochasticity from one final checkpoint, "
            "and it intentionally does not substitute bootstrap resampling or MC dropout for retraining.\n\n"
            "If you have repeated-seed prediction CSVs, rerun with:\n"
            "  --seed-results-dir /path/to/seed_outputs\n\n"
            "Recommended filenames should contain 'species' or 'direct'/'panel' and 'seed'. Each species CSV should "
            "contain true_species plus prob_ACAA...prob_STCP; each direct-routing CSV should contain a true routing label "
            "and 8 routing probability columns. The script will summarise whatever genuine seed outputs it can identify.\n\n"
            "To automate NEW seed training, provide the exact original training .py program; that implementation is not "
            "recoverable faithfully from the manuscript/checkpoint alone.\n",
            encoding="utf-8",
        )
        logging.warning("No --seed-results-dir supplied. Genuine seed retraining cannot be inferred from one checkpoint.")
        return

    root = Path(seed_dir)
    if not root.exists():
        raise FileNotFoundError(f"--seed-results-dir does not exist: {root}")
    files = sorted(root.rglob("*.csv"))
    if not files:
        logging.warning("No CSV files found under seed-results directory: %s", root)
        return

    rows = []
    for path in files:
        name = path.name.lower()
        seed_match = re.search(r"seed[_-]?(\d+)", name)
        seed = int(seed_match.group(1)) if seed_match else np.nan
        try:
            sdf = pd.read_csv(path)
        except Exception:
            continue
        # Species-trained output.
        spcols = detect_prob_columns_generic(sdf, SPECIES)
        true_sp_col = next((c for c in ("true_species", "species", "true_label") if c in sdf.columns), None)
        if spcols and true_sp_col:
            w = sdf[sdf[true_sp_col].astype(str).isin(SPECIES)].copy()
            Ps = normalise_probabilities(w[spcols].to_numpy())
            ti = np.array([SPECIES_TO_IDX[x] for x in w[true_sp_col].astype(str)], dtype=np.int16)
            pi = Ps.argmax(axis=1).astype(np.int16)
            m = calculate_observed_metrics(Ps, ti, pi)
            rows.append({"file": str(path), "seed": seed, "task": "species_then_pool", "accuracy": m["routing_probability_pooling_accuracy"], "macro_f1": m["routing_probability_pooling_macro_f1"]})
            continue

        # Direct-routing output: accept normalized routing labels/probability columns.
        true_g_col = next((c for c in ("true_group", "true_panel", "routing_group", "panel", "true_label") if c in sdf.columns), None)
        if true_g_col:
            # Try exact prob_<group> first, otherwise normalized-name matching.
            norm_cols = {_norm_text(c): c for c in sdf.columns}
            gcols = []
            for g in ROUTING_GROUPS:
                options = [_norm_text("prob_"+g), _norm_text(g)]
                col = next((norm_cols[o] for o in options if o in norm_cols), None)
                if col is None:
                    gcols = []
                    break
                gcols.append(col)
            if gcols:
                labels = sdf[true_g_col].astype(str)
                # map exact group names only; avoid silently guessing clinical labels
                valid = labels.isin(ROUTING_GROUPS)
                if valid.any():
                    Pd = normalise_probabilities(sdf.loc[valid, gcols].to_numpy())
                    tg = np.array([ROUTING_TO_IDX[x] for x in labels[valid]], dtype=int)
                    pg = Pd.argmax(axis=1)
                    rows.append({"file": str(path), "seed": seed, "task": "direct_routing", "accuracy": float(np.mean(pg==tg)), "macro_f1": float(f1_score(tg, pg, labels=np.arange(8), average="macro", zero_division=0))})

    if not rows:
        logging.warning("No recognizable genuine seed prediction CSVs found in %s", root)
        return
    res = pd.DataFrame(rows)
    res.to_csv(OUTPUT_DIR / "seed_run_metrics.csv", index=False)
    summary = res.groupby("task")[["accuracy", "macro_f1"]].agg(["count", "mean", "std", "min", "max"])
    summary.to_csv(OUTPUT_DIR / "seed_run_summary.csv")

    # Paired seed differences if both tasks share numeric seed IDs.
    p = res.dropna(subset=["seed"]).pivot_table(index="seed", columns="task", values="accuracy", aggfunc="first")
    if {"species_then_pool", "direct_routing"}.issubset(p.columns):
        p = p.dropna(subset=["species_then_pool", "direct_routing"])
        p["difference_species_pool_minus_direct"] = p["species_then_pool"] - p["direct_routing"]
        p.to_csv(OUTPUT_DIR / "seed_paired_accuracy_differences.csv")


# =============================================================================
# 13. REPORT
# =============================================================================

def write_human_report(observed, null_summary, bootstrap_summary, mixture_summary, unseen_manifest):
    lines = []
    lines.append("SC-MAPS Chris review analysis summary")
    lines.append("="*42)
    lines.append("")
    lines.append("Observed fixed-model holdout metrics")
    for k,v in observed.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Group-size-matched routing nulls")
    lines.append(null_summary.to_string(index=False))
    lines.append("")
    lines.append("Sample-cluster bootstrap")
    lines.append(bootstrap_summary.to_string(index=False))
    lines.append("")
    lines.append("Balanced 50:50 / 200-cell mixture analysis")
    for k,v in mixture_summary.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append(f"Unseen manifest used: {unseen_manifest if unseen_manifest else 'NONE — unseen UMAP/kNN skipped'}")
    lines.append(f"UMAP available: {HAVE_UMAP}")
    lines.append("")
    lines.append("Naive uniform-chance references (descriptive only; permutation nulls are the primary controls):")
    lines.append(f"  13-species top-1: {1/13:.4f} ({100/13:.2f}%)")
    lines.append(f"  8-group top-1: {1/8:.4f} (12.50%)")
    lines.append(f"  8-group top-2: {2/8:.4f} (25.00%)")
    (OUTPUT_DIR / "RESULTS_SUMMARY.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 14. MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="SC-MAPS all-in-one Chris review analyses on ARC")
    p.add_argument("--unseen-manifest", default=None, help="Optional CSV containing unseen species + crop paths")
    p.add_argument("--seed-results-dir", default=None, help="Optional directory of genuine repeated-seed prediction CSVs")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(OUTPUT_DIR)
    t0 = time.time()
    logging.info("Starting SC-MAPS Chris-review analysis")
    logging.info("Source ZIP: %s", TRILLIAN_ZIP)
    logging.info("Scratch parent: %s", SCRATCH_PARENT)
    logging.info("Expected scratch dataset root: %s", DATA_ROOT)
    logging.info("Checkpoint: %s", DENSENET_CONCAT_CHECKPOINT)
    logging.info("Output: %s", OUTPUT_DIR)

    # Stage 0: correct scratch preparation.
    ensure_trillian_dataset()

    # Stage 1: fixed-model holdout probabilities + penultimate features.
    logging.info("Stage 1: holdout inference/features")
    ensure_holdout_predictions_and_features()
    df, P, true_idx, pred_idx, sample_ids = load_holdout_predictions()

    # Stage 2: observed metrics.
    logging.info("Stage 2: observed routing metrics")
    observed = calculate_observed_metrics(P, true_idx, pred_idx)
    write_json(OUTPUT_DIR / "observed_holdout_metrics.json", observed)
    logging.info("Observed metrics: %s", observed)

    # Sanity warnings against manuscript values (not hard failures).
    targets = {
        "species_accuracy": 0.654,
        "routing_top1_species_mapping_accuracy": 0.769,
        "routing_probability_pooling_accuracy": 0.771,
        "routing_top2_accuracy": 0.973,
        "species_error_rescue_fraction": 0.345,
    }
    for k,t in targets.items():
        if abs(observed[k] - t) > 0.015:
            logging.warning("Sanity check: %s=%.4f differs from manuscript ~%.4f by >1.5 pp", k, observed[k], t)

    # Stage 3: routing nulls.
    null_summary = run_routing_nulls(P, true_idx, pred_idx, observed)

    # Stage 4: sample-cluster bootstrap.
    logging.info("Stage 4: sample-cluster bootstrap")
    bootstrap_summary = run_cluster_bootstrap(df, P, true_idx, pred_idx, sample_ids, observed)

    # Stage 5: mixture null.
    mixture_summary = run_mixture_null(df, P)

    # Stage 6: unseen inference if a manifest can be found, then UMAP/kNN.
    unseen_manifest = discover_unseen_manifest(args.unseen_manifest)
    _, unseen_features = run_unseen_inference(unseen_manifest)
    run_feature_space_analysis(FEATURES_NPZ, unseen_features)

    # Stage 7: aligned Fig 1C/D analysis.
    run_figure1_phenotype_alignment()

    # Stage 8: genuine repeated-seed audit if outputs exist.
    summarize_seed_results(args.seed_results_dir)

    write_human_report(observed, null_summary, bootstrap_summary, mixture_summary, unseen_manifest)
    logging.info("Finished in %.1f minutes", (time.time()-t0)/60)
    logging.info("Main summary: %s", OUTPUT_DIR / "RESULTS_SUMMARY.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
