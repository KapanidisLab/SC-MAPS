#!/usr/bin/env python3
"""
SC-MAPS paired bootstrap analysis for direct 8-class AST-routing classification.

Purpose
-------
Compare, on the SAME independent Plate-1 holdout cells and the SAME bootstrap
resamples:

  1) Species-trained DenseNet121 fusion -> pool 13 species probabilities into
     the 8 AST-routing groups.
  2) DenseNet121 fusion trained DIRECTLY on the 8 AST-routing groups.

The script analyses both concatenation and SE-attention models automatically.
IMPORTANT: the SE species-trained model is the validated four-channel run
from 2026-07-21 (DAPI + PhaseContrast + WGA + FM1), which is the run used
for the manuscript 75.62% species-derived routing result.
It uses the holdout prediction CSVs already written by the original training
runs, so it does NOT retrain the models and does NOT need GPU inference.

Bootstrap design
----------------
- Unit of resampling: holdout sample / imaging-series unit, NOT individual cells.
- Paired: the same resampled samples are used for both models.
- Stratified by TRUE SPECIES: each bootstrap replicate preserves the number of
  sample units contributed by each of the 13 species.
- 10,000 bootstrap replicates.
- Reports 95% percentile confidence intervals for accuracy, balanced accuracy,
  macro-F1, weighted-F1, within-sample mean/median accuracy, and the paired
  species-pooled minus direct-model difference.
- Runs both on all holdout samples and, as a sensitivity analysis, on the fixed
  cohort of samples containing >=200 cells (expected ~120 samples).

No command-line arguments are required. On ARC run simply:

    python SC_MAPS_direct_panel_bootstrap_ARC_CORRECTED.py

Outputs are written to:
/data/phys-deep-learning-bacteria/phys3115/Bacteria-data-15 species/Trillian/
SCMAPS_direct_panel_bootstrap_ARC_CORRECTED/
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# USER/PROJECT PATHS -- already filled in for the SC-MAPS ARC project
# =============================================================================

BASE = Path(
    "/data/phys-deep-learning-bacteria/phys3115/"
    "Bacteria-data-15 species/Trillian"
)

RUN_DIRS = {
    "concat": {
        "species": BASE / (
            "Trillian_attention_fusion_training_outputs_ARC/"
            "fusion_training_outputs_CONCAT/"
            "20260706_152625_DAPI-PHASECONTRAST-WGA-FM1_concat"
        ),
        "direct": BASE / (
            "Trillian_8class_fusion_training_outputs_ARC/"
            "run_20260824_122834_panel2_8class_fusion_concat"
        ),
    },
    "se": {
        "species": BASE / (
            "Trillian_attention_fusion_training_outputs_ARC/"
            "run_20260721_160005_BF-DAPI-PHASECONTRAST-WGA-FM1_se_attention"
        ),
        "direct": BASE / (
            "Trillian_8class_fusion_training_outputs_ARC/"
            "run_20260824_122834_panel2_8class_fusion_se_attention"
        ),
    },
}

HOLDOUT_MANIFEST = Path(
    "/scratch/phys3115/bacteria_data/Trillian_DATASET_with_holdout/"
    "splits/holdout_plate1.csv"
)

OUTPUT_DIR = BASE / "SCMAPS_direct_panel_bootstrap_ARC_CORRECTED"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAPS = 10_000
RANDOM_SEED = 42
MIN_CELLS_SENSITIVITY = 200


# =============================================================================
# SC-MAPS class definitions
# =============================================================================

SPECIES = [
    "ACAA", "ECOL", "ENTC", "KLAE", "KLPN", "PSAR", "PSMA",
    "SAENA", "SAUR", "SEPI", "SERM", "SLUG", "STCP",
]

PANELS = [
    "Routine Enterobacterales",
    "AmpC/caution Enterobacterales",
    "Salmonella",
    "Pseudomonas",
    "Acinetobacter",
    "Stenotrophomonas",
    "S. aureus-like staphylococci",
    "CoNS staphylococci",
]

SPECIES_TO_PANEL = {
    "ECOL": "Routine Enterobacterales",
    "KLPN": "Routine Enterobacterales",
    "KLAE": "AmpC/caution Enterobacterales",
    "ENTC": "AmpC/caution Enterobacterales",
    "SERM": "AmpC/caution Enterobacterales",
    "SAENA": "Salmonella",
    "PSAR": "Pseudomonas",
    "ACAA": "Acinetobacter",
    "PSMA": "Stenotrophomonas",
    "SAUR": "S. aureus-like staphylococci",
    "SLUG": "S. aureus-like staphylococci",
    "SEPI": "CoNS staphylococci",
    "STCP": "CoNS staphylococci",
}

PANEL_TO_INDEX = {name: i for i, name in enumerate(PANELS)}

# Reference values from the existing manuscript/results. These are used only as
# a sanity check; the script always calculates the values afresh from CSVs.
EXPECTED_REFERENCE = {
    "concat": {"species_pooled": 0.7711, "direct": 0.7491},
    "se": {"species_pooled": 0.7562, "direct": 0.7500},
}


# =============================================================================
# Utility functions
# =============================================================================

def norm_text(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def canonical_panel(value: object) -> str:
    """Map the label spellings used in different scripts to one panel name."""
    n = norm_text(value)
    aliases = {
        "routineenterobacterales": "Routine Enterobacterales",
        "ampcenterobacterales": "AmpC/caution Enterobacterales",
        "ampccautionenterobacterales": "AmpC/caution Enterobacterales",
        "salmonella": "Salmonella",
        "pseudomonas": "Pseudomonas",
        "acinetobacter": "Acinetobacter",
        "stenotrophomonas": "Stenotrophomonas",
        "saureuslike": "S. aureus-like staphylococci",
        "saureuslikestaphylococci": "S. aureus-like staphylococci",
        "cons": "CoNS staphylococci",
        "consstaphylococci": "CoNS staphylococci",
        "coagulasenegativestaphylococci": "CoNS staphylococci",
    }
    if n in aliases:
        return aliases[n]

    # Exact canonical-name fallback after punctuation removal.
    for panel in PANELS:
        if norm_text(panel) == n:
            return panel

    raise ValueError(f"Unrecognised AST-routing label: {value!r}")


def path_key(value: object) -> str:
    """Create a root-independent stable key for the same cell crop."""
    s = str(value).strip().replace("\\", "/")
    marker = "Trillian_DATASET_with_holdout/"
    low = s.lower()
    mlow = marker.lower()
    if mlow in low:
        pos = low.index(mlow)
        s = s[pos + len(marker):]
    else:
        # If an old output used a different absolute root, the last 5 path
        # components are normally enough to retain the unique cell filename and
        # its species/plate context without depending on /scratch vs /data.
        parts = [p for p in s.split("/") if p]
        s = "/".join(parts[-5:]) if len(parts) >= 5 else "/".join(parts)
    return s.lower()


def parse_sample_id_from_path(path_text: object, species: str) -> str:
    """
    Reproduce the Plate-1 imaging-series grouping used in the previous SC-MAPS
    within-sample and aggregation analyses.

    Example:
      ACAA_ACAA_plate1_rep1_series001_T000_Z000_cell_0001.npy
      -> ACAA__plate1__rep1__series001
    """
    stem = Path(str(path_text)).name

    def grab(pattern: str, default: Optional[str]) -> Optional[str]:
        m = re.search(pattern, stem, flags=re.I)
        return m.group(1) if m else default

    plate = grab(r"plate[_-]?(\d+)", "1")
    rep = grab(r"rep[_-]?(\d+)", "1")
    series = grab(r"series[_-]?(\d+)", None)

    if series is None:
        base = re.sub(
            r"_cell[_-]?\d+.*$", "", Path(stem).stem, flags=re.I
        )
        return f"{species}__plate{plate}__rep{rep}__{base}"

    return f"{species}__plate{plate}__rep{rep}__series{int(series):03d}"


def find_prediction_csv(run_dir: Path) -> Path:
    """Find the feature-fusion Plate-1 prediction CSV produced by training."""
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist:\n{run_dir}")

    preferred = [
        run_dir / "holdout_feature_fusion_predictions.csv",
        run_dir / "holdout_predictions.csv",
    ]
    for p in preferred:
        if p.is_file():
            return p

    candidates = sorted(
        [
            p for p in run_dir.rglob("*.csv")
            if "holdout" in p.name.lower()
            and "prediction" in p.name.lower()
            and "fusion" in p.name.lower()
        ],
        key=lambda p: ("feature_fusion" in p.name.lower(), p.stat().st_mtime),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "Could not find a holdout fusion prediction CSV below:\n"
            f"{run_dir}\n"
            "Expected e.g. holdout_feature_fusion_predictions.csv"
        )
    return candidates[0]


def pick_column(df: pd.DataFrame, candidates: Iterable[str], purpose: str) -> str:
    by_lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in by_lower:
            return by_lower[c.lower()]
    raise ValueError(
        f"Could not identify {purpose} column. Columns are:\n{list(df.columns)}"
    )


def load_manifest_sample_map() -> Optional[Dict[str, str]]:
    """Use an explicit sample column from the holdout manifest when present."""
    if not HOLDOUT_MANIFEST.is_file():
        print(
            f"NOTE: holdout manifest not found at {HOLDOUT_MANIFEST}; "
            "sample IDs will be parsed from crop filenames.",
            flush=True,
        )
        return None

    m = pd.read_csv(HOLDOUT_MANIFEST, low_memory=False)
    try:
        path_col = pick_column(
            m,
            ["crop_path", "npy_path", "path", "file_path", "mask_crop_path"],
            "manifest crop-path",
        )
    except ValueError:
        return None

    sample_col = None
    for candidate in (
        "sample_id", "sample", "sample_name", "experimental_sample",
        "source_sample",
    ):
        if candidate in m.columns:
            sample_col = candidate
            break
    if sample_col is None:
        print(
            "Holdout manifest has no explicit sample column; using the same "
            "plate/rep/series filename parser as the prior aggregation analysis.",
            flush=True,
        )
        return None

    tmp = pd.DataFrame({
        "cell_key": m[path_col].map(path_key),
        "sample_id": m[sample_col].astype(str),
    })
    if tmp["cell_key"].duplicated().any():
        print(
            "WARNING: manifest path keys are not unique; ignoring explicit sample "
            "column and parsing sample IDs from filenames.",
            flush=True,
        )
        return None

    print(f"Using explicit sample IDs from manifest column: {sample_col}")
    return dict(zip(tmp["cell_key"], tmp["sample_id"]))


def load_species_predictions(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    path_col = pick_column(df, ["path", "crop_path", "npy_path", "file_path"], "path")
    truth_col = pick_column(df, ["true_label", "true_species", "species"], "true species")

    true_species = df[truth_col].astype(str).str.upper().str.strip()
    invalid = sorted(set(true_species) - set(SPECIES))
    if invalid:
        raise ValueError(
            f"Species-model CSV contains unexpected true labels: {invalid[:20]}"
        )

    missing = [f"prob_{s}" for s in SPECIES if f"prob_{s}" not in df.columns]
    if missing:
        raise ValueError(
            f"Species-model CSV is missing probability columns: {missing}"
        )

    probs = df[[f"prob_{s}" for s in SPECIES]].to_numpy(dtype=np.float64)
    if not np.isfinite(probs).all():
        raise ValueError("Species probabilities contain non-finite values")
    row_sums = probs.sum(axis=1)
    if np.quantile(np.abs(row_sums - 1.0), 0.99) > 0.02:
        raise ValueError("Species probability rows do not sum to ~1")
    probs = probs / row_sums[:, None]

    pooled = np.zeros((len(df), len(PANELS)), dtype=np.float64)
    for j, sp in enumerate(SPECIES):
        pooled[:, PANEL_TO_INDEX[SPECIES_TO_PANEL[sp]]] += probs[:, j]

    pred_idx = pooled.argmax(axis=1)
    out = pd.DataFrame({
        "cell_key": df[path_col].map(path_key),
        "path_species": df[path_col].astype(str),
        "true_species": true_species,
        "true_panel_idx": [PANEL_TO_INDEX[SPECIES_TO_PANEL[s]] for s in true_species],
        "pooled_pred_idx": pred_idx,
        "pooled_correct": pred_idx == np.asarray(
            [PANEL_TO_INDEX[SPECIES_TO_PANEL[s]] for s in true_species]
        ),
    })
    if out["cell_key"].duplicated().any():
        raise ValueError(f"Duplicate cell keys in species predictions: {csv_path}")
    return out


def load_direct_predictions(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    path_col = pick_column(df, ["path", "crop_path", "npy_path", "file_path"], "path")
    truth_col = pick_column(df, ["true_label", "true_panel", "label"], "true panel")
    pred_col = pick_column(df, ["predicted_label", "pred_panel", "prediction"], "predicted panel")

    true_panel = [canonical_panel(x) for x in df[truth_col]]
    pred_panel = [canonical_panel(x) for x in df[pred_col]]

    out = pd.DataFrame({
        "cell_key": df[path_col].map(path_key),
        "path_direct": df[path_col].astype(str),
        "direct_true_idx": [PANEL_TO_INDEX[x] for x in true_panel],
        "direct_pred_idx": [PANEL_TO_INDEX[x] for x in pred_panel],
    })
    out["direct_correct"] = out["direct_true_idx"] == out["direct_pred_idx"]
    if out["cell_key"].duplicated().any():
        raise ValueError(f"Duplicate cell keys in direct predictions: {csv_path}")
    return out


def confusion_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((len(PANELS), len(PANELS)), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def metrics_from_cm(cm: np.ndarray) -> Dict[str, float]:
    cm = np.asarray(cm, dtype=np.float64)
    total = cm.sum()
    if total <= 0:
        return {
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "macro_f1": np.nan,
            "weighted_f1": np.nan,
        }

    tp = np.diag(cm)
    support = cm.sum(axis=1)
    pred_support = cm.sum(axis=0)

    accuracy = float(tp.sum() / total)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, pred_support, out=np.zeros_like(tp), where=pred_support > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )

    present = support > 0
    balanced_accuracy = float(recall[present].mean())
    macro_f1 = float(f1[present].mean())
    weighted_f1 = float((f1 * support).sum() / support.sum())

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def ci(values: np.ndarray) -> Tuple[float, float]:
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def build_sample_table(matched: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    sample_ids = matched["sample_id"].astype(str).unique().tolist()
    pooled_cms = []
    direct_cms = []
    rows = []

    for sid in sample_ids:
        g = matched[matched["sample_id"] == sid]
        species_values = g["true_species"].unique()
        if len(species_values) != 1:
            raise ValueError(
                f"Sample {sid!r} contains multiple species: {species_values.tolist()}"
            )
        species = species_values[0]
        true_panel = SPECIES_TO_PANEL[species]
        y = g["true_panel_idx"].to_numpy(dtype=int)
        pp = g["pooled_pred_idx"].to_numpy(dtype=int)
        dp = g["direct_pred_idx"].to_numpy(dtype=int)
        pcm = confusion_from_arrays(y, pp)
        dcm = confusion_from_arrays(y, dp)
        pooled_cms.append(pcm)
        direct_cms.append(dcm)
        rows.append({
            "sample_id": sid,
            "species": species,
            "true_panel": true_panel,
            "n_cells": len(g),
            "species_pooled_accuracy": float((pp == y).mean()),
            "direct_accuracy": float((dp == y).mean()),
            "difference_pp": 100.0 * float((pp == y).mean() - (dp == y).mean()),
        })

    sample_df = pd.DataFrame(rows)
    return sample_df, np.stack(pooled_cms), np.stack(direct_cms)


def bootstrap_cohort(
    architecture: str,
    cohort_name: str,
    matched: pd.DataFrame,
    sample_df: pd.DataFrame,
    pooled_cms: np.ndarray,
    direct_cms: np.ndarray,
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Paired species-stratified cluster bootstrap over sample units."""
    rng = np.random.default_rng(RANDOM_SEED)

    # sample_df order is aligned to pooled_cms/direct_cms.
    cohort_sample_ids = set(matched["sample_id"].astype(str).unique())
    keep = sample_df["sample_id"].astype(str).isin(cohort_sample_ids).to_numpy()
    sdf = sample_df.loc[keep].reset_index(drop=True)
    pcms = pooled_cms[keep]
    dcms = direct_cms[keep]

    if sdf.empty:
        raise ValueError(f"Cohort {cohort_name} contains no samples")

    strata: Dict[str, np.ndarray] = {}
    for sp in SPECIES:
        idx = np.where(sdf["species"].to_numpy() == sp)[0]
        if len(idx) == 0:
            raise ValueError(
                f"Bootstrap cohort {cohort_name!r} contains no sample for species {sp}"
            )
        strata[sp] = idx

    # Observed metrics on the original, non-resampled cohort.
    observed_pcm = pcms.sum(axis=0)
    observed_dcm = dcms.sum(axis=0)
    obs_p = metrics_from_cm(observed_pcm)
    obs_d = metrics_from_cm(observed_dcm)
    obs_p_sample = sdf["species_pooled_accuracy"].to_numpy(float)
    obs_d_sample = sdf["direct_accuracy"].to_numpy(float)

    observed = {
        "species_pooled": {
            **obs_p,
            "mean_sample_accuracy": float(obs_p_sample.mean()),
            "median_sample_accuracy": float(np.median(obs_p_sample)),
        },
        "direct": {
            **obs_d,
            "mean_sample_accuracy": float(obs_d_sample.mean()),
            "median_sample_accuracy": float(np.median(obs_d_sample)),
        },
    }

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "mean_sample_accuracy",
        "median_sample_accuracy",
    ]
    draws = {
        "species_pooled": {m: np.empty(N_BOOTSTRAPS, dtype=np.float64) for m in metric_names},
        "direct": {m: np.empty(N_BOOTSTRAPS, dtype=np.float64) for m in metric_names},
    }

    for b in range(N_BOOTSTRAPS):
        selected_chunks = []
        for sp in SPECIES:
            idx = strata[sp]
            selected_chunks.append(rng.choice(idx, size=len(idx), replace=True))
        selected = np.concatenate(selected_chunks)

        pcm = pcms[selected].sum(axis=0)
        dcm = dcms[selected].sum(axis=0)
        mp = metrics_from_cm(pcm)
        md = metrics_from_cm(dcm)

        for m in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"):
            draws["species_pooled"][m][b] = mp[m]
            draws["direct"][m][b] = md[m]

        p_sample = sdf.iloc[selected]["species_pooled_accuracy"].to_numpy(float)
        d_sample = sdf.iloc[selected]["direct_accuracy"].to_numpy(float)
        draws["species_pooled"]["mean_sample_accuracy"][b] = p_sample.mean()
        draws["direct"]["mean_sample_accuracy"][b] = d_sample.mean()
        draws["species_pooled"]["median_sample_accuracy"][b] = np.median(p_sample)
        draws["direct"]["median_sample_accuracy"][b] = np.median(d_sample)

    # Save raw bootstrap draws, including paired differences.
    draw_df = pd.DataFrame({
        "bootstrap": np.arange(1, N_BOOTSTRAPS + 1),
    })
    for model in ("species_pooled", "direct"):
        for m in metric_names:
            draw_df[f"{model}_{m}"] = draws[model][m]
            if model == "direct":
                draw_df[f"difference_{m}"] = (
                    draws["species_pooled"][m] - draws["direct"][m]
                )
    draw_file = out_dir / f"bootstrap_draws_{architecture}_{cohort_name}.csv"
    draw_df.to_csv(draw_file, index=False)

    # Long-format summary with CIs.
    rows = []
    for model in ("species_pooled", "direct"):
        for m in metric_names:
            low, high = ci(draws[model][m])
            rows.append({
                "architecture": architecture,
                "cohort": cohort_name,
                "model": model,
                "metric": m,
                "observed": observed[model][m],
                "ci95_low": low,
                "ci95_high": high,
                "n_samples": len(sdf),
                "n_cells": int(matched.shape[0]),
                "n_bootstraps": N_BOOTSTRAPS,
                "bootstrap_unit": "sample/imaging-series",
                "bootstrap_stratification": "true species",
            })

    for m in metric_names:
        delta = draws["species_pooled"][m] - draws["direct"][m]
        low, high = ci(delta)
        obs_delta = observed["species_pooled"][m] - observed["direct"][m]
        rows.append({
            "architecture": architecture,
            "cohort": cohort_name,
            "model": "species_pooled_minus_direct",
            "metric": m,
            "observed": obs_delta,
            "ci95_low": low,
            "ci95_high": high,
            "n_samples": len(sdf),
            "n_cells": int(matched.shape[0]),
            "n_bootstraps": N_BOOTSTRAPS,
            "bootstrap_unit": "sample/imaging-series",
            "bootstrap_stratification": "true species",
            "bootstrap_probability_difference_gt_0": float(np.mean(delta > 0)),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(
        out_dir / f"bootstrap_summary_{architecture}_{cohort_name}.csv",
        index=False,
    )

    # Difference plot for the main accuracy endpoint.
    delta_acc = 100.0 * (
        draws["species_pooled"]["accuracy"] - draws["direct"]["accuracy"]
    )
    obs_delta_pp = 100.0 * (
        observed["species_pooled"]["accuracy"] - observed["direct"]["accuracy"]
    )
    low_pp, high_pp = ci(delta_acc)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hist(delta_acc, bins=60)
    ax.axvline(0, linewidth=1)
    ax.axvline(obs_delta_pp, linewidth=2)
    ax.axvline(low_pp, linestyle="--", linewidth=1)
    ax.axvline(high_pp, linestyle="--", linewidth=1)
    ax.set_xlabel("Species-pooled minus direct routing accuracy (percentage points)")
    ax.set_ylabel("Bootstrap replicates")
    ax.set_title(
        f"{architecture.upper()} paired sample bootstrap: {cohort_name}\n"
        f"Observed {obs_delta_pp:+.2f} pp; 95% CI [{low_pp:+.2f}, {high_pp:+.2f}]"
    )
    fig.tight_layout()
    fig.savefig(
        out_dir / f"bootstrap_accuracy_difference_{architecture}_{cohort_name}.png",
        dpi=220,
    )
    plt.close(fig)

    return summary, draw_df


def make_ci_plot(all_summary: pd.DataFrame, output_path: Path) -> None:
    """Compact accuracy plot for the all-sample primary analysis."""
    q = all_summary[
        (all_summary["cohort"] == "all_holdout_samples")
        & (all_summary["metric"] == "accuracy")
        & (all_summary["model"].isin(["species_pooled", "direct"]))
    ].copy()
    if q.empty:
        return

    labels = []
    y = []
    lo = []
    hi = []
    for arch in ("concat", "se"):
        for model in ("species_pooled", "direct"):
            r = q[(q["architecture"] == arch) & (q["model"] == model)]
            if r.empty:
                continue
            r = r.iloc[0]
            labels.append(f"{arch.upper()}\n{model.replace('_', ' ')}")
            y.append(100 * r["observed"])
            lo.append(100 * (r["observed"] - r["ci95_low"]))
            hi.append(100 * (r["ci95_high"] - r["observed"]))

    x = np.arange(len(y))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(x, y, yerr=np.vstack([lo, hi]), fmt="o", capsize=5)
    ax.set_xticks(x, labels)
    ax.set_ylabel("AST-routing accuracy (%)")
    ax.set_title("Independent holdout performance with 95% sample-bootstrap CIs")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def sanity_check(architecture: str, summary: pd.DataFrame) -> None:
    q = summary[
        (summary["cohort"] == "all_holdout_samples")
        & (summary["metric"] == "accuracy")
    ]
    for model in ("species_pooled", "direct"):
        row = q[q["model"] == model]
        if row.empty:
            continue
        observed = float(row.iloc[0]["observed"])
        expected = EXPECTED_REFERENCE[architecture][model]
        diff_pp = 100 * (observed - expected)
        print(
            f"Sanity check {architecture}/{model}: observed={100*observed:.2f}% "
            f"vs prior={100*expected:.2f}% (difference {diff_pp:+.2f} pp)"
        )
        if abs(diff_pp) > 0.50:
            print(
                "WARNING: this differs by >0.5 percentage points from the prior "
                "reported value. Check that the intended holdout prediction CSV "
                "was selected.",
                flush=True,
            )


def analyse_architecture(
    architecture: str,
    sample_map: Optional[Dict[str, str]],
) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    print("\n" + "=" * 100)
    print(f"ANALYSING {architecture.upper()}")
    print("=" * 100)

    species_csv = find_prediction_csv(RUN_DIRS[architecture]["species"])
    direct_csv = find_prediction_csv(RUN_DIRS[architecture]["direct"])
    print("Species-trained prediction CSV:", species_csv)
    print("Direct 8-class prediction CSV:", direct_csv)

    s = load_species_predictions(species_csv)
    d = load_direct_predictions(direct_csv)

    matched = s.merge(d, on="cell_key", how="inner", validate="one_to_one")
    if len(matched) != len(s) or len(matched) != len(d):
        raise ValueError(
            f"Predictions are not on exactly the same cells for {architecture}: "
            f"species={len(s):,}, direct={len(d):,}, matched={len(matched):,}. "
            "Refusing an unmatched paired comparison."
        )
    if not np.array_equal(
        matched["true_panel_idx"].to_numpy(),
        matched["direct_true_idx"].to_numpy(),
    ):
        bad = matched[
            matched["true_panel_idx"] != matched["direct_true_idx"]
        ].head()
        raise ValueError(
            "Direct-model true routing labels disagree with species-derived true "
            f"routing labels. First disagreements:\n{bad.to_string()}"
        )

    # Attach the exact sample grouping used for clustered resampling.
    if sample_map is not None:
        mapped = matched["cell_key"].map(sample_map)
        if mapped.notna().all():
            matched["sample_id"] = mapped.astype(str)
        else:
            print(
                f"WARNING: explicit manifest sample IDs matched only "
                f"{mapped.notna().sum():,}/{len(mapped):,} cells; falling back "
                "to plate/rep/series parsing from filenames."
            )
            matched["sample_id"] = [
                parse_sample_id_from_path(p, sp)
                for p, sp in zip(matched["path_species"], matched["true_species"])
            ]
    else:
        matched["sample_id"] = [
            parse_sample_id_from_path(p, sp)
            for p, sp in zip(matched["path_species"], matched["true_species"])
        ]

    # Guard against accidental grouping across species.
    cross = matched.groupby("sample_id")["true_species"].nunique()
    if cross.max() > 1:
        raise ValueError("A sample_id contains cells from multiple true species")

    sample_df, pooled_cms, direct_cms = build_sample_table(matched)
    sample_df.to_csv(
        OUTPUT_DIR / f"sample_level_performance_{architecture}.csv",
        index=False,
    )

    print(
        f"Matched cells: {len(matched):,}; samples: {len(sample_df)}; "
        f"species: {sample_df['species'].nunique()}"
    )
    print("Samples per species:")
    print(sample_df.groupby("species").size().to_string())

    # Cell-level paired correctness breakdown.
    pgood = matched["pooled_pred_idx"].to_numpy() == matched["true_panel_idx"].to_numpy()
    dgood = matched["direct_pred_idx"].to_numpy() == matched["true_panel_idx"].to_numpy()
    breakdown = pd.DataFrame([
        {"category": "both_correct", "n_cells": int(np.sum(pgood & dgood))},
        {"category": "species_pooled_only_correct", "n_cells": int(np.sum(pgood & ~dgood))},
        {"category": "direct_only_correct", "n_cells": int(np.sum(~pgood & dgood))},
        {"category": "both_wrong", "n_cells": int(np.sum(~pgood & ~dgood))},
    ])
    breakdown["percentage"] = 100 * breakdown["n_cells"] / len(matched)
    breakdown.to_csv(
        OUTPUT_DIR / f"paired_correctness_breakdown_{architecture}.csv",
        index=False,
    )

    # Cohort 1: all independent holdout samples.
    all_ids = set(sample_df["sample_id"].astype(str))
    all_matched = matched[matched["sample_id"].astype(str).isin(all_ids)].copy()

    # Cohort 2: fixed >=200-cell cohort, matching the aggregation eligibility rule.
    eligible_ids = set(
        sample_df.loc[
            sample_df["n_cells"] >= MIN_CELLS_SENSITIVITY, "sample_id"
        ].astype(str)
    )
    eligible_matched = matched[
        matched["sample_id"].astype(str).isin(eligible_ids)
    ].copy()

    cohort_sizes = pd.DataFrame([
        {
            "architecture": architecture,
            "cohort": "all_holdout_samples",
            "n_samples": len(all_ids),
            "n_cells": len(all_matched),
        },
        {
            "architecture": architecture,
            "cohort": "samples_ge200_cells",
            "n_samples": len(eligible_ids),
            "n_cells": len(eligible_matched),
        },
    ])
    cohort_sizes.to_csv(
        OUTPUT_DIR / f"cohort_sizes_{architecture}.csv", index=False
    )

    print(
        f">=200-cell sensitivity cohort: {len(eligible_ids)} samples, "
        f"{len(eligible_matched):,} cells"
    )

    summaries = []
    for cohort_name, cohort_df in (
        ("all_holdout_samples", all_matched),
        ("samples_ge200_cells", eligible_matched),
    ):
        summary, _ = bootstrap_cohort(
            architecture,
            cohort_name,
            cohort_df,
            sample_df,
            pooled_cms,
            direct_cms,
            OUTPUT_DIR,
        )
        summaries.append(summary)

    combined = pd.concat(summaries, ignore_index=True)
    sanity_check(architecture, combined)

    # Save matched per-cell prediction labels without 13 probability columns;
    # useful for checking exactly which cells differ between models.
    compact = matched[[
        "cell_key", "path_species", "true_species", "sample_id",
        "true_panel_idx", "pooled_pred_idx", "direct_pred_idx",
    ]].copy()
    compact["true_panel"] = [PANELS[i] for i in compact["true_panel_idx"]]
    compact["species_pooled_prediction"] = [PANELS[i] for i in compact["pooled_pred_idx"]]
    compact["direct_prediction"] = [PANELS[i] for i in compact["direct_pred_idx"]]
    compact["species_pooled_correct"] = compact["pooled_pred_idx"] == compact["true_panel_idx"]
    compact["direct_correct"] = compact["direct_pred_idx"] == compact["true_panel_idx"]
    compact.to_csv(
        OUTPUT_DIR / f"paired_cell_predictions_{architecture}.csv",
        index=False,
    )

    return summaries, cohort_sizes


def write_text_report(summary: pd.DataFrame, cohort_sizes: pd.DataFrame) -> None:
    lines = []
    lines.append("SC-MAPS DIRECT 8-CLASS PANEL BOOTSTRAP ANALYSIS")
    lines.append("=" * 72)
    lines.append(
        f"Bootstrap: {N_BOOTSTRAPS:,} paired sample-level replicates, "
        "stratified by true species."
    )
    lines.append("")

    for arch in ("concat", "se"):
        lines.append(f"{arch.upper()}")
        lines.append("-" * 72)
        for cohort in ("all_holdout_samples", "samples_ge200_cells"):
            sub = summary[
                (summary["architecture"] == arch)
                & (summary["cohort"] == cohort)
                & (summary["metric"] == "accuracy")
            ]
            if sub.empty:
                continue
            size = cohort_sizes[
                (cohort_sizes["architecture"] == arch)
                & (cohort_sizes["cohort"] == cohort)
            ].iloc[0]
            lines.append(
                f"{cohort}: n={int(size.n_samples)} samples, "
                f"{int(size.n_cells):,} cells"
            )
            for model in ("species_pooled", "direct", "species_pooled_minus_direct"):
                r = sub[sub["model"] == model].iloc[0]
                if model == "species_pooled_minus_direct":
                    lines.append(
                        "  paired difference (species pooled - direct): "
                        f"{100*r.observed:+.2f} pp "
                        f"(95% CI {100*r.ci95_low:+.2f} to {100*r.ci95_high:+.2f} pp)"
                    )
                    if "bootstrap_probability_difference_gt_0" in r.index and not pd.isna(r["bootstrap_probability_difference_gt_0"]):
                        lines.append(
                            "  bootstrap Pr(difference > 0): "
                            f"{float(r.bootstrap_probability_difference_gt_0):.4f}"
                        )
                else:
                    lines.append(
                        f"  {model}: {100*r.observed:.2f}% "
                        f"(95% CI {100*r.ci95_low:.2f}-{100*r.ci95_high:.2f}%)"
                    )
            lines.append("")

    lines.append("Interpretation guide")
    lines.append("-" * 72)
    lines.append(
        "If the 95% CI for 'species pooled - direct' excludes 0, the paired "
        "sample-bootstrap supports a consistent accuracy difference between the "
        "two strategies on the independent holdout samples."
    )
    lines.append(
        "This bootstrap quantifies uncertainty across holdout sample/imaging-series "
        "units. It does NOT measure training-seed/retraining variability."
    )

    report = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "FINAL_BOOTSTRAP_REPORT.txt").write_text(report)
    print("\n" + report)


def main() -> None:
    print("SC-MAPS direct-panel paired bootstrap analysis")
    print("Output:", OUTPUT_DIR)
    print(f"Bootstrap replicates: {N_BOOTSTRAPS:,}")
    print("Resampling unit: sample/imaging-series; stratified by true species")

    sample_map = load_manifest_sample_map()

    all_summaries: List[pd.DataFrame] = []
    all_cohort_sizes: List[pd.DataFrame] = []

    for architecture in ("concat", "se"):
        summaries, cohort_sizes = analyse_architecture(architecture, sample_map)
        all_summaries.extend(summaries)
        all_cohort_sizes.append(cohort_sizes)

    final_summary = pd.concat(all_summaries, ignore_index=True)
    final_sizes = pd.concat(all_cohort_sizes, ignore_index=True)

    final_summary.to_csv(OUTPUT_DIR / "ALL_BOOTSTRAP_RESULTS.csv", index=False)
    final_sizes.to_csv(OUTPUT_DIR / "ALL_COHORT_SIZES.csv", index=False)

    # A compact manuscript-oriented table containing the main endpoints.
    manuscript = final_summary[
        final_summary["metric"].isin(["accuracy", "macro_f1"])
    ].copy()
    manuscript["observed_percent"] = 100 * manuscript["observed"]
    manuscript["ci95_low_percent"] = 100 * manuscript["ci95_low"]
    manuscript["ci95_high_percent"] = 100 * manuscript["ci95_high"]
    manuscript.to_csv(
        OUTPUT_DIR / "MANUSCRIPT_BOOTSTRAP_TABLE.csv", index=False
    )

    make_ci_plot(
        final_summary,
        OUTPUT_DIR / "all_models_accuracy_with_bootstrap_CI.png",
    )
    write_text_report(final_summary, final_sizes)

    metadata = {
        "n_bootstraps": N_BOOTSTRAPS,
        "random_seed": RANDOM_SEED,
        "bootstrap_unit": "sample/imaging-series",
        "bootstrap_design": "paired, stratified by true species",
        "minimum_cells_sensitivity_cohort": MIN_CELLS_SENSITIVITY,
        "holdout_manifest": str(HOLDOUT_MANIFEST),
        "run_dirs": {
            arch: {k: str(v) for k, v in d.items()}
            for arch, d in RUN_DIRS.items()
        },
        "output_dir": str(OUTPUT_DIR),
    }
    (OUTPUT_DIR / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print("\nDONE.")
    print("Main files to inspect:")
    print("  ", OUTPUT_DIR / "FINAL_BOOTSTRAP_REPORT.txt")
    print("  ", OUTPUT_DIR / "MANUSCRIPT_BOOTSTRAP_TABLE.csv")
    print("  ", OUTPUT_DIR / "ALL_BOOTSTRAP_RESULTS.csv")
    print("  ", OUTPUT_DIR / "all_models_accuracy_with_bootstrap_CI.png")


if __name__ == "__main__":
    main()
