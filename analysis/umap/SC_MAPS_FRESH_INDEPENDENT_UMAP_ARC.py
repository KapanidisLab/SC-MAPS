#!/usr/bin/env python3
"""
SC-MAPS | fresh, independent UMAP fits from DenseNet121 penultimate features.

ONE STANDALONE FILE. Does not retrain the model. Does not use or transform the
old known-species UMAP, old unseen UMAP coordinates, or previous PCA model.

By default makes 4 figures from THREE SEPARATELY FITTED maps:
  (a) Jointly refitted 13-known + 5-unseen map: known species and circled regions
  (b) SAME NEW JOINT MAP, with unseen organisms overlaid (valid comparison)
  (c) A fully independent 13-known-only map (new UMAP and new PCA)
  (d) A fully independent five-unseen-only map (new UMAP and new PCA)

With --per-unseen: five additional independently fitted maps, each containing
all known species plus ONE unseen species, refitted from scratch each time.

IMPORTANT: Different UMAP fits do not share meaningful coordinates. Only
organisms in the SAME fit can be compared for visual neighbourhood patterns.
These maps are descriptive, not species classification or a biological distance.

Run:
    python -u SC_MAPS_FRESH_INDEPENDENT_UMAP_ARC.py --inspect-only
    python -u SC_MAPS_FRESH_INDEPENDENT_UMAP_ARC.py
    python -u SC_MAPS_FRESH_INDEPENDENT_UMAP_ARC.py --per-unseen

Optional explicit inputs (use if automatic detection does not find your NPZ):
    python -u SC_MAPS_FRESH_INDEPENDENT_UMAP_ARC.py \\
       --known-npz /path/known_balanced_penultimate_features.npz \\
       --unseen-npz /path/unseen_balanced_penultimate_features.npz

Uses SAVED PENULTIMATE VECTORS (not UMAP coordinates, softmax probability CSV,
or raw microscopy crops). You may need to install umap-learn in your ARC env:
    python -m pip install --user umap-learn
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import warnings

# Avoid oversubscribing the ARC CPU node in OpenBLAS/Numba.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_name, "8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA

ARC_ROOT = Path(
    "/data/phys-deep-learning-bacteria/phys3115/"
    "Bacteria-data-15 species/Trillian"
)
KNOWN = ("ACAA", "ECOL", "ENTC", "KLAE", "KLPN", "PSAR", "PSMA",
         "SAENA", "SAUR", "SEPI", "SERM", "SLUG", "STCP")
UNSEEN = ("BSB", "EF", "EFM", "HFLU", "LMON")
ALIASES = {
    "HFLU": "HFLU", "HELU": "HFLU", "HI": "HFLU",
    "SAGALACTIAE": "BSB", "STREPTOCOCCUSAGALACTIAE": "BSB",
    "EFAECALIS": "EF", "ENTEROCOCCUSFAECALIS": "EF",
    "EFAECIUM": "EFM", "ENTEROCOCCUSFAECIUM": "EFM",
    "LISTERIAMONOCYTOGENES": "LMON", "HAEMOPHILUSINFLUENZAE": "HFLU",
    "HEMOPHILUSINFLUENZAE": "HFLU",
}


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--known-npz", type=Path, default=None, help="Explicit known feature NPZ")
    p.add_argument("--unseen-npz", type=Path, default=None, help="Explicit unseen feature NPZ")
    p.add_argument("--known-labels-csv", type=Path, default=None, help="Optional labels CSV aligned row-for-row to known NPZ")
    p.add_argument("--unseen-labels-csv", type=Path, default=None, help="Optional labels CSV aligned row-for-row to unseen NPZ")
    p.add_argument("--root", type=Path, default=ARC_ROOT)
    p.add_argument("--output", type=Path, default=ARC_ROOT / "Chris_review_analysis_ARC" / "fresh_independent_umap")
    p.add_argument("--max-per-species", type=int, default=2000, help="Maximum random feature rows per species per dataset; 0=all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pca-components", type=int, default=50)
    p.add_argument("--n-neighbors", type=int, default=30)
    p.add_argument("--min-dist", type=float, default=0.15)
    p.add_argument("--metric", type=str, default="euclidean", help="UMAP metric on per-fit PCA features")
    p.add_argument("--per-unseen", action="store_true", help="Also fit five independent known-plus-one-unseen maps")
    p.add_argument("--inspect-only", action="store_true", help="Locate/list NPZ keys and shapes; do not fit")
    return p.parse_args()


def discovered_npzs(root: Path):
    # Search only analysis-output directories, never millions of image crops.
    candidate_dirs = [Path.cwd(), Path(__file__).resolve().parent, root,
                      root / "Chris_review_analysis_ARC", root / "Chris_analysis_ARC",
                      root / "Chris_review_analysis_ARC_fixed", root / "Chris_Review_Analysis_ARC",
                      root / "Python" / "Chris_analysis",
                      Path("/scratch/phys3115/bacteria_data") / "Chris_review_analysis_ARC"]
    found = []
    seen = set()
    for folder in candidate_dirs:
        if not folder.exists() or not folder.is_dir():
            continue
        try:
            paths = list(folder.glob("*.npz")) + list(folder.glob("*/*features*.npz"))
        except (PermissionError, OSError):
            continue
        for path in paths:
            try:
                canonical = path.resolve()
            except OSError:
                continue
            if canonical not in seen:
                found.append(path)
                seen.add(canonical)
    return found


def auto_path(explicit: Path | None, kind: str, found: list[Path]):
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"--{kind}-npz not found: {explicit}")
        return explicit
    candidates = []
    for path in found:
        low = path.name.lower()
        if "feature" not in low or "npz" not in low:
            continue
        if kind == "known":
            score = (10 if "known" in low else 0) + (5 if "holdout" in low else 0) - (20 if "unseen" in low or "unknown" in low else 0)
        else:
            score = (10 if "unseen" in low else 0) + (5 if "unknown" in low else 0) - (20 if "known" in low and "unseen" not in low else 0)
        if score > 0:
            candidates.append((score, path))
    candidates.sort(key=lambda x: (-x[0], str(x[1])))
    if not candidates:
        raise FileNotFoundError(
            f"Could not locate {kind} penultimate feature NPZ. "
            "Supply --known-npz /path/file.npz and --unseen-npz /path/file.npz. "
            "Old UMAP-coordinate CSV files are NOT valid inputs for a fresh fit.\n"
            "On ARC, locate files with: find '/data/phys-deep-learning-bacteria/phys3115/"
            "Bacteria-data-15 species/Trillian' -name '*features*.npz'"
        )
    if len(candidates) > 1:
        logging.warning("Multiple %s NPZ candidates; selecting %s. Other possibilities: %s",
                        kind, candidates[0][1], [str(p) for _, p in candidates[1:4]])
    return candidates[0][1]


def normalise_species(labels: np.ndarray):
    vals = []
    for v in labels:
        if isinstance(v, bytes):
            v = v.decode("utf-8", errors="replace")
        s = str(v).strip().upper()
        key = "".join(ch for ch in s if ch.isalnum())
        vals.append(ALIASES.get(key, key))
    return np.asarray(vals, dtype="U40")


def print_npz_contents(path: Path):
    print(f"\nFEATURE FILE: {path}", flush=True)
    with np.load(path, allow_pickle=True) as d:
        for k in d.files:
            a = np.asarray(d[k])
            print(f"  {k:<30} {str(a.shape):<20} {a.dtype}", flush=True)


def pick_feature_key(d):
    priorities = ["features", "penultimate_features", "feature_vectors", "embeddings", "X", "x", "feature_matrix", "latent_features"]
    options = []
    for key in d.files:
        a = np.asarray(d[key])
        if a.ndim == 2 and a.shape[0] >= 10 and a.shape[1] >= 32 and np.issubdtype(a.dtype, np.number):
            rank = priorities.index(key) if key in priorities else 100
            options.append((rank, key, a))
    if not options:
        raise ValueError(f"No NxD (D>=32) numeric features in NPZ; keys: {d.files}")
    options.sort(key=lambda z: (z[0], -z[2].shape[1]))
    if len(options) > 1 and options[0][0] == 100:
        raise ValueError(f"Ambiguous feature arrays {[(z[1],z[2].shape) for z in options]}; rename desired array to 'features'.")
    return options[0][1], options[0][2]


def read_labels_csv(path: Path, n: int):
    df = pd.read_csv(path)
    for col in ("species", "true_species", "species_code", "label", "labels", "unseen_species"):
        if col in df.columns:
            if len(df) != n:
                raise ValueError(f"Labels file {path} contains {len(df)} rows, but NPZ features have {n}. "
                                 "You must guarantee identical row order and count.")
            logging.warning("Using row-ordered %s labels from %s. YOU must verify row alignment.", col, path)
            return df[col].to_numpy()
    raise ValueError(f"No species/label column in {path}; got {df.columns.tolist()}")


def read_npz(path: Path, kind: str, override_labels: Path | None):
    with np.load(path, allow_pickle=True) as d:
        feature_key, arr = pick_feature_key(d)
        X = np.asarray(arr, dtype=np.float32)
        label_keys = ("species", "true_species", "species_code", "species_name", "species_labels", "class_labels", "labels", "true_labels", "labels_str", "y_true", "y", "label")
        labels = None
        label_key = None
        for key in label_keys:
            if key not in d.files:
                continue
            a = np.asarray(d[key])
            if a.ndim == 1 and len(a) == len(X):
                labels = a
                label_key = key
                break
        if labels is None and override_labels:
            labels = read_labels_csv(override_labels, len(X))
            label_key = f"CSV:{override_labels.name}"
        if labels is None:
            raise ValueError(
                f"No per-cell labels in {path}; keys={d.files}. Provide explicit --{kind}-labels-csv "
                "with a species column IN THE EXACT FEATURE ROW ORDER. "
                "The old UMAP coordinate CSV may have compatible labels, but alignment MUST be checked."
            )
        if np.issubdtype(np.asarray(labels).dtype, np.integer):
            mapping = None
            for key in ("classes", "class_names", "species_names", "idx_to_class"):
                if key in d.files:
                    m = np.asarray(d[key])
                    if m.ndim == 1 and len(m) > int(np.max(labels)):
                        mapping = m
                        break
            if mapping is None:
                raise ValueError(f"Numeric labels in {path} without class names. Need text species labels or labelled CSV.")
            labels = mapping[labels.astype(int)]
        y = normalise_species(labels)
    if len(X) != len(y):
        raise ValueError(f"Feature/label row mismatch: {len(X)} vs {len(y)}")
    if not np.isfinite(X).all():
        raise ValueError(f"Nonfinite feature values in {path}; remove/repair before fitting")
    expected = set(KNOWN if kind == "known" else UNSEEN)
    actual = set(np.unique(y))
    if not actual.issubset(expected):
        raise ValueError(f"Unexpected {kind} labels {sorted(actual - expected)} from key {label_key}; expected {sorted(expected)}")
    if len(actual) != len(expected):
        logging.warning("Missing expected %s species: %s", kind, sorted(expected - actual))
    logging.info("Loaded %s: %s rows, %s features, labels=%s, featurekey=%s from %s",
                 kind, X.shape[0], X.shape[1], dict(zip(*np.unique(y, return_counts=True))),feature_key,path)
    return X, y


def balanced(X, y, max_each, seed):
    if max_each <= 0:
        return X, y, np.arange(len(y), dtype=np.int64)
    rng = np.random.default_rng(seed)
    inds = []
    for s in sorted(np.unique(y)):
        ix = np.flatnonzero(y == s)
        if len(ix) > max_each:
            ix = rng.choice(ix, size=max_each, replace=False)
        inds.extend(ix.tolist())
    ix = np.asarray(inds, dtype=np.int64)
    return X[ix], y[ix], ix


def fit_new_umap(X, seed, pca_components, n_neighbors, min_dist, metric, ident):
    try:
        import umap
    except ImportError as e:
        raise RuntimeError("Missing umap-learn. In your ARC Python environment run: python -m pip install --user umap-learn") from e
    n, d = X.shape
    k = min(pca_components, n - 1, d)
    if k < 2:
        raise ValueError("Not enough independent features to run PCA and UMAP")
    logging.info("[%s] Fitting a FRESH PCA(%d) on %d cells x %d features", ident, k, n, d)
    reducer = PCA(n_components=k, svd_solver="randomized", random_state=seed)
    coords_pca = reducer.fit_transform(X).astype(np.float32)
    logging.info("[%s] Explained feature variance in PCA(%d): %.2f%%", ident, k,
                 100 * float(reducer.explained_variance_ratio_.sum()))
    logging.info("[%s] Fitting a NEW UMAP model (not calling transform on another fit)", ident)
    model = umap.UMAP(n_components=2, random_state=seed, n_neighbors=min(n_neighbors,n-1),
                      min_dist=min_dist, metric=metric, low_memory=True)
    xy = model.fit_transform(coords_pca)
    if not np.isfinite(xy).all():
        raise RuntimeError("UMAP returned nonfinite coordinates")
    return xy.astype(np.float32), float(reducer.explained_variance_ratio_.sum())


def ellipse(ax, pts, label, edge, text_color=None):
    # Outlines are descriptive approximate cluster concentrations, not boundaries.
    if len(pts) < 10:
        return
    centre = np.median(pts, axis=0)
    dist = np.sum((pts - centre)**2,axis=1)
    subset = pts[dist <= np.quantile(dist, 0.82)]
    if len(subset) < 5:
        return
    cov = np.cov(subset.T) + 1e-5 * np.eye(2)
    eig, vec = np.linalg.eigh(cov)
    order = np.argsort(eig)[::-1]
    eig, vec = eig[order], vec[:,order]
    angle = float(np.degrees(np.arctan2(vec[1,0],vec[0,0])))
    width,height = np.maximum(2.9*np.sqrt(eig),0.2)
    ax.add_patch(Ellipse(centre, width,height,angle=angle,fill=False,
                         linewidth=1.35, edgecolor=edge, alpha=0.9))
    ax.annotate(label, centre, xytext=(3,4), textcoords="offset points",
                fontsize=8.5, weight="bold", color=text_color or edge,
                bbox={"facecolor":"white","edgecolor":"none","alpha":0.68,"pad":0.8})


def plot_coords(xy, y, file: Path, title: str, *, plot_mode: str, known_order=KNOWN, unseen_order=UNSEEN):
    known_colors = dict(zip(KNOWN, plt.get_cmap("tab20")(np.linspace(0.02,0.95,len(KNOWN)))))
    unseen_colors = dict(zip(UNSEEN, ["#e94e77", "#d47610", "#b83280", "#14a6a0", "#642cbb"]))
    fig,ax=plt.subplots(figsize=(13,8.5),dpi=140)
    is_known = np.isin(y, KNOWN)
    if plot_mode in ("joint_known", "known_only"):
        for s in known_order:
            mask = y == s
            if not mask.any():
                continue
            ax.scatter(xy[mask,0],xy[mask,1],s=3.7,alpha=0.19,
                       rasterized=True,color=known_colors[s],label=s,lw=0)
            ellipse(ax,xy[mask],s,known_colors[s])
    elif plot_mode in ("joint_overlay", "pair_overlay"):
        ax.scatter(xy[is_known,0],xy[is_known,1],s=2.0,alpha=0.10,
                   rasterized=True,color="#8c99a3",lw=0,label="Known species")
        for s in known_order:
            mask = y == s
            if mask.any():
                ellipse(ax,xy[mask],s,"#697782",text_color="#475569")
        for s in unseen_order:
            mask=y==s
            if mask.any():
                ax.scatter(xy[mask,0],xy[mask,1],s=5.5,alpha=0.51,
                           rasterized=True,color=unseen_colors[s],label=f"Unseen: {s}",lw=0)
    elif plot_mode=="unseen_only":
        for s in unseen_order:
            mask = y == s
            if mask.any():
                ax.scatter(xy[mask,0],xy[mask,1],s=4,alpha=0.23,
                           rasterized=True,color=unseen_colors[s],label=s,lw=0)
                ellipse(ax,xy[mask],s,unseen_colors[s])
    else:
        raise ValueError(plot_mode)
    ax.set_title(title,fontsize=15,weight="bold",pad=15)
    ax.set_xlabel("New UMAP-1 (arbitrary units)",fontsize=11)
    ax.set_ylabel("New UMAP-2 (arbitrary units)",fontsize=11)
    ax.spines[["right","top"]].set_visible(False)
    ax.legend(loc="upper left",bbox_to_anchor=(1.01,1.00),frameon=False,markerscale=3.1,fontsize=9)
    ax.text(0,-0.10,"Outlines show approximate cluster concentrations, not decision boundaries. "
            "Compare positions only within the SAME fit.",transform=ax.transAxes,fontsize=9,
            color="#586775",va="top")
    fig.savefig(file,dpi=220,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    logging.info("Saved %s",file)


def export_coords(xy,y,orig,fitid,origin,out):
    df=pd.DataFrame({"fit_id":fitid,"dataset":origin,"species":y,
                     "source_npz_row_index":orig,"UMAP1":xy[:,0],"UMAP2":xy[:,1]})
    df.to_csv(out,index=False)
    logging.info("Saved coordinate table %s", out)


def main():
    args=get_args()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",
                        stream=sys.stdout)
    possible=discovered_npzs(args.root)
    if args.inspect_only:
        logging.info("Discovered %d NPZ files in analysis directories",len(possible))
        for p in possible:
            try: print_npz_contents(p)
            except Exception as ex: logging.warning("Cannot inspect %s: %s",p,ex)
        if args.known_npz and args.known_npz not in possible: print_npz_contents(args.known_npz)
        if args.unseen_npz and args.unseen_npz not in possible: print_npz_contents(args.unseen_npz)
        return
    known_file=auto_path(args.known_npz,"known",possible)
    unseen_file=auto_path(args.unseen_npz,"unseen",possible)
    if known_file.resolve()==unseen_file.resolve():
        raise ValueError("Known and unseen must come from two distinct per-cell penultimate-feature NPZ files")
    Xk,yk=read_npz(known_file,"known",args.known_labels_csv)
    Xu,yu=read_npz(unseen_file,"unseen",args.unseen_labels_csv)
    if Xk.shape[1]!=Xu.shape[1]:
        raise ValueError(f"Known/unseen vectors have different dimensions ({Xk.shape[1]} vs {Xu.shape[1]}): "
                         "may be from incompatible networks or different stages. Do not combine.")
    Xk,yk,idxk=balanced(Xk,yk,args.max_per_species,args.seed)
    Xu,yu,idxu=balanced(Xu,yu,args.max_per_species,args.seed+1)
    logging.info("Using %d known and %d unseen cells after balancing",len(Xk),len(Xu))
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    config={"known_file":str(known_file),"unseen_file":str(unseen_file),
            "known_n":len(Xk),"unseen_n":len(Xu),"feature_dims":int(Xk.shape[1]),
            "seed":args.seed,"max_per_species":args.max_per_species,
            "pca_components":args.pca_components,"n_neighbors":args.n_neighbors,
            "min_dist":args.min_dist,"metric":args.metric,"per_unseen":args.per_unseen,
            "known_classes":dict(zip(*np.unique(yk,return_counts=True))),
            "unseen_classes":dict(zip(*np.unique(yu,return_counts=True))),
            "critical_note":"No reusing old UMAP. Different fit coordinates cannot be compared."}
    config["known_classes"]={k:int(v) for k,v in config["known_classes"].items()}
    config["unseen_classes"]={k:int(v) for k,v in config["unseen_classes"].items()}
    with (out/"run_manifest.json").open("w") as f: json.dump(config,f,indent=2)
    Xboth=np.concatenate((Xk,Xu),axis=0)
    yboth=np.concatenate((yk,yu),axis=0)
    ixall=np.concatenate((idxk,idxu))
    srcall=np.array(["known"]*len(Xk)+["unseen"]*len(Xu))
    pca_summary=[]
    xy,ev=fit_new_umap(Xboth,args.seed,args.pca_components,args.n_neighbors,args.min_dist,args.metric,"JOINT")
    pca_summary.append({"fit":"joint", "pca_variance_fraction":ev, "cells":len(xy)})
    export_coords(xy,yboth,ixall,"joint",srcall,out/"joint_fresh_umap_coordinates.csv")
    plot_coords(xy,yboth,out/"01_new_joint_KNOWN_circled.png",
                "NEW joint UMAP: known species (13) — fully refitted",plot_mode="joint_known")
    plot_coords(xy,yboth,out/"02_new_joint_UNSEEN_overlay.png",
                "SAME NEW joint UMAP: five unseen organisms overlaid",plot_mode="joint_overlay")
    # A true independent embedding, using a new PCA and UMAP for each data cohort.
    for fit_id,X,y,ix,src,title,filename,mode in (
        ("known_only",Xk,yk,idxk,"known", "INDEPENDENT map: 13 known species only", "03_independent_KNOWN_only.png", "known_only"),
        ("unseen_only",Xu,yu,idxu,"unseen", "INDEPENDENT map: five unseen organisms only", "04_independent_UNSEEN_only.png", "unseen_only")):
        xy_s,ev_s=fit_new_umap(X,args.seed,args.pca_components,args.n_neighbors,args.min_dist,args.metric,fit_id)
        pca_summary.append({"fit":fit_id,"pca_variance_fraction":ev_s,"cells":len(xy_s)})
        export_coords(xy_s,y,ix,fit_id,src,out/f"{fit_id}_fresh_umap_coordinates.csv")
        plot_coords(xy_s,y,out/filename,title,plot_mode=mode)
    if args.per_unseen:
        for s in UNSEEN:
            ix=np.flatnonzero(yu==s)
            if not len(ix): continue
            Xi=np.concatenate((Xk,Xu[ix]),axis=0)
            yi=np.concatenate((yk,yu[ix]),axis=0)
            rowids=np.concatenate((idxk,idxu[ix]))
            origins=np.array(["known"]*len(Xk)+["unseen"]*len(ix))
            xy_i,ev_i=fit_new_umap(Xi,args.seed,args.pca_components,args.n_neighbors,args.min_dist,args.metric,f"pair_{s}")
            pca_summary.append({"fit":f"pair_{s}","pca_variance_fraction":ev_i,"cells":len(Xi)})
            export_coords(xy_i,yi,rowids,f"pair_{s}",origins,out/f"pair_{s}_fresh_umap_coordinates.csv")
            plot_coords(xy_i,yi,out/f"05_independent_known_plus_{s}.png",
                        f"FRESH known + {s}: independently refitted UMAP", plot_mode="pair_overlay",
                        unseen_order=[s])
    pd.DataFrame(pca_summary).to_csv(out/"pca_variance_by_fresh_fit.csv",index=False)
    logging.info("COMPLETE: all figures and coordinates in %s",out)
    logging.info("CRITICAL: only compare known/unseen points in the SAME joint or pairwise fit; standalone independent maps have arbitrary axes")

if __name__=="__main__":
    try: main()
    except Exception as ex:
        logging.error("SC-MAPS fresh UMAP failed: %s",ex,exc_info=True)
        sys.exit(1)
