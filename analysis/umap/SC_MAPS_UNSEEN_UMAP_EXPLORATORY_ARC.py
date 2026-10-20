#!/usr/bin/env python3
"""SC-MAPS exploratory unseen-organism UMAP stability and PCA-space analysis.

ONE standalone CPU-only ARC script; no plate or strain analysis, no model training.
Uses previously saved DenseNet121 penultimate feature vectors, never UMAP coordinates.

Outputs: (1) unseen-only UMAP by species; (2) multi-seed/neighbour stability
plots and metrics; (3) PCA-space nearest-neighbour species composition,
enrichment and CSV; (4) sampling counts and coordinates.

ARC:
  python SC_MAPS_UNSEEN_UMAP_EXPLORATORY_ARC.py --inspect-only
  python -u SC_MAPS_UNSEEN_UMAP_EXPLORATORY_ARC.py
  # or submit CPU job using sbatch --wrap="python -u ..."
Dependencies: numpy pandas scikit-learn matplotlib scipy umap-learn.
"""
from __future__ import annotations
import argparse, json, logging, os, sys
from pathlib import Path
for _env in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS"):
    os.environ.setdefault(_env,"8")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
ARC_ROOT=Path("/data/phys-deep-learning-bacteria/phys3115/Bacteria-data-15 species/Trillian")
DEFAULT_OUT=ARC_ROOT/"Chris_review_analysis_ARC"/"unseen_UMAP_exploratory"
UNSEEN=("BSB","EF","EFM","HFLU","LMON")
COLORS={"BSB":"#e95880","EF":"#ed8d2e","EFM":"#c22a93","HFLU":"#1da7ad","LMON":"#7042c9"}
ALIASES={"HELU":"HFLU","HI":"HFLU","STREPTOCOCCUSAGALACTIAE":"BSB",
         "ENTEROCOCCUSFAECALIS":"EF","ENTEROCOCCUSFAECIUM":"EFM",
         "LISTERIAMONOCYTOGENES":"LMON","HAEMOPHILUSINFLUENZAE":"HFLU"}

def args_parser():
 p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 p.add_argument("--root",type=Path,default=ARC_ROOT)
 p.add_argument("--features-npz",type=Path,default=None)
 p.add_argument("--output",type=Path,default=DEFAULT_OUT)
 p.add_argument("--inspect-only",action="store_true")
 p.add_argument("--max-per-species",type=int,default=3000,help="0 uses all saved features")
 p.add_argument("--pca-components",type=int,default=50)
 p.add_argument("--seed",type=int,default=42)
 p.add_argument("--seeds",default="11,42,123")
 p.add_argument("--neighbors",default="15,30,50")
 p.add_argument("--min-dist",type=float,default=0.15)
 p.add_argument("--knn-k",type=int,default=25)
 p.add_argument("--stability-subset",type=int,default=1000)
 p.add_argument("--only-baseline",action="store_true")
 return p.parse_args()

def canon_species(a):
    out=[]
    for v in a:
        s=str(v.decode() if isinstance(v,bytes) else v).strip().upper()
        key="".join(c for c in s if c.isalnum())
        out.append(ALIASES.get(key,key))
    return np.array(out,dtype="U32")


def find_feature_file(args):
    if args.features_npz:
        if not args.features_npz.is_file():
            raise FileNotFoundError(args.features_npz)
        return args.features_npz
    roots=[Path.cwd(),Path(__file__).resolve().parent,args.root,
           args.root/"Chris_review_analysis_ARC",args.root/"Chris_analysis_ARC",
           args.root/"Python"/"Chris_analysis"]
    found=[]
    for root in roots:
        if root.is_dir():
            found += list(root.glob("unseen*features*.npz")) + list(root.glob("*/*unseen*features*.npz"))
    found=sorted(set(x.resolve() for x in found))
    if not found:
        raise FileNotFoundError("Could not find unseen penultimate feature NPZ. Provide --features-npz /path/...npz")
    if len(found)>1:
        logging.warning("Multiple feature files found: %s; choosing %s",found,found[0])
    return found[0]


def feature_key(z):
    priority=("features","penultimate_features","feature_vectors","embeddings","X","feature_matrix")
    for k in priority:
        if k in z.files:
            a=np.asarray(z[k]);
            if a.ndim==2 and a.shape[1]>=16 and np.issubdtype(a.dtype,np.number): return k
    poss=[k for k in z.files if np.asarray(z[k]).ndim==2 and
          np.issubdtype(np.asarray(z[k]).dtype,np.number) and np.asarray(z[k]).shape[1]>=16]
    if len(poss)==1: return poss[0]
    raise ValueError(f"Ambiguous feature matrix; NPZ keys={z.files}; expected named 'features'")


def vector(z,keys,n):
    for k in keys:
        if k in z.files:
            a=np.asarray(z[k])
            if a.ndim==1 and len(a)==n:
                return k,a
    return None,None


def read_features(path):
    with np.load(path,allow_pickle=True) as z:
        key=feature_key(z)
        X=np.asarray(z[key],dtype=np.float32)
        labkey,y=vector(z,("species","true_species","species_code","species_labels","labels_str","labels","y_true"),len(X))
        if y is None: raise ValueError("Feature NPZ lacks per-feature-row species labels. Cannot verify metadata joins.")
        if np.issubdtype(y.dtype,np.integer):
            _,classes=vector(z,("classes","class_names","species_names"),len(y))
            # Classes are usually only five elements rather than N, so handle separately.
            if classes is None:
                for ckey in ("classes","class_names","species_names"):
                    if ckey in z.files and len(z[ckey])>int(np.max(y)):
                        classes=np.asarray(z[ckey]);break
            if classes is None: raise ValueError("Numeric labels require a species_names/classes mapping in NPZ")
            y=classes[y.astype(int)]
        y=canon_species(y)
        metadata={}
        for name,keys in {
            "plate":("plate","plates","plate_id","plate_ids","strain_plate"),
            "sample_id":("sample_id","sample_ids","samples","source_sample_id","field_id"),
            "cell_id":("cell_id","cell_ids","cell_keys","object_id"),
            "crop_path":("crop_path","crop_paths","paths","file_paths"),
            "source_row":("source_row","source_rows","source_indices","source_row_indices","original_indices","row_indices","prediction_row_indices")
        }.items():
            k,v=vector(z,keys,len(X))
            if v is not None: metadata[name]=v
        keys=z.files.copy()
    if not np.isfinite(X).all(): raise ValueError("NPZ features contain NaN/Inf")
    if set(np.unique(y))-set(UNSEEN): raise ValueError(f"Unknown labels in NPZ: {set(np.unique(y))-set(UNSEEN)}")
    return X,y,metadata,{"keys":keys,"feature_key":key,"species_key":labkey}


def fit_pca(X,k,seed):
    k=min(k,X.shape[1],len(X)-1)
    if k<2: raise ValueError("Need at least 3 cells and 2 feature dimensions")
    pc=PCA(n_components=k,svd_solver="randomized",random_state=seed)
    return pc,pc.fit_transform(X).astype(np.float32)


def umap_fit(X,neighbors,seed,min_dist):
    try: import umap
    except ImportError as e:
        raise RuntimeError("Missing umap-learn; run 'python -m pip install umap-learn' in the ARC environment") from e
    model=umap.UMAP(n_neighbors=min(neighbors,len(X)-1),min_dist=min_dist,
                    metric="euclidean",n_components=2,random_state=seed,
                    low_memory=True,transform_seed=seed)
    xy=np.asarray(model.fit_transform(X),dtype=np.float32)
    if not np.isfinite(xy).all(): raise ValueError("Non-finite UMAP result")
    return xy


def scatter(ax,xy,y,colors=COLORS,size=3.3,alpha=.27):
    for s in UNSEEN:
        mask=y==s
        if mask.any(): ax.scatter(xy[mask,0],xy[mask,1],c=colors[s],s=size,alpha=alpha,
                                  linewidths=0,rasterized=True,label=s)


def axes_style(ax,title):
    ax.set_title(title,fontsize=15,fontweight="bold",pad=10)
    ax.set_xlabel("UMAP-1 (arbitrary units)")
    ax.set_ylabel("UMAP-2 (arbitrary units)")
    ax.spines[["top","right"]].set_visible(False)


def save_plot(path,fig):
    fig.savefig(path,dpi=210,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    logging.info("Saved %s",path)


def choose_balanced(y,max_per_species,seed):
 rng=np.random.default_rng(seed); chosen=[]
 for s in UNSEEN:
  candidates=np.flatnonzero(y==s)
  if not len(candidates): raise ValueError(f"Missing unseen species {s}")
  quota=len(candidates) if max_per_species<=0 else min(max_per_species,len(candidates))
  chosen.extend(rng.choice(candidates,size=quota,replace=False).tolist())
 return np.asarray(sorted(chosen),dtype=np.int64)

def main_figures(y,xy,out):
 fig,ax=plt.subplots(figsize=(11,7.5))
 scatter(ax,xy,y,size=4,alpha=.27)
 axes_style(ax,"Unseen organisms: independently fitted UMAP coloured by species")
 ax.legend(title="Unseen organism",markerscale=3,frameon=False,bbox_to_anchor=(1.01,1),loc="upper left")
 save_plot(out/"01_unseen_by_species.png",fig)

def highdim_knn(Xp,y,k,out):
 n=len(y); k=min(k,n-2)
 nbr=NearestNeighbors(n_neighbors=k+1,n_jobs=-1).fit(Xp)
 # Neighbour rows may contain exact distance ties: remove SELF by index explicitly.
 ids=nbr.kneighbors(Xp,return_distance=False)
 ranks=np.empty((n,k),dtype=np.int32)
 for i,row in enumerate(ids):
  others=row[row!=i][:k]
  if len(others)!=k: raise RuntimeError("Insufficient nonself neighbours")
  ranks[i]=others
 labels=y[ranks]; same=(labels==y[:,None]); outcomes=[]
 counts=pd.DataFrame(0.0,index=UNSEEN,columns=UNSEEN)
 for s in UNSEEN:
  mask=y==s; vals=pd.Series(labels[mask].ravel()).value_counts()
  counts.loc[s]=[vals.get(t,0) for t in UNSEEN]
  baseline=(int(mask.sum())-1)/(n-1)
  observed=float(same[mask].mean())
  outcomes.append(dict(species=s,n_cells=int(mask.sum()),same_species_knn_fraction=observed,
      random_label_baseline=baseline,same_species_enrichment=observed/baseline if baseline else np.nan))
 df=pd.DataFrame(outcomes);df.to_csv(out/"pca_space_species_knn_enrichment.csv",index=False)
 norm=counts.div(counts.sum(axis=1).replace(0,np.nan),axis=0).fillna(0)
 norm.to_csv(out/"pca_space_knn_label_matrix.csv")
 fig,ax=plt.subplots(figsize=(7.5,6.1))
 m=ax.imshow(norm.values,cmap="Blues",vmin=0,vmax=1,aspect="equal")
 ax.set_xticks(range(len(UNSEEN)),UNSEEN);ax.set_yticks(range(len(UNSEEN)),UNSEEN)
 ax.set_xlabel("PCA-space neighbour species");ax.set_ylabel("Query species")
 ax.set_title(f"PCA-space neighbour composition (k={k})")
 for i in range(len(UNSEEN)):
  for j in range(len(UNSEEN)):
   v=norm.iloc[i,j];ax.text(j,i,f"{v*100:.1f}",ha="center",va="center",fontsize=9,
                              color="white" if v>.57 else "#19314e")
 fig.colorbar(m,ax=ax,label="Fraction of nearest neighbours")
 save_plot(out/"03_pca_space_knn_heatmap.png",fig)
 fig,ax=plt.subplots(figsize=(9,5));x=np.arange(len(df));bw=.35
 ax.bar(x-bw/2,df.same_species_knn_fraction,width=bw,label="Observed",color="#2a7da8")
 ax.bar(x+bw/2,df.random_label_baseline,width=bw,label="Random-label baseline",color="#bac9d6")
 ax.set_xticks(x,df.species);ax.set_ylim(0,1.05);ax.set_ylabel("Fraction")
 ax.set_title("Species association in PCA-reduced DenseNet features")
 ax.legend(frameon=False);save_plot(out/"04_pca_space_knn_enrichment.png",fig)
 logging.info("PCA-space neighbours:\n%s",df.to_string(index=False))

def small_neighbor_sets(xy,idx,k=15):
    # point indices are the SAME across fits; local-neighbour overlap is meaningful.
    sub=xy[idx]
    ids=NearestNeighbors(n_neighbors=min(k+1,len(idx)),n_jobs=-1).fit(sub).kneighbors(sub,return_distance=False)
    return [set(row[row!=i][:k].tolist()) for i,row in enumerate(ids)]


def compare_knn_sets(ref,candidate):
    return float(np.mean([len(a&b)/max(1,len(a)) for a,b in zip(ref,candidate)]))


def stability_grid(Xp,y,params_seeds,params_neighbors,min_dist,idx,out):
    xy_ref=None
    refset=None
    result=[]
    nrow,ncol=len(params_seeds),len(params_neighbors)
    fig,axes=plt.subplots(nrow,ncol,figsize=(4.8*ncol,4.1*nrow),squeeze=False)
    for i,seed in enumerate(params_seeds):
        for j,nb in enumerate(params_neighbors):
            tag=f"seed_{seed}_nn_{nb}"
            logging.info("UMAP stability fit %s (%d/%d)",tag,i*ncol+j+1,nrow*ncol)
            xy=umap_fit(Xp,nb,seed,min_dist)
            if xy_ref is None:
                xy_ref=xy
                refset=small_neighbor_sets(xy_ref,idx)
            cellset=small_neighbor_sets(xy,idx)
            score=compare_knn_sets(refset,cellset)
            tw=float(trustworthiness(Xp[idx],xy[idx],n_neighbors=min(10,len(idx)//3)))
            result.append({"seed":seed,"n_neighbors":nb,"min_dist":min_dist,"n_cells":len(Xp),
                           "trustworthiness_subsample":tw,
                           "local_2d_neighbour_overlap_with_first_fit":score,
                           "comparison_note":"First-fit overlap is 1 by definition; other maps have unaligned axes"})
            ax=axes[i,j]
            scatter(ax,xy,y,size=1.2,alpha=.25)
            ax.set_title(f"seed={seed}; neighbours={nb}\ntrust={tw:.3f}; overlap={score:.3f}",fontsize=10)
            ax.set_xticks([]);ax.set_yticks([])
            ax.spines[["top","right"]].set_visible(False)
            if i==0 and j==0:
                pd.DataFrame({"species":y,"UMAP1":xy[:,0],"UMAP2":xy[:,1]}).to_csv(
                    out/"stability_first_fit_coords.csv",index=False)
    fig.suptitle("Unseen-only UMAP stability: separate fits, NOT aligned axes",fontsize=16,fontweight="bold")
    save_plot(out/"02_stability_grid.png",fig)
    pd.DataFrame(result).to_csv(out/"umap_stability_metrics.csv",index=False)
    logging.info("UMAP stability metrics: %s",pd.DataFrame(result).to_string(index=False))


def main():
 a=args_parser()
 logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",stream=sys.stdout)
 npz=find_feature_file(a);logging.info("DenseNet features: %s",npz)
 with np.load(npz,allow_pickle=True) as z:
  inventory=[{"key":k,"shape":list(np.asarray(z[k]).shape),"dtype":str(z[k].dtype)} for k in z.files]
 for info in inventory: logging.info("NPZ: %s",info)
 if a.inspect_only:
  logging.info("Inspection complete; no plate metadata required.");return
 X,y,meta,keys=read_features(npz)
 idx=choose_balanced(y,a.max_per_species,a.seed);X=X[idx];y=y[idx]
 output=a.output;output.mkdir(parents=True,exist_ok=True)
 pd.DataFrame({"species":UNSEEN,"n_cells":[int(sum(y==s)) for s in UNSEEN]}).to_csv(output/"unseen_species_counts.csv",index=False)
 pca,Xp=fit_pca(X,a.pca_components,a.seed)
 seeds=[int(x) for x in a.seeds.split(",") if x.strip()]; neighbors=[int(x) for x in a.neighbors.split(",") if x.strip()]
 if not seeds or not neighbors or min(neighbors)<2: raise ValueError("Invalid seed/neighbour settings")
 manifest={"source_npz":str(npz),"n_cells":len(y),"feature_dim":X.shape[1],
  "pca_components":Xp.shape[1],"pca_variance_retained":float(pca.explained_variance_ratio_.sum()),
  "sampling_seed":a.seed,"per_species_cap":a.max_per_species,"umap_seeds":seeds,
  "umap_neighbors":neighbors,"umap_min_dist":a.min_dist,"knn_k":a.knn_k,
  "note":"Exploratory, not a validated unseen-organism classifier. 2D distances are not quantitative."}
 (output/"run_manifest.json").write_text(json.dumps(manifest,indent=2))
 highdim_knn(Xp,y,a.knn_k,output)
 logging.info("Fitting primary unseen-only UMAP")
 xy=umap_fit(Xp,30,a.seed,a.min_dist)
 pd.DataFrame({"species":y,"source_npz_row_index":idx,"UMAP1":xy[:,0],"UMAP2":xy[:,1]}).to_csv(output/"unseen_umap_coordinates.csv",index=False)
 main_figures(y,xy,output)
 if not a.only_baseline:
  rng=np.random.default_rng(a.seed)
  subs=np.sort(rng.choice(len(y),size=min(len(y),a.stability_subset),replace=False))
  stability_grid(Xp,y,seeds,neighbors,a.min_dist,subs,output)
 logging.info("SUCCESS: %s",output)

if __name__=="__main__":
 try:main()
 except Exception as exc:
  logging.exception("Unseen UMAP analysis failed: %s",exc);sys.exit(1)
