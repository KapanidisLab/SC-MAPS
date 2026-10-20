#!/usr/bin/env python3
"""SC-MAPS OOD scoring: MSP, entropy, kNN, Mahalanobis, calibration, population tests.

Reconstructed implementation of the study methods; numeric identity to manuscript
requires the original checkpoint, calibration/reference cohort and random draws.
No unseen organism is used to fit the model, PCA, or select rejection thresholds.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import xlogy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score, average_precision_score

METHODS=['knn','mahalanobis','entropy','msp']
SIZES=[1,5,10,20,50]

def read_array(path):
    if not path.is_file(): raise FileNotFoundError(f'Missing {path}; first run 01_extract_ood_features_ARC.py')
    with np.load(path,allow_pickle=False) as z:
        d={k:np.array(z[k]) for k in ('features','probs','species','sample_id','crop_path','class_names')}
    if len(d['features'])!=len(d['probs']) or len(d['probs'])!=len(d['species']):
        raise ValueError(f'Inconsistent array lengths in {path}')
    if not (np.isfinite(d['features']).all() and np.isfinite(d['probs']).all()):
        raise ValueError(f'Non-finite features/probabilities in {path}')
    if not np.allclose(d['probs'].sum(axis=1),1,atol=1e-3):
        raise ValueError(f'Invalid probability sums in {path}')
    return d

def model_fit(ref,pca_components,knn_neighbors):
    scaler=StandardScaler().fit(ref['features'])
    x=scaler.transform(ref['features'])
    n=min(pca_components,x.shape[0]-1,x.shape[1])
    if n<2: raise ValueError('Too few known reference cells to fit PCA')
    pca=PCA(n_components=n,svd_solver='randomized',random_state=42).fit(x)
    x=pca.transform(x).astype(np.float32)
    knn=NearestNeighbors(n_neighbors=min(knn_neighbors,len(x)),metric='euclidean',algorithm='auto',n_jobs=-1)
    knn.fit(x)
    class_fits=[]
    for species in ref['class_names'].tolist():
        group=x[ref['species']==species]
        if len(group)<max(3,n//2):
            raise ValueError(f'Insufficient reference features for species {species}: {len(group)}')
        cov=LedoitWolf().fit(group)
        class_fits.append((species,cov.location_.astype(np.float32),cov.precision_.astype(np.float32)))
    return scaler,pca,knn,class_fits

def score_rows(d,models,batch_size):
    scaler,pca,knn,class_fits=models
    p=d['probs'].astype(np.float64)
    score={'msp':(1-p.max(axis=1)).astype(np.float32),
           'entropy':(-xlogy(p,p).sum(axis=1)/np.log(p.shape[1])).astype(np.float32)}
    knn_scores=[]; mahal_scores=[]
    for a in range(0,len(p),batch_size):
        b=min(len(p),a+batch_size)
        f=pca.transform(scaler.transform(d['features'][a:b])).astype(np.float32)
        distances=knn.kneighbors(f,return_distance=True)[0]
        knn_scores.append(distances.mean(axis=1).astype(np.float32))
        mahal=np.full(len(f),np.inf,dtype=np.float32)
        for species,mean,precision in class_fits:
            delta=f-mean
            dist=np.einsum('ni,ij,nj->n',delta,precision,delta,optimize=True)
            mahal=np.minimum(mahal,np.maximum(0,dist))
        mahal_scores.append(np.sqrt(mahal))
    score['knn']=np.concatenate(knn_scores)
    score['mahalanobis']=np.concatenate(mahal_scores)
    return score

def threshold(scores,retention):
    # Novelty high => OOD; scores above this threshold are rejected.
    return float(np.quantile(scores,retention))

def join_scores(data,score):
    rows=pd.DataFrame({'species':data['species'], 'sample_id':data['sample_id'],
                       'crop_path':data['crop_path']})
    for name in METHODS:rows[f'ood_{name}']=score[name]
    return rows

def sample_populations(data,scores,sizes,repeats,seed):
    """Equal sample weighting. No replacement within a population; between repeats may reuse."""
    rng=np.random.default_rng(seed)
    species=np.asarray(data['species'],dtype=str)
    ids=np.asarray(data['sample_id'],dtype=str)
    uniq, inverse=np.unique(ids,return_inverse=True)
    records=[]
    for j,sample in enumerate(uniq):
        ix=np.flatnonzero(inverse==j)
        if np.unique(species[ix]).size!=1:
            raise ValueError(f'Sample ID contains multiple true species: {sample}')
        for n in sizes:
            if len(ix)<n:continue
            for rep in range(repeats):
                chosen=rng.choice(ix,size=n,replace=False)
                entry={'species':species[ix[0]],'sample_id':sample,
                       'cells':n,'repeat':rep}
                for name in METHODS:entry[name]=float(np.mean(scores[name][chosen]))
                records.append(entry)
    return pd.DataFrame.from_records(records)

def report_ood_scores(scored,known,unseen,thresholds,output):
    records=[]; organism=[]
    for name in METHODS:
        y=np.r_[np.zeros(len(known[METHODS[0]]),dtype=int),np.ones(len(unseen[METHODS[0]]),dtype=int)]
        s=np.r_[known[name],unseen[name]]
        cutoff=thresholds[name]
        records.append({'method':name,'AUROC':float(roc_auc_score(y,s)),
            'AUPRC':float(average_precision_score(y,s)),
            'known_retention':float(np.mean(known[name]<=cutoff)),
            'unseen_rejection':float(np.mean(unseen[name]>cutoff)),
            'threshold':cutoff,'n_known':len(known[name]),'n_unseen':len(unseen[name])})
        for species in sorted(scored['unseen']['species'].unique()):
            sel=(scored['unseen']['species']==species).to_numpy()
            organism.append({'method':name,'unseen_species':species,'n_cells':int(sel.sum()),
                'rejection_rate':float(np.mean(unseen[name][sel]>cutoff))})
    pd.DataFrame(records).to_csv(output/'ood_single_cell_metrics.csv',index=False)
    pd.DataFrame(organism).to_csv(output/'ood_per_organism_rejection.csv',index=False)
    return records

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input-dir',type=Path,default=Path('ood_outputs/input_arrays'))
    p.add_argument('--output-dir',type=Path,default=Path('ood_outputs/results'))
    p.add_argument('--pca-dim',type=int,default=50)
    p.add_argument('--knn-neighbours',type=int,default=25)
    p.add_argument('--retention',type=float,default=0.95,help='Known calibration retention')
    p.add_argument('--population-sizes',type=int,nargs='+',default=SIZES)
    p.add_argument('--repeats',type=int,default=200)
    p.add_argument('--batch-size',type=int,default=2048)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if not (0<args.retention<1):p.error('retention must be between 0 and 1')
    if args.repeats<1 or any(n<1 for n in args.population_sizes):p.error('population sizes/repeats must be positive')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    splits={n:read_array(args.input_dir/f'{n}.npz') for n in ('reference','calibration','known_test','unseen')}
    order=splits['reference']['class_names'].tolist()
    if len(order)!=13:
        raise ValueError('Expected 13 species in original checkpoint')
    for name,d in splits.items():
        if d['class_names'].tolist()!=order:raise ValueError(f'{name} class probability order differs')
    # Defensive validation: no cell can be in multiple input cohorts.
    for a,b in (('reference','calibration'),('reference','known_test'),('calibration','known_test')):
        common=np.intersect1d(splits[a]['crop_path'],splits[b]['crop_path'])
        if common.size: raise ValueError(f'Leakage: {common.size} crops appear in {a} and {b}')
    print('Fitting scaler, PCA, 25-NN and class-conditional covariance on known reference only ...',flush=True)
    models=model_fit(splits['reference'],args.pca_dim,args.knn_neighbours)
    print('PCA components:',models[1].n_components_,flush=True)
    scores={}
    for split,d in splits.items():
        print('Scoring',split,len(d['features']),'cells',flush=True)
        scores[split]=score_rows(d,models,args.batch_size)
    # Operating points: threshold chosen on known calibration only.
    thresholds={name:threshold(scores['calibration'][name],args.retention) for name in METHODS}
    print('Cell-level calibration thresholds:',thresholds,flush=True)
    sc_frames={split:join_scores(d,scores[split]) for split,d in splits.items()}
    for split in ('calibration','known_test','unseen'):
        f=sc_frames[split].copy();f.insert(0,'dataset',split)
        f.to_csv(args.output_dir/f'ood_cell_scores_{split}.csv',index=False)
    metrics=report_ood_scores(sc_frames,
        {name:scores['known_test'][name] for name in METHODS},
        {name:scores['unseen'][name] for name in METHODS},thresholds,args.output_dir)
    sensitivity=[]
    for retain in (0.90,0.95,0.99):
        for method in METHODS:
            t=threshold(scores['calibration'][method],retain)
            sensitivity.append({'method':method,'target_known_retention':retain,'threshold':t,
                'observed_known_retention':float(np.mean(scores['known_test'][method]<=t)),
                'unseen_rejection':float(np.mean(scores['unseen'][method]>t))})
    pd.DataFrame(sensitivity).to_csv(args.output_dir/'ood_threshold_sensitivity.csv',index=False)
    (args.output_dir/'ood_thresholds.json').write_text(json.dumps({'thresholds':thresholds,
        'known_retention':args.retention,'pca_components':models[1].n_components_,
        'knn_neighbours':min(args.knn_neighbours,len(splits['reference']['features'])),
        'mahalanobis':'minimum class-specific LedoitWolf covariance distance',
        'direction':'high score indicates unseen',
        'calibration':'known-only val.csv, independent of unseen organisms'},indent=2))
    population={}
    for k in ('calibration','known_test','unseen'):
        population[k]=sample_populations(splits[k],scores[k],args.population_sizes,args.repeats,args.seed+{'calibration':1,'known_test':2,'unseen':3}[k])
        if population[k].empty:
            raise ValueError(f'No valid populations for {k}: sample groups too small or invalid sample IDs')
        population[k].insert(0,'dataset',k)
    pop=pd.concat(list(population.values()),ignore_index=True)
    pop.to_csv(args.output_dir/'ood_population_scores.csv',index=False)
    pop_metrics=[]
    for n in args.population_sizes:
        cal=population['calibration'][population['calibration']['cells']==n]
        known=population['known_test'][population['known_test']['cells']==n]
        unseen=population['unseen'][population['unseen']['cells']==n]
        if cal.empty or known.empty or unseen.empty:
            print(f'WARNING: cannot report n={n}, insufficient samples with {n} cells in at least one cohort')
            continue
        for method in METHODS:
            # Calibrate separately for each population size using known samples only.
            t=threshold(cal[method].to_numpy(),args.retention)
            observed_known=float(np.mean(known[method]<=t))
            observed_unseen=float(np.mean(unseen[method]>t))
            pop_metrics.append({'cells':n,'method':method,'threshold':t,
                'known_retention':observed_known,'unseen_rejection':observed_unseen,
                'known_samples':known['sample_id'].nunique(),'unseen_samples':unseen['sample_id'].nunique(),
                'calibration_samples':cal['sample_id'].nunique(),
                'repeats_per_sample':args.repeats,'calibration_mode':'population_size_specific'})
    pd.DataFrame(pop_metrics).to_csv(args.output_dir/'ood_population_metrics.csv',index=False)
    run={'methods':METHODS,'retention':args.retention,'population_sizes':args.population_sizes,
         'repeats':args.repeats,'seed':args.seed,'reference_n':len(splits['reference']['features']),
         'calibration_n':len(splits['calibration']['features']),'known_test_n':len(splits['known_test']['features']),
         'unseen_n':len(splits['unseen']['features']),
         'population_interpretation':'repeats reflect cell sampling, not independent biological samples',
         'published_results_warning':'Reconstructed methods implementation: verify checkpoint and original cohort/threshold conventions before claiming exact figure reproduction'}
    (args.output_dir/'run_summary.json').write_text(json.dumps(run,indent=2))
    print('\nSingle cell evaluation:')
    print(pd.DataFrame(metrics).to_string(index=False,float_format=lambda v:f'{v:.4f}'))
    print('\nFinished. Run 03_plot_ood_figures.py next.')

if __name__=='__main__':main()
