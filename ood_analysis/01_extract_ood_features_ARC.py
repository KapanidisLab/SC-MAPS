#!/usr/bin/env python3
"""SC-MAPS OOD input extraction from the original four-channel DenseNet121 concat checkpoint.

Reconstructed release script. Model module/state key conventions taken from
SC_MAPS_CHRIS_REVIEW_ALL_IN_ONE_ARC_FIXED2.py in the author's Python.zip.
DO NOT use output of a different checkpoint as if it reproduced the published OOD results.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import densenet121

SPECIES = ['ACAA','ECOL','ENTC','KLAE','KLPN','PSAR','PSMA','SAENA','SAUR','SEPI','SERM','SLUG','STCP']
UNSEEN = {'BSB','EF','EFM','HFLU','LMON'}
CHANNEL_INDEX = {'PHASECONTRAST':0, 'DAPI':2, 'WGA':3, 'FM1':4}

def canon_channel(name):
    s = re.sub('[^A-Z0-9]', '', str(name).upper())
    return {'PHASE':'PHASECONTRAST','PHASECONTRASTIMAGE':'PHASECONTRAST',
            'WGAAF647':'WGA','WGA647':'WGA','FM143':'FM1'}.get(s,s)

def canonical_species(s):
    s=str(s).strip().upper()
    return {'HELU':'HFLU', 'HINF':'HFLU'}.get(s,s)

def state_dict(checkpoint):
    for name in ('model_state_dict','state_dict','model'):
        if name in checkpoint and isinstance(checkpoint[name],dict):
            d=checkpoint[name];break
    else:
        if all(torch.is_tensor(v) for v in checkpoint.values()): d=checkpoint
        else: raise ValueError('Checkpoint lacks a state_dict')
    return {k.removeprefix('module.'):v for k,v in d.items()}

class FourBranchDenseNet121Concat(nn.Module):
    """Architecture matching the author's Chris-review inference code."""
    def __init__(self, channels, projection_dim, hidden_dim, classes, dropout):
        super().__init__()
        self.channels=list(channels)
        self.branches=nn.ModuleDict()
        self.projections=nn.ModuleDict()
        for name in channels:
            backbone=densenet121(weights=None)
            old=backbone.features.conv0
            backbone.features.conv0=nn.Conv2d(1,old.out_channels,kernel_size=old.kernel_size,
                stride=old.stride,padding=old.padding,bias=False)
            backbone.classifier=nn.Identity()
            self.branches[name]=backbone
            self.projections[name]=nn.Sequential(nn.Linear(1024,projection_dim),
                nn.LayerNorm(projection_dim),nn.GELU(),nn.Dropout(dropout))
        self.classifier=nn.Sequential(nn.Linear(projection_dim*len(channels),hidden_dim),
            nn.LayerNorm(hidden_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden_dim,classes))

    def forward_with_penultimate(self, inputs):
        x=torch.cat([self.projections[name](self.branches[name](inputs[name]))
                     for name in self.channels], dim=1)
        for layer in list(self.classifier.children())[:-1]: x=layer(x)
        return self.classifier[-1](x), x

def load_model(path,device):
    try: ckpt=torch.load(path,map_location='cpu',weights_only=False)
    except TypeError: ckpt=torch.load(path,map_location='cpu')
    sd=state_dict(ckpt)
    in_state=list(dict.fromkeys(re.match(r'^branches\.([^.]+)\.',k).group(1)
        for k in sd if re.match(r'^branches\.([^.]+)\.',k)))
    if len(in_state)!=4: raise ValueError(f'Expected four branches, found {in_state}')
    metadata=ckpt.get('selected_channels',[]) if isinstance(ckpt,dict) else []
    names=[]
    for c in (metadata if len(metadata)==4 else ['DAPI','PHASECONTRAST','WGA','FM1']):
        found=[x for x in in_state if canon_channel(x)==canon_channel(c)]
        if len(found)!=1: raise ValueError(f'Cannot uniquely match channel {c}: {in_state}')
        names.append(found[0])
    if set(map(canon_channel,names)) != set(CHANNEL_INDEX):
        raise ValueError(f'Unexpected channels: {names}')
    projection_dim=int(sd[f'projections.{names[0]}.0.weight'].shape[0])
    hidden_dim=int(sd['classifier.0.weight'].shape[0])
    n_classes=int(sd['classifier.4.weight'].shape[0])
    classes=ckpt.get('class_names',ckpt.get('classes',SPECIES)) if isinstance(ckpt,dict) else SPECIES
    if isinstance(ckpt,dict) and isinstance(ckpt.get('class_to_idx'),dict):
        classes=[c for c,_ in sorted(ckpt['class_to_idx'].items(),key=lambda q:q[1])]
    classes=list(classes)
    if n_classes!=13 or set(classes)!=set(SPECIES):
        raise ValueError(f'Checkpoint class-order mismatch; got {classes}')
    dropout=float(ckpt.get('fusion_dropout',ckpt.get('dropout',0.30))) if isinstance(ckpt,dict) else .30
    model=FourBranchDenseNet121Concat(names,projection_dim,hidden_dim,n_classes,dropout)
    model.load_state_dict(sd,strict=True)
    model=model.to(device).eval()
    return model,names,classes,hidden_dim

def resolve_crop(raw,root):
    raw=str(raw).strip().replace('\\','/')
    candidate=Path(raw)
    if candidate.is_file(): return candidate
    # Dataset-relative paths, including Windows paths from the original manifest.
    choices=[root/candidate,root/'crops'/candidate.name]
    for marker in ('Trillian_DATASET_with_holdout/', 'SC_MAPS_NOVEL_SPECIES_SINGLE_CELL_DATASET/'):
        if marker.lower() in raw.lower():
            pos=raw.lower().rfind(marker.lower())+len(marker)
            choices.insert(0,root/raw[pos:])
    cut=re.search(r'(?:^|/)(crops/.*)$',raw,re.I)
    if cut: choices.insert(0,root/cut.group(1))
    for c in choices:
        if c.is_file(): return c
    raise FileNotFoundError(f'Cannot resolve {raw} under {root}')

def sample_identifier(row,species,path):
    for col in ('sample_id','sample','sample_name','experimental_sample','source_sample'):
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            return species+'__'+str(row[col]).strip()
    plate=next((str(row[c]) for c in ('plate','strain') if c in row.index and pd.notna(row[c])),None)
    rep=next((str(row[c]) for c in ('replicate','rep') if c in row.index and pd.notna(row[c])),None)
    series=next((str(row[c]) for c in ('series','field_id','image_id') if c in row.index and pd.notna(row[c])),None)
    stem=Path(path).stem
    if plate is None:
        m=re.search(r'plate[_-]?(\d+)',stem,re.I);plate=m.group(1) if m else '?'
    if rep is None:
        m=re.search(r'rep[_-]?(\d+)',stem,re.I);rep=m.group(1) if m else '?'
    if series is None:
        m=re.search(r'series[_-]?(\d+)',stem,re.I)
        series=m.group(1) if m else re.sub(r'_cell[_-]?\d+.*$','',stem,flags=re.I)
    return f'{species}__plate{plate}__rep{rep}__{series}'

class CropDataset(Dataset):
    def __init__(self, paths, channels):
        self.paths=paths
        self.channels=channels
    def __len__(self): return len(self.paths)
    def __getitem__(self,i):
        path=self.paths[i]
        x=np.load(path,allow_pickle=False,mmap_mode='r')
        if x.ndim!=3: raise ValueError(f'Unexpected crop shape {x.shape}: {path}')
        if x.shape[0]==5: pass
        elif x.shape[-1]==5: x=np.moveaxis(x,-1,0)
        else: raise ValueError(f'Expected five channels, got {x.shape}: {path}')
        result={}
        for name in self.channels:
            plane=np.asarray(x[CHANNEL_INDEX[canon_channel(name)]],dtype=np.float32)
            plane=np.nan_to_num(plane,nan=0,posinf=0,neginf=0)
            if plane.shape!=(96,96): raise ValueError(f'Expected native 96x96 images, got {plane.shape}: {path}')
            mean=float(plane.mean());std=max(float(plane.std()),1e-6)
            result[name]=torch.from_numpy(np.ascontiguousarray(((plane-mean)/std)[None]))
        return result

def select_rows(df, species_col, max_per_species,seed):
    if not max_per_species: return df.reset_index(drop=True)
    rng=np.random.default_rng(seed)
    labels=df[species_col].to_numpy()
    inds=[]
    for species in np.unique(labels):
        idx=np.flatnonzero(labels==species)
        inds.extend(rng.choice(idx,size=min(len(idx),max_per_species),replace=False).tolist())
    return df.iloc[sorted(inds)].reset_index(drop=True)

def run_split(name,manifest,root,model,channels,classes,hidden_dim,args,cap,seed):
    df=pd.read_csv(manifest,low_memory=False)
    species_col=next((c for c in ('species','true_species','label','target') if c in df.columns),None)
    path_col=next((c for c in ('crop_path','npy_path','path','file_path') if c in df.columns),None)
    if not species_col or not path_col: raise ValueError(f'{manifest} missing species/crop_path columns: {df.columns.tolist()}')
    df[species_col]=df[species_col].astype(str).map(canonical_species)
    allowed=UNSEEN if name=='unseen' else set(SPECIES)
    df=df[df[species_col].isin(allowed)].copy()
    if df.empty: raise ValueError(f'No {name} rows in {manifest}; expected {sorted(allowed)}')
    df=select_rows(df,species_col,cap,seed)
    roots=[root]
    paths=[str(resolve_crop(p,root)) for p in df[path_col].tolist()]
    sample_ids=[sample_identifier(row,sp,p) for (_,row),sp,p in zip(df.iterrows(),df[species_col],paths)]
    file_tags=[str(Path(p).resolve()) for p in paths]
    batch=DataLoader(CropDataset(paths,channels),batch_size=args.batch_size,shuffle=False,
        num_workers=args.workers,pin_memory=args.device=='cuda',persistent_workers=args.workers>0)
    features=np.empty((len(df),hidden_dim),dtype=np.float32)
    probs=np.empty((len(df),len(classes)),dtype=np.float32)
    pos=0
    with torch.inference_mode():
        for num,item in enumerate(batch,1):
            item={key:val.to(args.device,non_blocking=True) for key,val in item.items()}
            with torch.autocast(device_type=args.device,dtype=torch.float16,enabled=(args.device=='cuda' and args.amp)):
                logits,hidden=model.forward_with_penultimate(item)
            count=len(hidden)
            features[pos:pos+count]=hidden.float().cpu().numpy()
            probs[pos:pos+count]=torch.softmax(logits.float(),dim=1).cpu().numpy()
            pos+=count
            if num%100==0: print(f'{name}: {pos:,}/{len(df):,}',flush=True)
    target=args.outdir/f'{name}.npz'
    np.savez_compressed(target,features=features,probs=probs,
        species=np.asarray(df[species_col].tolist(),dtype='U64'),
        sample_id=np.asarray(sample_ids,dtype='U512'),
        crop_path=np.asarray(file_tags,dtype='U2048'),class_names=np.asarray(classes,dtype='U64'))
    print(f'Saved {target}: {len(df):,} cells, {len(set(sample_ids))} samples',flush=True)
    return file_tags

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--checkpoint',required=True,type=Path,help='13-class four-channel DenseNet121 concat best_fusion_model.pt')
    p.add_argument('--known-root',required=True,type=Path,help='Extracted Trillian_DATASET_with_holdout folder')
    p.add_argument('--unseen-root',required=True,type=Path,help='Extracted SC_MAPS_NOVEL_SPECIES_SINGLE_CELL_DATASET folder')
    p.add_argument('--reference-manifest',type=Path,default=None,help='Default known-root/splits/train.csv')
    p.add_argument('--calibration-manifest',type=Path,default=None,help='Default known-root/splits/val.csv')
    p.add_argument('--known-test-manifest',type=Path,default=None,help='Default known-root/splits/holdout_plate1.csv')
    p.add_argument('--unseen-manifest',type=Path,default=None,help='Default unseen-root/cell_manifest.csv')
    p.add_argument('--outdir',type=Path,default=Path('ood_outputs/input_arrays'))
    p.add_argument('--reference-per-species',type=int,default=2000,help='0 selects ALL available cells')
    p.add_argument('--calibration-per-species',type=int,default=2000)
    p.add_argument('--known-test-per-species',type=int,default=2000)
    p.add_argument('--unseen-per-species',type=int,default=2000)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True)
    args=p.parse_args()
    args.device='cuda' if torch.cuda.is_available() else 'cpu'
    if any(x<0 for x in (args.reference_per_species,args.calibration_per_species,args.known_test_per_species,args.unseen_per_species)):
        p.error('Sampling limits must be non-negative')
    if not args.checkpoint.is_file(): p.error(f'Missing checkpoint: {args.checkpoint}')
    args.outdir.mkdir(parents=True,exist_ok=True)
    known=args.known_root; unseen=args.unseen_root
    manifests={
        'reference':(args.reference_manifest or known/'splits/train.csv',known,args.reference_per_species),
        'calibration':(args.calibration_manifest or known/'splits/val.csv',known,args.calibration_per_species),
        'known_test':(args.known_test_manifest or known/'splits/holdout_plate1.csv',known,args.known_test_per_species),
        'unseen':(args.unseen_manifest or unseen/'cell_manifest.csv',unseen,args.unseen_per_species),
    }
    for label,(csv,_,_) in manifests.items():
        if not csv.is_file(): p.error(f'Missing {label} manifest: {csv}')
    model,channels,classes,hidden_dim=load_model(args.checkpoint,args.device)
    paths={}
    for i,(name,(manifest,root,cap)) in enumerate(manifests.items()):
        paths[name]=run_split(name,manifest,root,model,channels,classes,hidden_dim,args,cap,args.seed+i)
    for a,b in [('reference','calibration'),('reference','known_test'),('calibration','known_test'),('unseen','known_test')]:
        overlap=set(paths[a]) & set(paths[b])
        if overlap: raise RuntimeError(f'DATA LEAKAGE: {len(overlap)} crops overlap between {a} and {b}')
    ckhash=hashlib.sha256()
    with args.checkpoint.open('rb') as fh:
        for piece in iter(lambda:fh.read(2**20),b''): ckhash.update(piece)
    meta={'checkpoint_sha256':ckhash.hexdigest(),'checkpoint':str(args.checkpoint),
          'channels':channels,'classes':classes,'seed':args.seed,
          'sample_limits_per_species':{k:v[2] for k,v in manifests.items()},
          'manifests':{k:str(v[0]) for k,v in manifests.items()},
          'input_count':{k:len(v) for k,v in paths.items()}}
    (args.outdir/'provenance.json').write_text(json.dumps(meta,indent=2))
    print('Extraction finished. Run 02_ood_scoring_and_population.py next.')

if __name__=='__main__': main()
