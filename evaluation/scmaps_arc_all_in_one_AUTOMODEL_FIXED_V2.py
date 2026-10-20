#!/usr/bin/env python3
"""SC-MAPS: checkpoint-driven four-model ARC inference and routing analysis (single file).

Modes:
  --inspect: print checkpoint candidates, checkpoint formats, and sample data.
  --check-models: strictly validate all four models before inference.
  normal: run four checkpoint inferences, then compare concat and SE routing.
  --compare-only: compare previously generated CSVs in --out, without loading weights.

SC-MAPS DenseNet121 concat and SE attention state_dict checkpoints are reconstructed
from saved tensor names/shapes and training metadata, then loaded STRICTLY. Training
script imports are needed only if checkpoints use a different architecture.
No retraining or re-extraction; independent Plate-1 cells are reused.

Input: per-cell five-channel .npy (PhaseContrast, BF, DAPI, WGA, FM1-43).
The four-channel default is DAPI,PhaseContrast,WGA,FM1 (stack indices 2,0,3,4).
Use --channel-indices to override if your checkpoint trained on other channels.

A CSV manifest with file path, species and pure-population sample ID is STRONGLY
recommended to avoid inferring metadata from filenames. If absent, supply
--data-root for .npy discovery and --sample-regex for sample IDs.
"""
import argparse, importlib.util, json, os, re, sys, warnings
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, silhouette_score, balanced_accuracy_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

RUNS = {
 'species_se': '/data/phys-deep-learning-bacteria/phys3115/Bacteria-data-15 species/Trillian/Trillian_attention_fusion_training_outputs_ARC/run_20260721_160005_BF-DAPI-PHASECONTRAST-WGA-FM1_se_attention',
 'panel_se': '/data/phys-deep-learning-bacteria/phys3115/Bacteria-data-15 species/Trillian/Trillian_8class_fusion_training_outputs_ARC/run_20260824_122834_panel2_8class_fusion_se_attention',
 'panel_concat': '/data/phys-deep-learning-bacteria/phys3115/Bacteria-data-15 species/Trillian/Trillian_8class_fusion_training_outputs_ARC/run_20260824_122834_panel2_8class_fusion_concat',
 'species_concat': '/data/phys-deep-learning-bacteria/phys3115/Bacteria-data-15 species/Trillian/Trillian_attention_fusion_training_outputs_ARC/fusion_training_outputs_CONCAT/20260706_152625_DAPI-PHASECONTRAST-WGA-FM1_concat',
}
# Exact Plate 1 dataset location, confirmed in Chris-review ARC log (2026-10-08).
DEFAULT_SCRATCH_PARENT = Path('/scratch/phys3115/bacteria_data')
DEFAULT_DATA_ROOT = DEFAULT_SCRATCH_PARENT / 'Trillian_DATASET_with_holdout'
DEFAULT_HOLDOUT_CSV = DEFAULT_DATA_ROOT / 'splits' / 'holdout_plate1.csv'
DEFAULT_OUTPUT = Path('/data/phys-deep-learning-bacteria/phys3115/Bacteria-data-15 species/Trillian/SCMAPS_8class_comparison_ARC')

SPECIES=['ACAA','ECOL','ENTC','KLAE','KLPN','PSAR','PSMA','SAENA','SAUR','SEPI','SERM','SLUG','STCP']
GROUPS=['Routine_Enterobacterales','AmpC_Enterobacterales','Salmonella','Pseudomonas','Acinetobacter','Stenotrophomonas','S_aureus_like','CoNS']
SP2GP={'ECOL':0,'KLPN':0,'KLAE':1,'ENTC':1,'SERM':1,'SAENA':2,'PSAR':3,'ACAA':4,'PSMA':5,'SAUR':6,'SLUG':6,'SEPI':7,'STCP':7}

def cli():
 p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 p.add_argument('--inspect', action='store_true')
 p.add_argument('--check-models',action='store_true',help='Strict-load all four fusion checkpoints and verify channel/class order without processing 221,726 cells')
 p.add_argument('--compare-only',action='store_true',help='Use existing *_predictions.csv files in --out, skipping checkpoint inference')
 p.add_argument('--skip-existing',action='store_true',help='Reuse any existing model prediction CSVs instead of rerunning those models')
 p.add_argument('--scratch-root',default=str(DEFAULT_SCRATCH_PARENT),help='Scratch parent directory (default: existing extracted bacteria_data)')
 p.add_argument('--manifest',default=str(DEFAULT_HOLDOUT_CSV),help='Defaults to splits/holdout_plate1.csv on scratch (Plate 1 only)')
 p.add_argument('--data-root',default=str(DEFAULT_DATA_ROOT),help='Defaults to extracted Trillian_DATASET_with_holdout on scratch')
 p.add_argument('--split-col',default='',help='Optional manifest column indicating holdout partition or plate')
 p.add_argument('--holdout-value',default='',help='Keep only rows matching this value in --split-col (e.g. Plate1)')
 p.add_argument('--file-col',default='npy_path');p.add_argument('--species-col',default='true_species');p.add_argument('--sample-col',default='sample_id')
 p.add_argument('--sample-regex',default='',help='Regex with named group (?P<sample>...) to extract biological sample from path')
 p.add_argument('--out',default=str(DEFAULT_OUTPUT))
 p.add_argument('--checkpoint-species-se');p.add_argument('--checkpoint-species-concat');p.add_argument('--checkpoint-panel-se');p.add_argument('--checkpoint-panel-concat')
 p.add_argument('--model-source',help='Original training .py file defining the custom fusion architecture')
 p.add_argument('--species-class',default='');p.add_argument('--panel-class',default='')
 p.add_argument('--model-kwargs',default='{}',help='JSON constructor kwargs; placeholders {n_classes}, {fusion}')
 for key in RUNS:
  p.add_argument('--model-source-'+key.replace('_','-'),default='',help='Override training source .py for '+key)
  p.add_argument('--model-class-'+key.replace('_','-'),default='',help='Override model class for '+key)
  p.add_argument('--model-kwargs-'+key.replace('_','-'),default='',help='Override JSON constructor kwargs for '+key)
 p.add_argument('--species-class-order',default='',help='Explicit species neuron order ONLY if checkpoint/run metadata lacks it; comma-separated')
 p.add_argument('--panel-class-order',default='',help='Explicit 8-group neuron order ONLY if checkpoint/run metadata lacks it; comma-separated')
 p.add_argument('--channel-indices',default='2,0,3,4',help='Default four channels and order in each checkpoint')
 for key in RUNS:p.add_argument('--channel-indices-'+key.replace('_','-'),default='',help='Override channel indices for '+key+' (supports 4 or 5 channels)')
 for key in RUNS:p.add_argument('--input-layout-'+key.replace('_','-'),default='',choices=['','branches','stack'],help='Override input layout for '+key)
 p.add_argument('--input-layout',choices=['branches','stack'],default='stack',help='Only used with custom model sources; automatic SC-MAPS loader uses stacked tensors')
 p.add_argument('--normalisation',choices=['per_channel_crop_zscore','none'],default='per_channel_crop_zscore')
 p.add_argument('--batch-size',type=int,default=64);p.add_argument('--workers',type=int,default=4)
 p.add_argument('--max-cells',type=int,default=0,help='0 means all matched cells')
 p.add_argument('--max-umap-cells',type=int,default=12000,help='Balanced cells per model to save penultimate features for')
 p.add_argument('--feature-layer',default='auto',help='auto selects last linear classifier, or set exact named module, or use --no-features')
 p.add_argument('--no-features',action='store_true');p.add_argument('--seed',type=int,default=42)
 p.add_argument('--bootstraps',type=int,default=2000);p.add_argument('--repeats',type=int,default=200)
 p.add_argument('--min-cells',type=int,default=200,help='Minimum sample size for all aggregation sizes; default 200')
 p.add_argument('--sizes',default='1,2,3,5,10,20,30,50,75,100,150,200')
 return p.parse_args()

def find_checkpoint(run):
 path=Path(run)
 if not path.exists():raise FileNotFoundError(f'Run directory missing on ARC: {path}')
 candidates=[]
 for ext in ('*.pt','*.pth','*.ckpt','*.torchscript'):
  candidates.extend(path.rglob(ext))
 candidates=[x for x in candidates if x.is_file() and x.stat().st_size>10240]
 if not candidates:raise FileNotFoundError(f'No weights found under {path}')
 def score(x):
  # Prefer the final checkpoint explicitly saved by the fusion training run.
  # Do not select a nested intermediate checkpoint merely because it has
  # a newer modification time than the run-root best_fusion_model.pt.
  filename=x.name.lower()
  exact=int(filename=='best_fusion_model.pt')
  root=int(x.parent.resolve()==path.resolve())
  score=(exact*1000+root*100+int('best' in filename)*60+
         int('fusion' in filename)*30+int('model' in filename)*10-
         int('optimizer' in filename)*1000)
  return (score,-len(x.relative_to(path).parts),x.stat().st_mtime)
 return sorted(candidates,key=score,reverse=True)

def torch_load(path):
 import torch
 try:return torch.load(path,map_location='cpu',weights_only=False)
 except Exception as e:
  try:return torch.jit.load(str(path),map_location='cpu')
  except Exception:raise RuntimeError(f'Cannot load {path}: {e}')

def inspect_run(name, directory, override):
 print(f'\n=== {name}: {directory} ===',flush=True)
 try:
  paths=[Path(override)] if override else find_checkpoint(directory)
  for x in paths[:10]:print(f' candidate: {x} ({x.stat().st_size/1e6:.1f} MB)')
  obj=torch_load(paths[0]);print(' selected type:',type(obj))
  if isinstance(obj,dict):
   print(' checkpoint keys:',list(obj)[:40])
   sd=state_dict(obj)
   if sd is not None:
    print(' tensor names (first 25):',list(sd)[:25]);print(' tensor tail:',[(k,tuple(v.shape)) for k,v in list(sd.items())[-8:] if hasattr(v,'shape')])
    print(' checkpoint metadata:',json.dumps(checkpoint_metadata_summary(obj),default=str)[:5000])
    bn=[]
    for k in sd:
     hit=re.match(r'^branches\.([^.]+)\.', k)
     if hit and hit.group(1) not in bn:bn.append(hit.group(1))
    print(' detected branch names:',bn,'(count:',len(bn),')')
    print(' saved class_names:',obj.get('class_names') if isinstance(obj,dict) else None)
  else:print(' loadable model:',type(obj).__name__)
 except Exception as e:print(' ERROR:',e)

def state_dict(obj):
 if not isinstance(obj,dict):return None
 for key in ('model_state_dict','state_dict','fusion_model_state_dict','model_weights','net_state_dict','weights'):
  if key in obj and isinstance(obj[key],dict):return obj[key]
 if obj and all(hasattr(v,'shape') for v in obj.values()):return obj
 return None

def import_class(source, cls):
 if not source or not cls:raise RuntimeError('This checkpoint contains a state_dict, not a serialized model. Pass --model-source and --species-class/--panel-class using your original training architecture.')
 spec=importlib.util.spec_from_file_location('scmaps_original_training',source)
 if not spec or not spec.loader:raise RuntimeError('Cannot import original model source: '+source)
 module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 if not hasattr(module,cls):raise RuntimeError(f'Class {cls} not found in {source}')
 return getattr(module,cls)

# ----- Embedded SC-MAPS fusion architectures -----
# Mirrors the original Trillian attention-fusion training code and the
# checkpoint-matched concatenation inference in SC_MAPS_polymicrobial_ALL_IN_ONE_ARC.py.
# Inspect each checkpoint and STRICTLY match ALL tensor names and dimensions.

def canonical_channel(name):
 aliases = {'PHASE':'PHASECONTRAST','PHASECONTRAST':'PHASECONTRAST',
            'PHASECONTRASTIMAGE':'PHASECONTRAST','BF':'BF','BRIGHTFIELD':'BF',
            'DAPI':'DAPI','WGA':'WGA','WGAAF647':'WGA','WGA647':'WGA',
            'FM1':'FM1','FM143':'FM1'}
 raw=re.sub(r'[^A-Z0-9]', '',str(name).upper())
 return aliases.get(raw,raw)

CROP_INDEX={'PHASECONTRAST':0,'BF':1,'DAPI':2,'WGA':3,'FM1':4}

def ckpt_state(sd):
 for prefix in ('module.','model.','_orig_mod.'):
  if sd and all(str(k).startswith(prefix) for k in sd):
   sd={str(k)[len(prefix):]:v for k,v in sd.items()}
 return sd

def get_ckpt_branch_names(ckpt,sd):
 names=[]
 for key in sd:
  m=re.match(r'^branches\.([^.]+)\.',key)
  if m and m.group(1) not in names:names.append(m.group(1))
 if len(names) not in (4,5):
  raise RuntimeError(
   'Automatic DenseNet loader expects 4 or 5 branches but this selected checkpoint '
   f'contains {len(names)}: {names}. Check the displayed SELECTED checkpoint path. '
   'If a checkpoint in a 5-channel run has only 3 branch tensors, it is not the '
   'expected full 5-channel fusion model. Inspect checkpoint candidates and '
   'select the correct best_fusion_model.pt using --checkpoint-species-se PATH. '
   'No missing branches will be invented.')
 selected=None
 if isinstance(ckpt,dict):
  for key in ('selected_channels','selected_channel_names','channel_names','channels'):
   val=ckpt.get(key)
   if isinstance(val,(list,tuple)) and len(val)==len(names) and all(isinstance(x,str) for x in val):
    selected=list(val);break
 if not selected:
  preferred=['BF','DAPI','PHASECONTRAST','WGA','FM1'] if len(names)==5 else ['DAPI','PHASECONTRAST','WGA','FM1']
  if set(canonical_channel(n) for n in names)!=set(preferred):
   raise RuntimeError('Cannot determine branch/channel ordering from checkpoint: '+str(names))
  selected=preferred
 ordered=[]
 for ch in selected:
  matches=[n for n in names if canonical_channel(n)==canonical_channel(ch)]
  if len(matches)!=1:raise RuntimeError(f'Channel metadata {ch!r} cannot be matched uniquely to branch names {names}')
  ordered.append(matches[0])
 if len(set(ordered))!=len(names):raise RuntimeError('Duplicate branch metadata: '+str(selected))
 return ordered

def checkpoint_class_labels(ckpt,run_dir,n_classes,override=''):
 candidates=[]
 if isinstance(ckpt,dict):
  class_to_idx=ckpt.get('class_to_idx')
  if isinstance(class_to_idx,dict) and len(class_to_idx)==n_classes:
   try:
    pairs=sorted(((int(i),str(s)) for s,i in class_to_idx.items()))
    if [i for i,s in pairs]==list(range(n_classes)):candidates.append(('checkpoint.class_to_idx',[s for i,s in pairs]))
   except (TypeError,ValueError):pass
  for key in ('class_names','classes','species','panel_names','group_names','routing_classes'):
   v=ckpt.get(key)
   if isinstance(v,(tuple,list)) and len(v)==n_classes and all(isinstance(x,str) for x in v):
    candidates.append(('checkpoint.'+key,list(v)))
  conf=ckpt.get('run_config')
  if isinstance(conf,dict):
   for key in ('CLASS_NAMES','PANEL_NAMES','GROUP_NAMES'):
    v=conf.get(key)
    if isinstance(v,(list,tuple)) and len(v)==n_classes:candidates.append(('run_config.'+key,list(v)))
 for folder in (Path(run_dir),Path(run_dir).parent,Path(run_dir)/'metadata',Path(run_dir)/'config'):
  for f in ('class_names.json','classes.json','panel_names.json'):
   path=folder/f
   if path.is_file():
    try:
     v=json.loads(path.read_text())
     if isinstance(v,dict):v=[s for s,i in sorted(v.items(),key=lambda x:int(x[1]))]
     if isinstance(v,list) and len(v)==n_classes:candidates.append((str(path),list(map(str,v))))
    except Exception:pass
 if override:
  labels=[c.strip() for c in override.split(',')]
  if len(labels)!=n_classes:raise ValueError('Explicit --*-class-order has wrong length')
  # Never let a manual order silently contradict explicit checkpoint metadata.
  if candidates and labels!=candidates[0][1]:
   raise ValueError(f'Manual output neuron order {labels} conflicts with checkpoint metadata {candidates[0]}')
  if not candidates:candidates.append(('user override',labels))
 if not candidates:
  raise RuntimeError(f'Output-neuron labels are absent from checkpoint and run directory {run_dir}. '
                     f'Provide --{"species" if n_classes==13 else "panel"}-class-order with the EXACT training order; do not guess.')
 # Two metadata sources can use different descriptive names for the *same*
 # routing category. Compare canonically, but NEVER reorder neuron positions.
 def semantic_order(raw):
  converted=[]
  for item in raw:
   if n_classes==13:
    converted.append(str(item).upper().strip())
   else:
    n=canon_group(item)
    converted.append(n if n is not None else str(item).lower().strip())
  return converted
 if any(semantic_order(v) != semantic_order(candidates[0][1]) for k,v in candidates[1:]):
  raise RuntimeError('Conflicting saved class labels / neuron order: '+str(candidates))
 return candidates[0]

def translate_classes(raw_labels,name):
 if name.startswith('species'):
  labels=[str(s).upper().strip() for s in raw_labels]
  if len(labels)!=13 or set(labels)!=set(SPECIES):
   raise ValueError(f'Expected 13 species names, got: {raw_labels}')
  return labels
 labels=[]
 for label in raw_labels:
  g=canon_group(label)
  if g is None:
   n=re.sub(r'[^a-z0-9]','',str(label).lower())
   # Tolerate descriptive names from the original Panel-2 training script.
   if 'routine' in n and 'enterobacter' in n:g='Routine_Enterobacterales'
   elif ('ampc' in n or 'caution' in n) and 'enterobacter' in n:g='AmpC_Enterobacterales'
   elif n in ('staphaureuslike','saureuslikegroup','saureuslikegroup') or ('aureus' in n and 'like' in n):g='S_aureus_like'
   elif (n.startswith('cons') and (n=='cons' or 'staph' in n)) or ('coagulase' in n and 'staph' in n):g='CoNS'
   elif 'salmonella' in n:g='Salmonella'
   elif 'pseudomonas' in n:g='Pseudomonas'
   elif 'acinetobacter' in n:g='Acinetobacter'
   elif 'stenotrophomonas' in n:g='Stenotrophomonas'
  if g is None:raise ValueError(f'Cannot map saved routing-class name {label!r} to one of the eight SC-MAPS groups. Saved raw class order: {list(raw_labels)!r}. Add a verified semantic alias; do NOT treat neuron indices as alphabetical.')
  labels.append(g)
 if len(labels)!=8 or set(labels)!=set(GROUPS):
  raise ValueError('Class labels do not map uniquely to the 8 routing groups: '+str(labels))
 return labels

def checkpoint_metadata_summary(ckpt):
 if not isinstance(ckpt,dict):return {}
 useful=('class_names','class_to_idx','selected_channels','selected_channel_names','fusion_mode','se_reduction_ratio',
         'projection_dim','fusion_hidden_dim','fusion_dropout','normalization_by_channel','channel_to_index','input_size')
 d={}
 for k in useful:
  if k in ckpt:
   val=ckpt[k]
   if isinstance(val,(str,int,float,list,tuple,dict,type(None))):d[k]=val
 return d

def construct_checkpoint_fusion(sd,branch_names,n_classes,is_se,ratio,dropout):
 import torch
 import torch.nn as nn
 from torchvision.models import densenet121
 class SEBlock(nn.Module):
  def __init__(self,dim,red):
   super().__init__()
   self.excitation=nn.Sequential(nn.Linear(dim,max(dim//red,1)),nn.ReLU(inplace=True),
                                 nn.Linear(max(dim//red,1),dim),nn.Sigmoid())
  def forward(self,x):return x*self.excitation(x)
 class Fusion(nn.Module):
  def __init__(self):
   super().__init__()
   self.branch_names=list(branch_names)
   self.branches=nn.ModuleDict()
   self.projections=nn.ModuleDict()
   self.se_blocks=nn.ModuleDict() if is_se else None
   for channel in branch_names:
    b=densenet121(weights=None)
    old=b.features.conv0
    b.features.conv0=nn.Conv2d(1,old.out_channels,kernel_size=old.kernel_size,
                               stride=old.stride,padding=old.padding,bias=False)
    b.classifier=nn.Identity()
    self.branches[channel]=b
    proj_key=f'projections.{channel}.0.weight'
    if proj_key not in sd:raise ValueError(f'Projection key absent: {proj_key}')
    fout,fin=map(int,sd[proj_key].shape)
    if fin!=1024:raise ValueError(f'{channel} projection expects {fin} input features, not 1024 DenseNet121 features')
    self.projections[channel]=nn.Sequential(nn.Linear(fin,fout),nn.LayerNorm(fout),nn.GELU(),nn.Dropout(dropout))
    if is_se:self.se_blocks[channel]=SEBlock(fout,ratio)
   first=sd.get('classifier.0.weight'); last=sd.get('classifier.4.weight')
   if first is None or last is None:raise ValueError('Expected classifier.0.weight and classifier.4.weight')
   hidden,combined=map(int,first.shape)
   if combined!=sum(self.projections[c][0].out_features for c in branch_names):
    raise ValueError('Classifier feature dimension incompatible with branch projections')
   if int(last.shape[0])!=n_classes or int(last.shape[1])!=hidden:
    raise ValueError('Classifier output size incompatible with expected classes')
   self.classifier=nn.Sequential(nn.Linear(combined,hidden),nn.LayerNorm(hidden),nn.GELU(),
                                 nn.Dropout(dropout),nn.Linear(hidden,n_classes))
  def forward(self,x):
   if x.ndim!=4 or x.shape[1]!=len(self.branch_names):
    raise ValueError(f'Expected [B,{len(self.branch_names)},H,W], got {tuple(x.shape)}')
   feats=[]
   for i,name in enumerate(self.branch_names):
    f=self.projections[name](self.branches[name](x[:,i:i+1]))
    if self.se_blocks is not None:f=self.se_blocks[name](f)
    feats.append(f)
   return self.classifier(torch.cat(feats,dim=1))
 return Fusion()

def auto_build_model(name,sd,ckpt,checkpoint_path,args):
 import torch
 if not all(k.startswith(('branches.','projections.','classifier.','se_blocks.')) for k in sd):
  raise ValueError('State-dict contains non-fusion prefixes: '+str([k for k in sd if not k.startswith(('branches.','projections.','classifier.','se_blocks.'))][:8]))
 branches=get_ckpt_branch_names(ckpt,sd)
 is_se=any(k.startswith('se_blocks.') for k in sd)
 expected_se=name.endswith('_se')
 if is_se!=expected_se:
  raise RuntimeError(f'{name}: checkpoint fusion architecture inferred as {"SE" if is_se else "concat"}, which conflicts with the requested run')
 n_classes=13 if name.startswith('species') else 8
 override=(args.species_class_order if n_classes==13 else args.panel_class_order)
 src,raw_labels=checkpoint_class_labels(ckpt,Path(checkpoint_path).parent,n_classes,override=override)
 labels=translate_classes(raw_labels,name)
 dropout=float(ckpt.get('fusion_dropout',ckpt.get('dropout',0.30))) if isinstance(ckpt,dict) else 0.30
 # The SE reduction dimension must match the actual saved weights.
 reduction=4
 if is_se:
  pname=branches[0]
  first=f'se_blocks.{pname}.excitation.0.weight'
  last=f'se_blocks.{pname}.excitation.2.weight'
  if first not in sd or last not in sd:
   raise RuntimeError('SE checkpoint uses a different block layout than original training code; inspect key names')
  hidden,out=map(int,sd[first].shape)
  if int(sd[last].shape[0])!=out or int(sd[last].shape[1])!=hidden:
   raise RuntimeError('SE weights inconsistent')
  if isinstance(ckpt,dict) and ckpt.get('se_reduction_ratio'):
   reduction=int(ckpt['se_reduction_ratio'])
   if max(out//reduction,1)!=hidden:raise RuntimeError('Checkpoint se_reduction_ratio metadata inconsistent with weights')
  else:
   reduction=max(1,out//hidden)
 model=construct_checkpoint_fusion(sd,branches,n_classes,is_se,reduction,dropout)
 model.load_state_dict(sd,strict=True)
 model.eval()
 indices=[CROP_INDEX[canonical_channel(ch)] for ch in branches]
 model.scmaps_indices=indices
 model.scmaps_class_order=labels
 model.scmaps_class_source=src
 model.scmaps_branch_names=branches
 model.scmaps_fusion='se_attention' if is_se else 'concat'
 model.scmaps_checkpoint_metadata=checkpoint_metadata_summary(ckpt)
 print(f'{name}: reconstructed {model.scmaps_fusion} DenseNet121 from checkpoint with STRICT tensor loading',flush=True)
 print(f'{name}: branches={branches} | crop_indices={indices} | class_order={labels} ({src})',flush=True)
 return model

def load_model(name, checkpoint, args):
 import torch
 obj=torch_load(checkpoint)
 if isinstance(obj,torch.nn.Module):return obj.eval()
 if isinstance(obj,dict):
  for k in ('model','network','net'):
   if isinstance(obj.get(k),torch.nn.Module):return obj[k].eval()
 sd=state_dict(obj)
 if sd is None:raise RuntimeError(f'Unknown checkpoint structure: {type(obj)}')
 sd=ckpt_state(sd)
 auto_error=None
 try:return auto_build_model(name,sd,obj,checkpoint,args)
 except Exception as e:auto_error=e
 classname=getattr(args,'model_class_'+name) or (args.species_class if name.startswith('species') else args.panel_class)
 source=getattr(args,'model_source_'+name) or args.model_source
 if not classname or not source:
  raise RuntimeError(f'{name}: automatic reconstruction from state_dict failed: {auto_error}\n'
                     'To diagnose, inspect checkpoint tensor names and saved class metadata using --inspect; '
                     'a non-standard architecture additionally requires --model-source and --species-class/--panel-class. '
                     'This script NEVER uses strict=False as a workaround.') from auto_error
 cls=import_class(source,classname)
 fusion='se_attention' if name.endswith('se') else 'concat'
 n=13 if name.startswith('species') else 8
 kwargs_str=getattr(args,'model_kwargs_'+name) or args.model_kwargs
 kwargs=json.loads(kwargs_str.replace('{n_classes}',str(n)).replace('{fusion}',fusion))
 try:model=cls(**kwargs)
 except Exception as e:raise RuntimeError(f'Constructor failed for {cls.__name__} with {kwargs}. Details: {e}')
 model.load_state_dict(sd,strict=True)
 model.eval()
 # The original model may not expose checkpoint channel metadata; user must verify manually.
 return model

def locate_data(args):
 """Read Plate-1 CSV; intelligently resolve common manifest schemas without guessing labels."""
 if args.manifest:
  csv=Path(args.manifest).expanduser()
  if not csv.is_file():
   raise FileNotFoundError(f'Holdout manifest not found: {csv}. Expected existing scratch dataset; check it is mounted.')
  m=pd.read_csv(csv,low_memory=False)
  print(f'Holdout manifest: {csv} | rows={len(m):,} | columns={list(m.columns)}',flush=True)
  if args.split_col or args.holdout_value:
   if not args.split_col or not args.holdout_value:raise ValueError('Supply BOTH --split-col and --holdout-value for holdout filtering')
   if args.split_col not in m:raise ValueError('Unknown split column: '+args.split_col)
   before=len(m);m=m[m[args.split_col].astype(str).str.strip().str.casefold()==args.holdout_value.strip().casefold()].copy()
   print(f'Holdout filter: {before:,} -> {len(m):,} rows for {args.split_col}={args.holdout_value}',flush=True)
  def pick(requested, alternatives, purpose):
   if requested in m:return requested
   for alt in alternatives:
    matches=[c for c in m if str(c).lower()==alt.lower()]
    if len(matches)==1:
     print(f'Inferred {purpose} column: {matches[0]!r} (requested {requested!r})',flush=True)
     return matches[0]
   raise ValueError(f'Could not identify {purpose} column in {csv}. Found: {list(m.columns)}. Pass --{purpose.replace(" ","-")}-col explicitly if necessary.')
  fcol=pick(args.file_col,['npy_path','file_path','filepath','path','relative_path','rel_path','crop_path','cell_path','npy_file','filename','file','image_path'], 'file')
  scol=pick(args.species_col,['true_species','species','species_code','species_name','organism','target_species','class_name','label'], 'species')
  sample_options=['sample_id','sample','sample_key','sample_name','replicate_id','replicate','biological_sample','sample_group']
  try:
   groupcol=pick(args.sample_col,sample_options, 'sample')
  except ValueError as e:
   raise ValueError(str(e)+'\nA valid sample grouping is required for aggregation/cluster bootstrap. Do not infer 120 samples from row order. If the CSV has no sample identifiers, provide a separate manifest with sample_id.') from e
  m=m[[fcol,scol,groupcol]].rename(columns={fcol:'npy_path',scol:'true_species',groupcol:'sample_id'})
  base=Path(args.data_root or csv.parent.parent).expanduser()
  # Original CSV was produced on Windows. On Linux, Path("D:\\Anuj\\...") is
  # NOT considered absolute; simply prepending `base` leads to 221,726 false
  # missing files. Normalise Windows separators/drives and retain the suffix
  # after the dataset directory instead.
  def resolve_path(value):
   raw=str(value).strip().strip('"').strip("'")
   if not raw or raw.lower() in ('nan','none'):
    return raw
   native=Path(raw)
   if native.is_file():
    return str(native.resolve())
   normed=raw.replace('\\','/')
   parts=[part for part in normed.split('/') if part and part != '.']
   folded=[part.casefold() for part in parts]
   dataset='trillian_dataset_with_holdout'
   anchor='holdout_plate1'
   candidates=[]
   if dataset in folded:
    tail=Path(*parts[folded.index(dataset)+1:])
    if tail.parts:
     candidates.append(base/tail)
   if anchor in folded:
    tail=Path(*parts[folded.index(anchor):])
    candidates.append(base/tail)
   # Some exports contain only a path relative to the dataset root.
   if not re.match(r'^[a-zA-Z]:/',normed) and not normed.startswith('/'):
    candidates.append(base/Path(normed))
   # Legacy manifests sometimes store just crops/<filename> or a basename.
   filename=parts[-1] if parts else raw
   candidates.extend([base/'holdout_plate1'/'crops'/filename,
                      base/'crops'/filename,
                      base/filename])
   # Keep only unique candidate paths, preserving their priority.
   seen=set();unique=[]
   for candidate in candidates:
    key=str(candidate)
    if key not in seen:
     unique.append(candidate);seen.add(key)
   for candidate in unique:
    if candidate.is_file():
     return str(candidate.resolve())
   return str(unique[0]) if unique else raw
  original_example=m.npy_path.iloc[0] if len(m) else None
  m['npy_path']=m.npy_path.map(resolve_path)
  if original_example is not None:
   print('Path remapping example:',repr(original_example),'->',repr(m.npy_path.iloc[0]),flush=True)
 else:
  root=Path(args.data_root or args.scratch_root)
  if not root.exists():raise FileNotFoundError(root)
  print('Searching .npy single-cell files under',root,flush=True)
  files=list(root.rglob('*.npy'))
  if not files:raise ValueError('No .npy cells found')
  pat=re.compile(args.sample_regex) if args.sample_regex else None
  rows=[]
  for f in files:
   matches=[s for s in SPECIES if re.search(r'(?<![A-Za-z])'+s+r'(?![A-Za-z])',str(f),re.I)]
   if len(matches)!=1:continue
   mt=pat.search(str(f)) if pat else None
   sample=mt.group('sample') if mt and 'sample' in mt.groupdict() else None
   if sample:rows.append((str(f),matches[0],sample))
  if not rows:raise ValueError('Could not infer reliably labelled cells AND sample IDs from paths; supply --manifest with npy_path,true_species,sample_id or --sample-regex.')
  m=pd.DataFrame(rows,columns=['npy_path','true_species','sample_id'])
 if m.empty:raise ValueError('Holdout manifest contains no records')
 if m[['npy_path','true_species','sample_id']].isna().any().any():raise ValueError('Null paths, species or sample IDs')
 m['true_species']=m.true_species.astype(str).str.upper().str.strip()
 invalid=sorted(set(m.true_species)-set(SPECIES))
 if invalid:raise ValueError(f'Unexpected labels in 13-species holdout manifest: {invalid[:20]} (resolve codes rather than silently dropping cells).')
 m['sample_id']=m.sample_id.astype(str).str.strip()
 if m.sample_id.eq('').any():raise ValueError('Blank sample IDs')
 if m.npy_path.duplicated().any():raise ValueError('Duplicate paths in manifest')
 available=m.npy_path.map(lambda p:Path(p).is_file())
 if not available.all():
  bad=m.loc[~available,'npy_path']
  raise FileNotFoundError(f'{len(bad):,} / {len(m):,} crop files are missing. First missing paths: {bad.head(5).tolist()}. Check --data-root and manifest path conventions.')
 if m.groupby('sample_id').true_species.nunique().max()>1:raise ValueError('Sample IDs contain multiple species; use a sample ID that uniquely identifies a pure species population')
 if args.max_cells and len(m)>args.max_cells:
  # Reproducible species-stratified *subset* for debugging; not publication analysis.
  rng=np.random.default_rng(args.seed)
  keep=[]
  for _, g in m.groupby('true_species'):
   quota=max(1,round(args.max_cells*len(g)/len(m)))
   keep.extend(rng.choice(g.index,size=min(quota,len(g)),replace=False).tolist())
  m=m.loc[keep].copy()
  print('WARNING: --max-cells subsampled the holdout set; metrics are exploratory only.',flush=True)
 m=m.sort_values('npy_path').reset_index(drop=True)
 m.insert(0,'cell_id',m.npy_path)
 print(f'Resolved manifest: {len(m):,} matched crops, {m.true_species.nunique()} species, {m.sample_id.nunique()} sample IDs',flush=True)
 return m

def build_dataset(m, args, name, model=None):
 import torch
 from torch.utils.data import Dataset
 override=getattr(args,'channel_indices_'+name)
 indices=list(model.scmaps_indices) if model is not None and hasattr(model,'scmaps_indices') else [int(s) for s in (override or args.channel_indices).split(',')]
 if not (1<=len(indices)<=5) or any(i not in range(5) for i in indices) or len(set(indices))!=len(indices):
  raise ValueError(f'{name}: invalid channel indices {indices}; specify unique indices from 0,1,2,3,4')
 class Cells(Dataset):
  def __len__(self):return len(m)
  def __getitem__(self,i):
   p=m.iloc[i].npy_path;x=np.load(p,allow_pickle=False)
   if x.ndim!=3:raise ValueError(f'Expected 3D single cell stack, got {x.shape}: {p}')
   if x.shape[0]>=5 and x.shape[1]==96:x=x[indices]
   elif x.shape[-1]>=5 and x.shape[0]==96:x=np.moveaxis(x,-1,0)[indices]
   else:raise ValueError(f'Unexpected stack shape {x.shape}: {p}')
   x=np.nan_to_num(x.astype('float32'),nan=0,posinf=0,neginf=0)
   if args.normalisation=='per_channel_crop_zscore':
    x=(x-x.mean(axis=(1,2),keepdims=True))/np.maximum(x.std(axis=(1,2),keepdims=True),1e-6)
   return torch.from_numpy(np.ascontiguousarray(x)),i
 return Cells()

def forward(model, x, layout):
 import torch
 if hasattr(model,'scmaps_indices') or layout=='stack':out=model(x)
 else:
  branches=[x[:,i:i+1] for i in range(x.shape[1])]
  try:out=model(*branches)
  except TypeError:out=model(branches)
 if isinstance(out,dict):
  for key in ('logits','output','outputs','predictions'):
   if key in out:out=out[key];break
 if isinstance(out,(list,tuple)):out=out[0]
 if not isinstance(out,torch.Tensor) or out.ndim!=2:raise RuntimeError(f'Model forward should yield [batch,classes] logits; got {type(out)} / {getattr(out,"shape",None)}')
 return out

def balanced_feature_selection(m, maximum, seed):
 """Select exactly the same stratified cells for each model's feature analysis."""
 if maximum < 2:
  return set()
 rng=np.random.default_rng(seed)
 groups=[np.asarray(list(g.index),dtype=int) for _,g in m.groupby('true_species',sort=True)]
 quota=max(1,maximum//max(len(groups),1))
 picked=[]
 for inds in groups:
  picked.extend(rng.choice(inds,size=min(quota,len(inds)),replace=False).tolist())
 leftover=max(0,maximum-len(picked))
 if leftover:
  others=np.setdiff1d(np.arange(len(m)),np.asarray(picked,dtype=int),assume_unique=False)
  if len(others):picked.extend(rng.choice(others,size=min(leftover,len(others)),replace=False).tolist())
 return set(picked)


def choose_feature_module(model, name, args):
 import torch
 if args.no_features:return None,None
 mods=dict(model.named_modules())
 if args.feature_layer!='auto':
  if args.feature_layer not in mods:raise ValueError(f'{name}: --feature-layer {args.feature_layer!r} missing. Last modules: {list(mods)[-40:]}')
  return args.feature_layer,mods[args.feature_layer]
 expected=13 if name.startswith('species') else 8
 candidates=[(k,v) for k,v in mods.items() if isinstance(v,torch.nn.Linear) and v.out_features==expected]
 if not candidates:
  print(f'WARNING: {name} has no identifiable final Linear({expected}) layer. Penultimate features skipped for this model; set --feature-layer explicitly.',flush=True)
  return None,None
 return candidates[-1]


def infer(name,ckpt,m,args,out,feature_indices):
 import torch
 from torch.utils.data import DataLoader
 model=load_model(name,ckpt,args)
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model.to(device)
 loader=DataLoader(build_dataset(m,args,name,model),batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=device.type=='cuda')
 allp=[];feature_blocks=[];feature_ids=[];hook=None;feature_capture=[]
 layer_name,module=choose_feature_module(model,name,args)
 if module is not None and feature_indices:
  def fn(module,ins,outs):
   v=ins[0] if ins else outs
   if isinstance(v,(tuple,list)):v=v[0]
   if isinstance(v,torch.Tensor):feature_capture.append(v.detach().flatten(1).cpu().numpy())
  hook=module.register_forward_hook(fn)
  print(f'{name}: extracting penultimate features from {layer_name} for <= {len(feature_indices)} balanced cells',flush=True)
 try:
  with torch.inference_mode():
   for batch,(x,ix) in enumerate(loader):
    logits=forward(model,x.to(device,non_blocking=True),getattr(args,'input_layout_'+name) or args.input_layout)
    expected=13 if name.startswith('species') else 8
    if logits.shape[1]!=expected:raise ValueError(f'{name} outputs {logits.shape[1]} classes; expected {expected}')
    p=logits.softmax(1).cpu().numpy()
    if not np.isfinite(p).all():raise ValueError('NaNs from model')
    allp.append(p)
    if hook is not None:
     if len(feature_capture)!=1:raise RuntimeError(f'{name}: layer {layer_name} executed {len(feature_capture)} times per batch; choose a unique final classifier module using --feature-layer')
     idx=ix.numpy();wanted=np.fromiter((int(j) in feature_indices for j in idx),dtype=bool,count=len(idx))
     if wanted.any():
      feature_blocks.append(feature_capture[0][wanted]);feature_ids.extend(m.iloc[idx[wanted]].cell_id.tolist())
     feature_capture.clear()
    if batch%250==0:print(f'{name}: {min((batch+1)*args.batch_size,len(m)):,}/{len(m):,} cells',flush=True)
 finally:
  if hook:hook.remove()
 probs=np.concatenate(allp)
 labels=(model.scmaps_class_order if hasattr(model,'scmaps_class_order') else [t.strip() for t in (args.species_class_order if name.startswith('species') else args.panel_class_order).split(',')])
 required=SPECIES if name.startswith('species') else GROUPS
 if set(labels)!=set(required) or len(labels)!=len(required):raise ValueError(f'Output class order for {name} is invalid: {labels}. Must supply true training class order!')
 d=m[['cell_id','true_species','sample_id']].copy()
 for label in required:d[f'prob_{label}']=probs[:,labels.index(label)]
 d.to_csv(out/f'{name}_predictions.csv',index=False)
 if hasattr(model,'scmaps_class_order'):
  (out/f'{name}_resolved_checkpoint.json').write_text(json.dumps({'checkpoint':str(ckpt),'channels':model.scmaps_branch_names,'indices':model.scmaps_indices,'class_order':model.scmaps_class_order,'class_source':model.scmaps_class_source,'fusion':model.scmaps_fusion,'checkpoint_metadata':model.scmaps_checkpoint_metadata},indent=2))
 if feature_blocks:
  feats=np.concatenate(feature_blocks).astype('float32')
  np.savez_compressed(out/f'{name}_features.npz',features=feats,cell_id=np.asarray(feature_ids,dtype=str))
 print(f'{name}: saved {len(probs):,} predictions and {len(feature_ids):,} feature vectors',flush=True)
 del model
 return out/f'{name}_predictions.csv'

GROUP_TO_SPECIES = {
    'Routine_Enterobacterales':['ECOL','KLPN'],
    'AmpC_Enterobacterales':['KLAE','ENTC','SERM'],
    'Salmonella':['SAENA'], 'Pseudomonas':['PSAR'],
    'Acinetobacter':['ACAA'], 'Stenotrophomonas':['PSMA'],
    'S_aureus_like':['SAUR','SLUG'], 'CoNS':['SEPI','STCP'],
}
SP_TO_GROUP={s:g for g,ls in GROUP_TO_SPECIES.items() for s in ls}
ALIASES={
 'Routine_Enterobacterales':['routine enterobacterales','routine_ent','routine'],
 'AmpC_Enterobacterales':['ampc/caution enterobacterales','ampc caution enterobacterales','ampc','ampc_enterobacterales'],
 'Salmonella':['salmonella'], 'Pseudomonas':['pseudomonas'],
 'Acinetobacter':['acinetobacter'], 'Stenotrophomonas':['stenotrophomonas'],
 'S_aureus_like':['s aureus like','s. aureus-like','s_aureus_like','saureuslike'],
 'CoNS':['cons','cons staphylococci','cons staphylococcus','coagulase negative staphylococci','coagulase-negative staphylococci']}

def norm(s): return re.sub(r'[^a-z0-9]', '', str(s).lower())
def canon_group(s):
    for g in GROUPS:
        if norm(s)==norm(g) or any(norm(s)==norm(a) for a in ALIASES[g]):return g
    return None

def args_parser():
 p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 p.add_argument('--species-csv',required=True,help='13-species prediction CSV (one row per cell)')
 p.add_argument('--direct-csv',required=True,help='direct 8-routing prediction CSV')
 p.add_argument('--out',default='routing_comparison')
 p.add_argument('--key-col',default='cell_id',help='unique cell key present in BOTH CSVs')
 p.add_argument('--truth-col',default='true_species',help='column in species CSV with ground-truth species codes')
 p.add_argument('--sample-col',default='sample_id',help='biological sample identifier; used for aggregation and clustered bootstrap')
 p.add_argument('--species-prefix',default='',help='optional prefix on probability columns e.g. prob_')
 p.add_argument('--direct-prefix',default='',help='optional prefix on probability columns e.g. prob_')
 p.add_argument('--species-features',help='optional species-model .npz holding features and cell IDs')
 p.add_argument('--direct-features',help='optional direct-model .npz holding features and cell IDs')
 p.add_argument('--feature-key',default='features',help='array key in both feature npz files')
 p.add_argument('--feature-id-key',default='cell_id',help='aligned cell ID key in feature npz files')
 p.add_argument('--max-umap',type=int,default=12000)
 p.add_argument('--repeats',type=int,default=200)
 p.add_argument('--bootstraps',type=int,default=2000)
 p.add_argument('--seed',type=int,default=42)
 p.add_argument('--min-cells',type=int,default=200,help='minimum sample cells to include in aggregation up to 200')
 p.add_argument('--sizes',default='1,2,3,5,10,20,30,50,75,100,150,200')
 return p.parse_args()

def probs(df, labels, prefix, is_group):
 out=[]; mapping={}
 for c in df.columns:
  if is_group:
   stem=str(c)
   if prefix and not stem.startswith(prefix):continue
   if prefix:stem=stem[len(prefix):]
   else:stem=re.sub(r'^(probability_|prob_|p_|score_|softmax_)','',stem,flags=re.I)
   g=canon_group(stem)
   if g: mapping.setdefault(g,[]).append(c)
  else:
   stem=str(c)
   if prefix and not stem.startswith(prefix):continue
   if prefix:stem=stem[len(prefix):]
   else:stem=re.sub(r'^(probability_|prob_|p_|score_|softmax_)','',stem,flags=re.I)
   for s in labels:
    if norm(stem)==norm(s):mapping.setdefault(s,[]).append(c)
 for lab in labels:
  cols=mapping.get(lab,[])
  if len(cols)!=1:raise ValueError(f'Expected exactly one probability column for {lab}, found {cols}. Available: {list(df.columns)[:80]}. Use --species-prefix or --direct-prefix.')
  out.append(cols[0])
 a=df[out].apply(pd.to_numeric,errors='raise').to_numpy(dtype=np.float64)
 if not np.isfinite(a).all() or (a < -1e-7).any():raise ValueError('Non-finite or negative probabilities')
 sums=a.sum(axis=1)
 if np.any(sums<=0):raise ValueError('Rows with zero probability sum')
 if np.quantile(np.abs(sums-1),0.99)>0.02:raise ValueError('Probability rows not normalized. Check columns are probabilities rather than logits or other features.')
 return np.clip(a,0,None)/sums[:,None],dict(zip(labels,out))

def ece(y,p,bins=15):
 conf=p.max(1);correct=(p.argmax(1)==y).astype(float)
 edges=np.linspace(0,1,bins+1);res=0.
 for j in range(bins):
  m=(conf>=edges[j]) & ((conf<edges[j+1]) if j<bins-1 else (conf<=edges[j+1]))
  if m.any():res+=m.mean()*abs(correct[m].mean()-conf[m].mean())
 return float(res)

def metrics(y,p):
 pred=p.argmax(1);eps=1e-12
 return {'n_cells':len(y),'accuracy':accuracy_score(y,pred),
 'balanced_accuracy':balanced_accuracy_score(y,pred),
 'macro_f1':f1_score(y,pred,labels=np.arange(8),average='macro',zero_division=0),
 'weighted_f1':f1_score(y,pred,labels=np.arange(8),average='weighted',zero_division=0),
 'nll':-np.log(np.maximum(p[np.arange(len(y)),y],eps)).mean(),
 'brier':np.mean(np.sum((p-np.eye(8)[y])**2,axis=1)), 'ece_15':ece(y,p)}

def save_cm(y,p,path,title):
 cm=confusion_matrix(y,p.argmax(1),labels=np.arange(8),normalize='true')
 fig,ax=plt.subplots(figsize=(10,8));im=ax.imshow(cm,cmap='Blues',vmin=0,vmax=1)
 ax.set_xticks(range(8),GROUPS,rotation=45,ha='right',fontsize=9)
 ax.set_yticks(range(8),GROUPS,fontsize=9)
 for i in range(8):
  for j in range(8):ax.text(j,i,f'{cm[i,j]:.0%}',ha='center',va='center',fontsize=8,color='white' if cm[i,j]>.52 else 'black')
 ax.set_xlabel('Predicted routing group');ax.set_ylabel('True routing group');ax.set_title(title)
 fig.colorbar(im,ax=ax,fraction=.046,pad=.04);fig.tight_layout();fig.savefig(path,dpi=190);plt.close(fig)

def feature_analysis(a,b,ids,ysp,ypanel,out,args,rng):
 try:import umap
 except ImportError:
  print('UMAP skipped: install umap-learn in ARC environment');return
 results=[]
 def load(path):
  z=np.load(path,allow_pickle=False)
  print(f'{path}: npz keys={z.files}')
  if args.feature_key not in z or args.feature_id_key not in z:raise KeyError('Feature NPZ requires --feature-key and --feature-id-key arrays')
  v=z[args.feature_key];k=np.asarray(z[args.feature_id_key]).astype(str)
  if len(v)!=len(k) or len(set(k))!=len(k):raise ValueError('Feature IDs must be unique and aligned')
  return pd.DataFrame({args.key_col:k,'index':np.arange(len(k))}),v
 ia,fa=load(a);ib,fb=load(b)
 lookup_a=dict(zip(ia[args.key_col],ia['index']));lookup_b=dict(zip(ib[args.key_col],ib['index']))
 eligible=np.array([i for i,k in enumerate(ids) if k in lookup_a and k in lookup_b])
 if not len(eligible):raise ValueError('No overlapping feature cell IDs')
 # Balanced subsampling by true species avoids a figure dominated by large classes.
 chosen=[];per=max(1,args.max_umap//len(SPECIES))
 for si in range(len(SPECIES)):
  pool=eligible[ysp[eligible]==si]
  chosen.extend(rng.choice(pool,size=min(len(pool),per),replace=False))
 chosen=np.asarray(sorted(chosen),dtype=int)
 print(f'Independent-model features: matched {len(eligible)} IDs, plotting {len(chosen)} cells')
 for label,f,look in [('species_model',fa,lookup_a),('direct_model',fb,lookup_b)]:
  x=np.asarray(f[[look[ids[i]] for i in chosen]],dtype=np.float32)
  x=np.nan_to_num(x,nan=0,posinf=0,neginf=0)
  x=PCA(n_components=min(50,x.shape[1],len(x)-1),random_state=args.seed).fit_transform(x)
  emb=umap.UMAP(n_neighbors=30,min_dist=.2,random_state=args.seed).fit_transform(x)
  for colorlabels,names,suffix in [(ysp[chosen],SPECIES,'species'),(ypanel[chosen],GROUPS,'routing')]:
   fig,ax=plt.subplots(figsize=(11,9))
   cmap=plt.get_cmap('tab20',len(names))
   for j,name in enumerate(names):
    m=colorlabels==j
    if m.any():ax.scatter(emb[m,0],emb[m,1],s=3,alpha=.5,label=name,color=cmap(j),rasterized=True)
   ax.set_title(f'{label}: independent UMAP, coloured by {suffix}')
   ax.set_xlabel('UMAP 1');ax.set_ylabel('UMAP 2');ax.legend(markerscale=3,fontsize=8,bbox_to_anchor=(1.02,1),loc='upper left')
   fig.tight_layout();fig.savefig(out/f'{label}_umap_{suffix}.png',dpi=190);plt.close(fig)
  # In original PCA-reduced features; kNN label agreement captures local structure.
  nn=NearestNeighbors(n_neighbors=min(16,len(x))).fit(x).kneighbors(x,return_distance=False)[:,1:]
  for vals,names,suffix in [(ysp[chosen],SPECIES,'species'),(ypanel[chosen],GROUPS,'routing')]:
   agree=(vals[nn]==vals[:,None]).mean()
   sil=np.nan
   if len(np.unique(vals))>1 and len(vals)>len(np.unique(vals)):
    sub=rng.choice(len(vals),size=min(len(vals),5000),replace=False)
    sil=float(silhouette_score(x[sub],vals[sub])) if len(np.unique(vals[sub]))>1 else np.nan
   results.append({'model':label,'labels':suffix,'n':len(vals),'knn15_agreement':agree,'silhouette_pca50':sil})
 pd.DataFrame(results).to_csv(out/'feature_space_metrics.csv',index=False)

def run_comparison(species_csv, direct_csv, output_dir, species_features, direct_features, run_args):
 """Evaluate one matched 13-class/8-class pair with one set of CLI settings."""
 args=SimpleNamespace(
  species_csv=str(species_csv),direct_csv=str(direct_csv),out=str(output_dir),
  species_features=str(species_features) if species_features and Path(species_features).exists() else None,
  direct_features=str(direct_features) if direct_features and Path(direct_features).exists() else None,
  key_col='cell_id',truth_col='true_species',sample_col='sample_id',
  species_prefix='prob_',direct_prefix='prob_',feature_key='features',feature_id_key='cell_id',
  max_umap=run_args.max_umap_cells,repeats=run_args.repeats,bootstraps=run_args.bootstraps,
  seed=run_args.seed,min_cells=run_args.min_cells,sizes=run_args.sizes)

 rng=np.random.default_rng(args.seed);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 a=pd.read_csv(args.species_csv,low_memory=False);b=pd.read_csv(args.direct_csv,low_memory=False)
 print('Species CSV columns:',list(a.columns));print('Direct CSV columns:',list(b.columns))
 for name,df in [('species',a),('direct',b)]:
  if args.key_col not in df:raise ValueError(f'{name} missing unique key {args.key_col}. Create a stable per-cell key in both prediction files before comparing; do not merge by row order.')
  if df[args.key_col].isna().any() or df[args.key_col].duplicated().any():raise ValueError(f'{name} has null or duplicate cell IDs')
 if args.truth_col not in a:raise ValueError(f'{args.truth_col} missing from species CSV')
 sa,scols=probs(a,SPECIES,args.species_prefix,False)
 db,dcols=probs(b,GROUPS,args.direct_prefix,True)
 species_indices=np.array([GROUPS.index(SP_TO_GROUP[s]) for s in SPECIES]); pooled=np.zeros((len(a),8))
 for j in range(8):pooled[:,j]=sa[:,species_indices==j].sum(axis=1)
 a=a[[args.key_col,args.truth_col]+([args.sample_col] if args.sample_col in a else [])].copy()
 for j in range(8):a[f's_{j}']=pooled[:,j]
 b=b[[args.key_col]+([args.truth_col] if args.truth_col in b else [])+([args.sample_col] if args.sample_col in b else [])].copy()
 for j in range(8):b[f'd_{j}']=db[:,j]
 matched=a.merge(b,on=args.key_col,how='inner',validate='one_to_one',suffixes=('_species','_direct'))
 print(f'Rows: species={len(a)}, direct={len(b)}, matched={len(matched)}')
 if len(matched)!=len(a) or len(matched)!=len(b):raise ValueError('The two model outputs are not evaluated on exactly the same cells. Check the saved manifests; refusing an unmatched comparison.')
 if not len(matched):raise ValueError('No matching cell IDs')
 species_truth=matched[args.truth_col if args.truth_col in matched else args.truth_col+'_species'].astype(str).str.upper().str.strip().to_numpy()
 invalid=set(species_truth)-set(SPECIES)
 if invalid:raise ValueError(f'Unknown true species labels: {invalid}')
 ysp=np.array([SPECIES.index(s) for s in species_truth]);y=np.array([GROUPS.index(SP_TO_GROUP[s]) for s in species_truth]);
 ps=matched[[f's_{j}' for j in range(8)]].to_numpy();pdirect=matched[[f'd_{j}' for j in range(8)]].to_numpy()
 if args.truth_col+'_direct' in matched:
  other=matched[args.truth_col+'_direct'].astype(str).str.upper().str.strip().to_numpy()
  if (other!=species_truth).any():raise ValueError('Ground-truth species disagree across matched CSVs')
 samplecol=args.sample_col+'_species' if args.sample_col+'_species' in matched else args.sample_col
 if samplecol in matched:
  sample=matched[samplecol].astype(str).to_numpy()
  if args.sample_col+'_direct' in matched and np.any(matched[args.sample_col+'_direct'].astype(str).to_numpy()!=sample):
   raise ValueError('Sample IDs differ across matched CSVs')
 else:
  sample=None;print('WARNING: no sample_id found. Aggregation and sample bootstrap SKIPPED; use --sample-col.')
 ids=matched[args.key_col].astype(str).to_numpy()
 if sample is not None:
  cross=pd.DataFrame({'sample':sample,'truth':species_truth}).groupby('sample')['truth'].nunique()
  if cross.max()>1:raise ValueError('Sample ID groups contain multiple true species. Use biological sample IDs for pure populations; do not aggregate mixed species.')
 perf=pd.DataFrame([dict(model='species_13_pooled',**metrics(y,ps)),dict(model='direct_8',**metrics(y,pdirect))]);perf.to_csv(out/'overall_metrics.csv',index=False)
 print(perf.to_string(index=False))
 save_cm(y,ps,out/'confusion_species_pooled.png','13-species model: pooled AST-routing probabilities')
 save_cm(y,pdirect,out/'confusion_direct.png','Direct 8-group model')
 sgood=ps.argmax(1)==y;dgood=pdirect.argmax(1)==y
 categories=np.select([sgood&dgood,sgood&~dgood,~sgood&dgood],['both_correct','species_only_correct','direct_only_correct'],default='both_wrong')
 detail=pd.DataFrame({'cell_id':ids,'true_species':species_truth,'true_group':[GROUPS[i] for i in y],
 'species_model_pred':[GROUPS[i] for i in ps.argmax(1)],'direct_model_pred':[GROUPS[i] for i in pdirect.argmax(1)],
 'species_model_true_prob':ps[np.arange(len(y)),y], 'direct_model_true_prob':pdirect[np.arange(len(y)),y],
 'disagreement_category':categories})
 if sample is not None:detail['sample_id']=sample
 detail.to_csv(out/'paired_cell_predictions.csv',index=False)
 breakdown=[]
 for s in SPECIES:
  mask=species_truth==s
  breakdown.append({'species':s,'routing_group':SP_TO_GROUP[s],'n_cells':int(mask.sum()),
  'species_pooled_recall':sgood[mask].mean() if mask.any() else np.nan,
  'direct_recall':dgood[mask].mean() if mask.any() else np.nan,
  'advantage_pp':100*(sgood[mask].mean()-dgood[mask].mean()) if mask.any() else np.nan,
  'species_only_correct':int((sgood[mask]&~dgood[mask]).sum()),
  'direct_only_correct':int((~sgood[mask]&dgood[mask]).sum())})
 bysp=pd.DataFrame(breakdown);bysp.to_csv(out/'by_species.csv',index=False)
 bygroup=detail.groupby('true_group',sort=False).agg(n=('cell_id','size'),species_only=('disagreement_category',lambda x:(x=='species_only_correct').sum()),direct_only=('disagreement_category',lambda x:(x=='direct_only_correct').sum())).reset_index()
 bygroup['net_species_advantage_pp']=100*(bygroup.species_only-bygroup.direct_only)/bygroup.n
 bygroup.to_csv(out/'by_routing_group.csv',index=False)
 pd.Series(categories).value_counts().rename_axis('category').reset_index(name='n_cells').to_csv(out/'disagreement_totals.csv',index=False)
 fig,ax=plt.subplots(figsize=(9,6));tmp=bysp.sort_values('advantage_pp');ax.barh(tmp.species,tmp.advantage_pp,color=['#2866a3' if v>=0 else '#b75858' for v in tmp.advantage_pp]);ax.axvline(0,color='black',lw=.8)
 ax.set_xlabel('Species-first minus direct routing recall (percentage points)');ax.set_title('Paired holdout routing performance, stratified by true species');fig.tight_layout();fig.savefig(out/'per_species_advantage.png',dpi=190);plt.close(fig)
 # Confidence reliability curves (uncalibrated probabilities: interpret ECE with care).
 fig,ax=plt.subplots(figsize=(6.7,6.3));ax.plot([0,1],[0,1],'k--',lw=1)
 for label,p in [('Species-first pooled',ps),('Direct routing',pdirect)]:
  conf=p.max(1);correct=(p.argmax(1)==y).astype(float);xx=[];yy=[]
  for low,hi in zip(np.linspace(0,1,16)[:-1],np.linspace(0,1,16)[1:]):
   m=(conf>=low)&(conf<=hi if hi==1 else conf<hi)
   if m.any():xx.append(conf[m].mean());yy.append(correct[m].mean())
  ax.plot(xx,yy,marker='o',label=label)
 ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted confidence',ylabel='Observed accuracy',title='Reliability diagram (15 equal-width bins)');ax.legend();fig.tight_layout();fig.savefig(out/'reliability.png',dpi=190);plt.close(fig)
 if sample is not None:
  # Biological-sample cluster bootstrap, preserving paired cells and within-sample correlation.
  groups=pd.DataFrame({'sample':sample,'s':sgood.astype(int),'d':dgood.astype(int)}).groupby('sample').agg(n=('s','size'),ns=('s','sum'),nd=('d','sum'))
  vals=groups[['n','ns','nd']].to_numpy(dtype=float);nunits=len(vals)
  boot=np.empty(args.bootstraps)
  for j in range(args.bootstraps):
   ix=rng.integers(0,nunits,size=nunits)
   arr=vals[ix].sum(axis=0);boot[j]=100*(arr[1]-arr[2])/arr[0]
  low,hi=np.quantile(boot,[.025,.975]);observed=100*(sgood.mean()-dgood.mean())
  pd.DataFrame({'bootstrap_difference_pp':boot}).to_csv(out/'bootstrap_draws.csv',index=False)
  pd.DataFrame([{'units':'sample','n_samples':nunits,'n_cells':len(y),'observed_species_minus_direct_pp':observed,'bootstrap_95_low_pp':low,'bootstrap_95_high_pp':hi}]).to_csv(out/'bootstrap_summary.csv',index=False)
  print(f'Paired cluster bootstrap: n={nunits} samples, difference={observed:.3f} pp, 95% interval [{low:.3f}, {hi:.3f}] pp')
  # Matched draws, with each group equally weighted at every repeat and N.
  sizes=sorted(set(int(x) for x in args.sizes.split(',')))
  sample_ix={sid:np.where(sample==sid)[0] for sid in np.unique(sample)}
  eligible={k:v for k,v in sample_ix.items() if len(v)>=args.min_cells}
  if not eligible:print(f'No sample meets --min-cells={args.min_cells}; aggregation skipped')
  else:
   print(f'Aggregation: {len(eligible)}/{len(sample_ix)} pure samples with >={args.min_cells} cells')
   records=[]
   for n in sizes:
    eligible_n={k:v for k,v in eligible.items() if len(v)>=n}
    if not eligible_n:continue
    pred_s=[];pred_d=[];truth=[]
    for sid,ix in eligible_n.items():
     for r in range(args.repeats):
      draw=rng.choice(ix,n,replace=False)
      pred_s.append(ps[draw].mean(axis=0));pred_d.append(pdirect[draw].mean(axis=0));truth.append(y[ix[0]])
    yp=np.asarray(truth);p1=np.asarray(pred_s);p2=np.asarray(pred_d)
    for label,p in [('species_13_pooled',p1),('direct_8',p2)]:
     records.append({'n_cells':n,'model':label,'n_samples':len(eligible_n),'repeats':args.repeats,
                     'accuracy':accuracy_score(yp,p.argmax(1)),
                     'macro_f1':f1_score(yp,p.argmax(1),labels=np.arange(8),average='macro',zero_division=0)})
   agg=pd.DataFrame(records);agg.to_csv(out/'aggregation_curves.csv',index=False)
   fig,ax=plt.subplots(figsize=(8,5.5))
   for label,g in agg.groupby('model'):ax.plot(g.n_cells,g.accuracy*100,marker='o',label=label)
   ax.set(xlabel='Cells per sample',ylabel='AST-routing accuracy (%)',title='Matched-draw probability aggregation');ax.legend();fig.tight_layout();fig.savefig(out/'aggregation_accuracy.png',dpi=190);plt.close(fig)
 if args.species_features or args.direct_features:
  if not(args.species_features and args.direct_features):raise ValueError('Both --species-features and --direct-features are required')
  feature_analysis(args.species_features,args.direct_features,ids,ysp,y,out,args,rng)
 (out/'run_config.json').write_text(json.dumps(vars(args),indent=2))
 (out/'probability_column_mapping.json').write_text(json.dumps({'species':scols,'direct':dcols},indent=2))
 print('Finished. Files saved to',out.resolve())


def main():
 a=cli();out=Path(a.out).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
 if sum(bool(v) for v in (a.compare_only,a.inspect,a.check_models))>1:raise ValueError('Choose only one of --inspect, --check-models, --compare-only')
 if not a.compare_only:
  import torch
  print('PyTorch',torch.__version__,'CUDA available:',torch.cuda.is_available(),flush=True)
  overrides={'species_se':a.checkpoint_species_se,'species_concat':a.checkpoint_species_concat,'panel_se':a.checkpoint_panel_se,'panel_concat':a.checkpoint_panel_concat}
  chosen={}
  for key,directory in RUNS.items():
   inspect_run(key,directory,overrides[key])
   try:
    chosen[key]=str(Path(overrides[key]) if overrides[key] else find_checkpoint(directory)[0])
    print(f'SELECTED CHECKPOINT [{key}]: {chosen[key]}',flush=True)
   except Exception:
    if not a.inspect:raise
  if a.check_models:
   import gc
   for key in RUNS:
    print('\nSTRICT MODEL VALIDATION:',key,flush=True)
    model=load_model(key,chosen[key],a)
    print(f'{key}: PASS; {sum(p.numel() for p in model.parameters()):,} parameters',flush=True)
    del model
    gc.collect()
   print('\nAll four models loaded strictly. Run without --check-models to begin inference.',flush=True)
   return
  if a.inspect:
   try:
    m=locate_data(a)
    print('\nData:',len(m),'cells;',m.sample_id.nunique(),'samples')
    print(m.head().to_string(index=False))
    print('First stack shape:',np.load(m.iloc[0].npy_path,mmap_mode='r').shape)
   except Exception as e:print('\nData inspection:',e)
   print('\nIMPORTANT: Verify the true species/panel class order and channel order from training configs.',flush=True)
   return
  prediction_paths=[out/f'{key}_predictions.csv' for key in RUNS]
  if a.skip_existing and all(p.is_file() for p in prediction_paths):
   print('Reusing all four existing prediction CSVs. Skipping data-loading and inference.',flush=True)
  else:
   # Fail fast on checkpoint/class-order problems before running cell inference.
   # The 4 full-model state_dicts are checked one by one, with strict loading.
   # Unlike a single preflight on the first model, this catches the last model
   # (e.g. panel_se descriptive labels) before GPU hours are spent.
   import gc
   missing_models=[key for key in RUNS if not (a.skip_existing and (out/f'{key}_predictions.csv').is_file())]
   if missing_models:
    print('Preflight: strict-loading all models requiring inference:',missing_models,flush=True)
    for key in missing_models:
     tested=load_model(key,chosen[key],a)
     print(f'Preflight PASS: {key}',flush=True)
     del tested
     gc.collect()
   m=locate_data(a)
   print('Input:',len(m),'cells from',m.sample_id.nunique(),'pure biological samples',flush=True)
   if not a.split_col:print('Holdout selection: supplied CSV should contain ONLY independent Plate 1 cells; no extra split filter applied.',flush=True)
   m.to_csv(out/'selected_cell_manifest.csv',index=False)
   (out/'run_metadata.json').write_text(json.dumps({'args':vars(a),'checkpoints':chosen,'cells':len(m)},indent=2))
   chosen_cells=balanced_feature_selection(m,a.max_umap_cells,a.seed) if not a.no_features else set()
   for key in RUNS:
    p=out/f'{key}_predictions.csv'
    if a.skip_existing and p.is_file():
     print('Using existing',p,flush=True)
     continue
    infer(key,chosen[key],m,a,out,chosen_cells)
 print('\nEvaluating paired routing classifiers (one combined script; no subprocesses)',flush=True)
 for method in ('concat','se'):
  s=out/f'species_{method}_predictions.csv';d=out/f'panel_{method}_predictions.csv'
  if not s.is_file() or not d.is_file():
   raise FileNotFoundError(f'Comparison for {method} requires BOTH {s} and {d}')
  sf=out/f'species_{method}_features.npz';df=out/f'panel_{method}_features.npz'
  print('\n=== COMPARISON',method.upper(),'===',flush=True)
  run_comparison(s,d,out/f'comparison_{method}',sf,df,a)
 print('\nAll done:',out,flush=True)

if __name__=='__main__':main()
