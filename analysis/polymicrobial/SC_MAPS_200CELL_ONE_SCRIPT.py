#!/usr/bin/env python3
"""ONE-FILE CPU-only SC-MAPS 200-cell synthetic mixture comparison.

No training, PyTorch, image files, checkpoints, five-seed jobs or external pair CSVs.

Replicates the *design* of the old vs new balanced 200-cell experiments:
  - use the identical FIXED Plate-1 per-cell DenseNet121 probability CSV
  - use each experiment's sample-pair selection embedded INSIDE THIS FILE
  - per pair/repeat draw 100 cells from each species without replacement
  - evaluate 200 repeats for each source-sample pair (default)
  - use pair-keyed random seeds so old/new overlapping pairs get identical draws
  - compare original real routing groups with exactly the SAME saved random
    group-size-matched partitions on the SAME drawn mixed populations
  - report both species and AST-group recoveries, matched permutation nulls

This is a new seeded reconstruction: does not claim to reproduce the old run's
exact random draws unless those original RNG states were retained.

To run on ARC, upload just this .py script, activate an environment with numpy
and pandas, and submit it in a CPU job, e.g. (adjust ARC partition/account):

  sbatch --job-name=scmaps200 --cpus-per-task=4 --mem=24G --time=08:00:00 \
    --output=scmaps200-%j.log --wrap='python -u SC_MAPS_200CELL_ONE_SCRIPT.py'

Run --dry-run first to confirm access to your existing saved probability CSV.
Use --probabilities /exact/path/file.csv if needed.

"""
from __future__ import annotations

import argparse
import base64
import io
import zlib
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_BASE = Path('/data/phys-deep-learning-bacteria/phys3115/Bacteria-data-15 species/Trillian')
DEFAULT_PROBS = DEFAULT_BASE / 'Polymicrobial_analysis_ARC' / 'holdout_plate1_DenseNet121_concat_cell_probabilities.csv'
SPECIES = ('ACAA', 'ECOL', 'ENTC', 'KLAE', 'KLPN', 'PSAR', 'PSMA', 'SAENA', 'SAUR', 'SEPI', 'SERM', 'SLUG', 'STCP')
GROUPS = ('Routine Enterobacterales', 'AmpC/caution Enterobacterales', 'Salmonella', 'Pseudomonas',
          'Acinetobacter', 'Stenotrophomonas', 'S. aureus-like', 'CoNS')
GROUP_BY_SPECIES = {'ACAA':4,'ECOL':0,'ENTC':1,'KLAE':1,'KLPN':0,'PSAR':3,'PSMA':5,'SAENA':2,
                    'SAUR':6,'SEPI':7,'SERM':1,'SLUG':6,'STCP':7}
G = np.array([GROUP_BY_SPECIES[s] for s in SPECIES], dtype=np.int8)
SIZES = np.bincount(G, minlength=8)
assert tuple(SIZES) == (2,3,1,1,1,1,2,2)
LOG = logging.getLogger('SCMAPS-200cell')

# Data embedded to keep this as ONE ARC-uploadable Python file.
EMBEDDED_PAIR_LISTS = {
    "old": """
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
""",
    "new": """
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
/r3smxFKk11JQpliz7Ejo2mJZUCK7vkSSrXoUiyUbtEls1H3fIhPUIj3kjDKEDtRHbHQjKbsHBNKkd3a3RlhSrwvvFjUpZRmqVEn
sdz6W9Gwk/1IxKm634Y4ycSRe5chZ9F9TcTpJp2bKa+h+5qIU0z1arqfvXFORynzST3pJGWKnaiMovbWMkqTfVZCqbKfuF/dbNFl
sehC1S2W8dLQeLl038JYRFeSUOIeXoUU3U8klLjnmyyxZEcsiVI6o2y9qjiWuONFKEl23whFfZHDEQvVJT8go2TJKFkq7dGF1qjL
DulKufSqQoromxFKFnuxGaXIfiKhqC+VMEqSPd/dutxH3S5dtm4toSyyK0koTexRd9ToFsulVxVOekWHlFCa7JsRypQ9PEJRb1px
xJIslGap0YSU+oCMPKOuf9raVNt6P+zSiR10jKLOgTOK2rXmiKU5YkkWdc/XEaOo0+QGXc5jl1GGJaMp3mGwUS6r+xgjVveQk00c
2dWBHNnVgZxkymua4pHdIchppnhk18tTr37jnI5y+ET7qhhlip6MI5YMKeoLv46MaCzqe7iMorpvjFItNcoWXTrKaO1gxE+6eL+Z
g5IgRX09llG6OKvPKIslo2SJJbNYzut9VaOoa8Aow0JRbyhmlGmhNPEOA4e63UKpkLJYdCmiV8UoCemyrvdFpcNuCqOoXWuMUkVP
xkBJVJdh0SWJzg6jLOJtyYzSxDeLDeMFZ6TeuXyhrOt9UWm93peUIbopjKJOcDOK+qouoxRLLP0BulSLLotFF0ppUt3zUv+2Qno+
/PI+DsjJesXmiGdATjLFk4ruHDj0wTonTzxKn2qKB+fVYL1OR1mueM1EKNlBkV8kYpQmdiwwSpbrWqJLs2RUHLGI3kODlCr7IPt1
uWV0WQWG40Ws1AElZfFFIhZLEvseGKWJnRyMUsSXmpgu3ZJRcqibquyD7B8v91i29VvwtgC5fiMU9T48RpniDCCipCLX+4Si3jbI
KFmu1IkuVXYNCCXJ9f7+jIZl1N3Gy2X9Fj7Vwl0liJK6ONPFKEl8HYllpL73xCjNEkuWq2NCKZZYdAdjf0bTkdH9OX1Zv4WjLtzJ
wShJrmsBZb0aw70pjJLQinQfBeoiugbFUqMGY2lib4qDQsdLEb2qy0Jtm/DG6zrhcFHOEDNnD2eaOM2kD9dZvcfcw0mYM8Vq1KOP
q173vNapdNxfED6ehXOe2VNOE44g5XSxRqCcYeKoEwKUU8Va2cN5Mul8vx+u02LJiebolFPF6oVy1IrMwymYk0z6JFM8Q6xjPHUv
Jk7HnMbiWafKahyGbh/lqD37lKN2p1viseU1MWeY8hqmvIZYr3n0qaZxeI9nnT6r64vMnyVHOayUM03xVLG68MTj0pnH00wc9X51
jz7TVK81ntPxNm05/eKuvidIqXKuSiiLOMfKKFnOwwhFvbvQkRFVN8kZM6E08TWgK2WbFsSVjmc7hDLl3JRQkuiJOzKakFIsuixy
brtfF6puk/N+QukWSpFz2tPxNgU4HX/1dCOjVLHDkFGK2EnHKPHutW7JKEFKEx6KgXK/GndlhClZdPkNlKfbqLs87k/HX5+dEUqy
UNQ+RUZR7y5klGmhFDlnJZSOZmT7Ylkslc6WSp/V3SYd2907mluo5xGiKE8fUVJXT3tEUe/hcOhSIaU7YhG6wFiSesfwhbLdMaN5
sXqqIUpV/RBEyeqphijNQslqzkAoT/kBsVRHjda5FMqoq/nLRtnudfH9JX42EkpSrjOLJaknLKJUS0bqC4KMok4CMIp6x7CDsqq7
3uYuoy64m8m7FKEsaqa6n7J2ThBFdloRJcurkVCS6vYjSrVk1NQ6wKButlS6XCmXKyAcu/EcE1EWeU0TykRX4z5KtWS0WNRNFgpV
d1h00ffd/bGcK70N/o0SrZiVuoSSmiUW6XWgWNSJPRZLd1DEczrDWJJDF/meeUYZjvEi51L/A13BM+4=
""",
}


def hash_seed(*parts: object) -> int:
    digest = hashlib.sha256('::'.join(map(str,parts)).encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'little')


def read_design(label: str) -> pd.DataFrame:
    """Load the exact historical pair selection from the embedded compressed data."""
    raw = zlib.decompress(base64.b64decode(EMBEDDED_PAIR_LISTS[label]))
    df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    columns = ['species_a','species_b','sample_a','sample_b']
    missing = set(columns)-set(df.columns)
    if missing: raise ValueError(f'{label} pair file is missing columns: {sorted(missing)}')
    # Uploaded bundle files are pre-filtered to exactly 200 total cells and 50:50.
    df = df[columns].drop_duplicates().reset_index(drop=True)
    if df.empty: raise ValueError(f'No sample pairs in {label} design')
    if (df[columns]=='').any().any(): raise ValueError(f'Missing species/sample IDs in {label}')
    if ((~df.species_a.isin(SPECIES))|(~df.species_b.isin(SPECIES))).any():
        raise ValueError(f'Unrecognised species in {label} pair file')
    if (df.species_a==df.species_b).any(): raise ValueError('Pairs must contain two different species')
    # Explicitly disallow reversed copies of the same sample pair.
    canonical = [tuple(sorted(((r.species_a,r.sample_a),(r.species_b,r.sample_b))))
                 for r in df.itertuples(index=False)]
    if len(set(canonical)) != len(df): raise ValueError(f'Duplicate/reversed sample pairs in {label} design')
    LOG.info('%s: %s unique 200-cell balanced sample pairs', label, len(df))
    return df


def load_predictions(path: Path) -> dict[tuple[str,str],np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f'Missing frozen Plate-1 cell probabilities at {path}\n'
                                'Use --probabilities with the path to your existing CSV on ARC.')
    cols = ['true_species','sample_id'] + [f'prob_{s}' for s in SPECIES]
    df = pd.read_csv(path, usecols=cols, low_memory=False, dtype={'true_species':str,'sample_id':str})
    if df.empty or not df.true_species.isin(SPECIES).all():
        raise ValueError('Prediction CSV is empty or has invalid species labels')
    v = df[[f'prob_{s}' for s in SPECIES]].to_numpy(dtype=np.float32)
    row_sums = v.sum(axis=1)
    if not np.isfinite(v).all() or np.any(v < 0) or np.any(np.abs(row_sums-1) > 0.002):
        raise ValueError('Probability file has NaNs, negative values, or rows not summing to one')
    sample_dict = {}
    for (sp,sample), sub in df.groupby(['true_species','sample_id'],sort=False):
        sample_dict[str(sp),str(sample)] = sub[[f'prob_{s}' for s in SPECIES]].to_numpy(dtype=np.float32,copy=True)
    LOG.info('Loaded %s cells across %s source microscopy samples', len(df),len(sample_dict))
    return sample_dict


def build_draws(old: pd.DataFrame, new: pd.DataFrame, samples:dict,
                repeats:int, seed:int, out:Path):
    combo = pd.concat((old.assign(_name='old'),new.assign(_name='new')),ignore_index=True)
    unique = combo[['species_a','species_b','sample_a','sample_b']].drop_duplicates().reset_index(drop=True)
    keys = list(map(tuple, unique.itertuples(index=False,name=None)))
    lookup = {k:i for i,k in enumerate(keys)}
    pair_indices = {
        name:np.array([lookup[tuple(r)] for r in df.itertuples(index=False,name=None)],dtype=np.int32)
        for name,df in (('old',old),('new',new))
    }
    log_n = len(unique)*repeats
    LOG.info('Generating %d unique sample pairs × %d repeats = %d mixtures',len(unique),repeats,log_n)
    # 13 species scores for every mixture. Float32 is sufficient for comparing ranked outputs.
    matrix = np.empty((log_n, len(SPECIES)),dtype=np.float32)
    for i,(sa,sb,ia,ib) in enumerate(keys):
        if (sa,ia) not in samples or (sb,ib) not in samples:
            raise KeyError(f'Pair not in probability CSV: {sa}/{ia} + {sb}/{ib}')
        a = samples[sa,ia];b=samples[sb,ib]
        if len(a)<100 or len(b)<100:
            raise ValueError(f'Need >=100 cells per source sample, got {len(a)}, {len(b)} at pair #{i}')
        for r in range(repeats):
            random = np.random.default_rng(hash_seed(seed,sa,ia,sb,ib,r))
            x = random.choice(len(a),100,replace=False)
            y = random.choice(len(b),100,replace=False)
            matrix[i*repeats+r] = (a[x].sum(axis=0,dtype=np.float64)+b[y].sum(axis=0,dtype=np.float64))/200
        if (i+1)%200==0: LOG.info('Sampled %d / %d unique source pairs',i+1,len(unique))
    if not np.isfinite(matrix).all(): raise AssertionError('Invalid sampled mean scores')
    true_a = np.array([SPECIES.index(k[0]) for k in keys],dtype=np.int8)
    true_b = np.array([SPECIES.index(k[1]) for k in keys],dtype=np.int8)
    # Retain sampled mean probabilities for transparent/reproducible downstream analyses.
    np.save(out/'all_unique_200cell_mean_probabilities.npy',matrix)
    unique.to_csv(out/'unique_sample_pairs.csv',index=False)
    return matrix,true_a,true_b,pair_indices,unique


def pooled(mat:np.ndarray, mapping:np.ndarray) -> np.ndarray:
    result=np.zeros((len(mat),8),dtype=np.float32)
    for col, g in enumerate(mapping): result[:,int(g)]+=mat[:,col]
    return result


def both_top2(prob:np.ndarray, a:np.ndarray,b:np.ndarray) -> np.ndarray:
    top = np.argpartition(prob, -2, axis=1)[:,-2:]
    return ((top[:,0]==a)&(top[:,1]==b))|((top[:,1]==a)&(top[:,0]==b))


def rows_for(idx:np.ndarray,repeats:int) -> np.ndarray:
    return (idx[:,None]*repeats + np.arange(repeats,dtype=np.int32)).reshape(-1)


def observed_rate_table(df:pd.DataFrame,idx:np.ndarray,r:int,matrix:np.ndarray,
                        sa:np.ndarray,sb:np.ndarray,species_hit:np.ndarray,
                        routing_hit:np.ndarray,shared_hit:np.ndarray) -> pd.DataFrame:
    tab=df.copy()
    tab['same_group'] = [GROUP_BY_SPECIES[a]==GROUP_BY_SPECIES[b] for a,b in zip(df.species_a,df.species_b)]
    tab['species_pair'] = [f'{a}|{b}' for a,b in zip(df.species_a,df.species_b)]
    selected = idx[:,None]*r + np.arange(r)
    tab['both_species_top2'] = species_hit[selected].mean(axis=1)
    tab['both_routing_groups_top2'] = routing_hit[selected].mean(axis=1)
    tab['shared_routing_group_top1'] = shared_hit[selected].mean(axis=1)
    return tab


def bootstrap_species_pair(values:pd.DataFrame,repeats:int,seed:int) -> list[float]:
    if repeats < 1:return [float('nan'),float('nan')]
    clusters=[x['both_routing_groups_top2'].to_numpy() for _,x in values.groupby('species_pair',sort=True)]
    n=len(clusters)
    rng=np.random.default_rng(seed)
    out=np.empty(repeats,dtype=float)
    for b in range(repeats):
        picks=rng.integers(0,n,size=n)
        out[b]=np.concatenate([clusters[i] for i in picks]).mean()
    return np.quantile(out,[.025,.975]).tolist()


def read_arguments():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--probabilities',type=Path,default=DEFAULT_PROBS,
                   help='Existing frozen Plate-1 DenseNet121 concat per-cell probability CSV on ARC')
    p.add_argument('--output',type=Path,default=DEFAULT_BASE/'Chris_200cell_only_ARC')
    p.add_argument('--repeats',type=int,default=200,help='Independent synthetic cell draws per source-sample pair')
    p.add_argument('--permutations',type=int,default=2000,help='Random group assignments, fixed across old/new')
    p.add_argument('--bootstrap',type=int,default=2000,help='Species-pair-cluster bootstrap replicates')
    p.add_argument('--draw-seed',type=int,default=20261008)
    p.add_argument('--null-seed',type=int,default=20261009)
    p.add_argument('--dry-run',action='store_true',help='Only check files, sample availability and matching counts')
    return p.parse_args()


def main():
    args=read_arguments()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s')
    for name,value in [('repeats',args.repeats),('permutations',args.permutations)]:
        if value<1:raise ValueError(f'--{name} must be >=1')
    if args.bootstrap<0:raise ValueError('--bootstrap cannot be negative')
    old=read_design('old');new=read_design('new')
    samples=load_predictions(args.probabilities)
    for name,d in (('old',old),('new',new)):
        count_under=sum(len(samples.get((sp,sid),[]))<100 for sp,sid in
                        set(zip(d.species_a,d.sample_a))|set(zip(d.species_b,d.sample_b)))
        if count_under:raise ValueError(f'{name}: {count_under} sources missing / with fewer than 100 cells')
    old_keys=set(map(tuple,old.itertuples(index=False,name=None)))
    new_keys=set(map(tuple,new.itertuples(index=False,name=None)))
    overlap=old_keys&new_keys
    LOG.info('Old=%d, New=%d, shared=%d; all source samples >=100 cells',len(old),len(new),len(overlap))
    if args.dry_run:
        print(json.dumps({'status':'PREFLIGHT_OK','old_pairs':len(old),'new_pairs':len(new),'shared_pairs':len(overlap),
                          'n_microscopy_samples':len(samples)},indent=2))
        return
    out=args.output.expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    settings=dict(probabilities=str(args.probabilities),pair_lists='embedded in script',pair_sha256={k:hashlib.sha256(zlib.decompress(base64.b64decode(v))).hexdigest() for k,v in EMBEDDED_PAIR_LISTS.items()},
                  repeats=args.repeats,permutations=args.permutations,bootstrap=args.bootstrap,
                  draw_seed=args.draw_seed,null_seed=args.null_seed,old_n=len(old),new_n=len(new),
                  shared_n=len(overlap),species=list(SPECIES),routing_mapping=GROUP_BY_SPECIES)
    (out/'run_configuration.json').write_text(json.dumps(settings,indent=2))
    matrix,a,b,indices,unique = build_draws(old,new,samples,args.repeats,args.draw_seed,out)
    del samples
    group_real = pooled(matrix,G)
    species_hit=both_top2(matrix,np.repeat(a,args.repeats),np.repeat(b,args.repeats))
    truth_a=G[a];truth_b=G[b]
    distinct=truth_a!=truth_b
    routing_hit=both_top2(group_real,np.repeat(truth_a,args.repeats),np.repeat(truth_b,args.repeats))
    routing_hit &= np.repeat(distinct,args.repeats)  # two groups must actually be different
    same_hit=(np.argmax(group_real,axis=1)==np.repeat(truth_a,args.repeats))&np.repeat(~distinct,args.repeats)
    tables={}; masks={};reports={}
    for name,df in [('old',old),('new',new)]:
        table=observed_rate_table(df,indices[name],args.repeats,matrix,a,b,species_hit,routing_hit,same_hit)
        table.to_csv(out/f'{name}_per_sample_pair_results.csv',index=False)
        tables[name]=table
        dif=table[~table.same_group]
        same=table[table.same_group]
        masks[name]=np.repeat(np.isin(np.arange(len(unique)),indices[name])&distinct,args.repeats)
        ci=bootstrap_species_pair(dif,args.bootstrap,hash_seed(args.draw_seed,name,'bootstrap'))
        reports[name]={
            'n_source_sample_pairs':len(table),
            'n_different_group_pairs':len(dif),
            'n_same_group_pairs':len(same),
            'n_synthetic_draws':len(table)*args.repeats,
            'both_exact_species_different_groups':float(dif.both_species_top2.mean()),
            'both_routing_groups_different_groups':float(dif.both_routing_groups_top2.mean()),
            'both_routing_95ci_species_pair_cluster':ci,
            'both_exact_species_same_group':float(same.both_species_top2.mean()),
            'shared_group_top1_same_group':float(same.shared_routing_group_top1.mean()),
        }
        LOG.info('%s observed: species dual %.3f%%; routing dual %.3f%% (different groups only)',name,
                 100*reports[name]['both_exact_species_different_groups'],
                 100*reports[name]['both_routing_groups_different_groups'])

    # Integrity check: matched draws on overlapping old/new source pairs are bit-identical.
    shared = tables['old'].merge(tables['new'],on=['species_a','species_b','sample_a','sample_b'],
                                suffixes=('_old','_new'))
    if len(shared)!=len(overlap):raise AssertionError('Mismatch in shared source-sample pair counts')
    for k in ('both_species_top2','both_routing_groups_top2','shared_routing_group_top1'):
        if not np.allclose(shared[f'{k}_old'],shared[f'{k}_new'],rtol=0,atol=0):
            raise AssertionError(f'Matched source pair outputs differ: {k}')
    shared.to_csv(out/'shared_source_pair_exact_match.csv',index=False)
    LOG.info('Integrity check passed: all %d shared source pairs exactly match',len(shared))

    rng=np.random.default_rng(args.null_seed)
    null_rows={'old':[],'new':[]}
    for i in range(args.permutations):
        permutation=rng.permutation(len(SPECIES))
        mapping=np.empty(len(SPECIES),dtype=np.int8)
        pos=0
        for group_index,size in enumerate(SIZES):
            mapping[permutation[pos:pos+size]]=group_index
            pos+=size
        gp=pooled(matrix,mapping)
        pa=mapping[a];pb=mapping[b]
        perm_distinct=np.repeat(pa!=pb,args.repeats)
        hit=both_top2(gp,np.repeat(pa,args.repeats),np.repeat(pb,args.repeats))&perm_distinct
        for label in ('old','new'):
            eligible=masks[label]             # FIXED real different-group mixture populations
            conditional=eligible & perm_distinct
            null_rows[label].append({'permutation':i,
                                     'strict_top2':float(hit[eligible].mean()),
                                     'conditional_top2':float(hit[conditional].mean()) if conditional.any() else np.nan,
                                     'conditional_n_draws':int(conditional.sum()),
                                     'fraction_remaining_distinct':float(conditional.sum()/eligible.sum())})
        if (i+1)%100==0:LOG.info('Matched null permutations: %d / %d',i+1,args.permutations)

    for name in ('old','new'):
        null=pd.DataFrame(null_rows[name])
        null.to_csv(out/f'{name}_group_size_matched_null.csv',index=False)
        obs=reports[name]['both_routing_groups_different_groups']
        for measure in ('strict','conditional'):
            vals=null[f'{measure}_top2'].dropna().to_numpy()
            if not len(vals):raise AssertionError(f'No valid {measure} permutations')
            reports[name].update({f'null_{measure}_mean':float(vals.mean()),
                                  f'null_{measure}_95pct':np.quantile(vals,[.025,.975]).tolist(),
                                  f'null_{measure}_p_value':(1+int((vals>=obs-1e-12).sum()))/(len(vals)+1)})
        LOG.info('%s matched null: conditional mean %.3f%%; p=%.6f',name,
                 100*reports[name]['null_conditional_mean'],reports[name]['null_conditional_p_value'])

    overview={
        'notes':[
            'Synthetic 200-cell mixtures: 100+100 cells, without replacement, 200 repeated draws per sample pair.',
            'Old/new overlap uses identical cells and same random partitions; different overall estimates reflect sample-pair selections.',
            'Only true different-routing-group species pairs enter the dual-routing null comparison.',
            'Strict null counts randomly merged species groups as failures on a fixed denominator.',
            'Conditional null excludes draws when a random grouping merges the two true species; denominator can vary by permutation.',
            'Species-pair bootstrap clusters by unordered species pair, not cell, but does not provide across-strain generalisation.',
            'This re-samples frozen individual-cell predictions, not real experimentally mixed cultures.',
            'Independent random draws approximate the original run, but cannot recover its exact RNG state unless supplied.'
        ],
        'matched_shared_sample_pairs':len(overlap),
        'old':reports['old'], 'new':reports['new'],
        'new_minus_old_routing_percentage_points':100*(reports['new']['both_routing_groups_different_groups']-
                                                        reports['old']['both_routing_groups_different_groups'])
    }
    (out/'OLD_VS_NEW_FINAL_COMPARISON.json').write_text(json.dumps(overview,indent=2))
    pd.DataFrame([{'design':name,**reports[name]} for name in ('old','new')]).to_csv(out/'OLD_VS_NEW_FINAL_COMPARISON.csv',index=False)
    print('\n=== OLD VS NEW: MATCHED 200-CELL RESULTS ===')
    print('                     OLD               NEW')
    for key in ('n_source_sample_pairs','n_different_group_pairs', 'both_exact_species_different_groups',
                'both_routing_groups_different_groups','null_conditional_mean','null_conditional_p_value'):
        def f(v):return f'{v:.4%}' if ('different_groups' in key or key=='null_conditional_mean') and not key.startswith('n_') else str(round(v,6) if isinstance(v,float) else v)
        print(f'{key:<37} {f(reports["old"][key]):<17}{f(reports["new"][key])}')
    print('\nShared sample pairs with identical draws:',len(overlap))
    print('Files saved to:',out)


if __name__=='__main__':
    main()
