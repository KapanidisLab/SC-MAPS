#!/usr/bin/env python3
"""Plot manuscript-ready SC-MAPS OOD results calculated by 02_ood_scoring_and_population.py.
Figures are generated from measured CSVs; no results are hardcoded.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score

LABELS={'knn':'kNN feature distance','mahalanobis':'Mahalanobis feature distance',
        'entropy':'Normalised entropy','msp':'1 − maximum softmax probability'}
ORDER=['knn','mahalanobis','entropy','msp']

def save(fig,base):
    fig.savefig(base.with_suffix('.png'),dpi=300,bbox_inches='tight',facecolor='white')
    fig.savefig(base.with_suffix('.pdf'),bbox_inches='tight',facecolor='white')
    plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('ood_outputs/results'))
    p.add_argument('--outdir',type=Path,default=None)
    args=p.parse_args()
    source=args.results
    out=args.outdir or (source/'figures');out.mkdir(parents=True,exist_ok=True)
    known=pd.read_csv(source/'ood_cell_scores_known_test.csv')
    unseen=pd.read_csv(source/'ood_cell_scores_unseen.csv')
    metrics=pd.read_csv(source/'ood_single_cell_metrics.csv').set_index('method')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,
                         'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,
                         'axes.spines.right':False,'savefig.dpi':300})
    fig,ax=plt.subplots(figsize=(6.9,6.2))
    y=np.r_[np.zeros(len(known),dtype=int),np.ones(len(unseen),dtype=int)]
    for method in ORDER:
        scores=np.r_[known['ood_'+method].to_numpy(),unseen['ood_'+method].to_numpy()]
        fpr,tpr,_=roc_curve(y,scores)
        ax.plot(fpr,tpr,lw=2,label=f'{LABELS[method]} (AUROC {roc_auc_score(y,scores):.3f})')
    ax.plot([0,1],[0,1],ls='--',color='0.6',lw=1)
    ax.set(xlabel='False-positive rate (known species)',ylabel='True-positive rate (unseen organisms)',
           xlim=(0,1),ylim=(0,1),title='A  Single-cell open-set discrimination')
    ax.legend(loc='lower right',frameon=False,fontsize=9)
    save(fig,out/'ood_A_ROC_methods')

    individual=pd.read_csv(source/'ood_per_organism_rejection.csv')
    df=individual[individual.method=='knn'].sort_values('unseen_species')
    fig,ax=plt.subplots(figsize=(7.4,4.9))
    x=np.arange(len(df)); ax.bar(x,df.rejection_rate.to_numpy()*100,color='#3274a1',width=.62)
    ax.set_xticks(x,df.unseen_species.to_numpy())
    ax.set_ylim(0,105);ax.set_ylabel('Rejected unseen cells (%)')
    ax.set_title('B  Rejection by unseen organism (kNN)')
    ax.grid(axis='y',alpha=.20);ax.set_axisbelow(True)
    for ix,val in enumerate(df.rejection_rate):
        ax.text(ix,100*val+2.0,f'{100*val:.1f}%',ha='center',fontsize=9)
    save(fig,out/'ood_B_organism_rejection')

    pop=pd.read_csv(source/'ood_population_metrics.csv')
    fig,ax=plt.subplots(figsize=(7.5,5.0))
    for method in ORDER:
        sel=pop[pop.method==method].sort_values('cells')
        if len(sel):ax.plot(sel.cells,sel.unseen_rejection*100,marker='o',lw=2,label=LABELS[method])
    ax.set(xlabel='Cells per sampled population',ylabel='Unseen population rejection (%)',
           ylim=(0,105),title='C  Population-level open-set rejection')
    ax.set_xticks(sorted(pop.cells.unique()))
    ax.grid(alpha=.2);ax.legend(loc='lower right',frameon=False,fontsize=8)
    save(fig,out/'ood_C_population_rejection')

    fig,ax=plt.subplots(figsize=(7.5,5.0))
    for method in ORDER:
        sel=pop[pop.method==method].sort_values('cells')
        if len(sel):ax.plot(sel.cells,sel.known_retention*100,marker='o',lw=2,label=LABELS[method])
    ax.axhline(95,ls='--',color='0.5',lw=1,label='Target calibration retention')
    ax.set(xlabel='Cells per sampled population',ylabel='Known population retention (%)',
           ylim=(0,105),title='D  Retention of known populations')
    ax.set_xticks(sorted(pop.cells.unique()))
    ax.grid(alpha=.2);ax.legend(loc='lower right',frameon=False,fontsize=8)
    save(fig,out/'ood_D_population_known_retention')
    print('Saved four PDF/PNG figure pairs in',out)

if __name__=='__main__':main()
