"""Plot only an admitted EXP0056 matrix, with all training seeds visible."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    root=parser.parse_args().root
    audit=json.loads((root/'AUDIT.json').read_text())
    if not audit['valid_comparison_gate']:
        raise ValueError('Refusing to plot a failed comparison matrix')
    order=['graph_anchor','graph_ppo','mappo_anchor','r_mappo','tpg','balanced_1s','memory_n1']
    labels=['Graph + anchor','Graph + PPO','MAPPO + anchor','MAPPO','Rule TPG','Balanced 1 s','Single cluster']
    colors=['#107b80','#72b2ad','#4677b2','#99b7de','#969696','#b6b6b6','#d0d0d0']
    summaries={s['arm']:s for s in audit['summaries']}
    fig,axes=plt.subplots(1,4,figsize=(15,4.7),layout='constrained')
    for ax,metric,title,scale in zip(axes,
        ['removal','removal_auc_180','wall_contact_s','spacing_violation_pair_s'],
        ['Thrombus removal (%)','Removal AUC (%)','Wall contact (cluster-s)','Spacing violations (pair-s)'],
        [100,100,1,1]):
        values=[summaries[a][metric]*scale for a in order]
        ax.barh(np.arange(len(order)),values,color=colors,height=.65)
        if metric=='spacing_violation_pair_s':
            ax.text(.02,order.index('memory_n1'),'N/A',transform=ax.get_yaxis_transform(),va='center',fontsize=9)
        for i,a in enumerate(order):
            points=[s[metric]*scale for s in audit['per_seed'] if s['arm']==a]
            if points:
                ax.scatter(points,[i]*len(points),s=21,facecolor='white',edgecolor='black',zorder=3,linewidth=.7)
        ax.set_yticks(np.arange(len(order)),labels if ax is axes[0] else ['']*len(order))
        ax.invert_yaxis();ax.set_title(title,fontsize=10)
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        if scale==100:ax.set_xlim(0,105)
        elif max(values)==0:ax.set_xlim(0,1)
        ax.spines[['top','right']].set_visible(False)
    safe='; '.join(f"{labels[i]} {summaries[a]['safe']}/{summaries[a]['rows']}" for i,a in enumerate(order))
    protocol=json.loads((root/'manifest.json').read_text())['protocol']
    tag=('EXP0057: frozen policies, clean CPU deployment' if 'parent_study' in protocol else
         'EXP0058: continuous residual study' if 'CONTINUOUS' in protocol['experiment'] else 'EXP0056: paired development study')
    fig.suptitle(tag+f" — 8 layouts, 3 training seeds, {protocol['training_steps']:,} training controls",fontsize=12)
    fig.supxlabel('Dots: individual training-seed means. Safety success: '+safe+'\nShared local TPG module comparison; 2 mm is an uncalibrated spacing proxy.',fontsize=8)
    destination=root/'figures';destination.mkdir(exist_ok=True)
    for suffix in ('png','svg'):
        fig.savefig(destination/f'factorial_comparison.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
