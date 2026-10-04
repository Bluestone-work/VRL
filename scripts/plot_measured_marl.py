"""Render admitted development results with every seed and checkpoint retained."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('root',type=Path)
    args=parser.parse_args(); root=args.root
    audit=json.loads((root/'AUDIT.json').read_text())
    if not audit['valid_comparison_gate']:
        raise RuntimeError('Incomplete/unadmitted matrix cannot produce a comparison plot')
    primary=audit['primary_checkpoint_steps']
    summaries=[s for s in audit['summaries'] if s['steps'] in (0,primary)]
    order=['vctpg_ac','vctpg_no_ac','r_mappo','r_ippo','tpg','balanced_1s','memory_n1']
    display={'vctpg_ac':'V-CTPG + action-conditioned RL','vctpg_no_ac':'V-CTPG no-AC ablation',
        'r_mappo':'Recurrent MAPPO adaptation','r_ippo':'Recurrent IPPO adaptation',
        'tpg':'Shared TPG rule','balanced_1s':'Balanced heuristic (1 s)','memory_n1':'Single-cluster rule'}
    colors=['#1f77b4','#8eb7d8','#009e73','#70baa1','#888888','#cc79a7','#c5a365']
    lookup={s['arm']:s for s in summaries}
    fig,axes=plt.subplots(1,4,figsize=(16,6),gridspec_kw={'width_ratios':[1.1,1.1,1.2,.8]})
    keys=['removal','removal_auc_180','wall_contact_s','safe_successes']
    titles=['Clot removal (%)','Removal AUC / 180 s (%)','Wall exposure (cluster-s)','Registered safe success (%)']
    for ax,key,title in zip(axes,keys,titles):
        values=[]
        for arm in order:
            s=lookup[arm]
            value=s[key]/s['rows']*100 if key=='safe_successes' else s[key]*(100 if key in keys[:2] else 1)
            values.append(value)
        ax.barh(range(len(order)),values,color=colors,height=.6,alpha=.85)
        for i,arm in enumerate(order):
            seeds=[s for s in audit['per_seed'] if s['variant']==arm and s['steps']==primary]
            for offset,s in zip([-.15,0,.15],seeds):
                val=s['safe']/lookup[arm]['independent_scenes']*100 if key=='safe_successes' else s[key]*(100 if key in keys[:2] else 1)
                ax.scatter(val,i+offset,s=22,color='black',zorder=3,marker='o')
            suffix=f"{lookup[arm]['safe_successes']}/{lookup[arm]['rows']}" if key=='safe_successes' else f'{values[i]:.1f}'
            ax.annotate(suffix,(values[i],i),xytext=(5,0),textcoords='offset points',va='center',fontsize=8)
        ax.set_yticks(range(len(order)),[display[x] for x in order] if ax is axes[0] else ['']*len(order))
        ax.invert_yaxis(); ax.set_title(title,fontsize=10); ax.grid(axis='x',alpha=.2); ax.set_axisbelow(True)
        ax.spines[['top','right','left']].set_visible(False)
        seed_values=[s[key] for s in audit['per_seed'] if s['steps']==primary] if key=='wall_contact_s' else []
        ax.set_xlim(0,max(values+seed_values+[1])*1.18 if key=='wall_contact_s' else 115)
    fig.suptitle('Matched measured-information MARL development study',fontsize=15,y=.96)
    fig.text(.30,.89,f'{primary:,} controls per seed | 3 training seeds | 8 paired development scenes',fontsize=10)
    fig.text(.02,.025,'Dots: individual training-seed means. All methods share local TPG mechanics; this is a module comparison.\n'
        'Safe success keeps the registered wall/spacing/contact/loss criteria. No hardware or confirmation-set claim.',fontsize=9)
    fig.subplots_adjust(left=.24,right=.98,top=.83,bottom=.15,wspace=.3)
    out=root/'figures'; out.mkdir(exist_ok=True)
    fig.savefig(out/'primary_comparison.png',dpi=160);fig.savefig(out/'primary_comparison.svg');plt.close(fig)
    (out/'plotted_values.json').write_text(json.dumps(dict(primary=primary,summary=summaries,
        seed_points=[s for s in audit['per_seed'] if s['steps']==primary]),indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for ax,key,label in zip(axes,['removal_auc_180','wall_contact_s'],['Removal AUC (%)','Wall exposure (cluster-s)']):
        for arm,color in zip(order[:4],colors[:4]):
            group=sorted([s for s in audit['summaries'] if s['arm']==arm],key=lambda s:s['steps'])
            scale=100 if key=='removal_auc_180' else 1
            ax.plot([s['steps'] for s in group],[s[key]*scale for s in group],marker='o',color=color,label=display[arm])
            for s in audit['per_seed']:
                if s['variant']==arm:ax.scatter(s['steps'],s[key]*scale,s=18,color=color,alpha=.5)
        ax.set_xlabel('Physical controls per training seed');ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Both preregistered checkpoints; no best-checkpoint selection')
    fig.tight_layout();fig.savefig(out/'checkpoint_diagnostic.png',dpi=160);fig.savefig(out/'checkpoint_diagnostic.svg');plt.close(fig)
    print(json.dumps(dict(figures=str(out),source_audit_passed=True)))


if __name__=='__main__':main()
