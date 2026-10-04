"""Plot both matched continuation regimes; never select a favorable reward arm."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    args=parser.parse_args();root=args.root
    audit=json.loads((root/'AUDIT.json').read_text())
    if not audit['valid_comparison_gate']:raise ValueError('Matrix not admitted')
    protocol=json.loads((root/'manifest.json').read_text())['protocol']
    project=Path(__file__).resolve().parents[1]
    parent=json.loads((project/protocol['parent_study']/'AUDIT.json').read_text())
    labels={'vctpg_ac':'V-CTPG + action-conditioned RL','vctpg_no_ac':'V-CTPG no-AC ablation',
        'r_mappo':'Recurrent MAPPO adaptation','r_ippo':'Recurrent IPPO adaptation'}
    variants=list(labels);regimes=['original_reward','safety_reward']
    colors=['#0072b2','#d55e00'];summary={(s['regime'],s['variant']):s for s in audit['summaries']}
    fig,axes=plt.subplots(1,4,figsize=(16,6))
    for axis,key,title in zip(axes,['removal','removal_auc_180','wall_contact_s','safe_successes'],
            ['Clot removal (%)','Removal AUC / 180 s (%)','Wall exposure (cluster-s)','Registered safe success (%)']):
        scale=100 if key in ('removal','removal_auc_180') else 1
        for j,(regime,color) in enumerate(zip(regimes,colors)):
            ys=np.arange(4)+(j-.5)*.32
            values=[summary[regime,v][key]*scale if key!='safe_successes' else summary[regime,v][key]/summary[regime,v]['rows']*100 for v in variants]
            axis.barh(ys,values,height=.29,color=color,alpha=.8,label=regime.replace('_',' '))
            for i,variant in enumerate(variants):
                points=[s for s in audit['per_seed'] if s['regime']==regime and s['variant']==variant]
                for offset,s in zip([-.07,0,.07],points):
                    val=s[key]*scale if key!='safe_successes' else s[key]/protocol['validation_scenes']*100
                    axis.scatter(val,ys[i]+offset,color='black',s=16,zorder=3)
                if key=='safe_successes':
                    s=summary[regime,variant]
                    axis.annotate(f"{s['safe_successes']}/{s['rows']}",(values[i],ys[i]),xytext=(5,0),textcoords='offset points',fontsize=8,va='center')
        if key!='safe_successes':
            for name,color,style in [('tpg','#666666','--'),('balanced_1s','#cc79a7',':')]:
                baseline=next(s for s in parent['summaries'] if s['arm']==name)
                axis.axvline(baseline[key]*scale,color=color,linestyle=style,linewidth=1.4,label=name)
        axis.set_yticks(range(4),list(labels.values()) if axis is axes[0] else ['']*4)
        axis.invert_yaxis();axis.set_title(title,fontsize=10);axis.grid(axis='x',alpha=.2);axis.set_axisbelow(True)
        axis.spines[['right','top','left']].set_visible(False)
        if key!='wall_contact_s':axis.set_xlim(0,110)
    axes[0].legend(loc='lower right',fontsize=8)
    fig.suptitle('Matched continuation: original vs safety training reward',fontsize=15,y=.96)
    fig.text(.27,.89,'32,768 parent + 16,384 new controls | 3 seeds | same 8 paired development scenes',fontsize=10)
    fig.text(.02,.025,'Dots: all individual seed means. Both regimes use fresh optimizers and separate actor/critic gradient clipping.\n'
        'Classical references are cached from the admitted parent study. Reward changes are not safety guarantees or algorithmic novelty.',fontsize=9)
    fig.subplots_adjust(left=.245,right=.98,top=.83,bottom=.15,wspace=.3)
    out=root/'figures';out.mkdir(exist_ok=True)
    fig.savefig(out/'matched_continuation.png',dpi=160);fig.savefig(out/'matched_continuation.svg');plt.close(fig)
    (out/'plotted_values.json').write_text(json.dumps(dict(summaries=audit['summaries'],seed_points=audit['per_seed']),indent=2)+'\n')
    print(json.dumps(dict(figures=str(out),admitted=True)))


if __name__=='__main__':main()
