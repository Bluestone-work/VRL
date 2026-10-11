"""Plot component-attribution metrics from report_nav_decomposition output."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); a=ap.parse_args()
    rows=json.loads((a.root/'aggregate.json').read_text())
    strengths=sorted(set(r['strength'] for r in rows)); variants=sorted(set(r['variant'] for r in rows));
    colors=plt.cm.tab10.colors
    fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    specs=[('task_success','raw/full %',100),('strict_success','strict %',100),('t90_s','T90 (s)',1),('wall_contact_s','wall robot·s',1)]
    for ax,(key,label,scale) in zip(axs.flat,specs):
        for vi,v in enumerate(variants):
            xs=[]; ys=[]
            for s in strengths:
                for n in (1,2,3):
                    z=[r for r in rows if r['strength']==s and r['clusters']==n and r['variant']==v]
                    if z: xs.append(f'{s}/N{n}'); ys.append(z[0][key]*scale)
            ax.plot(xs,ys,marker='o',label=v,color=colors[vi%len(colors)])
        ax.set_title(label); ax.grid(alpha=.25); ax.tick_params(axis='x',rotation=55)
    axs[0,0].legend(fontsize=8,ncol=2)
    fig.savefig(a.root/'component_metrics.png',dpi=180); fig.savefig(a.root/'component_metrics.pdf')
    print(a.root/'component_metrics.png')
if __name__=='__main__': main()
