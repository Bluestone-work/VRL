import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('research/runs/EXP0071_FOUR_ARM_20261009')
out=Path('research/figures/EXP0071_FOUR_ARM_20261009/shared')
out.mkdir(parents=True, exist_ok=True)
fig,axs=plt.subplots(1,3,figsize=(13,4),dpi=160)
arms=[('nav_tf_v3_16u','#1565c0'),('flow_aux_16u','#2e7d32'),('belief_16u','#c62828'),('belief_rolling_tpg_16u_repaired','#6a1b9a')]
for name,color in arms:
    p=root/name/'log.jsonl'
    rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    x=[r.get('it') for r in rows]
    for ax,key,label in [(axs[0],'ret','mean return'),(axs[1],'mean_a0','mean speed action'),(axs[2],'lr','learning rate')]:
        y=[r.get(key) for r in rows]
        ax.plot(x,y,'o-',label=name.replace('_retry',''),color=color)
axs[0].set_title('Training return'); axs[1].set_title('Mean speed action'); axs[2].set_title('Learning rate')
for ax in axs:
    ax.set_xlabel('update'); ax.grid(alpha=.25); ax.legend()
fig.tight_layout(); fig.savefig(out/'training_curves.png')
print(out/'training_curves.png')
