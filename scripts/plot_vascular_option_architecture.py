"""Scientific architecture diagram of the implemented EXP0060 model."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def main():
    out=Path('research/figures/EXP0060_20261006'); out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(14,8)); ax.set_xlim(0,14); ax.set_ylim(0,8); ax.axis('off')
    def box(x,y,w,h,text,c='#e4f1f6'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.12',fc=c,ec='#527080',lw=1.2))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10,color='#173345')
    def arrow(a,b,c='#527080',style='-'):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,color=c,lw=1.3,linestyle=style))
    ax.text(.15,7.7,'V-MORL: vascular memory option reinforcement learning',fontsize=19,weight='bold',color='#173345')
    ax.text(.15,7.25,'Implemented candidate architecture | Parameter sharing, decentralized actors | EXP0060',fontsize=11,color='#526a77')
    box(.2,5.1,2.4,1.3,'Preoperative map\nRegistered clot locations\nInitial measured positions')
    box(3.15,5.1,2.3,1.3,'Geodesic allocation A\nVascular conflict\ntemporal graph (VCTG)')
    box(6.0,5.1,2.4,1.3,'Route feedback options\n9 candidate commands\nSame options in baselines')
    box(10.4,5.1,3.1,1.3,'Common TPG hold +\nmeasured spacing filter\nWorld command at 10 Hz')
    arrow((2.72,5.75),(3.02,5.75)); arrow((5.57,5.75),(5.87,5.75)); arrow((8.52,5.75),(10.27,5.75))
    box(.2,2.95,2.4,1.25,'Noisy / delayed detections\nLocal particles and peers\n66 features per cluster')
    box(3.15,2.95,2.3,1.25,'Linear 66 to 64\nLayerNorm + tanh\n8 measured frames')
    box(6.,2.95,2.4,1.25,'GRU: hidden size 64\nWindow about 4 seconds\nNo recurrent truth input')
    box(9.05,2.95,2.35,1.25,'Shared actor: 64 to 9\nCategorical option\nDecision every 0.5 s','#d9efdf')
    arrow((2.72,3.55),(3.02,3.55)); arrow((5.57,3.55),(5.87,3.55)); arrow((8.52,3.55),(8.92,3.55))
    arrow((10.2,4.32),(10.2,5.75)); ax.text(9.,4.6,'select',fontsize=10,color='#267149')
    box(6.,.55,2.4,1.25,'Observed-agent pooling\nLocal + team: 128\nCritic: 128 to 128 to 1','#f5e9d8')
    box(9.05,.55,2.35,1.25,'PPO / GAE updates\nRemoval, contact,\nspacing and time rewards','#f5e9d8')
    arrow((7.2,2.82),(7.2,1.92),'#a17b46','--'); arrow((8.52,1.15),(8.92,1.15),'#a17b46','--')
    arrow((10.2,1.92),(10.2,2.82),'#a17b46','--')
    box(.2,.55,4.8,1.25,'Ablations / same information and actuator access\nFF-MAPPO: no temporal memory\nMemory-IPPO: local critic; fixed-option controls','#f2f4f5')
    ax.text(11.7,2.9,'TRAINING ONLY\nTruth supplies outcome\nrewards and evaluation.\nNever actor / critic input.',fontsize=9,color='#976c3a',va='top')
    fig.savefig(out/'network_architecture.png',dpi=220,bbox_inches='tight'); fig.savefig(out/'network_architecture.svg',bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
