"""Read-only research publisher: figures/Word/morning snapshot for EXP0061."""
from datetime import datetime
import json
import os
import subprocess
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from scripts.md_to_docx import convert

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/validation/EXP0061_ADAPTIVE_WORLD_RL_20261006'
FIG=ROOT/'research/figures/EXP0061_20261006'


def architecture():
    FIG.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(13,7));ax.set_xlim(0,13);ax.set_ylim(0,7);ax.axis('off')
    def box(x,y,w,h,label,color='#e4f1f6'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.10',fc=color,ec='#527080'))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9.5,color='#173345')
    def arrow(a,b,dashed=False):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,lw=1.2,color='#527080',linestyle='--' if dashed else '-'))
    ax.text(.1,6.65,'Predictive multi-agent DRL: implemented EXP0061 architecture',fontsize=17,weight='bold',color='#173345')
    ax.text(.1,6.2,'Learned one-step observation / outcome model; no simulator rollouts at inference',fontsize=11,color='#527080')
    box(.15,4.25,2.25,1.15,'Measured history\n8 frames x 66 features\nPreoperative route features')
    box(2.95,4.25,2.1,1.15,'Shared encoder + GRU\nHidden state: 64\nLocal actor information')
    box(5.6,4.25,2.1,1.15,'Base policy logits\n64 to 9 options\nWarm-started PPO')
    box(8.55,4.25,2.25,1.15,'Add predicted effects\nCategorical PPO actor\n0.5 s decisions','#d9efdf')
    arrow((2.5,4.82),(2.85,4.82));arrow((5.15,4.82),(5.5,4.82));arrow((7.8,4.82),(8.45,4.82))
    box(2.95,2.1,2.1,1.2,'9 candidate actions\nHistory + action one-hot\nInput size: 64 + 9')
    box(5.6,2.1,2.1,1.2,'3 model members\n73 to 128 to 28\nBootstrap auxiliary fitting','#f5e9d8')
    box(8.55,2.1,2.25,1.2,'Predicted reward + risks\nReward disagreement\nGate: 45 to 9','#f5e9d8')
    arrow((4,4.15),(4,3.4));arrow((5.15,2.7),(5.5,2.7));arrow((7.8,2.7),(8.45,2.7));arrow((9.7,3.4),(9.7,4.15))
    ax.text(10.98,4.38,'Common route feedback\n+ TPG holds\n+ spacing filter\n10 Hz execution',fontsize=9,color='#527080')
    arrow((10.9,4.82),(11.1,4.82))
    box(.15,.35,4.9,1.,'Next measured state (24), reward (1), hazard scores (3)\nTerminal measurements; never cross reset boundaries\nResidual dynamics initialized to persistence','#f4f5f6')
    box(5.6,.35,5.2,1.,'Training: PPO + observation / outcome prediction loss\nCritic pools observed agent encodings only\nSame-budget continued PPO and model-gate ablation','#f4f5f6')
    arrow((6.65,1.45),(6.65,2.0),True)
    fig.savefig(FIG/'predictive_network.png',dpi=200,bbox_inches='tight');fig.savefig(FIG/'predictive_network.svg',bbox_inches='tight');plt.close(fig)


def publish():
    status=json.loads((OUT/'STATUS.json').read_text());md=(OUT/'REPORT.md').read_text()
    md+='\n\n## 实现结构与验收资料\n\n'
    md+=f'![实际实现的预测辅助 DRL 网络]({FIG}/predictive_network.png)\n\n'
    md+='原始训练与评估保留在各 round 目录；模型门的预测来自训练好的网络，不是查询仿真真值。\n'
    md+='\n## 固定行为轨迹上的预测审计\n\n'
    md+='额外开发场景采用固定行为生成轨迹，各模型观察相同初始化；这些审计不进入自动调参规则。风险分数尚未校准，必须同时看阳性样本数量和零风险预测基线。\n\n'
    for audit in sorted(OUT.glob('round_*/world_model_audit.json')):
        r=json.loads(audit.read_text())
        md+=f'- {audit.parent.name}：下一测量 MSE {r["next_state_mse"]:.4f}，持久性基线 {r["persistence_mse"]:.4f}；奖励 MSE {r["reward_mse"]:.4f}，零奖励基线 {r["zero_reward_mse"]:.4f}；壁/粒子/间距阳性样本 {r["positive_counts"]}；Brier {r["hazard_brier"]}，零风险基线 {r["zero_risk_brier"]}。\n'
    # Honest progress curves; no surrogate training success shown as evaluation.
    fig,axes=plt.subplots(1,2,figsize=(11,3.5))
    for run in sorted((ROOT/'research/runs/EXP0061_ADAPTIVE_WORLD_RL_20261006').glob('r*/training.jsonl')):
        rs=[json.loads(l) for l in run.read_text().splitlines() if l.strip()]
        if not rs:continue
        axes[0].plot([r['iteration'] for r in rs],[r['state_mse'] for r in rs],alpha=.65,label=run.parent.name)
        if 'world_ppo' in run.parent.name:
            axes[1].plot([r['iteration'] for r in rs],[r['hazard_brier'] for r in rs],alpha=.65,label=run.parent.name)
    axes[0].set(xlabel='PPO updates',ylabel='Next measured state MSE (online slice)')
    axes[1].set(xlabel='PPO updates',ylabel='Hazard Brier score (online slice)')
    for ax in axes:
        if ax.lines:ax.legend(fontsize=5,ncol=2)
    fig.tight_layout();fig.savefig(FIG/'model_diagnostics.png',dpi=160);plt.close(fig)
    md+=f'\n![在线模型诊断，不是任务成功率]({FIG}/model_diagnostics.png)\n'
    guis=sorted(FIG.glob('round_*/gui_10s.png'))
    if guis:md+=f'\n![最近候选策略实际 GUI 回放，场景固定]({guis[-1]})\n'
    report=OUT/'验收报告_连续DRL与世界模型.md';report.write_text(md)
    tmp=OUT/'验收报告_连续DRL与世界模型.tmp.docx'
    convert(md,tmp,src_dir=OUT);tmp.replace(OUT/'验收报告_连续DRL与世界模型.docx')
    if datetime.now().hour>=9 and not (OUT/'09点阶段验收.docx').exists():
        import shutil
        shutil.copyfile(OUT/'验收报告_连续DRL与世界模型.docx',OUT/'09点阶段验收.docx')
    (OUT/'PUBLISH_STATUS.json').write_text(json.dumps(dict(time=datetime.now().isoformat(),queue_stage=status['stage'])))
    return status


def main():
    os.sched_setaffinity(0,os.sched_getaffinity(0)-{6,7});architecture()
    while True:
        try:
            status=publish()
            for ck in sorted((ROOT/'research/runs/EXP0061_ADAPTIVE_WORLD_RL_20261006').glob('r??_world_ppo/final.pt')):
                audit_dir=OUT/f'round_{ck.parent.name[1:3]}'
                audit=audit_dir/'world_model_audit.json';failed=audit_dir/'world_model_audit.failed'
                if not audit.exists() and not failed.exists():
                    env=dict(os.environ,PYTHONPATH=str(ROOT),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
                    with (audit_dir/'world_model_audit.log').open('w') as log:
                        result=subprocess.run(['nice','-n','10','taskset','-c','0-5',
                            '/home/wj/.cache/vascular-research/cpu312-clean-20261004/bin/python','-m','scripts.audit_vascular_world_model',
                            '--checkpoint',str(ck),'--out',str(audit)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
                    if result.returncode:failed.write_text(str(result.returncode))
            if status['stage'] in ['registered_24_round_budget_completed','stopped_by_local_control',
                                   'infrastructure_blocked_three_rounds','infrastructure_failure']:break
        except (OSError,ValueError,subprocess.SubprocessError) as e:print(repr(e),flush=True)
        time.sleep(60)


if __name__=='__main__':main()
