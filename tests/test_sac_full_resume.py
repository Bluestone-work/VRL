"""Real environment + optimizer continuation, not only tensor serialization."""
import json
from pathlib import Path
import subprocess
import sys
import torch


ROOT=Path(__file__).resolve().parents[1]


def test_full_resume_matches_uninterrupted_training(tmp_path):
    common=[sys.executable,'-m','scripts.train_finite_image_flow_sac','--seed','0',
        '--workers','2','--rollout','8','--gradient-steps','1','--batch','16','--capacity','64',
        '--device','cpu','--critic-warmup-iterations','4','--action-space','vector',
        '--dynamic-obstacles','--dynamic-obstacle-count','16','--obstacle-features','--team-reward','--auto-alpha']
    for name,iterations,extra in [('full',6,[]),('part',5,[]),('resumed',6,['--resume',str(tmp_path/'part/resume.pt')])]:
        result=subprocess.run(common+['--out',str(tmp_path/name),'--iterations',str(iterations)]+extra,
            cwd=ROOT,capture_output=True,text=True,timeout=180)
        assert result.returncode==0,result.stdout+result.stderr
    full=torch.load(tmp_path/'full/resume.pt',weights_only=False)
    resumed=torch.load(tmp_path/'resumed/resume.pt',weights_only=False)
    for key in ['actor','critic','target_critic','target_actor']:
        for name,value in full[key].items():
            torch.testing.assert_close(value,resumed[key][name],rtol=0,atol=0)
    torch.testing.assert_close(full['log_alpha'],resumed['log_alpha'],rtol=0,atol=0)
    assert full['replay_cursor']==resumed['replay_cursor']
    assert len(full['replay_rows'])==len(resumed['replay_rows'])
    import numpy as np
    for a,b in zip(full['replay_rows'],resumed['replay_rows']):
        assert a['reward']==b['reward'] and a['terminal']==b['terminal']
        np.testing.assert_array_equal(a['action'],b['action'])
        for obs in ('obs','next'):
            for field in a[obs]:np.testing.assert_array_equal(a[obs][field],b[obs][field])
    row=json.loads((tmp_path/'resumed/training.jsonl').read_text().splitlines()[-1])
    assert row['environment_steps']==96 and row['actor_gradient_updates']==2


def test_reject_policy_only_resume(tmp_path):
    file=tmp_path/'policy.pt'
    torch.save({'actor':{}},file)
    result=subprocess.run([sys.executable,'-m','scripts.train_finite_image_flow_sac',
        '--out',str(tmp_path/'invalid'),'--resume',str(file)],cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode!=0
    assert 'Not a full-state checkpoint' in result.stderr
    assert not (tmp_path/'invalid').exists()
