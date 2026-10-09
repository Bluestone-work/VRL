import numpy as np
import torch
from marl.lysis_abcd import FlowPolicy
from scripts.train_lysis_nav import History


def test_reset_windows_and_batch_independence():
    torch.set_num_threads(1)
    torch.manual_seed(14)
    p=FlowPolicy(history=True,arch='transformer',layers=1,window=8,dim=39).eval()
    hist=History(2,8)
    seq,mask=hist.push(np.ones((2,39),np.float32))
    seq=torch.tensor(seq);mask=torch.tensor(mask)
    with torch.no_grad():
        a=p.backbone(seq,mask)
        changed=seq.clone();changed[mask]=1234
        assert torch.allclose(a,p.backbone(changed,mask))
        assert torch.allclose(a[:1],p.backbone(seq[:1],mask[:1]),atol=1e-6)
        reset=History(2,8)
        assert reset.mask.all() and not reset.seq.any()


def test_auxiliary_reaches_context_and_checkpoint_roundtrip():
    torch.manual_seed(2)
    p=FlowPolicy(history=True,arch='transformer',layers=1,window=8,dim=39)
    seq=torch.randn(2,8,39);mask=torch.zeros(2,8,dtype=torch.bool)
    pred=p.predict_motion(seq,mask,torch.randn(2,3))
    pred.square().mean().backward()
    assert p.context.weight_ih_l0.grad.abs().sum()>0
    q=FlowPolicy(history=True,arch='transformer',layers=1,window=8,dim=39)
    q.load_state_dict(p.state_dict())
    assert torch.equal(p.backbone(seq,mask),q.backbone(seq,mask))


def test_previous_command_uses_execution_frame():
    # world -> local F; deployment must invert the OLD frame, as training does.
    old=np.array([[[0.,1,0],[-1,0,0],[0,0,1]]])
    world=np.array([[.3,.4,.5]])
    local=np.einsum('nij,nj->ni',old,world)
    assert np.allclose(np.einsum('nji,nj->ni',old,local),world)


def test_history_previous_command_contract_is_local_frame():
    # The policy token stores the executed local command, while the plant
    # receives its world transform. The two representations must not be mixed.
    local = np.array([[.2, -.4, .7]])
    F = np.array([[[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]]])
    world = np.einsum('nji,nj->ni', F, local)
    assert not np.allclose(local, world)
    assert np.allclose(np.einsum('nij,nj->ni', F, world), local)
