import torch
from marl.image_flow_sac import ImageActor


def test_direct_actor_does_not_add_parent_command_to_action_mean():
    actor=ImageActor('direct',zero_init=False).eval()
    hist=torch.zeros(2,24,11)
    mask=torch.zeros(2,24,dtype=torch.bool)
    ctx=torch.zeros(2,48)
    image=torch.zeros(2,4,32,32)
    ctx[0,19:22]=torch.tensor([.9,-.7,.4])
    ctx[1,19:22]=torch.tensor([-.8,.6,-.3])
    with torch.no_grad():
        out=actor.head(actor.encoder(hist,mask,ctx,image))
        action,_=actor.sample(hist,mask,ctx,image,deterministic=True)
    torch.testing.assert_close(action,torch.tanh(out[:,:3]))
    # The old implementation would add atanh(ctx[:,19:22]) here.
    assert not torch.allclose(action,torch.tanh(torch.atanh(ctx[:,19:22])+out[:,:3]))
