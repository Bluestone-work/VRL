import torch
from scripts.train_finite_image_flow_sac import entropy_temperature_loss


def test_temperature_increases_below_target_entropy():
    log_alpha=torch.tensor(-6.,requires_grad=True)
    logp=torch.tensor([4.],requires_grad=True)  # entropy -4, target -3
    entropy_temperature_loss(log_alpha,logp,-3.,torch.ones(1)).backward()
    assert log_alpha.grad < 0  # gradient descent increases alpha
    assert logp.grad is None


def test_temperature_decreases_above_target_and_masks_wait():
    log_alpha=torch.tensor(-6.,requires_grad=True)
    entropy_temperature_loss(log_alpha,torch.tensor([2.,1000.]),-3.,torch.tensor([1.,0.])).backward()
    assert log_alpha.grad == 1


def test_all_waiting_does_not_change_temperature():
    log_alpha=torch.tensor(-6.,requires_grad=True)
    loss=entropy_temperature_loss(log_alpha,torch.tensor([1000.]),-3.,torch.zeros(1))
    loss.backward()
    assert torch.isfinite(loss) and log_alpha.grad == 0
