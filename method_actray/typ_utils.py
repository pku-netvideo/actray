import torch
from nerfstudio.utils import writer
from nerfstudio.utils.misc import torch_compile
from nerfstudio.model_components.losses import ray_samples_to_sdist, lossfun_outer, lossfun_distortion

class Typ_TimeWriter:
    """Timer context manager that calculates duration around wrapped functions"""

    def __init__(self, name, step=None, writer = writer):
        self.name = name
        self.step = step
        self.writer = writer

        self.duration: float = 0.0

        self.starter = torch.cuda.Event(enable_timing = True)
        self.ender = torch.cuda.Event(enable_timing = True)

    def __enter__(self):
        torch.cuda.synchronize()
        self.starter.record()
        return self

    def __exit__(self, *args):
        # typ code:
        self.ender.record()
        torch.cuda.synchronize()

        self.duration = self.starter.elapsed_time(self.ender) / 1000.

        self.writer.put_time(
            name=self.name,
            duration=self.duration,
            step=self.step,
            avg_over_steps=False,
        )

@torch_compile(dynamic=True, mode="reduce-overhead", backend="eager")
def radial_and_tangential_distort(x_correct: torch.Tensor, y_correct: torch.Tensor, distortion_params: torch.Tensor) -> torch.Tensor:
    """
    Compute distorted coords given opencv distortion parameters.
    Correspond to nerfstudio.cameras.camera_utils.radial_and_tangential_distort

    Args:
        x_correct, y_correct: The ideal coordinates.
        distortion_params: The distortion parameters [k1, k2, k3, k4, p1, p2].
    Returns:
        The distorted coordinates.
    """
    k1 = distortion_params[..., 0]
    k2 = distortion_params[..., 1]
    k3 = distortion_params[..., 2]
    k4 = distortion_params[..., 3]
    p1 = distortion_params[..., 4]
    p2 = distortion_params[..., 5]

    r_2 = x_correct * x_correct + y_correct * y_correct
    d = 1.0 + r_2 * (k1 + r_2 * (k2 + r_2 * (k3 + r_2 * k4)))
    x_distort = d * x_correct + 2 * p1 * x_correct * y_correct + p2 * (r_2 + 2 * x_correct * x_correct)
    y_distort = d * y_correct + 2 * p2 * x_correct * y_correct + p1 * (r_2 + 2 * y_correct * y_correct)
    return x_distort, y_distort


def HuberLoss_per_ray(target: torch.Tensor, prediction: torch.Tensor, delta = 0.1, reduction = 'mean'):
    """The same as nn.HuberLoss. Default to 'mean' setting and 0.1 delta."""
    abs_difference = torch.abs(target - prediction)
    square_difference = (target - prediction) ** 2
    loss = torch.where(abs_difference < delta, 0.5 * square_difference, delta * (abs_difference - 0.5 * delta))
    if reduction == 'mean':
        loss = torch.mean(loss, dim = -1)
    elif reduction == 'sum':
        loss = torch.sum(loss, dim = -1)
    return loss

def distortion_loss_per_ray(weights_list, ray_samples_list):
    """The same as nerfstudio.model_components.losses.distortion_loss, but return the loss for each ray."""
    c = ray_samples_to_sdist(ray_samples_list[-1])
    w = weights_list[-1][..., 0]
    loss = lossfun_distortion(c, w)
    return loss

def interlevel_loss_per_ray(weights_list, ray_samples_list) -> torch.Tensor:
    """The same as nerfstudio.model_components.losses.interlevel_loss, but return the loss for each ray."""
    c = ray_samples_to_sdist(ray_samples_list[-1]).detach()
    w = weights_list[-1][..., 0].detach()
    assert len(ray_samples_list) > 0

    loss_interlevel = torch.zeros(c.shape[0], device = c.device)
    for ray_samples, weights in zip(ray_samples_list[:-1], weights_list[:-1]):
        sdist = ray_samples_to_sdist(ray_samples)
        cp = sdist  # (num_rays, num_samples + 1)
        wp = weights[..., 0]  # (num_rays, num_samples)
        loss_interlevel += torch.mean(lossfun_outer(c, w, cp, wp), dim = -1)

    return loss_interlevel

def mse_loss_per_ray(target, prediction):
    """MSE loss of each ray
    
    Args:
        target:     (..., 3)        target rgb
        prediction: (..., 3)        predicted rgb
    Returns:
                    (...)           mse loss for each ray
    """
    return torch.mean((target - prediction)**2, dim = -1)