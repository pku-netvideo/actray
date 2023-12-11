import torch
from nerfstudio.utils import writer
from nerfstudio.utils.misc import torch_compile

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