"""
A pipeline using ActRay inherited from VanillaPipeline.
"""

from dataclasses import dataclass, field
from typing import Literal, Type, Optional

import torch
from torch.cuda.amp.grad_scaler import GradScaler
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union, cast
from nerfstudio.utils import profiler


from method_actray.actray_pipeline import ActRayPipelineConfig, ActRayPipeline
from method_actray.typ_utils import Typ_TimeWriter, mse_loss_per_ray

from pathlib import Path

import cv2, os

@dataclass
class ActRayVanillaPipelineConfig(ActRayPipelineConfig):
    """ActRay version of Vanilla  Pipeline Config"""

    _target: Type = field(default_factory=lambda: ActRayVanillaPipeline)

class ActRayVanillaPipeline(ActRayPipeline):
    """ActRay version of Vanilla Pipeline."""

    config: ActRayVanillaPipelineConfig

    def __init__(
        self,
        config: ActRayVanillaPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        train_num_rays_per_batch = config.datamanager.train_num_rays_per_batch
        eval_num_rays_per_batch = config.datamanager.eval_num_rays_per_batch
        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler)
        self.datamanager.train_pixel_sampler.set_num_rays_per_batch(train_num_rays_per_batch)
        self.datamanager.eval_pixel_sampler.set_num_rays_per_batch(eval_num_rays_per_batch)
    
    def get_train_loss_dict(self, step: int):
        """Do the same things as VanillaPipeline and update our pixel_sampler.

        Args:
            step: current iteration step
        """
        # if using random background color, update background_color in datasets and model
        if self.config.use_random_background_color == True:
            self.bg_color = torch.rand(3).to(self.device)
            self.model.renderer_rgb.background_color = self.bg_color
            self.datamanager.train_dataset._dataparser_outputs.alpha_color = self.bg_color.cpu()
            self.datamanager.eval_dataset._dataparser_outputs.alpha_color = self.bg_color.cpu()

        # Below is the code from VanillaPipeline
        ray_bundle, batch = self.datamanager.next_train(step)
        height = self.datamanager.train_ray_generator.cameras.height[0, 0]
        width = self.datamanager.train_ray_generator.cameras.width[0, 0]
        pic_space_idx_ray_bundle = (batch['indices'][:, 0] * height * width) + (batch['indices'][:, 1] * width) + batch['indices'][:, 2]

        if 'image' in batch.keys():
            batch['image'] = self.apply_background_color(self.images[pic_space_idx_ray_bundle], self.bg_color)

        model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        if self.config.datamanager.camera_optimizer is not None:
            camera_opt_param_group = self.config.datamanager.camera_optimizer.param_group
            if camera_opt_param_group in self.datamanager.get_param_groups():
                # Report the camera optimization metrics
                metrics_dict["camera_opt_translation"] = (
                    self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, :3].norm()
                )
                metrics_dict["camera_opt_rotation"] = (
                    self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, 3:].norm()
                )

        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        # Our code: Update the information stored in PixelSampler
        if self.datamanager.train_pixel_sampler.config.actray_start_iter >= 0 and step >= self.datamanager.train_pixel_sampler.config.actray_start_iter:

            # re-calculate the loss values for each ray
            loss_per_ray = self.get_loss_per_ray(model_outputs, batch)
            if 'depth' in model_outputs.keys():
                self._update_active_pixel_samplers(loss_per_ray, model_outputs['depth'], ray_bundle, pic_space_idx_ray_bundle, step)
            else:
                self._update_active_pixel_samplers(loss_per_ray, model_outputs['expected_depth'], ray_bundle, pic_space_idx_ray_bundle, step)

        return model_outputs, loss_dict, metrics_dict

    # inherited from VanillaPipeline
    @profiler.time_function
    def get_eval_loss_dict(self, step: int) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict