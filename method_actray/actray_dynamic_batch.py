"""
A pipeline using ActRay inherited from DynamicBatchPipeline.
"""

from dataclasses import dataclass, field
from typing import Literal, Type, Optional

import torch
from torch.cuda.amp.grad_scaler import GradScaler


from method_actray.actray_pipeline import ActRayPipelineConfig, ActRayPipeline
from method_actray.typ_utils import Typ_TimeWriter

from pathlib import Path

import cv2, os

@dataclass
class ActRayDynamicBatchPipelineConfig(ActRayPipelineConfig):
    """ActRay version of Dynamic Batch Pipeline Config"""

    _target: Type = field(default_factory=lambda: ActRayDynamicBatchPipeline)

class ActRayDynamicBatchPipeline(ActRayPipeline):
    """ActRay version of Dynamic Batch Pipeline."""

    config: ActRayDynamicBatchPipelineConfig

    def __init__(
        self,
        config: ActRayDynamicBatchPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler)
            
    def get_train_loss_dict(self, step: int):
        """Do the same things as DynamicBatchPipeline and update our pixel_sampler.

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

        # Below is the code from DynamicBatchPipeline
        if "num_samples_per_batch" not in metrics_dict:
            raise ValueError(
                "'num_samples_per_batch' is not in metrics_dict."
                "Please return 'num_samples_per_batch' in the models get_metrics_dict function to use this method."
            )
        self._update_dynamic_num_rays_per_batch(int(metrics_dict["num_samples_per_batch"]))
        self._update_pixel_samplers()

        # add the number of rays
        assert "num_rays_per_batch" not in metrics_dict
        assert self.datamanager.train_pixel_sampler is not None
        metrics_dict["num_rays_per_batch"] = torch.tensor(self.datamanager.train_pixel_sampler.num_rays_per_batch)

        # Our code: Update the information stored in PixelSampler
        if self.datamanager.train_pixel_sampler.config.actray_start_iter >= 0 and step >= self.datamanager.train_pixel_sampler.config.actray_start_iter:

            # re-calculate the loss values for each ray
            loss_per_ray = self.get_loss_per_ray(model_outputs, batch)
            if 'depth' in model_outputs.keys():
                self._update_active_pixel_samplers(loss_per_ray, model_outputs['depth'], ray_bundle, pic_space_idx_ray_bundle, step)
            else:
                self._update_active_pixel_samplers(loss_per_ray, model_outputs['expected_depth'], ray_bundle, pic_space_idx_ray_bundle, step)

        return model_outputs, loss_dict, metrics_dict