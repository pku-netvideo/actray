"""
A pipeline using ActRay inherited from DynamicBatchPipeline.
"""

from dataclasses import dataclass, field
from typing import Literal, Type, Optional

import torch
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipeline, DynamicBatchPipelineConfig
from nerfstudio.cameras.rays import RayBundle
import nerfstudio.utils.poses as pose_utils

from method_actray.actray_datamanager import ActRayDataManager, ActRayDataManagerConfig
from method_actray.typ_utils import Typ_TimeWriter, radial_and_tangential_distort

from nerfacc.grid import ray_aabb_intersect

from pathlib import Path

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

import cv2, os

@dataclass
class ActRayDynamicBatchPipelineConfig(DynamicBatchPipelineConfig):
    """Dynamic Batch Pipeline Config"""

    _target: Type = field(default_factory=lambda: ActRayDynamicBatchPipeline)
    propogation_reduce_to: int = 5000
    stop_exploration_thresh: float = 5.35

class ActRayDynamicBatchPipeline(DynamicBatchPipeline):
    """Pipeline with logic for changing the number of rays per batch."""

    config: ActRayDynamicBatchPipelineConfig
    datamanager: ActRayDataManager
    images: torch.Tensor # instead of directly using cached data from dataloader(whose order is disrupted), we store a copy

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
        self.images = torch.empty_like(self.datamanager.train_image_dataloader.cached_collated_batch['image']).to(device = device)
        self.images[self.datamanager.train_image_dataloader.cached_collated_batch['image_idx']] = self.datamanager.train_image_dataloader.cached_collated_batch['image'].to(device = device)
        self.images = self.images.view(-1, 3)

        self.init_prop_finished = False
        self.propagation_ratio_max = 0.
    
        if type(self.datamanager.dataparser).__name__ == 'Blender':
            image_num = self.datamanager.train_dataset.__len__()
            height = self.datamanager.train_ray_generator.cameras.height[0, 0]
            width = self.datamanager.train_ray_generator.cameras.width[0, 0]
            background_mask = torch.empty(self.images.shape[0])
            for i in range(image_num):
                alpha_image = torch.from_numpy(self.datamanager.train_dataset.get_numpy_image(i).astype("float32") / 255.0)[..., 3]
                background_mask[i * height * width : (i+1) * height * width] = alpha_image.view(-1)
            background_mask = (background_mask == 0)

            tmp = self.datamanager.train_pixel_sampler.UCB_image.view(-1)
            tmp[background_mask] = 0
    
    def _projection(self, surface_points: torch.Tensor, step):
        """
        Project surface points to each camera.

        Args:
            surface_points: (point_num, 3, )      the surface points of each ray
            step:                                 iteration id
        Return:
            pic_space_idx: (point_num*cam_num, )  The projection of surface points(uv coordinate, where u is the row index and v is the colume index, but flattened to 1D
        """
        cam_num = self.datamanager.train_dataset.__len__()
        point_num = surface_points.shape[0]
        camera_indices = torch.arange(0, cam_num, 1, dtype = torch.int).expand(point_num, cam_num).to(device = self.device).reshape(-1) # (point_num * cam_num)
        cameras = self.datamanager.train_ray_generator.cameras[camera_indices]
        height = cameras.height[0]
        width = cameras.width[0]

        c2w = cameras.camera_to_worlds

        if self.datamanager.train_ray_generator.pose_optimizer.config.mode != 'off':
            camera_opt_to_camera = self.datamanager.train_ray_generator.pose_optimizer(camera_indices).view(-1, 3, 4)
            c2w = pose_utils.multiply(c2w, camera_opt_to_camera)

        rotation = c2w[..., :3, :3]
        origins = c2w[..., 3]


        surface_points = surface_points.unsqueeze(1).expand(point_num, cam_num, 3).reshape(-1, 3)
        shifted = surface_points - origins
        rotation_T = rotation.permute(0, 2, 1)
        shifted = shifted.unsqueeze(1)
        view_space_coords = torch.sum(shifted * rotation_T, dim = -1)
        

        # filter out points between near plane and far plane
        near, far = 1e-10, 1e30
        if hasattr(self.model.config, 'near_plane'):
            near = self.model.config.near_plane
        if hasattr(self.model.config, 'far_plane'):
            far = self.model.config.far_plane

        z_g_zero_mask = (near < -view_space_coords[..., 2]) & (-view_space_coords[..., 2] < far)
        view_space_coords = view_space_coords[z_g_zero_mask]
        camera_indices = camera_indices[z_g_zero_mask]
        cameras = cameras[z_g_zero_mask]

        fx = cameras.fx.view(-1)
        fy = cameras.fy.view(-1)
        cx = cameras.cx.view(-1)
        cy = cameras.cy.view(-1)

        norm_view_space_coords_x = view_space_coords[..., 0] / (-view_space_coords[..., 2])
        norm_view_space_coords_y = view_space_coords[..., 1] / (-view_space_coords[..., 2])

        distortion_params = cameras.distortion_params
        if distortion_params is not None: # from ideal coordinates to distorted coordinates
            norm_view_space_coords_x, norm_view_space_coords_y = radial_and_tangential_distort(norm_view_space_coords_x, norm_view_space_coords_y, distortion_params)
            
        pic_space_coords_x = norm_view_space_coords_x * fx + cx
        pic_space_coords_y = -norm_view_space_coords_y * fy + cy

        pic_space_coords_y = torch.clamp(pic_space_coords_y, 0.0, height.item() - 1.0).long()
        pic_space_coords_x = torch.clamp(pic_space_coords_x, 0.0, width.item() - 1.0).long()

        pic_space_idx = camera_indices * (height * width) + pic_space_coords_y * width + pic_space_coords_x # masked_size

        return pic_space_idx, z_g_zero_mask


    def _update_active_pixel_samplers(self, outputs_rgb: torch.Tensor, outputs_depth: torch.Tensor, ray_bundle: RayBundle, batch: torch.Tensor, pic_space_coords_ray_bundle: torch.Tensor, step: int):
        """
        Update the information(outdate, similarity, loss) stored in ActivePixelSampler.

        Args:
            outputs_rgb: (ray_num, 3)       rgb color of rendered rays
            outputs_depth: (ray_num, 1)     depth of rendered rays (in fact, it's not true 'depth', but the distance between the surface point and the origin along the ray)
            ray_bundle & batch:             ray information
            pic_space_coords_ray_bundle:    ray_num x 3  picture space coordinates of sampled ray bundle
            step:                           iteration id
        """

        # actray strategy won't be activated until certain number of iterations
        if self.datamanager.train_pixel_sampler.config.actray_start_iter < 0 or step < self.datamanager.train_pixel_sampler.config.actray_start_iter:
            return

        with torch.no_grad():

            image_num = self.datamanager.train_dataset.__len__()
            # supposing all images share the same resolution
            height = self.datamanager.train_ray_generator.cameras.height[0, 0]
            width = self.datamanager.train_ray_generator.cameras.width[0, 0]

            outdate_offset_sequence = self.datamanager.train_pixel_sampler.outdate_offset_image.view(-1)
            similarity_sequence = self.datamanager.train_pixel_sampler.similarity_image.view(-1)
            loss_sequence = self.datamanager.train_pixel_sampler.loss_image.view(-1)
            UCB_sequence = self.datamanager.train_pixel_sampler.UCB_image.view(-1)

            surface_points = ray_bundle.origins + outputs_depth * ray_bundle.directions

            # update the information of sampled rays
            pic_space_idx_ray_bundle = (pic_space_coords_ray_bundle[:, 0] * height * width) + (pic_space_coords_ray_bundle[:, 1] * width) + pic_space_coords_ray_bundle[:, 2]
            loss_ray = torch.sqrt(torch.sum((outputs_rgb - batch)**2, dim = -1))
            loss_ray[loss_ray < 1e-3] = 0

            self.datamanager.train_pixel_sampler.outdate_total += 1
            outdate_offset_sequence[pic_space_idx_ray_bundle] = -self.datamanager.train_pixel_sampler.outdate_total
            UCB_sequence[pic_space_idx_ray_bundle] = loss_ray / (1.0 + 1e-8)

            similarity_sequence[pic_space_idx_ray_bundle] = 1
            loss_sequence[pic_space_idx_ray_bundle] = loss_ray

            # select the rays of topk loss values in this batch for propagation
            ray_num = ray_bundle.shape[0]
            if self.config.propogation_reduce_to > 0:
                ray_num = min(ray_num, self.config.propogation_reduce_to)
            _, topk_ray_idx = torch.topk(input = loss_ray, k = ray_num)

            surface_points = surface_points[topk_ray_idx]
            loss_ray = loss_ray[topk_ray_idx]
            batch = batch[topk_ray_idx]
             
            pic_space_idx, seen_mask = self._projection(surface_points, step)

            gt_ray = (batch.unsqueeze(1).repeat(1, image_num, 1).view(-1, 3))[seen_mask, :]
            loss_new = (loss_ray.unsqueeze(1).repeat(1, image_num).view(-1))[seen_mask]

            gt_cast = self.images[pic_space_idx, ...]

            similarity_new = 1 - ((gt_ray - gt_cast)**2).sum(dim = 1).sqrt()
            similarity_new[similarity_new < 0.7] = 0

            # abandon failed propagations
            propagation_success_mask = (similarity_new > 0)
            pic_space_idx = pic_space_idx[propagation_success_mask]
            similarity_new = similarity_new[propagation_success_mask]
            loss_new = loss_new[propagation_success_mask]

            loss_updated = (loss_sequence[pic_space_idx] * similarity_sequence[pic_space_idx] + loss_new * similarity_new * similarity_new) / (similarity_sequence[pic_space_idx] + similarity_new + 1e-30)
            similarity_updated = (similarity_sequence[pic_space_idx] * similarity_sequence[pic_space_idx] + similarity_new * similarity_new) / (similarity_sequence[pic_space_idx] + similarity_new + 1e-30)

            loss_updated[loss_updated < 1e-3] = 0

            # whether should we stop the initial exploration process
            propagation_ratio_current = (UCB_sequence[pic_space_idx]==1e30).sum() / ray_num
            need_to_stop = (self.init_prop_finished == False) and \
                            (self.propagation_ratio_max / propagation_ratio_current > self.config.stop_exploration_thresh)

            UCB_sequence[pic_space_idx] = loss_updated / (similarity_updated + 1e-8)
            loss_sequence[pic_space_idx] = loss_updated
            similarity_sequence[pic_space_idx] = similarity_updated

            if need_to_stop == True:
                unpropagate_mask = (UCB_sequence == 1e30)
                loss_sequence[unpropagate_mask] = loss_sequence[~unpropagate_mask].mean()
                similarity_sequence[unpropagate_mask] = similarity_sequence[~unpropagate_mask].mean()
                UCB_sequence[unpropagate_mask] = loss_sequence[unpropagate_mask] / (similarity_sequence[unpropagate_mask] + 1e-8)
                self.init_prop_finished = True

            self.propagation_ratio_max = max(self.propagation_ratio_max, propagation_ratio_current)
    
    def get_train_loss_dict(self, step: int):
        # Below is the code from VanillaPipeline
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self._model(ray_bundle)
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
        gt_rgb = batch["image"][..., :3].to(self.device)
        pred_rgb, gt_rgb = self.model.renderer_rgb.blend_background_for_loss_computation(
            pred_image=model_outputs["rgb"],
            pred_accumulation=model_outputs["accumulation"],
            gt_image=gt_rgb,
        )

        if 'depth' in model_outputs.keys():
            self._update_active_pixel_samplers(pred_rgb, model_outputs['depth'], ray_bundle, gt_rgb, batch['indices'], step)
        else:
            self._update_active_pixel_samplers(pred_rgb, model_outputs['expected_depth'], ray_bundle, gt_rgb, batch['indices'], step)
        return model_outputs, loss_dict, metrics_dict