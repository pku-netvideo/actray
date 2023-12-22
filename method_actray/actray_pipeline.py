"""
The base pipeline for ActRayDynamicBatchPipeline and ActRayVanillaPipeline.
"""

from dataclasses import dataclass, field
from typing import Literal, Type, Optional

import torch
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipeline, DynamicBatchPipelineConfig
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.model_components.renderers import RGBRenderer
from nerfstudio.models.nerfacto import NerfactoModel
import nerfstudio.utils.poses as pose_utils

from method_actray.actray_datamanager import ActRayDataManager
from method_actray.typ_utils import Typ_TimeWriter, radial_and_tangential_distort, HuberLoss_per_ray, distortion_loss_per_ray, interlevel_loss_per_ray, mse_loss_per_ray

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
class ActRayPipelineConfig(DynamicBatchPipelineConfig):
    """ActRayPipeline Config"""

    _target: Type = field(default_factory=lambda: ActRayPipeline)
    propagation_reduce_to: int = 5000
    """how many rays in batch to be used for loss propagation"""
    stop_exploration_thresh: float = 3.5
    """the constant deciding when to terminate the initial exploration period (the smaller, the earlier)"""
    use_random_background_color: bool = False
    """whether to use random background color"""

class ActRayPipeline(DynamicBatchPipeline):
    """ActRayPipeline implementing functions shared by ActRayDynamicBatchPipeline and ActRayVanillaPipeline."""

    config: ActRayPipelineConfig
    datamanager: ActRayDataManager
    images: torch.Tensor # instead of directly using cached data from dataloader(whose order is disrupted), we store a copy
    bg_color = None

    def __init__(
        self,
        config: ActRayPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler)
        sample_image = torch.from_numpy(self.datamanager.train_dataset.get_numpy_image(0).astype("float32") / 255.0)
        num_channels = sample_image.shape[-1]

        image_num = self.datamanager.train_dataset.__len__()
        height = self.datamanager.train_ray_generator.cameras.height[0, 0]
        width = self.datamanager.train_ray_generator.cameras.width[0, 0]

        self.images = torch.empty((image_num * height * width, num_channels), device = self.device)
        for i in range(image_num):
            img = torch.from_numpy(self.datamanager.train_dataset.get_numpy_image(i).astype("float32") / 255.0)
            self.images[i * height * width : (i+1) * height * width, :] = img.view(-1, num_channels)

        self.init_prop_finished = False
        self.propagation_ratio_max = 0.

        if self.config.use_random_background_color == False:
            if self.model.config.background_color not in {"last_sample", "random"}:
                self.bg_color = RGBRenderer.get_background_color(background_color = self.model.config.background_color, shape = (3,), device = self.device)
            else:
                self.bg_color = self.model.config.background_color
        else:
            self.bg_color = torch.rand(3).to(self.device)
            self.model.renderer_rgb.background_color = self.bg_color
            self.datamanager.train_dataset._dataparser_outputs.alpha_color = self.bg_color.cpu()
            self.datamanager.eval_dataset._dataparser_outputs.alpha_color = self.bg_color.cpu()

        self.total_ray_num = self.datamanager.train_pixel_sampler.loss_image.shape[0]
    
        # set the UCB value in background area to 0, so that we won't sample them in the beginning
        if type(self.datamanager.dataparser).__name__ == 'Blender':
            background_mask = (self.images[..., 3] == 0)
            UCB_sequence = self.datamanager.train_pixel_sampler.UCB_image.view(-1)
            UCB_sequence[background_mask] = 0
            self.total_ray_num = (UCB_sequence == 1e30).sum()

    def apply_background_color(self, images, bg_color):
        """apply the background color if the given images have 4 channels

        Args:
            images:     (..., 3 or 4)       images to apply background color
            bg_color:   (3) or other        background color
        """
        if images.shape[-1] <= 3:
            return images
        return images[..., :3] * images[..., 3:] + bg_color * (1. - images[..., 3:])
    
    def _projection(self, surface_points: torch.Tensor):
        """
        Project surface points to each camera.

        Args:
            surface_points: (point_num, 3)              the surface points of each ray
        Return:
            pic_space_idx:  (valid_cast_point_num, )    the valid projection of surface points(uv coordinate, where u is the row index and v is the colume index, but flattened to 1D
            seen_mask:      (point_num*cam_num, )       the mask indicating which projections are valid
        """
        cam_num = self.datamanager.train_dataset.__len__()
        point_num = surface_points.shape[0]
        camera_indices = torch.arange(0, cam_num, 1, dtype = torch.int).expand(point_num, cam_num).to(device = self.device).reshape(-1) # (point_num * cam_num)
        cameras = self.datamanager.train_ray_generator.cameras[camera_indices]
        height = cameras.height[0] # supposing all images share the same resolution
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
        view_space_coords = torch.sum(shifted * rotation_T, dim = -1) # equivalent to premultiply rotation_T
        

        # filter out points between near plane and far plane
        near, far = 1e-10, 1e30
        if hasattr(self.model.config, 'near_plane'):
            near = self.model.config.near_plane
        if hasattr(self.model.config, 'far_plane'):
            far = self.model.config.far_plane

        seen_mask = (near < -view_space_coords[..., 2]) & (-view_space_coords[..., 2] < far) # (point_num * cam_num)
        view_space_coords = view_space_coords[seen_mask]
        camera_indices = camera_indices[seen_mask]
        cameras = cameras[seen_mask]

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

        pic_space_idx = camera_indices * (height * width) + pic_space_coords_y * width + pic_space_coords_x

        return pic_space_idx, seen_mask

    def _propagation(self, pic_space_idx, gt_ray, loss_new, loss_sequence, similarity_sequence, UCB_sequence):
        """Propagate the information to indices given by pic_space_idx.

        Args:
            pic_space_idx:          (cast_pixel_num)            the valid projection of surface points given by _projection
            gt_ray:                 (cast_pixel_num)            rgb of the sampled ray in the training batch for each cast pixel
            loss_new:               (cast_pixel_num)            loss value of the sampled ray in the training batch for each cast pixel
            loss_sequence:          (num_all_pixels)            maintained loss value of each pixel in training dataset
            similarity_sequence:    (num_all_pixels)            maintained similarity value of each pixel in training dataset
            UCB_sequence:           (num_all_pixels)            maintained UCB value (in fact only the loss term multiplying the similarity term) of each pixel in training dataset
        """
        gt_cast = self.apply_background_color(self.images[pic_space_idx, ...], self.bg_color) # (ray_num * image_num) * 3

        similarity_new = 1 - ((gt_ray - gt_cast)**2).sum(dim = 1).sqrt()
        similarity_new[similarity_new < 0.7] = 0

        # abandon failed propagations
        propagation_success_mask = (similarity_new > 0)
        pic_space_idx = pic_space_idx[propagation_success_mask]
        similarity_new = similarity_new[propagation_success_mask]
        loss_new = loss_new[propagation_success_mask]

        loss_updated = (loss_sequence[pic_space_idx] * similarity_sequence[pic_space_idx] + loss_new * similarity_new * similarity_new) / (similarity_sequence[pic_space_idx] + similarity_new + 1e-30)
        similarity_updated = (similarity_sequence[pic_space_idx] * similarity_sequence[pic_space_idx] + similarity_new * similarity_new) / (similarity_sequence[pic_space_idx] + similarity_new + 1e-30)
        
        ret = (UCB_sequence[pic_space_idx]==1e30).sum()

        UCB_sequence[pic_space_idx] = loss_updated / (similarity_updated + 1e-8)
        loss_sequence[pic_space_idx] = loss_updated
        similarity_sequence[pic_space_idx] = similarity_updated

        return ret

    def get_loss_per_ray(self, outputs, batch):
        """Rec-calculate the loss values of each sampled ray, for nerfstudio's code dosen't provide such data.

        Args:
            outputs:                    the pure outputs of model
            batch:                      the pure outputs from next_train()
        Return:
            loss_per_ray:   (ray_num)   re-calculated loss value for each sampled ray
        """
        with torch.no_grad():
            loss_per_ray = mse_loss_per_ray(outputs['rgb'], batch['image'])
            # for nerfacto there're distortion loss and interlevel loss
            if isinstance(self.model, NerfactoModel):
                loss_per_ray += self.model.config.distortion_loss_mult * distortion_loss_per_ray(outputs['weights_list'], outputs['ray_samples_list'])
                loss_per_ray += self.model.config.interlevel_loss_mult * interlevel_loss_per_ray(outputs['weights_list'], outputs['ray_samples_list'])
            return loss_per_ray

    def _update_active_pixel_samplers(self, loss_ray: torch.Tensor, outputs_depth: torch.Tensor, ray_bundle: RayBundle, pic_space_idx_ray_bundle: torch.Tensor, step: int):
        """Update the information(outdate, similarity, loss) stored in ActivePixelSampler.

        Args:
            loss_ray:                       (ray_num)       re-calculated loss value for each sampled ray
            outputs_depth:                  (ray_num)       depth of rendered rays (in fact, it's not true 'depth', but the distance between the surface point and the origin along the ray)
            ray_bundle:                                     ray information
            pic_space_idx_ray_bundle:       (ray_num)       picture space coordinates (in 1D) of sampled ray bundle
            step:                                           num of iterations
        """     
        with torch.no_grad():
        
            image_num = self.datamanager.train_dataset.__len__()
            height = self.datamanager.train_ray_generator.cameras.height[0, 0] # supposing all images share the same resolution
            width = self.datamanager.train_ray_generator.cameras.width[0, 0]

            outdate_offset_sequence = self.datamanager.train_pixel_sampler.outdate_offset_image.view(-1)
            similarity_sequence = self.datamanager.train_pixel_sampler.similarity_image.view(-1)
            loss_sequence = self.datamanager.train_pixel_sampler.loss_image.view(-1)
            UCB_sequence = self.datamanager.train_pixel_sampler.UCB_image.view(-1)

            surface_points = ray_bundle.origins + outputs_depth * ray_bundle.directions

            # update the information of sampled rays
            gt_ray = self.apply_background_color(self.images[pic_space_idx_ray_bundle, :], self.bg_color)

            self.datamanager.train_pixel_sampler.outdate_total += 1
            outdate_offset_sequence[pic_space_idx_ray_bundle] = -self.datamanager.train_pixel_sampler.outdate_total
            UCB_sequence[pic_space_idx_ray_bundle] = loss_ray / (1.0 + 1e-8)
            similarity_sequence[pic_space_idx_ray_bundle] = 1.
            loss_sequence[pic_space_idx_ray_bundle] = loss_ray

            # select the rays of topk loss values in this batch for propagation
            ray_num = ray_bundle.shape[0]
            if self.config.propagation_reduce_to > 0:
                ray_num = min(ray_num, self.config.propagation_reduce_to)
            _, topk_ray_idx = torch.topk(input = loss_ray, k = ray_num)

            surface_points = surface_points[topk_ray_idx]
            loss_ray = loss_ray[topk_ray_idx]
            gt_ray = gt_ray[topk_ray_idx]
            
            pic_space_idx, seen_mask = self._projection(surface_points)

            gt_ray = (gt_ray.unsqueeze(1).repeat(1, image_num, 1).view(-1, 3))[seen_mask, :]
            loss_new = (loss_ray.unsqueeze(1).repeat(1, image_num).view(-1))[seen_mask]

            valid_propagation_count = self._propagation(pic_space_idx, gt_ray, loss_new, loss_sequence, similarity_sequence, UCB_sequence)

            # whether should we stop the initial exploration process
            propagation_ratio_current = valid_propagation_count / ray_num
            need_to_stop = (self.init_prop_finished == False) and \
                            (self.propagation_ratio_max / propagation_ratio_current > self.config.stop_exploration_thresh)

            if need_to_stop == True:
                unpropagate_mask = (UCB_sequence == 1e30)
                loss_sequence[unpropagate_mask] = loss_sequence[~unpropagate_mask].mean()
                similarity_sequence[unpropagate_mask] = similarity_sequence[~unpropagate_mask].mean()
                UCB_sequence[unpropagate_mask] = loss_sequence[unpropagate_mask] / (similarity_sequence[unpropagate_mask] + 1e-8)
                self.init_prop_finished = True

            self.propagation_ratio_max = max(self.propagation_ratio_max, propagation_ratio_current)