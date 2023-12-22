import random

import torch
from jaxtyping import Int
from torch import Tensor

from dataclasses import dataclass, field
from nerfstudio.data.utils.pixel_sampling_utils import erode_mask
from typing import (
    Dict,
    Optional,
    Type,
    Union,
)

from nerfstudio.configs.base_config import (
    InstantiateConfig,
)

from nerfstudio.data.pixel_samplers import PixelSamplerConfig
from nerfstudio.data.pixel_samplers import PixelSampler

from method_actray.typ_utils import Typ_TimeWriter

def multinomial_large_scale(sampling_prob, sampling_num):
    all_idx = []
    sampling_batch = 16777200 # in torch.multinomial, number of categories cannot exceed 2^24
    prob_sum = torch.sum(sampling_prob)
    prob_len = len(sampling_prob)
    for sampling_begin in range(0, prob_len, sampling_batch):
        sampling_end = min(sampling_begin + sampling_batch, prob_len)
        sampling_prob_batch = sampling_prob[sampling_begin:sampling_end]
        prob_sum_batch = torch.sum(sampling_prob_batch)
        sampling_num_batch = int(torch.ceil((prob_sum_batch / prob_sum) * sampling_num).item())

        if(sampling_num_batch <= 0):
            continue

        idx_batch = torch.multinomial(sampling_prob_batch, num_samples=sampling_num_batch, replacement=False)
        idx_batch += sampling_begin
        all_idx.append(idx_batch)
    all_idx = torch.cat(all_idx)
    all_idx = all_idx[0:sampling_num]
    return all_idx

@dataclass
class ActivePixelSamplerConfig(PixelSamplerConfig):
    """Configuration for active pixel sampler instantiation."""

    _target: Type = field(default_factory=lambda: ActivePixelSampler)
    actray_start_iter: int = 300
    """when to start the ActRay sampling strategy"""
    prefetch_scale: int = 5
    """an acceleration strategy: in one batch prefetch the rays used in the several following batches (prefetch_scale gives the number)."""

class ActivePixelSampler(PixelSampler):
    
    config: ActivePixelSamplerConfig
    # outdate = outdate_total - outdate_offset, where outdate_total will increment each iteration
    outdate_offset_image: torch.Tensor
    outdate_total: float
    similarity_image: torch.Tensor
    loss_image: torch.Tensor
    
    # UCB here refers to the loss term multiplying the similarity term in our formula
    UCB_image: torch.Tensor
    img_shape: list # num_images, height, width

    # acceleration: pre-fetch rays for multiple batches
    indices_buffer: torch.Tensor
    indices_buffer_size: int

    def __init__(
            self, 
            config: ActivePixelSamplerConfig, 
            img_shape: list, 
            device: Union[torch.device, str] = "cpu",
            **kwargs
        ):
        self.img_shape = img_shape
        self.device = device
        self.outdate_offset_image = torch.zeros(img_shape).to(device = device)
        self.outdate_total = 0
        self.similarity_image = torch.zeros(img_shape).to(device = device)

        self.loss_image = torch.zeros(img_shape).to(device = device)
        self.loss_gzero_count = 0

        self.UCB_image = torch.ones(img_shape).to(device = device)*1e30
        
        self.indices_buffer = torch.empty((1)).to(device = device)
        self.indices_buffer_size = 0
        
        super().__init__(config, **kwargs)
    
    def calc_probability(self, img_idx: Tensor, batch_size: int, importance_ratio: float = 0.5):
        outdate_image = self.outdate_total + self.outdate_offset_image[img_idx, ...].view(-1)
        UCB_image = self.UCB_image[img_idx, ...].view(-1)

        if UCB_image.max() == 1e30:
            return UCB_image

        mask = (UCB_image > 0)
        UCB_image_max, UCB_image_median = UCB_image[mask].max(), UCB_image[mask].median()
        outdate_image_median = outdate_image[mask].median()
        UCB_ratio = UCB_image_max / UCB_image_median
        scale_factor = torch.log(UCB_ratio * importance_ratio) / outdate_image_median
        outdate_image = outdate_image.clamp(max = outdate_image_median) * scale_factor

        UCB = UCB_image * torch.exp(outdate_image)
        assert UCB.max() != torch.nan and UCB.max() != torch.inf

        return UCB


    def sample_method(
            self,
            batch_size: int,
            img_idx: Tensor,
            step: int,
    ) -> Int[Tensor, "batch_size 3"]:
        """Pixel Sampler using ActRay. It samples pixels according to possibilities calculated with loss values.
        
        Args:
            batch_size:         number of rays in a batch
            img_idx:            indices of images being sampled in this batch
            step:               iteration index
        """
        with torch.no_grad():
            if self.config.actray_start_iter >= 0 and step > self.config.actray_start_iter:
                fetch_len = min(self.indices_buffer_size, batch_size)
                indices_part1 = self.indices_buffer[self.indices_buffer_size - fetch_len : self.indices_buffer_size]
                self.indices_buffer_size -= fetch_len
                if fetch_len < batch_size:
                    UCB = self.calc_probability(img_idx = img_idx, batch_size = batch_size)
                    self.indices_buffer = multinomial_large_scale(sampling_prob = UCB, sampling_num = batch_size * self.config.prefetch_scale)
                        
                    self.indices_buffer_size = batch_size * self.config.prefetch_scale
                    fetch_len = batch_size - fetch_len
                    indices_part2 = self.indices_buffer[self.indices_buffer_size - fetch_len : self.indices_buffer_size]
                    self.indices_buffer_size -= fetch_len
                    indices = torch.cat([indices_part1, indices_part2])
                else:
                    indices = indices_part1
                
                indices_x = ( indices // (self.img_shape[1] * self.img_shape[2]) ).int()
                indices_y = (( indices - indices_x*(self.img_shape[1] * self.img_shape[2]) ) // self.img_shape[2]).int()
                indices_z = ( indices - indices_x*(self.img_shape[1] * self.img_shape[2]) - indices_y * (self.img_shape[2]) ).int()
                indices = torch.stack((indices_x, indices_y, indices_z), dim = 1)
                return indices
            else:
                return super().sample_method(batch_size, self.img_shape[0], self.img_shape[1], self.img_shape[2], device = 'cpu').to(self.device)
    

    def collate_image_dataset_batch(self, batch: Dict, num_rays_per_batch: int, step: int, keep_full_image: bool = False):
        """The same as PixelSampler.collate_image_dataset_batch"""
        indices = self.sample_method(num_rays_per_batch, batch["image_idx"], step)

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        c, y, x = c.cpu(), y.cpu(), x.cpu()
        collated_batch = {
            key: value[c, y, x] for key, value in batch.items() if key != "image_idx" and value is not None
        }

        assert collated_batch["image"].shape[0] == num_rays_per_batch

        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices

        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch
    

    def sample(self, image_batch: Dict, step: int):
        """The same as PixelSampler.sample"""
        if isinstance(image_batch["image"], torch.Tensor):
            pixel_batch = self.collate_image_dataset_batch(
                image_batch, self.num_rays_per_batch, step, keep_full_image=self.config.keep_full_image
            )
        else:
            raise ValueError("image_batch['image'] must be a list or torch.Tensor")
        return pixel_batch