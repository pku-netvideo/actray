from __future__ import annotations

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.pixel_samplers import PixelSamplerConfig
from nerfstudio.data.dataparsers.instant_ngp_dataparser import InstantNGPDataParserConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig
from nerfstudio.models.instant_ngp import InstantNGPModelConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification


from method_actray.actray_datamanager import ActRayDataManagerConfig
from method_actray.actray_pixel_samplers import ActivePixelSamplerConfig
from method_actray.actray_dynamic_batch import ActRayDynamicBatchPipelineConfig
from method_actray.actray_vanilla_pipeline import ActRayVanillaPipelineConfig



method_actray_instant_ngp = MethodSpecification(
    config=TrainerConfig(
        method_name="method-actray-instant-ngp",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=ActRayDynamicBatchPipelineConfig(
            propagation_reduce_to=-1,
            stop_exploration_thresh=3.5,
            use_random_background_color=True,
            datamanager=ActRayDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                train_pixel_sampler_config=ActivePixelSamplerConfig(prefetch_scale=5, actray_start_iter=300)
            ),
            model=InstantNGPModelConfig(eval_num_rays_per_chunk=8192),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            }
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 12),
        vis="viewer",
    ),
    description="Nerfstudio method actray on instant-ngp.",
)


method_actray_instant_ngp_bounded = MethodSpecification(
    config=TrainerConfig(
        method_name="method-actray-instant-ngp-bounded",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=ActRayDynamicBatchPipelineConfig(
            propagation_reduce_to=5000,
            stop_exploration_thresh=3.5,
            use_random_background_color=True,
            datamanager=ActRayDataManagerConfig(
                dataparser=BlenderDataParserConfig(),
                train_num_rays_per_batch=8192,
                train_pixel_sampler_config=ActivePixelSamplerConfig(prefetch_scale=5, actray_start_iter=300)
            ),
            model=InstantNGPModelConfig(
                eval_num_rays_per_chunk=8192,
                grid_levels=1,
                alpha_thre=0.0,
                cone_angle=0.0,
                disable_scene_contraction=True,
                near_plane=0.01,
                background_color="white",
            ),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            }
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 12),
        vis="viewer",
    ),
    description="Nerfstudio method actray on instant-ngp-bounded.",
)


method_actray_nerfacto = MethodSpecification(
    config=TrainerConfig(
        method_name="method-actray-nerfacto",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=ActRayVanillaPipelineConfig(
            propagation_reduce_to=-1,
            stop_exploration_thresh=3.5,
            use_random_background_color=True,
            datamanager=ActRayDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3",
                    optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                    scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
                ),
                train_pixel_sampler_config=ActivePixelSamplerConfig(prefetch_scale=5, actray_start_iter=300)
            ),
            model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Nerfstudio method actray on nerfacto.",
)