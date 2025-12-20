# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# rob6323_go2_env_cfg.py
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.terrains.height_field import hf_terrains_cfg as hf_cfg
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns

from isaaclab.actuators import ImplicitActuatorCfg


@configclass
class Rob6323Go2EnvCfg(DirectRLEnvCfg):
    """Rough-terrain variant (bonus task)."""

    # ---- training practicality ----
    # Rough terrain generation is heavier; start smaller and scale later if you want.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=4.0, replicate_physics=True)

    # ---- height scanner observation ----
    use_height_scanner = True
    height_scanner_num_points = 64     # we downsample the raw ray grid to a fixed size
    height_scan_clip = 1.0             # meters (clip observation)
    observation_space = 48 + 4 + height_scanner_num_points

    # Ray-cast height scanner (grid under/around the base, pointing down)
    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        update_period=0.0,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        attach_yaw_only=True,
        ray_direction=(0.0, 0.0, -1.0),
        ray_far=2.0,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )

    # ---- uneven terrain generator ----
    # NOTE: TerrainImporterCfg in generator mode needs a terrain_generator config. :contentReference[oaicite:6]{index=6}
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
        terrain_generator=TerrainGeneratorCfg(
            seed=0,                 # set !=0 for deterministic terrain layouts
            curriculum=True,
            size=(4.0, 4.0),        # patch size (meters) ~ env_spacing
            num_rows=32,            # 32*32 = 1024 patches => matches num_envs above
            num_cols=32,
            horizontal_scale=0.2,   # heightfield cell size (bigger = fewer cells)
            vertical_scale=0.005,   # height quantization
            slope_threshold=0.75,
            sub_terrains={
                # proportions should sum to 1.0
                "random_uniform": hf_cfg.HfRandomUniformTerrainCfg(
                    proportion=0.40, noise_range=(0.0, 0.05), noise_step=0.01
                ),
                "pyramid_sloped": hf_cfg.HfPyramidSlopedTerrainCfg(
                    proportion=0.20, slope_range=(0.0, 0.35)
                ),
                "pyramid_stairs": hf_cfg.HfPyramidStairsTerrainCfg(
                    proportion=0.20, step_height_range=(0.02, 0.10), step_width=0.30
                ),
                "discrete_obstacles": hf_cfg.HfDiscreteObstaclesTerrainCfg(
                    proportion=0.20,
                    obstacle_height_range=(0.02, 0.15),
                    obstacle_width_range=(0.20, 0.60),
                    num_obstacles=30,
                ),
            },
        ),
    )

    # (optional but recommended on rough terrain)
    base_height_min = 0.18  # now interpreted as "height above ground" (we’ll implement that below)
