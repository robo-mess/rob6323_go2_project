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
    observation_space = 48 + 4 + (height_scanner_num_points if use_height_scanner else 0)

    # termination threshold (avoid instant terminations from tiny contacts/noise)
    termination_base_contact_force = 25.0

    ## Ray-cast height scanner (casts a grid downwards and returns hit points)
    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        update_period=0.0,
        mesh_prim_paths=["/World/ground"],  # one static mesh target (recommended)
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        ray_alignment="yaw",  # ignore roll/pitch for a “heightmap-like” scan
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.20,   # 0.20m grid spacing
            size=(1.6, 1.0),   # (length, width) in meters around the base
        ),
        max_distance=2.0,
    )

    terrain = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFG,
    max_init_terrain_level=5,
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    ),
)

    # (optional but recommended on rough terrain)
    bbase_height_min = 0.25  # now interpreted as "height above ground" (we’ll implement that below)
