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
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG

# Official rough terrain config (stable import)
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG


def _grid_num_rays(size_xy: tuple[float, float], resolution: float) -> int:
    nx = int(size_xy[0] / resolution) + 1
    ny = int(size_xy[1] / resolution) + 1
    return nx * ny


@configclass
class Rob6323Go2EnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 30.0

    # spaces
    action_scale = 0.50          # more authority => easier to move
    action_space = 12

    # height scanner settings (perceptive locomotion)
    height_scan_size = (1.6, 1.0)
    height_scan_resolution = 0.10
    height_scan_num_rays = _grid_num_rays(height_scan_size, height_scan_resolution)

    # base obs(48) + clock(4) + height_scan
    observation_space = 48 + 4 + height_scan_num_rays
    state_space = 0

    # visualize arrows
    debug_vis = True

    # keep termination simple (avoid “kills” on rough early learning)
    base_height_min = 0.05

    # Commands: FORWARD ONLY so it doesn't spin
    command_resample_time_s = 2.0
    command_smoothing_tau_s = 0.25
    command_range_vx = (0.5, 0.9)
    command_range_vy = (0.0, 0.0)
    command_range_yaw = (0.0, 0.0)

    # Spawn jitter (not same start each episode)
    spawn_xy_jitter = 0.5
    spawn_yaw_jitter = 0.0   # keep 0 so arrows don’t “rotate forever”

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # ONE TYPE of terrain (rough), fixed seed, fixed init level
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG.replace(
            curriculum=False,
            seed=0,
        ),
        max_init_terrain_level=9,   # fixed difficulty; single env => one surface
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # robot
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene (LOW for debugging)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=4.0,
        replicate_physics=True,
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # height scanner
    height_scanner_cfg: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.6)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=height_scan_resolution,
            size=list(height_scan_size),
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # arrows
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    # Rewards: make “move forward” the easiest thing to learn
    lin_vel_reward_scale = 10.0
    yaw_rate_reward_scale = 0.0

    # smoothness (light)
    action_rate_reward_scale = -0.02

    # stability (light early so it doesn’t freeze)
    orient_reward_scale = -1.0
    lin_vel_z_reward_scale = -0.2
    dof_vel_reward_scale = -0.00002
    ang_vel_xy_reward_scale = -0.01

    # shaping (very light)
    raibert_heuristic_reward_scale = -0.5
    feet_clearance_target_m = 0.06
    feet_clearance_reward_scale = -2.0
    tracking_contacts_shaped_force_reward_scale = 0.5
    contact_force_scale = 50.0

    base_height_target_m = 0.32
    base_height_reward_scale = -1.0
    non_foot_contact_reward_scale = -0.5
