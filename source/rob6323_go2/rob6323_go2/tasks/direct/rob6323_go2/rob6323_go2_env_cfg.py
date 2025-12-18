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
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfPyramidStairsTerrainCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.actuators import ImplicitActuatorCfg


def _grid_num_rays(size_xy: tuple[float, float], resolution: float) -> int:
    nx = int(size_xy[0] / resolution) + 1
    ny = int(size_xy[1] / resolution) + 1
    return nx * ny


@configclass
class Rob6323Go2EnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 30.0  # enough time to go up + down

    # spaces
    action_scale = 0.25
    action_space = 12

    # -----------------------------
    # Height scanner settings
    # -----------------------------
    height_scan_size = (1.6, 1.0)       # (length, width) meters
    height_scan_resolution = 0.10       # meters
    height_scan_num_rays = _grid_num_rays(height_scan_size, height_scan_resolution)

    # base obs(48) + clock(4) + height scan
    observation_space = 48 + 4 + height_scan_num_rays
    state_space = 0

    # arrows for video
    debug_vis = True

    # terminate if base is too low relative to local ground
    base_height_min = 0.10

    # -----------------------------
    # Commands (STAIRS DEMO: forward only)
    # -----------------------------
    command_resample_time_s = 2.0
    command_smoothing_tau_s = 0.25

    command_range_vx = (0.30, 0.60)
    command_range_vy = (0.0, 0.0)
    command_range_yaw = (0.0, 0.0)

    # PD gains
    Kp = 20.0
    Kd = 0.5
    torque_limits = 23.5

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

    # -----------------------------
    # Terrain (STAIRS ONLY, EASY)
    # -----------------------------
    # NOTE: env_spacing must be >= tile size to avoid overlap
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
            seed=0,
            curriculum=False,
            # each env gets its own 8x8m tile (enough space for up+down)
            size=(8.0, 8.0),
            num_rows=1,
            num_cols=1,
            horizontal_scale=0.10,
            vertical_scale=0.005,
            slope_threshold=0.75,
            sub_terrains={
                "stairs": HfPyramidStairsTerrainCfg(
                    proportion=1.0,                  # ONLY STAIRS
                    step_height_range=(0.03, 0.06),  # easy
                    step_width=0.30,                 # easy
                    platform_width=1.5,              # top platform
                ),
            },
        ),
    )

    # robot
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # disable implicit actuator gains (your explicit PD is in control)
    robot_cfg.actuators["base_legs"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        effort_limit=23.5,
        velocity_limit=30.0,
        stiffness=0.0,
        damping=0.0,
    )

    # scene
    # For training you can increase num_envs; for video set num_envs=1.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=8.0, replicate_physics=True)

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

    # -----------------------------
    # Rewards (tuned to avoid spin/run-in-place)
    # -----------------------------
    lin_vel_reward_scale = 10.0
    yaw_rate_reward_scale = 0.0

    action_rate_reward_scale = -0.05
    raibert_heuristic_reward_scale = -1.0  # keep weak on stairs

    # stability
    orient_reward_scale = -2.0
    lin_vel_z_reward_scale = -0.5
    dof_vel_reward_scale = -0.00005
    ang_vel_xy_reward_scale = -0.02

    # clearance/contact
    feet_clearance_target_m = 0.08
    feet_clearance_reward_scale = -8.0

    tracking_contacts_shaped_force_reward_scale = 1.0
    contact_force_scale = 50.0

    # ground-relative base height (weak)
    base_height_target_m = 0.32
    base_height_reward_scale = -2.0
    non_foot_contact_reward_scale = -1.0

    torque_reward_scale = -1.0e-4
