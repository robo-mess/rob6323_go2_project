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
from isaaclab.actuators import ImplicitActuatorCfg

# Official rough terrain config (stable)
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG


def _grid_num_rays(size_xy: tuple[float, float], resolution: float) -> int:
    nx = int(size_xy[0] / resolution) + 1
    ny = int(size_xy[1] / resolution) + 1
    return nx * ny


@configclass
class Rob6323Go2EnvCfg(DirectRLEnvCfg):
    # -----------------------------
    # ENV
    # -----------------------------
    decimation = 4
    episode_length_s = 30.0  # longer so it traverses multiple patches

    action_scale = 0.25
    action_space = 12

    # -----------------------------
    # Height scanner
    # -----------------------------
    height_scan_size = (1.6, 1.0)
    height_scan_resolution = 0.10
    height_scan_num_rays = _grid_num_rays(height_scan_size, height_scan_resolution)

    observation_space = 48 + 4 + height_scan_num_rays
    state_space = 0

    debug_vis = True

    # terminate if base too low relative to local ground
    base_height_min = 0.12

    # -----------------------------
    # Commands (forward-only = no spin)
    # -----------------------------
    command_resample_time_s = 2.0
    command_smoothing_tau_s = 0.25

    command_range_vx = (0.4, 0.8)
    command_range_vy = (0.0, 0.0)
    command_range_yaw = (0.0, 0.0)

    # -----------------------------
    # Spawn behavior (IMPORTANT)
    # - start near the beginning of the 4-tile strip
    # - randomize XY so start isn't identical every reset
    # -----------------------------
    terrain_tile_size = 8.0
    terrain_num_rows = 4   # 4 tiles in +x (progression)
    terrain_num_cols = 1

    # Put spawn near the "left" side of the strip (approx)
    spawn_base_x = -0.5 * terrain_num_rows * terrain_tile_size + 1.5
    spawn_base_y = 0.0

    # Randomization (per reset)
    spawn_xy_jitter = 0.5     # +/- meters
    spawn_yaw_jitter = 0.15   # +/- radians (small)

    # PD
    Kp = 20.0
    Kd = 0.5
    torque_limits = 23.5

    # -----------------------------
    # SIM
    # -----------------------------
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
    # Terrain: official ROUGH_TERRAINS_CFG,
    # but stretched into a 4x1 strip so one robot walks across
    # multiple terrain patches in one episode.
    # -----------------------------
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG.replace(
            # make a strip of tiles
            size=(terrain_tile_size, terrain_tile_size),
            num_rows=terrain_num_rows,
            num_cols=terrain_num_cols,
            curriculum=False,     # fixed layout, no curriculum
            seed=0,
        ),
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

    # -----------------------------
    # Robot
    # -----------------------------
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    robot_cfg.actuators["base_legs"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        effort_limit=23.5,
        velocity_limit=30.0,
        stiffness=0.0,
        damping=0.0,
    )

    # -----------------------------
    # Scene (KEEP SMALL for debugging)
    # -----------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,          # <<< small so you can verify it works
        env_spacing=4.0,
        replicate_physics=True,
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # Height scanner
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

    # Arrows
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    # -----------------------------
    # Rewards (tuned to actually move forward)
    # -----------------------------
    lin_vel_reward_scale = 8.0
    yaw_rate_reward_scale = 0.0

    action_rate_reward_scale = -0.05
    raibert_heuristic_reward_scale = -1.0

    orient_reward_scale = -2.0
    lin_vel_z_reward_scale = -0.5
    dof_vel_reward_scale = -0.00005
    ang_vel_xy_reward_scale = -0.02

    feet_clearance_target_m = 0.06
    feet_clearance_reward_scale = -5.0

    tracking_contacts_shaped_force_reward_scale = 1.0
    contact_force_scale = 50.0

    base_height_target_m = 0.32
    base_height_reward_scale = -2.0
    non_foot_contact_reward_scale = -1.0

    torque_reward_scale = -1.0e-4
