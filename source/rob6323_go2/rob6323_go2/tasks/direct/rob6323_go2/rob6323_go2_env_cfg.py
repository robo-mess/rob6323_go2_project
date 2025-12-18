# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# rob6323_go2_env_cfg.py
# SPDX-License-Identifier: BSD-3-Clause

from ast import pattern
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
from isaaclab.actuators import ImplicitActuatorCfg

# --- NEW: terrain generator + stairs sub-terrain
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfPyramidStairsTerrainCfg

# --- NEW: height scanner (ray caster)
from isaaclab.sensors.ray_caster import RayCasterCfg
from isaaclab.sensors.patterns import GridPatternCfg


def _grid_num_rays(size_xy: tuple[float, float], resolution: float) -> int:
    nx = int(size_xy[0] / resolution) + 1
    ny = int(size_xy[1] / resolution) + 1
    return nx * ny


@configclass
class Rob6323Go2EnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 30.0  # longer so you clearly see up + down

    # spaces
    action_scale = 0.25
    action_space = 12

    # -----------------------------
    # Height scanner settings (NEW)
    # -----------------------------
    height_scan_size = (1.8, 1.2)        # (length, width) in meters
    height_scan_resolution = 0.20        # meters
    height_scan_num_rays = _grid_num_rays(height_scan_size, height_scan_resolution)

    # base obs (48) + clock(4) + height scan
    observation_space = 48 + 4 + height_scan_num_rays
    state_space = 0

    # enable for video (green/blue arrows)
    debug_vis = True

    # Terminate if base is too low relative to local ground (handled in env code)
    base_height_min = 0.10

    # -----------------------------
    # Commands (STAIRS DEMO: forward only)
    # -----------------------------
    command_resample_time_s = 2.0
    command_smoothing_tau_s = 0.25

    # Force straight stair traversal (reliable up+down demo)
    command_range_vx = (0.6, 1.0)
    command_range_vy = (0.0, 0.0)
    command_range_yaw = (0.0, 0.0)

    # PD control gains
    Kp = 20.0
    Kd = 0.5

    # IMPORTANT: match actuator effort_limit (keeps clipping + torque penalty consistent)
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
    # Terrain (STAIRS ONLY)
    # -----------------------------
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
            # one big tile so the robot can traverse up then down within same env
            size=(10.0, 10.0),
            num_rows=1,
            num_cols=1,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            sub_terrains={
                "stairs": HfPyramidStairsTerrainCfg(
                    proportion=1.0,                 # ONLY STAIRS
                    step_height_range=(0.05, 0.10), # difficulty knob
                    step_width=0.25,
                    platform_width=2.0,
                ),
            },
        ),
    )

    # robot(s)
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    print("Actuator keys:", list(UNITREE_GO2_CFG.actuators.keys()))


    # disable implicit actuator gains (so your explicit PD is in control)
    robot_cfg.actuators["base_legs"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        effort_limit=23.5,
        velocity_limit=30.0,
        stiffness=0.0,
        damping=0.0,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # -----------------------------
    # Height scanner config (NEW)
    # -----------------------------
    height_scanner_cfg: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.6)),
        ray_alignment="yaw",
        pattern_cfg=GridPatternCfg(
            resolution=height_scan_resolution,
            size=list(height_scan_size),
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # arrows (command vs current velocity)
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    # -----------------------------
    # reward scales
    # -----------------------------
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5

    action_rate_reward_scale = -0.1
    raibert_heuristic_reward_scale = -10.0

    # stairs need a bit more clearance target
    feet_clearance_target_m = 0.12
    feet_clearance_reward_scale = -30.0

    tracking_contacts_shaped_force_reward_scale = 4.0
    contact_force_scale = 50.0

    # stability penalties
    orient_reward_scale = -5.0
    lin_vel_z_reward_scale = -0.02
    dof_vel_reward_scale = -0.0001
    ang_vel_xy_reward_scale = -0.001

    # base height relative to local ground (enforced in env code)
    base_height_target_m = 0.32
    base_height_reward_scale = -20.0
    non_foot_contact_reward_scale = -2.0

    # torque regularization
    torque_reward_scale = -1.0e-4


