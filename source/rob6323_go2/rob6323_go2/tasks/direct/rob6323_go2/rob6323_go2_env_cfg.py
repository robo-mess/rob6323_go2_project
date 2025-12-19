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

# SAME import style as your working code (NO omni fallback)
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfRandomUniformTerrainCfg

from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.actuators import ImplicitActuatorCfg


@configclass
class Rob6323Go2EnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 20.0

    # spaces
    action_scale = 0.25
    action_space = 12
    observation_space = 48 + 4  # +4 clock inputs (matches your env_latest obs)
    state_space = 0

    # IMPORTANT: enable arrows for rubric video check
    debug_vis = True

    # termination (keep same as your working latest; not ground-relative here)
    base_height_min = 0.05

    # -----------------------------
    # Commands (UNCHANGED)
    # -----------------------------
    command_resample_time_s = 2.0
    command_smoothing_tau_s = 0.25
    command_range_vx = (-1.0, 1.0)
    command_range_vy = (-0.5, 0.5)
    command_range_yaw = (-1.0, 1.0)

    # PD control gains
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
    # Terrain (UNEVENTERRAIN; ONE style, randomized bumps)
    #
    # CRASH-PROOFING:
    # - use_terrain_origins=True so env_origins exists
    # - num_rows*num_cols MUST equal scene.num_envs
    # - size must match env_spacing
    # -----------------------------
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        collision_group=-1,
        use_terrain_origins=True,
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
            size=(4.0, 4.0),
            num_rows=32,
            num_cols=32,  # 32*32 = 1024 envs (matches scene.num_envs below)
            horizontal_scale=0.10,
            vertical_scale=0.005,
            slope_threshold=0.75,
            sub_terrains={
                "rough_uniform": HfRandomUniformTerrainCfg(
                    proportion=1.0,
                    noise_range=(-0.04, 0.04),  # mild bumps (stable)
                    noise_step=0.01,
                    downsampled_scale=0.5,
                ),
            },
        ),
    )

    # robot(s)
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
    # IMPORTANT: num_envs must match num_rows*num_cols above
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=4.0, replicate_physics=True)

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    # -----------------------------
    # reward scales (UNCHANGED vs your latest)
    # -----------------------------
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5

    action_rate_reward_scale = -0.1
    raibert_heuristic_reward_scale = -10.0
    feet_clearance_reward_scale = -30.0
    tracking_contacts_shaped_force_reward_scale = 4.0

    orient_reward_scale = -5.0
    lin_vel_z_reward_scale = -0.02
    dof_vel_reward_scale = -0.0001
    ang_vel_xy_reward_scale = -0.001

    feet_clearance_target_m = 0.08
    contact_force_scale = 50.0

    base_height_target_m = 0.32
    base_height_reward_scale = -20.0
    non_foot_contact_reward_scale = -2.0

    torque_reward_scale = -1.0e-4
