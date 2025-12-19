# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
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

from isaaclab.actuators import ImplicitActuatorCfg


@configclass
class Rob6323Go2EnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 20.0

    # safer defaults for bipedal training
    action_scale = 0.12
    action_space = 12
    observation_space = 48 + 4
    state_space = 0

    debug_vis = True
    base_height_min = 0.05

    # command sampling + smoothing
    command_resample_time_s = 2.0
    command_smoothing_tau_s = 0.35

    command_range_vx = (-1.0, 1.0)
    command_range_vy = (-0.5, 0.5)
    command_range_yaw = (-1.0, 1.0)

    # -----------------------------
    # New skill: bipedal locomotion (SAFE MODE)
    # -----------------------------
    enable_bipedal_skill = True

    # bipedal commands (vx only)
    bipedal_command_resample_time_s = 10.0
    bipedal_command_range_vx = (-0.20, 0.20)
    bipedal_command_range_vy = (0.0, 0.0)
    bipedal_command_range_yaw = (-0.7, 0.7)
    bipedal_command_vx_bin = 0.10

    # stand-up + safety
    bipedal_grace_steps = 40
    bipedal_front_contact_force_threshold = 25.0  # N
    bipedal_front_feet_min_height_m = 0.12

    # gate
    bipedal_height_min_m = 0.33
    bipedal_height_max_m = 0.60
    bipedal_track_gate_power = 1.0

    # bipedal reward scales (safer)
    bipedal_gate_reward_scale = 0.0
    bipedal_lift_reward_scale = 0.6
    bipedal_upright_reward_scale = 0.8
    bipedal_front_air_reward_scale = -6.0
    bipedal_front_contact_penalty_scale = -1.5

    # PD gains (safer)
    Kp = 12.0
    Kd = 0.6

    # torque limit consistent with actuator
    torque_limits = 18.0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        disable_contact_processing=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene (reduce env count if you hit OOM / instability)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=2.5, replicate_physics=True)

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
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

    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # disable implicit gains
    robot_cfg.actuators["base_legs"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        effort_limit=torque_limits,
        velocity_limit=30.0,
        stiffness=0.0,
        damping=0.0,
    )

    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        track_air_time=True,
        debug_vis=False,
    )

    goal_vel_visualizer_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/CommandVel",
        markers={"arrow": GREEN_ARROW_X_MARKER_CFG.replace(scale=(0.5, 0.1, 0.1))},
    )
    current_vel_visualizer_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/CurrentVel",
        markers={"arrow": BLUE_ARROW_X_MARKER_CFG.replace(scale=(0.5, 0.1, 0.1))},
    )

    # -----------------------------
    # Reward scales
    # -----------------------------
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5

    action_rate_reward_scale = -0.01
    raibert_heuristic_reward_scale = -3.0

    feet_clearance_reward_scale = -20.0
    tracking_contacts_shaped_force_reward_scale = 3.0

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
