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
from isaaclab.actuators import ImplicitActuatorCfg


@configclass
class Rob6323Go2EnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 20.0

    # spaces
    action_scale = 0.25
    action_space = 12
    observation_space = 48 + 4  # Added 4 for clock inputs
    state_space = 0

    # IMPORTANT: enable arrows for rubric video check
    debug_vis = True

    base_height_min = 0.05  # Terminate if base is lower than this

    # -----------------------------
    # Command following curric
    # -----------------------------
    command_resample_time_s = 2.0       # resample target commands every 2s
    command_smoothing_tau_s = 0.25      # low-pass filter time constant (slow change)

    command_range_vx = (-1.0, 1.0)
    command_range_vy = (-0.5, 0.5)
    command_range_yaw = (-1.0, 1.0)


    # -----------------------------
    # New skill: bipedal locomotion on hind legs (bonus task)
    # -----------------------------
    # Keep this False to train the original quadruped locomotion.
    enable_bipedal_skill = False

    # In bipedal mode we command forward speed + yaw-rate only (vy forced to 0).
    bipedal_command_resample_time_s = 10.0
    bipedal_command_range_vx = (-0.30, 0.30)
    bipedal_command_range_vy = (0.0, 0.0)
    bipedal_command_range_yaw = (-1.0, 1.0)
    bipedal_command_vx_bin = 0.10

    # Safety termination: after a short grace period, terminate if front feet hit the ground hard.
    bipedal_grace_steps = 30
    bipedal_front_contact_force_threshold = 25.0  # N

    # Shaping targets for bipedal standing
    bipedal_height_min_m = 0.33
    bipedal_height_max_m = 0.65
    bipedal_front_feet_min_height_m = 0.12

    # Bipedal shaping reward scales (additive). Set to 0 to disable any term.
    bipedal_lift_reward_scale = 0.8
    bipedal_upright_reward_scale = 0.6
    bipedal_front_air_reward_scale = -6.0
    bipedal_front_contact_penalty_scale = -1.5

    # PD control gains
    Kp = 20.0
    Kd = 0.5

    # IMPORTANT: match actuator effort_limit (prevents mismatched clipping/penalty)
    torque_limits = 23.5

    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=decimation)

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

    # robot(s)
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # disable implicit actuator gains (so your explicit PD is in control)
    robot_cfg.actuators["base_legs"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        effort_limit=23.5,
        velocity_limit=30.0,
        stiffness=0.0,
        damping=0.0,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=4.0, replicate_physics=True)

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
    )

    # debug arrows / markers (rubric)
    marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Markers",
        markers={
            "command": sim_utils.MarkerCfg(
                prim_path="/Visuals/Markers/command",
                usd_path="path/to/arrow.usd",
                scale=(0.2, 0.2, 0.2),
            ),
            "heading": sim_utils.MarkerCfg(
                prim_path="/Visuals/Markers/heading",
                usd_path="path/to/arrow.usd",
                scale=(0.2, 0.2, 0.2),
            ),
        },
    )

    # reward scales
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5
    action_rate_reward_scale = -0.01
    raibert_heuristic_reward_scale = -0.05

    # Part 5
    orient_reward_scale = -1.0
    lin_vel_z_reward_scale = -2.0
    dof_vel_reward_scale = -0.0001
    ang_vel_xy_reward_scale = -0.001

    # Part 6 shaping constants
    feet_clearance_target_m = 0.08
    contact_force_scale = 50.0

    feet_clearance_reward_scale = -0.2
    tracking_contacts_shaped_force_reward_scale = 0.5

    # base height + collision avoidance
    base_height_target_m = 0.32
    base_height_reward_scale = -20.0
    non_foot_contact_reward_scale = -2.0

    # torque regularization (rubric wants tiny scale)
    torque_reward_scale = -1.0e-4
