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
    # Command following curriculum
    # -----------------------------
    command_resample_time_s = 2.0       # resample target commands every 2s
    command_smoothing_tau_s = 0.25      # low-pass filter time constant (slow change)

    command_range_vx = (-1.0, 1.0)
    command_range_vy = (-0.5, 0.5)
    command_range_yaw = (-1.0, 1.0)

    # -----------------------------
    # New skill: bipedal locomotion (stand/walk on hind legs)
    # -----------------------------
    # Toggle this ON to train the new bipedal skill without changing the rest of the pipeline.
    enable_bipedal_skill = True

    # Support feet (stance) and air feet (must stay off the ground)
    bipedal_support_feet = ("RL_foot", "RR_foot")
    bipedal_air_feet = ("FL_foot", "FR_foot")

    # Command ranges for bipedal mode (vy is fixed to 0 for stability).
    # Matches the common setup in bipedal-on-quadruped work: vx in [-0.3, 0.3] with 0.1 bins, yaw-rate in [-1, 1].
    bipedal_command_resample_time_s = 10.0
    bipedal_command_range_vx = (-0.3, 0.3)
    bipedal_command_range_vy = (0.0, 0.0)
    bipedal_command_range_yaw = (-1.0, 1.0)
    bipedal_command_vx_bin = 0.1

    # Stand-up / safety thresholds
    bipedal_grace_steps = 30
    bipedal_front_contact_force_threshold = 20.0  # N

    # Stand gate (used to "unlock" tracking rewards only after standing up)
    bipedal_height_min_m = 0.35
    bipedal_height_max_m = 0.65
    bipedal_track_gate_power = 1.0

    # Front feet must stay above this height (helps prevent "cheating" with extra support)
    bipedal_front_feet_min_height_m = 0.10

    # Extra bipedal reward scales (additive with existing terms)
    # NOTE: front_air and front_contact are penalties -> keep scales NEGATIVE.
    bipedal_gate_reward_scale = 0.0              # purely for logging; keep 0
    bipedal_lift_reward_scale = 0.5
    bipedal_upright_reward_scale = 1.0
    bipedal_front_air_reward_scale = -10.0
    bipedal_front_contact_penalty_scale = -2.0

    # PD control gains
    Kp = 20.0
    Kd = 0.5

    # IMPORTANT: match actuator effort_limit (prevents mismatched clipping/penalty)
    torque_limits = 23.5

    # simulation
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

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=2.5, replicate_physics=True)

    # terrain
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

    # contact sensor (forces)
    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        track_air_time=True,
        debug_vis=False,
    )

    # --- Debug markers: green = command, blue = current ---
    goal_vel_visualizer_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/CommandVel",
        markers={
            "arrow": GREEN_ARROW_X_MARKER_CFG.replace(
                scale=(0.5, 0.1, 0.1),
            )
        },
    )
    current_vel_visualizer_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/CurrentVel",
        markers={
            "arrow": BLUE_ARROW_X_MARKER_CFG.replace(
                scale=(0.5, 0.1, 0.1),
            )
        },
    )

    # -----------------------------
    # Reward scales
    # -----------------------------
    # Part 1 & 2: tracking
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5

    # Part 3: action smoothness
    action_rate_reward_scale = -0.01

    # Part 4: Raibert heuristic
    raibert_heuristic_reward_scale = -5.0

    # Part 6: foot clearance + contacts
    feet_clearance_reward_scale = -30.0
    tracking_contacts_shaped_force_reward_scale = 4.0

    # Part 5 stability penalties
    orient_reward_scale = -5.0
    lin_vel_z_reward_scale = -0.02
    dof_vel_reward_scale = -0.0001
    ang_vel_xy_reward_scale = -0.001

    # Part 6 shaping constants
    feet_clearance_target_m = 0.08
    contact_force_scale = 50.0

    # base height + collision avoidance
    base_height_target_m = 0.32
    base_height_reward_scale = -20.0
    non_foot_contact_reward_scale = -2.0

    # torque regularization (rubric wants tiny scale)
    torque_reward_scale = -1.0e-4
