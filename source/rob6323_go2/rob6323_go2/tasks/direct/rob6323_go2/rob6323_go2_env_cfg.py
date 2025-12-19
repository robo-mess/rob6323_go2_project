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

# Terrain imports (support both IsaacLab namespaces)
try:
    from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
    try:
        from isaaclab.terrains.height_field import HfRandomUniformTerrainCfg
    except Exception:
        from isaaclab.terrains.height_field.hf_terrains_cfg import HfRandomUniformTerrainCfg
except Exception:
    from omni.isaac.lab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
    try:
        from omni.isaac.lab.terrains.height_field import HfRandomUniformTerrainCfg
    except Exception:
        from omni.isaac.lab.terrains.height_field.hf_terrains_cfg import HfRandomUniformTerrainCfg

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

    # -----------------------------
    # Command following curriculum
    # -----------------------------
    command_resample_time_s = 2.0       # resample target commands every 2s
    command_smoothing_tau_s = 0.25      # low-pass filter time constant (slow change)

    command_range_vx = (-1.0, 1.0)
    command_range_vy = (-0.5, 0.5)
    command_range_yaw = (-1.0, 1.0)

    # PD control gains
    Kp = 20.0
    Kd = 0.5

    # IMPORTANT: match actuator effort_limit (prevents mismatched clipping/penalty)
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
    # UNEVEN TERRAIN (single style, randomized tiles)
    # -----------------------------
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        collision_group=-1,
        # Use terrain tile origins as environment origins (one tile per env).
        use_terrain_origins=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        terrain_generator=TerrainGeneratorCfg(
            # Match your env spacing (4.0m): each env gets its own 4x4m rough patch.
            size=(4.0, 4.0),
            num_rows=64,
            num_cols=64,
            curriculum=False,
            # Mild roughness range so training is unlikely to collapse.
            difficulty_range=(0.0, 0.6),
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            # One terrain *type*, randomized tiles (different bumps) across the grid.
            sub_terrains={
                "rough_uniform": HfRandomUniformTerrainCfg(
                    proportion=1.0,
                    noise_range=(-0.04, 0.04),
                    noise_step=0.01,
                    downsampled_scale=0.5,
                ),
            },
            seed=0,
            use_cache=True,
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

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
    # reward scales
    # -----------------------------
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5

    # Part 2 gait shaping
    foot_clearance_reward_scale = 0.4
    foot_slip_reward_scale = -0.25
    feet_air_time_reward_scale = 0.25

    # Part 3 gait symmetry
    symmetry_reward_scale = 0.25

    # Part 4 contact tracking (Raibert heuristic)
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
