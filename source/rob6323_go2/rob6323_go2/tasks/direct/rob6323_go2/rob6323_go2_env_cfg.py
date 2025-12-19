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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.actuators import ImplicitActuatorCfg

# Mesh stairs (looks like real steps)
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPyramidStairsTerrainCfg


@configclass
class Rob6323Go2EnvCfg(DirectRLEnvCfg):
    # -----------------------------
    # ENV
    # -----------------------------
    decimation = 4
    episode_length_s = 30.0

    action_scale = 0.25
    action_space = 12
    observation_space = 48 + 4  # base obs + gait clock
    state_space = 0

    debug_vis = True

    # Spawn “before” the stairs pyramid center (in +x direction)
    spawn_offset_x = -3.0

    # Terminate if base too low (simple safeguard)
    base_height_min = 0.10

    # -----------------------------
    # COMMANDS (forward-only so they traverse stairs)
    # -----------------------------
    command_resample_time_s = 2.0
    command_smoothing_tau_s = 0.25

    command_range_vx = (0.6, 1.0)
    command_range_vy = (0.0, 0.0)
    command_range_yaw = (0.0, 0.0)

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
    # MANY ENVS (many bots) — IMPORTANT:
    # num_rows * num_cols MUST be >= num_envs
    # -----------------------------
    num_envs = 64        # change to 256 later if you want (but start with 64 for video)
    grid_rows = 8
    grid_cols = 8        # 8*8 = 64

    # Stairs tile size (meters). Bigger tile => longer climb
    tile_size = 8.0

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=num_envs,
        env_spacing=tile_size,  # keep spacing >= tile_size so tiles don't overlap
        replicate_physics=True,
    )

    # -----------------------------
    # TERRAIN: STAIRS ONLY with MANY STEPS
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
            # size per tile:
            size=(tile_size, tile_size),
            num_rows=grid_rows,
            num_cols=grid_cols,
            slope_threshold=0.75,
            sub_terrains={
                "stairs": MeshPyramidStairsTerrainCfg(
                    proportion=1.0,
                    size=(tile_size, tile_size),
                    border_width=0.25,
                    # ↓ smaller step_width => more steps (what you want)
                    step_width=0.15,
                    # keep height easy enough to learn
                    step_height_range=(0.05, 0.07),
                    # smaller platform = longer staircase visible
                    platform_width=0.8,
                    holes=False,
                ),
            },
        ),
    )

    # -----------------------------
    # ROBOT
    # -----------------------------
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    robot_cfg.actuators["base_legs"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        effort_limit=23.5,
        velocity_limit=30.0,
        stiffness=0.0,
        damping=0.0,
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # -----------------------------
    # VISUAL ARROWS
    # -----------------------------
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    # -----------------------------
    # REWARDS (keep your option-2 structure)
    # -----------------------------
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5

    action_rate_reward_scale = -0.1
    raibert_heuristic_reward_scale = -10.0

    orient_reward_scale = -5.0
    lin_vel_z_reward_scale = -0.02
    dof_vel_reward_scale = -0.0001
    ang_vel_xy_reward_scale = -0.001

    feet_clearance_target_m = 0.12
    feet_clearance_reward_scale = -30.0
    tracking_contacts_shaped_force_reward_scale = 4.0
    contact_force_scale = 50.0

    base_height_target_m = 0.32
    base_height_reward_scale = -20.0
    non_foot_contact_reward_scale = -2.0

    torque_reward_scale = -1.0e-4
