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
from isaaclab.terrains.height_field import HfRandomUniformTerrainCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers import VisualizationMarkersCfg

# -----------------------------------------------------------------------------
# Marker configs (from IsaacLab defaults)
# -----------------------------------------------------------------------------
from isaaclab.markers import RED_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG


@configclass
class Rob6323Go2EnvCfg(DirectRLEnvCfg):
    """Configuration for ROB6323 Go2 direct RL environment."""

    # -----------------------------
    # simulation
    # -----------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=4,
        disable_contact_processing=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # -----------------------------
    # scene
    # -----------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # -----------------------------
    # termination / safety
    # -----------------------------
    base_height_min = 0.05  # Terminate if base is lower than this

    # -----------------------------
    # robot
    # -----------------------------
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # -----------------------------
    # terrain (UNEVENTERRAIN: single style, randomized tiles)
    # -----------------------------
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        # For generator terrains, env origins come from terrain tile origins (one tile per env below).
        use_terrain_origins=True,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        # One terrain "style" (random uniform roughness), but many different tiles (different difficulty/seed per tile).
        terrain_generator=TerrainGeneratorCfg(
            # Each tile is 4m x 4m so each env has its own patch (matches env_spacing=4.0).
            size=(4.0, 4.0),
            num_rows=64,
            num_cols=64,
            curriculum=False,
            # Keep difficulty mild to avoid early training collapse, but still uneven.
            difficulty_range=(0.0, 0.6),
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            # Terrain type dictionary: keep ONE terrain type, but randomize its parameters via difficulty.
            sub_terrains={
                "rough_uniform": HfRandomUniformTerrainCfg(
                    proportion=1.0,
                    # Peak-to-peak roughness is modest (few cm) so rewards/termination from flat-ground tutorial stay stable.
                    noise_range=(-0.04, 0.04),
                    noise_step=0.01,
                    downsampled_scale=0.5,
                ),
            },
            # Deterministic terrain generation across runs (training reproducibility)
            seed=0,
            use_cache=True,
        ),
        debug_vis=False,
    )

    # robot(s)
    robot: ArticulationCfg = robot_cfg

    # -----------------------------
    # contact sensor
    # -----------------------------
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=1,
        track_air_time=True,
        debug_vis=False,
    )

    # -----------------------------
    # command ranges + smoothing
    # -----------------------------
    command_range_vx = (-1.0, 1.0)
    command_range_vy = (-0.5, 0.5)
    command_range_yaw = (-1.0, 1.0)
    command_resample_time_s = 4.0
    command_smoothing_tau_s = 0.5

    # -----------------------------
    # episode settings
    # -----------------------------
    episode_length_s = 20.0

    # -----------------------------
    # visualization markers
    # -----------------------------
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = RED_ARROW_X_MARKER_CFG.replace(
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
