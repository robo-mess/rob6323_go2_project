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

# -----------------------------
# Terrain (IsaacLab-only imports; do NOT fall back to omni.*)
# -----------------------------
from isaaclab.terrains import TerrainImporterCfg
try:
    # newer IsaacLab
    from isaaclab.terrains import TerrainGeneratorCfg
except Exception:
    # some older layouts
    from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg  # type: ignore

try:
    # newer IsaacLab
    from isaaclab.terrains.height_field import HfRandomUniformTerrainCfg
except Exception:
    # older layout
    from isaaclab.terrains.height_field.hf_terrains_cfg import HfRandomUniformTerrainCfg  # type: ignore

from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG

from isaaclab.actuators import ImplicitActuatorCfg


@configclass
class Rob6323Go2EnvCfg(DirectRLEnvCfg):
    # ---------------------------------------------------------------------
    # Core env settings (match your working "latest" setup)
    # ---------------------------------------------------------------------
    decimation = 4
    episode_length_s = 20.0
    debug_vis = False

    # spaces
    action_scale = 0.25
    action_space = 12
    # obs = lin_vel(3)+ang_vel(3)+proj_grav(3)+cmd(3)+dof_pos(12)+dof_vel(12)+actions(12)=48 + clock(4)=52
    observation_space = 52
    state_space = 0

    # command sampling + smoothing (slow changes => fewer catastrophic failures)
    command_resample_time_s = 2.0
    command_smoothing_tau_s = 0.25
    command_range_vx = (-1.0, 1.0)
    command_range_vy = (-0.5, 0.5)
    command_range_yaw = (-1.0, 1.0)

    # PD control gains (your env file uses these)
    Kp = 20.0
    Kd = 0.5

    # IMPORTANT: keep consistent with actuator effort_limit
    torque_limits = 23.5

    # termination
    base_height_min = 0.05

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

    # ---------------------------------------------------------------------
    # UNEVEN TERRAIN (single terrain style, randomized tiles across grid)
    # - This is the ONLY functional change vs flat plane.
    # ---------------------------------------------------------------------
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        collision_group=-1,
        use_terrain_origins=True,  # critical for env_origins to exist
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        terrain_generator=TerrainGeneratorCfg(
            # one 4x4m tile per env (matches env_spacing=4.0)
            size=(4.0, 4.0),
            num_rows=64,
            num_cols=64,          # 64*64 = 4096 envs
            curriculum=False,
            # mild difficulty to reduce collapse (still clearly uneven)
            difficulty_range=(0.0, 0.6),
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            # ONE terrain TYPE, randomized parameters/seed per tile
            sub_terrains={
                "rough_uniform": HfRandomUniformTerrainCfg(
                    proportion=1.0,
                    noise_range=(-0.04, 0.04),  # ~few cm roughness
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

    # disable implicit actuator gains (so your explicit PD in env controls torques)
    robot_cfg.actuators["base_legs"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        effort_limit=23.5,
        velocity_limit=30.0,
        stiffness=0.0,
        damping=0.0,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # contact sensor
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # command arrows (green = command, blue = current)
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    # ---------------------------------------------------------------------
    # reward scales (match what your env code uses)
    # ---------------------------------------------------------------------
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5

    # tutorial gait shaping already in your env
    action_rate_reward_scale = -0.1
    raibert_heuristic_reward_scale = -10.0

    # Part 6: foot clearance + contact shaping
    feet_clearance_reward_scale = -30.0
    tracking_contacts_shaped_force_reward_scale = 4.0

    # Part 5: stability penalties
    orient_reward_scale = -5.0
    lin_vel_z_reward_scale = -0.02
    dof_vel_reward_scale = -0.0001
    ang_vel_xy_reward_scale = -0.001

    # Part 6 shaping constants
    feet_clearance_target_m = 0.08
    contact_force_scale = 50.0

    # rubric extras
    base_height_target_m = 0.32
    base_height_reward_scale = -20.0
    non_foot_contact_reward_scale = -2.0

    # torque regularization (rubric wants tiny scale)
    torque_reward_scale = -1.0e-4
