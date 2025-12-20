# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# rob6323_go2_env_cfg.py
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

# ------------------------------------------------------------
# Uneven terrain generator (IsaacLab provides ROUGH_TERRAINS_CFG)
# ------------------------------------------------------------
try:
    from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # type: ignore
except Exception:
    # Fallback if your IsaacLab install doesn't ship ROUGH_TERRAINS_CFG.
    try:
        from isaaclab.terrains import TerrainGeneratorCfg
        from isaaclab.terrains.height_field.hf_terrains_cfg import HfRandomUniformTerrainCfg
    except Exception as e:
        raise ImportError(
            "Could not import ROUGH_TERRAINS_CFG (and fallback TerrainGeneratorCfg). "
            "Your IsaacLab install may be missing terrain generator configs."
        ) from e

    ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=10,
        num_cols=20,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        curriculum=True,
        sub_terrains={
            "random_rough": HfRandomUniformTerrainCfg(
                proportion=1.0,
                noise_range=(0.02, 0.10),
                noise_step=0.02,
                border_width=0.25,
            ),
        },
    )

# ------------------------------------------------------------
# Height scanner (RayCaster) config
# GridPattern: resolution=0.1, size=[1.6, 1.0] => 17 * 11 = 187 rays
# ------------------------------------------------------------
HEIGHT_SCAN_RESOLUTION = 0.1
HEIGHT_SCAN_SIZE = (1.6, 1.0)
HEIGHT_SCAN_DIM = (int(round(HEIGHT_SCAN_SIZE[0] / HEIGHT_SCAN_RESOLUTION)) + 1) * (
    int(round(HEIGHT_SCAN_SIZE[1] / HEIGHT_SCAN_RESOLUTION)) + 1
)


@configclass
class Rob6323Go2EnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 20.0

    # expose expected height scan dim for the env (shape-safe observations)
    HEIGHT_SCAN_DIM = HEIGHT_SCAN_DIM

    # spaces
    action_scale = 0.25
    action_space = 12
    observation_space = 48 + 4 + HEIGHT_SCAN_DIM  # base obs + clock + height scan
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
        debug_vis=False,
    )

    # -----------------------------
    # terrain (UNEVENTERRAIN)
    # -----------------------------
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
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
        effort_limit=torque_limits,
        velocity_limit=30.0,
        stiffness=0.0,
        damping=0.0,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # sensors
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=HEIGHT_SCAN_RESOLUTION, size=list(HEIGHT_SCAN_SIZE)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
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

    action_rate_reward_scale = -0.01


    # gait heuristic shaping
    raibert_heuristic_reward_scale = -0.05
    # Part 5 shaping constants
    upright_reward_scale = -1.0
    lin_vel_z_reward_scale = -0.02
    dof_vel_reward_scale = -0.0001
    ang_vel_xy_reward_scale = -0.001

    # Part 6 shaping constants
    feet_clearance_target_m = 0.08
    contact_force_scale = 50.0

    feet_clearance_reward_scale = 0.5
    tracking_contacts_shaped_force_reward_scale = 0.5
    # base height + collision avoidance
    base_height_target_m = 0.32
    base_height_reward_scale = -20.0
    non_foot_contact_reward_scale = -2.0

    # torque regularization
    torque_reward_scale = -1.0e-4
