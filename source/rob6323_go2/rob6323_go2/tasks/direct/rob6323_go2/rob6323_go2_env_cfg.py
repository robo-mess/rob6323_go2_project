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
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.markers import VisualizationMarkersCfg


# --------------------------------------------------------------------------------------
# Uneven / rough terrain generator config + height scanner config
# --------------------------------------------------------------------------------------

def _load_rough_terrains_cfg():
    """Return a TerrainGeneratorCfg for rough terrain.

    Tries to import the built-in ROUGH_TERRAINS_CFG if available; otherwise builds a compatible
    generator config inline (keeps this assignment robust across IsaacLab versions).
    """
    try:
        from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # type: ignore
        return ROUGH_TERRAINS_CFG
    except Exception:
        try:
            from isaaclab.terrains import TerrainGeneratorCfg
        except Exception:
            from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg  # type: ignore

        from isaaclab.terrains.trimesh.mesh_terrains_cfg import (
            MeshInvertedPyramidStairsTerrainCfg,
            MeshPyramidStairsTerrainCfg,
            MeshRandomGridTerrainCfg,
        )
        from isaaclab.terrains.height_field.hf_terrains_cfg import (
            HfInvertedPyramidSlopedTerrainCfg,
            HfPyramidSlopedTerrainCfg,
            HfRandomUniformTerrainCfg,
        )

        return TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            curriculum=True,
            difficulty_range=(0.0, 1.0),
            use_cache=False,
            sub_terrains={
                "pyramid_stairs": MeshPyramidStairsTerrainCfg(
                    proportion=0.2,
                    step_height_range=(0.05, 0.23),
                    step_width=0.3,
                    platform_width=3.0,
                    border_width=1.0,
                    holes=False,
                ),
                "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(
                    proportion=0.2,
                    step_height_range=(0.05, 0.23),
                    step_width=0.3,
                    platform_width=3.0,
                    border_width=1.0,
                    holes=False,
                ),
                "boxes": MeshRandomGridTerrainCfg(
                    proportion=0.2,
                    grid_width=0.45,
                    grid_height_range=(0.05, 0.2),
                    platform_width=2.0,
                ),
                "random_rough": HfRandomUniformTerrainCfg(
                    proportion=0.2,
                    noise_range=(0.02, 0.10),
                    noise_step=0.02,
                    border_width=0.25,
                ),
                "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
                    proportion=0.1,
                    slope_range=(0.0, 0.4),
                    platform_width=2.0,
                    border_width=0.25,
                ),
                "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
                    proportion=0.1,
                    slope_range=(0.0, 0.4),
                    platform_width=2.0,
                    border_width=0.25,
                ),
            },
        )


ROUGH_TERRAINS_CFG = _load_rough_terrains_cfg()

HEIGHT_SCAN_RESOLUTION = 0.10
HEIGHT_SCAN_SIZE = (1.6, 1.0)  # (length, width) meters


def _grid_num_points(size_xy: tuple[float, float], resolution: float) -> int:
    nx = int(round(size_xy[0] / resolution)) + 1
    ny = int(round(size_xy[1] / resolution)) + 1
    return nx * ny


HEIGHT_SCAN_DIM = _grid_num_points(HEIGHT_SCAN_SIZE, HEIGHT_SCAN_RESOLUTION)


def _make_height_scanner_cfg() -> RayCasterCfg:
    kwargs = dict(
        prim_path="/World/envs/env_.*/Robot",
        update_period=0.02,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        pattern_cfg=patterns.GridPatternCfg(resolution=HEIGHT_SCAN_RESOLUTION, size=list(HEIGHT_SCAN_SIZE)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=100.0,
    )
    # API compatibility across IsaacLab versions
    if hasattr(RayCasterCfg, "ray_alignment"):
        kwargs["ray_alignment"] = "yaw"
    else:
        kwargs["attach_yaw_only"] = True
    return RayCasterCfg(**kwargs)


@configclass
class Rob6323Go2EnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 20.0

    # spaces
    action_scale = 0.25
    action_space = 12
    observation_space = 48 + 4 + HEIGHT_SCAN_DIM  # + height scan
    state_space = 0

    # IMPORTANT: enable arrows for rubric video check
    debug_vis = True

    base_height_min = 0.05  # Terminate if base is lower than this

    # -----------------------------
    # Command following curric
    # -----------------------------
    command_range_vx = (-1.0, 1.0)
    command_range_vy = (-0.5, 0.5)
    command_range_wz = (-1.0, 1.0)
    command_resample_time_s = 2.0
    command_smoothing_tau_s = 0.10

    # -----------------------------
    # PD control
    # -----------------------------
    Kp = 20.0
    Kd = 0.5
    torque_limits = 23.5

    # -----------------------------
    # simulation
    # -----------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0,
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
    # terrain (UNEVENT / ROUGH)
    # -----------------------------
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=1,
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

    # -----------------------------
    # robot
    # -----------------------------
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # -----------------------------
    # sensors
    # -----------------------------
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.0,
        track_air_time=True,
    )

    height_scanner: RayCasterCfg = _make_height_scanner_cfg()

    # -----------------------------
    # scene
    # -----------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # -----------------------------
    # visualization markers
    # -----------------------------
    goal_vel_visualizer_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/velocity_goal",
        markers={
            "arrow": sim_utils.UsdFileCfg(
                usd_path="omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Props/UIElements/arrow_x.usd",
                scale=(0.5, 0.5, 0.5),
            )
        },
    )

    current_vel_visualizer_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/velocity_current",
        markers={
            "arrow": sim_utils.UsdFileCfg(
                usd_path="omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Props/UIElements/arrow_x.usd",
                scale=(0.5, 0.5, 0.5),
            )
        },
    )

    # -----------------------------
    # reward scales
    # -----------------------------
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5
    action_rate_reward_scale = -0.01

    # Part 5 shaping constants
    upright_reward_scale = -1.0
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

    # torque regularization
    torque_reward_scale = -1.0e-4
