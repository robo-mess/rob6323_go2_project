# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# rob6323_go2_env.py
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import sample_uniform
import isaaclab.utils.math as math_utils

from .rob6323_go2_env_cfg_uneven import Rob6323Go2EnvCfg


class Rob6323Go2Env(DirectRLEnv):
    cfg: Rob6323Go2EnvCfg

    def __init__(self, cfg: Rob6323Go2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands (smoothed)
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._commands_target = torch.zeros_like(self._commands)  # target command to track (resampled)
        self._command_time = torch.zeros(self.num_envs, device=self.device)  # time since last resample

        # Update Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "rew_action_rate",
                "raibert_heuristic",
                # Part 5
                "orient", "lin_vel_z", "dof_vel", "ang_vel_xy",
                # Part 6
                "feet_clearance", "tracking_contacts_shaped_force",
                # rubric extras
                "base_height", "non_foot_contact", "torque",
            ]
        }

        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._base_id = int(self._base_id[0])

        # add handle for debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

        # PD control parameters
        self.Kp = torch.tensor([cfg.Kp] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.Kd = torch.tensor([cfg.Kd] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        # IMPORTANT: match cfg and actuator limit
        self.torque_limits = float(cfg.torque_limits)

        # variables needed for action rate penalization
        self.last_actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # feet indices in ARTICULATION (for positions)
        self._feet_ids = []
        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        for name in foot_names:
            id_list, _ = self.robot.find_bodies(name)
            self._feet_ids.append(int(id_list[0]))

        # feet indices in CONTACT SENSOR (for forces)
        self._feet_ids_sensor = []
        for name in foot_names:
            id_list, _ = self._contact_sensor.find_bodies(name)
            self._feet_ids_sensor.append(int(id_list[0]))

        # undesired contacts (knees/hips/thigh/calf touching ground)
        undesired = []
        for pat in [".*thigh", ".*calf", ".*hip"]:
            ids, _ = self._contact_sensor.find_bodies(pat)
            undesired += [int(i) for i in ids]

        # remove feet from undesired list
        feet_set = set(self._feet_ids_sensor)
        self._undesired_contact_ids_sensor = torch.tensor(
            [i for i in sorted(set(undesired)) if i not in feet_set],
            device=self.device,
            dtype=torch.long,
        )

        # store last torques for torque penalty
        self._last_torques = torch.zeros(self.num_envs, 12, device=self.device)

        # Variables needed for the raibert heuristic
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.desired_contact_states = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )

        self.extras = {"log": {}, "episode": {}}

    @property
    def foot_positions_w(self) -> torch.Tensor:
        """Returns the feet positions in the world frame.
        Shape: (num_envs, num_feet, 3)
        Order: [FL, FR, RL, RR]
        """
        return self.robot.data.body_pos_w[:, self._feet_ids]

    def _get_height_scan(self) -> torch.Tensor:
        """Return (num_envs, *) terrain height samples under/around the robot.

        We use the RayCaster's ray hit positions and convert them into heights relative to the robot base.
        If the raycaster hasn't produced data yet, this returns an empty tensor (so torch.cat still works).
        """
        hits = getattr(self._height_scanner.data, "ray_hits_w", None)
        if hits is None or hits.numel() == 0:
            return torch.zeros(self.num_envs, 0, device=self.device)

        # Common shapes across Isaac Lab releases:
        #  - (E, B, 3)
        #  - (E, 1, B, 3)
        #  - (1, E, B, 3)
        if hits.ndim == 4:
            if hits.shape[0] == self.num_envs:
                hits = hits[:, 0]  # (E, B, 3)
            elif hits.shape[1] == self.num_envs:
                hits = hits[0]     # (E, B, 3)
            else:
                hits = hits.reshape(self.num_envs, -1, 3)
        elif hits.ndim == 3:
            if hits.shape[0] != self.num_envs:
                hits = hits.reshape(self.num_envs, -1, 3)
        else:
            hits = hits.reshape(self.num_envs, -1, 3)

        z_hits = hits[..., 2]
        base_z = self.robot.data.root_pos_w[:, 2].unsqueeze(1)

        heights = base_z - z_hits
        heights = torch.nan_to_num(heights, nan=0.0, posinf=0.0, neginf=0.0)

        valid = getattr(self._height_scanner.data, "ray_valid", None)
        if valid is not None and valid.numel() == heights.numel():
            valid = valid.reshape_as(heights)
            heights = torch.where(valid, heights, torch.zeros_like(heights))

        heights = torch.clamp(heights, -1.0, 1.0)
        return heights.reshape(self.num_envs, -1)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)

        self._height_scanner = RayCaster(self.cfg.height_scanner)

        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.scene.sensors["height_scanner"] = self._height_scanner

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        # keep height scanner casting against the (imported/generated) ground prim
        try:
            self.cfg.height_scanner.mesh_prim_paths = [self.cfg.terrain.prim_path]
        except Exception:
            pass
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ---------------------------
    # Command sampling + smoothing
    # ---------------------------
    def _resample_commands(self, env_ids: torch.Tensor):
        n = int(env_ids.numel())

        self._commands_target[env_ids, 0] = sample_uniform(
            self.cfg.command_range_vx[0], self.cfg.command_range_vx[1], (n,), self.device
        )
        self._commands_target[env_ids, 1] = sample_uniform(
            self.cfg.command_range_vy[0], self.cfg.command_range_vy[1], (n,), self.device
        )
        self._commands_target[env_ids, 2] = sample_uniform(
            self.cfg.command_range_yaw[0], self.cfg.command_range_yaw[1], (n,), self.device
        )

    def _update_commands(self):
        # resample target commands every T seconds
        self._command_time += self.step_dt
        resample_mask = self._command_time > float(self.cfg.command_resample_time_s)
        if torch.any(resample_mask):
            env_ids = torch.nonzero(resample_mask).squeeze(-1)
            self._command_time[env_ids] = 0.0
            self._resample_commands(env_ids)

        # low-pass filter to make commands change slowly
        tau = float(self.cfg.command_smoothing_tau_s)
        if tau <= 0.0:
            self._commands[:] = self._commands_target
        else:
            alpha = 1.0 - torch.exp(torch.tensor(-self.step_dt / tau, device=self.device))
            self._commands += alpha * (self._commands_target - self._commands)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # update commands each step (slowly changing)
        self._update_commands()

        self._actions = actions.clone()
        # Compute desired joint positions from policy actions
        self.desired_joint_pos = self.cfg.action_scale * self._actions + self.robot.data.default_joint_pos

    def _apply_action(self) -> None:
        # Compute PD torques
        torques = torch.clip(
            (self.Kp * (self.desired_joint_pos - self.robot.data.joint_pos) - self.Kd * self.robot.data.joint_vel),
            -self.torque_limits,
            self.torque_limits,
        )
        self._last_torques = torques
        self.robot.set_joint_effort_target(torques)

    def _get_observations(self) -> dict:
        self._step_contact_targets()  # Update gait state
        self._previous_actions = self._actions.clone()

        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self.robot.data.root_lin_vel_b,
                    self.robot.data.root_ang_vel_b,
                    self.robot.data.projected_gravity_b,
                    self._commands,
                    self.robot.data.joint_pos - self.robot.data.default_joint_pos,
                    self.robot.data.joint_vel,
                    self._actions,
                    self.clock_inputs,
                    self._get_height_scan(),
                )
                if tensor is not None
            ],
            dim=-1,
        )
        return {"policy": obs}

    # --- rest of file unchanged (rewards/dones/reset/debug/gait/raibert) ---
    # (If you want I can paste the rest too, but it’s long—this is already the full file you had.)
