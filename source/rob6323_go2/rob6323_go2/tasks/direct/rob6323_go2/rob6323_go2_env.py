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

from .rob6323_go2_env_cfg import Rob6323Go2EnvCfg


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
                "orient",
                "lin_vel_z",
                "dof_vel",
                "ang_vel_xy",
                # Part 6
                "feet_clearance",
                "tracking_contacts_shaped_force",
                # rubric extras
                "base_height",
                "non_foot_contact",
                "torque",
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

    def _get_height_scan_obs(self) -> torch.Tensor:
        """Return (num_envs, HEIGHT_SCAN_DIM) terrain height samples around the robot.

        Uses RayCaster hit points (ray_hits_w) relative to the robot base height.
        Enforces a fixed output dimension to prevent PPO observation-shape crashes.
        """
        exp_dim = int(getattr(self.cfg, "HEIGHT_SCAN_DIM", 0))
        if exp_dim <= 0:
            return torch.zeros((self.num_envs, 0), device=self.device)

        if not hasattr(self, "_height_scanner") or self._height_scanner is None:
            return torch.zeros((self.num_envs, exp_dim), device=self.device)

        data = self._height_scanner.data
        hits = getattr(data, "ray_hits_w", None)

        # If hits aren't ready yet, return zeros (safe at episode start)
        if hits is None or hits.numel() == 0:
            return torch.zeros((self.num_envs, exp_dim), device=self.device)

        # Normalize hit tensor to (E, R, 3)
        if hits.ndim == 4:
            # common shapes: (E,1,R,3) or (1,E,R,3)
            if hits.shape[0] == self.num_envs:
                hits = hits[:, 0]
            elif hits.shape[1] == self.num_envs:
                hits = hits[0]
            else:
                hits = hits.reshape(self.num_envs, -1, 3)
        elif hits.ndim == 3:
            if hits.shape[0] != self.num_envs:
                hits = hits.reshape(self.num_envs, -1, 3)
        else:
            hits = hits.reshape(self.num_envs, -1, 3)

        z_hits = hits[..., 2]  # (E,R)
        base_z = self.robot.data.root_pos_w[:, 2:3]  # (E,1)
        heights = base_z - z_hits  # positive if ground is below base

        # Optional validity mask
        valid = getattr(data, "ray_valid", None)
        if valid is not None and valid.numel() == heights.numel():
            valid = valid.reshape_as(heights)
            heights = torch.where(valid, heights, torch.zeros_like(heights))

        heights = torch.nan_to_num(heights, nan=0.0, posinf=0.0, neginf=0.0)
        heights = torch.clamp(heights, -1.0, 1.0).reshape(self.num_envs, -1)

        # Enforce fixed dimension
        if heights.shape[1] != exp_dim:
            if heights.shape[1] > exp_dim:
                heights = heights[:, :exp_dim]
            else:
                pad = torch.zeros((self.num_envs, exp_dim - heights.shape[1]), device=self.device)
                heights = torch.cat([heights, pad], dim=1)

        return heights

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)

        # Height scanner (RayCaster) for uneven terrain perception
        self._height_scanner = RayCaster(self.cfg.height_scanner)

        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.scene.sensors["height_scanner"] = self._height_scanner

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
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
        num = env_ids.numel()
        self._commands_target[env_ids, 0] = sample_uniform(
            self.cfg.command_range_vx[0], self.cfg.command_range_vx[1], (num,), device=self.device
        )
        self._commands_target[env_ids, 1] = sample_uniform(
            self.cfg.command_range_vy[0], self.cfg.command_range_vy[1], (num,), device=self.device
        )
        self._commands_target[env_ids, 2] = sample_uniform(
            self.cfg.command_range_yaw[0], self.cfg.command_range_yaw[1], (num,), device=self.device
        )

    def _update_commands(self):
        # resample target commands every T seconds
        self._command_time += self.step_dt
        resample_mask = self._command_time > self.cfg.command_resample_time_s
        if torch.any(resample_mask):
            env_ids = resample_mask.nonzero(as_tuple=False).flatten()
            self._command_time[env_ids] = 0.0
            self._resample_commands(env_ids)

        # low-pass filter to make commands change slowly
        tau = self.cfg.command_smoothing_tau_s
        if tau <= 0.0:
            self._commands[:] = self._commands_target
        else:
            alpha = 1.0 - torch.exp(torch.tensor(-self.step_dt / tau, device=self.device))
            self._commands += alpha * (self._commands_target - self._commands)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        # Update commands with smoothing
        self._update_commands()

        # Compute PD joint targets
        self.desired_joint_pos = self.cfg.action_scale * self._actions + self.robot.data.default_joint_pos

    def _apply_action(self):
        # PD control in torque space (your original behavior)
        torques = self.Kp * (self.desired_joint_pos - self.robot.data.joint_pos) - self.Kd * self.robot.data.joint_vel
        torques = torch.clip(torques, -self.torque_limits, self.torque_limits)

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
                    self._get_height_scan_obs(),
                )
                if tensor is not None
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # tracking rewards
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)

        yaw_rate_error = torch.square(self._commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)

        # action rate penalization
        rew_action_rate = torch.sum(torch.square(self._actions - self.last_actions[:, :, 0]), dim=1) * (
            self.cfg.action_scale**2
        )
        rew_action_rate += torch.sum(
            torch.square(self._actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]), dim=1
        ) * (self.cfg.action_scale**2)

        self.last_actions = torch.roll(self.last_actions, 1, 2)
        self.last_actions[:, :, 0] = self._actions[:]

        rew_raibert_heuristic = self._reward_raibert_heuristic()

        # Part 5 shaping
        upright = torch.square(1.0 - self.robot.data.projected_gravity_b[:, 2])
        lin_vel_z = torch.square(self.robot.data.root_lin_vel_b[:, 2])
        dof_vel = torch.sum(torch.square(self.robot.data.joint_vel), dim=1)
        ang_vel_xy = torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=1)

        # Part 6 shaping terms
        feet_clearance = self._reward_feet_clearance()
        tracking_contacts_force = self._reward_tracking_contacts_shaped_force()

        # rubric extras
        base_height = torch.square(self.robot.data.root_pos_w[:, 2] - self.cfg.base_height_target_m)
        non_foot_contact = self._reward_non_foot_contact()
        torque = torch.sum(torch.square(self._last_torques), dim=1)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "rew_action_rate": rew_action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "raibert_heuristic": rew_raibert_heuristic * self.cfg.raibert_heuristic_reward_scale * self.step_dt,
            "orient": upright * self.cfg.upright_reward_scale * self.step_dt,
            "lin_vel_z": lin_vel_z * self.cfg.lin_vel_z_reward_scale * self.step_dt,
            "dof_vel": dof_vel * self.cfg.dof_vel_reward_scale * self.step_dt,
            "ang_vel_xy": ang_vel_xy * self.cfg.ang_vel_xy_reward_scale * self.step_dt,
            "feet_clearance": feet_clearance * self.cfg.feet_clearance_reward_scale * self.step_dt,
            "tracking_contacts_shaped_force": tracking_contacts_force
            * self.cfg.tracking_contacts_shaped_force_reward_scale
            * self.step_dt,
            "base_height": base_height * self.cfg.base_height_reward_scale * self.step_dt,
            "non_foot_contact": non_foot_contact * self.cfg.non_foot_contact_reward_scale * self.step_dt,
            "torque": torque * self.cfg.torque_reward_scale * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        base_height = self.robot.data.root_pos_w[:, 2]
        cstr_base_height_min = base_height < self.cfg.base_height_min
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        net_contact_forces = self._contact_sensor.data.net_forces_w_history  # (N,H,B,3)
        base_id = int(self._base_id)

        base_force_mag_hist = torch.linalg.norm(net_contact_forces[:, :, base_id, :], dim=-1)  # (N,H)
        cstr_termination_contacts = torch.any(base_force_mag_hist > 1.0, dim=1)

        cstr_upsidedown = self.robot.data.projected_gravity_b[:, 2] > 0
        died = cstr_termination_contacts | cstr_upsidedown | cstr_base_height_min
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._resample_commands(torch.tensor(env_ids, device=self.device, dtype=torch.long))

        for key in self._episode_sums.keys():
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = {}

        # log episode sums
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            self.extras["log"][key] = episodic_sum_avg / self.max_episode_length_s

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_goal_vel_vis"):
                self._goal_vel_vis = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self._current_vel_vis = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
        else:
            if hasattr(self, "_goal_vel_vis"):
                self._goal_vel_vis = None
                self._current_vel_vis = None

    def _debug_vis_callback(self, event):
        if not self.cfg.debug_vis:
            return
        # goal velocity marker
        pos = self.robot.data.root_pos_w
        quat_goal = self._resolve_xy_velocity_to_arrow(self._commands[:, :2])
        self._goal_vel_vis.visualize(pos, quat_goal)

        # current velocity marker
        quat_cur = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        self._current_vel_vis.visualize(pos, quat_cur)

    def _resolve_xy_velocity_to_arrow(self, xy_vel_b: torch.Tensor) -> torch.Tensor:
        # Arrow points along +X, so compute yaw from velocity vector
        yaw = torch.atan2(xy_vel_b[:, 1], xy_vel_b[:, 0])
        zeros = torch.zeros_like(yaw)
        quat = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
        return quat

    def _step_contact_targets(self):
        # trot gait with period 1s
        gait_period = 1.0
        self.gait_indices = torch.remainder(self.gait_indices + self.step_dt / gait_period, 1.0)

        phase = 2 * np.pi * self.gait_indices
        # clock inputs (4 dims)
        self.clock_inputs[:, 0] = torch.sin(phase)
        self.clock_inputs[:, 1] = torch.cos(phase)
        self.clock_inputs[:, 2] = torch.sin(phase + np.pi)
        self.clock_inputs[:, 3] = torch.cos(phase + np.pi)

        # desired contacts (trot: FL+RR then FR+RL)
        self.desired_contact_states[:, 0] = (self.clock_inputs[:, 0] > 0).float()  # FL
        self.desired_contact_states[:, 3] = (self.clock_inputs[:, 0] > 0).float()  # RR
        self.desired_contact_states[:, 1] = (self.clock_inputs[:, 2] > 0).float()  # FR
        self.desired_contact_states[:, 2] = (self.clock_inputs[:, 2] > 0).float()  # RL

    def _reward_raibert_heuristic(self):
        # current footsteps in body frame
        cur_footsteps_translated = self.foot_positions_w - self.robot.data.root_pos_w.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros_like(cur_footsteps_translated)

        for i in range(4):
            footsteps_in_body_frame[:, i, :] = math_utils.quat_apply_yaw(
                math_utils.quat_conjugate(self.robot.data.root_quat_w), cur_footsteps_translated[:, i, :]
            )

        desired_stance_width = 0.25
        desired_ys_nom = torch.tensor(
            [desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2],
            device=self.device,
        ).unsqueeze(0)

        desired_stance_length = 0.25
        desired_xs_nom = torch.tensor(
            [desired_stance_length / 2, desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2],
            device=self.device,
        ).unsqueeze(0)

        phases = torch.abs(1.0 - (self.gait_indices.unsqueeze(1) * 2.0)) * 1.0 - 0.5
        frequencies = torch.tensor([1.0], device=self.device)

        x_vel_des = self._commands[:, 0:1]
        y_vel_des = self._commands[:, 1:2]

        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])
        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))
        return reward

    # NOTE: these 3 were in your base env file; keeping them intact
    def _reward_feet_clearance(self) -> torch.Tensor:
        foot_z = self.foot_positions_w[..., 2]
        swing = 1.0 - self.desired_contact_states
        target = float(self.cfg.feet_clearance_target_m)
        err = torch.square(foot_z - target)
        return -torch.sum(err * swing, dim=1)

    def _reward_tracking_contacts_shaped_force(self) -> torch.Tensor:
        forces = self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor, :]
        fmag = torch.linalg.norm(forces, dim=-1)
        contact_strength = torch.tanh(fmag / float(self.cfg.contact_force_scale))
        match = 1.0 - torch.abs(contact_strength - self.desired_contact_states)
        return torch.mean(match, dim=1)

    def _reward_non_foot_contact(self) -> torch.Tensor:
        if self._undesired_contact_ids_sensor.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        forces = self._contact_sensor.data.net_forces_w[:, self._undesired_contact_ids_sensor, :]
        bad_mag = torch.linalg.norm(forces, dim=-1).sum(dim=1)
        bad = torch.tanh(bad_mag)
        return self.cfg.non_foot_contact_reward_scale * bad
