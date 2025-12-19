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
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import sample_uniform
import isaaclab.utils.math as math_utils

from .rob6323_go2_env_cfg import Rob6323Go2EnvCfg


class Rob6323Go2Env(DirectRLEnv):
    cfg: Rob6323Go2EnvCfg

    def __init__(self, cfg: Rob6323Go2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.last_actions = torch.zeros((self.num_envs, self.cfg.action_space, 2), device=self.device)

        # Command buffers
        self._commands = torch.zeros((self.num_envs, 3), device=self.device)
        self._commands_target = torch.zeros((self.num_envs, 3), device=self.device)
        self._command_time = torch.zeros((self.num_envs,), device=self.device)

        # Gait clock + desired contact states (FL, FR, RL, RR)
        self.clock_inputs = torch.zeros((self.num_envs, 4), device=self.device)
        self.desired_contact_states = torch.zeros((self.num_envs, 4), device=self.device)
        self.gait_indices = torch.zeros((self.num_envs,), device=self.device)
        self.foot_indices = torch.zeros((self.num_envs, 4), device=self.device)

        # Reward logging
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
                # bipedal skill shaping
                "bipedal_lift", "bipedal_upright", "bipedal_front_air", "bipedal_front_contact",
            ]
        }

        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._base_id = int(self._base_id[0])

        # Feet ids in contact sensor order
        self._feet_ids_sensor, _ = self._contact_sensor.find_bodies(".*_foot")
        self._feet_ids_sensor = self._feet_ids_sensor.to(dtype=torch.long, device=self.device)

        # Undesired contact bodies (everything except feet + base)
        self._undesired_contact_ids_sensor, _ = self._contact_sensor.find_bodies(".*")
        # remove feet + base indices from undesired list (safe even if empty)
        mask = torch.ones_like(self._undesired_contact_ids_sensor, dtype=torch.bool)
        for i in torch.cat([self._feet_ids_sensor, torch.tensor([self._base_id], device=self.device)]):
            mask = mask & (self._undesired_contact_ids_sensor != i)
        self._undesired_contact_ids_sensor = self._undesired_contact_ids_sensor[mask].to(dtype=torch.long, device=self.device)

        # Torque limits
        self.torque_limits = float(getattr(self.cfg, "torque_limits", 23.5))
        self._last_torques = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        # debug markers
        if self.cfg.debug_vis:
            self._markers = VisualizationMarkers(self.cfg.marker_cfg)
        else:
            self._markers = None

    def _is_bipedal(self) -> bool:
        return bool(getattr(self.cfg, "enable_bipedal_skill", False))

    def _setup_scene(self):
        # terrain + robot + sensors
        self.robot = Articulation(self.cfg.robot_cfg)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)

        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.cfg.terrain.func("/World/ground", self.cfg.terrain)

        # lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ---------------------------
    # Command sampling + smoothing
    # ---------------------------
    def _resample_commands(self, env_ids: torch.Tensor):
        n = int(env_ids.numel())

        if self._is_bipedal():
            # forward speed + yaw-rate; no lateral command
            vx_lo, vx_hi = self.cfg.bipedal_command_range_vx
            bin_sz = float(getattr(self.cfg, "bipedal_command_vx_bin", 0.0))
            if bin_sz and bin_sz > 0.0:
                n_bins = int(round((vx_hi - vx_lo) / bin_sz)) + 1
                idx = torch.randint(0, n_bins, (n,), device=self.device)
                self._commands_target[env_ids, 0] = float(vx_lo) + idx.to(torch.float32) * bin_sz
            else:
                self._commands_target[env_ids, 0] = sample_uniform(vx_lo, vx_hi, (n,), self.device)

            self._commands_target[env_ids, 1] = 0.0
            self._commands_target[env_ids, 2] = sample_uniform(
                self.cfg.bipedal_command_range_yaw[0], self.cfg.bipedal_command_range_yaw[1], (n,), self.device
            )
        else:
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

        resample_T = float(self.cfg.bipedal_command_resample_time_s) if self._is_bipedal() else float(self.cfg.command_resample_time_s)
        resample_mask = self._command_time > resample_T
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

        # hard-enforce vy=0 in bipedal mode (prevents sideways "cheating")
        if self._is_bipedal():
            self._commands[:, 1] = 0.0
            self._commands_target[:, 1] = 0.0

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions[:] = torch.clamp(actions, -1.0, 1.0)

        # PD targets (relative to default)
        joint_pos_target = self.robot.data.default_joint_pos + self._actions * self.cfg.action_scale
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel

        # Explicit PD torque
        torques = self.cfg.Kp * (joint_pos_target - joint_pos) - self.cfg.Kd * joint_vel
        torques = torch.clamp(torques, -self.torque_limits, self.torque_limits)
        self._last_torques = torques

        self.robot.set_joint_effort_target(torques)

        # commands + gait clocks
        self._update_commands()
        self._step_contact_targets()

        # debug arrows
        if self._markers is not None:
            self._update_debug_arrows()

    def _apply_action(self):
        # already applied as torques in _pre_physics_step
        pass

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # typical observation: base vel, ang vel, projected gravity, joint pos/vel, commands, last actions, clock
        obs = torch.cat(
            [
                self.robot.data.root_lin_vel_b,
                self.robot.data.root_ang_vel_b,
                self.robot.data.projected_gravity_b,
                self.robot.data.joint_pos - self.robot.data.default_joint_pos,
                self.robot.data.joint_vel,
                self._commands,
                self._actions,
                self.clock_inputs,
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _reward_raibert_heuristic(self) -> torch.Tensor:
        # (baseline implementation, plus a safe bipedal slice)
        hip_offset = torch.tensor([0.19, 0.05, 0.0], device=self.device)  # approx
        desired_speed = self._commands[:, 0]
        desired_yaw = self._commands[:, 2]

        base_quat = self.robot.data.root_quat_w
        base_pos = self.robot.data.root_pos_w
        foot_pos = self.foot_positions_w  # (N,4,3)

        # footsteps in body frame
        footsteps_in_body_frame = math_utils.quat_rotate_inverse(
            base_quat.unsqueeze(1).repeat(1, 4, 1),
            foot_pos - base_pos.unsqueeze(1),
        )

        desired_xs_nom = desired_speed * 0.2
        desired_ys_nom = torch.zeros_like(desired_xs_nom)

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(1).repeat(1, 4, 1),
                                                  desired_ys_nom.unsqueeze(1).repeat(1, 4, 1)), dim=2)

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

        if self._is_bipedal():
            err_raibert_heuristic = err_raibert_heuristic[:, 2:4, :]

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))
        return reward

    def _get_rewards(self) -> torch.Tensor:
        # ----------------------------
        # tracking rewards
        # ----------------------------
        if self._is_bipedal():
            # Track planar velocity in the yaw frame (ignores pitch/roll), which is more stable for bipedal.
            base_quat_w = self.robot.data.root_quat_w
            w, x, y, z = base_quat_w.unbind(dim=-1)
            yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            cy, sy = torch.cos(yaw), torch.sin(yaw)

            lin_vel_w = math_utils.quat_apply(base_quat_w, self.robot.data.root_lin_vel_b)
            vx_w, vy_w = lin_vel_w[:, 0], lin_vel_w[:, 1]
            vel_yaw_x = cy * vx_w + sy * vy_w
            vel_yaw_y = -sy * vx_w + cy * vy_w
            lin_vel_meas_xy = torch.stack([vel_yaw_x, vel_yaw_y], dim=-1)

            ang_vel_w = math_utils.quat_apply(base_quat_w, self.robot.data.root_ang_vel_b)
            yaw_rate_meas = ang_vel_w[:, 2]
        else:
            lin_vel_meas_xy = self.robot.data.root_lin_vel_b[:, :2]
            yaw_rate_meas = self.robot.data.root_ang_vel_b[:, 2]

        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - lin_vel_meas_xy), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)

        yaw_rate_error = torch.square(self._commands[:, 2] - yaw_rate_meas)
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

        # ============================================================
        # Part 5: stability (upright + smooth base)
        # ============================================================
        pg = self.robot.data.projected_gravity_b          # (N,3)
        lin = self.robot.data.root_lin_vel_b              # (N,3)
        ang = self.robot.data.root_ang_vel_b              # (N,3)
        qd  = self.robot.data.joint_vel                   # (N,12)

        # In bipedal mode, don't punish pitch (the body will pitch up). Only punish roll.
        if self._is_bipedal():
            rew_orient = (pg[:, 1] ** 2)
        else:
            rew_orient = (pg[:, 0] ** 2 + pg[:, 1] ** 2)

        rew_lin_vel_z  = (lin[:, 2] ** 2)
        rew_dof_vel    = (qd ** 2).sum(dim=1)
        rew_ang_vel_xy = (ang[:, 0] ** 2 + ang[:, 1] ** 2)

        # ============================================================
        # Part 6: foot clearance + contact shaping
        # ============================================================
        foot_pos_w = self.foot_positions_w               # (N,4,3) [FL,FR,RL,RR]
        foot_h = foot_pos_w[:, :, 2]                     # (N,4)

        swing_w = 1.0 - self.desired_contact_states      # (N,4)

        clearance = float(self.cfg.feet_clearance_target_m)
        err = torch.clamp(clearance - foot_h, min=0.0)
        rew_feet_clearance = (swing_w * (err ** 2)).sum(dim=1)

        forces_w = self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor, :]  # (N,4,3)
        fmag = torch.linalg.norm(forces_w, dim=-1)                                       # (N,4)
        contact_strength = torch.tanh(fmag / float(self.cfg.contact_force_scale))        # (N,4)

        match = 1.0 - torch.abs(contact_strength - self.desired_contact_states)         # (N,4)
        rew_track_contacts = match.mean(dim=1)                                           # (N,)

        # ============================================================
        # nominal base height + avoid knee/hip ground hits
        # ============================================================
        base_h = self.robot.data.root_pos_w[:, 2]
        rew_base_height = (base_h - float(self.cfg.base_height_target_m)) ** 2

        if self._undesired_contact_ids_sensor.numel() > 0:
            bad_forces = self._contact_sensor.data.net_forces_w[:, self._undesired_contact_ids_sensor, :]  # (N,K,3)
            bad_mag = torch.linalg.norm(bad_forces, dim=-1).sum(dim=1)                                      # (N,)
            rew_non_foot_contact = torch.tanh(bad_mag / float(self.cfg.contact_force_scale))
        else:
            rew_non_foot_contact = torch.zeros(self.num_envs, device=self.device)

        # ============================================================
        # torque magnitude regularization
        # ============================================================
        tau = self._last_torques
        rew_torque = ((tau / self.torque_limits) ** 2).sum(dim=1)

        # ============================================================
        # bipedal shaping (adds to the original reward; does not change the baseline behavior)
        # ============================================================
        bipedal_lift = torch.zeros(self.num_envs, device=self.device)
        bipedal_upright = torch.zeros(self.num_envs, device=self.device)
        bipedal_front_air = torch.zeros(self.num_envs, device=self.device)
        bipedal_front_contact = torch.zeros(self.num_envs, device=self.device)

        if self._is_bipedal():
            hmin = float(self.cfg.bipedal_height_min_m)
            hmax = float(self.cfg.bipedal_height_max_m)
            bipedal_lift = torch.clamp((base_h - hmin) / (hmax - hmin + 1.0e-6), 0.0, 1.0)

            roll_cost = (pg[:, 1] ** 2)
            bipedal_upright = torch.exp(-roll_cost / 0.05)

            front_h = foot_h[:, 0:2]
            front_low = torch.clamp(float(self.cfg.bipedal_front_feet_min_height_m) - front_h, min=0.0)
            bipedal_front_air = torch.sum(front_low ** 2, dim=1)

            bipedal_front_contact = torch.sum(contact_strength[:, 0:2], dim=1)

            # Gate tracking until it stands up, otherwise it will thrash on the floor and explode.
            gate = bipedal_lift * bipedal_upright
            lin_vel_error_mapped = lin_vel_error_mapped * gate
            yaw_rate_error_mapped = yaw_rate_error_mapped * gate

        # ----------------------------
        # final rewards dict (IMPORTANT: * step_dt)
        # ----------------------------
        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,

            "rew_action_rate": rew_action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "raibert_heuristic": rew_raibert_heuristic * self.cfg.raibert_heuristic_reward_scale * self.step_dt,

            "orient": rew_orient * self.cfg.orient_reward_scale * self.step_dt,
            "lin_vel_z": rew_lin_vel_z * self.cfg.lin_vel_z_reward_scale * self.step_dt,
            "dof_vel": rew_dof_vel * self.cfg.dof_vel_reward_scale * self.step_dt,
            "ang_vel_xy": rew_ang_vel_xy * self.cfg.ang_vel_xy_reward_scale * self.step_dt,

            "feet_clearance": rew_feet_clearance * self.cfg.feet_clearance_reward_scale * self.step_dt,
            "tracking_contacts_shaped_force": rew_track_contacts * self.cfg.tracking_contacts_shaped_force_reward_scale * self.step_dt,

            "base_height": rew_base_height * self.cfg.base_height_reward_scale * self.step_dt,
            "non_foot_contact": rew_non_foot_contact * self.cfg.non_foot_contact_reward_scale * self.step_dt,
            "torque": rew_torque * self.cfg.torque_reward_scale * self.step_dt,

            # bipedal skill shaping
            "bipedal_lift": bipedal_lift * float(getattr(self.cfg, "bipedal_lift_reward_scale", 0.0)) * self.step_dt,
            "bipedal_upright": bipedal_upright * float(getattr(self.cfg, "bipedal_upright_reward_scale", 0.0)) * self.step_dt,
            "bipedal_front_air": bipedal_front_air * float(getattr(self.cfg, "bipedal_front_air_reward_scale", 0.0)) * self.step_dt,
            "bipedal_front_contact": bipedal_front_contact * float(getattr(self.cfg, "bipedal_front_contact_penalty_scale", 0.0)) * self.step_dt,
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

        # bipedal safety: terminate if front feet contact hard (after a small grace period)
        if self._is_bipedal():
            grace = int(getattr(self.cfg, "bipedal_grace_steps", 30))
            after_grace = self.episode_length_buf > grace

            front_forces = self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor[0:2], :]  # (N,2,3)
            front_mag = torch.linalg.norm(front_forces, dim=-1).max(dim=1).values
            front_contact = front_mag > float(getattr(self.cfg, "bipedal_front_contact_force_threshold", 25.0))

            died = died | (after_grace & front_contact)

        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, device=self.device)
        env_ids = env_ids.to(dtype=torch.long, device=self.device)

        # reset robot state
        self.robot.reset(env_ids)

        # reset buffers
        self._actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0

        self._command_time[env_ids] = 0.0
        self._commands[env_ids] = 0.0
        self._commands_target[env_ids] = 0.0
        self._resample_commands(env_ids)

        self.gait_indices[env_ids] = 0.0
        self.clock_inputs[env_ids] = 0.0
        self.desired_contact_states[env_ids] = 0.0

        # reset episode sums
        for k in self._episode_sums:
            self._episode_sums[k][env_ids] = 0.0

    @property
    def foot_positions_w(self) -> torch.Tensor:
        # (N,4,3) from rigid body states (expects order matches feet ids)
        foot_states = self.robot.data.body_state_w[:, self._feet_ids_sensor, :]
        return foot_states[..., 0:3]

    def _step_contact_targets(self):
        frequencies = 3.0
        phases = 0.5
        offsets = 0.0
        bounds = 0.0
        durations = 0.5 * torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
        self.gait_indices = torch.remainder(self.gait_indices + self.step_dt * frequencies, 1.0)

        foot_indices = [
            self.gait_indices + phases + offsets + bounds,  # FL
            self.gait_indices + offsets,                    # FR
            self.gait_indices + bounds,                     # RL
            self.gait_indices + phases,                     # RR
        ]

        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

        for idxs in foot_indices:
            stance_idxs = torch.remainder(idxs, 1) < durations
            swing_idxs = torch.remainder(idxs, 1) > durations

        smoothing_multiplier_FL = torch.zeros((self.num_envs,), device=self.device)
        smoothing_multiplier_FR = torch.zeros((self.num_envs,), device=self.device)
        smoothing_multiplier_RL = torch.zeros((self.num_envs,), device=self.device)
        smoothing_multiplier_RR = torch.zeros((self.num_envs,), device=self.device)

        for i, idxs in enumerate(foot_indices):
            stance_idxs = torch.remainder(idxs, 1) < durations
            swing_idxs = torch.remainder(idxs, 1) > durations
            smoothing = torch.zeros((self.num_envs,), device=self.device)

            smoothing[stance_idxs] = 0.5 * torch.tanh(
                10 * (torch.remainder(idxs[stance_idxs], 1) / durations[stance_idxs] - 0.5)
            ) + 0.5

            smoothing[swing_idxs] = 0.5 * torch.tanh(
                10 * (0.5 - (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) / (1 - durations[swing_idxs]))
            ) + 0.5

            if i == 0:
                smoothing_multiplier_FL = smoothing
            elif i == 1:
                smoothing_multiplier_FR = smoothing
            elif i == 2:
                smoothing_multiplier_RL = smoothing
            else:
                smoothing_multiplier_RR = smoothing

        if self._is_bipedal():
            # Only hind legs have a gait clock; front legs are "in the air".
            self.clock_inputs[:, 0] = 0.0
            self.clock_inputs[:, 1] = 0.0
            self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
            self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])
        else:
            self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
            self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
            self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
            self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

        if self._is_bipedal():
            self.desired_contact_states[:, 0] = 0.0
            self.desired_contact_states[:, 1] = 0.0
            self.desired_contact_states[:, 2] = smoothing_multiplier_RL
            self.desired_contact_states[:, 3] = smoothing_multiplier_RR
        else:
            self.desired_contact_states[:, 0] = smoothing_multiplier_FL
            self.desired_contact_states[:, 1] = smoothing_multiplier_FR
            self.desired_contact_states[:, 2] = smoothing_multiplier_RL
            self.desired_contact_states[:, 3] = smoothing_multiplier_RR

    def _update_debug_arrows(self):
        # Minimal debug marker update; keep your existing implementation if you had one.
        # This placeholder prevents crashes if debug_vis=True but markers aren't configured.
        try:
            if self._markers is None:
                return
        except Exception:
            return
