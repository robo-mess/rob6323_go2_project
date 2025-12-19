# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from collections.abc import Sequence

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import sample_uniform
import isaaclab.utils.math as math_utils

from .rob6323_go2_env_cfg_bipedal_safe import Rob6323Go2EnvCfg


class Rob6323Go2Env(DirectRLEnv):
    cfg: Rob6323Go2EnvCfg

    def __init__(self, cfg: Rob6323Go2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)

        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._commands_target = torch.zeros_like(self._commands)
        self._command_time = torch.zeros(self.num_envs, device=self.device)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "rew_action_rate",
                "raibert_heuristic",
                "orient",
                "lin_vel_z",
                "dof_vel",
                "ang_vel_xy",
                "feet_clearance",
                "tracking_contacts_shaped_force",
                "base_height",
                "non_foot_contact",
                "torque",
                "bipedal_gate",
                "bipedal_lift",
                "bipedal_upright",
                "bipedal_front_air",
                "bipedal_front_contact",
            ]
        }

        self.robot: Articulation = Articulation(cfg.robot_cfg)
        self._terrain = self.scene.terrain
        self._contact_sensor: ContactSensor = ContactSensor(cfg.contact_sensor)

        self.Kp = float(cfg.Kp)
        self.Kd = float(cfg.Kd)
        self.torque_limits = torch.tensor([float(cfg.torque_limits)], device=self.device)

        self._last_torques = torch.zeros(self.num_envs, cfg.action_space, device=self.device)
        self.last_actions = torch.zeros((self.num_envs, cfg.action_space, 2), device=self.device)

        self.desired_contact_states = torch.zeros((self.num_envs, 4), device=self.device)
        self.clock_inputs = torch.zeros((self.num_envs, 4), device=self.device)
        self.gait_indices = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self._feet_ids = [self.robot.find_bodies(name)[0][0] for name in foot_names]

        self._feet_ids_sensor = []
        for name in foot_names:
            id_list, _ = self._contact_sensor.find_bodies(name)
            self._feet_ids_sensor.append(int(id_list[0]))

        self._front_feet_ids_sensor = torch.tensor(self._feet_ids_sensor[0:2], device=self.device, dtype=torch.long)
        self._rear_feet_ids_sensor = torch.tensor(self._feet_ids_sensor[2:4], device=self.device, dtype=torch.long)
        self._front_feet_ids = torch.tensor(self._feet_ids[0:2], device=self.device, dtype=torch.long)
        self._rear_feet_ids = torch.tensor(self._feet_ids[2:4], device=self.device, dtype=torch.long)

        base_list, _ = self._contact_sensor.find_bodies("base")
        self._base_id = int(base_list[0])

        undesired_names = ["FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh", "FL_calf", "FR_calf", "RL_calf", "RR_calf"]
        undesired_ids = []
        for name in undesired_names:
            ids, _ = self._contact_sensor.find_bodies(name)
            if len(ids) > 0:
                undesired_ids.append(int(ids[0]))
        self._undesired_contact_ids_sensor = (
            torch.tensor(undesired_ids, device=self.device, dtype=torch.long)
            if len(undesired_ids) > 0
            else torch.zeros((0,), device=self.device, dtype=torch.long)
        )

        if cfg.debug_vis:
            self.goal_vel_visualizer = VisualizationMarkers(cfg.goal_vel_visualizer_cfg)
            self.current_vel_visualizer = VisualizationMarkers(cfg.current_vel_visualizer_cfg)

        self.default_joint_pos = self.robot.data.default_joint_pos.clone()
        self.desired_joint_pos = self.default_joint_pos.clone()

    def _setup_scene(self):
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone()
        self._update_commands()
        self.desired_joint_pos = self.default_joint_pos + self._actions * float(self.cfg.action_scale)

    def _apply_action(self) -> None:
        torques = (
            self.Kp * (self.desired_joint_pos - self.robot.data.joint_pos) - self.Kd * self.robot.data.joint_vel
        )
        torques = torch.clip(torques, -self.torque_limits, self.torque_limits)
        self._last_torques = torques
        self.robot.set_joint_effort_target(torques)

    def _get_observations(self) -> dict:
        self._step_contact_targets()
        self._previous_actions = self._actions.clone()
        obs = torch.cat(
            [
                self.robot.data.root_lin_vel_b,
                self.robot.data.root_ang_vel_b,
                self.robot.data.projected_gravity_b,
                self._commands,
                self.robot.data.joint_pos - self.default_joint_pos,
                self.robot.data.joint_vel,
                self._actions,
                self.clock_inputs,
            ],
            dim=-1,
        )
        return {"policy": obs}

    # ---------------------------
    # Commands
    # ---------------------------
    def _resample_commands(self, env_ids: torch.Tensor):
        n = int(env_ids.numel())
        if bool(getattr(self.cfg, "enable_bipedal_skill", False)):
            vx_lo, vx_hi = self.cfg.bipedal_command_range_vx
            bin_sz = float(self.cfg.bipedal_command_vx_bin)
            n_bins = int(round((vx_hi - vx_lo) / bin_sz)) + 1
            idx = torch.randint(0, n_bins, (n,), device=self.device)
            self._commands_target[env_ids, 0] = float(vx_lo) + idx.to(torch.float32) * bin_sz
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
        self._command_time += self.step_dt
        resample_time = float(
            self.cfg.bipedal_command_resample_time_s
            if bool(getattr(self.cfg, "enable_bipedal_skill", False))
            else self.cfg.command_resample_time_s
        )
        resample_mask = self._command_time > resample_time
        if torch.any(resample_mask):
            env_ids = torch.nonzero(resample_mask).squeeze(-1)
            self._command_time[env_ids] = 0.0
            self._resample_commands(env_ids)

        tau = float(self.cfg.command_smoothing_tau_s)
        if tau <= 0.0:
            self._commands[:] = self._commands_target
        else:
            alpha = 1.0 - torch.exp(torch.tensor(-self.step_dt / tau, device=self.device))
            self._commands += alpha * (self._commands_target - self._commands)

        if bool(getattr(self.cfg, "enable_bipedal_skill", False)):
            self._commands[:, 1] = 0.0
            self._commands_target[:, 1] = 0.0

    def _get_rewards(self) -> torch.Tensor:
        bipedal = bool(getattr(self.cfg, "enable_bipedal_skill", False))

        # tracking (yaw-frame planar) for bipedal
        if bipedal:
            base_quat_w = self.robot.data.root_quat_w
            w, x, y, z = base_quat_w.unbind(dim=-1)
            yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            cy, sy = torch.cos(yaw), torch.sin(yaw)

            lin_vel_w = math_utils.quat_apply(base_quat_w, self.robot.data.root_lin_vel_b)
            vx_w, vy_w = lin_vel_w[:, 0], lin_vel_w[:, 1]
            vel_yaw_x = cy * vx_w + sy * vy_w
            vel_yaw_y = -sy * vx_w + cy * vy_w
            lin_vel_yaw_xy = torch.stack([vel_yaw_x, vel_yaw_y], dim=-1)

            ang_vel_w = math_utils.quat_apply(base_quat_w, self.robot.data.root_ang_vel_b)
            yaw_rate_meas = ang_vel_w[:, 2]
        else:
            lin_vel_yaw_xy = self.robot.data.root_lin_vel_b[:, :2]
            yaw_rate_meas = self.robot.data.root_ang_vel_b[:, 2]

        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - lin_vel_yaw_xy), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)

        yaw_rate_error = torch.square(self._commands[:, 2] - yaw_rate_meas)
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)

        # action rate
        rew_action_rate = torch.sum(torch.square(self._actions - self.last_actions[:, :, 0]), dim=1) * (
            self.cfg.action_scale**2
        )
        rew_action_rate = rew_action_rate + torch.sum(
            torch.square(self._actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]), dim=1
        ) * (self.cfg.action_scale**2)

        self.last_actions = torch.roll(self.last_actions, 1, 2)
        self.last_actions[:, :, 0] = self._actions[:]

        rew_raibert_heuristic = self._reward_raibert_heuristic()

        # stability
        pg = self.robot.data.projected_gravity_b
        lin = self.robot.data.root_lin_vel_b
        ang = self.robot.data.root_ang_vel_b
        qd = self.robot.data.joint_vel

        if bipedal:
            rew_orient = (pg[:, 1] ** 2)  # roll only
        else:
            rew_orient = (pg[:, 0] ** 2 + pg[:, 1] ** 2)

        rew_lin_vel_z = (lin[:, 2] ** 2)
        rew_dof_vel = (qd ** 2).sum(dim=1)
        rew_ang_vel_xy = (ang[:, 0] ** 2 + ang[:, 1] ** 2)

        # foot clearance + contact match
        foot_pos_w = self.foot_positions_w
        foot_h = foot_pos_w[:, :, 2]

        swing_w = 1.0 - self.desired_contact_states
        clearance = float(self.cfg.feet_clearance_target_m)
        err = torch.clamp(clearance - foot_h, min=0.0)
        rew_feet_clearance = (swing_w * (err**2)).sum(dim=1)

        forces_w = self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor, :]
        fmag = torch.linalg.norm(forces_w, dim=-1)
        contact_strength = torch.tanh(fmag / float(self.cfg.contact_force_scale))

        match = 1.0 - torch.abs(contact_strength - self.desired_contact_states)
        rew_track_contacts = match.mean(dim=1)

        # base height (off in bipedal)
        base_h = self.robot.data.root_pos_w[:, 2]
        if bipedal:
            rew_base_height = torch.zeros_like(base_h)
        else:
            rew_base_height = (base_h - float(self.cfg.base_height_target_m)) ** 2

        # collisions
        if self._undesired_contact_ids_sensor.numel() > 0:
            bad_forces = self._contact_sensor.data.net_forces_w[:, self._undesired_contact_ids_sensor, :]
            bad_mag = torch.linalg.norm(bad_forces, dim=-1).sum(dim=1)
            rew_non_foot_contact = torch.tanh(bad_mag / float(self.cfg.contact_force_scale))
        else:
            rew_non_foot_contact = torch.zeros(self.num_envs, device=self.device)

        # torque
        tau = self._last_torques
        rew_torque = ((tau / self.torque_limits) ** 2).sum(dim=1)

        # bipedal shaping + gating
        bipedal_gate = torch.zeros(self.num_envs, device=self.device)
        bipedal_lift = torch.zeros(self.num_envs, device=self.device)
        bipedal_upright = torch.zeros(self.num_envs, device=self.device)
        bipedal_front_air = torch.zeros(self.num_envs, device=self.device)
        bipedal_front_contact = torch.zeros(self.num_envs, device=self.device)

        if bipedal:
            base_quat_w = self.robot.data.root_quat_w
            w, x, y, z = base_quat_w.unbind(dim=-1)
            yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

            x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
            v_f = math_utils.quat_apply(base_quat_w, x_axis)

            c, s = torch.cos(yaw), torch.sin(yaw)
            v_u = torch.stack([0.2 * c, 0.2 * s, -torch.ones_like(c)], dim=-1)
            v_u_norm = torch.linalg.norm(v_u, dim=-1) + 1.0e-6
            upright_cos = torch.sum(v_f * v_u, dim=-1) / v_u_norm

            hmin = float(self.cfg.bipedal_height_min_m)
            hmax = float(self.cfg.bipedal_height_max_m)
            height_clip = torch.clamp((base_h - hmin) / (hmax - hmin + 1.0e-6), 0.0, 1.0)

            upright_gate = torch.clamp((upright_cos - 0.65) / (0.88 - 0.65), 0.0, 1.0)
            bipedal_gate = (height_clip ** float(getattr(self.cfg, "bipedal_track_gate_power", 1.0))) * upright_gate

            bipedal_lift = height_clip
            bipedal_upright = (0.5 * upright_cos + 0.5) ** 2

            front_h = foot_h[:, 0:2]
            front_low = torch.clamp(float(self.cfg.bipedal_front_feet_min_height_m) - front_h, min=0.0)
            bipedal_front_air = torch.sum(front_low**2, dim=1)

            bipedal_front_contact = torch.sum(contact_strength[:, 0:2], dim=1)

            lin_vel_error_mapped = lin_vel_error_mapped * bipedal_gate
            yaw_rate_error_mapped = yaw_rate_error_mapped * bipedal_gate

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
            "tracking_contacts_shaped_force": rew_track_contacts
            * self.cfg.tracking_contacts_shaped_force_reward_scale
            * self.step_dt,
            "base_height": rew_base_height * self.cfg.base_height_reward_scale * self.step_dt,
            "non_foot_contact": rew_non_foot_contact * self.cfg.non_foot_contact_reward_scale * self.step_dt,
            "torque": rew_torque * self.cfg.torque_reward_scale * self.step_dt,
            "bipedal_gate": bipedal_gate * float(getattr(self.cfg, "bipedal_gate_reward_scale", 0.0)) * self.step_dt,
            "bipedal_lift": bipedal_lift * float(getattr(self.cfg, "bipedal_lift_reward_scale", 0.0)) * self.step_dt,
            "bipedal_upright": bipedal_upright
            * float(getattr(self.cfg, "bipedal_upright_reward_scale", 0.0))
            * self.step_dt,
            "bipedal_front_air": bipedal_front_air
            * float(getattr(self.cfg, "bipedal_front_air_reward_scale", 0.0))
            * self.step_dt,
            "bipedal_front_contact": bipedal_front_contact
            * float(getattr(self.cfg, "bipedal_front_contact_penalty_scale", 0.0))
            * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        base_height = self.robot.data.root_pos_w[:, 2]
        cstr_base_height_min = base_height < self.cfg.base_height_min
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        base_id = int(self._base_id)
        base_force_mag_hist = torch.linalg.norm(net_contact_forces[:, :, base_id, :], dim=-1)
        cstr_termination_contacts = torch.any(base_force_mag_hist > 1.0, dim=1)

        cstr_upsidedown = self.robot.data.projected_gravity_b[:, 2] > 0
        died = cstr_termination_contacts | cstr_upsidedown | cstr_base_height_min

        if bool(getattr(self.cfg, "enable_bipedal_skill", False)):
            grace_steps = int(getattr(self.cfg, "bipedal_grace_steps", 40))
            after_grace = self.episode_length_buf > grace_steps

            front_forces = self._contact_sensor.data.net_forces_w[:, self._front_feet_ids_sensor, :]
            front_mag = torch.linalg.norm(front_forces, dim=-1).max(dim=1).values
            front_contact = front_mag > float(getattr(self.cfg, "bipedal_front_contact_force_threshold", 25.0))

            died = died | (after_grace & front_contact)

        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._command_time[env_ids_t] = 0.0
        self._commands[env_ids_t] = 0.0
        self._resample_commands(env_ids_t)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids)

        log_dict = {}
        for key in self._episode_sums.keys():
            log_dict[key] = torch.mean(self._episode_sums[key][env_ids]).item() / float(self.cfg.episode_length_s)
            self._episode_sums[key][env_ids] = 0.0

        log_dict["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        log_dict["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()

        self.extras["log"].update(log_dict)
        self.extras["episode"].update(log_dict)

        self.last_actions[env_ids] = 0.0
        self.gait_indices[env_ids] = 0

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self._commands[:, :2])

        if bool(getattr(self.cfg, "enable_bipedal_skill", False)):
            base_quat_w = self.robot.data.root_quat_w
            w, x, y, z = base_quat_w.unbind(dim=-1)
            yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            cy, sy = torch.cos(yaw), torch.sin(yaw)
            lin_vel_w = math_utils.quat_apply(base_quat_w, self.robot.data.root_lin_vel_b)
            vx_w, vy_w = lin_vel_w[:, 0], lin_vel_w[:, 1]
            vel_yaw_x = cy * vx_w + sy * vy_w
            vel_yaw_y = -sy * vx_w + cy * vy_w
            vel_meas_xy = torch.stack([vel_yaw_x, vel_yaw_y], dim=-1)
        else:
            vel_meas_xy = self.robot.data.root_lin_vel_b[:, :2]

        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(vel_meas_xy)

        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0

        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)

        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat

    # gait/contact plan
    def _step_contact_targets(self):
        frequencies = 3.0
        durations = 0.5 * torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)

        self.gait_indices = torch.remainder(self.gait_indices + self.step_dt * frequencies, 1.0)

        if bool(getattr(self.cfg, "enable_bipedal_skill", False)):
            foot_indices = [
                self.gait_indices,       # FL (unused)
                self.gait_indices,       # FR (unused)
                self.gait_indices,       # RL
                self.gait_indices + 0.5, # RR
            ]
        else:
            phases = 0.5
            offsets = 0.0
            bounds = 0.0
            foot_indices = [
                self.gait_indices + phases + offsets + bounds,
                self.gait_indices + offsets,
                self.gait_indices + bounds,
                self.gait_indices + phases,
            ]

        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

        for idxs in foot_indices:
            stance_idxs = torch.remainder(idxs, 1) < durations
            swing_idxs = torch.remainder(idxs, 1) > durations
            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                0.5 / (1 - durations[swing_idxs])
            )

        if bool(getattr(self.cfg, "enable_bipedal_skill", False)):
            self.clock_inputs[:, 0] = 0.0
            self.clock_inputs[:, 1] = 0.0
            self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
            self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])
        else:
            self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
            self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
            self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
            self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

        kappa = 0.07
        smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf

        smoothing_multiplier_FL = (
            smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0))
            * (1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5))
            + smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1)
            * (1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5 - 1))
        )
        smoothing_multiplier_FR = (
            smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0))
            * (1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5))
            + smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1)
            * (1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5 - 1))
        )
        smoothing_multiplier_RL = (
            smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0))
            * (1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5))
            + smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1)
            * (1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5 - 1))
        )
        smoothing_multiplier_RR = (
            smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0))
            * (1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5))
            + smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1)
            * (1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5 - 1))
        )

        if bool(getattr(self.cfg, "enable_bipedal_skill", False)):
            self.desired_contact_states[:, 0] = 0.0
            self.desired_contact_states[:, 1] = 0.0
            self.desired_contact_states[:, 2] = smoothing_multiplier_RL
            self.desired_contact_states[:, 3] = smoothing_multiplier_RR
        else:
            self.desired_contact_states[:, 0] = smoothing_multiplier_FL
            self.desired_contact_states[:, 1] = smoothing_multiplier_FR
            self.desired_contact_states[:, 2] = smoothing_multiplier_RL
            self.desired_contact_states[:, 3] = smoothing_multiplier_RR

    def _reward_raibert_heuristic(self):
        cur_footsteps_translated = self.foot_positions_w - self.robot.data.root_pos_w.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = math_utils.quat_apply_yaw(
                math_utils.quat_inv(self.robot.data.root_quat_w), cur_footsteps_translated[:, i, :]
            )

        foot_positions_b = self.robot.data.default_body_state_w[:, self._feet_ids, :3]
        foot_positions_translated = foot_positions_b - self.robot.data.default_root_state[:, :3].unsqueeze(1)
        foot_positions_translated_b = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            foot_positions_translated_b[:, i, :] = math_utils.quat_apply_yaw(
                math_utils.quat_inv(self.robot.data.default_root_state[:, 3:7]), foot_positions_translated[:, i, :]
            )

        desired_xs_nom = foot_positions_translated_b[:, :, 0]
        desired_ys_nom = foot_positions_translated_b[:, :, 1]

        frequencies = 3.0
        phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5

        x_vel_des = self._commands[:, 0:1]
        y_vel_des = self._commands[:, 1:2]

        desired_ys_nom = desired_ys_nom + phases * y_vel_des * (0.5 / frequencies)
        desired_xs_nom = desired_xs_nom + phases * x_vel_des * (0.5 / frequencies)

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)
        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

        if bool(getattr(self.cfg, "enable_bipedal_skill", False)):
            err_raibert_heuristic = err_raibert_heuristic[:, 2:4, :]

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))
        return reward
