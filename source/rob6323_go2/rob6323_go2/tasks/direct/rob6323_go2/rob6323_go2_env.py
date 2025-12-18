# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# rob6323_go2_env.py
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import math
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import sample_uniform
import isaaclab.utils.math as math_utils

from isaaclab.sensors.ray_caster import RayCaster

from .rob6323_go2_env_cfg import Rob6323Go2EnvCfg


class Rob6323Go2Env(DirectRLEnv):
    cfg: Rob6323Go2EnvCfg

    def __init__(self, cfg: Rob6323Go2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # actions
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)

        # commands (smoothed)
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._commands_target = torch.zeros_like(self._commands)
        self._command_time = torch.zeros(self.num_envs, device=self.device)

        # logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "rew_action_rate",
                "raibert_heuristic",
                "orient", "lin_vel_z", "dof_vel", "ang_vel_xy",
                "feet_clearance", "tracking_contacts_shaped_force",
                "base_height", "non_foot_contact", "torque",
            ]
        }

        # base body id in contact sensor
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._base_id = int(self._base_id[0])

        # debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

        # PD gains
        self.Kp = torch.tensor([cfg.Kp] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.Kd = torch.tensor([cfg.Kd] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.torque_limits = float(cfg.torque_limits)

        # action history for smoothness penalty
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

        feet_set = set(self._feet_ids_sensor)
        self._undesired_contact_ids_sensor = torch.tensor(
            [i for i in sorted(set(undesired)) if i not in feet_set],
            device=self.device,
            dtype=torch.long,
        )

        # last torques for torque penalty
        self._last_torques = torch.zeros(self.num_envs, 12, device=self.device)

        # gait / clock
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)

        # IMPORTANT FIX: foot_indices is used in rewards (raibert) — must exist before first reward call
        self.foot_indices = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)

        # height scan obs buffer
        self._height_scan_obs = torch.zeros(self.num_envs, self.cfg.height_scan_num_rays, device=self.device)

    @property
    def foot_positions_w(self) -> torch.Tensor:
        """Feet positions in world. Order: [FL, FR, RL, RR]."""
        return self.robot.data.body_pos_w[:, self._feet_ids]

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)

        # height scanner
        self._height_scanner = RayCaster(self.cfg.height_scanner_cfg)

        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.scene.sensors["height_scanner"] = self._height_scanner

        # terrain env settings
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing

        # IMPORTANT FIX: if num_envs is large and generator grid is 1x1, all envs share same origin -> crash
        gen = getattr(self.cfg.terrain, "terrain_generator", None)
        if gen is not None and getattr(gen, "num_rows", 1) == 1 and getattr(gen, "num_cols", 1) == 1:
            n = int(self.scene.cfg.num_envs)
            rows = int(math.ceil(math.sqrt(n)))
            cols = int(math.ceil(n / rows))
            gen.num_rows = rows
            gen.num_cols = cols

        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ---------------------------
    # Commands
    # ---------------------------
    def _resample_commands(self, env_ids: torch.Tensor):
        n = int(env_ids.numel())
        self._commands_target[env_ids, 0] = sample_uniform(self.cfg.command_range_vx[0], self.cfg.command_range_vx[1], (n,), self.device)
        self._commands_target[env_ids, 1] = sample_uniform(self.cfg.command_range_vy[0], self.cfg.command_range_vy[1], (n,), self.device)
        self._commands_target[env_ids, 2] = sample_uniform(self.cfg.command_range_yaw[0], self.cfg.command_range_yaw[1], (n,), self.device)

    def _update_commands(self):
        self._command_time += self.step_dt
        resample_mask = self._command_time > float(self.cfg.command_resample_time_s)
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

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # IMPORTANT FIX: rewards/dones may be computed before observations in DirectRLEnv
        # so gait/contact targets (and foot_indices) must be updated here (early).
        self._update_commands()
        self._step_contact_targets()

        self._actions = actions.clone()
        self.desired_joint_pos = self.cfg.action_scale * self._actions + self.robot.data.default_joint_pos

    def _apply_action(self) -> None:
        torques = torch.clip(
            (self.Kp * (self.desired_joint_pos - self.robot.data.joint_pos) - self.Kd * self.robot.data.joint_vel),
            -self.torque_limits,
            self.torque_limits,
        )
        self._last_torques = torques
        self.robot.set_joint_effort_target(torques)

    # ---------------------------
    # Height scan -> observation
    # ---------------------------
    def _update_height_scan_obs(self):
        hits_w = getattr(self._height_scanner.data, "ray_hits_w", None)
        if hits_w is None:
            self._height_scan_obs.zero_()
            return

        hits_z = hits_w[..., 2]  # (N, R)
        base_z = self.robot.data.root_pos_w[:, 2].unsqueeze(1)  # (N, 1)

        # if a ray missed, make it "ground at base height" -> relative height 0
        hits_z = torch.where(torch.isfinite(hits_z), hits_z, base_z)

        rel = hits_z - base_z  # negative means ground below base
        rel = torch.clamp(rel, -1.0, 0.5)

        # Ensure buffer matches configured observation length
        if rel.shape[1] != self._height_scan_obs.shape[1]:
            self._height_scan_obs = torch.zeros(self.num_envs, rel.shape[1], device=self.device)

        self._height_scan_obs[:] = rel

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        self._update_height_scan_obs()

        obs = torch.cat(
            [
                self.robot.data.root_lin_vel_b,
                self.robot.data.root_ang_vel_b,
                self.robot.data.projected_gravity_b,
                self._commands,
                self.robot.data.joint_pos - self.robot.data.default_joint_pos,
                self.robot.data.joint_vel,
                self._actions,
                self.clock_inputs,
                self._height_scan_obs,
            ],
            dim=-1,
        )
        self.extras = {}
        return {"policy": obs}

    # ---------------------------
    # Rewards
    # ---------------------------
    def _get_rewards(self) -> torch.Tensor:
        # tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)

        yaw_rate_error = torch.square(self._commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)

        # action rate
        rew_action_rate = torch.sum(torch.square(self._actions - self.last_actions[:, :, 0]), dim=1) * (self.cfg.action_scale**2)
        rew_action_rate += torch.sum(
            torch.square(self._actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]), dim=1
        ) * (self.cfg.action_scale**2)

        self.last_actions = torch.roll(self.last_actions, 1, 2)
        self.last_actions[:, :, 0] = self._actions[:]

        # raibert
        rew_raibert_heuristic = self._reward_raibert_heuristic()

        # stability
        pg = self.robot.data.projected_gravity_b
        lin = self.robot.data.root_lin_vel_b
        ang = self.robot.data.root_ang_vel_b
        qd = self.robot.data.joint_vel

        rew_orient = (pg[:, 0] ** 2 + pg[:, 1] ** 2)
        rew_lin_vel_z = (lin[:, 2] ** 2)
        rew_dof_vel = (qd ** 2).sum(dim=1)
        rew_ang_vel_xy = (ang[:, 0] ** 2 + ang[:, 1] ** 2)

        # foot clearance
        foot_h = self.foot_positions_w[:, :, 2]
        swing_w = 1.0 - self.desired_contact_states
        clearance = float(self.cfg.feet_clearance_target_m)
        err = torch.clamp(clearance - foot_h, min=0.0)
        rew_feet_clearance = (swing_w * (err ** 2)).sum(dim=1)

        # contact shaping
        forces_w = self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor, :]
        fmag = torch.linalg.norm(forces_w, dim=-1)
        contact_strength = torch.tanh(fmag / float(self.cfg.contact_force_scale))
        match = 1.0 - torch.abs(contact_strength - self.desired_contact_states)
        rew_track_contacts = match.mean(dim=1)

        # base height relative to local ground (estimated from ray hits)
        hits_w = getattr(self._height_scanner.data, "ray_hits_w", None)
        if hits_w is None:
            rew_base_height = torch.zeros(self.num_envs, device=self.device)
        else:
            hits_z = hits_w[..., 2]
            base_z = self.robot.data.root_pos_w[:, 2].unsqueeze(1)
            hits_z = torch.where(torch.isfinite(hits_z), hits_z, base_z)
            ground_z = torch.mean(hits_z, dim=1)
            base_h_rel = self.robot.data.root_pos_w[:, 2] - ground_z
            rew_base_height = (base_h_rel - float(self.cfg.base_height_target_m)) ** 2

        # non-foot contact
        if self._undesired_contact_ids_sensor.numel() > 0:
            bad_forces = self._contact_sensor.data.net_forces_w[:, self._undesired_contact_ids_sensor, :]
            bad_mag = torch.linalg.norm(bad_forces, dim=-1).sum(dim=1)
            rew_non_foot_contact = torch.tanh(bad_mag / float(self.cfg.contact_force_scale))
        else:
            rew_non_foot_contact = torch.zeros(self.num_envs, device=self.device)

        # torque penalty
        tau = self._last_torques
        rew_torque = ((tau / self.torque_limits) ** 2).sum(dim=1)

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
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    # ---------------------------
    # Termination
    # ---------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # terminate if base is too low relative to local ground
        hits_w = getattr(self._height_scanner.data, "ray_hits_w", None)
        if hits_w is None:
            cstr_base_height_min = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            hits_z = hits_w[..., 2]
            base_z = self.robot.data.root_pos_w[:, 2].unsqueeze(1)
            hits_z = torch.where(torch.isfinite(hits_z), hits_z, base_z)
            ground_z = torch.mean(hits_z, dim=1)
            base_h_rel = self.robot.data.root_pos_w[:, 2] - ground_z
            cstr_base_height_min = base_h_rel < float(self.cfg.base_height_min)

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        net_contact_forces = self._contact_sensor.data.net_forces_w_history  # (N,H,B,3)
        base_id = int(self._base_id)
        base_force_mag_hist = torch.linalg.norm(net_contact_forces[:, :, base_id, :], dim=-1)
        cstr_termination_contacts = torch.any(base_force_mag_hist > 1.0, dim=1)

        cstr_upsidedown = self.robot.data.projected_gravity_b[:, 2] > 0
        died = cstr_termination_contacts | cstr_upsidedown | cstr_base_height_min
        return died, time_out

    # ---------------------------
    # Reset + logging
    # ---------------------------
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
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # logging
        self.extras.setdefault("log", {})
        self.extras.setdefault("episode", {})

        log_dict = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            val = episodic_sum_avg / self.max_episode_length_s

            if key in ("track_lin_vel_xy_exp", "track_ang_vel_z_exp"):
                val = val / self.step_dt

            log_dict["Episode_Reward/" + key] = val
            self._episode_sums[key][env_ids] = 0.0

        log_dict["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        log_dict["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()

        self.extras["log"].update(log_dict)
        self.extras["episode"].update(log_dict)

        self.last_actions[env_ids] = 0.0
        self.gait_indices[env_ids] = 0
        self.foot_indices[env_ids] = 0
        self.clock_inputs[env_ids] = 0
        self.desired_contact_states[env_ids] = 0

    # ---------------------------
    # Debug viz
    # ---------------------------
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
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])

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

    # ---------------------------
    # Gait clock + contact targets
    # ---------------------------
    def _step_contact_targets(self):
        frequencies = 3.0
        phases = 0.5
        offsets = 0.0
        bounds = 0.0
        durations = 0.5 * torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
        self.gait_indices = torch.remainder(self.gait_indices + self.step_dt * frequencies, 1.0)

        # Order consistent with feet: [FL, FR, RL, RR]
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

            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                0.5 / (1 - durations[swing_idxs])
            )

        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

        kappa = 0.07
        smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf

        sm_FL = (
            smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0))
            * (1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5))
            + smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1)
            * (1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1.5))
        )
        sm_FR = (
            smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0))
            * (1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5))
            + smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1)
            * (1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1.5))
        )
        sm_RL = (
            smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0))
            * (1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5))
            + smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1)
            * (1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1.5))
        )
        sm_RR = (
            smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0))
            * (1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5))
            + smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1)
            * (1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1.5))
        )

        self.desired_contact_states[:, 0] = sm_FL
        self.desired_contact_states[:, 1] = sm_FR
        self.desired_contact_states[:, 2] = sm_RL
        self.desired_contact_states[:, 3] = sm_RR

    # ---------------------------
    # Raibert heuristic
    # ---------------------------
    def _reward_raibert_heuristic(self):
        cur_footsteps_translated = self.foot_positions_w - self.robot.data.root_pos_w.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)

        for i in range(4):
            footsteps_in_body_frame[:, i, :] = math_utils.quat_apply_yaw(
                math_utils.quat_conjugate(self.robot.data.root_quat_w), cur_footsteps_translated[:, i, :]
            )

        # Order: [FL, FR, RL, RR]
        desired_stance_width = 0.25
        desired_ys_nom = torch.tensor(
            [desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2],
            device=self.device,
        ).unsqueeze(0)

        desired_stance_length = 0.45
        desired_xs_nom = torch.tensor(
            [desired_stance_length / 2, desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2],
            device=self.device,
        ).unsqueeze(0)

        phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = torch.tensor([3.0], device=self.device)

        x_vel_des = self._commands[:, 0:1]
        y_vel_des = self._commands[:, 1:2]

        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

        err = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])
        return torch.sum(torch.square(err), dim=(1, 2))
