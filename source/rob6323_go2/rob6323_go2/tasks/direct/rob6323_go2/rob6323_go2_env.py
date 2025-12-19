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

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._commands_target = torch.zeros_like(self._commands)
        self._command_time = torch.zeros(self.num_envs, device=self.device)

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

    def _get_height_scan(self) -> torch.Tensor:
        """Return (num_envs, R) terrain height samples around the robot.

        Uses RayCaster hit points and converts them into heights relative to the robot base.
        Robust to slight IsaacLab shape differences in RayCaster outputs.
        """
        hits = getattr(self._height_scanner.data, "ray_hits_w", None)
        if hits is None or hits.numel() == 0:
            return torch.zeros(self.num_envs, 0, device=self.device)

        # Common shapes:
        #  - (E, R, 3)
        #  - (E, 1, R, 3) or (1, E, R, 3)
        if hits.ndim == 4:
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

        z_hits = hits[..., 2]
        base_z = self.robot.data.root_pos_w[:, 2].unsqueeze(1)
        heights = base_z - z_hits

        heights = torch.nan_to_num(heights, nan=0.0, posinf=0.0, neginf=0.0)

        valid = getattr(self._height_scanner.data, "ray_valid", None)
        if valid is not None and valid.numel() == heights.numel():
            valid = valid.reshape_as(heights)
            heights = torch.where(valid, heights, torch.zeros_like(heights))

        return torch.clamp(heights, -1.0, 1.0).reshape(self.num_envs, -1)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)

        self._height_scanner = RayCaster(self.cfg.height_scanner)

        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.scene.sensors["height_scanner"] = self._height_scanner

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        # ensure height scanner casts against the terrain prim
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

    # ---------------------------------------------------------------------
    # Command resampling + smoothing
    # ---------------------------------------------------------------------
    def _resample_commands(self, env_ids: torch.Tensor):
        num = env_ids.numel()
        self._commands_target[env_ids, 0] = sample_uniform(
            self.cfg.command_range_vx[0], self.cfg.command_range_vx[1], (num,), device=self.device
        )
        self._commands_target[env_ids, 1] = sample_uniform(
            self.cfg.command_range_vy[0], self.cfg.command_range_vy[1], (num,), device=self.device
        )
        self._commands_target[env_ids, 2] = sample_uniform(
            self.cfg.command_range_wz[0], self.cfg.command_range_wz[1], (num,), device=self.device
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

    # ---------------------------------------------------------------------
    # RL step hooks
    # ---------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._update_commands()
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

    def _get_observations(self) -> dict:
        self._step_contact_targets()
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

    # ---------------------------------------------------------------------
    # Rewards
    # ---------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        # tracking rewards
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)

        yaw_rate_error = torch.square(self._commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)

        # action rate penalty
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)

        # raibert heuristic term (Part 6)
        raibert = self._raibert_heuristic()

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

        rewards = (
            self.cfg.lin_vel_reward_scale * lin_vel_error_mapped
            + self.cfg.yaw_rate_reward_scale * yaw_rate_error_mapped
            + self.cfg.action_rate_reward_scale * action_rate
            + raibert
            + self.cfg.upright_reward_scale * upright
            + self.cfg.lin_vel_z_reward_scale * lin_vel_z
            + self.cfg.dof_vel_reward_scale * dof_vel
            + self.cfg.ang_vel_xy_reward_scale * ang_vel_xy
            + feet_clearance
            + tracking_contacts_force
            + self.cfg.base_height_reward_scale * base_height
            + non_foot_contact
            + self.cfg.torque_reward_scale * torque
        )

        # logging
        self._episode_sums["track_lin_vel_xy_exp"] += lin_vel_error_mapped
        self._episode_sums["track_ang_vel_z_exp"] += yaw_rate_error_mapped
        self._episode_sums["rew_action_rate"] += action_rate
        self._episode_sums["raibert_heuristic"] += raibert
        self._episode_sums["orient"] += upright
        self._episode_sums["lin_vel_z"] += lin_vel_z
        self._episode_sums["dof_vel"] += dof_vel
        self._episode_sums["ang_vel_xy"] += ang_vel_xy
        self._episode_sums["feet_clearance"] += feet_clearance
        self._episode_sums["tracking_contacts_shaped_force"] += tracking_contacts_force
        self._episode_sums["base_height"] += base_height
        self._episode_sums["non_foot_contact"] += non_foot_contact
        self._episode_sums["torque"] += torque

        return rewards

    # ---------------------------------------------------------------------
    # Dones / resets
    # ---------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = self.robot.data.root_pos_w[:, 2] < float(self.cfg.base_height_min)
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # reset robot
        self.robot.reset(env_ids)
        self.robot.write_root_pose_to_sim(self.robot.data.default_root_state[env_ids, :7])
        self.robot.write_root_velocity_to_sim(self.robot.data.default_root_state[env_ids, 7:])
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos[env_ids], self.robot.data.default_joint_vel[env_ids]
        )

        # reset actions
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        # reset commands
        self._commands[env_ids] = 0.0
        self._commands_target[env_ids] = 0.0
        self._command_time[env_ids] = 0.0
        self._resample_commands(torch.as_tensor(env_ids, device=self.device))

        # reset gait / clock
        self.gait_indices[env_ids] = 0.0
        self.clock_inputs[env_ids] = 0.0
        self.desired_contact_states[env_ids] = 0.0

        # reset episodic sums
        for key in self._episode_sums:
            self._episode_sums[key][env_ids] = 0.0

    # ---------------------------------------------------------------------
    # Debug visualization
    # ---------------------------------------------------------------------
    def set_debug_vis(self, debug_vis: bool):
        self.cfg.debug_vis = debug_vis
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

        root_pos_w = self.robot.data.root_pos_w
        # goal velocity marker
        goal_quat = math_utils.quat_from_euler_xyz(
            torch.zeros(self.num_envs, device=self.device),
            torch.zeros(self.num_envs, device=self.device),
            torch.atan2(self._commands[:, 1], self._commands[:, 0]),
        )
        self._goal_vel_vis.visualize(root_pos_w, goal_quat)

        # current velocity marker
        cur_vel = self.robot.data.root_lin_vel_b[:, :2]
        cur_quat = math_utils.quat_from_euler_xyz(
            torch.zeros(self.num_envs, device=self.device),
            torch.zeros(self.num_envs, device=self.device),
            torch.atan2(cur_vel[:, 1], cur_vel[:, 0]),
        )
        self._current_vel_vis.visualize(root_pos_w, cur_quat)

    # ---------------------------------------------------------------------
    # Gait / Raibert heuristic and Part-6 style rewards
    # ---------------------------------------------------------------------
    def _step_contact_targets(self):
        """Update gait clock and desired contact states."""
        self.gait_indices = torch.remainder(self.gait_indices + self.step_dt, 1.0)

        # trot gait clocks (FL+RR, FR+RL)
        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * self.gait_indices)
        self.clock_inputs[:, 1] = torch.cos(2 * np.pi * self.gait_indices)
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * (self.gait_indices + 0.5))
        self.clock_inputs[:, 3] = torch.cos(2 * np.pi * (self.gait_indices + 0.5))

        # desired contact: stance when sine > 0
        self.desired_contact_states[:, 0] = (self.clock_inputs[:, 0] > 0).float()  # FL
        self.desired_contact_states[:, 1] = (self.clock_inputs[:, 0] > 0).float()  # RR
        self.desired_contact_states[:, 2] = (self.clock_inputs[:, 2] > 0).float()  # FR
        self.desired_contact_states[:, 3] = (self.clock_inputs[:, 2] > 0).float()  # RL

    def _raibert_heuristic(self) -> torch.Tensor:
        # your baseline raibert term
        # (kept identical to your provided file logic)
        # If your original env had a different implementation, paste it here unchanged.
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_feet_clearance(self) -> torch.Tensor:
        """Reward feet swing clearance toward target height."""
        foot_pos = self.foot_positions_w
        # (num_envs, 4) heights
        h = foot_pos[..., 2]
        target = float(self.cfg.feet_clearance_target_m)
        # Encourage being near target during swing
        # Use desired_contact_states: 0 => swing, 1 => stance
        swing = 1.0 - self.desired_contact_states
        err = torch.square(h - target)
        return -torch.sum(err * swing, dim=1)

    def _reward_tracking_contacts_shaped_force(self) -> torch.Tensor:
        """Reward matching desired contacts using contact forces."""
        # ContactSensor forces are (num_envs, num_bodies, 3)
        forces = self._contact_sensor.data.net_forces_w
        foot_forces = forces[:, self._feet_ids_sensor, 2]  # vertical component
        contact = (foot_forces > float(self.cfg.contact_force_scale)).float()
        # match contact to desired_contact_states
        return -torch.sum(torch.square(contact - self.desired_contact_states), dim=1)

    def _reward_non_foot_contact(self) -> torch.Tensor:
        """Penalty if any non-foot link collides with terrain."""
        if self._undesired_contact_ids_sensor.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        forces = self._contact_sensor.data.net_forces_w
        mag = torch.norm(forces[:, self._undesired_contact_ids_sensor], dim=-1)
        bad = (mag > 1.0).any(dim=1).float()
        return self.cfg.non_foot_contact_reward_scale * bad
