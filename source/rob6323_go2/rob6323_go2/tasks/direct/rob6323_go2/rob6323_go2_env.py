# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# rob6323_go2_env.py
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, sample_uniform

from rob6323_go2_env_cfg_latest import Rob6323Go2EnvCfg


class Rob6323Go2Env(DirectRLEnv):
    cfg: Rob6323Go2EnvCfg

    def __init__(self, cfg: Rob6323Go2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, 12, device=self.device)

        # Indices for feet (body ids in articulation)
        # Order: [FL, FR, RL, RR]
        self._feet_names = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
        self._feet_ids = [self.robot.find_bodies(name)[0] for name in self._feet_names]

        # Contact sensor filtering: get ids from sensor body ordering
        body_names_sensor = self._contact_sensor.body_names
        self._feet_ids_sensor = [body_names_sensor.index(n) for n in self._feet_names if n in body_names_sensor]

        # Undesired contacts: everything except feet
        undesired = list(range(len(body_names_sensor)))

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

        # command buffers (smoothed)
        self._commands_target = torch.zeros(self.num_envs, 3, device=self.device)
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._command_time = torch.zeros(self.num_envs, device=self.device)

        # Logging
        self._episode_sums = {
            "track_lin_vel_xy_exp": torch.zeros(self.num_envs, device=self.device),
            "track_ang_vel_z_exp": torch.zeros(self.num_envs, device=self.device),
            "orient": torch.zeros(self.num_envs, device=self.device),
            "lin_vel_z": torch.zeros(self.num_envs, device=self.device),
            "dof_vel": torch.zeros(self.num_envs, device=self.device),
            "ang_vel_xy": torch.zeros(self.num_envs, device=self.device),
            "feet_clearance": torch.zeros(self.num_envs, device=self.device),
            "tracking_contacts_shaped_force": torch.zeros(self.num_envs, device=self.device),
            "base_height": torch.zeros(self.num_envs, device=self.device),
            "non_foot_contact": torch.zeros(self.num_envs, device=self.device),
            "torque": torch.zeros(self.num_envs, device=self.device),
            "symmetry": torch.zeros(self.num_envs, device=self.device),
            "feet_air_time": torch.zeros(self.num_envs, device=self.device),
            "foot_slip": torch.zeros(self.num_envs, device=self.device),
        }

    @property
    def foot_positions_w(self) -> torch.Tensor:
        """Returns the feet positions in the world frame.
        Shape: (num_envs, num_feet, 3)
        Order: [FL, FR, RL, RR]
        """
        return self.robot.data.body_pos_w[:, self._feet_ids]

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)

        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self._contact_sensor

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

        # store action
        self._actions = actions.clone()

    def _apply_action(self) -> None:
        # Action is joint position delta around default pose
        # Use internal actuator in robot cfg (PD)
        self.robot.set_joint_position_target(self._actions)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # Observations: base lin/ang vel, projected gravity, dof pos/vel, commands
        base_lin_vel_b = self.robot.data.root_lin_vel_b
        base_ang_vel_b = self.robot.data.root_ang_vel_b
        projected_gravity_b = quat_rotate_inverse(self.robot.data.root_quat_w, self.robot.data.gravity_vec_w)
        dof_pos = self.robot.data.joint_pos
        dof_vel = self.robot.data.joint_vel

        obs = torch.cat(
            [
                base_lin_vel_b,
                base_ang_vel_b,
                projected_gravity_b,
                dof_pos,
                dof_vel,
                self._commands,
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _compute_gait(self):
        # Simple trot clock (4 legs)
        # Clock is used for desired contact schedule
        freq = 1.0  # Hz
        self.gait_indices += self.step_dt * freq * 2 * math.pi
        self.gait_indices %= 2 * math.pi

        # trot: FL+RR in phase, FR+RL opposite
        phase = self.gait_indices
        self.clock_inputs[:, 0] = torch.sin(phase)           # FL
        self.clock_inputs[:, 1] = torch.sin(phase + math.pi) # FR
        self.clock_inputs[:, 2] = torch.sin(phase + math.pi) # RL
        self.clock_inputs[:, 3] = torch.sin(phase)           # RR

        # desired contact: stance when sin > 0
        self.desired_contact_states = (self.clock_inputs > 0.0).float()

    def _get_rewards(self) -> torch.Tensor:
        # ============================================================
        # command tracking (same as tutorial)
        # ============================================================
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)

        yaw_rate_error = torch.square(self._commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)

        # ============================================================
        # gait shaping
        # ============================================================
        self._compute_gait()

        feet_pos = self.foot_positions_w  # (N,4,3)
        feet_h = feet_pos[..., 2]         # (N,4)

        # Foot clearance reward (encourage swing legs to lift)
        desired_contact = self.desired_contact_states
        rew_feet_clearance = torch.square(feet_h - float(self.cfg.feet_clearance_target_m)) * (1.0 - desired_contact)
        rew_feet_clearance = rew_feet_clearance.mean(dim=1)

        # Feet air time reward (from contact sensor)
        if len(self._feet_ids_sensor) == 4:
            air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids_sensor]  # (N,4)
            rew_feet_air_time = air_time.mean(dim=1)
        else:
            rew_feet_air_time = torch.zeros(self.num_envs, device=self.device)

        # Foot slip penalty: penalize feet moving when in contact
        if len(self._feet_ids_sensor) == 4:
            feet_vel = self.robot.data.body_lin_vel_w[:, self._feet_ids, :2]  # (N,4,2)
            slip = torch.linalg.norm(feet_vel, dim=-1) * desired_contact      # (N,4)
            rew_foot_slip = slip.mean(dim=1)
        else:
            rew_foot_slip = torch.zeros(self.num_envs, device=self.device)

        # Symmetry reward (simple)
        # encourage diagonal pairs to have similar heights (trot-like symmetry)
        rew_symmetry = torch.abs(feet_h[:, 0] - feet_h[:, 3]) + torch.abs(feet_h[:, 1] - feet_h[:, 2])

        # ============================================================
        # stability penalties
        # ============================================================
        projected_gravity_b = quat_rotate_inverse(self.robot.data.root_quat_w, self.robot.data.gravity_vec_w)
        rew_orient = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

        rew_lin_vel_z = torch.square(self.robot.data.root_lin_vel_b[:, 2])
        rew_dof_vel = torch.sum(torch.square(self.robot.data.joint_vel), dim=1)
        rew_ang_vel_xy = torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=1)

        # ============================================================
        # contact tracking (shaped)
        # ============================================================
        if len(self._feet_ids_sensor) == 4:
            contact_forces = self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor, :]  # (N,4,3)
            contact_strength = torch.linalg.norm(contact_forces, dim=-1)  # (N,4)
            contact_strength = torch.tanh(contact_strength / float(self.cfg.contact_force_scale))
        else:
            contact_strength = torch.zeros(self.num_envs, 4, device=self.device)

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
        # torque penalty
        # ============================================================
        torques = self.robot.data.applied_torque
        rew_torque = torch.sum(torch.square(torques), dim=1)
        self._last_torques = torques

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,

            "feet_clearance": rew_feet_clearance * self.cfg.foot_clearance_reward_scale * self.step_dt,
            "tracking_contacts_shaped_force": rew_track_contacts * self.cfg.tracking_contacts_shaped_force_reward_scale * self.step_dt,

            "base_height": rew_base_height * self.cfg.base_height_reward_scale * self.step_dt,
            "non_foot_contact": rew_non_foot_contact * self.cfg.non_foot_contact_reward_scale * self.step_dt,
            "torque": rew_torque * self.cfg.torque_reward_scale * self.step_dt,

            "orient": rew_orient * self.cfg.orient_reward_scale * self.step_dt,
            "lin_vel_z": rew_lin_vel_z * self.cfg.lin_vel_z_reward_scale * self.step_dt,
            "dof_vel": rew_dof_vel * self.cfg.dof_vel_reward_scale * self.step_dt,
            "ang_vel_xy": rew_ang_vel_xy * self.cfg.ang_vel_xy_reward_scale * self.step_dt,

            "symmetry": rew_symmetry * self.cfg.symmetry_reward_scale * self.step_dt,
            "feet_air_time": rew_feet_air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "foot_slip": rew_foot_slip * self.cfg.foot_slip_reward_scale * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # time-out
        time_out = self.episode_length_buf >= int(self.max_episode_length) - 1

        # base height termination (still works with mild rough terrain)
        base_low = self.robot.data.root_pos_w[:, 2] < float(self.cfg.base_height_min)

        terminated = base_low
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # reset robot state
        default_root_state = self.robot.data.default_root_state.clone()
        default_joint_pos = self.robot.data.default_joint_pos.clone()
        default_joint_vel = self.robot.data.default_joint_vel.clone()

        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        self.robot.write_root_state_to_sim(default_root_state[env_ids])
        self.robot.write_joint_state_to_sim(default_joint_pos[env_ids], default_joint_vel[env_ids])

        # reset buffers
        self._actions[env_ids] = 0.0
        self._last_torques[env_ids] = 0.0
        self._command_time[env_ids] = 0.0
        self._commands_target[env_ids] = 0.0
        self._commands[env_ids] = 0.0

        # resample commands
        self._resample_commands(env_ids)

        # reset logging
        for k in self._episode_sums:
            self._episode_sums[k][env_ids] = 0.0

        super()._reset_idx(env_ids)
