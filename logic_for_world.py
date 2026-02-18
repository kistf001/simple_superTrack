"""
World dynamics network training.
SuperTrack Algorithm 1 implementation.
"""

import torch
from utils.quaternion import torch as quaternion_torch
from transforms import local_torch, integrate_torch
from config import config


class WorldDynamicsTrainer:
    """
    World dynamics network trainer.

    The world model W approximates physics simulation as a differentiable model:
    - Input: Local(P_i) + T_i (current state + PD targets)
    - Output: local space acceleration
    - Loss computed in WORLD space
    """

    def __init__(self, world_dynamics_net, optimizer, replay_buffer):
        self.net = world_dynamics_net
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer

    def train(self, epochs=1):
        """
        Train world dynamics network.

        Algorithm:
        - Sample 8-frame chunks from buffer
        - P_0 ← S_0 (initialize from ground truth)
        - For i=1..7: predict P_i, compare with S_i
        """
        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # ===== Step 1: Sample batch from buffer =====
            pos, vel, rot, ang_vel, dt, ctrl = self.replay_buffer.pop_world()

            # ===== Step 2: Initialize prediction from ground truth =====
            # P_0 ← S_0: Use first frame as initial state
            pred_pos = pos[0].detach()
            pred_vel = vel[0].detach()
            pred_rot = rot[0].detach()
            pred_ang = ang_vel[0].detach()

            # ===== Step 3: Rollout 7 steps =====
            total_loss = 0.0

            for i in range(1, 8):
                # Convert to local space for network input
                local_state = local_torch(pred_pos, pred_vel, pred_rot, pred_ang)

                # Predict acceleration in local space
                acc, ang_acc = self.net(local_state, ctrl[i])

                # Transform local acceleration to world space
                # v_world = v_local @ R^T (row vector convention)
                root_rot = quaternion_torch.to_rotation_matrix(pred_rot[..., 0:1, :])
                combined = torch.stack([acc, ang_acc], dim=-2) @ root_rot.transpose(-2, -1)
                acc_world = combined[..., 0, :]
                ang_acc_world = combined[..., 1, :]

                # Integrate to get next state
                pred_pos, pred_vel, pred_rot, pred_ang = integrate_torch(
                    pred_pos, pred_vel, pred_rot, pred_ang,
                    acc_world, ang_acc_world, dt[i]
                )

                # Compute loss in WORLD space
                loss_pos = torch.abs(pos[i] - pred_pos).mean()
                loss_vel = torch.abs(vel[i] - pred_vel).mean()
                loss_rot = torch.abs(quaternion_torch.quat_log_residual(rot[i], pred_rot)).mean()
                loss_ang = torch.abs(ang_vel[i] - pred_ang).mean()

                total_loss += (
                    config.loss_weights.world['pos'] * loss_pos +
                    config.loss_weights.world['vel'] * loss_vel +
                    config.loss_weights.world['rot'] * loss_rot +
                    config.loss_weights.world['ang'] * loss_ang
                )

            # ===== Step 4: Backward and optimize =====
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.net.parameters(),
                max_norm=config.optimizer.gradient_clip_max_norm
            )
            self.optimizer.step()
