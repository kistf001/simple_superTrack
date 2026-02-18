"""
Policy control network training.
SuperTrack Algorithm 2 implementation.
"""

import torch
from utils.quaternion import torch as quaternion_torch
from transforms import local_torch, local_torch_components, integrate_torch
from config import config


class PolicyTrainer:
    """
    Policy control network trainer.

    The policy network P generates joint corrections:
    - Input: Local(P_{i-1}) + Local(K_i) (predicted state + target state)
    - Output: O (joint position corrections)
    - Loss computed in LOCAL space (rotation/translation invariant)
    """

    def __init__(self, policy_net, world_net, optimizer, replay_buffer):
        self.policy_net = policy_net
        self.world_net = world_net
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.iteration_count = 0

    def _calculate_reg_weights(self):
        """
        Calculate decayed regularization weights.

        3-phase schedule: hold -> exponential decay -> hold
        - Phase 1 (0 ~ decay_start): Use initial weights
        - Phase 2 (decay_start ~ decay_end): Exponential decay
        - Phase 3 (decay_end ~ inf): Use final weights
        """
        decay_cfg = config.loss_weights.policy.reg_decay

        if self.iteration_count >= decay_cfg.decay_end:
            # Phase 3: Final values
            return decay_cfg.l1_final, decay_cfg.l2_final
        elif self.iteration_count < decay_cfg.decay_start:
            # Phase 1: Initial values
            return decay_cfg.l1_initial, decay_cfg.l2_initial
        else:
            # Phase 2: Exponential decay
            progress = (self.iteration_count - decay_cfg.decay_start) / (
                decay_cfg.decay_end - decay_cfg.decay_start
            )
            w_l1 = decay_cfg.l1_initial * (decay_cfg.l1_final / decay_cfg.l1_initial) ** progress
            w_l2 = decay_cfg.l2_initial * (decay_cfg.l2_final / decay_cfg.l2_initial) ** progress
            return w_l1, w_l2

    def train(self, epochs=1000):
        """
        Train policy control network.

        Algorithm (SuperTrack Algorithm 2):
        - Sample 32-frame chunks from buffer
        - P_0 <- S_0 (initialize from ground truth)
        - For i=1..31:
            * Policy: [Local(P_{i-1}), Local(K_i)] -> O
            * Add noise: O_hat = O + epsilon
            * World model: Local(P_{i-1}) + target_qpos -> acc
            * Integrate: P_i = Integrate(P_{i-1}, acc, dt)
            * Loss: ||Local(P_i) - Local(K_i)||_1
        """
        total_losses = {
            "total": 0.0, "pos": 0.0, "vel": 0.0, "rot": 0.0,
            "ang": 0.0, "height": 0.0, "normal": 0.0,
            "l1_reg": 0.0, "l2_reg": 0.0,
        }
        max_total_loss = 0.0

        for epoch in range(epochs):
            self.iteration_count += 1
            self.optimizer.zero_grad()

            # ===== Step 1: Sample batch from buffer =====
            pos, vel, rot, ang_vel, dt, _, pos_gt, vel_gt, rot_gt, ang_gt, joint_qpos_gt = (
                self.replay_buffer.pop_policy()
            )

            # ===== Step 2: Initialize prediction from ground truth =====
            # P_0 <- S_0: Use first frame as initial state
            pred_pos = pos[0].detach()
            pred_vel = vel[0].detach()
            pred_rot = rot[0].detach()
            pred_ang = ang_vel[0].detach()

            # ===== Step 3: Rollout 31 steps =====
            corrections = []
            loss_pos_sum = 0.0
            loss_vel_sum = 0.0
            loss_rot_sum = 0.0
            loss_ang_sum = 0.0
            loss_height_sum = 0.0
            loss_up_sum = 0.0

            for i in range(1, 32):
                # Convert current prediction to local space
                local_P = local_torch(pred_pos, pred_vel, pred_rot, pred_ang)

                # Convert target (GT) to local space
                local_K = local_torch(pos_gt[i], vel_gt[i], rot_gt[i], ang_gt[i])

                # Policy inference: [Local(P_{i-1}), Local(K_i)] -> O
                combined = torch.cat([local_P, local_K], dim=-1)
                O = self.policy_net(combined)
                corrections.append(O)

                # Add exploration noise: O_hat = O + epsilon
                noise = torch.randn_like(O) * config.policy_training.noise_std
                O_hat = O + noise

                # Compute target joint positions
                target_qpos = joint_qpos_gt[i] + O_hat

                # World model prediction: Local(P_{i-1}) + target_qpos -> acc
                acc, ang_acc = self.world_net(local_P, target_qpos)

                # Transform local acceleration to world space
                # v_world = v_local @ R^T (row vector convention)
                root_rot = quaternion_torch.to_rotation_matrix(pred_rot[..., 0:1, :])
                combined_acc = torch.stack([acc, ang_acc], dim=-2) @ root_rot.transpose(-2, -1)
                acc_world = combined_acc[..., 0, :]
                ang_acc_world = combined_acc[..., 1, :]

                # Integrate to get next state
                pred_pos, pred_vel, pred_rot, pred_ang = integrate_torch(
                    pred_pos, pred_vel, pred_rot, pred_ang,
                    acc_world, ang_acc_world, dt[i - 1]
                )

                # Compute loss in LOCAL space (rotation/translation invariant)
                lpos_pred, lvel_pred, lrot_pred, lang_pred, hei_pred, up_pred = (
                    local_torch_components(pred_pos, pred_vel, pred_rot, pred_ang)
                )
                lpos_gt, lvel_gt, lrot_gt, lang_gt, hei_gt, up_gt = (
                    local_torch_components(pos_gt[i], vel_gt[i], rot_gt[i], ang_gt[i])
                )

                loss_pos_sum += torch.abs(lpos_pred - lpos_gt).mean()
                loss_vel_sum += torch.abs(lvel_pred - lvel_gt).mean()
                loss_rot_sum += torch.abs(lrot_pred - lrot_gt).mean()
                loss_ang_sum += torch.abs(lang_pred - lang_gt).mean()
                loss_height_sum += torch.abs(hei_pred - hei_gt).mean()
                loss_up_sum += torch.abs(up_pred - up_gt).mean()

            # ===== Step 4: Compute total loss =====
            # Average over 31 steps
            loss_pos = loss_pos_sum / 31
            loss_vel = loss_vel_sum / 31
            loss_rot = loss_rot_sum / 31
            loss_ang = loss_ang_sum / 31
            loss_height = loss_height_sum / 31
            loss_up = loss_up_sum / 31

            # Regularization losses
            reg_tensor = torch.stack(corrections)
            loss_l1_reg = torch.abs(reg_tensor).mean()
            loss_l2_reg = torch.square(reg_tensor).mean()

            # Get loss weights
            w_l1_reg, w_l2_reg = self._calculate_reg_weights()

            total_loss = (
                config.loss_weights.policy['pos'] * loss_pos +
                config.loss_weights.policy['vel'] * loss_vel +
                config.loss_weights.policy['rot'] * loss_rot +
                config.loss_weights.policy['ang'] * loss_ang +
                config.loss_weights.policy['height'] * loss_height +
                config.loss_weights.policy['normal'] * loss_up +
                w_l1_reg * loss_l1_reg +
                w_l2_reg * loss_l2_reg
            )

            # ===== Step 5: Backward and optimize =====
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy_net.parameters(),
                max_norm=config.optimizer.gradient_clip_max_norm
            )
            self.optimizer.step()

            # Accumulate losses for reporting
            current_total = total_loss.item()
            total_losses["total"] += current_total
            total_losses["pos"] += loss_pos.item()
            total_losses["vel"] += loss_vel.item()
            total_losses["rot"] += loss_rot.item()
            total_losses["ang"] += loss_ang.item()
            total_losses["height"] += loss_height.item()
            total_losses["normal"] += loss_up.item()
            total_losses["l1_reg"] += loss_l1_reg.item()
            total_losses["l2_reg"] += loss_l2_reg.item()
            max_total_loss = max(max_total_loss, current_total)

        # Return average losses
        avg_losses = {key: total_losses[key] / epochs for key in total_losses}
        return avg_losses, max_total_loss
