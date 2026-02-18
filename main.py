"""
Single-process training for robot control.

Implements SuperTrack-style training with sequential:
1. Data collection (simulation + policy inference)
2. World model training (physics prediction)
3. Policy training (control generation)

Usage:
    python main.py
    python main.py --config configs/exp.toml
"""

import argparse
import os
import torch

# CLI parsing before config import (sets CONFIG_FILE env var)
parser = argparse.ArgumentParser(description="Single-process training")
parser.add_argument("--config", "-c", type=str, help="TOML config path")
args, _ = parser.parse_known_args()

if args.config:
    os.environ["CONFIG_FILE"] = args.config

from config import config
from env import ObservationCollector
from network import PolicyControlNetwork, WorldDynamicsNetwork
from buffer import ReplayBuffer
from logic_for_collect import DataCollector
from logic_for_world import WorldDynamicsTrainer
from logic_for_policy import PolicyTrainer


def main():
    """
    Main training loop.

    Training pipeline:
    1. Initialize environment, networks, buffer
    2. Fill buffer with initial data
    3. Train for 150000 iterations:
       - Collect new trajectory data
       - Train world dynamics model
       - Train policy network
       - Sync networks periodically
    4. Save final models
    """
    # ===== Environment Setup =====
    env = ObservationCollector(config.process.model_path)
    device = config.process.device

    # ===== Network Initialization =====
    world_net = WorldDynamicsNetwork(
        observation_size=env.local_size,
        control_size=env.ctrl_size,
        hidden_size=config.network_architecture.world_hidden_size,
        num_bodies=env.real_nbody,
        device=device,
    )

    policy_net = PolicyControlNetwork(
        observation_size=env.local_size * 2,
        control_size=env.ctrl_size,
        device=device,
    )

    # ===== Optimizer Setup =====
    world_optimizer = torch.optim.RAdam(
        world_net.parameters(),
        lr=config.optimizer.world_learning_rate
    )
    policy_optimizer = torch.optim.RAdam(
        policy_net.parameters(),
        lr=config.optimizer.policy_learning_rate
    )

    # ===== Buffer Setup =====
    buffer = ReplayBuffer(
        capacity=config.process.buffer_size,
        world_chunk_size=config.buffer.world_chunk_size,
        policy_chunk_size=config.buffer.policy_chunk_size,
        nbody=env.real_nbody,
        ctrl_size=env.ctrl_size,
        device=device,
    )

    # ===== Data Collector Setup =====
    # Uses CPU policy for inference during data collection
    policy_net_cpu = PolicyControlNetwork(
        observation_size=env.local_size * 2,
        control_size=env.ctrl_size,
        device="cpu",
    )
    policy_net_cpu.eval()

    data_collector = DataCollector(
        env,
        policy_net_cpu,
        use_standing_pose=config.data_collection.use_standing_pose
    )

    # ===== Trainers =====
    world_trainer = WorldDynamicsTrainer(world_net, world_optimizer, buffer)
    policy_trainer = PolicyTrainer(policy_net, world_net, policy_optimizer, buffer)

    # ===== Initial Buffer Fill =====
    while len(buffer) < config.process.buffer_size:
        collected = data_collector.collect(num_samples=config.data_collection.total_samples)
        for data in collected:
            frame_idx, pos, vel, rot, ang_vel, dt, ctrl, gt_data = data
            buffer.push(frame_idx, pos, vel, rot, ang_vel, dt, ctrl, gt_data)

    # ===== Training Loop (150000 iterations) =====
    for iteration in range(1, 150001):

        # Collect new trajectory data
        with torch.no_grad():
            collected = data_collector.collect(num_samples=config.data_collection.total_samples)
            for data in collected:
                frame_idx, pos, vel, rot, ang_vel, dt, ctrl, gt_data = data
                buffer.push(frame_idx, pos, vel, rot, ang_vel, dt, ctrl, gt_data)

        # Train world dynamics model
        world_net.train()
        world_trainer.train(epochs=config.optimizer.training_epochs)

        # Train policy network
        policy_net.train()
        world_net.eval()  # World model used as frozen simulator
        policy_trainer.train(epochs=config.optimizer.training_epochs)

        # Sync policy weights to CPU for data collection
        if iteration % config.process.network_sync_period == 0:
            policy_net_cpu.load_state_dict(policy_net.state_dict())

        # Save checkpoint
        if iteration % config.training.model_save_interval == 0:
            torch.save(world_net.state_dict(), f"world_{iteration}.pth")
            torch.save(policy_net.state_dict(), f"policy_{iteration}.pth")

    # ===== Save Final Models =====
    torch.save(policy_net.state_dict(), "policy_final.pth")
    torch.save(world_net.state_dict(), "world_final.pth")


if __name__ == "__main__":
    main()
