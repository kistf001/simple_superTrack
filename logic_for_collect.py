"""
Data collection for training.
Simulates trajectories using policy network and collects state transitions.
"""

import numpy as np
import torch
import mujoco
import pickle
from pathlib import Path
from utils.quaternion import numpy as quaternion_numpy
from transforms import local_numpy
from config import config


MIN_FRAMES = 1024


def generate_standing_gt_data(model, num_frames=1000):
    """
    Generate standing pose ground truth data.

    Args:
        model: MuJoCo model
        num_frames: Number of frames to generate

    Returns:
        tuple: (gt_data list, initial_qpos)
    """
    data = mujoco.MjData(model)
    standing_qpos = np.zeros(model.nq)
    standing_qpos[2] = 1.282  # Height
    standing_qpos[3] = 1.0    # Quaternion w

    data.qpos[:] = standing_qpos
    mujoco.mj_forward(model, data)

    pos = data.xpos[1:].copy()
    rot = data.xquat[1:].copy()
    vel = np.zeros_like(pos)
    ang_vel = np.zeros((pos.shape[0], 3))
    joint_qpos = standing_qpos[7:]

    gt_data = [
        (pos, vel, rot, ang_vel, joint_qpos, standing_qpos.copy())
        for _ in range(num_frames)
    ]
    return gt_data, standing_qpos.copy()


def load_motion_file(model, pkl_path):
    """
    Load a single motion file and compute velocities.

    Args:
        model: MuJoCo model
        pkl_path: Path to pkl file

    Returns:
        tuple: (gt_data list, initial_qpos) or (None, None) if < MIN_FRAMES
    """
    with open(pkl_path, "rb") as f:
        motion_data = pickle.load(f)

    qpos_array = motion_data["qpos"]
    if len(qpos_array) < MIN_FRAMES:
        return None, None

    fps = motion_data["fps"]
    dt = 1.0 / fps
    initial_qpos = qpos_array[0].copy()

    # Forward kinematics for each frame
    data = mujoco.MjData(model)
    positions = []
    rotations = []
    for qpos in qpos_array:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        positions.append(data.xpos[1:].copy())
        rotations.append(data.xquat[1:].copy())

    positions = np.array(positions)
    rotations = np.array(rotations)

    # Compute velocities via finite differences
    velocities = np.zeros_like(positions)
    velocities[1:] = (positions[1:] - positions[:-1]) / dt

    # Compute angular velocities from quaternion differences
    angular_vels = np.zeros((len(qpos_array), positions.shape[1], 3))
    for i in range(1, len(qpos_array)):
        angular_vels[i] = quaternion_numpy.quat_differentiate_angular_velocity(
            rotations[i], rotations[i - 1], dt
        )

    # Build gt_data list
    joint_qpos_array = qpos_array[:, 7:]
    gt_data = [
        (positions[i], velocities[i], rotations[i], angular_vels[i],
         joint_qpos_array[i], qpos_array[i])
        for i in range(len(qpos_array))
    ]
    return gt_data, initial_qpos


def load_all_motions(model, motions_dir=None):
    """
    Load all motion files from directory (1024+ frames only).

    Args:
        model: MuJoCo model
        motions_dir: Directory containing pkl files

    Returns:
        tuple: (motions dict, initial_qpos)
    """
    if motions_dir is None:
        motions_dir = config.data_collection.motions_dir

    pkl_files = sorted(Path(motions_dir).glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No pkl files in {motions_dir}")

    motions = {}
    initial_qpos = None

    for pkl_file in pkl_files:
        gt_data, qpos = load_motion_file(model, pkl_file)
        if gt_data is not None:
            motions[pkl_file.stem] = gt_data
            if initial_qpos is None:
                initial_qpos = qpos

    return motions, initial_qpos


def load_gt_data(model, motions_dir=None):
    """
    Load first motion file (for visualization).

    Args:
        model: MuJoCo model
        motions_dir: Directory containing pkl files

    Returns:
        tuple: (gt_data list, initial_qpos)
    """
    if motions_dir is None:
        motions_dir = config.data_collection.motions_dir

    pkl_files = sorted(Path(motions_dir).glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No pkl files in {motions_dir}")

    gt_data, initial_qpos = load_motion_file(model, pkl_files[0])
    return gt_data, initial_qpos


class DataCollector:
    """
    Collects training data by simulating trajectories.

    Uses policy network to generate actions and collects
    (state, action, ground_truth) tuples for training.
    """

    def __init__(self, env, policy_net, motions_dir=None, use_standing_pose=False):
        self.env = env
        self.policy_net = policy_net

        if use_standing_pose:
            gt_data, self.initial_qpos = generate_standing_gt_data(env.model)
            self.motions = {"standing": gt_data}
        else:
            if motions_dir is None:
                motions_dir = config.data_collection.motions_dir
            self.motions, self.initial_qpos = load_all_motions(env.model, motions_dir)

        self.motion_names = list(self.motions.keys())

    def _should_reset(self, pos_gt):
        """Check if simulation should reset based on head height deviation."""
        HEAD_IDX = 1
        HEAD_HEIGHT_THRESHOLD = 0.30
        head_height_diff = abs(self.pos[HEAD_IDX, 2] - pos_gt[HEAD_IDX, 2])
        return head_height_diff > HEAD_HEIGHT_THRESHOLD

    def _simulate_trajectory(self, num_frames=1024):
        """
        Simulate a trajectory using policy network.

        Returns:
            list: Trajectory data [(pos, vel, rot, ang_vel, dt, ctrl, gt_data), ...]
        """
        CHUNK_SIZE = config.buffer.policy_chunk_size
        trajectory = [None] * num_frames

        # Select random motion
        motion_name = np.random.choice(self.motion_names)
        gt_data = self.motions[motion_name]

        # Initialize environment
        self.reset_flag = False
        initial_qpos = gt_data[0][5]
        self.pos, self.vel, self.rot, self.ang_vel = self.env.reset(qpos=initial_qpos)

        # Store initial state
        gt_data_0 = gt_data[0]
        _, _, _, _, gt_joint_qpos_0, _ = gt_data_0
        initial_pos = self.pos.copy()
        initial_vel = self.vel.copy()
        initial_rot = self.rot.copy()
        initial_ang = self.ang_vel.copy()
        dt_sum = 0.0

        # Simulate trajectory
        for i in range(1, num_frames):
            gt_data_curr = gt_data[i % len(gt_data)]

            # Reset at chunk boundaries if needed
            if (i % CHUNK_SIZE) == 0 and self.reset_flag:
                _, _, _, _, _, gt_qpos_current = gt_data_curr
                self.pos, self.vel, self.rot, self.ang_vel = self.env.reset(qpos=gt_qpos_current)
                self.reset_flag = False

            # Check for reset condition
            if self._should_reset(gt_data_curr[0]):
                self.reset_flag = True

            # Get ground truth data
            pos_gt, vel_gt, rot_gt, ang_gt, gt_joint_qpos, _ = gt_data_curr

            # Compute policy input (local space)
            local_P = local_numpy(self.pos, self.vel, self.rot, self.ang_vel).astype(np.float32)
            local_K = local_numpy(pos_gt, vel_gt, rot_gt, ang_gt).astype(np.float32)
            combined_obs = np.concatenate([local_P, local_K])

            # Policy inference
            combined_obs_torch = torch.from_numpy(combined_obs).unsqueeze(0)
            O = self.policy_net(combined_obs_torch).squeeze(0).cpu().numpy()

            # Add exploration noise
            noise = np.random.randn(*O.shape).astype(np.float32) * config.policy_training.noise_std
            O_hat = O + noise

            # Step environment
            target_qpos = gt_joint_qpos + O_hat
            self.pos, self.vel, self.rot, self.ang_vel, dt = self.env.step(target_qpos)
            dt_sum += dt

            # Store trajectory data
            trajectory[i] = (
                self.pos.copy(), self.vel.copy(), self.rot.copy(), self.ang_vel.copy(),
                dt, target_qpos.copy(), gt_data_curr
            )

        # Store initial frame with average dt
        dt_avg = dt_sum / (num_frames - 1) if num_frames > 1 else 0.01
        trajectory[0] = (
            initial_pos, initial_vel, initial_rot, initial_ang,
            dt_avg, gt_joint_qpos_0.copy(), gt_data_0
        )
        return trajectory

    def _sample_from_zones(self, trajectory, num_samples=8, zones_per_sample=16):
        """
        Sample data from trajectory using zone-based sampling.

        Divides trajectory into zones and samples contiguous zone sequences.
        """
        ZONE_SIZE = config.buffer.world_chunk_size
        total_zones = len(trajectory) // ZONE_SIZE
        POLICY_CHUNK = config.buffer.policy_chunk_size

        max_start_zone = total_zones - zones_per_sample
        start_zones = np.random.choice(max_start_zone + 1, size=num_samples, replace=False)

        collected_data = []
        for start_zone in start_zones:
            for zone_offset in range(zones_per_sample):
                zone_idx = start_zone + zone_offset
                frame_start = zone_idx * ZONE_SIZE
                for frame_offset in range(ZONE_SIZE):
                    idx = frame_start + frame_offset
                    pos, vel, rot, ang_vel, dt, ctrl, gt_data = trajectory[idx]
                    global_offset = zone_offset * ZONE_SIZE + frame_offset
                    frame_idx = global_offset % POLICY_CHUNK
                    collected_data.append((frame_idx, pos, vel, rot, ang_vel, dt, ctrl, gt_data))

        return collected_data

    def collect(self, num_samples=1024):
        """
        Collect training data.

        Returns:
            list: Collected data tuples
        """
        NUM_SAMPLES = 8
        ZONES_PER_SAMPLE = 16

        with torch.no_grad():
            trajectory = self._simulate_trajectory(num_samples)
            collected_data = self._sample_from_zones(trajectory, NUM_SAMPLES, ZONES_PER_SAMPLE)

        return collected_data
