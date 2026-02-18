"""
Replay buffer for single-process training.

Provides chunk-based circular buffer for storing simulation trajectories
and sampling batches for world model and policy training.
"""

import numpy as np
import torch
from typing import Tuple
from config import config


class ReplayBuffer:
    """
    Circular replay buffer with chunk-based sampling.

    Stores simulation trajectories and provides batched sampling
    for world model (8-frame) and policy (32-frame) training.

    The buffer is organized into fixed-size chunks (policy_chunk_size).
    Only complete chunks are committed and available for sampling.
    """

    def __init__(
        self,
        capacity: int,
        world_chunk_size: int,
        policy_chunk_size: int,
        nbody: int = 22,
        ctrl_size: int = 21,
        device: str = "cpu",
    ):
        """
        Initialize buffer with given capacity and chunk sizes.

        Args:
            capacity: Total buffer capacity (must be divisible by policy_chunk_size)
            world_chunk_size: Chunk size for world model training (typically 8)
            policy_chunk_size: Chunk size for policy training (typically 32)
            nbody: Number of bodies in the model
            ctrl_size: Control dimension (number of actuators)
            device: Device for output tensors
        """
        assert capacity % policy_chunk_size == 0, "Capacity must be divisible by policy_chunk_size"

        # ===== Buffer Configuration =====
        self.device = torch.device(device)
        self.capacity = capacity
        self.world_chunk_size = world_chunk_size
        self.policy_chunk_size = policy_chunk_size
        self.max_chunks = capacity // policy_chunk_size

        # ===== Simulation Data Buffers =====
        self.pos = np.zeros((capacity, nbody, 3), dtype=np.float32)
        self.vel = np.zeros((capacity, nbody, 3), dtype=np.float32)
        self.rot = np.zeros((capacity, nbody, 4), dtype=np.float32)
        self.ang = np.zeros((capacity, nbody, 3), dtype=np.float32)
        self.dt = np.zeros(capacity, dtype=np.float32)
        self.ctrl = np.zeros((capacity, ctrl_size), dtype=np.float32)

        # ===== Ground Truth Buffers =====
        self.pos_gt = np.zeros((capacity, nbody, 3), dtype=np.float32)
        self.vel_gt = np.zeros((capacity, nbody, 3), dtype=np.float32)
        self.rot_gt = np.zeros((capacity, nbody, 4), dtype=np.float32)
        self.ang_gt = np.zeros((capacity, nbody, 3), dtype=np.float32)
        self.joint_qpos_gt = np.zeros((capacity, ctrl_size), dtype=np.float32)

        # ===== Chunk Tracking =====
        # Circular buffer uses head/tail pointers for valid chunks
        self.write_ptr = 0              # Current write position
        self.chunk_start_ptr = 0        # Start of current chunk being written
        self.committed_size = 0         # Total committed frames
        self.valid_chunks = np.full(self.max_chunks, -1, dtype=np.int32)
        self.chunk_head = 0             # Next chunk slot to write
        self.chunk_tail = 0             # Oldest valid chunk
        self.num_valid_chunks = 0       # Number of valid chunks
        self.chunk_frame_count = 0      # Frames written in current chunk
        self.chunk_valid = True         # Current chunk validity flag

    def push(self, frame_idx: int, pos, vel, rot, ang, dt, ctrl, gt_data) -> bool:
        """
        Add a single frame to the buffer.

        Frames are accumulated into chunks. Only complete chunks with
        contiguous frame indices are committed to the buffer.

        Args:
            frame_idx: Frame index within trajectory (0 to policy_chunk_size-1)
            pos: Body positions [nbody, 3]
            vel: Body velocities [nbody, 3]
            rot: Body rotations (quaternions) [nbody, 4]
            ang: Body angular velocities [nbody, 3]
            dt: Timestep
            ctrl: Control input [ctrl_size]
            gt_data: Ground truth tuple (pos, vel, rot, ang, joint_qpos, qpos)

        Returns:
            True if a chunk was committed, False otherwise
        """
        frame_in_chunk = frame_idx % self.policy_chunk_size

        # ===== Handle Chunk Boundaries =====
        if frame_in_chunk == 0:
            # Finalize previous chunk if any
            if self.chunk_frame_count > 0:
                if self.chunk_frame_count == self.policy_chunk_size and self.chunk_valid:
                    self._commit_chunk()
                else:
                    self._rollback_chunk()
            # Start new chunk
            self.chunk_start_ptr = self.write_ptr
            self.chunk_frame_count = 0
            self.chunk_valid = True

        # ===== Validate Frame Continuity =====
        if not self.chunk_valid or frame_in_chunk != self.chunk_frame_count:
            self.chunk_valid = False
            return False

        # ===== Write Data =====
        idx = self.write_ptr
        self.pos[idx] = pos
        self.vel[idx] = vel
        self.rot[idx] = rot
        self.ang[idx] = ang
        self.dt[idx] = dt
        self.ctrl[idx] = ctrl

        # Write ground truth
        pos_gt, vel_gt, rot_gt, ang_gt, joint_qpos_gt, _ = gt_data
        self.pos_gt[idx] = pos_gt
        self.vel_gt[idx] = vel_gt
        self.rot_gt[idx] = rot_gt
        self.ang_gt[idx] = ang_gt
        self.joint_qpos_gt[idx] = joint_qpos_gt

        # ===== Advance Pointer =====
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.chunk_frame_count += 1

        # Auto-commit when chunk is complete
        if self.chunk_frame_count == self.policy_chunk_size:
            self._commit_chunk()
            return True
        return False

    def _commit_chunk(self):
        """Commit current chunk to valid chunk list."""
        if self.chunk_frame_count != self.policy_chunk_size:
            return

        # Record chunk start in circular valid_chunks array
        self.valid_chunks[self.chunk_head] = self.chunk_start_ptr
        self.chunk_head = (self.chunk_head + 1) % self.max_chunks

        # Update valid chunk count
        if self.num_valid_chunks < self.max_chunks:
            self.num_valid_chunks += 1
        else:
            # Overwrite oldest chunk
            self.chunk_tail = (self.chunk_tail + 1) % self.max_chunks

        self.committed_size = min(self.committed_size + self.policy_chunk_size, self.capacity)
        self.chunk_frame_count = 0
        self.chunk_valid = True

    def _rollback_chunk(self):
        """Rollback incomplete chunk by resetting write pointer."""
        self.write_ptr = self.chunk_start_ptr
        self.chunk_frame_count = 0
        self.chunk_valid = True

    def _get_valid_chunk_starts(self) -> np.ndarray:
        """
        Get array of valid chunk start indices.

        Returns:
            Array of buffer indices where valid chunks start
        """
        if self.num_valid_chunks == 0:
            return np.array([], dtype=np.int32)

        # Get chunk indices from tail to head in circular buffer
        indices = (self.chunk_tail + np.arange(self.num_valid_chunks)) % self.max_chunks
        starts = self.valid_chunks[indices]
        return starts[starts >= 0]

    def pop_world(self) -> Tuple[torch.Tensor, ...]:
        """
        Sample 8-frame chunks for world model training.

        Samples policy-sized chunks, then splits each into world-sized
        sub-chunks for training the world dynamics model.

        Returns:
            Tuple of tensors: (pos, vel, rot, ang, dt, ctrl)
            Each tensor has shape [chunk_size, batch, ...]
        """
        valid_starts = self._get_valid_chunk_starts()
        n_chunks = len(valid_starts)
        if n_chunks == 0:
            raise ValueError("No valid chunks available for sampling")

        # Sample policy chunks
        n_samples = max(1, int(n_chunks * config.buffer.world_sample_percent))
        selected = np.random.choice(valid_starts, size=min(n_samples, n_chunks), replace=False)

        # Split into world-sized sub-chunks
        sub_offsets = np.arange(0, self.policy_chunk_size, self.world_chunk_size, dtype=np.int32)
        sub_starts = (selected[:, None] + sub_offsets).ravel() % self.capacity

        # Generate frame indices for each sub-chunk
        offsets = np.arange(self.world_chunk_size, dtype=np.int32)
        idx = (sub_starts[:, None] + offsets) % self.capacity

        # Convert to tensors with shape [time, batch, ...]
        return (
            torch.from_numpy(self.pos[idx]).transpose(0, 1).to(self.device),
            torch.from_numpy(self.vel[idx]).transpose(0, 1).to(self.device),
            torch.from_numpy(self.rot[idx]).transpose(0, 1).to(self.device),
            torch.from_numpy(self.ang[idx]).transpose(0, 1).to(self.device),
            torch.from_numpy(self.dt[idx]).transpose(0, 1).unsqueeze(-1).unsqueeze(-1).to(self.device),
            torch.from_numpy(self.ctrl[idx]).transpose(0, 1).to(self.device),
        )

    def pop_policy(self) -> Tuple[torch.Tensor, ...]:
        """
        Sample 32-frame chunks for policy training.

        Returns:
            Tuple of tensors: (pos, vel, rot, ang, dt, ctrl,
                              pos_gt, vel_gt, rot_gt, ang_gt, joint_qpos_gt)
            Each tensor has shape [chunk_size, batch, ...]
        """
        valid_starts = self._get_valid_chunk_starts()
        n_chunks = len(valid_starts)
        if n_chunks == 0:
            raise ValueError("No valid chunks available for sampling")

        # Sample policy chunks
        n_samples = max(1, int(n_chunks * config.buffer.policy_sample_percent))
        selected = np.random.choice(valid_starts, size=min(n_samples, n_chunks), replace=False)

        # Generate frame indices
        offsets = np.arange(self.policy_chunk_size, dtype=np.int32)
        idx = (selected[:, None] + offsets) % self.capacity

        # Convert to tensors with shape [time, batch, ...]
        return (
            torch.from_numpy(self.pos[idx]).transpose(0, 1).to(self.device),
            torch.from_numpy(self.vel[idx]).transpose(0, 1).to(self.device),
            torch.from_numpy(self.rot[idx]).transpose(0, 1).to(self.device),
            torch.from_numpy(self.ang[idx]).transpose(0, 1).to(self.device),
            torch.from_numpy(self.dt[idx]).transpose(0, 1).unsqueeze(-1).unsqueeze(-1).to(self.device),
            torch.from_numpy(self.ctrl[idx]).transpose(0, 1).to(self.device),
            torch.from_numpy(self.pos_gt[idx]).transpose(0, 1).to(self.device),
            torch.from_numpy(self.vel_gt[idx]).transpose(0, 1).to(self.device),
            torch.from_numpy(self.rot_gt[idx]).transpose(0, 1).to(self.device),
            torch.from_numpy(self.ang_gt[idx]).transpose(0, 1).to(self.device),
            torch.from_numpy(self.joint_qpos_gt[idx]).transpose(0, 1).to(self.device),
        )

    def __len__(self) -> int:
        """Return number of committed frames in buffer."""
        return self.committed_size
