import numpy as np
import torch
from utils.quaternion import numpy as quaternion_numpy
from utils.quaternion import torch as quaternion_torch


def local_numpy(pos, vel, rot, ang_vel):
    """Transform world coordinates to local coordinates using NumPy.

    NOTE: This function expects single samples (NOT batched).
    For batched processing, use local_torch instead.

    Args:
        pos: Body positions in world frame (nbody, 3) - single sample only
        vel: Body velocities in world frame (nbody, 3) - single sample only
        rot: Body rotations as quaternions (nbody, 4) - format: (w, x, y, z) - single sample only
        ang_vel: Body angular velocities in world frame (nbody, 3) - single sample only

    Returns:
        np.ndarray: Flattened concatenated vector containing all local features
            [local_pos, local_vel, local_rot_two_axis, local_ang_vel, local_pos_z, heading_vec]
            Total size: nbody*3 + nbody*3 + nbody*6 + nbody*3 + nbody + 3
    """
    # Get rotation matrices from quaternions
    rot_matrices = quaternion_numpy.to_rotation_matrix(rot)  # (nbody, 3, 3)

    # Root transform (first body)
    root_pos = pos[0]
    root_rot = rot_matrices[0]
    root_rot_T = root_rot.T  # For rotation matrix transformation (R_root^T @ R_body)

    # Local coordinates transformation
    # World→Local: v_local = v_world @ R_root (row vector convention)
    local_pos = (pos - root_pos) @ root_rot
    local_vel = vel @ root_rot
    local_ang_vel = ang_vel @ root_rot

    # Local rotations (two-axis representation)
    local_rot = (
        root_rot_T @ rot_matrices
    )  # NOTE: NumPy uses @ for matrix multiplication
    local_rot_two_axis = local_rot[:, :, :2].reshape(
        -1, 6
    )  # NOTE: NumPy reshape syntax

    # Z-coordinate extraction
    local_pos_z = local_pos[:, 2:3]  # Extract Z-coordinates only

    # Heading vector: root's forward direction (X-axis) in world frame
    # This is simply the first column of root rotation matrix
    heading_vec = root_rot[:, 0]

    # Concatenate all features
    # NOTE: NumPy uses np.concatenate and flatten() method
    flattened = np.concatenate(
        [
            local_pos.flatten(),
            local_vel.flatten(),
            local_rot_two_axis.flatten(),
            local_ang_vel.flatten(),
            local_pos_z.flatten(),
            heading_vec,
        ]
    )

    return flattened


def local_torch(pos, vel, rot, ang_vel):
    """Transform world coordinates to local coordinates using PyTorch.

    NOTE: This function always expects batch input.
    For single samples, use local_numpy instead.

    Args:
        pos: Body positions (..., nbody, 3) - always batched
        vel: Body velocities (..., nbody, 3) - always batched
        rot: Body rotations as quaternions (..., nbody, 4) - format: (w, x, y, z) - always batched
        ang_vel: Body angular velocities (..., nbody, 3) - always batched

    Returns:
        torch.Tensor: Flattened concatenated vector containing all local features
            [local_pos, local_vel, local_rot_two_axis, local_ang_vel, local_pos_z, heading_vec]
            Shape: (..., feature_dim) where feature_dim = nbody*3 + nbody*3 + nbody*6 + nbody*3 + nbody + 3
    """
    # Get rotation matrices from quaternions
    rot_matrices = quaternion_torch.to_rotation_matrix(
        rot
    )  # (..., nbody, 3, 3)

    # Root transform (first body)
    # NOTE: PyTorch uses slicing [..., 0:1, :] to keep dimensions
    root_pos = pos[..., 0:1, :]
    root_rot = rot_matrices[..., 0, :, :]  # For vector transformation (World→Local)
    root_rot_T = root_rot.transpose(-2, -1)  # For rotation matrix transformation (R_root^T @ R_body)

    # Local coordinates transformation
    # World→Local: v_local = v_world @ R_root (row vector convention)
    local_pos = (pos - root_pos) @ root_rot
    local_vel = vel @ root_rot
    local_ang_vel = ang_vel @ root_rot

    # Local rotations (two-axis representation)
    # NOTE: PyTorch uses unsqueeze to add dimension for broadcasting
    local_rot = root_rot_T.unsqueeze(-3) @ rot_matrices
    local_rot_two_axis = local_rot[..., :, :2].flatten(
        start_dim=-3
    )  # NOTE: PyTorch flatten with start_dim

    # Heading vector: root's forward direction (X-axis) in world frame
    # This is simply the first column of root rotation matrix
    heading = rot_matrices[..., 0, :, 0]

    # Concatenate all features
    # NOTE: PyTorch uses torch.cat and flatten(start_dim=-2) for batched data
    return torch.cat(
        [
            local_pos.flatten(start_dim=-2),
            local_vel.flatten(start_dim=-2),
            local_rot_two_axis,
            local_ang_vel.flatten(start_dim=-2),
            local_pos[..., 2:3].flatten(
                start_dim=-2
            ),  # Z-coordinate extraction
            heading,
        ],
        dim=-1,
    )


def local_torch_components(pos, vel, rot, ang_vel):
    """Transform world coordinates to local coordinates, returning separate components.

    Used for LOCAL space loss computation (SuperTrack Algorithm 2).

    Args:
        pos: Body positions (..., nbody, 3)
        vel: Body velocities (..., nbody, 3)
        rot: Body rotations as quaternions (..., nbody, 4) - format: (w, x, y, z)
        ang_vel: Body angular velocities (..., nbody, 3)

    Returns:
        tuple: (local_pos, local_vel, local_rot_two_axis, local_ang_vel, heights, up_vector)
            - local_pos: (..., nbody, 3)
            - local_vel: (..., nbody, 3)
            - local_rot_two_axis: (..., nbody, 6)
            - local_ang_vel: (..., nbody, 3)
            - heights: (..., nbody)
            - up_vector: (..., 3)
    """
    # Get rotation matrices from quaternions
    rot_matrices = quaternion_torch.to_rotation_matrix(rot)  # (..., nbody, 3, 3)

    # Root transform (first body)
    root_pos = pos[..., 0:1, :]
    root_rot = rot_matrices[..., 0, :, :]  # For vector transformation (World→Local)
    root_rot_T = root_rot.transpose(-2, -1)  # For rotation matrix transformation (R_root^T @ R_body)

    # Local coordinates transformation
    # World→Local: v_local = v_world @ R_root (row vector convention)
    local_pos = (pos - root_pos) @ root_rot
    local_vel = vel @ root_rot
    local_ang_vel = ang_vel @ root_rot

    # Local rotations (two-axis representation) - keep shape (..., nbody, 6)
    local_rot = root_rot_T.unsqueeze(-3) @ rot_matrices
    local_rot_two_axis = local_rot[..., :, :2].flatten(start_dim=-2)  # (..., nbody, 6)
    # Reshape to (..., nbody, 6)
    nbody = pos.shape[-2]
    local_rot_two_axis = local_rot_two_axis.view(*pos.shape[:-2], nbody, 6)

    # Heights (Z-coordinate of local positions)
    heights = local_pos[..., 2]  # (..., nbody)

    # Up vector: gravity direction in local space
    # SuperTrack: x^up = Inv(root_rot) ⊗ [0, 0, 1]^T
    # World→Local: v_local = v_world @ R_root (row vector convention)
    up_world = torch.tensor([0.0, 0.0, 1.0], device=pos.device, dtype=pos.dtype)
    up_vector = up_world @ root_rot  # (..., 3)

    return local_pos, local_vel, local_rot_two_axis, local_ang_vel, heights, up_vector


def integrate_torch(pos, vel, rot, ang_vel, acc, ang_acc, dt):
    """Integrate position and rotation using accelerations (PyTorch version).

    Args:
        pos: Current positions (..., nbody, 3)
        vel: Current velocities (..., nbody, 3)
        rot: Current rotations as quaternions (..., nbody, 4) - format: (w, x, y, z)
        ang_vel: Current angular velocities (..., nbody, 3)
        acc: Linear accelerations (..., nbody, 3)
        ang_acc: Angular accelerations (..., nbody, 3)
        dt: Time step - can be:
            - scalar (float)
            - tensor with shape (..., 1, 1) for broadcasting
            - tensor with shape matching batch dimensions

    Returns:
        tuple: (new_pos, new_vel, new_rot, new_ang_vel) - Updated states
    """
    # Handle dt shape for proper broadcasting
    # If dt has shape (..., 1, 1), it will broadcast correctly
    # If dt is scalar, it will also work

    # Integrate velocities using accelerations
    new_vel = vel + acc * dt

    # Integrate angular velocities using angular accelerations
    new_ang_vel = ang_vel + ang_acc * dt

    # Integrate positions using new velocities
    new_pos = pos + new_vel * dt

    # Integrate rotations using angular velocities
    # quat_integrate_angular_velocity can handle tensor dt
    new_rot = quaternion_torch.quat_integrate_angular_velocity(
        new_ang_vel, rot, dt
    )

    return new_pos, new_vel, new_rot, new_ang_vel
