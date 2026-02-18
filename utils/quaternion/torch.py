"""
PyTorch quaternion operations.

Format: (w, x, y, z) where w is the scalar part.
All operations are PyTorch-native with no type checking overhead.
"""

import torch


# =============================================================================
# Core Operations
# =============================================================================


def conjugate(q: torch.Tensor) -> torch.Tensor:
    """Compute quaternion conjugate: (w, -x, -y, -z)."""
    q_conj = q.clone()
    q_conj[..., 1:] = -q_conj[..., 1:]
    return q_conj


def multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions (Hamilton product)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)


def normalize(q: torch.Tensor) -> torch.Tensor:
    """Normalize quaternion to unit length."""
    norm = torch.norm(q, dim=-1, keepdim=True)
    return q / torch.clamp(norm, min=1e-8)


# =============================================================================
# Conversions
# =============================================================================

def from_euler(roll, pitch, yaw) -> torch.Tensor:
    """Convert Euler angles to quaternion (ZYX order)."""
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z], dim=-1)


def to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to 3x3 rotation matrix."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    matrix = torch.zeros(q.shape[:-1] + (3, 3), dtype=q.dtype, device=q.device)

    matrix[..., 0, 0] = 1 - 2 * (yy + zz)
    matrix[..., 0, 1] = 2 * (xy - wz)
    matrix[..., 0, 2] = 2 * (xz + wy)
    matrix[..., 1, 0] = 2 * (xy + wz)
    matrix[..., 1, 1] = 1 - 2 * (xx + zz)
    matrix[..., 1, 2] = 2 * (yz - wx)
    matrix[..., 2, 0] = 2 * (xz - wy)
    matrix[..., 2, 1] = 2 * (yz + wx)
    matrix[..., 2, 2] = 1 - 2 * (xx + yy)
    return matrix


def from_rotation_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Convert 3x3 rotation matrix to quaternion (Shepperd method)."""
    trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]

    q = torch.zeros(matrix.shape[:-2] + (4,), dtype=matrix.dtype, device=matrix.device)
    mask1 = trace > 0
    mask2 = (~mask1) & (matrix[..., 0, 0] > matrix[..., 1, 1]) & (matrix[..., 0, 0] > matrix[..., 2, 2])
    mask3 = (~mask1) & (~mask2) & (matrix[..., 1, 1] > matrix[..., 2, 2])
    mask4 = (~mask1) & (~mask2) & (~mask3)

    trace_safe = torch.where(mask1, trace, torch.zeros_like(trace))
    s1 = 2.0 * torch.sqrt(torch.clamp(trace_safe + 1.0, min=1e-8))
    s1 = torch.where(mask1, s1, torch.ones_like(s1))
    q[..., 0] = torch.where(mask1, 0.25 * s1, q[..., 0])
    q[..., 1] = torch.where(mask1, (matrix[..., 2, 1] - matrix[..., 1, 2]) / s1, q[..., 1])
    q[..., 2] = torch.where(mask1, (matrix[..., 0, 2] - matrix[..., 2, 0]) / s1, q[..., 2])
    q[..., 3] = torch.where(mask1, (matrix[..., 1, 0] - matrix[..., 0, 1]) / s1, q[..., 3])

    diag = torch.where(mask2, 1.0 + matrix[..., 0, 0] - matrix[..., 1, 1] - matrix[..., 2, 2], torch.zeros_like(trace))
    s2 = 2.0 * torch.sqrt(torch.clamp(diag, min=1e-8))
    s2 = torch.where(mask2, s2, torch.ones_like(s2))
    q[..., 0] = torch.where(mask2, (matrix[..., 2, 1] - matrix[..., 1, 2]) / s2, q[..., 0])
    q[..., 1] = torch.where(mask2, 0.25 * s2, q[..., 1])
    q[..., 2] = torch.where(mask2, (matrix[..., 0, 1] + matrix[..., 1, 0]) / s2, q[..., 2])
    q[..., 3] = torch.where(mask2, (matrix[..., 0, 2] + matrix[..., 2, 0]) / s2, q[..., 3])

    diag = torch.where(mask3, 1.0 + matrix[..., 1, 1] - matrix[..., 0, 0] - matrix[..., 2, 2], torch.zeros_like(trace))
    s3 = 2.0 * torch.sqrt(torch.clamp(diag, min=1e-8))
    s3 = torch.where(mask3, s3, torch.ones_like(s3))
    q[..., 0] = torch.where(mask3, (matrix[..., 0, 2] - matrix[..., 2, 0]) / s3, q[..., 0])
    q[..., 1] = torch.where(mask3, (matrix[..., 0, 1] + matrix[..., 1, 0]) / s3, q[..., 1])
    q[..., 2] = torch.where(mask3, 0.25 * s3, q[..., 2])
    q[..., 3] = torch.where(mask3, (matrix[..., 1, 2] + matrix[..., 2, 1]) / s3, q[..., 3])

    diag = torch.where(mask4, 1.0 + matrix[..., 2, 2] - matrix[..., 0, 0] - matrix[..., 1, 1], torch.zeros_like(trace))
    s4 = 2.0 * torch.sqrt(torch.clamp(diag, min=1e-8))
    s4 = torch.where(mask4, s4, torch.ones_like(s4))
    q[..., 0] = torch.where(mask4, (matrix[..., 1, 0] - matrix[..., 0, 1]) / s4, q[..., 0])
    q[..., 1] = torch.where(mask4, (matrix[..., 0, 2] + matrix[..., 2, 0]) / s4, q[..., 1])
    q[..., 2] = torch.where(mask4, (matrix[..., 1, 2] + matrix[..., 2, 1]) / s4, q[..., 2])
    q[..., 3] = torch.where(mask4, 0.25 * s4, q[..., 3])

    return normalize(q)


# =============================================================================
# Interpolation
# =============================================================================

def slerp(q1: torch.Tensor, q2: torch.Tensor, t) -> torch.Tensor:
    """Spherical linear interpolation between two quaternions."""
    q1, q2 = normalize(q1), normalize(q2)

    dot = torch.sum(q1 * q2, dim=-1, keepdim=True)
    q2 = torch.where(dot < 0, -q2, q2)
    dot = torch.abs(dot)
    dot = torch.clamp(dot, min=-1.0, max=1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    small = sin_theta < 1e-6
    t1 = torch.where(small, 1.0 - t, torch.sin((1.0 - t) * theta) / sin_theta)
    t2 = torch.where(small, t, torch.sin(t * theta) / sin_theta)

    return normalize(t1 * q1 + t2 * q2)


# =============================================================================
# Utilities
# =============================================================================

def quat_invert(q: torch.Tensor) -> torch.Tensor:
    """Compute quaternion inverse (conjugate for unit quaternions)."""
    return conjugate(q)


def quat_abs(q: torch.Tensor) -> torch.Tensor:
    """Make quaternion scalar part positive (canonical form)."""
    mask = q[..., :1] < 0.0
    return torch.where(mask, -q, q)


# =============================================================================
# Exponential/Log Maps
# =============================================================================

def quat_exp(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Exponential map from R^3 to quaternion."""
    half_angle = torch.norm(v, dim=-1, keepdim=True)
    c = torch.cos(half_angle)
    s = torch.sin(half_angle) / torch.clamp(half_angle, min=eps)
    return torch.cat([c, s * v], dim=-1)


def quat_log(q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Logarithm map from unit quaternion to R^3."""
    w, v = q[..., :1], q[..., 1:]
    v_norm = torch.norm(v, dim=-1, keepdim=True)
    w_clamped = torch.clamp(w, min=-1.0 + eps, max=1.0 - eps)
    angle = torch.acos(w_clamped)
    return (v / torch.clamp(v_norm, min=eps)) * angle


# =============================================================================
# Angular Velocity
# =============================================================================

def quat_differentiate_angular_velocity(
    next_q: torch.Tensor, curr_q: torch.Tensor, dt: float = 1.0, eps: float = 1e-8
) -> torch.Tensor:
    """Compute angular velocity between two quaternions."""
    # Normalize
    next_q = next_q / torch.clamp(torch.norm(next_q, dim=-1, keepdim=True), min=eps)
    curr_q = curr_q / torch.clamp(torch.norm(curr_q, dim=-1, keepdim=True), min=eps)

    # Relative rotation: delta_q = next * curr^{-1}
    w1, v1 = next_q[..., 0], next_q[..., 1:]
    w2, v2 = curr_q[..., 0], -curr_q[..., 1:]

    dot_v = torch.sum(v1 * v2, dim=-1)
    cross_v = torch.cross(v1, v2, dim=-1)
    delta_w = w1 * w2 - dot_v
    delta_v = w1.unsqueeze(-1) * v2 + w2.unsqueeze(-1) * v1 + cross_v
    delta_q = torch.cat([delta_w.unsqueeze(-1), delta_v], dim=-1)
    delta_q = delta_q / torch.clamp(torch.norm(delta_q, dim=-1, keepdim=True), min=eps)
    sign = torch.where(delta_q[..., :1] < 0, -torch.ones_like(delta_q[..., :1]), torch.ones_like(delta_q[..., :1]))
    delta_q = delta_q * sign
    w, v = delta_q[..., :1], delta_q[..., 1:]
    v_norm = torch.norm(v, dim=-1, keepdim=True)
    half_angle = torch.atan2(v_norm, w)
    scale = torch.where(v_norm > eps, half_angle / v_norm, torch.ones_like(v_norm))

    return (2.0 / dt) * v * scale


def quat_integrate_angular_velocity(
    vel: torch.Tensor, curr_q: torch.Tensor, dt, eps: float = 1e-8
) -> torch.Tensor:
    """Integrate angular velocity to get new quaternion."""
    v = 0.5 * vel * dt

    theta = torch.norm(v, dim=-1, keepdim=True)
    s_over_t = torch.sin(theta) / torch.clamp(theta, min=eps)
    s_over_t = torch.where(theta > eps, s_over_t, 1.0 - (theta * theta) / 6.0)
    c = torch.cos(theta)
    delta_w, delta_v = c.squeeze(-1), s_over_t * v
    w1, v1 = delta_w, delta_v
    w2, v2 = curr_q[..., 0], curr_q[..., 1:]
    dot_v = torch.sum(v1 * v2, dim=-1)
    cross_v = torch.cross(v1, v2, dim=-1)
    new_w = w1 * w2 - dot_v
    new_v = w1.unsqueeze(-1) * v2 + w2.unsqueeze(-1) * v1 + cross_v
    new_q = torch.cat([new_w.unsqueeze(-1), new_v], dim=-1)
    return new_q / torch.clamp(torch.norm(new_q, dim=-1, keepdim=True), min=eps)


# =============================================================================
# Advanced Operations
# =============================================================================

def quat_log_residual(q_t: torch.Tensor, q_s: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute log residual between two quaternions."""
    q_t = q_t / torch.clamp(torch.norm(q_t, dim=-1, keepdim=True), min=eps)
    q_s = q_s / torch.clamp(torch.norm(q_s, dim=-1, keepdim=True), min=eps)
    w1, v1 = q_t[..., :1], q_t[..., 1:]
    w2, v2 = q_s[..., :1], -q_s[..., 1:]
    delta_w = w1 * w2 - torch.sum(v1 * v2, dim=-1, keepdim=True)
    delta_v = w1 * v2 + w2 * v1 + torch.cross(v1, v2, dim=-1)
    delta = torch.cat([delta_w, delta_v], dim=-1)
    delta = delta / torch.clamp(torch.norm(delta, dim=-1, keepdim=True), min=eps)
    delta = torch.where(delta[..., :1] < 0, -delta, delta)
    w, v = delta[..., :1], delta[..., 1:]
    v_norm = torch.norm(v, dim=-1, keepdim=True)
    half_angle = torch.atan2(v_norm, w)
    scale = torch.where(v_norm > eps, half_angle / v_norm, torch.ones_like(v_norm))

    return 2.0 * v * scale


def quat_geodesic_angle(q_t: torch.Tensor, q_s: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute geodesic angle between two quaternions."""
    q_t = q_t / torch.clamp(torch.norm(q_t, dim=-1, keepdim=True), min=eps)
    q_s = q_s / torch.clamp(torch.norm(q_s, dim=-1, keepdim=True), min=eps)
    d = torch.abs(torch.sum(q_t * q_s, dim=-1))
    d = torch.clamp(d, max=1.0 - eps)
    return 2.0 * torch.atan2(torch.sqrt(torch.clamp(1.0 - d * d, min=0.0)), d)
