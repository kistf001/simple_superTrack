"""
NumPy quaternion operations.

Format: (w, x, y, z) where w is the scalar part.
All operations are NumPy-native with no type checking overhead.
"""

import numpy as np


# =============================================================================
# Core Operations
# =============================================================================

def conjugate(q: np.ndarray) -> np.ndarray:
    """Compute quaternion conjugate: (w, -x, -y, -z)."""
    q_conj = q.copy()
    q_conj[..., 1:] = -q_conj[..., 1:]
    return q_conj


def multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (Hamilton product)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.stack([w, x, y, z], axis=-1)


def normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit length."""
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(norm, 1e-8)


# =============================================================================
# Conversions
# =============================================================================

def from_euler(roll, pitch, yaw) -> np.ndarray:
    """Convert Euler angles to quaternion (ZYX order)."""
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.stack([w, x, y, z], axis=-1)


def to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to 3x3 rotation matrix."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    matrix = np.zeros(q.shape[:-1] + (3, 3), dtype=q.dtype)

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


def from_rotation_matrix(matrix: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (Shepperd method)."""
    trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]

    q = np.zeros(matrix.shape[:-2] + (4,), dtype=matrix.dtype)
    mask1 = trace > 0
    mask2 = (~mask1) & (matrix[..., 0, 0] > matrix[..., 1, 1]) & (matrix[..., 0, 0] > matrix[..., 2, 2])
    mask3 = (~mask1) & (~mask2) & (matrix[..., 1, 1] > matrix[..., 2, 2])
    mask4 = (~mask1) & (~mask2) & (~mask3)

    trace_safe = np.where(mask1, trace, 0.0)
    s1 = 2.0 * np.sqrt(np.maximum(trace_safe + 1.0, 1e-8))
    s1 = np.where(mask1, s1, 1.0)
    q[..., 0] = np.where(mask1, 0.25 * s1, q[..., 0])
    q[..., 1] = np.where(mask1, (matrix[..., 2, 1] - matrix[..., 1, 2]) / s1, q[..., 1])
    q[..., 2] = np.where(mask1, (matrix[..., 0, 2] - matrix[..., 2, 0]) / s1, q[..., 2])
    q[..., 3] = np.where(mask1, (matrix[..., 1, 0] - matrix[..., 0, 1]) / s1, q[..., 3])

    diag = np.where(mask2, 1.0 + matrix[..., 0, 0] - matrix[..., 1, 1] - matrix[..., 2, 2], 0.0)
    s2 = 2.0 * np.sqrt(np.maximum(diag, 1e-8))
    s2 = np.where(mask2, s2, 1.0)
    q[..., 0] = np.where(mask2, (matrix[..., 2, 1] - matrix[..., 1, 2]) / s2, q[..., 0])
    q[..., 1] = np.where(mask2, 0.25 * s2, q[..., 1])
    q[..., 2] = np.where(mask2, (matrix[..., 0, 1] + matrix[..., 1, 0]) / s2, q[..., 2])
    q[..., 3] = np.where(mask2, (matrix[..., 0, 2] + matrix[..., 2, 0]) / s2, q[..., 3])

    diag = np.where(mask3, 1.0 + matrix[..., 1, 1] - matrix[..., 0, 0] - matrix[..., 2, 2], 0.0)
    s3 = 2.0 * np.sqrt(np.maximum(diag, 1e-8))
    s3 = np.where(mask3, s3, 1.0)
    q[..., 0] = np.where(mask3, (matrix[..., 0, 2] - matrix[..., 2, 0]) / s3, q[..., 0])
    q[..., 1] = np.where(mask3, (matrix[..., 0, 1] + matrix[..., 1, 0]) / s3, q[..., 1])
    q[..., 2] = np.where(mask3, 0.25 * s3, q[..., 2])
    q[..., 3] = np.where(mask3, (matrix[..., 1, 2] + matrix[..., 2, 1]) / s3, q[..., 3])

    diag = np.where(mask4, 1.0 + matrix[..., 2, 2] - matrix[..., 0, 0] - matrix[..., 1, 1], 0.0)
    s4 = 2.0 * np.sqrt(np.maximum(diag, 1e-8))
    s4 = np.where(mask4, s4, 1.0)
    q[..., 0] = np.where(mask4, (matrix[..., 1, 0] - matrix[..., 0, 1]) / s4, q[..., 0])
    q[..., 1] = np.where(mask4, (matrix[..., 0, 2] + matrix[..., 2, 0]) / s4, q[..., 1])
    q[..., 2] = np.where(mask4, (matrix[..., 1, 2] + matrix[..., 2, 1]) / s4, q[..., 2])
    q[..., 3] = np.where(mask4, 0.25 * s4, q[..., 3])

    return normalize(q)


# =============================================================================
# Interpolation
# =============================================================================

def slerp(q1: np.ndarray, q2: np.ndarray, t) -> np.ndarray:
    """Spherical linear interpolation between two quaternions."""
    q1, q2 = normalize(q1), normalize(q2)

    dot = np.sum(q1 * q2, axis=-1, keepdims=True)
    q2 = np.where(dot < 0, -q2, q2)
    dot = np.abs(dot)
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    small = sin_theta < 1e-6
    t1 = np.where(small, 1.0 - t, np.sin((1.0 - t) * theta) / sin_theta)
    t2 = np.where(small, t, np.sin(t * theta) / sin_theta)

    return normalize(t1 * q1 + t2 * q2)


# =============================================================================
# Utilities
# =============================================================================

def quat_invert(q: np.ndarray) -> np.ndarray:
    """Compute quaternion inverse (conjugate for unit quaternions)."""
    return conjugate(q)


def quat_abs(q: np.ndarray) -> np.ndarray:
    """Make quaternion scalar part positive (canonical form)."""
    mask = q[..., :1] < 0.0
    return np.where(mask, -q, q)


# =============================================================================
# Exponential/Log Maps
# =============================================================================

def quat_exp(v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Exponential map from R^3 to quaternion."""
    half_angle = np.linalg.norm(v, axis=-1, keepdims=True)
    c = np.cos(half_angle)
    s = np.sin(half_angle) / np.maximum(half_angle, eps)
    return np.concatenate([c, s * v], axis=-1)


def quat_log(q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Logarithm map from unit quaternion to R^3."""
    w, v = q[..., :1], q[..., 1:]
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    w_clamped = np.clip(w, -1.0 + eps, 1.0 - eps)
    angle = np.arccos(w_clamped)
    return (v / np.maximum(v_norm, eps)) * angle


# =============================================================================
# Angular Velocity
# =============================================================================

def quat_differentiate_angular_velocity(
    next_q: np.ndarray, curr_q: np.ndarray, dt: float = 1.0, eps: float = 1e-8
) -> np.ndarray:
    """Compute angular velocity between two quaternions."""
    # Normalize
    next_q = next_q / np.maximum(np.linalg.norm(next_q, axis=-1, keepdims=True), eps)
    curr_q = curr_q / np.maximum(np.linalg.norm(curr_q, axis=-1, keepdims=True), eps)

    # Relative rotation: delta_q = next * curr^{-1}
    w1, v1 = next_q[..., 0], next_q[..., 1:]
    w2, v2 = curr_q[..., 0], -curr_q[..., 1:]

    dot_v = np.sum(v1 * v2, axis=-1)
    cross_v = np.cross(v1, v2, axis=-1)
    delta_w = w1 * w2 - dot_v
    delta_v = np.expand_dims(w1, -1) * v2 + np.expand_dims(w2, -1) * v1 + cross_v
    delta_q = np.concatenate([np.expand_dims(delta_w, -1), delta_v], axis=-1)
    delta_q = delta_q / np.maximum(np.linalg.norm(delta_q, axis=-1, keepdims=True), eps)
    sign = np.where(delta_q[..., :1] < 0, -np.ones_like(delta_q[..., :1]), np.ones_like(delta_q[..., :1]))
    delta_q = delta_q * sign
    w, v = delta_q[..., :1], delta_q[..., 1:]
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    half_angle = np.arctan2(v_norm, w)
    scale = np.where(v_norm > eps, half_angle / v_norm, np.ones_like(v_norm))

    return (2.0 / dt) * v * scale


def quat_integrate_angular_velocity(
    vel: np.ndarray, curr_q: np.ndarray, dt: float = 1.0, eps: float = 1e-8
) -> np.ndarray:
    """Integrate angular velocity to get new quaternion."""
    v = 0.5 * vel * dt

    theta = np.linalg.norm(v, axis=-1, keepdims=True)
    s_over_t = np.sin(theta) / np.maximum(theta, eps)
    s_over_t = np.where(theta > eps, s_over_t, 1.0 - (theta * theta) / 6.0)
    c = np.cos(theta)
    delta_w, delta_v = c.squeeze(-1), s_over_t * v
    w1, v1 = delta_w, delta_v
    w2, v2 = curr_q[..., 0], curr_q[..., 1:]
    dot_v = np.sum(v1 * v2, axis=-1)
    cross_v = np.cross(v1, v2, axis=-1)
    new_w = w1 * w2 - dot_v
    new_v = np.expand_dims(w1, -1) * v2 + np.expand_dims(w2, -1) * v1 + cross_v
    new_q = np.concatenate([np.expand_dims(new_w, -1), new_v], axis=-1)
    return new_q / np.maximum(np.linalg.norm(new_q, axis=-1, keepdims=True), eps)


# =============================================================================
# Advanced Operations
# =============================================================================

def quat_log_residual(q_t: np.ndarray, q_s: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compute log residual between two quaternions."""
    q_t = q_t / np.maximum(np.linalg.norm(q_t, axis=-1, keepdims=True), eps)
    q_s = q_s / np.maximum(np.linalg.norm(q_s, axis=-1, keepdims=True), eps)
    w1, v1 = q_t[..., :1], q_t[..., 1:]
    w2, v2 = q_s[..., :1], -q_s[..., 1:]
    delta_w = w1 * w2 - np.sum(v1 * v2, axis=-1, keepdims=True)
    delta_v = w1 * v2 + w2 * v1 + np.cross(v1, v2, axis=-1)
    delta = np.concatenate([delta_w, delta_v], axis=-1)
    delta = delta / np.maximum(np.linalg.norm(delta, axis=-1, keepdims=True), eps)
    delta = np.where(delta[..., :1] < 0, -delta, delta)
    w, v = delta[..., :1], delta[..., 1:]
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    half_angle = np.arctan2(v_norm, w)
    scale = np.where(v_norm > eps, half_angle / v_norm, np.ones_like(v_norm))

    return 2.0 * v * scale


def quat_geodesic_angle(q_t: np.ndarray, q_s: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compute geodesic angle between two quaternions."""
    q_t = q_t / np.maximum(np.linalg.norm(q_t, axis=-1, keepdims=True), eps)
    q_s = q_s / np.maximum(np.linalg.norm(q_s, axis=-1, keepdims=True), eps)
    d = np.abs(np.sum(q_t * q_s, axis=-1))
    d = np.minimum(d, 1.0 - eps)
    return 2.0 * np.arctan2(np.sqrt(np.maximum(1.0 - d * d, 0.0)), d)
