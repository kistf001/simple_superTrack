import torch
import numpy as np


@torch.jit.script
def quat_norm(x):
    """
    quaternion norm (unit quaternion represents a 3D rotation, which has norm of 1)
    """
    x = x.norm(p=2, dim=-1)
    return x


@torch.jit.script
def quat_unit(x):
    """
    normalized quaternion with norm of 1
    """
    norm = quat_norm(x).unsqueeze(-1)
    return x / (norm.clamp(min=1e-9))


@torch.jit.script
def quat_mul(Q0, Q1):
    # Extract the values from Q0
    w0, x0, y0, z0 = Q0[..., 0], Q0[..., 1], Q0[..., 2], Q0[..., 3]

    # Extract the values from Q1
    w1, x1, y1, z1 = Q1[..., 0], Q1[..., 1], Q1[..., 2], Q1[..., 3]

    # Computer the product of the two quaternions, term by term
    Q0Q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    Q0Q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    Q0Q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    Q0Q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

    return quat_unit(torch.stack([Q0Q1_w, Q0Q1_x, Q0Q1_y, Q0Q1_z], dim=-1))


@torch.jit.script
def quat_abs(x):
    return torch.where(x[..., 0:1] < 0.0, -x, x)


@torch.jit.script
def quat_exp(v):
    halfangle = v.norm(p=2, dim=-1).unsqueeze(-1)
    c = torch.cos(halfangle)
    s = torch.sin(halfangle) / halfangle
    ones = halfangle.clone() * 0 + 1
    return torch.where(
        halfangle < 0.0000001,
        # torch.concat((torch.ones(halfangle.shape), v), -1),
        torch.concat((ones, v), -1),
        torch.concat((c, s * v), -1),  # if small than eps
    )


@torch.jit.script
def quat_log(q):
    eps = 0.0000001
    length = q[..., 1:].norm(p=2, dim=-1).unsqueeze(-1)
    halfangle = torch.acos(torch.clamp(q[..., 0:1], min=-0.99999, max=0.99999))
    return torch.where(
        length < eps,
        q[..., 1:4],
        halfangle * (q[..., 1:4] / length),
    )


@torch.jit.script
def quat_differentiate_angular_velocity(_next, curr, dt):
    curr[..., 1:] = -curr[..., 1:]  # quat_inv
    q_mul = quat_mul(_next, curr)
    q_abs = quat_abs(q_mul)
    q_log = 2.0 * quat_log(q_abs) / dt  # quat_to_scaled_angle_axis_approx
    return q_log


@torch.jit.script
def quat_integrate_angular_velocity(vel, curr, dt):
    q_exp = quat_exp((vel * dt) / 2.0)
    q_mul = quat_mul(q_exp, curr)
    return q_mul


@torch.jit.script
def q_2_m(Q):
    q0, q1, q2, q3 = Q[..., 0:1], Q[..., 1:2], Q[..., 2:3], Q[..., 3:4]
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
    # Matrix complete
    matrix = torch.concat(
        (
            torch.stack(
                (
                    r00,
                    r01,
                    r02,
                ),
                -1,
            ),
            torch.stack(
                (
                    r10,
                    r11,
                    r12,
                ),
                -1,
            ),
            torch.stack(
                (
                    r20,
                    r21,
                    r22,
                ),
                -1,
            ),
        ),
        -2,
    )
    return matrix


@torch.jit.script
def q_2_m_2axis(Q):
    q0, q1, q2, q3 = Q[..., 0:1], Q[..., 1:2], Q[..., 2:3], Q[..., 3:4]
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    # r02 = 2 * (q1 * q3 + q0 * q2)
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    # r12 = 2 * (q2 * q3 - q0 * q1)
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    # r22 = 2 * (q0 * q0 + q3 * q3) - 1
    # Matrix complete
    matrix = torch.concat(
        (
            torch.stack(
                (
                    r00,
                    r10,
                    r20,
                ),
                -1,
            ),
            torch.stack(
                (
                    r01,
                    r11,
                    r21,
                ),
                -1,
            ),
        ),
        -2,
    )
    return matrix


if __name__ == "__main__":
    print(quat_unit(torch.tensor([1.0, 1.0, 0.0, 0]).reshape(1, 1, 1, -1)))
    print(quat_mul(torch.tensor([1.0, 0.0, 0.0, 0]).reshape(1, 1, 1, -1), torch.tensor([1.0, 0.0, 0.0, 0]).reshape(1, 1, 1, -1)))
    print(quat_abs(torch.tensor([-1.0, 0.0, 0.0, 0]).reshape(1, 1, 1, -1)))
    print(quat_exp(torch.tensor([3.141592, 0.0, 0.0]).reshape(1, 1, 1, 1, 1, -1)))
    print(quat_log(torch.tensor([0.0, 1, 0.0, 0]).reshape(1, 1, 1, -1)))
    print(
        quat_differentiate_angular_velocity(
            torch.tensor([0.0, 1, 0.0, 0]).reshape(1, 1, 1, -1),
            torch.tensor([1.0, 0.0, 0.0, 0]).reshape(1, 1, 1, -1),
            torch.tensor([1]),
        )
    )
    print(
        quat_integrate_angular_velocity(
            torch.tensor([3.141592, 0, 0.0]).reshape(1, 1, 1, -1),
            torch.tensor([1.0, 0.0, 0.0, 0]).reshape(1, 1, 1, -1),
            torch.tensor([1]),
        )
    )
    print(q_2_m(torch.tensor([1.0, 2.0, 3.0, 4]).reshape(1, 1, 1, -1)))
    print("q_2_m_2axis", q_2_m_2axis(torch.tensor([1.0, 2.0, 3.0, 4]).reshape(1, 1, 1, -1)))

    from pyquaternion import Quaternion

    wwe = Quaternion(w=0, x=1, y=0, z=0).rotation_matrix
    print(wwe)
