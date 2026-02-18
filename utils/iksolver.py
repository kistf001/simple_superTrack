#!/usr/bin/env python3
"""
MuJoCo Inverse Kinematics Solver
Multi-body IK with position and rotation constraints
"""

import numpy as np
import mujoco
from utils.quaternion import numpy as quaternion_numpy


def _compute_position_error(current_pos, target_pos):
    """위치 오차 계산"""
    return target_pos - current_pos


def _compute_rotation_error(current_quat, target_quat):
    """회전 오차를 axis-angle로 계산"""
    quat_error = quaternion_numpy.multiply(
        target_quat.reshape(1, 4),
        quaternion_numpy.conjugate(current_quat.reshape(1, 4)),
    )[0]
    return 2.0 * quat_error[1:4]


def _compute_floor_constraints(model, data, floor_bodies, floor_z):
    """바닥 관통 방지 제약 계산 (부등식 제약)

    Z < floor_z 인 body에 대해서만 제약 활성화
    """
    constraints = {}
    for body_name in floor_bodies:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        current_z = data.xpos[bid, 2]
        if current_z < floor_z:
            constraints[body_name] = {
                'bid': bid,
                'error': floor_z - current_z  # 위로 밀어야 할 양
            }
    return constraints


def _compute_horizontal_constraints(model, data, horizontal_bodies):
    """발 수평 제약: pitch(앞뒤 꺾임)와 roll(좌우 기울임) 모두 0으로

    - Pitch: local X축(발끝 방향)의 Z성분 = 0 → 발끝이 위/아래로 안 꺾임
    - Roll: local Y축(발 측면 방향)의 Z성분 = 0 → 발이 좌/우로 안 기울어짐
    """
    constraints = {}
    for body_name in horizontal_bodies:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        xmat = data.xmat[bid].reshape(3, 3)

        local_x_world = xmat[:, 0]  # 발끝 방향
        local_y_world = xmat[:, 1]  # 발 측면 방향

        constraints[body_name] = {
            'bid': bid,
            'pitch_error': -local_x_world[2],  # pitch: local X의 Z = 0
            'roll_error': -local_y_world[2],   # roll: local Y의 Z = 0
        }
    return constraints


def _compute_errors(
    data, body_ids, targets_pos, targets_quat, weights_pos, weights_rot
):
    """모든 body의 위치/회전 오차 계산"""
    errors_pos = {}
    errors_rot = {}
    total_error_norm = 0.0

    for name, bid in body_ids.items():
        # 위치 오차
        if targets_pos and name in targets_pos:
            current_pos = data.xpos[bid]
            error_pos = _compute_position_error(current_pos, targets_pos[name])
            errors_pos[name] = error_pos
            total_error_norm += weights_pos.get(name, 0.0) * np.linalg.norm(error_pos)

        # 회전 오차
        if targets_quat and name in targets_quat:
            current_quat = data.xquat[bid]
            error_rot = _compute_rotation_error(current_quat, targets_quat[name])
            errors_rot[name] = error_rot
            total_error_norm += weights_rot.get(name, 0.0) * np.linalg.norm(error_rot)

    return errors_pos, errors_rot, total_error_norm


def _build_jacobian(
    model,
    data,
    body_ids,
    targets_pos,
    targets_quat,
    errors_pos,
    errors_rot,
    weights_pos,
    weights_rot,
    floor_constraints=None,
    floor_weight=10.0,
    horizontal_constraints=None,
    horizontal_weight=10.0,
    reg_weight=0.0,
):
    """제약조건에 대한 stacked Jacobian 구성 (regularization 포함)"""
    jacobians = []
    error_vector = []

    for name, bid in body_ids.items():
        # Jacobian 계산
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, bid)

        # 위치 제약
        if name in errors_pos:
            w_p = weights_pos.get(name, 1.0)
            jacobians.append(np.sqrt(w_p) * jacp)
            error_vector.append(np.sqrt(w_p) * errors_pos[name])

        # 회전 제약
        if name in errors_rot:
            w_r = weights_rot.get(name, 1.0)
            jacobians.append(np.sqrt(w_r) * jacr)
            error_vector.append(np.sqrt(w_r) * errors_rot[name])

    # 바닥 제약 (Z 높이 부등식 제약)
    if floor_constraints:
        for name, constraint in floor_constraints.items():
            bid = constraint['bid']
            error_z = constraint['error']

            # Z축 Jacobian만 사용
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, jacr, bid)

            # Z 성분만 추출 (jacp[2, :])
            jacobians.append(np.sqrt(floor_weight) * jacp[2:3, :])
            error_vector.append(np.sqrt(floor_weight) * np.array([error_z]))

    # 수평 제약 (pitch + roll)
    if horizontal_constraints:
        for name, constraint in horizontal_constraints.items():
            bid = constraint['bid']
            pitch_error = constraint['pitch_error']
            roll_error = constraint['roll_error']

            # body의 rotation matrix
            xmat = data.xmat[bid].reshape(3, 3)
            local_x_world = xmat[:, 0]  # 발끝 방향
            local_y_world = xmat[:, 1]  # 발 측면 방향

            # 회전 Jacobian
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, jacr, bid)

            # Pitch 제약: d(local_x[2])/dt = omega_x * local_x[1] - omega_y * local_x[0]
            jac_pitch = local_x_world[1] * jacr[0, :] - local_x_world[0] * jacr[1, :]
            jacobians.append(np.sqrt(horizontal_weight) * jac_pitch.reshape(1, -1))
            error_vector.append(np.sqrt(horizontal_weight) * np.array([pitch_error]))

            # Roll 제약: d(local_y[2])/dt = omega_x * local_y[1] - omega_y * local_y[0]
            jac_roll = local_y_world[1] * jacr[0, :] - local_y_world[0] * jacr[1, :]
            jacobians.append(np.sqrt(horizontal_weight) * jac_roll.reshape(1, -1))
            error_vector.append(np.sqrt(horizontal_weight) * np.array([roll_error]))

    # Regularization (변화량 최소화)
    if reg_weight > 0:
        # J_reg = sqrt(reg_weight) * I, e_reg = 0
        # → min ||dq||^2 (변화량이 작은 것을 선호)
        jacobians.append(np.sqrt(reg_weight) * np.eye(model.nv))
        error_vector.append(np.zeros(model.nv))

    J_stacked = np.vstack(jacobians)
    e_stacked = np.concatenate(error_vector)

    return J_stacked, e_stacked


def _solve_damped_least_squares(J_stacked, e_stacked, damping, learning_rate):
    """Damped Least Squares로 qvel 업데이트 계산"""
    JJT = J_stacked @ J_stacked.T
    n_constraints = len(e_stacked)
    damped_inv = np.linalg.inv(JJT + damping * np.eye(n_constraints))
    delta_qvel = learning_rate * J_stacked.T @ damped_inv @ e_stacked
    return delta_qvel


def _clamp_joint_limits(model, data):
    """조인트 한계 강제 (클램핑)"""
    for i in range(model.njnt):
        if model.jnt_limited[i]:
            qpos_idx = model.jnt_qposadr[i]
            low, high = model.jnt_range[i]
            data.qpos[qpos_idx] = np.clip(data.qpos[qpos_idx], low, high)


def _update_qpos(data, delta_qvel, model, fix_base=True, clamp_limits=True):
    """qpos 업데이트

    Args:
        fix_base: True면 free joint 고정, False면 whole-body IK
        clamp_limits: True면 조인트 한계 강제
    """
    if fix_base:
        if model.nq > 7:
            data.qpos[7:] += delta_qvel[6:]
    else:
        mujoco.mj_integratePos(model, data.qpos, delta_qvel, 1.0)

    if clamp_limits:
        _clamp_joint_limits(model, data)


def _compute_final_errors(data, body_ids, targets_pos, targets_quat):
    """최종 오차 계산"""
    final_pos_errors = {}
    final_rot_errors = {}

    for name, bid in body_ids.items():
        # 위치 오차
        if targets_pos and name in targets_pos:
            current_pos = data.xpos[bid]
            final_pos_errors[name] = np.linalg.norm(targets_pos[name] - current_pos)

        # 회전 오차
        if targets_quat and name in targets_quat:
            current_quat = data.xquat[bid]
            final_rot_errors[name] = 1.0 - abs(
                np.dot(current_quat, targets_quat[name])
            )

    return final_pos_errors, final_rot_errors


def solve_ik_multi_body(
    model,
    data,
    targets_pos=None,
    targets_quat=None,
    weights_pos=None,
    weights_rot=None,
    damping=0.01,
    learning_rate=0.3,
    max_iterations=30,
    tolerance=0.01,
    verbose=False,
    fix_base=True,
    clamp_limits=True,
    floor_bodies=None,
    floor_z=0.027,
    floor_weight=10.0,
    horizontal_bodies=None,
    horizontal_weight=10.0,
    reg_weight=0.0,
):
    """
    Multi-body IK solver with position and rotation constraints

    Args:
        model: MuJoCo model
        data: MuJoCo data (will be modified in-place)
        targets_pos: dict {"body_name": np.array([x, y, z]), ...}
        targets_quat: dict {"body_name": np.array([w, x, y, z]), ...} (optional)
        weights_pos: dict {"body_name": weight, ...} (default: 1.0 for all)
        weights_rot: dict {"body_name": weight, ...} (default: 0.0 for all)
        damping: Damping factor for Damped Least Squares
        learning_rate: Step size multiplier
        max_iterations: Maximum number of iterations
        tolerance: Convergence threshold (average error)
        verbose: Print iteration progress
        fix_base: True면 free joint(qpos[0:7]) 고정, False면 whole-body IK
        clamp_limits: True면 조인트 한계 강제 (기본값: True)
        floor_bodies: 바닥 제약 적용할 body 목록 (예: ["foot_left", "foot_right"])
        floor_z: 최소 Z 높이 (기본값: 0.027, 발 geom 반지름)
        floor_weight: 바닥 제약 가중치 (기본값: 10.0)
        horizontal_bodies: 수평 제약(pitch+roll) 적용할 body 목록 (예: ["foot_left", "foot_right"])
        horizontal_weight: 수평 제약 가중치 (기본값: 10.0)
        reg_weight: Regularization 가중치 (변화량 최소화, 기본값: 0.0)

    Returns:
        dict {
            "success": bool,
            "iterations": int,
            "final_error": float,
            "pos_errors": dict {"body_name": error, ...},
            "rot_errors": dict {"body_name": error, ...} (if targets_quat provided)
        }
    """
    # Body ID lookup (합집합)
    all_names = set()
    if targets_pos:
        all_names.update(targets_pos.keys())
    if targets_quat:
        all_names.update(targets_quat.keys())

    body_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in all_names
    }

    # Default weights
    if weights_pos is None:
        weights_pos = {name: 1.0 for name in (targets_pos or {})}
    if weights_rot is None:
        weights_rot = {name: 1.0 for name in (targets_quat or {})}

    # IK loop
    converged = False
    final_avg_error = float("inf")

    for iteration in range(max_iterations):
        mujoco.mj_forward(model, data)

        # 오차 계산
        errors_pos, errors_rot, total_error_norm = _compute_errors(
            data, body_ids, targets_pos, targets_quat, weights_pos, weights_rot
        )

        # 수렴 확인
        total_weight = sum(weights_pos.values()) + sum(weights_rot.values())
        avg_error = total_error_norm / total_weight if total_weight > 0 else 0.0
        final_avg_error = avg_error

        if verbose and iteration % 50 == 0:
            print(f"  Iter {iteration:3d}: avg error {avg_error:.4f}")

        if avg_error < tolerance:
            converged = True
            if verbose:
                print(f"  ✅ Converged at iteration {iteration}")
            break

        # 바닥 제약 계산 (부등식 제약)
        floor_constraints = None
        if floor_bodies:
            floor_constraints = _compute_floor_constraints(
                model, data, floor_bodies, floor_z
            )

        # 수평 제약 계산 (pitch + roll)
        horizontal_constraints = None
        if horizontal_bodies:
            horizontal_constraints = _compute_horizontal_constraints(
                model, data, horizontal_bodies
            )

        # Jacobian 구성
        J_stacked, e_stacked = _build_jacobian(
            model,
            data,
            body_ids,
            targets_pos,
            targets_quat,
            errors_pos,
            errors_rot,
            weights_pos,
            weights_rot,
            floor_constraints,
            floor_weight,
            horizontal_constraints,
            horizontal_weight,
            reg_weight,
        )

        # Damped Least Squares 풀이
        delta_qvel = _solve_damped_least_squares(
            J_stacked, e_stacked, damping, learning_rate
        )

        # qpos 업데이트
        _update_qpos(data, delta_qvel, model, fix_base, clamp_limits)

    # 최종 forward kinematics
    mujoco.mj_forward(model, data)

    # 최종 오차 계산
    final_pos_errors, final_rot_errors = _compute_final_errors(
        data, body_ids, targets_pos, targets_quat
    )

    result = {
        "success": converged,
        "iterations": iteration + 1,
        "final_error": final_avg_error,
        "pos_errors": final_pos_errors,
    }

    if targets_quat is not None:
        result["rot_errors"] = final_rot_errors

    return result
