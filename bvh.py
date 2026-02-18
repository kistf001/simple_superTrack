"""
BVH Motion Loader for MuJoCo Humanoid.

Provides BVH file parsing, forward kinematics, and motion loading utilities.
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

import numpy as np
from scipy.interpolate import interp1d

from utils.quaternion import numpy as quaternion_numpy


# =============================================================================
# 쿼터니언 유틸리티
# =============================================================================


def euler_to_rotation_matrix(euler_xyz_deg):
    """Euler angles (ZYX order) → rotation matrix"""
    euler_rad = np.deg2rad(euler_xyz_deg)
    x_rot, y_rot, z_rot = euler_rad

    cos_x, sin_x = np.cos(x_rot), np.sin(x_rot)
    cos_y, sin_y = np.cos(y_rot), np.sin(y_rot)
    cos_z, sin_z = np.cos(z_rot), np.sin(z_rot)

    Rx = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
    Ry = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
    Rz = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


def quat_from_axis_angle(axis, angle_deg):
    """축-각도 → quaternion (w, x, y, z)"""
    axis = np.array(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    angle_rad = np.deg2rad(angle_deg)

    half_angle = angle_rad / 2.0
    w = np.cos(half_angle)
    xyz = axis * np.sin(half_angle)

    return np.array([w, xyz[0], xyz[1], xyz[2]])


def quat_multiply(q1, q2):
    """두 quaternion 곱셈 (q1 * q2)"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conjugate(q):
    """Quaternion conjugate (역회전)"""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def ensure_quat_continuity(quats):
    """쿼터니언 연속성 보장 (double cover 문제 해결)"""
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]
    return quats


def extract_yaw_quat(quat):
    """쿼터니언에서 Z축 회전(yaw)만 추출하여 수평 쿼터니언 생성

    X, Y축 회전(pitch, roll) → 0 (수평 유지)
    Z축 회전(yaw) → 보존 (진행 방향 유지)
    """
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    half_yaw = yaw / 2.0
    return np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)])


# =============================================================================
# BVH 파싱
# =============================================================================


def parse_bvh(filepath):
    """BVH 파일 파싱

    Returns:
        joints: dict - 관절 계층 구조
        joint_names: list - 관절 이름 순서
        motion_data: np.ndarray - 모션 데이터 (frames, channels)
        frame_time: float - 프레임 간격
        num_frames: int - 프레임 수
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    motion_idx = lines.index("MOTION\n")
    hierarchy_lines = lines[:motion_idx]
    motion_lines = lines[motion_idx + 1:]

    # 계층 구조 파싱
    joints = {}
    joint_names = []
    stack = []
    channel_index = 0

    for line in hierarchy_lines:
        stripped = line.strip()
        if not stripped or stripped == "HIERARCHY" or stripped in ["{", "}"]:
            continue

        indent = len(line) - len(line.lstrip())

        if stripped.startswith("ROOT") or stripped.startswith("JOINT"):
            while stack and stack[-1][1] >= indent:
                stack.pop()

            joint_name = stripped.split()[1]
            parent = stack[-1][0] if stack else None

            joints[joint_name] = {
                "parent": parent,
                "offset": None,
                "channels": [],
                "channel_indices": [],
                "children": [],
            }
            joint_names.append(joint_name)

            if parent:
                joints[parent]["children"].append(joint_name)

            stack.append((joint_name, indent))

        elif stripped.startswith("OFFSET"):
            if stack:
                joint_name = stack[-1][0]
                offset = [float(x) for x in stripped.split()[1:4]]
                joints[joint_name]["offset"] = np.array(offset)

        elif stripped.startswith("CHANNELS"):
            if stack:
                joint_name = stack[-1][0]
                parts = stripped.split()
                num_channels = int(parts[1])
                channels = parts[2:2 + num_channels]

                joints[joint_name]["channels"] = channels
                joints[joint_name]["channel_indices"] = list(
                    range(channel_index, channel_index + num_channels)
                )
                channel_index += num_channels

    # 모션 데이터 파싱
    num_frames = 0
    frame_time = 0.0
    motion_start = 0

    for i, line in enumerate(motion_lines):
        if line.startswith("Frames:"):
            num_frames = int(line.split()[1])
        elif line.startswith("Frame Time:"):
            frame_time = float(line.split()[2])
        elif not line.strip().startswith("Frames") and not line.strip().startswith("Frame"):
            motion_start = i
            break

    motion_data_raw = []
    for line in motion_lines[motion_start:]:
        if line.strip():
            values = [float(x) for x in line.split()]
            motion_data_raw.append(values)

    motion_data = np.array(motion_data_raw)

    return joints, joint_names, motion_data, frame_time, num_frames


# =============================================================================
# Forward Kinematics
# =============================================================================


def compute_forward_kinematics(
    joints,
    joint_names,
    motion_data,
    frame_idx,
    root_rotation_offset=None,
):
    """단일 프레임의 FK 계산

    Args:
        joints: 관절 계층 구조
        joint_names: 관절 이름 리스트
        motion_data: 모션 데이터 배열
        frame_idx: 프레임 인덱스
        root_rotation_offset: 루트 회전 오프셋 (euler xyz deg)

    Returns:
        world_positions: dict - 월드 좌표 위치
        world_rotations: dict - 월드 좌표 회전 행렬
    """
    if root_rotation_offset is not None:
        offset_rotation_matrix = euler_to_rotation_matrix(root_rotation_offset)
    else:
        offset_rotation_matrix = np.eye(3)

    root_name = joint_names[0]

    # 프레임 데이터 추출
    frame_data = {}
    for jn in joint_names:
        joint_info = joints[jn]
        channels = joint_info["channels"]
        indices = joint_info["channel_indices"]

        data = {"position": np.zeros(3), "rotation": np.zeros(3)}

        for channel, idx in zip(channels, indices):
            value = motion_data[frame_idx, idx]

            if "position" in channel.lower():
                axis_map = {"Xposition": 0, "Yposition": 1, "Zposition": 2}
                data["position"][axis_map[channel]] = value
            elif "rotation" in channel.lower():
                axis_map = {"Xrotation": 0, "Yrotation": 1, "Zrotation": 2}
                data["rotation"][axis_map[channel]] = value

        frame_data[jn] = data

    # FK 계산
    world_positions = {}
    world_rotations = {}

    def compute_transform(joint_name, parent_pos, parent_rot_mat):
        joint_info = joints[joint_name]

        local_pos = joint_info["offset"]
        if local_pos is None:
            local_pos = np.zeros(3)

        local_euler = frame_data[joint_name]["rotation"]
        local_rot_mat = euler_to_rotation_matrix(local_euler)

        if joint_name == root_name:
            local_rot_mat = offset_rotation_matrix @ local_rot_mat

        if "Xposition" in joint_info["channels"]:
            root_offset = frame_data[joint_name]["position"]
            root_offset = offset_rotation_matrix @ root_offset
            local_pos = local_pos + root_offset

        world_pos = parent_pos + parent_rot_mat @ local_pos
        world_rot_mat = parent_rot_mat @ local_rot_mat

        world_positions[joint_name] = world_pos
        world_rotations[joint_name] = world_rot_mat

        for child_name in joint_info["children"]:
            compute_transform(child_name, world_pos, world_rot_mat)

    compute_transform(root_name, np.zeros(3), np.eye(3))
    return world_positions, world_rotations


# =============================================================================
# 모션 로딩
# =============================================================================

# 기본 관절 오프셋 (BVH → MuJoCo 좌표계 변환)
DEFAULT_JOINT_OFFSETS = {
    "Hips": [0, 90, 90],
    "Spine": [0, 90, 90],
    "Spine1": [0, 90, 90],
    "Spine2": [0, 90, 90],
    "Neck": [0, 90, 90],
    "Head": [0, 90, 90],
    "LeftShoulder": [0, 90, 90],
    "LeftArm": [0, 90, 90],
    "LeftForeArm": [0, 90, 90],
    "LeftHand": [0, 90, 90],
    "RightShoulder": [0, 90, 90],
    "RightArm": [0, 90, 90],
    "RightForeArm": [0, 90, 90],
    "RightHand": [0, 90, 90],
    "LeftUpLeg": [0, 90, 90],
    "LeftLeg": [0, 90, 90],
    "RightUpLeg": [0, 90, 90],
    "RightLeg": [0, 90, 90],
    "LeftFoot": [0, 90, 90],
    "RightFoot": [0, 90, 90],
    "LeftToe": [0, 90, 90],
    "RightToe": [0, 90, 90],
}

DEFAULT_QUAT_OFFSET = [0, 90, 90]


def _compute_quat_offset(offset_xyz):
    """[x, y, z] 도 단위 오프셋 → quaternion"""
    q_x = quat_from_axis_angle([1, 0, 0], offset_xyz[0])
    q_y = quat_from_axis_angle([0, 1, 0], offset_xyz[1])
    q_z = quat_from_axis_angle([0, 0, 1], offset_xyz[2])
    return quat_multiply(q_x, quat_multiply(q_y, q_z))


def load_bvh_motion(
    filepath,
    max_frames=2048,
    joint_offsets=None,
    root_rotation_offset=None,
):
    """BVH 파일 로드 → joint_data dict

    Args:
        filepath: BVH 파일 경로
        max_frames: 최대 프레임 수
        joint_offsets: 관절별 쿼터니언 오프셋 (None이면 기본값 사용)
        root_rotation_offset: 루트 회전 오프셋 (기본: [90, 0, 90])

    Returns:
        joint_data: dict {joint_name: {"pos": np.ndarray, "quat": np.ndarray}}
        joint_names: list
        frame_time: float
        num_frames: int
    """
    if joint_offsets is None:
        joint_offsets = DEFAULT_JOINT_OFFSETS
    if root_rotation_offset is None:
        root_rotation_offset = np.array([90.0, 0.0, 90.0])

    # 오프셋 캐시 생성
    quat_offset_cache = {}
    for joint_name, offset in joint_offsets.items():
        if offset is not None:
            quat_offset_cache[joint_name] = _compute_quat_offset(offset)
    default_quat_offset = _compute_quat_offset(DEFAULT_QUAT_OFFSET)

    # BVH 파싱
    joints, joint_names, motion_data, frame_time, num_frames = parse_bvh(filepath)
    num_frames = min(max_frames, num_frames)

    # Ground offset 계산 (첫 프레임 기준)
    first_pos, _ = compute_forward_kinematics(
        joints, joint_names, motion_data, 0, root_rotation_offset
    )
    left_foot = first_pos.get("LeftFoot", np.zeros(3))
    right_foot = first_pos.get("RightFoot", np.zeros(3))
    ground_offset = np.array([
        (left_foot[0] + right_foot[0]) / 2.0,
        (left_foot[1] + right_foot[1]) / 2.0,
        min(left_foot[2], right_foot[2]),
    ])

    # 프레임 처리 함수
    def process_frame(frame_idx):
        positions, rotations = compute_forward_kinematics(
            joints, joint_names, motion_data, frame_idx, root_rotation_offset
        )

        frame_result = {}
        for joint_name in joint_names:
            pos = (positions[joint_name] - ground_offset) / 100.0
            quat = quaternion_numpy.from_rotation_matrix(rotations[joint_name])

            # 관절별 오프셋 적용
            q_offset = quat_offset_cache.get(joint_name, default_quat_offset)
            if q_offset is not None:
                quat = quat_multiply(quat, q_offset)

            frame_result[joint_name] = {"pos": pos, "quat": quat}

        return frame_result

    # 병렬 처리
    num_workers = mp.cpu_count()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_frame, range(num_frames)))

    # 결과 수집
    joint_data = {name: {"pos": [], "quat": []} for name in joint_names}
    for frame_result in results:
        for joint_name in joint_names:
            joint_data[joint_name]["pos"].append(frame_result[joint_name]["pos"])
            joint_data[joint_name]["quat"].append(frame_result[joint_name]["quat"])

    # numpy 배열로 변환 + 쿼터니언 연속성 보장
    for joint_name in joint_names:
        joint_data[joint_name]["pos"] = np.array(joint_data[joint_name]["pos"])
        joint_data[joint_name]["quat"] = ensure_quat_continuity(
            np.array(joint_data[joint_name]["quat"])
        )

    return joint_data, joint_names, frame_time, num_frames


# =============================================================================
# 리샘플링
# =============================================================================


def resample_motion(joint_data, joint_names, original_fps, target_fps, warmup_frames=40):
    """모션 리샘플링

    Args:
        joint_data: 관절 데이터 dict
        joint_names: 관절 이름 리스트
        original_fps: 원본 FPS
        target_fps: 목표 FPS
        warmup_frames: 스킵할 웜업 프레임 수

    Returns:
        resampled_data: 리샘플링된 관절 데이터
        num_frames: 리샘플링된 프레임 수
    """
    num_frames = len(joint_data[joint_names[0]]["pos"])
    start_frame = warmup_frames
    end_frame = num_frames

    original_times = np.arange(end_frame - start_frame) / original_fps
    target_duration = (end_frame - start_frame - 1) / original_fps
    target_times = np.arange(0, target_duration, 1.0 / target_fps)

    resampled_data = {}
    for jn in joint_names:
        pos_data = joint_data[jn]["pos"][start_frame:end_frame]
        quat_data = joint_data[jn]["quat"][start_frame:end_frame]

        pos_interp = interp1d(original_times, pos_data, axis=0, kind='linear')
        quat_interp = interp1d(original_times, quat_data, axis=0, kind='linear')

        resampled_pos = pos_interp(target_times)
        resampled_quat = quat_interp(target_times)
        resampled_quat = resampled_quat / np.linalg.norm(resampled_quat, axis=1, keepdims=True)

        resampled_data[jn] = {"pos": resampled_pos, "quat": resampled_quat}

    return resampled_data, len(target_times)
