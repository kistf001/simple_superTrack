"""
BVH Motion to MuJoCo qpos converter.

Converts BVH motion files to MuJoCo joint positions using IK.
Outputs pkl files to mjmotions/ directory.
"""

import glob
import pickle
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from bvh import (
    load_bvh_motion,
    resample_motion,
    extract_yaw_quat,
)
from utils.iksolver import solve_ik_multi_body


# =============================================================================
# 설정
# =============================================================================

MOTIONS_DIR = "data/motions/"
MODEL_PATH = "model/humanoid.xml"
OUTPUT_DIR = Path("mjmotions")

MAX_FRAMES = 2048
TARGET_FPS = 50
WARMUP_FRAMES = 40
EXTRACT_FRAMES = 1024

# BVH 조인트 → MuJoCo 바디 매핑
BVH_TO_MUJOCO = {
    "Hips": {"body": "pelvis", "w_pos": 0.01, "w_rot": 0.01},
    "Head": {"body": "head", "w_pos": 1.0, "w_rot": None},
    "LeftFoot": {"body": "foot_left", "w_pos": 1.0, "w_rot": None, "z_scale": 1.1},
    "RightFoot": {"body": "foot_right", "w_pos": 1.0, "w_rot": None, "z_scale": 1.1},
}


# =============================================================================
# IK 유틸리티
# =============================================================================


def build_ik_targets(joint_data, frame_idx, bvh_to_mujoco):
    """IK 타겟 구성"""
    targets_pos = {}
    targets_quat = {}
    weights_pos = {}
    weights_rot = {}

    for bvh_name, cfg in bvh_to_mujoco.items():
        mj_body = cfg["body"]

        if cfg.get("w_pos") is not None:
            pos = joint_data[bvh_name]["pos"][frame_idx].copy()
            if cfg.get("z_scale") is not None:
                pos[2] *= cfg["z_scale"]
            targets_pos[mj_body] = pos
            weights_pos[mj_body] = cfg["w_pos"]

        if cfg.get("w_rot") is not None:
            bvh_quat = joint_data[bvh_name]["quat"][frame_idx]
            if cfg.get("horizontal"):
                targets_quat[mj_body] = extract_yaw_quat(bvh_quat)
            else:
                targets_quat[mj_body] = bvh_quat
            weights_rot[mj_body] = cfg["w_rot"]

    return targets_pos, targets_quat, weights_pos, weights_rot


def warmup_ik(model, data, joint_data, bvh_to_mujoco, warmup_frames=40):
    """IK 워밍업: T-pose에서 BVH 모션으로 점진적 전환"""
    print(f"워밍업 중... ({warmup_frames} 프레임)")

    for idx in range(warmup_frames):
        targets_pos, targets_quat, weights_pos, weights_rot = build_ik_targets(
            joint_data, idx, bvh_to_mujoco
        )
        solve_ik_multi_body(
            model, data,
            targets_pos=targets_pos,
            targets_quat=targets_quat,
            weights_pos=weights_pos,
            weights_rot=weights_rot,
            damping=0.05,
            learning_rate=0.1,
            max_iterations=100,
            tolerance=0.01,
            fix_base=False,
            floor_bodies=["foot_left", "foot_right"],
            floor_z=0.027,
            horizontal_bodies=["foot_left", "foot_right"],
        )

    print("워밍업 완료")
    return warmup_frames


# =============================================================================
# 메인 함수
# =============================================================================


def visualize_bvh(bvh_path=None, show_viewer=True, export_qpos=True):
    """BVH 모션 시각화 및 qpos 추출

    Args:
        bvh_path: BVH 파일 경로 (None이면 MOTIONS_DIR에서 첫 번째 파일)
        show_viewer: MuJoCo 뷰어 표시 여부
        export_qpos: qpos 추출 및 저장 여부
    """
    # BVH 파일 찾기
    if bvh_path is None:
        bvh_files = sorted(glob.glob(str(Path(MOTIONS_DIR) / "*.bvh")))
        if not bvh_files:
            raise FileNotFoundError(f"{MOTIONS_DIR}에서 BVH 파일을 찾을 수 없습니다!")
        bvh_path = bvh_files[0]

    print(f"BVH 파일 로드: {bvh_path}")

    # 모션 로드
    joint_data, joint_names, frame_time, num_frames = load_bvh_motion(
        bvh_path, max_frames=MAX_FRAMES
    )

    print(f"  조인트: {len(joint_names)}개")
    print(f"  프레임: {num_frames}개")

    # 리샘플링
    original_fps = 1.0 / frame_time
    print(f"\n{TARGET_FPS}Hz 리샘플링 중...")

    resampled_data, resampled_num_frames = resample_motion(
        joint_data, joint_names, original_fps, TARGET_FPS, warmup_frames=WARMUP_FRAMES
    )

    print(f"  원본: {num_frames - WARMUP_FRAMES}프레임 @ {original_fps:.0f}fps")
    print(f"  리샘플링: {resampled_num_frames}프레임 @ {TARGET_FPS}fps")

    target_frame_time = 1.0 / TARGET_FPS

    # MuJoCo 환경 로드
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # 시각화
    if show_viewer:
        print(f"\nMuJoCo 시각화 시작 (IK)")
        print(f"  프레임: {resampled_num_frames}개, frame_time: {target_frame_time:.4f}s")

        with mujoco.viewer.launch_passive(model, data) as viewer:
            frame_idx = warmup_ik(model, data, resampled_data, BVH_TO_MUJOCO, warmup_frames=40)

            while viewer.is_running():
                targets_pos, targets_quat, weights_pos, weights_rot = build_ik_targets(
                    resampled_data, frame_idx, BVH_TO_MUJOCO
                )

                solve_ik_multi_body(
                    model, data,
                    targets_pos=targets_pos,
                    targets_quat=targets_quat,
                    weights_pos=weights_pos,
                    weights_rot=weights_rot,
                    damping=0.05,
                    learning_rate=0.01,
                    max_iterations=100,
                    tolerance=0.01,
                    fix_base=False,
                    floor_bodies=["foot_left", "foot_right"],
                    floor_z=0.027,
                    horizontal_bodies=["foot_left", "foot_right"],
                )

                # BVH 조인트 시각화 (빨간 구)
                i = 0
                for name in joint_names:
                    pos = resampled_data[name]["pos"][frame_idx]
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[i],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.02, 0, 0],
                        pos=pos + np.array([0, 0.5, 0.0]),
                        mat=np.eye(3).flatten(),
                        rgba=[1, 0, 0, 0.5],
                    )
                    i += 1
                viewer.user_scn.ngeom = i

                print(frame_idx)
                viewer.sync()

                frame_idx = (frame_idx + 1) % resampled_num_frames
                time.sleep(target_frame_time)

    # qpos 추출
    if export_qpos:
        OUTPUT_DIR.mkdir(exist_ok=True)

        model_export = mujoco.MjModel.from_xml_path(MODEL_PATH)
        data_export = mujoco.MjData(model_export)

        extract_num_frames = min(EXTRACT_FRAMES, resampled_num_frames - 40)

        print(f"\n{extract_num_frames}프레임 qpos 추출 중...")

        # 웜업
        print("  웜업 중...")
        for idx in range(40):
            targets_pos, targets_quat, weights_pos, weights_rot = build_ik_targets(
                resampled_data, idx, BVH_TO_MUJOCO
            )
            solve_ik_multi_body(
                model_export, data_export,
                targets_pos=targets_pos,
                targets_quat=targets_quat,
                weights_pos=weights_pos,
                weights_rot=weights_rot,
                damping=0.05,
                learning_rate=0.1,
                max_iterations=100,
                tolerance=0.01,
                fix_base=False,
                floor_bodies=["foot_left", "foot_right"],
                floor_z=0.027,
                horizontal_bodies=["foot_left", "foot_right"],
            )

        # 추출
        qpos_list = []
        print(f"  추출 중... (0/{extract_num_frames})")

        for i, frame_idx in enumerate(range(40, 40 + extract_num_frames)):
            targets_pos, targets_quat, weights_pos, weights_rot = build_ik_targets(
                resampled_data, frame_idx, BVH_TO_MUJOCO
            )
            solve_ik_multi_body(
                model_export, data_export,
                targets_pos=targets_pos,
                targets_quat=targets_quat,
                weights_pos=weights_pos,
                weights_rot=weights_rot,
                damping=0.05,
                learning_rate=0.01,
                max_iterations=100,
                tolerance=0.01,
                fix_base=False,
                floor_bodies=["foot_left", "foot_right"],
                floor_z=0.027,
                horizontal_bodies=["foot_left", "foot_right"],
            )
            qpos_list.append(data_export.qpos.copy())

            if (i + 1) % 200 == 0:
                print(f"  추출 중... ({i + 1}/{extract_num_frames})")

        qpos_array = np.array(qpos_list)

        # 저장
        bvh_name_stem = Path(bvh_path).stem
        output_path = OUTPUT_DIR / f"{bvh_name_stem}.pkl"

        motion_export = {
            "qpos": qpos_array,
            "fps": TARGET_FPS,
            "num_frames": extract_num_frames,
            "source_bvh": str(bvh_path),
        }

        with open(output_path, "wb") as f:
            pickle.dump(motion_export, f)

        print(f"  저장 완료: {output_path}")
        print(f"  qpos shape: {qpos_array.shape}")


if __name__ == "__main__":
    visualize_bvh(show_viewer=True, export_qpos=True)
