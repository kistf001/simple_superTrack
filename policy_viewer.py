"""
MuJoCo Viewer - Real-time policy visualization.

Visualizes trained policy network controlling a humanoid
to track ground truth motion data.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch

from env import ObservationCollector
from logic_for_collect import generate_standing_gt_data, load_gt_data
from network import PolicyControlNetwork
from transforms import local_numpy


@dataclass
class ViewerConfig:
    """Visualization configuration."""

    model_path: str = "model/humanoid.xml"
    policy_path: str = "data/weight/policy.pth"
    chunk_size: int = 32
    truss_distance_threshold: float = 0.30
    use_standing_pose: bool = False
    device: str = "cpu"

    # Recording settings
    record: bool = False
    record_delay: float = 4.0      # Delay before recording (seconds)
    record_duration: float = 5.0   # Recording duration (seconds)
    record_fps: int = 30           # Recording FPS
    record_output: str = "assets/demo.gif"

    @property
    def truss_distance_sq(self) -> float:
        return self.truss_distance_threshold ** 2


@dataclass
class SimulationState:
    """Simulation state management."""

    frame_idx: int = 0
    reset_flag: bool = False
    reset_count: int = 0
    first_run: bool = True
    pos: Optional[np.ndarray] = None
    vel: Optional[np.ndarray] = None
    rot: Optional[np.ndarray] = None
    ang_vel: Optional[np.ndarray] = None

    def mark_initialized(self) -> None:
        self.first_run = False

    def trigger_reset(self) -> None:
        self.reset_flag = True

    def complete_reset(self) -> None:
        self.reset_flag = False
        self.reset_count += 1

    def advance_frame(self, total_frames: int) -> None:
        self.frame_idx = (self.frame_idx + 1) % total_frames

    def needs_reset(self, chunk_size: int) -> bool:
        return self.first_run or (self.frame_idx % chunk_size == 0 and self.reset_flag)


class PolicyVisualizer:
    """Visualizes policy network in MuJoCo environment."""

    def __init__(self, config: ViewerConfig):
        self.config = config
        self.state = SimulationState()

        # Initialize environment
        self.env = ObservationCollector(config.model_path)
        self.model = self.env.model
        self.data = self.env.data

        # Timing setup (50Hz)
        self.step_time = self.model.opt.timestep * 4

        # Load policy network
        self.policy_net = self._load_policy()

        # Load ground truth data
        self.gt_data = self._load_gt_data()

        # Recording state
        self.frames: List[np.ndarray] = []
        self.recording = False
        self.record_start_time = None

    def _load_policy(self) -> PolicyControlNetwork:
        """Load policy network from checkpoint."""
        net = PolicyControlNetwork(
            observation_size=self.env.local_size * 2,  # [Local(P), Local(K)]
            control_size=self.env.ctrl_size,
            device=self.config.device,
        )

        policy_path = Path(self.config.policy_path)
        if policy_path.exists():
            try:
                net.load_state_dict(
                    torch.load(policy_path, map_location=self.config.device, weights_only=True)
                )
                print(f"Policy loaded: {policy_path}")
            except Exception as e:
                print(f"Policy load failed: {e}")
        else:
            print("No policy found - using random weights")

        net.eval()
        return net

    def _load_gt_data(self) -> list:
        """Load ground truth motion data."""
        if self.config.use_standing_pose:
            gt_data, _ = generate_standing_gt_data(self.model)
        else:
            gt_data, _ = load_gt_data(self.model)

        print(f"GT data: {len(gt_data)} frames")
        return gt_data

    @property
    def total_frames(self) -> int:
        return len(self.gt_data)

    @property
    def current_gt(self) -> tuple:
        """Get current frame's ground truth data."""
        return self.gt_data[self.state.frame_idx % self.total_frames]

    def _draw_gt_markers(self, viewer, pos_gt: np.ndarray) -> None:
        """Draw GT positions as red spheres."""
        viewer.user_scn.ngeom = 0
        for p in pos_gt:
            if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                break
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                [0.02, 0, 0],
                p,
                np.eye(3).flatten(),
                [1, 0, 0, 0.7],
            )
            viewer.user_scn.ngeom += 1

    def _handle_reset(self) -> bool:
        """Handle reset. Returns True if reset occurred."""
        if not self.state.needs_reset(self.config.chunk_size):
            return False

        _, _, _, _, _, gt_qpos = self.current_gt
        pos, vel, rot, ang_vel = self.env.reset(qpos=gt_qpos)
        self.state.pos, self.state.vel = pos, vel
        self.state.rot, self.state.ang_vel = rot, ang_vel

        if self.state.first_run:
            self.state.mark_initialized()
            print(f"[{self.state.frame_idx:4d}] Initialized")
        else:
            self.state.complete_reset()
            print(f"[{self.state.frame_idx:4d}] Reset executed (#{self.state.reset_count})")

        self.state.advance_frame(self.total_frames)
        return True

    def _check_reset_condition(self, pos_gt: np.ndarray) -> None:
        """Check reset condition (head height difference > 30cm triggers flag)."""
        HEAD_IDX = 1  # Head body index (excluding world)
        HEAD_HEIGHT_THRESHOLD = 0.30  # 30cm

        head_height_diff = abs(self.state.pos[HEAD_IDX, 2] - pos_gt[HEAD_IDX, 2])

        if head_height_diff > HEAD_HEIGHT_THRESHOLD:
            if not self.state.reset_flag:
                self.state.trigger_reset()
                print(f"[{self.state.frame_idx:4d}] Reset flag ON (head_z_diff={head_height_diff:.3f}m)")

    def _infer_action(self, pos_gt: np.ndarray, vel_gt: np.ndarray,
                       rot_gt: np.ndarray, ang_gt: np.ndarray) -> np.ndarray:
        """Policy inference - Local(P) + Local(K) (SuperTrack Algorithm 2)."""
        local_P = local_numpy(
            self.state.pos, self.state.vel, self.state.rot, self.state.ang_vel
        ).astype(np.float32)
        local_K = local_numpy(pos_gt, vel_gt, rot_gt, ang_gt).astype(np.float32)
        combined = np.concatenate([local_P, local_K])

        with torch.no_grad():
            action = self.policy_net(torch.from_numpy(combined).unsqueeze(0))
            return action.squeeze(0).cpu().numpy()

    def _log_action_stats(self, action: np.ndarray) -> None:
        """Print action statistics (every 100 frames)."""
        if self.state.frame_idx % 100 != 0:
            return

        print(f"\n[Frame {self.state.frame_idx}] Action stats:")
        print(f"  Mean:  {action.mean():8.5f}  Std: {action.std():8.5f}")
        print(f"  Min:   {action.min():8.5f}  Max: {action.max():8.5f}")
        print(f"  |Mean|: {np.abs(action).mean():8.5f}")

    def _simulation_step(self) -> None:
        """Execute one simulation step."""
        pos_gt, vel_gt, rot_gt, ang_gt, gt_joint_qpos, _ = self.current_gt

        self._check_reset_condition(pos_gt)

        action = self._infer_action(pos_gt, vel_gt, rot_gt, ang_gt)
        self._log_action_stats(action)

        # Position control: GT qpos + policy correction
        target_qpos = gt_joint_qpos + action

        pos, vel, rot, ang_vel, _ = self.env.step(target_qpos)
        self.state.pos, self.state.vel = pos, vel
        self.state.rot, self.state.ang_vel = rot, ang_vel

        self.state.advance_frame(self.total_frames)

    def _wait_for_timing(self, last_time: float) -> float:
        """Synchronize timing."""
        elapsed = time.perf_counter() - last_time
        if elapsed < self.step_time:
            time.sleep(self.step_time - elapsed)
        return time.perf_counter()

    def _save_gif(self) -> None:
        """Save recorded frames as GIF."""
        if not self.frames:
            print("No frames to save")
            return

        try:
            import imageio
        except ImportError:
            print("imageio not installed. Run: pip install imageio")
            return

        output_path = Path(self.config.record_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Downsample to target FPS if needed
        target_frames = int(self.config.record_duration * self.config.record_fps)
        step = max(1, len(self.frames) // target_frames)
        frames_to_save = self.frames[::step][:target_frames]

        print(f"Saving {len(frames_to_save)} frames to {output_path}...")
        imageio.mimsave(
            str(output_path),
            frames_to_save,
            fps=self.config.record_fps,
            loop=0
        )
        print(f"GIF saved: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")

    def run(self) -> None:
        """Main visualization loop."""
        print("=" * 40)
        print("Red dots = GT positions")
        if self.config.record:
            print(f"Recording: {self.config.record_delay}s delay, {self.config.record_duration}s duration")
        print("=" * 40)

        # Setup renderer for recording
        renderer = None
        if self.config.record:
            renderer = mujoco.Renderer(self.model, height=480, width=640)

        start_time = time.perf_counter()

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            last_time = time.perf_counter()

            while viewer.is_running():
                elapsed = time.perf_counter() - start_time

                # Recording logic
                if self.config.record:
                    if not self.recording and elapsed >= self.config.record_delay:
                        self.recording = True
                        self.record_start_time = time.perf_counter()
                        print("Recording started...")

                    if self.recording:
                        record_elapsed = time.perf_counter() - self.record_start_time
                        if record_elapsed < self.config.record_duration:
                            # Capture frame
                            renderer.update_scene(self.data)
                            self.frames.append(renderer.render().copy())
                        else:
                            # Stop recording
                            print("Recording finished.")
                            self._save_gif()
                            self.config.record = False  # Disable further recording

                if self._handle_reset():
                    continue

                pos_gt, *_ = self.current_gt
                self._simulation_step()
                self._draw_gt_markers(viewer, pos_gt)

                last_time = self._wait_for_timing(last_time)
                viewer.sync()

        print(f"\nTotal resets: {self.state.reset_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Policy Viewer")
    parser.add_argument("--record", "-r", action="store_true", help="Record demo GIF")
    parser.add_argument("--delay", type=float, default=4.0, help="Delay before recording (seconds)")
    parser.add_argument("--duration", type=float, default=5.0, help="Recording duration (seconds)")
    parser.add_argument("--output", "-o", type=str, default="assets/demo.gif", help="Output GIF path")
    args = parser.parse_args()

    config = ViewerConfig(
        record=args.record,
        record_delay=args.delay,
        record_duration=args.duration,
        record_output=args.output,
    )
    visualizer = PolicyVisualizer(config)
    visualizer.run()


if __name__ == "__main__":
    main()
