"""
MuJoCo environment for physics simulation and observation collection.

Provides step/reset interface and computes body states
(positions, velocities, rotations, angular velocities).
"""

import numpy as np
import mujoco
from utils.quaternion import numpy as quaternion_numpy
from transforms import local_numpy
from dataclasses import dataclass
from typing import Optional, Tuple
from config import config


# ===== Constants =====
IDENTITY_QUATERNION = np.array([1.0, 0.0, 0.0, 0.0])


@dataclass
class Observations:
    """Container for observation data with type safety."""

    xpos: Optional[np.ndarray] = None           # Body positions [nbody, 3]
    xquat: Optional[np.ndarray] = None          # Body quaternions [nbody, 4]
    linear_vel: Optional[np.ndarray] = None     # Linear velocities [nbody, 3]
    angular_vel: Optional[np.ndarray] = None    # Angular velocities [nbody, 3]
    time: Optional[float] = None                # Simulation time

    def __repr__(self) -> str:
        attrs = []
        for key, value in self.__dict__.items():
            if value is not None:
                if isinstance(value, np.ndarray):
                    attrs.append(f"{key}:shape{value.shape}")
                else:
                    attrs.append(f"{key}:{value}")
        return f"Observations({', '.join(attrs)})"


class ObservationCollector:
    """
    MuJoCo environment wrapper for physics simulation.

    Provides step/reset interface and computes body states
    (positions, velocities, rotations, angular velocities).
    Velocities are computed via finite differences from position/rotation changes.
    """

    def __init__(self, model_path: str) -> None:
        """
        Initialize MuJoCo model and data structures.

        Args:
            model_path: Path to MuJoCo XML model file
        """
        if model_path is None:
            raise ValueError("MuJoCo model path is required for ObservationCollector")

        # ===== MuJoCo Initialization =====
        self.model: mujoco.MjModel = mujoco.MjModel.from_xml_path(model_path)
        self.data: mujoco.MjData = mujoco.MjData(self.model)
        self.ctrl_size: int = self.model.nu

        # Real nbody (excluding world body at index 0)
        self.real_nbody: int = self.model.nbody - 1

        # ===== Observation Container =====
        self.obs: Observations = Observations()

        # ===== Size Calculations =====
        self.world_net_output_size: int = self.real_nbody * 6
        self.joint_qpos_size: int = self.model.nq - 7  # Exclude root freejoint (7 DoF)

        # Calculate local_size using dummy data
        dummy_pos = np.zeros((self.real_nbody, 3))
        dummy_vel = np.zeros((self.real_nbody, 3))
        dummy_rot = np.tile(IDENTITY_QUATERNION, (self.real_nbody, 1))
        dummy_ang_vel = np.zeros((self.real_nbody, 3))
        self.local_size: int = len(
            local_numpy(dummy_pos, dummy_vel, dummy_rot, dummy_ang_vel)
        )

        # ===== State Tracking for Velocity Computation =====
        self.prev_xquat: np.ndarray
        self.prev_xpos: np.ndarray
        self.prev_time: float

        # Initialize simulation
        self.reset()

    def reset(self, qpos: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Reset simulation to initial or given state.

        Args:
            qpos: Optional initial joint positions. If None, uses default T-pose.

        Returns:
            Tuple of (xpos, linear_vel, xquat, angular_vel) with zero velocities
        """
        # Reset MuJoCo data
        mujoco.mj_resetData(self.model, self.data)

        if qpos is not None:
            # Initialize with specified qpos (zero velocity)
            self.data.qpos[:] = qpos
            self.data.qvel[:] = 0
            # Set ctrl to joint qpos to prevent initial jump
            self.data.ctrl[:] = qpos[7:]
            mujoco.mj_forward(self.model, self.data)
        else:
            # Default T-pose initialization
            mujoco.mj_forward(self.model, self.data)

        # Store current values as previous (velocities will be zero)
        self.prev_xquat = self.data.xquat[1:].copy()
        self.prev_xpos = self.data.xpos[1:].copy()
        self.prev_time = self.data.time

        # Use collect for consistency (discard dt)
        pos, vel, rot, ang_vel, _ = self.collect()
        return pos, vel, rot, ang_vel

    def step(self, ctrl: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Step simulation with given control input.

        Args:
            ctrl: Joint position targets [ctrl_size]

        Returns:
            Tuple of (xpos, linear_vel, xquat, angular_vel, dt)
        """
        # Apply control input
        if ctrl is not None:
            self.data.ctrl[:] = ctrl

        # Run simulation step
        mujoco.mj_step(self.model, self.data, config.simulation.mujoco_substeps)

        # Collect observations
        return self.collect()

    def collect(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Collect current observations from MuJoCo data.

        Computes velocities via finite differences from previous state.
        Excludes world body (index 0).

        Returns:
            Tuple of (xpos, linear_vel, xquat, angular_vel, dt)
        """
        # ===== Read Current State =====
        # Exclude world body at index 0
        self.obs.xpos = self.data.xpos[1:].copy()
        self.obs.xquat = self.data.xquat[1:].copy()
        self.obs.time = self.data.time

        # ===== Compute Time Delta =====
        dt = self.data.time - self.prev_time
        # Use minimum dt to prevent division by zero
        safe_dt = max(dt, config.simulation.min_dt)

        # ===== Compute Velocities via Finite Differences =====
        self.obs.angular_vel = self._compute_angular_velocity(
            self.data.xquat[1:].copy(), safe_dt
        )
        self.obs.linear_vel = self._compute_linear_velocity(
            self.data.xpos[1:].copy(), safe_dt
        )

        # ===== Update Previous State =====
        self.prev_xquat = self.data.xquat[1:].copy()
        self.prev_xpos = self.data.xpos[1:].copy()
        self.prev_time = self.data.time

        return (
            self.obs.xpos,
            self.obs.linear_vel,
            self.obs.xquat,
            self.obs.angular_vel,
            dt,  # Return actual dt (not safe_dt)
        )

    def _compute_angular_velocity(self, current_xquat: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute angular velocity from quaternion difference.

        Args:
            current_xquat: Current quaternions [nbody, 4]
            dt: Time delta

        Returns:
            Angular velocities [nbody, 3]
        """
        angular_vel = quaternion_numpy.quat_differentiate_angular_velocity(
            current_xquat, self.prev_xquat, dt
        )
        return angular_vel

    def _compute_linear_velocity(self, current_xpos: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute linear velocity from position difference.

        Args:
            current_xpos: Current positions [nbody, 3]
            dt: Time delta

        Returns:
            Linear velocities [nbody, 3]
        """
        linear_vel = (current_xpos - self.prev_xpos) / dt
        return linear_vel
