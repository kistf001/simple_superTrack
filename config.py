"""
Configuration for single-process training using dataclasses.

Supports both programmatic defaults and TOML file overrides.
Set CONFIG_FILE environment variable to load from TOML.
"""

import os
import tomllib
from dataclasses import dataclass, field, fields
from typing import Any, List, get_args, get_origin


class ConfigBase:
    """Base class that supports both attribute and dictionary-style access."""

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


# ===== Process Configuration =====

@dataclass(frozen=True)
class ProcessConfig(ConfigBase):
    """Single-process training configuration."""
    device: str = "cuda:0"
    buffer_size: int = 8192 * 8 * 4  # 262144 frames
    model_path: str = "model/humanoid.xml"
    network_sync_period: int = 5


@dataclass(frozen=True)
class TrainingConfig(ConfigBase):
    """Training loop configuration."""
    model_save_interval: int = 100


@dataclass(frozen=True)
class DataCollectionConfig(ConfigBase):
    """Data collection configuration."""
    motions_dir: str = "mjmotions/"
    total_samples: int = 1024
    use_standing_pose: bool = True


# ===== Buffer Configuration =====

@dataclass(frozen=True)
class BufferConfig(ConfigBase):
    """Replay buffer configuration."""
    world_chunk_size: int = 8       # Frames per world model training chunk
    policy_chunk_size: int = 32     # Frames per policy training chunk
    world_sample_percent: float = 0.05   # Fraction of chunks to sample
    policy_sample_percent: float = 0.05  # Fraction of chunks to sample
    recent_data_ratio: float = 0.05      # Reserved for future use
    recent_sample_ratio: float = 0.2     # Reserved for future use


# ===== Network Configuration =====

@dataclass(frozen=True)
class NetworkArchitectureConfig(ConfigBase):
    """Neural network architecture configuration."""
    world_hidden_size: int = 1024
    policy_hidden_size: int = 512


@dataclass(frozen=True)
class OptimizerConfig(ConfigBase):
    """Optimizer configuration."""
    world_learning_rate: float = 1e-4
    policy_learning_rate: float = 1e-5
    gradient_clip_max_norm: float = 1.0
    training_epochs: int = 1


# ===== Loss Configuration =====

@dataclass(frozen=True)
class WorldLossWeights(ConfigBase):
    """Loss weights for world dynamics training (WORLD space)."""
    pos: float = 1.0
    vel: float = 1.0
    rot: float = 1.0
    ang: float = 1.0


@dataclass(frozen=True)
class RegDecayConfig(ConfigBase):
    """
    Regularization weight decay configuration.

    3-phase schedule: hold -> exponential decay -> hold
    - Phase 1 (0 ~ decay_start): Use initial weights
    - Phase 2 (decay_start ~ decay_end): Exponential decay
    - Phase 3 (decay_end ~ inf): Use final weights
    """
    l1_initial: float = 1.0
    l1_final: float = 0.01
    l2_initial: float = 1.0
    l2_final: float = 0.01
    decay_start: int = 800
    decay_end: int = 1500


@dataclass(frozen=True)
class PolicyLossWeights(ConfigBase):
    """Loss weights for policy training (LOCAL space)."""
    pos: float = 1.0
    vel: float = 0.2
    rot: float = 1.0
    ang: float = 0.2
    height: float = 1.0
    normal: float = 1.0
    reg_decay: RegDecayConfig = field(default_factory=RegDecayConfig)


@dataclass(frozen=True)
class LossWeightsConfig(ConfigBase):
    """Combined loss weights configuration."""
    world: WorldLossWeights = WorldLossWeights()
    policy: PolicyLossWeights = PolicyLossWeights()


# ===== Training Configuration =====

@dataclass(frozen=True)
class PolicyTrainingConfig(ConfigBase):
    """Policy-specific training configuration."""
    noise_std: float = 0.1          # Exploration noise standard deviation
    output_scale: float = 2         # Output scaling factor
    target_upward_vector: List[float] = None  # Gravity direction target

    def __post_init__(self):
        if self.target_upward_vector is None:
            object.__setattr__(self, 'target_upward_vector', [-0.001, 0.0, 1.0])


@dataclass(frozen=True)
class SimulationConfig(ConfigBase):
    """MuJoCo simulation configuration."""
    mujoco_substeps: int = 4    # Physics substeps per control step
    min_dt: float = 1e-6        # Minimum timestep (prevents division by zero)


# ===== Main Configuration =====

@dataclass(frozen=True)
class Config:
    """
    Unified configuration class.

    Contains all configuration sections as nested dataclasses.
    Supports loading from TOML files with automatic merging of defaults.
    """
    process: ProcessConfig = field(default_factory=ProcessConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data_collection: DataCollectionConfig = field(default_factory=DataCollectionConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    network_architecture: NetworkArchitectureConfig = field(default_factory=NetworkArchitectureConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss_weights: LossWeightsConfig = field(default_factory=LossWeightsConfig)
    policy_training: PolicyTrainingConfig = field(default_factory=PolicyTrainingConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)


# ===== TOML Loading Utilities =====

def _get_nested_dataclass_type(field_type) -> type | None:
    """
    Extract dataclass type from Optional or complex type hints.

    Examples:
        ProcessConfig -> ProcessConfig
        Optional[BufferConfig] -> BufferConfig
        List[float] -> None (not a dataclass)

    Args:
        field_type: Type annotation to analyze

    Returns:
        Dataclass type if found, None otherwise
    """
    origin = get_origin(field_type)
    if origin is None:
        if hasattr(field_type, "__dataclass_fields__"):
            return field_type
        return None
    for arg in get_args(field_type):
        if hasattr(arg, "__dataclass_fields__"):
            return arg
    return None


def _dict_to_dataclass(cls: type, data: dict[str, Any], defaults=None):
    """
    Recursively convert dictionary to dataclass instance.

    Handles nested dataclasses and preserves default values
    for missing keys in the input dictionary.

    Args:
        cls: Target dataclass type
        data: Dictionary with configuration values
        defaults: Default instance to use for missing values

    Returns:
        Dataclass instance with merged values
    """
    if not data:
        return defaults if defaults else cls()

    kwargs = {}
    for f in fields(cls):
        name = f.name
        default_val = getattr(defaults, name, None) if defaults else None

        if name not in data:
            if default_val is not None:
                kwargs[name] = default_val
            continue

        value = data[name]
        nested_type = _get_nested_dataclass_type(f.type)

        if nested_type and isinstance(value, dict):
            # Recursively convert nested dataclass
            kwargs[name] = _dict_to_dataclass(nested_type, value, default_val)
        else:
            kwargs[name] = value

    return cls(**kwargs) if kwargs else cls()


def _load_from_toml(path: str) -> Config:
    """
    Load configuration from TOML file.

    Merges TOML values with dataclass defaults.
    Missing keys use default values from Config dataclass.

    Args:
        path: Path to TOML configuration file

    Returns:
        Config instance with merged values
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)

    defaults = Config()
    kwargs = {}

    for f in fields(Config):
        section_name = f.name
        section_class = _get_nested_dataclass_type(f.type) or getattr(defaults, section_name).__class__

        if section_name in data:
            default_section = getattr(defaults, section_name)
            kwargs[section_name] = _dict_to_dataclass(
                section_class, data[section_name], default_section
            )

    return Config(**kwargs)


# ===== Global Configuration Instance =====

_config_file = os.environ.get("CONFIG_FILE")
if _config_file:
    config: Config = _load_from_toml(_config_file)
else:
    config: Config = Config()
