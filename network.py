"""
Neural network models for robot dynamics prediction and control.

Contains two networks:
- WorldDynamicsNetwork: Predicts physics accelerations from state and control
- PolicyControlNetwork: Generates control corrections from current and target states
"""

import torch
import torch.nn as nn


class WorldDynamicsNetwork(nn.Module):
    """
    Predicts physics accelerations from local state and control inputs.

    Architecture: Dual-encoder MLP with skip connections
    - Observation encoder: local_state -> hidden features
    - Action encoder: control -> hidden features
    - Combined features -> 5-layer MLP -> output heads

    Input: local_state (flattened) + control
    Output: (linear_acc, angular_acc) per body
    """

    def __init__(
        self,
        observation_size: int,
        control_size: int,
        hidden_size: int = 512,
        num_bodies: int = 5,
        dropout_rate: float = 0.2,
        device: str = "cpu",
    ):
        """
        Initialize world dynamics network.

        Args:
            observation_size: Dimension of local state observation
            control_size: Dimension of control input (number of actuators)
            hidden_size: Hidden layer dimension
            num_bodies: Number of bodies to predict accelerations for
            dropout_rate: Dropout rate (currently unused)
            device: Device to run network on
        """
        super().__init__()

        self.num_bodies = num_bodies

        # ===== Encoders =====
        self.obs_encoder = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(control_size, hidden_size),
            nn.ReLU(),
        )

        # ===== Main Network (5 layers) =====
        self.main = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # ===== Output Heads =====
        self.velocity_head = nn.Linear(hidden_size, num_bodies * 3)
        self.angular_velocity_head = nn.Linear(hidden_size, num_bodies * 3)

        # ===== Initialization =====
        self.apply(self._init_weights)
        self.device = torch.device(device)
        self.to(self.device)

    def _init_weights(self, module: nn.Module):
        """
        Initialize weights using Xavier uniform initialization.

        Args:
            module: Network module to initialize
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> tuple:
        """
        Forward pass: predict accelerations from state and control.

        Args:
            observations: Local space state [batch, observation_size]
            actions: Joint targets [batch, control_size]

        Returns:
            tuple: (linear_acc [batch, num_bodies, 3],
                    angular_acc [batch, num_bodies, 3])
        """
        # Encode inputs separately
        obs_features = self.obs_encoder(observations)
        action_features = self.action_encoder(actions)

        # Combine and process
        combined = torch.cat([obs_features, action_features], dim=-1)
        features = self.main(combined)

        # Generate output accelerations
        vel = self.velocity_head(features)
        ang_vel = self.angular_velocity_head(features)

        # Reshape to [batch, num_bodies, 3]
        vel = vel.view(-1, self.num_bodies, 3)
        ang_vel = ang_vel.view(-1, self.num_bodies, 3)

        return vel, ang_vel


class PolicyControlNetwork(nn.Module):
    """
    Generates joint corrections from current and target states.

    Architecture: Simple MLP with dropout
    - Observation encoder: concat(local_current, local_target) -> hidden
    - Policy network: hidden -> hidden/2
    - Output head: hidden/2 -> control corrections

    Input: concat(local_current_state, local_target_state)
    Output: joint position corrections (added to GT qpos)
    """

    def __init__(
        self,
        observation_size: int,
        control_size: int,
        hidden_size: int = 512,
        dropout_rate: float = 0.1,
        device: str = "cpu",
    ):
        """
        Initialize policy control network.

        Args:
            observation_size: Dimension of concatenated observations (2x local_size)
            control_size: Dimension of control output (number of actuators)
            hidden_size: Hidden layer dimension
            dropout_rate: Dropout rate for regularization
            device: Device to run network on
        """
        super().__init__()

        self.control_size = control_size

        # Load output scale from config
        from config import config
        self.output_scale = config.policy_training.output_scale

        # ===== Observation Encoder =====
        self.obs_encoder = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # ===== Policy Network =====
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
        )

        # ===== Output Head =====
        self.control_head = nn.Linear(hidden_size // 2, control_size)

        # ===== Initialization =====
        self.apply(self._init_weights)
        self.device = torch.device(device)
        self.to(self.device)

    def _init_weights(self, module: nn.Module):
        """
        Initialize weights using Xavier uniform initialization.

        Args:
            module: Network module to initialize
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: generate control corrections from observations.

        Args:
            observations: Concatenated [local_current, local_target] [batch, observation_size]

        Returns:
            Control corrections [batch, control_size]
            Output is scaled tanh, typically in range [-0.01, 0.01] radians
        """
        # Encode observations
        obs_features = self.obs_encoder(observations)

        # Process through policy network
        policy_features = self.policy_net(obs_features)

        # Generate control corrections
        controls = self.control_head(policy_features)

        # Apply tanh and scale output
        # tanh bounds to [-1, 1], then scale reduces magnitude
        # For position control: output is added to GT qpos as correction
        controls = torch.tanh(controls) * self.output_scale

        return controls
