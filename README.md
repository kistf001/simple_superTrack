# SuperTrack: Motion Tracking for Physically Simulated Characters using Supervised Learning

A PyTorch implementation of physics-based motion tracking for humanoid characters in MuJoCo using a **differentiable world model** (learned dynamics) and a **policy network**.  
The goal is to train a control policy that tracks motion capture targets while remaining physically plausible.

> Note on “differentiable physics”  
> This project does **not** differentiate through MuJoCo itself. Instead, it learns a **differentiable approximation of the simulator dynamics** (world model) and backpropagates through that model during policy training.

## Demo

![Policy Tracking Demo](assets/demo.gif)

*Humanoid tracking a motion target. Red dots show target/ground-truth positions.*

---

## Dataset (LAFAN1: single clip)

This repo is set up to validate the pipeline using **one motion clip sampled from LAFAN1**:
- A single BVH file is converted into MuJoCo joint targets (`mjmotions/*.pkl`)
- The included example weights / demo are trained on **that one clip** (single-sequence training)

Place your chosen BVH under `data/motions/` (e.g., one LAFAN1 clip), run the converter, and train.

> If you want multi-clip or full-dataset training, you can extend the motion list and buffer collection logic, but the default workflow assumes “one-clip” for simplicity.

---

## Overview

This implementation follows the SuperTrack approach with two neural networks:

1. **World Dynamics Network (Learned World Model)**
   - A differentiable approximation of physics dynamics
   - Predicts **linear/angular accelerations** from state + action
   - Enables gradient-based optimization through a learned simulator surrogate

2. **Policy Control Network**
   - Generates **residual joint target corrections** (Δqpos-style) for tracking
   - Takes current state and target pose as input
   - Outputs corrections that are applied as **position-actuator targets** (e.g., PD target residuals)

---

## Training Pipeline (High-level)

```

┌─────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Data         │ -> │ World Model  │ -> │ Policy       │       │
│  │ Collection   │    │ Training     │    │ Training     │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         v                   v                   v                │
│  ┌──────────────────────────────────────────────────────┐       │
│  │                   Replay Buffer                       │       │
│  │      (circular buffer with chunk-based sampling)      │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

```

- **Collection**: rollouts in MuJoCo to populate a replay buffer
- **World model**: trained to predict short-horizon dynamics (chunked)
- **Policy**: trained via differentiable rollouts through the learned world model

---

## Project Structure

```

├── main.py                 # Training entry point
├── config.py               # Configuration (dataclasses + TOML)
├── env.py                  # MuJoCo environment wrapper
├── network.py              # Neural networks (World/Policy)
├── buffer.py               # Replay buffer
├── transforms.py           # Coordinate transforms
├── logic_for_collect.py    # Data collection
├── logic_for_world.py      # World model training
├── logic_for_policy.py     # Policy training
├── bvh.py                  # BVH parser
├── motion_export.py        # BVH → MuJoCo converter
├── policy_viewer.py        # Visualization + GIF recording
├── configs/                # TOML configs
├── model/                  # MuJoCo XML model
├── data/weight/            # Model weights
├── mjmotions/              # Converted motions (pkl)
├── assets/                 # Demo GIF
└── utils/
├── quaternion/         # Quaternion ops
└── iksolver.py         # IK solver

````

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- MuJoCo 3.0+ (Python bindings via `pip install mujoco`)
- NumPy, SciPy
- imageio (for GIF recording)

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate

pip install torch numpy scipy mujoco imageio
````

> If you use CUDA, install the CUDA-enabled PyTorch build according to the official PyTorch instructions.

---

## Usage

### 1) Motion Data Preparation (LAFAN1 single clip)

Put **one** BVH file into `data/motions/`:

```bash
python motion_export.py
# Outputs pkl files to mjmotions/
```

### 2) Training

```bash
# Run with default configuration
python main.py

# Run with custom TOML config
python main.py --config configs/exp.toml
```

The training loop typically alternates between:

* collecting trajectories into the replay buffer
* fitting the world model on short chunks
* training the policy on longer chunks via differentiable rollouts

(Exact iteration counts / chunk sizes are defined in `configs/*.toml`.)

### 3) Visualization

```bash
# View policy tracking
python policy_viewer.py

# Record demo GIF (4s delay, 5s duration)
python policy_viewer.py --record

# Custom recording
python policy_viewer.py -r --delay 2 --duration 10 -o assets/custom.gif
```

---

## Configuration

Configuration uses dataclasses with optional TOML overrides:

```toml
# configs/example.toml
[process]
device = "cuda:0"
buffer_size = 262144

[optimizer]
world_learning_rate = 1e-4
policy_learning_rate = 1e-5

[loss_weights.policy]
pos = 1.0
vel = 0.2
rot = 1.0
```

Key sections:

* `process`: device, buffer size, model paths
* `optimizer`: learning rates, grad clipping
* `buffer`: chunk sizes for world/policy training
* `loss_weights`: loss term weights
* `policy_training`: noise std, output scaling, etc.

---

## Algorithm Details (Implementation-oriented)

Notation:

* `S_t`: reference (kinematic) target state at time t (from mocap / converted motion)
* `P_t`: predicted simulated state at time t (via learned world model rollout)
* `a_t`: control action / residual target generated by policy
* `W(·)`: learned world model predicting accelerations
* `Π(·)`: policy network producing residual joint target corrections

### World Model Training (short-horizon)

1. Sample **short chunks** from replay buffer (e.g., 8 frames)
2. Initialize `P_0 ← S_0`
3. For t = 1..T:

   * Predict accelerations with world model:
     `W(Local(P_{t-1}), a_{t-1}) → (lin_acc, ang_acc)`
   * Integrate forward one step:
     `P_t ← Integrate(P_{t-1}, lin_acc, ang_acc, dt)`
4. Compute loss in **world space** between predicted and recorded states.

> The integration method follows the repo’s implementation (dt from the env). Keep this consistent between data collection and model training.

### Policy Training (longer-horizon, through learned dynamics)

1. Sample **longer chunks** from replay buffer (e.g., 32 frames)
2. Initialize `P_0 ← S_0`
3. For t = 1..T:

   * Policy inference:
     `Δq_t ← Π([Local(P_{t-1}), Local(S_t)])`
   * Optional exploration noise: `Δq_t ← Δq_t + ε`
   * Convert residual target into control input for world model / actuators
     (e.g., position-target residuals for PD-style actuators)
   * World prediction + integration:
     `P_t ← Rollout(W, P_{t-1}, Δq_t, dt)`
4. Compute tracking loss in **local space** to encourage invariance:

   * position/velocity/rotation tracking terms (see `loss_weights`)

---

## Coordinate Transforms (Local Space)

All observations are converted to **root-relative local space**:

* Positions: relative to root position and rotated into root frame
* Velocities: rotated into root frame
* Rotations: expressed relative to root orientation
* Angular velocities: rotated into root frame

This improves translation/rotation invariance and stabilizes learning.

---

## Network Architecture (as implemented)

### WorldDynamicsNetwork

* Encoders for observation + action
* Multi-layer MLP (hidden size defined in config)
* Outputs linear & angular accelerations (per body / per feature block)

### PolicyControlNetwork

* Observation/target encoders
* MLP policy head
* Tanh-scaled residual outputs (Δq targets), scaled by config

(See `network.py` for exact layer sizes and activations.)

---

## License

MIT License

---

## References

* [SuperTrack: Motion Tracking for Physically Simulated Characters using Supervised Learning](https://arxiv.org/abs/2105.08936)
* [MuJoCo Documentation](https://mujoco.readthedocs.io/)
* [LAFAN1 Dataset (Ubisoft La Forge)](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)
