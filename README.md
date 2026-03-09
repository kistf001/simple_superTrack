# SuperTrack: Motion Tracking for Physically Simulated Characters using Supervised Learning

<p align="center">
  <img src="./assets/demo.gif" alt="SuperTrack policy tracking demo" width="760">
</p>

Training-oriented **PyTorch + MuJoCo** implementation of the paper **"SuperTrack: Motion Tracking for Physically Simulated Characters using Supervised Learning"**.

---

## What this repository provides

- **SuperTrack-style training pipeline**
- **PyTorch implementation**
- **MuJoCo humanoid simulation**
- **BVH to target-motion conversion**
- **Learned world dynamics model**
- **Policy training through differentiable rollouts**
- **Policy viewer with GIF recording**
- Code organized for reading and modification

---

## Dataset setup

This repository is currently structured around a **single BVH clip workflow**.

Recommended usage:

1. place one BVH file under `data/motions/`
2. convert it into MuJoCo-friendly motion targets
3. train the world model and policy using the generated motion data

Generated motion targets are written to:

```text
mjmotions/
```

If you want multi-clip or full-dataset training later, you can extend the motion loading and replay-buffer logic, but the default flow is intentionally kept simple.

---

## Requirements

Recommended environment:

- **Python 3.11+**
- **PyTorch 2.0+**
- **MuJoCo 3.0+**
- **NumPy**
- **SciPy**
- **imageio** (for GIF recording)

> Python 3.11+ is recommended because the current configuration code uses `tomllib`.

---

## Installation

```bash
git clone https://github.com/kistf001/supertrack-pytorch-mujoco.git
cd supertrack-pytorch-mujoco

python -m venv .venv
source .venv/bin/activate
pip install torch numpy scipy mujoco imageio
```

For CUDA, install the appropriate PyTorch build for your environment.

---

## Quick start

### 1) Prepare motion data

Put one BVH file into:

```text
data/motions/
```

Then run:

```bash
python motion_export.py
```

This converts the motion into MuJoCo-friendly target data and writes the output into:

```text
mjmotions/
```

---

### 2) Train

Run the default training configuration:

```bash
python main.py
```

To use a TOML config override:

```bash
python main.py --config path/to/your_config.toml
```

The training loop follows this high-level cycle:

1. collect trajectories in MuJoCo
2. fill / update the replay buffer
3. train the world model on short chunks
4. train the policy using differentiable rollouts through the learned world model
5. periodically save checkpoints

---

### 3) Visualize policy behavior

Launch the viewer:

```bash
python policy_viewer.py
```

Record a demo GIF:

```bash
python policy_viewer.py --record
```

Custom recording:

```bash
python policy_viewer.py --record --delay 2 --duration 10 --output assets/custom.gif
```

---

## Important checkpoint note

There is currently a small path mismatch between training and visualization:

- `main.py` saves checkpoints such as:
  - `policy_final.pth`
  - `world_final.pth`
- `policy_viewer.py` currently looks for:
  - `data/weight/policy.pth`

So after training, you may need to move or copy the trained policy checkpoint manually, for example:

```bash
mkdir -p data/weight
cp policy_final.pth data/weight/policy.pth
```

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force data/weight
Copy-Item policy_final.pth data/weight/policy.pth
```

Alternatively, you can edit `ViewerConfig.policy_path` inside `policy_viewer.py`.

---

## Training pipeline

```text
Data Collection -> Replay Buffer -> World Model Training -> Policy Training
```

More concretely:

- **Data collection**
  - roll out trajectories in MuJoCo
  - gather observations, controls, and target motion information
  - push samples into the replay buffer

- **World model training**
  - train a learned dynamics model on short motion chunks
  - predict future motion evolution from state + action

- **Policy training**
  - use the learned world model as a differentiable surrogate
  - optimize the policy to reduce tracking error over longer horizons

---

## Configuration

Configuration is defined through dataclasses in `config.py`, with optional TOML overrides.

Main categories include:

- `process`
- `training`
- `data_collection`
- `buffer`
- `network_architecture`
- `optimizer`
- `loss_weights`
- `policy_training`
- `simulation`

Typical configurable items include:

- device
- buffer size
- world / policy chunk sizes
- learning rates
- gradient clipping
- hidden sizes
- loss weights
- exploration noise
- output scaling
- model path

For the most reliable first run, start with:

```bash
python main.py
```

and only then introduce custom TOML overrides.

---

## References

- Paper: https://dl.acm.org/doi/10.1145/3478513.3480527
- Orange Duck article / paper page: https://theorangeduck.com/page/supertrack-motion-tracking-physically-simulated-characters-using-supervised-learning

---

## Citation

If this repository helps your work, please cite the original paper.

---

## License

MIT License
