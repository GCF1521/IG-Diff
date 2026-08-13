# GCF-DDPM: Goal-Conditioned Trajectory Generation with Diffusion

Diffusion-based trajectory prediction for autonomous driving (Argoverse 2). The model learns residual = trajectory - prior (Hermite spline, lane centerline, or linear interpolation from history endpoint to goal), using a Transformer denoiser with cross-attention + adaLN-Zero timestep injection.

## Project Structure

```
gcf-DDPM/
├── config/
│   └── default.yaml             # All training/inference/model config
├── data/
│   ├── av2_dataset.py           # Argoverse2 PyTorch Dataset (real AV2 data support)
│   ├── coordinate_utils.py      # Global ↔ local coordinate transforms
│   ├── normalization.py         # Fixed-scale normalization (SCALE=50m, RESIDUAL_SCALE=2.0m, chord-frame: LON=2.5m, LAT=1.0m, lane prior)
│   ├── map_encoding.py          # Lane segment → 46-dim token (real centerline from AV2 API)
│   └── neighbor_encoding.py     # Neighbor agent → 6-dim token per frame (vehicle/pedestrian/cyclist)
├── model/
│   ├── tf_cross_denoiser.py     # CrossAttn denoiser (ConditionEncoder + 6 AdaLNBlocks)
│   ├── diffusion.py             # DiffusionProcess: cosine schedule, add_noise, DDIM step
│   ├── dps_guidance.py          # CFG + DPS gradient guidance at inference
│   ├── auxiliary_losses.py      # Auxiliary losses (smoothness, velocity, endpoint, curvature)
│   └── smoothing.py             # Savitzky-Golay trajectory post-processing
├── src/
│   ├── train.py                 # Single-stage training loop
│   ├── inference.py             # DDIM sampling + visualization generation
│   ├── eval.py                  # Metric evaluation (minADE, minFDE, etc.)
│   └── collision_partner.py     # Isomorphic collision trajectory generation
├── viz/
│   ├── viz_trajectory.py        # Trajectory, endpoint, confidence ellipse plots
│   ├── viz_scene.py             # BEV scene (map + agents + trajectories + ped crossings)
│   ├── viz_denoising.py         # Step-by-step denoising visualization
│   ├── viz_attention.py         # Self-attention & cross-attention heatmaps
│   ├── viz_score_heatmap.py     # Driving score spatial heatmap
│   ├── viz_training.py          # Training-time visualizations (noise, residual)
│   ├── viz_animation.py        # BEV scene animation (GIF/MP4, 11s scenario playback)
│   ├── viz_collision.py         # Collision scene (AV2-style global BEV, map + agents + generated overlay)
│   ├── viz_collision_animation.py  # Collision MP4 (av2-api scenario_visualization style, top-5 pairs)
│   ├── viz_evaluation.py        # Eval metric distribution & per-scenario plots
│   ├── viz_prior_comparison.py  # Hermite vs Lane vs GT prior comparison per modality
│   ├── utils.py                 # Visualization utility functions
│   └── style.py                 # Dark theme + save_figure utility
├── docs/
│   ├── visualization_guide.md   # Detailed viz interpretation & diagnostic guide
│   └── model_documentation.md   # Comprehensive model/architecture documentation for paper writing
├── test.py                      # Self-test script (validate core components)
├── requirements.txt             # Python dependencies
├── generate_av2_samples.py      # Synthetic AV2 scenario generator
├── convert_pkl_to_av2.py        # Convert pickle data to AV2 format
├── download_and_convert_av2.sh  # Download & convert AV2 data script
└── all_info.md                  # Original project design document (Chinese)
```

## 1. Environment Setup

```bash
# Create conda environment
conda create -n gcf-ddpm python=3.10
conda activate gcf-ddpm

# Install dependencies (use mirror if default pip is slow)
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# System dependencies (apt, for matplotlib rendering)
sudo apt install libxcb1 libxext6 libxrender1 libgl1-mesa-glx libglib2.0-0 libsm6 libice6 libfontconfig1
```

**Requirements:** Python 3.10+, PyTorch 2.0+, CUDA GPU. Recommended: RTX 4090 (24GB VRAM).

> **Important:** All scripts under `src/` must be run with `python -m` from the project root (e.g. `python -m src.train`), NOT `python src/train.py`. The `-m` flag ensures Python resolves cross-package imports (`data.*`, `model.*`, `viz.*`) correctly.

## 2. Self-Test

After installation, run the self-test to verify all components:

```bash
python test.py
```

Expected output (all 10 tests should pass):

```
[PASS] Normalization roundtrip
[PASS] Chord-frame residual roundtrip (numpy + torch + batched + degenerate)
[PASS] Lane centerline prior (6 sub-tests)
[PASS] Forward diffusion (add_noise)
[PASS] DDIM step: recovery error=0.0080
[PASS] Map encoding: feature dim=46
[PASS] Model dimensions: output shape=torch.Size([2, 60, 2])
[PASS] Gradient flow: 96/120 params have nonzero gradients
[PASS] DPS sampling: output shape=(1, 60, 2), intermediates=10 snapshots
[PASS] Dataset loading: 1000 scenarios, dimensions verified
All tests passed!
```

## 3. Prepare Data

### Option A: Use real Argoverse 2 data (recommended)

The dataset loader fully supports real AV2 data and extracts all available information:

- **Lane segments**: centerline (from AV2 API), left/right boundaries, lane type, intersection flag, connectivity (predecessors/successors), mark types (14 types)
- **Drivable areas**: polygon geometry for visualization
- **Pedestrian crossings**: edge geometry for visualization
- **Neighbor agents**: vehicles, buses, motorcyclists, pedestrians, cyclists — with position, heading, velocity history
- **Focal track**: identified via `focal_track_id` field from AV2 scenario

Download the Argoverse 2 Motion Forecasting dataset from [argoverse.github.io](https://argoverse.github.io/).

The dataset loader supports two directory layouts:

#### Official Argoverse 2 layout (recommended)

```
parent_dir/
├── train/                          ← data.train_dir
│   ├── {scenario_id}/
│   │   └── scenario_{scenario_id}.parquet
│   └── ...
└── log_map_archive/                ← auto-detected as map_dir
    ├── {log_id}/
    │   └── log_map_archive_{log_id}.json
    └── ...
```

Set `data.train_dir` in `config/default.yaml` to the `train/` directory path. `data.map_dir` is auto-detected from `train_dir/../log_map_archive/` — no manual configuration needed.

#### Flat layout (HuggingFace / synthetic)

Both parquet and JSON files live in the same directory. Set `map_dir=None` (default) and it will be inferred from `data_dir`.

```
data_dir/
├── scenario_{id}.parquet
├── log_map_archive_{id}.json
└── ...
```

The map_dir is resolved with this priority:
1. Explicit `map_dir` parameter
2. `data_dir/../log_map_archive/` (official layout)
3. `data_dir` itself (flat layout: map files in same dir as parquet files)

If your data uses a different layout, set `data.map_dir` explicitly in the config.

### Option B: Use included sample data

The repo includes `av2_sample/` with 63 synthetic scenarios. This is enough for a smoke test but **not enough for real training** (you'll see mode collapse / poor convergence).

### Option C: Generate more synthetic data

```bash
# Generate 200 synthetic scenarios (for extended smoke testing)
python generate_av2_samples.py -n 200 -o av2_sample_large
# Then change data.train_dir in config to av2_sample_large/
```

### Option D: Download from HuggingFace

```bash
# Download and convert AV2 data using the provided script
bash download_and_convert_av2.sh
```

## 4. Training

### Single-GPU training

```bash
# Edit config/default.yaml to set data.train_dir and training.epochs
python -m src.train --config config/default.yaml --gpus 0
```

### Multi-GPU training (DDP)

Supports DistributedDataParallel via `torchrun`. Requires `NCCL_SHM_DISABLE=1` in Docker (workaround for small `/dev/shm`).

```bash
# Train on 8 GPUs
NCCL_SHM_DISABLE=1 torchrun --nproc_per_node=8 -m src.train --config config/default.yaml --gpus 0,1,2,3,4,5,6,7

# Train on 4 GPUs
NCCL_SHM_DISABLE=1 torchrun --nproc_per_node=4 -m src.train --config config/default.yaml --gpus 0,1,2,3
```

- **Batch size** in config is per-GPU. Effective batch = `batch_size × n_gpus`.
- **Dataset cache**: first run parses all parquet+JSON (~70 min) and saves to `output/dataset_cache.pt`. Subsequent runs load the cache in seconds.
- Only rank 0 writes TensorBoard logs and saves checkpoints.
- `torchrun` sets up DDP automatically; only rank 0 writes TensorBoard logs and saves checkpoints.

Key config parameters (in `config/default.yaml`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.epochs` | 300 | Number of training epochs |
| `training.batch_size` | 128 | Batch size **per GPU** (effective = 128 × n_gpus) |
| `training.lr` | 6e-4 | Learning rate (linear scaled for global batch 1024) |
| `training.lambda_endpoint` | 5.0 | Endpoint weight starting value (decays to `lambda_endpoint_end`) |
| `training.lambda_endpoint_end` | 3.0 | Endpoint weight final value |
| `training.ema_decay` | 0.9999 | EMA model decay (half-life ~6931 steps) |
| `training.use_amp` | true | Mixed precision (bf16 on RTX 4090) |
| `training.cross_attn_lr_mult` | 10.0 | LR multiplier for adaLN gate params (overcome zero-init bottleneck) |
| `training.output_head_weight_decay` | 0.1 | Higher weight decay for output head (balance gradient dominance) |
| `data.num_workers` | 0 | DataLoader workers (0 = in-process, data already cached) |
| `training.drop_goal_p` | 0.1 | CFG: goal dropout probability |
| `training.drop_map_p` | 0.1 | CFG: map dropout probability |
| `training.drop_neighbor_p` | 0.1 | CFG: neighbor dropout probability |
| `model.dim` | 256 | Transformer hidden dimension |
| `model.n_layers` | 6 | Number of transformer blocks |
| `model.n_heads` | 4 | Attention heads |
| `data.residual_scale_lat` | 1.0 | Lateral residual normalization (chord frame, reduced for precision) |
| `data.filter_parking` | true | Filter parking scenarios from training batches |
| `data.parking_speed_kph` | 3.0 | Speed threshold for parking detection |
| `data.parking_min_consecutive_sec` | 3.0 | Min consecutive low-speed duration for parking |
| `diffusion.inference_steps` | 100 | DDIM inference steps |
| `diffusion.inference_spacing` | quadratic | Step schedule: "linear" or "quadratic" (more steps at high-t) |

### Auxiliary Training Losses

Beyond the denoising MSE loss, auxiliary losses are computed on the denoised x_0 estimate in **normalized residual space** (gradients flow through noise_pred). Computing in normalized space avoids the numerical instability of denormalizing x_0 estimates at high timesteps.

| Loss | Weight | Purpose |
|------|--------|---------|
| Smoothness | 0.2 | Penalize zigzag predictions: `mean(||x_{t+1} - 2x_t + x_{t-1}||²)` (2nd-order) |
| Velocity consistency | 0.1 | Penalize large velocity: `mean(||v_t||²)` (1st-order) |
| Endpoint consistency | 0.1 | Drive predicted residual endpoint toward zero (prior already reaches goal) |
| Curvature regularization | 0.005 | Steering smoothness: `mean((v×a)² / ||v||³)` (softened curvature) |

Key controls:

- **Timestep mask**: Only samples with `t ∈ [min_t, max_t]` contribute to aux losses. At high timesteps, x_0 estimates are amplified by `sqrt_recipm1_alpha` (≈64178 at t=999), making aux losses meaningless.
- **Gradient clipping**: Aux loss gradients are backward-propagated and clipped separately (`grad_clip=1.0`) before merging with main loss gradients, preventing aux gradient spikes from destabilizing training.
- **Stochastic dropout**: With `drop_p=0.5`, aux losses are randomly skipped on 50% of steps, acting as a regularizer similar to stochastic depth.

Config:

```yaml
training:
  auxiliary_losses:
    enabled: true             # Enable/disable auxiliary losses
    interval: 4               # Compute every N steps
    min_t: 0                  # Only compute when t >= this
    max_t: 800                # Only compute when t <= this (cover 80% of schedule)
    grad_clip: 1.0            # Max gradient norm for aux loss before merging with main gradients
    lambda_smooth: 0.2        # Smoothness loss weight (strengthened for temporal continuity)
    lambda_vel: 0.1           # Velocity consistency loss weight
    lambda_endpoint: 0.1      # Endpoint consistency loss weight
    lambda_curv: 0.005        # Curvature regularization weight
```

Auxiliary losses are logged to TensorBoard as `loss/aux_smoothness`, `loss/aux_velocity_consistency`, `loss/aux_endpoint_consistency`, `loss/aux_curvature`, `loss/aux_total`, `loss/aux_n_valid`.

### Resume from checkpoint

```bash
python -m src.train --config config/default.yaml --gpus 0 --resume output/checkpoints/train/<timestamp>/latest.pt
```

或恢复多卡训练：

```bash
NCCL_SHM_DISABLE=1 torchrun --nproc_per_node=8 -m src.train --config config/default.yaml --gpus 0,1,2,3,4,5,6,7 --resume output/checkpoints/train/<timestamp>/latest.pt
```

> `<timestamp>` is the auto-generated directory name under `output/checkpoints/train/` (Beijing time, UTC+8). Check `ls output/checkpoints/train/` for the actual value.

### Monitor with TensorBoard

```bash
tensorboard --logdir output/logs/train
```

Each training run creates a timestamped log directory, making it easy to compare runs.

**Logged scalars:**

| Scalar | Frequency | Description |
|--------|-----------|-------------|
| `loss/residual_denoising` | Every step | Endpoint-weighted denoising loss |
| `loss/base` | Every step | Base MSE loss (all frames) |
| `loss/endpoint` | Every step | Endpoint MSE loss (last frame) |
| `loss/total` | Every step | Total loss (denoising + auxiliary) |
| `loss/aux_smoothness` | Every step | Smoothness auxiliary loss |
| `loss/aux_velocity_consistency` | Every step | Velocity consistency loss |
| `loss/aux_endpoint_consistency` | Every step | Endpoint consistency loss |
| `loss/aux_curvature` | Every step | Curvature regularization loss |
| `loss/aux_total` | Every step | Weighted sum of auxiliary losses |
| `loss/aux_n_valid` | Every step | Number of batch samples with t ∈ [min_t, max_t] used for aux loss |
| `loss/ema_denoising` | Every epoch | EMA model denoising loss |
| `lr` | Every step | Current learning rate |

**Logged histograms / norms (configurable intervals):**

| Log | Frequency | Description |
|-----|-----------|-------------|
| `grad_norm/{module}` | Every `grad_norm_interval` steps | Per-module gradient norms (output_head, adaLN, self_attn, cross_attn, cond_encoder, traj_embed, timestep_embedder) |
| `grad_norm/aux_before_clip` | Every `grad_norm_interval` steps | Aux loss gradient norm before clipping |
| `condition/{name}` | Every `condition_stats_interval` steps | Goal x/y histograms, map/neighbor valid counts, residual magnitude per frame |
| `param_norm/{param_name}` | Every `param_norm_interval` steps | Per-parameter norms (dots replaced with `/` for flat TensorBoard layout) |
| `param_norm_group/{group}` | Every `param_norm_interval` steps | Average parameter norm per module group |

Config:

```yaml
logging:
  grad_norm_interval: 200
  condition_stats_interval: 1000
  param_norm_interval: 1000
  ema_compare_interval: 1       # Compare EMA vs raw model every N epochs
```

### Training tips

- **adaLN-Zero slow start**: The model uses zero-initialized gates, so self-attention and cross-attention are effectively disabled at step 0. Gradients unlock progressively (output head → adaLN gates → attention). This is normal — don't panic if attention heatmaps look uniform in the first few epochs.
- **Minimum viable training**: With 1000 real AV2 scenarios, expect ~5-10 epochs for loss to plateau. For production results, use the full Argoverse 2 dataset (~200k scenarios) with 100+ epochs.
- **Checkpoint naming**: Saved under `output/checkpoints/train/<timestamp>/best.pt` and `latest.pt`. The timestamp uses Beijing time (UTC+8) and prevents overwriting previous runs. Run `ls output/checkpoints/train/` to find the actual timestamp.

## 5. Inference

### 5.1 Single-Modal Inference (GT Goal)

All generated trajectories use the same GT endpoint. Typically `n_samples=1` for deterministic prediction:

```bash
python -m src.inference \
    --config config/default.yaml \
    --checkpoint output/checkpoints/train/<timestamp>/best.pt \
    --n_samples 1 \
    --cfg_weight 2.0 \
    --gpus 0 \
    --viz_scenarios 5 \
    --viz_denoising \
    --viz_attention \
    --viz_score_heatmap \
    --animation gif \
    --output_dir output/samples
```

Set `--n_samples > 1` without `--goal_sigma_lon` / `--goal_sigma_lat` to generate multiple trajectories all targeting the same GT goal (useful for measuring model variance).

### 5.2 Multi-Modal Inference (Goal Sampling)

Each trajectory independently samples a goal from `N(GT_goal, sigma²)`, producing diverse endpoints that cover multiple possible futures. The number of trajectories is controlled by `--n_samples`. By default (`goal_sigma_lon=0, goal_sigma_lat=0` in config), sigma **auto-scales with trajectory length**: `sigma_lon = max(chord·0.10, 0.5)m`, `sigma_lat = max(chord·0.04, 0.3)m`, giving a 2.5:1 anisotropic ratio that keeps endpoints near the drivable area across short and long scenarios.

```bash
# Generate 20 diverse trajectories with auto-scaled goal sampling
python -m src.inference \
    --config config/default.yaml \
    --checkpoint output/checkpoints/train/<timestamp>/best.pt \
    --n_samples 20 \
    --cfg_weight 2.0 \
    --gpus 0 \
    --viz_scenarios 5 \
    --output_dir output/samples
```

To override the auto-scaled sigma with fixed values (e.g. wider longitudinal spread for highway scenarios):

```bash
python -m src.inference \
    --config config/default.yaml \
    --checkpoint output/checkpoints/train/<timestamp>/best.pt \
    --n_samples 20 \
    --goal_sigma_lon 10.0 \
    --goal_sigma_lat 2.0 \
    --cfg_weight 2.0 \
    --gpus 0 \
    --output_dir output/samples
```

Goal sampling parameters:
- `--goal_sigma_lon`: perturbation along GT endpoint heading (longitudinal). Default = `None` → use config (`0.0` = auto: `max(chord·0.10, 0.5)m`)
- `--goal_sigma_lat`: perturbation perpendicular to heading (lateral). Default = `None` → use config (`0.0` = auto: `max(chord·0.04, 0.3)m`)

Recommended values:
- `lon=0, lat=0` (auto): adaptive spread that scales with trajectory length — good default
- `lon=5.0, lat=2.0`: moderate fixed spread
- `lon=10.0, lat=2.0`: wide longitudinal spread for high-speed / highway scenarios

**Visualization of multi-modal output:** all N trajectories are rendered in the **same figure** with color coding:
- Individual trajectories: steelblue, low opacity (alpha=0.15)
- Mean trajectory: solid blue line
- 95% confidence band: light blue fill
- GT trajectory: white dashed line
- Sampled goal endpoints: red dots
- GT goal: red star

### Multi-GPU Inference

Scenarios are split across GPUs (each GPU loads its own model copy and processes a subset of scenarios):

```bash
# Single-modal on 4 GPUs
python -m src.inference \
    --checkpoint output/checkpoints/train/<timestamp>/best.pt \
    --n_samples 1 \
    --gpus 0,1,2,3

# Multi-modal on 2 GPUs
python -m src.inference \
    --checkpoint output/checkpoints/train/<timestamp>/best.pt \
    --n_samples 20 --goal_sigma_lon 5.0 --goal_sigma_lat 2.0 \
    --gpus 0,1
```

### With Visualizations

```bash
python -m src.inference \
    --config config/default.yaml \
    --checkpoint output/checkpoints/train/<timestamp>/best.pt \
    --n_samples 20 \
    --goal_sigma_lon 2.5 \
    --goal_sigma_lat 1.0 \
    --cfg_weight 2.0 \
    --gpus 0 \
    --viz_scenarios 5 \
    --viz_denoising \
    --viz_attention \
    --viz_score_heatmap \
    --animation gif \
    --output_dir output/samples
```

### Collision Partner Inference (Isomorphic Guidance)

Generate collision trajectories by running the **same diffusion model** for a second vehicle (partner) with the **same goal endpoint** as ego. Both vehicles converge to the shared collision point, producing collision scenarios without modifying training.

```bash
python -m src.inference \
    --checkpoint output/checkpoints/train/<timestamp>/best.pt \
    --gpus 0 \
    --collision_partner \
    --collision_threshold 1.5 \
    --n_samples 20 \
    --n_scenarios 10
```

| Flag | Default | Description |
|------|---------|-------------|
| `--collision_partner` | off | Enable partner generation + collision_rate metric |
| `--collision_threshold` | 1.5 | Distance threshold for collision detection (meters) |
| `--n_scenarios` | 10 | Max scenarios to evaluate |

**Output**: For each scenario with a valid partner, generates:
- `scenario_X_collision.png` — Full AV2-style BEV scene in global coordinates: drivable areas (grey), lane boundaries (light grey), surrounding agent bounding boxes with trajectory tails, ego + partner generated futures (blue/red solid), GT (dark dashed), shared collision goal (red star + threshold circle)
- `scenario_X_collision_heatmap.png` — (N_ego × N_partner) minimum-distance matrix with collision threshold contour

**How it works**:
1. Select the closest non-focal, non-parking SCORED vehicle as partner
2. Set partner's goal = ego's endpoint (same collision point)
3. Re-encode all partner conditions (history, map, neighbors) in partner's local frame
4. Run DDIM sampling → partner futures → convert to global coordinates
5. Compute `collision_rate = fraction of (ego, partner) pairs with min-distance < threshold`

**Partner start heading**: Instead of using the AV2 box heading at t=50 (which can diverge from actual motion direction under side-slip / low-speed / annotation noise), the partner's Hermite start tangent is estimated from the **last 3 history frames** via `atan2(dy, dx)`. Falls back to the box heading when the two diverge by more than 45°. This mirrors the ego-side logic in `data/av2_dataset.py:_load_item`. See `src/collision_partner.py:_compute_travel_heading`.

**Evaluation** includes `collision_rate`, `collision_rate_strict` (threshold=1.0m), and `collision_mean_min_dist` in the metrics output.

### Collision Animation (av2-api style)

`--collision_animation` generates a per-scenario MP4 that strictly follows av2-api's `visualize_scenario` pipeline (BEV, focal trajectory bounds + 30m buffer, cv2.VideoWriter + mp4v codec, 10 fps, in-memory PNG buffer per frame). History phase (t=0..49) plots each track's observed states up to t; future phase (t=50..109) replaces focal & partner with the selected driving pair while **other tracks continue along their observed GT future states** (AV2 provides full 110-timestep observations for all tracks, so surrounding vehicles keep moving, not frozen).

```bash
python -m src.inference \
    --checkpoint output/checkpoints/train/<timestamp>/best.pt \
    --gpus 0 \
    --collision_partner \
    --collision_animation \
    --collision_anim_background 5 \
    --collision_anim_fps 10 \
    --n_scenarios 10
```

| Flag | Default | Description |
|------|---------|-------------|
| `--collision_animation` | off | Generate per-scenario collision MP4 (requires `--collision_partner`) |
| `--collision_anim_background` | 5 | Number of background ego trajectories (lowest minFDE) to draw |
| `--collision_anim_fps` | 10 | Animation FPS |

Output: `output/samples/<timestamp>/figures/scenario_X_collision.mp4` (110 frames, 11s, 1200×1000).

#### Top-5 collision pair selection

For each scenario, 20 ego × 20 partner = 400 candidate pairs are scored and the top 5 are visualized (pair #1 is the driving pair shown bold; #2–5 are thin overlays). The scoring pipeline:

1. **Hard constraint (disqualify)**: a pair is sent to the bottom of the ranking if **any** of these fails:
   - Either the ego or partner trajectory leaves the drivable-area polygon union at any frame (ray-casting point-in-polygon test).
   - Either the ego or partner collides with any **other in-scene vehicle** (any non-focal, non-partner VEHICLE/BUS/MOTORCYCLIST) at any future frame — "collides" = frame distance < 1.5m. AV2 tracks carry observed states across all 110 timesteps, so surrounding vehicles are checked against their real future motion, not frozen positions.
   
   Disqualified pairs are only used to fill the top-5 if fewer than 5 eligible pairs exist.
2. **Composite score** on eligible pairs (each metric min-max normalized across the eligible set, then weighted):
   - `end_dist` = `||ego_end − partner_end||` — weight 0.50 (terminal collision convergence)
   - `goal_completion` = `||ego_end − mid|| + ||partner_end − mid||` where `mid` is the pair's endpoint midpoint — weight 0.20
   - `smoothness` = mean jerk² of ego + partner (3rd finite difference) — weight 0.15
   - `curvature` = mean curvature² of ego + partner — weight 0.15

The driving pair (rank #1) is rendered bold; ranks #2–5 thin. Disqualified pairs (when shown as backfill) are drawn dashed grey. The collision point marker is the **driving pair's endpoint midpoint** (not the GT collision point), so it stays aligned with the generated trajectories actually shown.

### Saving Animations (GIF / MP4)

Add `--animation gif` or `--animation mp4` to any inference command to save a BEV scene animation that plays back the full 11-second scenario (5s history + 6s future):

```bash
python -m src.inference \
    --checkpoint output/checkpoints/train/<timestamp>/best.pt \
    --gpus 0 \
    --viz_scenarios 3 \
    --animation gif
```

| Format | Flag | Notes |
|--------|------|-------|
| GIF | `--animation gif` | No extra dependencies, ~2MB per scenario |
| MP4 | `--animation mp4` | Requires `opencv-python`, ~500KB per scenario |

For collision animations, combine with `--collision_partner`:

```bash
python -m src.inference \
    --checkpoint output/checkpoints/train/<timestamp>/best.pt \
    --gpus 0 \
    --collision_partner \
    --n_scenarios 5 \
    --viz_scenarios 5 \
    --animation mp4
```

Animations are saved to `output/samples/<timestamp>/figures/`.

| Flag | Description |
|------|-------------|
| `--gpus GPUS` | Comma-separated GPU IDs, e.g. '0,1,2,3' (required) |
| `--n_samples N` | Number of trajectory samples per scenario (default=20) |
| `--n_scenarios N` | Max number of scenarios to evaluate (default=10) |
| `--goal_sigma_lon SIGMA` | Goal perturbation sigma along endpoint heading in meters (longitudinal, default=None→config 0.0=auto) |
| `--goal_sigma_lat SIGMA` | Goal perturbation sigma perpendicular to endpoint heading in meters (lateral, default=None→config 0.0=auto) |
| `--collision_partner` | Enable collision partner generation + collision_rate metric |
| `--collision_threshold THR` | Collision distance threshold in meters (default=1.5) |
| `--collision_animation` | Generate per-scenario collision MP4 (av2-api style, requires `--collision_partner`) |
| `--collision_anim_background N` | Number of background ego trajs (lowest minFDE) in animation (default=5) |
| `--collision_anim_fps N` | Collision animation FPS (default=10) |
| `--viz_scenarios N` | Generate visualizations for first N scenarios (0 = none) |
| `--viz_denoising` | Denoising process step-by-step plots |
| `--viz_attention` | Self-attention & cross-attention heatmaps |
| `--viz_score_heatmap` | Driving score spatial heatmap |
| `--animation FORMAT` | Save BEV animation: `gif` or `mp4` (default=disabled) |
| `--dps_eta 0.3` | Enable DPS gradient guidance (0 = disabled) |
| `--dynamic_threshold 1.5` | Imagen-style x_0 clipping threshold (0 = disabled, default=1.5) |
| `--smoothing` | Apply Savitzky-Golay smoothing to generated trajectories |
| `--smoothing_window 7` | Smoothing window length (odd integer, default=7) |
| `--smoothing_polyorder 3` | Smoothing polynomial order (default=3) |
| `--kinematic_projection` | Project smoothed trajectory onto kinematic manifold (jerk/accel/curvature limits) |

### Dynamic Thresholding (DDIM Sampling)

At high timesteps (t≈999), the cosine schedule produces very small ᾱ values (~1.56e-5), causing `sqrt_recip_alphas_cumprod` to reach ~64178. This amplifies any noise prediction error in the x_0 estimate, potentially causing numerical explosion in DDIM sampling and producing trajectories thousands of meters in scale.

The `--dynamic_threshold` flag applies Imagen-style dynamic thresholding: for each sample, if the maximum absolute value in the x_0 estimate exceeds the threshold, all values are rescaled to fit. This prevents explosion while preserving the relative structure of the prediction. Default threshold=1.5 (matching the ~95th percentile of normalized residual magnitudes, adjusted for RESIDUAL_SCALE_LAT=1.0).

### Trajectory Smoothing

The `--smoothing` flag applies Savitzky-Golay filtering as post-processing after DDIM sampling. `--kinematic_projection` further projects the smoothed trajectory onto a kinematically feasible manifold (forward-backward clamping on jerk / acceleration / speed / curvature, then position reintegration). Both flags apply to **ego and partner** trajectories identically — the same `args.*` parameters control both.

Key properties:
- **Endpoint preservation**: Start and end points are held fixed via linear blending at boundaries — critical for goal-conditioned generation.
- **Start-tangent preservation**: the kinematic projection's endpoint-correction drift (the gap between the reintegrated endpoint and the goal) is distributed **only over the second half** of the trajectory (ramp 0→1 from midpoint to end). Applying it uniformly across all frames shifts frame 1 by `drift/T`, which corrupts the start-point heading — a 40m endpoint drift yields ~0.7m at frame 1, enough to flip the start heading by 90°+. Keeping the first half untouched preserves the start tangent continuity with the observed history.
- **Configurable**: Window length and polynomial order control smoothness vs. fidelity trade-off.
- Default `window=7, polyorder=3` works well for 10Hz trajectories (60 frames over 6 seconds).

Example:
```bash
python -m src.inference \
    --checkpoint output/checkpoints/train/<timestamp>/best.pt \
    --smoothing --smoothing_window 7 --smoothing_polyorder 3 \
    --kinematic_projection --gpus 0
```

Output: `output/samples/<timestamp>/results.pt` + `figures/` subdirectory with PNG images.

### BEV Scene Visualization

The BEV scene visualization renders all available AV2 map and agent data:

- **Lane boundaries**: white left/right boundaries, yellow dashed centerlines, orange intersection markers
- **Drivable areas**: dark gray filled polygons
- **Pedestrian crossings**: purple filled polygons with edge lines
- **Neighbor agents**: orange rectangles with heading, trajectory history lines
- **Focal vehicle**: cyan rectangle with blue edge
- **Generated trajectories**: green lines (individual + mean)
- **Ground truth**: white dashed line
- **Goal**: red star

## 6. Evaluation

```bash
python -m src.eval --results output/samples/<timestamp>/results.pt
```

This computes 18 metrics across all scenarios, organized into four categories:

**Accuracy Metrics:**

| Metric | Description | Good range (trained model) |
|--------|-------------|---------------------------|
| minADE | Min average displacement error (best of N samples) | < 1.0m |
| minFDE | Min final displacement error (best of N samples) | < 2.0m |
| b-minFDE | Probability-weighted minFDE (softmin over endpoint distances) | < 2.0m |
| miss_rate | Fraction where best-of-N FDE > 2.0m | < 0.3 |
| endpoint_error | Mean endpoint-to-goal distance across all samples | < 2.0m |
| goal_hit_rate | Fraction of samples with endpoint < 2.0m from goal | > 0.5 |
| collision_rate | Fraction of (ego, partner) pairs with min-distance < 1.5m | > 0.3 (with collision_partner) |
| collision_rate_strict | Same but threshold = 1.0m | > 0.1 (with collision_partner) |
| collision_mean_min_dist | Mean minimum distance between ego and partner futures | < 3.0m (with collision_partner) |

**Diversity Metrics:**

| Metric | Description | Good range |
|--------|-------------|-----------|
| diversity | Mean pairwise distance between generated endpoints | 1-5m |
| goal_diversity | Mean pairwise distance between sampled goal endpoints (requires goal_sigma_lon/lat > 0) | 0m (no sampling) / 1-5m (with sampling) |

**Kinematic Feasibility Metrics** (computed on best-of-K trajectory):

| Metric | Description | Threshold |
|--------|-------------|-----------|
| jerk_mean / jerk_max / jerk_violation_frac | Third finite difference of position | 4.0 m/s³ |
| accel_mean / accel_max / accel_violation_frac | Second finite difference of position | 3.0 m/s² |
| curvature_mean / curvature_max / curvature_rate_mean | Path curvature via cross-product formula | — |

**Efficiency & Safety Metrics:**

| Metric | Description |
|--------|-------------|
| path_efficiency | Straight-line distance / actual path length (1.0 = straight) |
| off_road_rate | Fraction of waypoints outside drivable area (requires map) |
| constraint_violation_rate | Fraction of frames violating any constraint |

Off-road rate and constraint violation rate default to NaN if map/drivable-area data is unavailable.

**Parking Scenario Filtering**: The evaluation script detects parking scenarios (speed < 3 kph for ≥ 3 consecutive seconds) and reports metrics separately for non-parking scenarios. This avoids inflating error metrics with scenarios where the vehicle is stationary and the prior/GT alignment is meaningless.

Output: printed summary table (overall + non-parking subset) + `eval/figures/` with distribution and per-scenario plots.

### Important: eval requires inference first

Eval reads the `results.pt` file generated by inference. The workflow is:

```
training → inference (generates results.pt) → evaluation (reads results.pt)
```

## 7. Data Processing Details

### Map Encoding (46-dim per lane segment)

Each lane segment is encoded into a 46-dimensional feature vector:

| Feature | Dims | Description |
|---------|------|-------------|
| centerline_local | 10×2=20 | 10 resampled centerline points in local coords, ÷ 50 (uses real AV2 API centerline when available) |
| left_boundary | 5×2=10 | 5 resampled left boundary points in local coords, ÷ 50 |
| right_boundary | 5×2=10 | 5 resampled right boundary points in local coords, ÷ 50 |
| lane_type_onehot | 3 | VEHICLE / BIKE / BUS |
| is_intersection | 1 | Boolean intersection flag |
| start_dist | 1 | Distance from centerline start to focal agent, ÷ 50 |
| start_heading_diff | 1 | Heading difference from centerline start to focal agent, ÷ π (range [-1, 1]) |

Lanes are sorted by distance to focal agent (closest first), padded to `n_lanes=24`.

Additional lane metadata (passed through for potential future use):
- **Connectivity**: `has_predecessor`, `has_successor` (from AV2 lane segment predecessors/successors)
- **Mark types**: `left_mark_type`, `right_mark_type` (14 types: SOLID_WHITE, DASHED_WHITE, SOLID_YELLOW, etc.)

### Neighbor Encoding (6-dim per frame)

Each neighbor agent is encoded as `(n_history+1, 6)` tokens:

| Token | Dims | Description |
|-------|------|-------------|
| Type token (frame 0) | 6 | [normalized_type_id, 0, 0, 0, 0, valid_mask] |
| History frame (1..20) | 6 | [Δx÷50, Δy÷50, Δheading÷π, vx÷10, vy÷10, valid_mask] |

Valid neighbor types: vehicle, bus, motorcyclist, pedestrian, cyclist.

Neighbors are sorted by priority (SCORED > UNSCORED > FRAGMENT), then by distance to focal agent, limited to `n_neighbors=6`.

### Coordinate System

All coordinates are transformed to an ego-centric local frame centered at the focal agent's position at t=50, with heading aligned to the focal agent's heading at t=50. This is the standard convention used by Diffusion-Planner, MTR, and other motion forecasting models.

### Normalization

- **Coordinates**: normalized by `SCALE=50m` (divide by 50) — applies to goal, history, map position features, and neighbor position features
- **Residuals**: chord-frame parameterization (default) — decomposed into longitudinal (along chord) and lateral (perpendicular), normalized independently by `RESIDUAL_SCALE_LON=2.5m` and `RESIDUAL_SCALE_LAT=1.0m`. Legacy isotropic mode (`residual_frame: xy`) uses `RESIDUAL_SCALE=2.0m` for both axes
- **Velocities**: normalized by `VELOCITY_SCALE=10m/s` (divide by 10) — applies to neighbor velocity features
- **Headings**: normalized by `HEADING_SCALE=π` (divide by π, range [-1, 1]) — applies to neighbor heading differences and map lane start heading differences
- Residual = trajectory - prior, where prior is a Hermite spline (default) or linear interpolation from history endpoint to goal
- **Padding mask**: invalid map/neighbor tokens are excluded from attention via `key_padding_mask` (True=ignore), preventing phantom representations from zero-padded inputs

### Chord-Frame Residual Parameterization

By default (`residual_frame: chord`), residuals are rotated into the chord-aligned frame before normalization:

```
chord = goal - history_end
chord_dir = chord / ||chord||           # unit direction along chord
perp_dir = [-chord_dir[1], chord_dir[0]] # perpendicular direction

r_lon = dot(residual, chord_dir)        # longitudinal (meters)
r_lat = dot(residual, perp_dir)         # lateral (meters)

# Independent normalization per axis
r_lon_norm = r_lon / 2.5               # RESIDUAL_SCALE_LON
r_lat_norm = r_lat / 1.0               # RESIDUAL_SCALE_LAT

# Diffusion target: pack(r_lon_norm, r_lat_norm) → (T, 2)
```

**Why this matters**: In isotropic (x,y) space, residual distributions are highly anisotropic — turning scenarios have lateral residuals 2x larger than longitudinal, while straight scenarios are 7.5x elongated longitudinally. Isotropic Gaussian noise doesn't match these elliptical distributions, causing mode collapse to straight lines for turns. Chord-frame parameterization decorrelates the axes so both lon and lat residuals have similar magnitude after normalization, matching the isotropic Gaussian noise assumption of diffusion models.

**Edge case**: When `||chord|| < 0.1m` (nearly stationary), falls back to isotropic normalization.

**Reconstruction**:
```
r_lon = r_lon_norm * 2.5
r_lat = r_lat_norm * 1.0
residual = r_lon * chord_dir + r_lat * perp_dir
trajectory = prior + residual
```

**Backward compatibility**: Set `residual_frame: xy` in config to revert to isotropic (x,y) residual normalization.

### Hermite Spline Prior

The prior provides a baseline trajectory from the history endpoint to the goal. A **cubic Hermite spline** uses start/end headings to produce a curved baseline that follows turn geometry, dramatically reducing residuals for turning and U-turn scenarios.

```
prior(t) = h00(t)·P₀ + h10(t)·M₀ + h01(t)·P₁ + h11(t)·M₁

h00(t) = 2t³ - 3t² + 1      h10(t) = t³ - 2t² + t
h01(t) = -2t³ + 3t²         h11(t) = t³ - t²

P₀ = history_end,  P₁ = goal
```

**Tangent decomposition**: Tangent vectors are decomposed into along-chord and perpendicular components with clamping to prevent overshooting (U-turns) and spurious curvature (nearly-straight paths):

```
M = dist·(along·chord_dir + perp·perp_dir)
along ∈ [0.1, 1.5]    # along-chord component
perp  ∈ [-0.75, 0.75]  # perpendicular component
```

**Smoothstep blend**: For nearly-straight trajectories where heading deviates little from the chord direction, a smoothstep blend interpolates between linear and Hermite priors:

```
α = smoothstep(max(|start_perp|, |end_perp|), 0.05, 0.21)
prior = (1-α)·linear + α·hermite
```

This avoids introducing spurious curvature for straight/slight-turn scenarios while still leveraging Hermite benefits for turns.

- `start_heading`: computed from the last two history points
- `end_heading`: computed from the last two future points
- Tangent magnitude scales with distance, so short paths get gentle arcs and long paths get wider arcs

**Why it matters**: For a left turn, the linear prior is a straight line from start to end, so the residual = entire arc (large). The Hermite prior curves naturally, so the residual is only a small deviation — much easier for the diffusion model to learn. For U-turns, the decomposed tangent clamping prevents the spline from overshooting past the goal.

Set `prior_type: linear` in config to revert to the original linear interpolation (for backward compatibility with old checkpoints).

### Lane Centerline Prior

The lane centerline prior uses AV2 map topology to produce a prior trajectory that follows actual road geometry, further reducing residuals for turning scenarios where the Hermite prior (a mathematical curve) doesn't match the real road shape.

**Pipeline**:

1. Compute centerlines from left/right lane boundary midpoints (AV2 LaneSegment has no centerline attribute)
2. Find the best start lane: closest to history_end + heading-aligned toward goal direction
3. Follow the successor topology chain (goal-aware successor selection, depth ≤ 6)
4. Bridge intersection gaps: when a lane has no centerline (intersection), follow predecessor/successor links to find the next valid centerline and interpolate across the gap
5. If successor chain endpoint is too far from goal, try greedy lane-following path as fallback
6. Arc-length resample to `n_future=60` points
7. Extract centerline curvature profile and adaptively blend with Hermite curvature:
   - High-curvature segments (turns) → use centerline curvature (follows road)
   - Low-curvature segments (straight) → use Hermite curvature (smoother)
8. **Physics constraints** on the blended curvature:
   - Lateral acceleration limit: `a_lat = v² × κ ≤ 4.0 m/s²` — clips curvature to stay within comfort driving limits
   - Curvature rate limit: `|dκ/ds| ≤ 0.5 rad/m²` — iterative forward pass to enforce continuous curvature change
   - Savitzky-Golay smoothing (window=5, poly=3) — removes residual noise from intersection interpolation artifacts
9. Integrate curvature → reconstruct path → align endpoints to history_end and goal
10. **Boundary safety push**: for each point within 1.5m (vehicle half-width 1.0m + clearance 0.5m) of a drivable-area boundary edge, push inward along the boundary's inward normal; push vectors are S-G smoothed to avoid discontinuities; endpoints are pinned
11. **Degradation logic** (3 levels):
    - **Level 0**: lane prior used as-is (boundary violations < 30%, lateral accel < 6 m/s²)
    - **Level 1** (blended): `α × lane + (1-α) × hermite` with `α = max(0.3, 1 - severity)` — triggered when boundary violations > 30% or lateral accel > 6 m/s²
    - **Level 2** (full fallback): returns Hermite — triggered when Level 1 conditions combine with small heading change (< 15°), or when no map/path/endpoint alignment is found
12. Fallback: no map / path too short / endpoint offset > 10m → Hermite

**Key features**:
- Follows real road geometry (AV2 centerlines + successor topology)
- Adaptive curvature blending: centerline curvature for turns, Hermite for straights
- Physics constraints: lateral acceleration (4 m/s²) and curvature rate (0.5 rad/m²) limits ensure smooth, drivable paths
- Boundary safety: pushes trajectories away from drivable-area edges (1.5m minimum clearance)
- Three-level degradation: gracefully falls back to Hermite when lane prior quality is poor
- Intersection gap bridging: AV2 intersections have no lane centerlines — the algorithm bridges gaps by following topology links
- Parking scenario detection: marks low-speed scenarios (< 3 kph for ≥ 3s consecutive) to filter from training/evaluation

**When to use**: `prior_type: lane` is recommended when training with real AV2 data (maps available). It provides the most accurate prior for turning scenarios (~14% RMSE improvement) while matching Hermite quality for straight scenarios via automatic fallback.

**Config**:
```yaml
data:
  prior_type: lane    # "lane", "hermite", or "linear"
  filter_parking: true              # Filter parking scenarios during training
  parking_speed_kph: 3.0            # Speed threshold for parking detection
  parking_min_consecutive_sec: 3.0  # Min consecutive low-speed duration
```

**Comparison with Hermite prior (measured on 300 AV2 scenarios)**:
| Modality | Hermite RMSE | Lane RMSE | Improvement |
|----------|-------------|-----------|-------------|
| Straight | 1.90m       | 1.90m     | ~0% (fallback) |
| Turn     | 3.11m       | 2.66m     | +14.4% |
| U-turn   | 3.93m       | 3.76m     | +4.3% |
| Overall  | 2.37m       | 2.24m     | +5.7% |

### Physics-Aware DPS Guidance

Beyond the endpoint and jerk penalties, the DPS scoring function incorporates **physics-based acceleration constraints** that enforce realistic vehicle dynamics during inference:

**Acceleration continuity penalty**: Real vehicles cannot change lateral or longitudinal acceleration instantaneously. The penalty measures the fraction of frames where the rate of change of lateral/longitudinal acceleration (i.e., lateral/longitudinal jerk) exceeds physically plausible thresholds:
- Lateral jerk threshold: 6 m/s³ (comfort boundary)
- Longitudinal jerk threshold: 8 m/s³ (emergency boundary)

**Acceleration magnitude penalty**: Constrains acceleration magnitudes to physically achievable ranges:
- Lateral acceleration: < 6 m/s² (tire grip limit, dry road ~8-10 m/s²)
- Longitudinal acceleration: < 8 m/s² (hard braking limit)

Combined score: `score = 1 - 0.6·endpoint - 0.5·jerk - 0.4·accel_continuity - 0.3·accel_magnitude`

These physics constraints guide the diffusion model to produce trajectories that transition smoothly across lateral and longitudinal directions, mimicking real vehicle kinematics rather than generating abrupt direction changes.

### AV2 Scenario Timing

- Total: 110 timesteps at 10Hz (11 seconds)
- Observed: t=0..49 (5 seconds)
- Future: t=50..109 (6 seconds, `n_future=60`)
- History window: t=30..49 (2 seconds, `n_history=20`)

## 8. Visualization Guide

See `docs/visualization_guide.md` for detailed interpretation of each visualization type. Quick reference:

| Visualization | File pattern | What to check |
|---------------|-------------|---------------|
| Trajectories | `scenario_X_trajectories.png` | Does the mean trajectory follow GT? |
| Endpoints | `scenario_X_endpoints.png` | Do endpoints cluster near goal? |
| Confidence ellipses | `scenario_X_ellipses.png` | Does uncertainty grow over time? |
| BEV scene | `scenario_X_scene.png` | Are map/neighbors/ped crossings rendered correctly? |
| Collision scene | `scenario_X_collision.png` | AV2 BEV with drivable areas, lanes, agents, ego+partner futures, shared goal |
| Collision heatmap | `scenario_X_collision_heatmap.png` | (N_ego×N_partner) distance matrix |
| Collision animation | `scenario_X_collision.mp4` | av2-api style BEV video, 11s, top-5 pairs + driving pair |
| BEV animation | `scenario_X_animation.gif` | Ego-centric view — does map scroll as vehicle moves? |
| Denoising | `scenario_X_denoising.png` | Does noise reduce from left to right? |
| Self-attention | `scenario_X_self_attn_layerN.png` | Is there diagonal/local structure? |
| Cross-attention | `scenario_X_cross_attn_layerN.png` | Does goal get attention at trajectory end? |
| Score heatmap | `scenario_X_score_heatmap.png` | Gradient centered on goal? |

### Common diagnostic flow

```
1. Trajectories → noise or shape?
   ├── All noise → training didn't converge, check loss
   └── Has shape → continue

2. Cross-attention → goal/map/neighbor used?
   ├── goal ignored → goal conditioning broken
   └── All uniform → model under-trained

3. Denoising → progressive noise reduction?
   ├── All same (just scaled) → DDIM step bug (fixed in this version)
   └── Progressive → healthy
```

## 9. Troubleshooting

### Self-attention is completely uniform (~0.0167)

This equals 1/60 (uniform over 60 tokens). Causes:
- **Early training (normal)**: adaLN-Zero gates start at 0, self-attention is bypassed. Gates open after ~5-10 training steps. Check that nonzero gradient count increases from 2 → 80+ over the first epoch.
- **After many epochs still uniform**: model capacity issue, or loss not decreasing. Check TensorBoard loss curve.

### Denoising visualization shows identical patterns (only axis scale changes)

This was a **DDIM step formula bug** (fixed in current version). The old formula used `noise_pred` directly instead of the direction vector `(x_t - sqrt(alpha_t) * x_0_est) / sqrt(1 - alpha_t)`. If you see this symptom, make sure you're using the updated `model/diffusion.py`.

### Cross-attention only active at token 21 (map start)

Token 21 = goal(1) + history(20) boundary. This indicates:
- Model hasn't learned semantic attention structure yet (under-trained)
- Or the map/neighbor tokens dominate attention because they have larger magnitude — ensure map position features are normalized by SCALE=50 and neighbor positions/velocities by SCALE=50/VELOCITY_SCALE=10 respectively
- Or zero-padded map/neighbor tokens produce phantom embeddings that attract attention — ensure padding mask (`key_padding_mask=True` for invalid tokens) is passed through ConditionEncoder self-attention and AdaLNBlock cross-attention

### Loss not decreasing

- Check `data.train_dir` points to valid data with enough scenarios (>50 minimum, >1000 recommended)
- Increase `training.epochs` (10 is few for this model; use 50-200 for real training)
- Reduce `training.batch_size` if GPU OOM
- Check `lane_feat_dim` consistency: config should match model input (currently 46)

### EMA model produces noisy trajectories despite training loss decreasing

The EMA model lags behind the training model. Its convergence speed depends on `ema_decay`:
- `0.9995` → half-life ~1386 steps (needs 100K+ steps to fully converge)
- `0.99` → half-life ~69 steps (converges within ~300 steps)

With short training runs (e.g., 500 steps / 15 epochs), `ema_decay=0.9995` leaves the EMA model at ~80% random-init weights. Use `ema_decay: 0.99` for short runs. Training-time sample visualization uses the raw training model (not EMA) so TensorBoard reflects actual progress.

### GPU out of memory

```yaml
# In config/default.yaml, reduce:
training:
  batch_size: 16        # from 32
model:
  dim: 128              # from 256
  n_layers: 4           # from 6
  ffn_dim: 512          # from 1024
```

### map_id is None in AV2 scenarios (HuggingFace data)

Some AV2 datasets from HuggingFace have `scenario.map_id = None`. The dataset loader handles this via flat-layout glob fallback: it searches for `log_map_archive_{scenario_file_id}*.json` in the map directory. No manual intervention needed.

## 10. Architecture Summary

```
Input:
  noisy_residual (B, 60, 2) + timestep t
  Conditions: goal (B, 2), history (B, 20, 2), map (B, 24, 46), neighbors (B, 6, 21, 6)

Processing:
  1. TimestepEmbedder(t) → t_emb (B, 256)
  2. traj_embed(noisy_residual) + sin_pos + pos_embed → traj_tokens (B, 60, 256)
  3. ConditionEncoder(goal, history, map, neighbors) → cond_tokens (B, 171, 256), cond_mask (B, 171)
     - goal: 1 token, history: 20 tokens, map: 24 tokens, neighbors: 126 tokens
     - 2 layers of condition self-attention (with padding mask) for inter-condition interaction
     - All position features normalized by SCALE=50, velocities by VELOCITY_SCALE=10, headings by HEADING_SCALE=π
     - Padding mask (True=ignore) excludes zero-padded map/neighbor tokens from self-attention
  4. 6 AdaLNBlocks: traj self-attend + cross-attend to conditions + FFN
     - adaLN-Zero: timestep modulates γ/β/α for norm, attention, FFN
     - Zero-init: blocks start as identity, gradually activate
     - Cross-attention uses cond_mask to exclude padding condition tokens
  5. LayerNorm + output_head → noise_pred (B, 60, 2)

Output: predicted noise eps (same shape as input residual)
Loss: (T-1)/T * MSE(eps_pred[:-1], eps[:-1]) + lambda_ep/T * MSE(eps_pred[-1], eps[-1])
  lambda_ep decays from lambda_endpoint to lambda_endpoint_end over training

Sampling:
  x_T = randn → DDIM 100 steps → residual_pred
  trajectory = prior + denormalize_residual(residual_pred)
  Optional: CFG (weight=2.0) + DPS gradient guidance
```

## 11. Config Reference

Full config in `config/default.yaml`:

```yaml
model:
  dim: 256           # Transformer hidden dim
  n_heads: 4         # Attention heads (dim must be divisible by n_heads)
  n_layers: 6        # Transformer blocks
  ffn_dim: 1024      # FFN intermediate dim
  dropout: 0.0       # Dropout rate

data:
  train_dir: av2_dataset_1k/train/     # Training data directory
  val_dir: av2_dataset_1k/train/       # Validation data directory
  map_dir: null              # Map directory (null = auto-detect: official layout or flat layout)
  n_future: 60               # Future trajectory length (6s @ 10Hz)
  n_history: 20              # History length (2s @ 10Hz)
  n_lanes: 24                # Max lane segments per scenario
  lane_feat_dim: 46          # Map token feature dimension
  n_neighbors: 6             # Max neighbor agents
  neighbor_feat_dim: 6       # Neighbor per-frame feature dimension
  velocity_scale: 10.0      # m/s — velocity normalization for neighbor features
  heading_scale: 3.14159265 # radians — heading normalization (range [-pi, pi] → [-1, 1])
  scale: 50.0               # meters — coordinate normalization
  prior_type: hermite        # "hermite" (cubic spline), "lane" (centerline path), or "linear" (straight line)
  residual_frame: chord        # "chord" (lon/lat decomposition) or "xy" (isotropic, legacy)
  residual_scale_lon: 2.5    # meters — longitudinal residual normalization (chord frame)
  residual_scale_lat: 1.0    # meters — lateral residual normalization (chord frame)
  chord_epsilon: 0.1         # meters — chord length fallback threshold (below → isotropic)
  dt: 0.1                   # seconds — AV2 10Hz timestep

diffusion:
  n_steps: 1000              # Diffusion timesteps
  inference_steps: 100        # DDIM inference steps
  inference_spacing: quadratic # "linear" or "quadratic" (more steps at high-t where alpha_bar changes fast)
  schedule: cosine           # Noise schedule (cosine recommended)

training:
  epochs: 300                 # Training epochs
  batch_size: 128             # Batch size (per GPU)
  lr: 6.0e-4                 # Learning rate (linear scaled for global batch 1024)
  weight_decay: 0.01         # AdamW weight decay (base)
  output_head_weight_decay: 0.1 # Higher weight decay for output head
  cross_attn_lr_mult: 10.0     # LR multiplier for adaLN gate params (overcome zero-init bottleneck)
  ema_decay: 0.9999              # EMA model decay (half-life ~6931 steps)
  warmup_steps: 500              # Linear warmup steps
  use_amp: true                # Mixed precision (bf16 on RTX 4090)
  drop_goal_p: 0.1            # CFG: goal dropout probability
  drop_map_p: 0.1             # CFG: map dropout probability
  drop_neighbor_p: 0.1        # CFG: neighbor dropout probability
  lambda_endpoint: 5.0       # starting weight (decays to lambda_endpoint_end)
  lambda_endpoint_end: 3.0   # final endpoint weight
  auxiliary_losses:
    enabled: true             # Enable auxiliary losses
    interval: 4               # Compute every N steps
    min_t: 0                  # Only compute when t >= this
    max_t: 800                # Only compute when t <= this (80% of schedule)
    grad_clip: 1.0            # Max aux gradient norm
    lambda_smooth: 0.2        # Smoothness loss weight (strengthened)
    lambda_vel: 0.1           # Velocity consistency loss weight
    lambda_endpoint: 0.1      # Endpoint consistency loss weight
    lambda_curv: 0.005        # Curvature regularization weight

inference:
  n_samples: 20              # Number of trajectory samples per scenario
  cfg_weight: 2.0            # Classifier-free guidance weight
  dps_eta: 0.1               # DPS guidance strength (0 = disabled)
  dynamic_threshold: 1.5     # Imagen-style x0 clipping (0=disabled, >0=enabled)
  smoothing: false            # Apply Savitzky-Golay smoothing
  smoothing_window: 7         # smoothing window length (odd)
  smoothing_polyorder: 3       # smoothing polynomial order
  goal_sigma_lon: 0.0         # meters — 0=auto: max(chord*0.10, 0.5), scales with trajectory length
  goal_sigma_lat: 0.0         # meters — 0=auto: max(chord*0.04, 0.3), scales with trajectory length

logging:
  grad_norm_interval: 200      # Log gradient norms every N steps
  condition_stats_interval: 1000  # Log condition statistics every N steps
  param_norm_interval: 1000     # Log parameter norms every N steps
  ema_compare_interval: 1      # Compare EMA vs raw model every N epochs

visualization:
  train_sample_interval: 25       # Generate sample trajectory every N epochs
  train_noise_interval: 1000      # Log noise comparison every N steps
  train_residual_interval: 1000   # Log residual distribution every N steps
  inference_sample_steps: 10     # DDIM steps for training-time sampling
  viz_scenarios: 3               # Number of scenarios to visualize during inference (0=none)
```
