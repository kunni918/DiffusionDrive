# DiffusionDrive algorithm + code map

This note links the paper’s truncated diffusion design to the training recipe in `docs/train_eval.md` and the concrete implementation in this repo.

## High-level pipeline (paper → code)
- Perception: TransFuser-style fusion of front cameras + LiDAR BEV to build tokens and BEV feature maps. Implemented in `navsim/agents/diffusiondrive/transfuser_model_v2.py:24-144` (backbone call at lines 108-114, token packing at 116-130).
- Planning: A truncated diffusion model predicts ego trajectories conditioned on anchors, BEV features, detected agents, and ego status. Implemented in the TrajectoryHead stack (`transfuser_model_v2.py:390-568`).
- Multi-task outputs: BEV semantic map, agent boxes/labels, and multi-modal ego plan (lines 132-142).
- Training cmd (from `docs/train_eval.md`): `run_training.py agent=diffusiondrive_agent ...` drives the above model; key assets are the ResNet-34 backbone weights and `kmeans_navsim_traj_20.npy` anchors (see TransfuserConfig paths).

## Training recipe (docs/train_eval.md)
- Dataset caching for speed: `run_dataset_caching.py` and `run_metric_caching.py` (train_eval.md section 1).
- Main training command (section 2) launches Hydra config `agent=diffusiondrive_agent`, sets epochs=100, and points to cached data. Ensure `bkb_path` and `plan_anchor_path` are set in `navsim/agents/diffusiondrive/transfuser_config.py`.
- Evaluation uses `run_pdm_score.py` with a checkpoint path (section 3).

## Model components and data flow
- Config + anchors: `transfuser_config.py:9-112` holds anchor path, lidar extents, LR settings, and loss weights. Anchors are loaded in `TrajectoryHead.__init__` (`transfuser_model_v2.py:408-430`).
- Tokenization:
  - BEV feature map from backbone → downscaled and flattened tokens (`transfuser_model_v2.py:108-129`).
  - Ego status vector encoded to a token and concatenated (`transfuser_model_v2.py:114-118`).
  - Query embeddings split into trajectory and agent heads (`transfuser_model_v2.py:132-138`).
- Cross-BEV lookup (lightweight attention):
  - `GridSampleCrossBEVAttention` samples BEV features at predicted waypoints instead of dense attention (`modules/blocks.py:42-109`), normalizing waypoints using lidar bounds from config.
- Diffusion trajectory head:
  - Anchors → sinusoidal positional encodings → MLP to get trajectory tokens (`transfuser_model_v2.py:487-492`).
  - Diffusion timestep embedding via `time_mlp` (`transfuser_model_v2.py:425-430`) and injected by FiLM modulation (`ModulationLayer`, `transfuser_model_v2.py:235-274`).
  - Stacked custom decoder layers (2 by default) perform:
    1) Cross BEV sampling,
    2) Cross-attention with detected agents,
    3) Cross-attention with ego query,
    4) FFN + time modulation,
    5) Residual prediction of offsets/headings (`transfuser_model_v2.py:276-351`).
  - Multi-layer refinement feeds the newly denoised coordinates to the next layer to mimic iterative diffusion (`transfuser_model_v2.py:357-388`).
- Denoising schedule (truncation from the paper):
  - Training: noise levels sampled only from first 50 DDIM steps, then clipped and denormalized (`transfuser_model_v2.py:467-499`).
  - Inference: start from partially noisy anchors at t=8, run only 2 reverse steps (`transfuser_model_v2.py:513-565`), yielding ~10× speedup vs full diffusion.
- Output selection:
  - Each decoder layer predicts 20 modes × 8 poses; losses summed across layers (`transfuser_model_v2.py:499-507`).
  - Mode chosen by max classification logit; regression picked with `torch.gather` (`transfuser_model_v2.py:508-511` and inference at 564-567).

## Losses and optimization
- Trajectory loss:
  - Anchors matched to GT by closest-average L2; focal loss on mode classification + L1 on matched trajectory (`modules/multimodal_loss.py:117-164`).
- Agent head: simple MLP for boxes + objectness (`transfuser_model_v2.py:145-185`).
- BEV semantic head: conv + upsample branch (`transfuser_model_v2.py:47-70`).
- LR schedule: warmup + cosine decay utility in `modules/scheduler.py:6-56` (Hydra can swap schedulers via config; this mirrors paper’s fast/steady training).

## Diffusion UNet building block (for completeness)
- `ConditionalUnet1D` (`modules/conditional_unet1d.py:9-289`) defines a 1D denoiser with timestep embeddings, optional global/local conditioning, and FiLM residual blocks. It mirrors the standard diffusion-policy UNet; the trajectory head instead uses transformer-style refinement but reuses the same sinusoidal/time-conditioning patterns.

## How the training command exercises the code
- `run_training.py agent=diffusiondrive_agent` instantiates `V2TransfuserModel` with `TransfuserConfig` (backbone, anchors, loss weights).
- Backprop signals:
  - BEV segmentation loss (if enabled) from `bev_semantic_map`.
  - Agent detection loss from `agent_states/agent_labels`.
  - Trajectory diffusion loss from `trajectory_loss` dict (sums over decoder layers).
- Caching paths set in the command let loaders serve preprocessed sensor+label batches to `TransfuserBackbone` and trajectory head without extra I/O, matching the paper’s runtime claims.

## Quick reference of key files
- Model: `navsim/agents/diffusiondrive/transfuser_model_v2.py`
- Diffusion blocks/utilities: `navsim/agents/diffusiondrive/modules/blocks.py`, `conditional_unet1d.py`
- Loss: `navsim/agents/diffusiondrive/modules/multimodal_loss.py`
- Scheduler: `navsim/agents/diffusiondrive/modules/scheduler.py`
- Config: `navsim/agents/diffusiondrive/transfuser_config.py`
- Train/Eval commands: `docs/train_eval.md`
