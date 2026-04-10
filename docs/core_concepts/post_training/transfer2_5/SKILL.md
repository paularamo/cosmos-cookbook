---
name: cosmos-transfer-postrain
description: Use this skill when the user wants to post-train, fine-tune, or run training on NVIDIA Cosmos Transfer 2.5 models. Triggers on phrases like "post-train cosmos", "fine-tune cosmos transfer", "cosmos transfer training", "cosmos 2.5 posttraining", "train on my data with cosmos", "run cosmos transfer", "cosmos control signal", "check my data for cosmos", "prepare dataset cosmos", "cosmos checkpoint", "cosmos FPS check", "cosmos captions". Covers the full pipeline: data validation, control signal selection, experiment configuration, training launch, checkpoint management, and inference validation.
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, AskUserQuestion
argument-hint: "[validate-data|select-control|configure|train|convert-checkpoint|infer] [path]"
---

<!--
Token Budget:
- Level 1 (YAML): ~120 tokens
- Level 2 (This file): ~2000 tokens (target <2200)
- Level 3 (references/): Loaded on demand
-->

# Cosmos Transfer 2.5 — Post-Training Skill

End-to-end skill for post-training NVIDIA Cosmos Transfer 2.5 on custom video data. Covers data validation, control signal selection, experiment configuration, distributed training, checkpoint conversion, and inference validation.

**Repo:** `github.com/nvidia-cosmos/cosmos-transfer2.5`

---

## When to Use

- User wants to fine-tune or adapt Cosmos Transfer 2.5 to a new domain (sim2real, dashcam, robotics, medical, etc.)
- User needs to validate video dataset format before training
- User needs guidance on which control signal (edge/depth/seg/blur) fits their task
- User needs to launch distributed training and manage checkpoints
- User wants to run inference with a post-trained checkpoint

---

## Pipeline Overview

```
1. ENVIRONMENT SETUP       → validate GPU, install deps, set env vars
2. DATA VALIDATION         → FPS, resolution, duration, caption format
3. CONTROL SIGNAL CHOICE   → edge / depth / seg / blur (or combination)
4. CONTROL VIDEO GENERATION→ auto-generate or provide control MP4s
5. EXPERIMENT CONFIG       → write/edit Python config file
6. TRAINING LAUNCH         → torchrun distributed launch
7. CHECKPOINT MANAGEMENT   → monitor, convert DCP → .pt
8. INFERENCE VALIDATION    → run inference with post-trained model
```

---

## Quick Start

### 1. Environment Setup
See **[references/environment-setup.md](references/environment-setup.md)** for the full guide.

```bash
# Clone
git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git && cd cosmos-transfer2.5

# Install (uv, CUDA 12.8)
uv sync --extra=cu128 && source .venv/bin/activate

# Extra deps: DepthAnything V2, SAM, OpenCV
pip install transformers accelerate opencv-python numpy Pillow \
            git+https://github.com/facebookresearch/sam2.git

# Env vars (add to ~/.bashrc)
export HF_HOME=/path/to/large/storage/hf_cache
export IMAGINAIRE_OUTPUT_ROOT=/path/to/large/storage/output

# HuggingFace auth (required for NVIDIA model weights)
huggingface-cli login
```

### 2. Validate Your Data
See [references/data-validation.md](references/data-validation.md) for full checks.

> **⚠️ Videos only.** Cosmos Transfer 2.5 is a video generation model — it relies
> on temporal motion and inter-frame reasoning. Still images have no temporal
> information and will degrade training quality. Always use real video clips.

```bash
# Quick FPS probe on a sample video
ffprobe -v error -select_streams v:0 \
  -show_entries stream=r_frame_rate,nb_frames,duration \
  -of default=noprint_wrappers=1 your_video.mp4
```

**Hard requirements:**
| Property | Requirement |
|---|---|
| Container | MP4 (H.264) |
| Input type | **Real video clips only — no still images** |
| FPS | 16 FPS (resample if different) |
| Resolution | 1280×720 recommended; ≥480P minimum |
| Frame count | Multiples of 93 (93, 186, 279…) |
| Caption length | ≤ 300 words per clip |
| Caption format | JSON: `{"prompt": "...", "negative_prompt": "..."}` |

**Resample video to 16 FPS:**
```bash
ffmpeg -i input.mp4 -vf fps=16 -c:v libx264 -crf 18 output_16fps.mp4
```

**Pad/trim to nearest multiple of 93 frames:**
```bash
# Check frame count
ffprobe -v error -select_streams v:0 -count_frames \
  -show_entries stream=nb_read_frames -of csv=p=0 video.mp4
```

### 3. Select Control Signal
See [references/control-signals.md](references/control-signals.md) for decision tree.

| Signal | Use When | Auto-Generated |
|---|---|---|
| **edge** (Canny) | Texture/lighting transfer, preserve shapes | Yes |
| **depth** | Maintain 3D spatial consistency | No — must supply |
| **seg** | Semantic replacement / scene re-composition | No — must supply |
| **blur** (vis) | Subtle smoothing, supplementary only | Yes |

**Rule of thumb:**
- Sim2Real (CARLA→real): `edge` + optional `seg`
- Dashcam/AV domain: `depth` + `edge`
- Robotics manipulation: `depth` + `seg` + `edge`
- Style transfer only: `edge` alone

### 4. Dataset Directory Structure
```
dataset_root/
├── videos/
│   ├── clip_001.mp4          # 16 FPS, 720P, 93-frame multiples
│   └── clip_002.mp4
├── control_edge/             # or control_depth / control_seg
│   ├── clip_001.mp4          # Same dims & duration as source
│   └── clip_002.mp4
└── captions/
    ├── clip_001.json         # {"prompt": "...", "negative_prompt": "..."}
    └── clip_002.json
```

### 5. Configure Experiment

Copy and edit the example config:
```bash
cp cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/experiments/\
transfer2_post_train_example.py \
cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/experiments/\
my_experiment.py
```

Key fields to set in config (see [references/training-config.md](references/training-config.md)):
- `data.train_dataset.data_path`: path to `dataset_root/`
- `data.train_dataset.control_type`: `"edge"` / `"depth"` / `"seg"` / `"blur"`
- `training.max_iter`: total iterations
- `training.save_iter`: checkpoint frequency
- `training.learning_rate`

### 6. Launch Training
```bash
export EXP=my_experiment   # must match filename without .py
export NUM_GPUS=8

IMAGINAIRE_OUTPUT_ROOT=/large/storage \
torchrun --nproc_per_node=$NUM_GPUS --master_port=12341 \
  -m scripts.train \
  --config=cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py \
  -- experiment=${EXP} job.wandb_mode=disabled
```

Checkpoints saved to:
`${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer_v2p5/.../checkpoints/`

### 7. Convert Checkpoint (DCP → PyTorch)
```bash
python scripts/convert_distcp_to_pt.py \
  ${CKPT_DIR}/model \
  ${CKPT_DIR}
# Outputs: checkpoint_ema_bfloat16.pt (use this for inference)
```

See [references/checkpoints.md](references/checkpoints.md) for monitoring and resumption.

### 8. Run Inference with Post-Trained Model
```bash
python examples/inference.py \
  --params_file my_spec.json \
  --checkpoint_path ${CKPT_DIR}/checkpoint_ema_bfloat16.pt \
  -o outputs/validation/
```

---

## Common Workflows

1. **Sim2Real Augmentation** — [prompts/sim2real.md](prompts/sim2real.md)
2. **AV Dashcam Domain Adaptation** — [prompts/dashcam.md](prompts/dashcam.md)
3. **Robotics Manipulation Transfer** — configure depth+seg+edge multicontrol
4. **BioTrove-Balanced (Biodiversity)** — see **[../biotrove-balanced/SKILL.md](../biotrove-balanced/SKILL.md)**
   - Dataset: `BGLab/BioTrove-Train` → `BioTrove-benchmark/BioTrove-Balanced.parquet`
   - Prep: 93-frame MP4 @ 10 FPS + taxonomy-aware captions + edge + seg control
   - Config: `control_type="edge+seg"`, `learning_rate=1e-5`, `max_iter=2000`
   - **Note:** Edge alone is unreliable for biodiversity data — texture, shadows, and uncontrolled field conditions produce noisy edges. Use segmentation as primary + edge as supplementary.

---

## Safety / Cost Rules

- Always run `--dry-run` or validate a 1-clip test before full dataset training
- Multi-node runs require `--nnodes` and `--rdzv_backend=c10d` (ask user to confirm cluster config)
- Checkpoint conversion overwrites directory — confirm path before running
- Do NOT set `IMAGINAIRE_OUTPUT_ROOT=/tmp` (default) for real training runs

---

## References

- **[references/environment-setup.md](references/environment-setup.md)** — Hardware requirements, uv/conda setup, DepthAnything V2, SAM, env vars, HF auth, verification checklist
- **[references/data-validation.md](references/data-validation.md)** — Full data checks: FPS, resolution, captions, frame count validator script, HTML control signal report
- **[references/control-signals.md](references/control-signals.md)** — Control signal decision tree, weight tuning, generation commands
- **[references/training-config.md](references/training-config.md)** — Config parameter reference, experiment file anatomy
- **[references/checkpoints.md](references/checkpoints.md)** — Checkpoint format, monitoring, conversion, resumption
