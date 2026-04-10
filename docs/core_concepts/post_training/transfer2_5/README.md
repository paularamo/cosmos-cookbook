# Cosmos Transfer 2.5 — Post-Training Toolkit

Fine-tune NVIDIA Cosmos Transfer 2.5 on your own video data to bridge the gap between synthetic and real-world domains, adapt to new environments, or transfer visual style while preserving scene structure.

> **Repo:** `github.com/nvidia-cosmos/cosmos-transfer2.5`

---

## Why This Matters

Training generative video models from scratch requires millions of clips and
weeks of compute. Cosmos Transfer 2.5 sidesteps that by giving you a
pre-trained world model that already understands physics, lighting, and motion —
you only need to teach it your domain.

```
┌─────────────────────────────────────────────────────────────────┐
│                    THE DOMAIN GAP PROBLEM                       │
│                                                                 │
│  Synthetic / Source domain        Real / Target domain          │
│  ┌──────────────────────┐        ┌──────────────────────┐       │
│  │  CARLA render        │        │  Dashcam footage     │       │
│  │  Isaac Sim           │   ──►  │  Field robots        │       │
│  │  Lab conditions      │   ???  │  Production env.     │       │
│  │  Controlled images   │        │  Wild specimens      │       │
│  └──────────────────────┘        └──────────────────────┘       │
│                                                                 │
│  Direct training: needs 100k+ paired clips                      │
│  Cosmos Transfer post-training: works with ~50–500 clips.⚠       |
|  Note that this number can vary depending of the adaptation     |
|  or domain shift or your problem.                               │
└─────────────────────────────────────────────────────────────────┘
```

### What Post-Training Unlocks

| Use Case | Before | After |
|---|---|---|
| Sim2Real (robotics) | Sim-trained models fail on real hardware | Smooth domain transfer with edge + seg control |
| AV data augmentation | Limited real-world edge cases | Generate photorealistic weather/lighting variants |
| Species documentation | Manual photography at scale | Transfer appearance from reference images |
| Industrial inspection | Synthetic defect renders | Realistic defect appearance on real parts |

---

## How It Works

Post-training teaches the model to follow **control signals** — structured
representations of scene geometry, edges, or depth — while generating output
in your target domain.

```
                        POST-TRAINING
                        ┌───────────┐
  Source clip ─────────►│           │
                        │  Cosmos   │──────► Generated clip
  Control signal ───────►  Transfer │         (target domain)
  (edge / depth /       │   2.5     │
   seg / blur)          └───────────┘
                              ▲
                        Your dataset
                        (50–500 clips)⚠
```

### Control Signal Guide

```
What do you want to transfer?
│
├─ Texture / appearance only, keep shapes?
│     └──► edge (Canny)  weight 0.7–1.0        ← auto-generated ✓
│
├─ Lighting / weather, keep 3D depth?
│     └──► depth  weight 0.8–1.0               ← DepthAnything V2
│          + edge  weight 0.2–0.4  [optional]
│
├─ Replace semantic regions (sky, road, foliage)?
│     └──► seg  weight 0.5–0.7                 ← SAM 2.1
│          + edge  weight 0.3–0.5
│          ⚠ to avoid hallucinations use seg with another control signal
│
├─ Soft smoothing / minor appearance shift?
│     └──► blur  weight 0.4–0.6  [supplementary only]  ← auto-generated ✓
│
├─ Sim2Real  (CARLA / Isaac → real)?
│     └──► edge (0.6) + seg (0.4)
│          or edge (0.7) + depth (0.3)
│
├─ AV dashcam domain adaptation?
│     └──► depth (0.6) + edge (0.4)
│
└─ Robotics manipulation transfer?
      └──► depth (0.4) + seg (0.3) + edge (0.3)
```

---

## Pipeline Overview

```
┌──────────────────────────────────────────────────────────────┐
│                   8-STAGE PIPELINE                           │
│                                                              │
│  0. ENVIRONMENT SETUP                                        │
│     uv venv · PyTorch cu128 · transformers · SAM             │
│                    │                                         │
│  1. DATA INGEST                                              │
│     Use your preference way to ingest your data              |
|     For example Hugging Face Hub                             │
│                    │                                         │
│  2. FORMAT VALIDATION                  Stage 1               │
│     FPS=16 · resolution ≥480P · frames ×93 · MP4/H.264       │
│     Real video clips only — still images not supported       │
│                    │                                         │
│  3. VLM CAPTION GENERATION             Stage 2               │
│     Cosmos Reason2 (8B/2B) per clip                          │
│     + metadata injection (weather · time · species · tags)   |
│     You can use other models as Gemini for captioning        |
│                    │                                         │
│  4. CONTROL SIGNAL GENERATION          Stage 3               │
│     edge  ── auto (OpenCV Canny)                             │
│     blur  ── auto (Gaussian)                                 │
│     depth ── DepthAnything V2 on GPU                         │
│     seg   ── SAM 2.1 automatic mask generation               │
│                    │                                         │
│  5. HTML VALIDATION REPORT                                   │
│     Per-clip visual inspection: original · edge variants ·   │
│     blur variants · depth colormap · segmentation masks      │
│     + dataset inventory table (what exists / what's missing) │
│                    │                                         │
│  6. EXPERIMENT CONFIG                  Stage 5               │
│     Python config: data path · control type · LR · iters     │
│                    │                                         │
│  7. DISTRIBUTED TRAINING               Stage 6               │
│     torchrun --nproc_per_node=N                              │
│                    │                                         │
│  8. CHECKPOINT → INFERENCE             Stage 7–8             │
│     DCP → checkpoint_ema_bfloat16.pt → inference validation  │
└──────────────────────────────────────────────────────────────┘
```

---

## Dataset Requirements

> **Videos only.** Cosmos Transfer 2.5 is a video generation and temporal
> reasoning model. Still images have no motion or inter-frame information —
> using them degrades training quality. Collect real video clips.

| Property | Requirement | Fix |
|---|---|---|
| Input type | **Real video clips only** | Re-collect as video — no image workarounds |
| Container | MP4 (H.264) | `ffmpeg -c:v libx264` |
| FPS | **16 FPS** | `ffmpeg -vf fps=16` |
| Resolution | ≥ 480P (720P recommended) | `ffmpeg -vf scale=1280:720` |
| Frame count | **Multiples of 93** (93, 186, 279…) | Trim to nearest ×93 |
| Caption | JSON `{"prompt": "...", "negative_prompt": "..."}` | Cosmos Reason2 pipeline |
| Caption length | ≤ 300 words | Auto-validated |
| Minimum clips | ~50 for fine-tuning | More = better generalization |

### Expected Directory Structure

```
dataset_root/
├── videos/              ← 16 FPS · 720P · ×93 frames
│   ├── clip_001.mp4
│   └── clip_002.mp4
├── captions/            ← one JSON per clip
│   ├── clip_001.json    {"prompt": "...", "negative_prompt": "..."}
│   └── clip_002.json
├── metadata/            ← optional domain metadata per clip
│   └── clip_001.json    {"weather": "sunny", "hour_of_day": "14:30", ...}
├── control_edge/        ← auto-generated or provided
├── control_blur/        ← auto-generated or provided
├── control_depth/       ← DepthAnything V2 output
├── control_seg/         ← SAM 2.1 output
└── colormap.json        ← optional: color → class name mapping
```

---

## Quick Start

### 1. Environment Setup

```bash
# Clone the repo
git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git
cd cosmos-transfer2.5

# Create venv and install (CUDA 12.8 — works on CUDA 13.x too)
uv venv .venv --python 3.12 && source .venv/bin/activate
uv sync --extra=cu128

# Extra deps: data validation + control signal scripts
uv pip install opencv-python numpy Pillow \
               transformers accelerate huggingface_hub \
               "git+https://github.com/facebookresearch/sam2.git"

# HuggingFace auth (required for NVIDIA model weights)
huggingface-cli login
```

See [`references/environment-setup.md`](references/environment-setup.md) for the full guide including troubleshooting.

### 2. Prepare Your Dataset

Organize your video clips into the expected directory structure (see below).
You can use any data management approach that fits your workflow:

```bash
# Example: download from HuggingFace Hub
pip install huggingface_hub
huggingface-cli download your-org/your-dataset --local-dir dataset_root/

# Example: copy from local storage
cp -r /path/to/your/clips/ dataset_root/videos/

# Example: use a dataset management tool (FiftyOne, DVC, etc.)
```

### 3. Validate Data

```bash
# Check FPS, resolution, frame count across all clips
python scripts/check_control.py \
  --videos_dir dataset_root/videos/ \
  --control_dir dataset_root/control_edge/ \
  --control_type edge \
  --auto_generate --validate_alignment
```

### 4. Generate & Inspect Control Signals

```bash
# Generate HTML validation report for visual inspection
python scripts/visualize_controls.py \
  --input dataset_root/videos/clip_001.mp4 \
  --depth_model vitb

# Open the self-contained report in any browser
open clip_001_control_report.html
```

The HTML report includes:

| Panel | Contents |
|---|---|
| Inventory table | Which control signals exist for this clip (with thumbnails) |
| Original | Source frame at full resolution |
| Canny Edge | 5 threshold presets: Very Loose (20/50) → Very Tight (100/300) |
| Gaussian Blur | 5 sigma presets: σ=3 (light) → σ=15 (max) |
| Depth | DepthAnything V2 — INFERNO colormap (dark=near, bright=far) |
| Segmentation | SAM 2.1 automatic masks — each region color-coded |

### 5. Generate Captions

```bash
python scripts/generate_captions.py \
  --videos_dir    dataset_root/videos/ \
  --metadata_dir  dataset_root/metadata/ \
  --seg_dir       dataset_root/control_seg/ \
  --output_dir    dataset_root/captions/ \
  --model         nvidia/Cosmos-Reason2-8B \
  --device        cuda
```

### 6. Configure & Launch Training

```bash
# Copy and edit the example experiment config
cp cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/experiments/\
transfer2_post_train_example.py \
cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/experiments/\
my_experiment.py

# Launch distributed training
export EXP=my_experiment NUM_GPUS=8
IMAGINAIRE_OUTPUT_ROOT=/large/storage \
torchrun --nproc_per_node=$NUM_GPUS --master_port=12341 \
  -m scripts.train \
  --config=cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py \
  -- experiment=${EXP} job.wandb_mode=disabled
```

### 7. Convert Checkpoint & Run Inference

```bash
# Convert DCP → PyTorch .pt
python scripts/convert_distcp_to_pt.py ${CKPT_DIR}/model ${CKPT_DIR}

# Inference
python examples/inference.py \
  --params_file my_spec.json \
  --checkpoint_path ${CKPT_DIR}/checkpoint_ema_bfloat16.pt \
  -o outputs/validation/
```

---

## Tested Example — `moth_biotrove`

Pipeline validation run on [`pjramg/moth_biotrove`](https://huggingface.co/datasets/pjramg/moth_biotrove)
(1 000 moth species samples) to exercise Stages 1–4 of the toolkit.

> **Note:** This dataset contains still images, not videos. It was used to
> validate the pipeline scripts (captioning, control signal generation, HTML
> report) only. For real training, replace with actual video clips of moth
> behaviour. Still images are **not recommended** for Cosmos Transfer training.

| Step | Result |
|---|---|
| Captions | Auto-built from taxonomy labels (scientific name, family, genus) — 45–46 words ✓ |
| Edge control | Auto-generated via OpenCV Canny, alignment validated ✓ |
| HTML report | Depth (DepthAnything V2 vitb) + SAM 2.1 automatic masks per clip ✓ |
| Final checklist | All Stage 1–4 checks passed ✓ |

**The HTML control report is the most actionable output** — it reveals that
edge (Canny) alone is unreliable for specimens with complex textures, shadows,
and uncontrolled field conditions, and that **edge + segmentation** is the right
combination for this domain. See the full walkthrough and report:

→ **[`examples/moth_biotrove/`](examples/moth_biotrove/README.md)** — worked example with interactive HTML report

---

## Reference Docs

| File | Contents |
|---|---|
| [`references/environment-setup.md`](references/environment-setup.md) | Hardware requirements, uv/conda setup, verified package versions, troubleshooting |
| [`references/data-validation.md`](references/data-validation.md) | Full 4-stage validation pipeline, HTML report guide, Canny/blur threshold tables |
| [`references/control-signals.md`](references/control-signals.md) | Control signal decision tree, weight tuning, generation commands |
| [`references/caption-generation.md`](references/caption-generation.md) | Cosmos Reason2 setup, metadata injection, colormap object extraction |
| [`references/training-config.md`](references/training-config.md) | Config parameter reference, learning rate guide, iteration counts |
| [`references/checkpoints.md`](references/checkpoints.md) | DCP vs PyTorch formats, conversion, resumption, size reference |

---

## Hardware Requirements

| Scenario | GPU | VRAM |
|---|---|---|
| Data validation + HTML report | Any CUDA GPU | 4 GB+ |
| Caption generation (Cosmos Reason2-2B) | A100 / H100 | 24 GB |
| Caption generation (Cosmos Reason2-8B) | H100 | 32 GB |
| Training (single GPU) | A100 40 GB | 40 GB |
| Training (recommended) | 8× H100 80 GB | 640 GB total |
| Inference | A100 / H100 | 24–80 GB |
