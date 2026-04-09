# Control Signal Reference — Cosmos Transfer 2.5

Guide for selecting, generating, and weighting control signals for post-training.

---

## Control Signal Types

| Signal | Key Strength | Limitation | Auto-Generated |
|---|---|---|---|
| **edge** (Canny) | Preserves outlines, structure, layout | Loses color/texture info | Yes |
| **depth** | Maintains 3D spatial relationships | Needs depth sensor or estimator | No |
| **seg** | Semantic class-level structure | Prone to hallucination if used alone | No |
| **blur** (vis) | Soft appearance continuity | Adds temporal noise if weight > 0.6 | Yes |

---

## Decision Tree

```
What is your transfer goal?
│
├─ Change texture/appearance, keep structure?
│   └─ edge (weight 0.7–1.0)
│
├─ Change lighting/weather, keep depth realism?
│   └─ depth (weight 0.8–1.0)
│       + edge (weight 0.2–0.4) [optional]
│
├─ Replace semantic regions (sky→cloudy, road→wet)?
│   └─ seg (weight 0.5–0.7) + edge (weight 0.3–0.5)
│       NEVER use seg alone — causes hallucinations
│
├─ Smooth/soften + minor appearance shift?
│   └─ blur/vis (weight 0.4–0.6)
│       Use as supplementary; never primary
│
├─ Sim2Real (CARLA/Isaac/synthetic → real)?
│   └─ edge (0.6) + seg (0.4)
│       or edge (0.7) + depth (0.3)
│
├─ AV domain adaptation (dashcam quality)?
│   └─ depth (0.6) + edge (0.4)
│
└─ Robotics manipulation domain transfer?
    └─ depth (0.4) + seg (0.3) + edge (0.3)
```

**Weight normalization:** If weights sum > 1.0, they are automatically rescaled. Explicit normalization:
```python
weights = {"edge": 0.6, "seg": 0.4}  # already sum to 1.0 — OK
weights = {"edge": 0.7, "depth": 0.5}  # sums to 1.2 → rescaled to 0.583, 0.417
```

---

## Generating Control Videos

### Edge (Canny) — Auto-generated at inference if not provided
```bash
# If you need to pre-generate edge control videos:
python tools/generate_control.py \
  --type edge \
  --input_dir dataset_root/videos/ \
  --output_dir dataset_root/control_edge/
```

### Depth — Must be provided externally
Options for depth estimation:
```bash
# Using Depth Anything V2 (recommended open-source)
pip install depth-anything-v2
python -c "
from depth_anything_v2.dpt import DepthAnythingV2
# See depth_anything_v2 docs for video inference
"

# Using ZoeDepth
pip install zoedepth
# See ZoeDepth repo for video pipeline

# NVIDIA internal: use internal depth estimation service
```

Depth output format:
- Same MP4 container and FPS as source
- Depth encoded as grayscale (0=near, 255=far) or normalized float
- Must be pixel-aligned with source video

### Segmentation — Must be provided externally
```bash
# Using Grounded-SAM or Mask2Former for video segmentation
# Output: color-coded segmentation mask MP4

# Using NVIDIA's internal segmentation tools if available
```

Segmentation tips:
- Use masking (`mask_path`) to restrict seg to specific regions
- Avoid full-frame segmentation without masking
- Color-code semantic classes consistently across clips

### Blur (Visual) — Auto-generated at inference if not provided
```bash
# Pre-generate if needed:
ffmpeg -i source.mp4 -vf "gblur=sigma=5" -c:v libx264 control_blur.mp4
```

---

## Multi-Control JSON Spec

```json
{
  "prompt_path": "captions/clip_001.json",
  "output_dir": "outputs/",
  "video_path": "videos/clip_001.mp4",
  "guidance": 3.0,
  "num_steps": 50,
  "edge": {
    "control_path": "control_edge/clip_001.mp4",
    "control_weight": 0.6
  },
  "depth": {
    "control_path": "control_depth/clip_001.mp4",
    "control_weight": 0.4
  }
}
```

For segmentation with masking:
```json
{
  "seg": {
    "control_path": "control_seg/clip_001.mp4",
    "control_weight": 0.5,
    "mask_path": "masks/clip_001.mp4"
  }
}
```

---

## Distilled Model — Inference-Only Control

For the distilled edge model (low-latency, released Feb 2026):
```bash
python examples/inference.py \
  --params_file spec.json \
  --num_steps 4 \       # distilled = 4 steps (not 50)
  -o outputs/
```

---

## Weight Tuning Guide

| Scenario | Recommended Starting Weights |
|---|---|
| Edge-only structural transfer | edge=1.0 |
| Depth-dominant spatial | depth=0.8, edge=0.2 |
| Seg + edge balanced | seg=0.5, edge=0.5 |
| Subtle smoothing blend | vis=0.5, edge=0.5 |
| Full sim2real (3-signal) | depth=0.4, edge=0.4, seg=0.2 |
| Robotics (3-signal) | depth=0.4, seg=0.3, edge=0.3 |

Iterative tuning: adjust by ±0.1 increments; run 1-clip inference to evaluate.
