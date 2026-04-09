# Worked Example — `moth_biotrove` Pipeline Validation

Demonstrates the Cosmos Transfer 2.5 pre-training validation pipeline on a
real dataset, highlighting why **understanding your data and control signals
before training** prevents wasted GPU hours and poor results.

> **Dataset:** [`pjramg/moth_biotrove`](https://huggingface.co/datasets/pjramg/moth_biotrove)
> — 1 000 moth species samples from BioTrove, loaded via FiftyOne.

---

## What This Example Validates

| Pipeline Stage | What Was Tested | Outcome |
|---|---|---|
| **Stage 1 — Format Validation** | FPS, resolution, frame count (×93), container format | All clips resampled to 16 FPS @ 1280×720, trimmed to 93 frames |
| **Stage 2 — Caption Generation** | Taxonomy-aware captioning from dataset metadata | Captions auto-built from scientific name, family, genus — 45–46 words avg |
| **Stage 3 — Control Signal Inspection** | Edge / blur / depth / segmentation quality | HTML report reveals only **1 of 4** control types present per clip |

---

## The HTML Control Report

Open the report in any browser — it is fully self-contained (all images
base64-encoded, no external dependencies):

**[`biotrove_85278_control_report.html`](biotrove_85278_control_report.html)**

### What you will find inside

| Section | What It Shows | Why It Matters |
|---|---|---|
| **Inventory Table** | Per-signal check: directory exists, file present, size, thumbnail | Immediately see that only 1/4 signals are available — training with a missing signal silently degrades output |
| **Original Frame** | Source frame at 1280×720 | Baseline reference for all comparisons |
| **Canny Edge (5 presets)** | Very Loose (20/50) → Very Tight (100/300) | Loose thresholds pick up texture noise, shadows, and background clutter from uncontrolled field conditions; tight thresholds lose fine wing patterns. **Edge alone is unreliable for this domain** |
| **Gaussian Blur (5 presets)** | σ=3 (light) → σ=15 (max) | Shows how aggressively blur erases subject detail — helps choose supplementary weight |
| **Depth — DepthAnything V2** | INFERNO colormap (dark=near, bright=far) | Flat subjects (pinned specimens) produce near-uniform depth maps — **depth is a poor primary signal for this domain** |
| **Segmentation — SAM 2.1** | Automatic mask generation, color-coded regions | Cleanly isolates specimen from background despite texture and shadow noise — **strong primary signal for biodiversity data** |

### Key Insight

For a biodiversity / specimen dataset like moth_biotrove, the report makes the
control signal decision concrete:

- **Edge (Canny) alone is unreliable** — texture detail, shadows, and uncontrolled field conditions produce noisy, inconsistent edge maps across samples
- **Segmentation** is the strongest signal — cleanly separates the specimen from cluttered backgrounds regardless of lighting or texture variation
- **Edge + Segmentation combined** is the recommended approach — segmentation provides robust semantic boundaries while edge adds structural detail within regions
- **Depth** is nearly useless — flat specimens have no meaningful depth variation
- **Blur** works only as a light supplementary signal

Without this visual inspection, a user might default to `edge` alone (the
easiest auto-generated signal) and waste a full training run before discovering
it picks up too much noise from uncontrolled backgrounds in their domain.

---

## Reproducing This Example

```bash
# 1. Load dataset
python -c "
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub
dataset = load_from_hub('pjramg/moth_biotrove')
session = fo.launch_app(dataset)
"

# 2. Generate HTML control report for a sample clip
python scripts/visualize_controls.py \
  --input moth_biotrove/videos/biotrove_85278.mp4 \
  --depth_model vitb

# 3. Open in browser
xdg-open biotrove_85278_control_report.html   # Linux
# open biotrove_85278_control_report.html      # macOS
```

For the full captioning pipeline and training setup, see the
[main Transfer 2.5 toolkit guide](../../README.md).

---

## Hardware Used

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX PRO 5000 (Blackwell) |
| CUDA | 13.1 |
| PyTorch | 2.11.0+cu128 |
| Depth Model | DepthAnything V2 ViT-B |
| Segmentation | SAM 2.1 (automatic mask generation) |
