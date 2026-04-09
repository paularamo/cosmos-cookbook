# Data Validation & Preparation — Cosmos Transfer 2.5

End-to-end pipeline: video format validation → caption generation → control signal check.

> **⚠️ Videos only — no still images.**
> Cosmos Transfer 2.5 is a video generation and temporal reasoning model.
> Still images looped into MP4s carry no motion or inter-frame information,
> which degrades training quality and produces poor transfer results.
> Collect real video clips before proceeding with this pipeline.

```
Stage 1: Format validation (FPS, resolution, frame count)
Stage 2: Caption generation (Cosmos Reason → metadata injection)
Stage 3: Control signal check / auto-generation
Stage 4: Final checklist
```

---

## Stage 1 — Format Validation

### 1a. Bulk FPS & Frame Count Check

```bash
#!/usr/bin/env bash
# scripts/validate_format.sh <videos_dir>
VIDEO_DIR="${1:-videos}"
ERRORS=0

for f in "$VIDEO_DIR"/*.mp4; do
  fps=$(ffprobe -v error -select_streams v:0 \
    -show_entries stream=r_frame_rate \
    -of default=noprint_wrappers=1:nokey=1 "$f")
  fps_num=$(echo "$fps" | awk -F/ '{printf "%.2f", $1/$2}')

  frames=$(ffprobe -v error -select_streams v:0 -count_frames \
    -show_entries stream=nb_read_frames -of csv=p=0 "$f")

  if [[ "$fps_num" != "16.00" ]]; then
    echo "WARN FPS:    $f → ${fps_num} (need 16.00)"
    ERRORS=$((ERRORS+1))
  fi

  mod=$((frames % 93))
  if [[ "$mod" -ne 0 ]]; then
    nearest=$(( (frames / 93) * 93 ))
    echo "WARN FRAMES: $f → $frames frames (not ×93; trim to $nearest)"
    ERRORS=$((ERRORS+1))
  fi
done
echo "--- $ERRORS issues found in $VIDEO_DIR"
```

### 1b. Fix FPS

```bash
ffmpeg -i input.mp4 -vf fps=16 -c:v libx264 -crf 18 -preset fast output_16fps.mp4
```

### 1c. Trim to Nearest Multiple of 93

```bash
TARGET_FRAMES=93   # or 186, 279, 372
DURATION=$(echo "scale=6; $TARGET_FRAMES / 16" | bc)   # 5.8125 s
ffmpeg -i input_16fps.mp4 -t $DURATION -c copy trimmed.mp4
```

### 1d. Resolution

```bash
# Check
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height -of csv=p=0 video.mp4

# Downscale to 720P
ffmpeg -i input.mp4 -vf scale=1280:720 -c:v libx264 -crf 18 out_720p.mp4
```

---

## Stage 2 — Caption Generation Pipeline

**Formula:** `Caption = Video Description (Cosmos Reason) + Metadata Injection`

### 2a. Metadata JSON Schema

Each clip's metadata JSON (same stem as video, in `metadata/`) must contain:

```json
{
  "weather": "overcast",
  "hour_of_day": "14:30",
  "turf_variety": "bermuda_grass",
  "location": "field_A",
  "custom_tags": []
}
```

| Field | Type | Examples |
|---|---|---|
| `weather` | string | `"sunny"`, `"overcast"`, `"rain"`, `"fog"`, `"night"` |
| `hour_of_day` | string (HH:MM) | `"06:00"`, `"14:30"`, `"20:45"` |
| `turf_variety` | string | `"bermuda_grass"`, `"kentucky_bluegrass"`, `"synthetic"` |
| `location` | string | optional site/field identifier |
| `custom_tags` | list[str] | optional freeform labels |

If a field is missing it is omitted from the injected caption (not an error).

### 2b. Segmentation Mask → Object List

The seg mask is read to extract the list of object classes present in the scene.
Script automatically maps mask colors to class names using the provided colormap
(COCO by default; supply a custom `colormap.json` for domain-specific classes).

```python
# colormap.json example
{
  "0,128,0":   "grass",
  "128,64,128": "road",
  "70,70,70":  "building",
  "220,20,60": "person",
  "0,0,142":   "vehicle",
  "107,142,35": "vegetation"
}
```

### 2c. Run the Full Caption Pipeline

```bash
python scripts/generate_captions.py \
  --videos_dir    dataset_root/videos/ \
  --metadata_dir  dataset_root/metadata/ \
  --seg_dir       dataset_root/control_seg/ \
  --colormap      dataset_root/colormap.json \
  --output_dir    dataset_root/captions/ \
  --model         nvidia/Cosmos-Reason2-8B \
  --device        cuda \
  --fps           4
```

Script: **[scripts/generate_captions.py](../scripts/generate_captions.py)**

See [caption-generation.md](caption-generation.md) for Cosmos Reason setup, prompt template, and output format.

### 2d. Caption Output Format

```json
{
  "prompt": "A robotic arm moves over a bermuda grass field under overcast sky at 14:30. The scene includes: grass, vehicle, person. [Cosmos Reason description follows] The camera captures a close-up of the end effector descending toward a small object on the turf surface. Motion is steady and precise.",
  "negative_prompt": "blurry, overexposed, motion blur, synthetic render"
}
```

**Structure of injected prompt:**
```
{metadata_prefix} [Cosmos Reason video description]
```

Metadata prefix template:
```
A {turf_variety} field under {weather} conditions at {hour_of_day}.
Objects present: {object_list}.
```

### 2e. Validate Captions

```bash
#!/usr/bin/env bash
# scripts/validate_captions.sh <captions_dir> <videos_dir>
CAP_DIR="$1"; VID_DIR="$2"; ERRORS=0
for f in "$VID_DIR"/*.mp4; do
  stem=$(basename "$f" .mp4)
  cap="$CAP_DIR/${stem}.json"
  if [[ ! -f "$cap" ]]; then
    echo "MISSING: $cap"; ERRORS=$((ERRORS+1)); continue
  fi
  words=$(python3 -c "import json; d=json.load(open('$cap')); print(len(d.get('prompt','').split()))")
  if [[ "$words" -gt 300 ]]; then
    echo "LONG ($words words): $cap"; ERRORS=$((ERRORS+1))
  fi
done
echo "--- $ERRORS caption issues"
```

---

## Stage 3 — Control Signal Check / Auto-Generation

Script: **[scripts/check_control.py](../scripts/check_control.py)**

```bash
python scripts/check_control.py \
  --videos_dir   dataset_root/videos/ \
  --control_dir  dataset_root/control_edge/ \
  --control_type edge \
  --auto_generate
```

**Behavior per control type:**

| Type | Auto-Generate | How |
|---|---|---|
| `edge` | Yes | Canny filter via OpenCV on each frame |
| `blur` | Yes | Gaussian blur via OpenCV |
| `depth` | No — warns | Must supply from Depth Anything V2 or ZoeDepth |
| `seg` | No — warns | Must supply from Grounded-SAM or Mask2Former |

**Alignment validation** (always run, regardless of auto-generate):

```bash
# Manual check: source vs control must match exactly
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,nb_frames \
  -of default=noprint_wrappers=1 source.mp4

ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,nb_frames \
  -of default=noprint_wrappers=1 control.mp4
```

### 3b. HTML Control Signal Inspection Report

Before bulk-generating controls for the full dataset, visually inspect how each
signal looks on a representative frame. This catches bad thresholds, wrong
colormap directions, and model failures early.

Script: **[scripts/visualize_controls.py](../scripts/visualize_controls.py)**

```bash
# Input inside videos/ → dataset root auto-inferred, inventory table included
python scripts/visualize_controls.py --input dataset_root/videos/clip_001.mp4

# Explicit dataset root (use when input is not inside a videos/ subdir)
python scripts/visualize_controls.py --input clip_001.mp4 --dataset_root /data/myproject

# Extract frame 30 from a video
python scripts/visualize_controls.py --input dataset_root/videos/clip_001.mp4 --frame 30

# Choose DepthAnything V2 model size (vitl = best quality, vitb/vits = less VRAM)
python scripts/visualize_controls.py --input sample.jpg --depth_model vitb

# Fast local preview — skip GPU-heavy models (inventory table still shown)
python scripts/visualize_controls.py --input sample.jpg --no_depth --no_sam
```

**Output:** Self-contained HTML file (`<stem>_control_report.html`) with all
signal panels embedded as base64 images — open in any browser, no server needed.

**Panels in the report:**

| Panel | Contents |
|---|---|
| **Inventory table** | Per-signal presence check: directory exists · file exists · size · frame count · thumbnail |
| **Original** | Source image / extracted video frame |
| **Canny Edge** | Dataset frame (if found) + 5 computed threshold presets |
| **Gaussian Blur** | Dataset frame (if found) + 5 computed sigma presets |
| **Depth** | Dataset frame (if found in `control_depth/`) + DepthAnything V2 model output |
| **Segmentation** | Dataset frame (if found in `control_seg/`) + SAM 3 / SAM 2 model output |

**Inventory table columns:**

| Column | Description |
|---|---|
| Signal | edge / blur / depth / seg / captions / metadata |
| Directory | Whether `control_<type>/` directory exists |
| File (this clip) | Whether `<stem>.mp4` (or `.json`) exists for this clip |
| Size | File size in MB |
| How generated | auto-generated / external tool / user-supplied |
| Preview | Thumbnail of the extracted frame from the existing control file |

**Dataset root auto-inference:** if `--input` is inside a directory named
`videos/` or `images/`, the grandparent is used as the dataset root automatically.
Otherwise pass `--dataset_root` explicitly.

**Dependencies:**

```bash
# Core (always needed)
pip install opencv-python numpy Pillow

# Depth panel
pip install torch torchvision transformers

# Segmentation panel
pip install git+https://github.com/facebookresearch/sam2.git
```

**Canny threshold guide** (from the report panels):

| Preset | low / high | When to use |
|---|---|---|
| Very Loose (20/50) | Fine detail, noisy | Never for training — too much noise |
| Loose (30/80) | Moderate detail | Scenes with fine textures |
| **Default (50/150)** | **Balanced** | **`check_control.py` default** |
| Tight (80/200) | Strong edges only | Clean synthetic data, strong outlines |
| Very Tight (100/300) | Major contours only | High-contrast sim2real scenes |

**Blur sigma guide** (from the report panels):

| Preset | σ | kernel | Recommended use |
|---|---|---|---|
| Light (σ=3) | 3 | 19×19 | Subtle smoothing |
| Medium (σ=5) | 5 | 31×31 | Fine appearance transfer |
| **Default (σ=7)** | **7** | **43×43** | **`check_control.py` default** |
| Heavy (σ=10) | 10 | 61×61 | Aggressive texture transfer |
| Max (σ=15) | 15 | 91×91 | Near-silhouette only |

**Converting depth output for `control_depth/`** (report shows colorized preview;
save the grayscale version for training):

```python
import cv2, numpy as np
from transformers import pipeline as hf_pipeline

estimator = hf_pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf")
result = estimator(pil_image)
depth_np = np.array(result["depth"])
depth_u8 = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
# depth_u8 is what goes into control_depth/*.mp4  (0=near, 255=far)
```

**Converting SAM output for `control_seg/`** (report shows colorized preview;
map masks to COCO palette for caption object extraction):

```python
# After SAM mask generation, assign each mask the COCO class color
# that best matches the dominant color in the source region, or
# apply a fixed mapping from semantic class → COCO RGB from colormap.json
```

---

## Stage 4 — Final Checklist

```
FORMAT
[ ] All inputs are MP4 (H.264) — images converted in Stage 0
[ ] FPS = 16 on every clip
[ ] Frame counts are multiples of 93
[ ] Resolution ≥ 480P (720P recommended)

CAPTIONS
[ ] JSON caption file exists for every clip
[ ] Each JSON has "prompt" key
[ ] Prompt ≤ 300 words
[ ] Metadata fields injected (weather, hour_of_day, turf_variety, objects)

CONTROL SIGNALS
[ ] Control video exists for every source clip
[ ] Control type matches experiment config
[ ] Control video dimensions = source dimensions
[ ] Control video frame count = source frame count

DATASET STRUCTURE
[ ] dataset_root/videos/*.mp4
[ ] dataset_root/captions/*.json
[ ] dataset_root/metadata/*.json
[ ] dataset_root/control_<type>/*.mp4
[ ] dataset_root/control_seg/*.mp4  (if using objects from seg)
```

---

## Expected Dataset Directory Layout

```
dataset_root/
├── videos/
│   ├── clip_001.mp4          # 16 FPS, 720P, ×93 frames
│   └── clip_002.mp4
├── metadata/
│   ├── clip_001.json         # weather, hour_of_day, turf_variety
│   └── clip_002.json
├── control_edge/             # auto-generated or provided
│   ├── clip_001.mp4
│   └── clip_002.mp4
├── control_seg/              # optional: for object extraction + seg control
│   ├── clip_001.mp4
│   └── clip_002.mp4
├── captions/                 # generated by generate_captions.py
│   ├── clip_001.json
│   └── clip_002.json
└── colormap.json             # optional: color→class name mapping
```
