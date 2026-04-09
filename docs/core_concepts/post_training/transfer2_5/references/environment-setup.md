# Environment Setup — Cosmos Transfer 2.5

Full setup guide: hardware check → repo clone → Python environment → extra
dependencies (FiftyOne, DepthAnything V2, SAM) → env vars → HuggingFace auth →
verification.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | 1× A100 40 GB | 8× H100 80 GB |
| VRAM (training) | 40 GB | 80 GB × N GPUs |
| VRAM (inference) | 24 GB | 80 GB |
| CUDA | 12.4 | **12.8** |
| Driver | 550+ | 570+ |
| Storage (model cache) | 100 GB | 500 GB |
| Storage (dataset + outputs) | 200 GB | 2 TB |

Check your GPU and CUDA version:

```bash
nvidia-smi                          # GPU name, VRAM, driver
nvcc --version                      # CUDA toolkit version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

---

## 1. Clone the Repository

```bash
git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git
cd cosmos-transfer2.5
```

---

## 2. Python Environment

### Option A — uv (recommended)

`uv` is faster than pip and manages the virtualenv automatically.

```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env   # or restart shell

# Create env and install with CUDA 12.8 extras
uv sync --extra=cu128

# Activate
source .venv/bin/activate
```

For CUDA 12.4 systems:

```bash
uv sync --extra=cu124
```

**Standalone skill venv** (without the full cosmos repo — for data validation only):

```bash
# Create venv with uv
uv venv .venv --python 3.12
source .venv/bin/activate

# Core deps
uv pip install opencv-python numpy Pillow

# PyTorch — use cu128 wheel even on CUDA 13.x (backward compatible)
uv pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128

# DepthAnything V2 + SAM
uv pip install transformers accelerate huggingface_hub \
               "git+https://github.com/facebookresearch/sam2.git"
```

> **CUDA 13.x note:** CUDA drivers are backward compatible. The `cu128` PyTorch
> wheel runs correctly on systems with CUDA 13.1 (e.g. Blackwell GPUs).

### Option B — conda

```bash
conda create -n cosmos-transfer python=3.10 -y
conda activate cosmos-transfer

# Install PyTorch for CUDA 12.8
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

# Install the package in editable mode
pip install -e .
```

---

## 3. Extra Dependencies

These are required by the skill scripts but not part of the core package.

### 3a. FiftyOne — Dataset Management & HuggingFace Download

Used to download and inspect datasets from HuggingFace Hub.

```bash
pip install fiftyone
```

**Download a HuggingFace dataset with FiftyOne:**

```python
import fiftyone as fo
import fiftyone.utils.huggingface as fouh

# Download and open in the FiftyOne App
dataset = fouh.load_from_hub("pjramg/moth_biotrove")
session = fo.launch_app(dataset)
```

Or from the CLI:

```bash
fiftyone zoo datasets load pjramg/moth_biotrove --source huggingface
```

### 3b. DepthAnything V2 — Depth Control Signal

Used by `scripts/visualize_controls.py` and for generating `control_depth/`.

```bash
uv pip install transformers accelerate huggingface_hub
# verified: transformers==5.5.0
```

Usage in Python:

```python
from transformers import pipeline
estimator = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Large-hf",
    device=0,   # GPU index; -1 for CPU
)
```

Model sizes and VRAM:

| Model ID | Size | VRAM |
|---|---|---|
| `Depth-Anything-V2-Small-hf` | vits | ~2 GB |
| `Depth-Anything-V2-Base-hf`  | vitb | ~4 GB |
| `Depth-Anything-V2-Large-hf` | vitl | ~8 GB |

### 3c. SAM 3 / SAM 2 — Segmentation Control Signal

Used by `scripts/visualize_controls.py` and for generating `control_seg/`.

```bash
uv pip install "git+https://github.com/facebookresearch/sam2.git"
# installs latest SAM 2 (sam2 package); visualize_controls.py tries SAM 3 → SAM 2.1 → SAM 2
```

`visualize_controls.py` tries SAM 3 → SAM 2.1 → SAM 2 in order and uses the
first that loads. No manual model selection needed.

### 3d. Control Signal Scripts (OpenCV, Pillow)

Used by `scripts/check_control.py` and `scripts/visualize_controls.py`.

```bash
pip install opencv-python numpy Pillow
```

### 3e. Full Extra Install (all at once)

```bash
pip install fiftyone transformers accelerate \
            opencv-python numpy Pillow \
            git+https://github.com/facebookresearch/sam2.git
```

---

## 4. Environment Variables

Set these before training. Add to `~/.bashrc` or a project `.env` file.

```bash
# HuggingFace model cache — needs ~100 GB free
export HF_HOME=/path/to/large/storage/hf_cache

# Training output root — needs ~500 GB free
# Checkpoints land at: $IMAGINAIRE_OUTPUT_ROOT/cosmos_transfer_v2p5/<group>/<exp>/
export IMAGINAIRE_OUTPUT_ROOT=/path/to/large/storage/output

# Optional: speed up HF downloads
export HF_HUB_ENABLE_HF_TRANSFER=1
pip install hf-transfer   # if not already installed
```

**`/tmp` is the default for `IMAGINAIRE_OUTPUT_ROOT` — never use it for real
training runs; checkpoints will be lost on reboot.**

---

## 5. HuggingFace Authentication

Required to download NVIDIA Cosmos model weights (NVIDIA Open Model License).

```bash
# Install CLI if missing
pip install huggingface_hub

# Login (opens browser or accepts token from stdin)
huggingface-cli login
# or:
hf auth login
```

Then accept the license on the model card page:
`https://huggingface.co/nvidia/Cosmos-Transfer2-7B`

Verify access:

```bash
python -c "
from huggingface_hub import whoami
print(whoami()['name'])
"
```

---

## 6. Verification Checklist

Run these after setup to confirm everything is in order.

```bash
# 1. PyTorch + CUDA
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"

# 2. GPU count visible to torch
python -c "import torch; print(torch.cuda.device_count(), 'GPU(s) available')"

# 3. FiftyOne
python -c "import fiftyone; print('fiftyone', fiftyone.__version__)"

# 4. DepthAnything V2 (transformers)
python -c "from transformers import pipeline; print('transformers OK')"

# 5. SAM
python -c "from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator; print('SAM OK')"

# 6. OpenCV
python -c "import cv2; print('cv2', cv2.__version__)"

# 7. Cosmos package
python -c "import cosmos_transfer2; print('cosmos_transfer2 OK')"

# 8. HuggingFace auth
huggingface-cli whoami
```

Expected output (example):

```
2.11.0+cu128  NVIDIA RTX PRO 5000 Blackwell Generation Laptop GPU
1 GPU(s) available
fiftyone 1.3.0
transformers OK   # verified: transformers==5.5.0, torch==2.11.0+cu128
SAM OK            # verified: sam2 from github.com/facebookresearch/sam2
cv2 4.13.0        # verified: opencv-python==4.13.0.92
cosmos_transfer2 OK
your-hf-username
```

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `CUDA error: no kernel image` | PyTorch / CUDA version mismatch | Reinstall torch matching your `nvcc --version` |
| `uv sync` fails on cu128 extra | Driver < 550 or CUDA < 12.4 | Update driver or use `--extra=cu124` |
| `HfHubHTTPError: 401` | Not authenticated | Run `huggingface-cli login` |
| `HfHubHTTPError: 403` | License not accepted | Visit model page and accept NVIDIA Open Model License |
| `No module named 'sam2'` | SAM not installed | `pip install git+https://github.com/facebookresearch/sam2.git` |
| OOM during training | Batch too large or resolution too high | Set `batch_size=1`, `resolution=(480, 854)` in config |
| `IMAGINAIRE_OUTPUT_ROOT` not set | Env var missing | `export IMAGINAIRE_OUTPUT_ROOT=/your/path` before torchrun |
| FiftyOne port conflict | Another FiftyOne session open | `fo.close_app()` or pass `port=5152` to `launch_app()` |
