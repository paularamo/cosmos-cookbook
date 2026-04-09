# Training Config Reference — Cosmos Transfer 2.5

Anatomy of the experiment config file and key parameters for post-training.

---

## Config File Location

```
cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/
├── config.py                          ← base config (do not edit)
└── experiments/
    ├── transfer2_post_train_example.py   ← copy this as starting point
    └── my_experiment.py                 ← your config (filename = experiment name)
```

**For multiview/AV (HDMap) post-training:**
```
cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/
├── config.py
└── experiments/
    └── transfer2_auto_multiview_post_train_example.py
```

---

## Experiment Config Anatomy

```python
# my_experiment.py
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.config import (
    make_config,
)

# --- DATA ---
data = dict(
    train_dataset=dict(
        data_path="/path/to/dataset_root",   # ← your dataset root
        control_type="edge",                 # "edge" | "depth" | "seg" | "blur"
        # For multi-control:
        # control_types=["edge", "depth"],
        # control_weights=[0.6, 0.4],
        caption_dir="captions",              # relative to data_path
        video_dir="videos",
        control_dir="control_edge",          # relative to data_path
        resolution=(720, 1280),              # (H, W)
        num_frames=93,                       # or 186, 279
        fps=16,
    ),
    batch_size=1,                            # per-GPU batch size
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    shuffle=True,
)

# --- TRAINING ---
training = dict(
    max_iter=1000,             # total training iterations (start small: 500–1000)
    save_iter=100,             # save checkpoint every N iterations
    learning_rate=1e-5,        # recommended starting LR for fine-tuning
    grad_clip=1.0,
    mixed_precision="bfloat16",
)

# --- CHECKPOINT ---
checkpoint = dict(
    load_path=None,            # path to base checkpoint (auto-downloaded if None)
    # load_path="/path/to/Cosmos-Transfer2.5-2B/model.pt"  # force local
    resume=False,              # set True to resume from IMAGINAIRE_OUTPUT_ROOT
)

# --- LOGGING ---
job = dict(
    wandb_mode="disabled",     # "disabled" | "online" | "offline"
    # wandb_project="my-cosmos-project",
    # wandb_run_name="my_experiment",
)

config = make_config(data=data, training=training, checkpoint=checkpoint, job=job)
```

---

## Key Parameter Guide

### Learning Rate
| Scenario | Recommended LR |
|---|---|
| Light domain adaptation (few clips, close domain) | 5e-6 |
| Standard fine-tuning (100–1000 clips) | 1e-5 |
| Large domain gap (synthetic → real) | 2e-5 |

### Iterations
| Dataset Size | Recommended max_iter |
|---|---|
| Tiny (< 50 clips) | 200–500 |
| Small (50–500 clips) | 500–2000 |
| Medium (500–5000 clips) | 2000–10000 |
| Large (> 5000 clips) | 10000+ |

**Rule:** Aim for ~3–5 passes over the dataset. At batch_size=1 with 8 GPUs, effective batch = 8 clips/iter.

### Save Frequency
- Set `save_iter` = 10–20% of `max_iter`
- At minimum save at: 25%, 50%, 75%, 100% of training

---

## Multi-Control Config

```python
data = dict(
    train_dataset=dict(
        control_types=["depth", "edge"],     # order matters for indexing
        control_weights=[0.6, 0.4],
        control_dirs=["control_depth", "control_edge"],
        ...
    )
)
```

---

## Multiview Config (AV / HDMap)

Use the multiview base config for multi-camera setups:
```bash
torchrun --nproc_per_node=8 --master_port=12341 \
  -m scripts.train \
  --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py \
  -- experiment=transfer2_auto_multiview_post_train_example
```

Multiview dataset structure requires camera subdirectories:
```
dataset_root/
├── videos/
│   ├── camera_front/
│   ├── camera_rear/
│   ├── camera_left/
│   └── camera_right/
├── control_hdmap/
│   ├── camera_front/
│   └── ...
└── captions/
```

---

## Training Launch Command (Full)

```bash
export EXP=my_experiment         # must match experiments/my_experiment.py
export NUM_GPUS=8
export MASTER_PORT=12341

IMAGINAIRE_OUTPUT_ROOT=/large/storage/output \
torchrun \
  --nproc_per_node=$NUM_GPUS \
  --master_port=$MASTER_PORT \
  -m scripts.train \
  --config=cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py \
  -- experiment=${EXP} \
     job.wandb_mode=disabled

# Resume from latest checkpoint:
IMAGINAIRE_OUTPUT_ROOT=/large/storage/output \
torchrun ... -- experiment=${EXP} checkpoint.resume=True
```

**Multi-node (2 nodes × 8 GPUs):**
```bash
torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=<HEAD_NODE_IP>:29500 \
  -m scripts.train \
  --config=... -- experiment=${EXP}
```

---

## Output Directory Structure

```
${IMAGINAIRE_OUTPUT_ROOT}/
└── cosmos_transfer_v2p5/
    └── ${GROUP}/
        └── ${EXP}/
            ├── checkpoints/
            │   ├── step_000100/        ← DCP checkpoint (multi-file)
            │   │   ├── model/
            │   │   └── optimizer/
            │   └── step_000200/
            ├── logs/
            └── config.yaml             ← auto-saved config snapshot
```
