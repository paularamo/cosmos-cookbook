# Checkpoint Management — Cosmos Transfer 2.5

Monitoring, converting, and using checkpoints from post-training runs.

---

## Checkpoint Formats

| Format | Description | Use For |
|---|---|---|
| **DCP** (Distributed Checkpoint) | Multi-file sharded directory | Training resumption, distributed training |
| **PyTorch .pt** | Single consolidated file | Inference, sharing, deployment |
| `checkpoint_ema_bfloat16.pt` | EMA weights in bfloat16 | **Best for inference** |
| `checkpoint_ema_float32.pt` | EMA weights in float32 | High-precision evaluation |
| `checkpoint.pt` | Full model (non-EMA) | Alternative if EMA unavailable |

---

## Monitor Training Progress

```bash
# Watch checkpoint directory for new saves
watch -n 30 ls -lht ${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer_v2p5/.../checkpoints/

# Check latest step
ls ${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer_v2p5/.../checkpoints/ | sort -V | tail -1

# Read training loss from logs (if W&B disabled)
tail -f ${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer_v2p5/.../logs/train.log
```

---

## Convert DCP → PyTorch

Run after training completes or at any saved checkpoint:

```bash
# Identify checkpoint directory
CKPT_DIR="${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer_v2p5/.../checkpoints/step_001000"

# Convert
python scripts/convert_distcp_to_pt.py \
  "${CKPT_DIR}/model" \
  "${CKPT_DIR}"

# Outputs:
#   ${CKPT_DIR}/checkpoint.pt
#   ${CKPT_DIR}/checkpoint_ema_float32.pt
#   ${CKPT_DIR}/checkpoint_ema_bfloat16.pt  ← use this for inference
```

**Warning:** Conversion writes into the checkpoint directory. Backup if needed:
```bash
cp -r "${CKPT_DIR}" "${CKPT_DIR}_backup"
```

---

## Resume Training from Checkpoint

```bash
IMAGINAIRE_OUTPUT_ROOT=/large/storage/output \
torchrun --nproc_per_node=8 --master_port=12341 \
  -m scripts.train \
  --config=cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py \
  -- experiment=${EXP} \
     checkpoint.resume=True \
     job.wandb_mode=disabled
```

Resume loads the **latest** DCP checkpoint found in the output directory automatically.

To resume from a specific step:
```bash
  -- experiment=${EXP} \
     checkpoint.resume=True \
     checkpoint.resume_step=500
```

---

## Validate Checkpoint Before Full Inference

Run a quick 1-clip test to confirm the checkpoint is usable:

```bash
# Create a minimal test spec
cat > /tmp/test_spec.json << 'EOF'
{
  "prompt_path": "captions/clip_001.json",
  "output_dir": "/tmp/cosmos_test_output/",
  "video_path": "videos/clip_001.mp4",
  "guidance": 3.0,
  "num_steps": 10,
  "edge": {
    "control_path": "control_edge/clip_001.mp4",
    "control_weight": 1.0
  }
}
EOF

python examples/inference.py \
  --params_file /tmp/test_spec.json \
  --checkpoint_path "${CKPT_DIR}/checkpoint_ema_bfloat16.pt" \
  -o /tmp/cosmos_test_output/
```

---

## Checkpoint Selection Guide

| Training Progress | Expected Behavior |
|---|---|
| 0–20% | Loss still high; output may look like base model |
| 20–50% | Domain starts appearing; check for artifacts |
| 50–80% | Target domain features converging |
| 80–100% | Fine-grained detail; watch for overfitting |

**Signs of overfitting:** Output exactly copies input clips; no generalization to unseen clips.
**Signs of underfitting:** Still looks like pre-trained domain; increase iterations.

---

## Download Base Checkpoint Manually

```bash
# Ensure HF auth is set
hf auth login

# Download model files
huggingface-cli download nvidia/Cosmos-Transfer2.5-2B \
  --local-dir ./checkpoints/Cosmos-Transfer2.5-2B

# Available model variants on HuggingFace:
# - nvidia/Cosmos-Transfer2.5-2B           (general, all modalities)
# - nvidia/Cosmos-Transfer2.5-2B-Edge      (distilled edge, 4-step)
```

---

## Checkpoint Size Reference

| Checkpoint Type | Approximate Size |
|---|---|
| DCP (per GPU shard) | ~2–4 GB |
| Full DCP directory (8 GPU) | ~16–32 GB |
| PyTorch .pt (bfloat16) | ~4–5 GB |
| PyTorch .pt (float32) | ~8–10 GB |

Ensure `IMAGINAIRE_OUTPUT_ROOT` has at minimum **100 GB** free for training with multiple checkpoint saves.
