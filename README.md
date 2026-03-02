# RA-Det: Robustness Asymmetry Detection for AI-Generated Images

## Overview

RA-Det is a behavior-driven framework for detecting AI-generated images based on **Robustness Asymmetry**:
- **Natural images** preserve stable semantic representations under small perturbations
- **Generated images** exhibit significantly larger feature drift under the same perturbations

Unlike appearance-driven methods that analyze how images look, RA-Det examines how images **behave** under controlled perturbations, making it more robust to modern generators that minimize visual artifacts.

### Key Components

1. **Differential Robustness Probing (DRP)**: A learnable UNet module that applies bounded perturbations to amplify the robustness discrepancy between real and fake images

2. **Multi-Branch Detector**: Aggregates three complementary cues:
   - **Semantic features** from frozen foundation models (DINOv3)
   - **Discrepancy features** (distance, similarity, covariance) between clean and perturbed embeddings
   - **Low-level residual features** for high-frequency artifacts

### Main Results
- Evaluated on 14+ diverse generative models
- Outperforms 10+ existing detection methods
- Achieves **7.81% average performance improvement**
- Data- and model-agnostic, transfers across unseen generators

---

## Quick Start

### Training

Use `scripts/main.sh` for distributed training:

```bash
# 8 GPUs, default settings (eps=16/255, margin=1.0)
bash scripts/main.sh --gpus 8 --niter 1
```

**Important**: The recommended **batch size is 256** (32 per GPU x 8 GPUs). Small batch sizes significantly degrade performance due to the statistics-based discrepancy features.

### Validation

Use `scripts/validate.sh` for evaluation:

```bash
# Validate with default checkpoint
bash scripts/validate.sh --gpus 8
```

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Data Preparation

Organize data in Wang2020 format:

```
data/
├── progan_train/          # Training data
│   ├── airplane/
│   │   ├── 0_real/
│   │   └── 1_fake/
│   └── ...
└── AIGCTestset/test/      # Validation data
    ├── progan/
    ├── stable_diffusion_v_1_4/
    ├── DALLE2/
    └── ...
```

Update paths in `paths.py`:
```python
PROGAN_TRAIN_DATA_PATH = "/path/to/progan_train"
AIGCTEST_DATA_PATH = "/path/to/AIGCTestset/test"
```

### Model Preparation

Download DINOv3 pretrained weights and place in `models/dino/`:
- `dinov3_vitl16_pretrain.pth`

Clone DINOv3 repository:
```bash
cd models
git clone https://github.com/facebookresearch/dinov3.git dinov3_repo
```

---

## Training Details

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--eps` | 16/255 | Perturbation budget (epsilon) |
| `--margin` | 1.0 | Margin for discrepancy loss |
| `--niter` | 1 | Training epochs |
| `--gpus` | 8 | Number of GPUs |
| Batch size | 256 | **Important: 256 recommended** |

### Why Batch Size Matters

RA-Det uses statistics-based discrepancy features (covariance, distance) that require sufficient samples for stable estimation. **Batch size 256 is strongly recommended** - smaller batches lead to noisy statistics and degraded detection performance.

### Advanced Options

Edit `scripts/main.sh` or use `train.py` directly:

```bash
# Four-branch ensemble (foundation + discrepancy + low-level + L2)
torchrun --nproc_per_node=8 train.py \
    --config ensemble_vitl16_raw_lpd_discrepancy \
    --four-branch-ensemble \
    --normalize-loss

# Custom epsilon
python train.py --config ensemble_vitl16_raw_lpd_discrepancy --eps 0.0625
```

---

## Citation

If you use this code, please cite:

```bibtex
@misc{wang2026radet,
  title={RA-Det: Towards Universal Detection of AI-Generated Images via Robustness Asymmetry},
  author={Xinchang Wang and Yunhao Chen and Yuechen Zhang and Congcong Bian and Zihao Guo and Xingjun Ma and Hui Li},
  year={2026},
  url={https://openreview.net/forum?id=zM2l4MGoeu&noteId=zM2l4MGoeu},
  note={OpenReview Archive Direct Upload}
}
```

