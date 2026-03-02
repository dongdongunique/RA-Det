# RA-Det Iteration Workflow

## Rules
1. **No changes to validation** - Do not modify validation data paths or validation functions
2. **CPU/GPU separation** - All code changes happen on CPU cluster; user runs training on GPU cluster
3. **Only 1 iteration** - Make focused improvements, not multiple attempts
4. **No try/except** - Write clean code without exception handling
5. **Feature branch only** - All changes in `feature-dev` branch, never modify `master` directly

## Iteration Loop

### Step 1: Document Improvement
Write to `IMPROVEMENT.md`:
- Iteration number
- Planned improvement
- Theoretical reasoning (based on paper theory)

### Step 2: Implement Changes
Modify code based on the improvement plan.

### Step 3: Provide Command
Give user the training command to run on GPU cluster.

### Step 4: User Feedback
User runs the command and provides results.

### Step 5: Document Results
Update `IMPROVEMENT.md` with:
- What was changed
- Results from user's run
- Next steps (if needed)

## Paper Theory Summary

### Core Concept: Robustness Asymmetry
- **Real images**: Stable semantic representations under perturbations (high similarity)
- **Fake images**: Large feature drift under perturbations (low similarity)

### Architecture Components
1. **DRP (Differential Robustness Probing)**: Learnable UNet generating bounded perturbations δ
2. **Semantic Branch**: Frozen foundation encoder (DINOv3) + classification head
3. **Discrepancy Branches**: Distance (||e - e'||₂) and vector difference (Δ = e - e')
4. **Low-level Residual Branch**: Median filter residual + CNN

### Training Objective
```
L_comp = L_bce + L_ra
L_ra = ReLU((s_fake - s_real) + γ)  # γ = margin
```

### Key Hyperparameters
- `eps`: Perturbation budget (default: 16/255)
- `margin`: Contrastive loss margin (default: 1.0)
- `batch_size`: 256 (critical for stable statistics)