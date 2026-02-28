# RA-Det: Robustness Asymmetry Detection for AI-Generated Images

> **Preprint Version - Under Review at ICML 2026**

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
