# RA-Det Improvement Log

## Iteration 1: Epsilon Randomization for Domain Generalization

### Improvement Plan
Add **epsilon randomization** during training: instead of using a fixed epsilon (perturbation budget), randomly sample epsilon from a range [eps_min, eps_max] for each batch.

### Theoretical Reasoning (from paper)

The paper states in Section 3.2.1 that DRP generates bounded perturbations with magnitude constrained by epsilon. The paper also mentions domain generalization through "epsilon randomization and loss normalization for cross-generator robustness."

**Why this helps:**
1. **Memorization varies across generators** - Different generative models have different memorization tendencies, leading to varying robustness asymmetry magnitudes
2. **Single epsilon is brittle** - Training with fixed epsilon may overfit to a specific perturbation scale
3. **Randomization improves generalization** - By exposing the model to varying perturbation budgets during training, it learns to detect robustness asymmetry at multiple scales

**Paper reference (Theorem 4.3):**
> "The expected feature-shift under p_θ exceeds that under p by a margin controlled below by ε²/n × ∆ - B√(M/2) up to O(ε⁴)"

This shows the relationship between epsilon and the detectable shift - larger epsilon amplifies the signal but may also introduce noise. Randomizing epsilon helps find the optimal range.

### Implementation Changes

1. Add `eps_randomization` flag to config
2. Add `eps_min` and `eps_max` parameters
3. Modify training loop to sample epsilon per batch
4. Scale the perturbation δ accordingly

### Expected Outcome
- Improved cross-generator generalization
- More robust detection across varying perturbation levels
- Better stability during training

---

### Results (to be filled after user runs)
*Waiting for user feedback...*