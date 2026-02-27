# GRPO Hyperparameter Tuning v4: KL Compromise

| Field | Value |
|-------|-------|
| Date | 2026-02-27 |
| Commit | `7e51f92` |
| RunAI Job | `archer-qwen3-math-v4` (4x H100) |
| W&B Experiment | `Archer2.0-Qwen3-0.6B-Math-v4` |
| Status | Submitted, monitoring |

## Hyperparameter Search Summary (v0-v4)

| Version | entropy_coeff | kl_loss_coef | Score@~100 | Entropy@~100 | Outcome |
|---------|---------------|--------------|------------|--------------|---------|
| V0 | 0 | 0.001 | — | 0.07 | Collapsed (no regularization) |
| V1 | 0.01 | 0.01 | **0.23** | 8.0 | High score, but gibberish outputs |
| V2 | 0.005 | 0.01 | 0.09 | 0.18 | Collapsed at step 78 |
| V3 | 0.01 | 0.05 | 0.072 | 8.5 | Best quality, but score too low |
| V4 | 0.01 | **0.03** | TBD | TBD | Running |

## V3 Diagnosis

V3 (kl_loss_coef=0.05) achieved the best quality metrics:
- Valid=27/64 (vs v1's 15), solve_none=37 (vs v1's 47)
- Logical reasoning 14x better than v1 at similar entropy
- Positive response length 90-315 tokens (real reasoning chains)
- Near-zero repetition

BUT score/mean only 0.072 at step 114 (vs v1's 0.23 at step 100).

Root cause: the 5x KL penalty was too strong, preventing the policy from learning to produce consistently correct answers. The KL effectively anchored the model too close to the reference policy.

## V4 Change

| Parameter | V3 | V4 | Rationale |
|-----------|-----|-----|-----------|
| `kl_loss_coef` | 0.05 | **0.03** | 3x original (v1), 60% of v3. Less restrictive for learning |

Expected behavior at kl=0.03:
- At KL_loss=2.0: penalty = 0.06 (vs v3's 0.10, v1's 0.02)
- At entropy=8: bonus = 0.08
- Balance point: KL_loss ≈ 2.67 (achievable)
- Should allow faster score improvement while maintaining some quality

## V4 Monitoring Criteria

| Metric | Target | Warning |
|--------|--------|---------|
| Score/mean @ step 100 | > 0.15 | < 0.10 |
| Entropy | 5-8 | > 9 or < 2 |
| Valid | > 20 | < 10 |
| Logical_conn | > 0.10 | < 0.03 |
