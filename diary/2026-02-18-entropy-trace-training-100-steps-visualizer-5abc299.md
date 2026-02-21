## 2026-02-18 - Entropy Trace Training (100 Steps) + Visualizer

| Item | Details |
| --- | --- |
| Request | Run the newly implemented entropy-trace training + visualization pipeline end-to-end, then commit and push. |
| Delivery | Added token-level entropy trace writer, integrated it into the trainer loop, provided a Gradio visualizer, and verified the pipeline on RunAI for 100 training steps with W&B online logging. |
| Scope | GPU workloads executed via RunAI on `h100` (1 GPU). Local machine used only for static checks (no CUDA). |

### Changes
| Path | Change | Why |
| --- | --- | --- |
| `Archer2.0/dapo/entropy_trace.py` | Added | Persist per-step token-level entropy traces to `steps/step_*.npz` plus `manifest.jsonl`/`summary.json` for later inspection. |
| `Archer2.0/dapo/dapo_ray_trainer.py` | Updated | Hook entropy trace writing after `old_log_prob` when entropy is available; add config knobs `trainer.entropy_trace.*`. |
| `Archer2.0/scripts/train/run_archer2.0_qwen2.5_1.5b_math_entropy_trace.sh` | Added | Training wrapper forcing `ppo_epochs=1`, `total_training_steps=100`, and enabling entropy tracing. |
| `Archer2.0/scripts/train/run_archer2.0_qwen2.5_1.5b_math_entropy_trace_1gpu.sh` | Added | A reproducible, fixed-override 1GPU recipe used to run the successful `v9` job on RunAI without override loss across shell layers. |
| `Archer2.0/scripts/train/visualize_rollout_entropy.py` | Added | Gradio UI to browse steps/responses and inspect per-token entropies with filtering, sorting, and CSV export. |
| `Archer2.0/scripts/train/README_entropy_trace.md` | Added | Minimal usage guide for trace generation and visualization. |
| `Archer2.0/.gitignore` | Updated | Ignore `models/` to avoid accidentally committing local model checkpoints. |

### Validation
| Check | Evidence | Result |
| --- | --- | --- |
| RunAI training (entropy trace) reaches 100 steps | RunAI job `archer-entropy-trace-100-v9` and `Archer2.0/output/train_entropy_trace_100_v9.log` | Pass (`step:100`, job `Succeeded`). |
| Trace artifacts written for all steps | `Archer2.0/output/Archer2.0/Archer2.0-Qwen2.5-1.5B-Math-EntropyTrace-v9/entropy_trace/steps/` | Pass (100 `*.npz` files; `manifest.jsonl` includes `step=100`). |
| W&B online logging | `https://wandb.ai/doub7e/Archer2.0/runs/Archer2.0-Qwen2.5-1.5B-Math-EntropyTrace-v9` | Pass (run synced; summary written). |
| Static checks | `python3 -m py_compile ...` and `bash -n ...` | Pass (no syntax errors). |

### Git
| Field | Value |
| --- | --- |
| Commit | `5abc299` |
| Branch | `main` |
| Remote | `doub7e` |
| Push | Yes (`c9b5405..5abc299`) |

### Notes
- Repro command (RunAI container): `bash scripts/train/run_archer2.0_qwen2.5_1.5b_math_entropy_trace_1gpu.sh` with `WANDB_API_KEY` set via environment.
- Visualizer command (CPU): `python3 scripts/train/visualize_rollout_entropy.py --trace-dir output/Archer2.0/Archer2.0-Qwen2.5-1.5B-Math-EntropyTrace-v9/entropy_trace --model-path ./models/DeepSeek-R1-Distill-Qwen-1.5B --port 7862`.

