# Team group31: Sim to Table submission

**Status (2026-10-18 ~16:30 UTC):** the submitted policy is the **sim-only run** (`ablation_sim_only`, checkpoint-3000). Real-teleop recording failed after calibration issues — only 2 of the planned ≤50 episodes were captured — so the "main" (sim + Cosmos + real) training never started. Everything below reflects the actual VM state.

## Pipeline (the "ensemble")
1. **Isaac Sim / Isaac Lab:** a scripted expert collects SO-101 pick-and-place demos over the 3×3 grid — 150 episodes (46,316 frames), +100 more with a second seed (30,885 frames, collected but unused).
2. **Cosmos Reason 2 (2B), self-hosted on the VM** (vLLM, 20% of the GPU): scores every clip against the stated rule — keep if `task_success` ∧ `cube_and_gripper_visible` ∧ `overall_score ≥ 6`.
3. **GR00T N1.7-3B:** fine-tuned on Nebius AI Cloud on the sim dataset (ablation "sim-only" configuration; it became the final run).

## Curation decision (Cosmos Reason)
- **150 of 150 sim clips kept** (the scripted expert produces clean episodes).
- **Validation with negative controls:** we hand-crafted 4 corrupted clips (episodes cut before the release). Reason **dropped exactly the 2 whose cube ended off the pad** (`final_cube_location: on_table_not_on_pad`) and kept the 2 that genuinely finished on-pad — **correct verdict on 4 of 4 controls**. See `curation_negative_controls.json`.

## Data (see `manifest.json`)
| Source | Submitted policy | Notes |
|---|---|---|
| Sim (seed 0) | 150 episodes / 46,316 frames | used in training |
| Sim (seed 1) | 100 episodes / 30,885 frames | collected, unused |
| Cosmos Transfer (kept) | 0 | endpoint not available in time |
| Real teleop (≤50) | 2 episodes / 1,494 frames | recorded, unused (recording failed) |
| **Total** | **150** (≤300 cap) | |

## Training on Nebius
VM `so101-gpu` (RTX PRO 6000 Blackwell, 96 GB, uk-south2). 3,000 steps, batch 16, train loss 0.251 (final 0.105), wall time **29 min**, finished 14:51 UTC. Checkpoint: `checkpoints/ablation_sim_only/checkpoint-3000`.

## Results
| Policy | Eval (9 grid spots, closed-loop in Isaac Lab) |
|---|---|
| Sim-only (submitted) | **0/9** — all episodes timed out at 600 steps; inference pipeline healthy (~130 ms/step). Honest result: the 3k-step sim-only policy does not yet solve the task in closed loop. |

## Known gaps (honest disclosure)
- Main "with Cosmos" run not trained: blocked on real episodes that never materialized.
- Cosmos Transfer 2.5 not run: no hosted endpoint; skipping it protected the training window on the single GPU.
- Real teleop: 2 episodes only — leader-arm calibration (gripper span) broke mid-session; recalibration was in progress at the deadline.
