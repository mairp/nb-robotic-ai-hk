# Next steps (written 2026-10-18 12:12 UTC)

This updates `STRATEGY.md` with what's actually in `group31-project` (uk-south2). Every line below was checked against the live project.

## What's actually in the project
- **GPU:** 1× RTX PRO 6000 Blackwell (96 GB, *has RT cores*), `gpu-rtx6000-a` / `1gpu-24vcpu-218gb`. **There is no L40S or H100.** This one GPU runs sim, Cosmos Reason, training and serving, **one at a time**.
- **VMs:** none of our own. ("Nebius VM ready" in STRATEGY is wrong for this project; check whether a teammate has one somewhere else.)
- **Endpoint `cosmos-reason2`:** deleted by a teammate ~12:40 UTC. Run Reason inside the VM (vLLM) when needed.
- **Bucket:** `REDACTED_BUCKET` (ENHANCED_THROUGHPUT, AWS profile `hackathon-storage`, 200 GiB, `https://storage.uk-south2.nebius.cloud`). Keys are in SecretStash `hackathon-object-storage`.
- **Limits left:** 2 more VMs, 400 GiB of disk, 20 vCPUs for CPU-only VMs, 1 public IP.

## The key decision: who gets the GPU
The endpoint blocks Isaac Sim and training. Recommendation:
1. **Score clips with Reason now**, while sim is still being set up. Smoke-test it first with 2–3 clips.
2. **Then stop the endpoint** and create **one GPU VM** for sim collection, training (main and ablation together, 96 GB) and the GR00T server.
3. **Reason is self-hosted on the VM:** vLLM `nvidia/Cosmos-Reason2-2B` (Docker `cosmos-reason`, 127.0.0.1:8000, 20% of the GPU). Token Factory lists no Cosmos model for our key. Stop the container before the two training runs if memory is tight.
4. **Cosmos Transfer 2.5 (`nvidia/Cosmos-Transfer2.5-2B`) is not on Token Factory.** The same repo runs it on its own GPU. Ask the organizers whether they host it; if not, run it on our VM *between* sim and training (it needs the GPU).

## Order of work (owner → output)
| # | Step | Owner | Done when |
|---|---|---|---|
| 0 | Confirm the clock: is T0 14:28 Berlin time? | all | hard-stop time agreed |
| 1 | ✅ Test `cosmos-reason2` with a real request (bearer token): done 12:23 UTC, image→JSON in 0.7 s | P3 | JSON score for 1 clip |
| 2 | ✅ Create `submission/` in the bucket; check the upload works (bucket class switched to ENHANCED_THROUGHPUT: Standard quota is 0) | P3 | `aws s3 ls` OK |
| 3 | Calibrate the arm, lock the station, take the station photo | P1 | photo + ids fixed |
| 4 | Score clips with Reason → `curation.json`, then **stop the endpoint** | P3 | GPU free |
| 5 | ✅ GPU VM `so101-gpu` created 12:43 UTC: `ssh ubuntu@REDACTED_VM_IP` | P2 | SSH works |
| 6 | Sim collection (`NUM_EPISODES=150`) using the real grid and a fixed pad | P2 | episodes on disk + bucket |
| 7 | Teleop ~40 episodes (≤50) | P1 | real dataset |
| 8 | Transfer 8–15 clips via the organizers' endpoint → Reason → keep/drop | P3 | `cosmos_aug` + "N of M dropped" |
| 9 | Benchmark 50 steps → train main and ablation side by side | P2 | 2 checkpoints |
| 10 | Serve the main checkpoint (5556; 5555 is DCGM) over an SSH tunnel; evaluate on 9 spots | P1+P2 | success counts |
| 11 | Upload: checkpoints, manifest, curation, README | P3 | `--summarize` OK before the hard stop |

## Risks this adds
- **One GPU** means sim, Reason and training can't overlap. Every minute the endpoint sits idle delays sim.
- A new GPU VM uses one of the 2 remaining VM slots and a public IP if SSH goes direct.
- The endpoint's disk is wiped when it stops: the model downloads again on restart (2B, minutes).
