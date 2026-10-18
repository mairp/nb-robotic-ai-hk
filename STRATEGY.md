# Sim to Table: 3-hour strategy

Challenge: see `sim-to-table-hackathon.md`. Detailed runbooks are in `runbooks/` (these are research notes; none of the commands have been run yet).

- **Clock:** T0 ≈ 14:28 (estimated; correct it if wrong). **Hard stop for uploads: T+2:45 ≈ 17:13.** Cutoff T+3:00 ≈ 17:28.
- **Decisions so far:** 3+ people. Arm **not calibrated** yet. Policy is **GR00T N1.7-3B**.
- **Infra (live, 12:50 UTC; details in `NEXT-STEPS.md`):** project `group31-project`, region `uk-south2`.
  - **One** GPU VM `so101-gpu` (RTX PRO 6000, 96 GB), `ssh ubuntu@REDACTED_VM_IP`. The starter repo's `setup_vm.sh` is installing into `~/work`.
  - Bucket `REDACTED_BUCKET`: class ENHANCED_THROUGHPUT (Standard quota is 0), AWS profile `hackathon-storage`, `submission/` already exists.
  - Cosmos **Reason**: **Token Factory has no Cosmos model for our key** (checked 13:15 UTC). It is self-hosted on the VM instead: vLLM `nvidia/Cosmos-Reason2-2B` in Docker container `cosmos-reason`, at 127.0.0.1:8000, using 20% of the GPU. Cosmos **Transfer** goes through the organizers if they host it; otherwise we run `nvidia/Cosmos-Transfer2.5-2B` on the VM (our HF token can download it).
  - **Sim verified:** 2 of 2 smoke-test episodes succeeded, ~18 s each, LeRobot v2.1 with front and wrist cameras. **Isaac Lab is pinned to `24a2ebfd7`** (Isaac Sim 6.1.0, still has `mdp.image`) with the `teleop` and `mimic` extras.
- **Goal:** secure the ~70 points that don't depend on success (Cosmos 20, Nebius 20, deployment 15, creativity 15), then go for accuracy (30).

## Roles and timeline

| Person | T+0:00–0:30 | T+0:30–1:15 | T+1:15–2:15 | T+2:15–3:00 |
|---|---|---|---|---|
| **P1 Table** | Calibrate leader + follower; lock the camera and lamp; take a station photo | Teleop ~40 episodes, 4–5 per grid spot, each ending with a clean release | GR00T client on the laptop + SSH tunnel to the server | Eval main + ablation on the 9 spots |
| **P2 Sim + train** | Check the VM GPU type; fix `env.sh`; set cube range and fixed pad pose; start `collect.sh` | Compare a sim frame with a real frame (camera, units, gripper); convert real v3 → v2.1 | Benchmark 50 steps; train main + ablation | Serve the main checkpoint (port 5556) |
| **P3 Cosmos + submission** | ~~Bucket + `submission/`~~ (done). Get a Token Factory key (promo code) and ask the organizers about Transfer hosting | Transfer 8–15 clips (edge); Reason on all; `curation.json` | Manifest + README (ensemble story) | Upload, then confirm with `aws s3 ls --summarize` |

## Blockers to clear first
1. ~~Isaac Sim needs RT cores (no H100/H200).~~ **Resolved:** our RTX PRO 6000 has RT cores. Sim, training and serving all run on the one VM, **one after another**, because there is only one GPU.
2. **Cameras:** the sim has `front` + `wrist`. Either record the real data with both, or drop `wrist` from `modality.json` everywhere. Decide this before P1 records.
3. **Layout:** the sim randomizes both cube and pad. Fix the pad pose and set `CUBE_POSE_RANGE` to the real 3×3 grid area, measured in meters from the robot base.
4. **Consistency:** the cube color and the **task string** must be identical in sim, real and inference.
5. **Cosmos:**
   - **Reason:** self-hosted `Cosmos-Reason2-2B` on the VM. `nebius-physical-ai` claimed Token Factory has `Cosmos3-Super-Reasoner`; our key's `/v1/models` doesn't list it.
   - **Transfer** (`nvidia/Cosmos-Transfer2.5-2B`) is not on Token Factory. Ask the organizers. Fallback: run it on our VM between sim and training.

## Key commands
```bash
# P1: calibration (keep the same ids from here on)
lerobot-find-port
lerobot-calibrate --robot.type=so101_follower --robot.port=<PORT_F> --robot.id=follower_1
lerobot-calibrate --teleop.type=so101_leader  --teleop.port=<PORT_L> --teleop.id=leader_1
# record: --dataset.episode_time_s=25 --dataset.reset_time_s=10 ; --resume=true to continue

# P2: sim (counts only successful episodes; ~27 s each; resumable)
NUM_EPISODES=150 ./collect.sh --resume && ./prepare_dataset.sh
# train: examples/finetune.sh, batch 32, --dataset-path sim:real[:cosmos_kept], MAX_STEPS from 50-step benchmark
# serve: run_gr00t_server.py (5556: 5555 is taken by nv-hostengine on Nebius); laptop: ssh -L 5556:localhost:5556 ubuntu@REDACTED_VM_IP; eval_so100.py

# P3: storage (profile set up by infra/setup-storage-profile.sh; the endpoint is in the profile)
aws --profile hackathon-storage s3 sync ./submission s3://REDACTED_BUCKET/submission/
aws --profile hackathon-storage s3 ls s3://REDACTED_BUCKET/submission/ --recursive --summarize

# P3: Reason (Token Factory, OpenAI-compatible)
curl -s https://api.tokenfactory.nebius.com/v1/models -H "Authorization: Bearer $NEBIUS_TOKEN_FACTORY_KEY" | grep -o '"id":"[^"]*Cosmos[^"]*"'
```

## Cosmos plan
- **Transfer:**
  - `POST /v1/infer` with `edge.control_weight` 0.9–1.0.
  - **Leave `control` out** so the edge map is extracted automatically.
  - Pass the station photo in `image_context`, and optionally add `vis` at about 0.2 to keep the cube's color.
  - Clips must be 93–480 frames, so subsample long episodes.
  - Map the output back to the actions **by frame index** (the output may come back at 16 fps).
  - Write the results to a new `cosmos_aug` dataset, keeping actions and states unchanged.
- **Time the first Transfer request** before planning how many clips to run.
- **Reason** (`cosmos/reason_score.py` → vLLM `Cosmos-Reason2-2B` on the VM):
  - Send frames as base64 `image_url` data URLs; the guide shows 4–8 frames per request. Ask for JSON only, then strip `<think>…</think>` and any ```` ```json ```` fences before parsing.
  - Sanity-check it on 2–3 original sim clips.
  - Then score all clips and save the results to `curation.json`.
- **The stated curation decision:** keep a clip only if `task_success` is true, the cube and gripper are visible throughout, and `overall_score` ≥ 6. Report it as "N of M dropped".
- **Fallback:** don't let training wait on Cosmos. Start the ablation at T+1:15, and start the main run as soon as the kept clips exist.

## Training
- **Main:** sim + kept Cosmos clips + real. **Ablation:** sim + real.
- Both fit on our one 96 GB GPU running side by side, at about half speed each. Don't run sim or Transfer at the same time.
- Both runs must finish by T+2:05. Benchmark first and size `MAX_STEPS` to fit (estimate: 4–5k steps).

## Submission checklist (Nebius Object Storage)
- [ ] Main checkpoint, plus the ablation checkpoint (or its results)
- [ ] Manifest: episode counts per source (sim / cosmos / real). Limits: real ≤ 50, total ≤ 300
- [ ] `curation.json` + the stated curation decision
- [ ] README: pipeline, Nebius training, ensemble (Transfer → Reason → GR00T), ablation results, eval success counts
- [ ] Confirmed with `aws s3 ls`

## Blind spots to keep watching
- Sim and real data that don't match: joint units ([-100,100] arm, [0,100] gripper), camera pose, gripper range.
- Recalibrating after recording invalidates the data you already recorded.
- The shared Cosmos endpoint will have queues and 429 rate limits. Back off as long as `Retry-After` says.
- Wifi latency to the remote GR00T server. Test it before T+2:00.
- Policies that never open the gripper. Demos must end with a clean release.
- Moving the station after recording.

## Questions for the organizers
- Do Cosmos-augmented clips count toward the 300-episode cap?
- Do you host **Cosmos Transfer 2.5**? If so: URL, key, request schema (`/v1/infer`?), rate limits, parallel requests.
- Submit to our bucket (`REDACTED_BUCKET`, uk-south2) or to yours? How do judges get access, and is there a required folder layout?
- The exact cutoff time, with time zone.
