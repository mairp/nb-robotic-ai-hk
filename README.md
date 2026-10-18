# nb-robotic-ai-hk: Sim to Table hackathon (team group31)

Our work from the **NVIDIA × Nebius AI Builders Hackathon "Sim to Table"** (Berlin, 18 Oct 2026). The challenge
was to train an **SO-101** robot arm to pick up a 30 mm cube and place it on a green pad. The required pipeline was
Isaac Sim demos → NVIDIA Cosmos (Transfer + Reason) → a few real teleop demos → policy training on Nebius → evaluation
on the real arm.

The full brief and scoring are in [`sim-to-table-hackathon.md`](sim-to-table-hackathon.md).

## What we actually got done

| Stage | Result |
|---|---|
| **Sim** (Isaac Sim 6.1 / Isaac Lab, scripted expert) | 150 episodes / 46,316 frames over the 3×3 grid, plus 100 more with a second seed (unused) |
| **Cosmos Reason 2 (2B)** | Self-hosted with vLLM on our GPU VM. Scored all clips: 150 of 150 kept. It gave the right verdict on 4 of 4 hand-made negative controls |
| **Cosmos Transfer 2.5** | Not run: there was no hosted endpoint, and running it on our only GPU would have cost the training window |
| **Real teleop** | Only 2 episodes. Leader-arm calibration (gripper span) broke mid-session |
| **Training** (GR00T N1.7-3B on Nebius) | Sim-only run: 3,000 steps, batch 16, final loss 0.105, 29 min on one RTX PRO 6000 (96 GB) |
| **Eval** (9 grid spots, closed loop in Isaac Lab) | **0/9**: every episode timed out. Inference itself was healthy (~130 ms/step) |

The submitted package (manifest, curation JSON, write-up) is in [`submission/`](submission/). Its
[README](submission/README.md) is the honest account we handed to the judges.

## Repository layout

| Path | What's in it |
|---|---|
| [`STRATEGY.md`](STRATEGY.md) | The 3-hour plan: roles (table / sim+train / Cosmos+submission), timeline, blockers, key commands |
| [`NEXT-STEPS.md`](NEXT-STEPS.md) | Mid-event update checked against the live Nebius project (GPU, bucket, quotas) |
| [`runbooks/`](runbooks/) | Pre-event research notes on Cosmos, the GR00T repo and misc lookups |
| [`infra/`](infra/) | Nebius setup: GPU VM spec (`gpu-vm.yaml`), Object Storage AWS profile, `env.sh` fix for the starter repo on the VM |
| [`robot/`](robot/) | Robot-laptop tooling for the SO-101 leader/follower + two cameras (see below) |
| [`cosmos/reason_score.py`](cosmos/reason_score.py) | Scores clips with Cosmos Reason (OpenAI-compatible API) and writes the curation JSON |
| [`vm-artifacts/`](vm-artifacts/) | Pulled from the GPU VM: setup scripts, `env.sh` versions, logs, curation outputs, training config + log, a 2-episode sim smoke-test dataset (LeRobot v2.1) |
| [`submission/`](submission/) | What went into the bucket: `manifest.json`, `curation.json`, `README.md` |
| `nebius-starter-skill/` | Git submodule: the organizers' starter skill. Clone with `--recurse-submodules` |

## Reusing the pieces

**Robot laptop (Linux).** Fill in [`robot/robot.env`](robot/robot.env) first. Use the stable `/dev/serial/by-id`
and `/dev/v4l/by-id` paths, because `/dev/ttyACM0` and `/dev/video0` swap on replug.

```bash
cd robot
./robot-ops.sh setup        # LeRobot 0.6.1 + deps, serial access (once)
./robot-ops.sh ports        # find the arm ports
./robot-ops.sh cameras      # find the cameras, save a test frame each
./robot-ops.sh calibrate    # follower, then leader
./robot-ops.sh teleop       # live check, nothing recorded
./robot-ops.sh record 5     # record 5 more episodes (resumable)
./robot-ops.sh upload       # rsync to the GPU VM
./recal.sh                  # recalibrate both arms from scratch (backs up the old files)
```

`wrist_guard.py` shows the raw joint positions before calibrating. Every joint must be away from the encoder
wrap point, or `lerobot-calibrate` fails with `Magnitude ... exceeds 2047`. `session.sh` is the guided first
session. It calls a `which_leader.py` helper that isn't in this repo.

**GPU VM.** `vm-artifacts/scripts/setup_vm.sh` + `install_isaaclab.sh` install everything into `~/work`. Isaac Lab
is pinned to `24a2ebfd7` (it still has `mdp.image`). Then run `infra/vm-fix-env.sh` to point the starter repo at that
layout. The exact versions are in `vm-artifacts/config/versions.txt`.

**Cosmos Reason.** Run vLLM `nvidia/Cosmos-Reason2-2B` in Docker on the VM (port 8000, about 20% of the GPU), then
`cosmos/reason_score.py`. Send 4–8 frames per request as base64 `image_url`s. Ask for JSON only, then strip
`<think>` blocks and code fences before parsing.

> The project-specific values (VM IP `REDACTED_VM_IP`, bucket `REDACTED_BUCKET`, project `group31-project`
> in `uk-south2`) belonged to the event and probably no longer exist. No credentials are stored in this repo.

## Lessons learned

- **Calibrate and lock the real station first.** Bad calibration cost us the real data. Without real data the main
  (sim + Cosmos + real) run never started, so we only had the ablation to submit.
- **One GPU means everything runs in sequence.** Sim, Cosmos and training can't overlap, so plan the GPU timeline
  as the critical path.
- **Match sim and real exactly:** camera keys and resolution (`front` + `wrist`, 640×480 @ 30 fps), joint units, the
  task string and the cube color.
- **On Nebius images, port 5555 is taken** by NVIDIA DCGM (`nv-hostengine`). Serve GR00T on 5556.
- **Token Factory had no Cosmos model for our key.** Self-hosting Reason on the VM worked fine and fast.
- **Object Storage:** the Standard class had 0 quota. Use ENHANCED_THROUGHPUT.
- **3k steps of sim-only training doesn't solve the task in closed loop.** Next time: more steps, and real or
  Cosmos-transferred data in the mix.
