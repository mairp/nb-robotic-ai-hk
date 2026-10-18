# SO-101 sim pick-place -> GR00T runbook (research only; nothing was run)

Repo found: https://github.com/ddiddi/so101-sim-pickplace-gr00t (Isaac Lab 3.0 / Isaac Sim 6.1, **GR00T N1.7-3B**, not N1.5/N1.6; has a Nebius guide).
Other refs: NVIDIA course https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/ (code: https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop , its checkpoint name says "grootn16");
GR00T SO100/SO101 guide https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/SO100/README.md ;
https://github.com/NVIDIA/skills has NO SO-101/GR00T/Isaac Sim skill (only i4h-*, Cosmos TAO skills) - skip it.
Legend: [V] verified in source text I read; [E] my estimate/inference - verify.

## 0. Critical infra gotcha
Isaac Sim needs RT cores. Repo README: use L40S (`gpu-l40s-d`, eu-north1) or RTX PRO 6000 (`gpu-rtx6000`, us-central1); **NOT H100/H200** for sim [V].
=> Two VMs: (A) L40S for sim collection; (B) H100/H200 for training+serving. Or do everything on one L40S (README: 6000 steps batch 8 = 3-4 h, too slow for 1h budget).
Gated model: accept terms for `nvidia/Cosmos-Reason2-2B` on HF and `huggingface-cli login` before training [V].

## 1. Sim data generation
```bash
# VM A (L40S)  - nebius CLI: https://docs.nebius.com/cli/install
NAME=so101-sim PLATFORM=gpu-l40s-d PRESET=1gpu-16vcpu-96gb ./nebius/create_vm.sh   # 300GB ubuntu24.04-cuda13.0
ssh ubuntu@<ip>; curl -sL https://raw.githubusercontent.com/ddiddi/so101-sim-pickplace-gr00t/main/nebius/setup_vm.sh | bash
cd ~/work/so101_sim_pickplace
# edit env.sh: ISAACLAB_PY=$HOME/work/env_isaaclab/bin/python ISAACLAB_DIR=$HOME/work/IsaacLab GR00T_REPO=$HOME/work/Isaac-GR00T
# env.sh hardcodes /home/solotech007/... for PROJECT_DIR/DATASETS_DIR too -> fix those; drop the 2 DGX-Spark lines in gr00t_activate
source env.sh; export OMNI_KIT_ACCEPT_EULA=YES
NUM_EPISODES=3 ./collect.sh && ./prepare_dataset.sh        # smoke test (first run ~5 min asset download)
TASK="pick up the red block and place it on the green pad" NUM_EPISODES=150 ./collect.sh --resume   # or: ./collect.sh --num_episodes N
./prepare_dataset.sh                                        # copies modality.json, writes stats.json/relative_stats.json, loader check
```
- Episode count: `NUM_EPISODES` env (default 50) = *successful* episodes; failures are discarded; give-up at 3x attempts (`--max_attempts`). `--resume` appends. `--seed` (default 0) exists [V].
- Speed: ~25 s/attempt on DGX Spark, "~20 s per episode on L40S"; expert success ~75% => ~27 s per kept episode => 100 eps ~45 min, 150 ~70 min [E from V numbers]. Run 2-3 collectors in parallel with different `--seed` and `DATASET_NAME` (separate roots), merge later - parallelism is not built in [E].
- Randomization [V, pickplace_env_cfg.py L73-74]: cube x in (0.16,0.27) m, y in (0.02,0.13), yaw (-0.8,0.8) rad; **pad also randomized**: x (0.16,0.27), y (-0.15,-0.04). Cube is red 30 mm by default (`CUBE_SIZE_MM=30`, `CUBE_COLOR=red|green|blue|yellow`). Pad 90x90x4 mm green.
- **No built-in 3x3 grid.** Random uniform already covers it, but to bias coverage: edit `CUBE_POSE_RANGE` or the reset event (`pickplace_env_cfg.py` ~L262, `params={"pose_range": CUBE_POSE_RANGE ...}`) to sample from 9 measured centres + ~1 cm jitter. Minimal approach: measure your real taped spots relative to the robot base, then set x/y ranges to the bounding box of the 9 spots (+1-2 cm margin) and fix `PAD_POSE_RANGE` to the real pad location (x==y min==max). Otherwise policy learns a moving pad that the judges' fixed pad doesn't need. Check the sim frame: arm faces +X, base-frame origin is the seat [V comments]. Verify with `scripts/view_env.py --run_policy 600 --viz kit` / `logs/view/*.png`.
- Output [V]: **LeRobot v2.1**, `$DATASETS_DIR/local/so101_sim_pickplace` (DATASET_ROOT); 30 fps; videos h264 mp4 **640x480**; cameras `front` (fixed) + `wrist` -> keys `observation.images.front`, `observation.images.wrist`; `observation.state`/`action` 6-D `[shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos, gripper.pos]`, arm in **[-100,100]** (mapped from USD joint limits), gripper **[0,100]** (closed..open); ~300 frames/episode (10 s). state=measured pos, action=target at that frame.

## 2. GR00T data format / merging
- `meta/modality.json` [V, gr00t/modality.json]: state & action `single_arm` 0:5, `gripper` 5:6; video `front`->`observation.images.front`, `wrist`->`observation.images.wrist`; annotation `human.task_description`->`task_index`.
- Embodiment tag: `NEW_EMBODIMENT`; modality config `gr00t/so101_sim_config.py` (same as `examples/SO100/so100_config.py`): video [front,wrist], state [single_arm,gripper], action horizon 16, arm RELATIVE, gripper ABSOLUTE, language `annotation.human.task_description`. Horizon baked in at training.
- GR00T needs LeRobot **v2.x**. `lerobot-record` in new LeRobot writes **v3** -> convert: 
```bash
uv run --project scripts/lerobot_conversion python scripts/lerobot_conversion/convert_v3_to_v2.py --repo-id <hf_user>/<real_ds> --root <out>   # in Isaac-GR00T
cp $GR00T_REPO/examples/SO100/modality.json <out>/<hf_user>/<real_ds>/meta/modality.json
```
  [V, SO100 README]. Real recording must use cameras literally named `front` and `wrist`, 640x480@30, `so101_follower` (names `*.pos`).
- **Merging: no merge script is documented** (NVIDIA course says only "Merge this dataset with your simulation dataset"). Two options:
  1. *Preferred, zero-copy*: GR00T `launch_finetune.py` splits `--dataset-path` on `os.pathsep` (`:` on Linux) into multiple datasets [V, code line 59]; `--ds-weights-alpha` / `DS_WEIGHTS_ALPHA` env sets mixing weights [V]. Each dataset needs its own `meta/modality.json` + stats (run `gr00t/data/stats.py` on each). Untested by me: confirm in the log that all dirs load.
     `--dataset-path /data/sim:/data/cosmos:/data/real`
  2. Physical merge: `lerobot` has `lerobot.scripts.lerobot_edit_dataset --operation.type merge` (newer versions) [E, unverified]; or write your own concat renumbering episode_index/index/task_index in meta/episodes.jsonl, episodes_stats.jsonl, info.json.
- Cosmos: NVIDIA course ran Cosmos Transfer 2.5 with depth+edge+seg+vis controls, guidance 3; mixes tested: 75 sim + 7 Cosmos, 75 sim + 70 Cosmos; real = only 5 teleop eps [V: .../14-strategy3-cosmos.html, 13-strategy2-cotraining.html]. No commands given; see https://nvidia-cosmos.github.io/cosmos-cookbook/ . Cosmos output = re-rendered videos with same actions/states: copy the sim dataset dir, replace `videos/chunk-000/observation.images.{front,wrist}/episode_*.mp4` with the augmented ones (same frame count/fps/res!), keep parquet identical [E].

## 3. Fine-tune
Repo wrapper (`finetune.sh`) = `examples/finetune.sh` with `--no-tune-llm --tune-visual --tune-projector --tune-diffusion-model`, 4000 steps, batch 16, ckpt every 1000, `--save-only-model`, lr 1e-4, warmup 0.05, color-jitter default on [V]. Base: `nvidia/GR00T-N1.7-3B`.
```bash
# VM B (H100/H200). cd $GR00T_REPO ; uv env as per Isaac-GR00T README (https://github.com/NVIDIA/Isaac-GR00T)
export NUM_GPUS=1 USE_WANDB=0 MAX_STEPS=5000 SAVE_STEPS=1000 GLOBAL_BATCH_SIZE=32 DATALOADER_NUM_WORKERS=8
M=/path/so101_sim_config.py          # from the repo's gr00t/ dir
# MAIN: sim + cosmos + real
CUDA_VISIBLE_DEVICES=0 uv run bash examples/finetune.sh --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /data/sim:/data/cosmos:/data/real --modality-config-path $M --embodiment-tag NEW_EMBODIMENT \
  --output-dir /ckpt/main --save-only-model -- --no-tune-llm --tune-visual --tune-projector --tune-diffusion-model
# ABLATION: sim + real only (no cosmos)  -> same, --dataset-path /data/sim:/data/real --output-dir /ckpt/abl
```
- Wall time: only data point is ~2 s/step at batch 16 on GB10 (DGX Spark) [V]; generic NVIDIA H100 run was 2000 steps batch 32 (no time given) [V]. H100/H200 estimate **~0.5-1 s/step at batch 16-32 => 5000 steps ~45-80 min** [E - benchmark first 50 steps and size MAX_STEPS = 50 min / s-per-step]. Memory ~25-40 GB, so **both trainings can run concurrently on one 80GB H100/H200** (`CUDA_VISIBLE_DEVICES=0`, different `MASTER_PORT`, `--output-dir`) but each slows ~2x; with 2 GPUs use one each. Budget: 3h hackathon => run both with MAX_STEPS ~3000-4000 concurrently, or a 2-GPU VM.
- Real episodes are few (<=50): upweight via `DS_WEIGHTS_ALPHA` or duplicate dataset path entries [E]. Use `--save-steps 1000` and pick checkpoint via `gr00t/eval/open_loop_eval.py --dataset-path <real> --model-path <ckpt> --embodiment-tag NEW_EMBODIMENT --traj-ids 0 --execution-horizon 16 --steps 400` [V].

## 4. Deployment
- Server [V]: `uv run python gr00t/eval/run_gr00t_server.py --model-path <ckpt> --embodiment-tag NEW_EMBODIMENT --modality-config-path $M --device cuda --host 0.0.0.0 --port 5555`. Protocol: ZeroMQ REQ/REP + msgpack(numpy), **TCP 5555** (repo's `Gr00tClient` re-implements it; has optional `api_token`).
- Client on robot laptop [V]: `Isaac-GR00T/gr00t/eval/real_robot/SO100/eval_so100.py` (setup: `cd gr00t/eval/real_robot/SO100 && uv sync && uv pip install --no-deps -e ../../../../`):
```bash
uv run --no-sync python eval_so100.py --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=<id> \
 --robot.cameras="{ front: {type: opencv, index_or_path: $FRONT, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: $WRIST, width: 640, height: 480, fps: 30}}" \
 --policy_host=<VM_IP or localhost> --policy_port=5555 --lang_instruction="pick up the red block and place it on the green pad"
```
  Client sends front+wrist images and 6-D state, executes first `action_horizon` (default 8; max 16) actions of each chunk at 30 Hz (`sleep(1/30)`).
- Remote server works in principle (plain TCP). Safest: `ssh -N -L 5555:localhost:5555 ubuntu@<vm>` and `--policy_host=localhost` (avoids opening a firewall/security-group port; ZMQ port has no auth otherwise). Latency: images 2x640x480 per request + ~50-150 ms network RTT + ~50-100 ms inference on H100 (4 denoising steps) [E]. Because actions are chunked (8 steps = 267 ms executed per query), a few hundred ms is tolerable but motion may pause between chunks. Fallback: copy checkpoint to laptop GPU and serve locally (README "Datasets and Models"/course does localhost). Ping the VM region from the venue first; Nebius eu-north1 may be 50-150 ms from wherever the venue is.
- Repo's `eval.sh` (`scripts/eval_in_sim.py`) does closed-loop sim eval over the same server: `./eval.sh --num_episodes 20` - needs RT-core GPU, so run on VM A with tunnel to VM B.

## 5. Pitfalls
1. **Units**: sim dataset uses LeRobot "normalized" `use_degrees=False` convention: arm [-100,100], gripper [0,100]. Real client must read the same (`so101_follower` default, i.e. NOT `--robot.use_degrees=true`). Sim mapping uses USD joint limits (`so101_units.py`), real maps via *your calibration* min/max ranges: the same number means a different angle in sim vs real unless calibration ranges match. Sim and real same-valued commands can differ by several degrees per joint -> do `lerobot-calibrate` carefully through full range (course: .../07-calibrating-so101.html) and compare a pose: command sim and real arm to `[0,0,0,0,0]` and `[50,...]` and check visually. Sim data are also "joint targets" from IK with perfect tracking; real follower lags (the course's SAGE/GapONet strategy addresses this - skip in 3h).
2. **Normalization**: GR00T normalizes with dataset stats; with merged datasets stats are per-dataset in the multi-path mode but one shared embodiment's stats at inference come from checkpoint (`statistics.json`). Sim-only stat range vs real can mismatch; always fine-tune on some real data; check `stats.json` min/max of gripper (sim 0-100 vs real maybe 0-40ish; check real range after calibration).
3. **Gripper**: sim open = 1.745 rad, close = -0.1745 rad mapped to 100/0 [V]; real gripper rarely reaches 100 and a 30 mm cube stops closing at ~ mid value (action commands lower than state while grasping). Gripper action is ABSOLUTE in the config - verify real leader's range when open/closed vs sim numbers; record a real episode and compare histograms of `action[:,5]`.
4. **Cameras**: real front cam must replicate sim `FRONT_CAM_POS/LOOK_AT` (`pickplace_env_cfg.py` L107; focal 18 mm, aperture 20.955 mm, 640x480) and wrist cam on the lateral side of the gripper looking down jaws (L101). Measure the sim camera pose with `view_env.py` and mount a webcam at the same place; overlay `docs/front_cam.png` on the live preview. Keys must be `front`,`wrist` and image size 640x480. Use the identical lighting/table color if possible; Cosmos only fixes appearance, not pose.
5. Sim cube is **red** and pad **green**, prompt says "red block"; change `CUBE_COLOR`/`TASK` and use the identical instruction string in training and at inference.
6. Randomized pad in sim vs fixed pad on the real table (see section 1).
7. 16-step action horizon is fixed at training; N1.7 weights/CLI differ from N1.5/N1.6 (older HF blog https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning). Do not mix GR00T versions between training and server.
8. `env.sh` has hard-coded `/home/solotech007/...` paths; `collect.sh` pipes via tee, `set -e` may hide failures - check `logs/collect.log`.
9. Expert success ~75%; only successes stored. Verify a couple of mp4s visually.

## Sources
- https://github.com/ddiddi/so101-sim-pickplace-gr00t (README, env.sh, collect.sh, finetune.sh, start_server.sh, prepare_dataset.sh, gr00t/*, so101_pickplace/*)
- https://github.com/NVIDIA/Isaac-GR00T (examples/SO100/README.md, examples/finetune.sh, gr00t/experiment/launch_finetune.py, getting_started/finetune_new_embodiment.md, gr00t/eval/real_robot/SO100/eval_so100.py)
- https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/ (pages 13, 14, quick_reference)
- https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop
- https://github.com/NVIDIA/skills (nothing relevant)
- https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning ; https://nvidia-cosmos.github.io/cosmos-cookbook/
