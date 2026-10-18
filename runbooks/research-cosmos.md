# Cosmos Transfer 2.5 + Cosmos Reason runbook: SO-101 sim-to-real (hackathon)

Researched 2026-10-18. Some facts are confirmed in docs; others are my inference and are labelled "(unverified)".
Context: both `cosmos-transfer2.5` and `cosmos-reason2` GitHub repos now say they get only limited maintenance and point to Cosmos 3.
Use whatever the organizers host. To find out what that is, call `GET /v1/models` (Reason) and `GET /openapi.json` or the docs (Transfer).

---

## 0. Time plan (about 40 min for Transfer and about 20 min for Reason)

| Step | Time |
|---|---|
| Smoke test: 1 clip through Transfer at 480p, edge only | 5-10 min |
| Batch Transfer: N clips (see budget below) | ~30 min |
| Rebuild the dataset with the new videos | 10 min |
| Reason: score every clip (sim originals plus augmented), save JSON, apply the rule | 15-20 min |

---

## 1. Cosmos Transfer 2.5

### 1.1 Request shape (NIM `POST /v1/infer`)
Source: https://docs.nvidia.com/nim/cosmos/latest/api-reference.html and
https://docs.nvidia.com/nim/cosmos/latest/sampling-params.html

```json
{
  "prompt": "<style/scene description, < ~300 words>",
  "negative_prompt": "<optional; has a CG/game-look default>",
  "video": "<base64 mp4 OR URL>",
  "image_context": "<optional base64/URL photo of the REAL station -> style reference>",
  "resolution": "480",            // "256" | "480" | "512" | "720"; default "480"
  "guidance": 3,                  // int 0-7, default 3
  "num_steps": 35,                // default 35; fewer = faster, lower quality
  "seed": 42,
  "num_conditional_frames": 1,    // 1 or 2
  "edge": {
    "control_weight": 1.0,        // 0.0-1.0, default 1.0
    "preset_edge_threshold": "medium"   // very_low|low|medium|high|very_high
    // "control": "<b64/URL edge mp4>"  -> OMIT the key to auto-extract (do not set null/"")
    // "mask": "<b64/URL binary mp4>"   -> white = apply control
  }
  // optional extras: "depth": {...}, "vis": {"control_weight":..,"preset_blur_strength":..},
  // "seg": {"control_weight":..,"control_prompt":"robot arm cube pad"}
}
```
The response is `{"b64_video": "...", "seed": 42}`. Decode it with `base64.b64decode` and write it to an `.mp4`.

Constraints from the docs:
- The input mp4 must have **93 to 480 frames**.
- The **output resolution matches the input resolution**. The `resolution` field only sets the internal processing resolution.
- Height and width must be **multiples of 8**.
- Every request needs at least one control type: edge, depth, vis or seg.
- If you have only one control, its weight is used as given. If several weights add up to more than 1.0, they are scaled down in proportion.
- If a video passed as a URL times out, send it base64-encoded instead (NIM tip).

The build.nvidia.com model card adds these points (https://build.nvidia.com/nvidia/cosmos-transfer2_5-2b):
- The prompt should be under 300 words.
- The model works in 93-frame chunks, so lengths of **93, 186 or 279 frames work best**.
- Output is at **16 FPS**.
- Known weaknesses: temporal inconsistency and long or high-resolution clips.

Repo CLI (if you self-host): https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/inference.md
It runs as `python examples/inference.py -i spec.json -o out/`. In the spec, edge is set with `{"edge": {"control_path": ..., "control_weight": 1.0}}`.
`--model=edge/distilled` with `num_steps: 4` selects a distilled edge model, and it is much faster.

Hosted or OpenAI-style alternative: the Nebius Token Factory docs list no Cosmos model, and there is no `/v1/infer` there.
https://docs.tokenfactory.nebius.com/llms.txt
If the organizers' endpoint is Nebius, ask them for the exact path. It is probably a dedicated endpoint wrapping the NIM (unverified).

### 1.2 Python client
```python
import base64, json, pathlib, requests, time
URL = "http://<host>:8000/v1/infer"          # organizer endpoint
HDR = {"Content-Type": "application/json"}   # + {"Authorization": f"Bearer {KEY}"} if hosted

def b64(p): return base64.b64encode(pathlib.Path(p).read_bytes()).decode()

def transfer(src_mp4, out_mp4, prompt, style_img=None, seed=1, w=1.0):
    body = {"prompt": prompt, "negative_prompt": NEG, "video": b64(src_mp4),
            "resolution": "480", "guidance": 3, "seed": seed,
            "edge": {"control_weight": w, "preset_edge_threshold": "medium"}}
    if style_img: body["image_context"] = b64(style_img)
    for attempt in range(5):
        r = requests.post(URL, headers=HDR, json=body, timeout=1800)
        if r.status_code in (429, 502, 503, 504):
            time.sleep(int(r.headers.get("Retry-After", 2 ** attempt * 5))); continue
        r.raise_for_status()
        pathlib.Path(out_mp4).write_bytes(base64.b64decode(r.json()["b64_video"]))
        return r.json()["seed"]
    raise RuntimeError("gave up")
```

### 1.3 Edge conditioning: keeping the robot geometry and motion
- **Use edge only, at `control_weight` 0.9-1.0.** Edge is the strongest structure lock: arm links, gripper jaws, cube outline and pad outline all keep their position frame by frame.
  Below about 0.7 the model starts to invent geometry. NVIDIA's doc note says high weights suit precision use cases and low weights allow "creative freedom".
- **Optional second control: `vis` (blurred RGB) at 0.2-0.3.** It keeps the sim's coarse colours and layout, so the cube stays the same colour. Skip it if you want the strongest change of look.
  `depth` at about 0.3 also helps keep 3-D shape, but it costs time.
- **Edge threshold.** The default is `medium`. Use `high` if the sim render is cluttered and its texture edges pin the sim look in place.
  Use `low` or `very_low` if thin parts (gripper fingers) lose their edges.
- **Extracting the edge video yourself** (optional; the NIM extracts it automatically if you omit `control`). Doing it yourself lets you inspect it and mask out the background:
```python
import cv2, numpy as np
def canny_video(src, dst, lo=100, hi=200):
    cap = cv2.VideoCapture(src); fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    while True:
        ok, f = cap.read()
        if not ok: break
        g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        e = cv2.Canny(g, lo, hi)
        out.write(cv2.cvtColor(e, cv2.COLOR_GRAY2BGR))
    out.release()
```
  Check that the edge frames clearly show the gripper and the cube before you spend GPU time on them.
  Pro move: send a `mask` that is white on the robot, cube and pad and black on the background. Then the robot is locked and the background is free to change to the real station.
- **Writing the prompt** (describe the REAL station, and also the robot and objects so they stay):
  > "A real-world photograph-quality video of a low-cost SO-101 robot arm (white 3D-printed plastic links, black servo motors, two-finger gripper) mounted on a light wooden table in a lab. The arm reaches for a small red cube, grasps it, lifts it and places it on a flat blue square pad. Soft overhead fluorescent lighting, slight shadows under the arm and cube, matte table texture, plain grey wall in the background, static camera, realistic colors, sharp focus, natural sensor noise."
  - Use the colours, materials, lighting and background of the actual station. Check them against a phone photo.
  - Mention every object that must persist: arm, gripper, cube, pad. Say "static camera".
  - **Style reference:** set `image_context` to a photo of the real station taken from about the same viewpoint.
    It is in the NIM schema, but I have not tested how strongly it acts (unverified).
  - Negative prompt: add hallucination terms to the default, for example
    `"extra objects, duplicate cubes, missing gripper, deformed robot arm, floating objects, flickering, texture popping, cartoon, CG render, video game"`.
- **Diversity for augmentation.** Use 2-3 prompt variants (lighting warm/cool, table colour, background clutter) times different seeds.
  Keep the edge input, and therefore the motion, the same.

### 1.4 Latency and budget
Published numbers:
- Repo, 1 GPU, edge, average diffusion time per 93-frame chunk at 720p (https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/inference.md):

  | Model | B200 | H100 NVL | H200 NVL |
  |---|---|---|---|
  | Base | ~180 s | ~503 s | ~374 s |
  | Distilled (4 steps) | ~24 s | ~65 s | ~50 s |

- End-to-end for 121 frames (2 chunks) on B200: ~187 s.
- The model needs about 65 GB VRAM. The NIM needs Hopper or newer with 80 GB or more.

Assumptions for a shared hosted endpoint at 480p: **about 2-5 min per 93-frame clip**. That gives **about 8-15 clips sequentially in 40 min**. More fit only if the endpoint accepts concurrent requests (throughput profile).

Plan:
- Trim or subsample each episode to **93 frames** (one chunk). For example, take a 30 fps episode of about 6 s and keep every 2nd frame. Or cut out the grasp and place window.
- Run 2-3 requests at once and back off on 429 or 5xx.
- Time the first request and recompute N from it.

---

## 2. Putting augmented videos back into a LeRobot dataset

Facts:
- The Cosmos output has the **same resolution and length as the input** (per the docs). It may be written at **16 fps**.
- LeRobot pairs frame i with action/state row i by `timestamp = i / fps`. Rewrite or rebuild the video so that it has **exactly `episode_length` frames at the dataset fps**.
- Format v3.0 (lerobot >= 0.4) packs many episodes into `videos/<key>/chunk-XXX/file-XXX.mp4`, and `meta/episodes/` stores per-episode offsets.
  Format v2.1 has one `videos/chunk-000/<key>/episode_XXXXXX.mp4` per episode.
  https://huggingface.co/docs/lerobot/lerobot-dataset-v3

**Safest approach, which works with either version:** build a *new* dataset.
Copy every non-video feature from the source, and feed decoded Cosmos frames as the image.
This also keeps the original sim dataset intact, so you can train on both.

```python
import av, cv2, numpy as np
from lerobot.datasets import LeRobotDataset      # older lerobot: lerobot.common.datasets.lerobot_dataset

SRC, DST, CAM = "local/sim_pnp", "local/sim_pnp_cosmos", "observation.images.front"
src = LeRobotDataset(SRC)
dst = LeRobotDataset.create(DST, fps=src.fps, features=src.meta.features,
                            robot_type=src.meta.robot_type, use_videos=True)

def read_frames(mp4):
    with av.open(mp4) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])

for ep, mp4 in cosmos_outputs.items():                 # {episode_index: path}, kept clips only
    s, e = src.meta.episodes["dataset_from_index"][ep], src.meta.episodes["dataset_to_index"][ep]  # v3; v2.1: episode_data_index
    T = e - s
    frames = read_frames(mp4)
    H, W = src.meta.features[CAM]["info"]["video.height"], src.meta.features[CAM]["info"]["video.width"]
    if frames.shape[1:3] != (H, W):                            # should match; resize if not
        frames = np.stack([cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) for f in frames])
    if len(frames) != T:                                       # e.g. trimmed/subsampled for Cosmos
        idx = np.round(np.linspace(0, len(frames) - 1, T)).astype(int)
        frames = frames[idx]                                    # nearest-frame time alignment
    for i in range(T):
        row = src.hf_dataset[s + i]
        frame = {k: row[k] for k in src.meta.features
                 if k not in ("index", "episode_index", "frame_index", "timestamp", "task_index")
                 and src.meta.features[k]["dtype"] not in ("video", "image")}
        frame[CAM] = frames[i]                                  # HWC uint8
        frame["task"] = src.meta.tasks.index[row["task_index"]] if hasattr(src.meta.tasks, "index") else src.meta.tasks[row["task_index"]]
        dst.add_frame(frame)
    dst.save_episode()
dst.finalize()
```
- Check the exact attribute names against your installed lerobot version. `episode_data_index` and `meta.tasks` changed between 2.1 and 3.0.
- If you **trimmed** the episode before running Cosmos, also trim the action/state rows to the same window. Do not stretch 93 frames over the full episode.
  Better: subsample uniformly so that the 93 frames span the whole episode, and then the `linspace` mapping above is correct.
- Checks after the rebuild:
  - `len(dst) == len(src)` for the kept episodes.
  - `dst.meta.features[CAM]["info"]["video.fps"] == src.fps`.
  - Open 2 episodes in `lerobot-dataset-viz` and check that the gripper closes when the action says it closes.
- Multi-camera: augment every camera with the **same prompt and seed**, or keep the other cameras as the original sim views. Otherwise the views will not agree with each other.

---

## 3. Cosmos Reason: scoring and curation

### 3.1 Model and call
- Current models: `nvidia/Cosmos-Reason2-2B` (24 GB GPU or more), `-8B` (32 GB or more) and `-32B`. They are based on Qwen3-VL and served with vLLM >= 0.11 through the OpenAI-compatible API.
  https://github.com/nvidia-cosmos/cosmos-reason2 , https://huggingface.co/nvidia/Cosmos-Reason2-8B
- build.nvidia.com now sends the Reason2 page to **Cosmos 3 Nano Reasoner** (also vLLM, mp4 at 4 fps).
  Use whichever model id the organizers' `/v1/models` lists.
- Nebius Token Factory: base URL `https://api.tokenfactory.nebius.com/v1/` with a Bearer `NEBIUS_API_KEY`.
  Its docs show only `image_url` and do not list Cosmos (https://docs.tokenfactory.nebius.com/api-reference/examples/vision-capabilities).
  Call `client.models.list()` first.
- **Use about 4 fps input**, which matches the training setup. Use **max_tokens 4096 or more** if reasoning is on.
- In NVIDIA's client, the **video goes before the text** in the user content. Video is sent as `{"type":"video_url","video_url":{"url":...}}` with `extra_body={"mm_processor_kwargs":{"fps":4,"do_sample_frames":True}}`.
  Source: https://github.com/nvidia-cosmos/cosmos-reason2/blob/main/cosmos_reason2_utils/cosmos_reason2_utils/script/inference.py
- Per-request mm kwargs need the server to run with `--trust-request-mm-kwargs`. If the server rejects them, just drop `extra_body`.
  https://docs.vllm.ai/en/latest/features/multimodal_inputs.html

```python
import base64, json, os, re, pathlib
from openai import OpenAI
client = OpenAI(base_url=os.environ["REASON_BASE_URL"], api_key=os.environ["REASON_API_KEY"])
MODEL = os.environ.get("REASON_MODEL") or client.models.list().data[0].id

def score(mp4, task="pick up the cube and place it on the pad"):
    url = "data:video/mp4;base64," + base64.b64encode(pathlib.Path(mp4).read_bytes()).decode()
    msgs = [{"role": "system", "content": "You are a strict robotics video quality inspector."},
            {"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": url}},
                {"type": "text", "text": PROMPT.format(task=task)}]}]
    kw = dict(model=MODEL, messages=msgs, max_tokens=4096, temperature=0.2)
    try:
        r = client.chat.completions.create(**kw, response_format={"type": "json_schema",
              "json_schema": {"name": "clip_score", "schema": SCHEMA}},
              extra_body={"mm_processor_kwargs": {"fps": 4, "do_sample_frames": True}})
    except Exception:
        r = client.chat.completions.create(**kw)           # fallback: no schema / mm kwargs
    txt = r.choices[0].message.content
    txt = txt.split("</think>")[-1]                         # strip reasoning if inline
    return json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
```
If the endpoint refuses `video_url` data URIs (the upload is too big, or video is unsupported):
- Shrink the clip to about 480p and 4 fps with ffmpeg (`ffmpeg -i in.mp4 -vf fps=4,scale=640:-2 out.mp4`).
- Or send about 8-16 extracted JPEG frames as multiple `image_url` parts, and say in the prompt that they are "frames in temporal order".

### 3.2 Prompt
```
You are evaluating a short video of an SO-101 robot arm doing the task: "{task}".
The video may be a real recording, a simulation, or an AI-augmented (style-transferred) simulation.
Watch the whole clip and judge:
1. task_success: does the arm grasp the cube, lift it, and release it so it rests ON the pad at the end?
2. cube_on_pad_final: in the last frames, is the cube on the pad?
3. visual_artifacts: list any of: object appears/disappears, cube teleports or duplicates, gripper or arm links vanish/deform/merge,
   hallucinated extra objects, texture flicker, color popping, blur, geometry warping. Empty list if none.
4. robot_realism (0-10): does the arm look like a physical robot with consistent geometry and plausible motion?
5. visual_realism (0-10): does the scene look like a real camera recording (lighting, materials, noise)?
6. overall_score (0-10): fitness as a TRAINING clip; heavily penalize any artifact touching the gripper or cube.
Answer ONLY with a JSON object matching this schema, no prose:
{"task_success": bool, "cube_on_pad_final": bool, "gripper_visible_throughout": bool,
 "cube_visible_throughout": bool, "visual_artifacts": [string], "temporal_flicker": "none|minor|severe",
 "robot_realism": int, "visual_realism": int, "overall_score": int, "rationale": string (<=40 words)}
```
(Reason2 reasoning mode: NVIDIA appends "Answer the question using the following format: <think>…</think> Write your final answer immediately after the </think> tag." Keep reasoning OFF for speed and use JSON only. Turn it on only if the scores look random.)

### 3.3 JSON to save (`reason_scores.jsonl`, one line per clip)
```json
{
  "clip_id": "ep012_cosmos_s1_promptA",
  "source": "cosmos_transfer|sim|real",
  "episode_index": 12,
  "video_path": "out/ep012_s1.mp4",
  "model": "<model id>",
  "fps_sampled": 4,
  "prompt_version": "v1",
  "result": {
    "task_success": true, "cube_on_pad_final": true,
    "gripper_visible_throughout": true, "cube_visible_throughout": true,
    "visual_artifacts": [], "temporal_flicker": "minor",
    "robot_realism": 8, "visual_realism": 7, "overall_score": 8,
    "rationale": "..."
  },
  "decision": "keep",
  "decision_rule": "drop if overall_score<6 OR task_success=false OR cube/gripper not visible throughout OR flicker=severe",
  "timestamp": "2026-10-18T..."
}
```
Schema for `response_format` (SCHEMA): `type: object` with the fields above as `required`. Integers are `minimum 0, maximum 10`. `temporal_flicker` is an `enum`.

### 3.4 The ONE curation decision (state it like this)
> **Rule:** keep an augmented episode in the training set only if Cosmos Reason reports `task_success == true` AND
> `cube_visible_throughout && gripper_visible_throughout` AND `overall_score >= 6`. Every other episode is dropped.
> Reason: an augmented clip whose visuals no longer match the unchanged action labels (the cube vanishes, or the gripper deforms) teaches the policy wrong visuomotor pairs.

Report it as "X of N clips kept, Y dropped (breakdown by reason)" and save the JSONL plus the kept list.
Sanity check: also score 2-3 original sim clips. They should come out `task_success=true` with high robot_realism.
If they do not, the judge or prompt is miscalibrated. Fix the prompt before trusting the rule.
NVIDIA's own critic recipe (https://docs.nvidia.com/cosmos/latest/reason1/video_critic.html) scores each video several times (4) and uses the fraction of artifact-free verdicts.
Use 2-3 calls per clip and take the majority or mean if time allows, because scores at temperature > 0 are noisy.

---

## 4. Pitfalls

| Problem | Symptom | Mitigation |
|---|---|---|
| Hallucinated objects | Extra cubes, tools or cables show up on the table | Edge weight about 1.0. Name the scene contents exactly in the prompt. Add negative prompt terms. Use a mask on the background region if allowed. Reason drops what is left. |
| Gripper or cube disappears or merges | Thin fingers and the small cube lose edges; the cube changes colour | `preset_edge_threshold` low. Add `vis` at about 0.2 to keep the cube colour. State the cube colour and the gripper in the prompt. Higher input resolution, or crop closer. |
| Temporal flicker / chunk seams | Texture popping, a jump at frame 93 | Use exactly 93-frame clips (one chunk). Static camera. Keep guidance at the default of 3. Higher guidance gives more artifacts. |
| fps / length mismatch | Augmented frames drift from actions | Output may be 16 fps. Always count decoded frames and remap to episode length. Never trust container fps. |
| Input rejected | 400 errors | Need 93-480 frames, H and W multiples of 8, mp4. Do not send `"control": null` (omit the key). |
| URL timeouts | Request hangs, then fails | Send base64 (NIM tip). |
| Rate limits (Nebius) | HTTP 429 | Default about 60 RPM / 400k TPM; it rises 1.2x per 15 min of high use. Read `x-ratelimit-remaining-*` and `Retry-After`. Use exponential backoff. Long video requests are token-heavy, so keep 4 fps and 480p. https://docs.tokenfactory.nebius.com/ai-models-inference/rate-limits |
| Transfer queue | Each request takes minutes | Use a small concurrency (2-3), a 30-min client timeout and idempotent output names, so you can resume. Log the seed. |
| JSON from Reason | Prose or `<think>` around the JSON | Use `response_format` json_schema when supported. Otherwise split on `</think>` and regex-extract `{...}`. Retry once on a parse failure. |
| Multi-camera inconsistency | Wrist and front views disagree | Use the same prompt and seed per episode across cameras, or augment only the main camera. |

## Sources
- Cosmos Transfer 2.5 repo and inference doc: https://github.com/nvidia-cosmos/cosmos-transfer2.5 , https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/inference.md
- NIM for Cosmos API reference: https://docs.nvidia.com/nim/cosmos/latest/api-reference.html
- NIM sampling params (Transfer 2.5 schema, control_weight, image_context): https://docs.nvidia.com/nim/cosmos/latest/sampling-params.html
- NIM supported models/GPUs: https://docs.nvidia.com/nim/cosmos/latest/support-matrix.html
- Transfer 2.5 model card: https://build.nvidia.com/nvidia/cosmos-transfer2_5-2b
- Cosmos Reason2 repo and model card: https://github.com/nvidia-cosmos/cosmos-reason2 , https://huggingface.co/nvidia/Cosmos-Reason2-8B
- Cosmos Reason video critic (rejection sampling): https://docs.nvidia.com/cosmos/latest/reason1/video_critic.html
- vLLM multimodal inputs: https://docs.vllm.ai/en/latest/features/multimodal_inputs.html
- Nebius Token Factory: https://docs.tokenfactory.nebius.com/llms.txt , vision: https://docs.tokenfactory.nebius.com/api-reference/examples/vision-capabilities , JSON: https://docs.tokenfactory.nebius.com/ai-models-inference/json , rate limits: https://docs.tokenfactory.nebius.com/ai-models-inference/rate-limits
- LeRobot dataset v3: https://huggingface.co/docs/lerobot/lerobot-dataset-v3
- Cosmos 3 (successor): https://github.com/NVIDIA/Cosmos
