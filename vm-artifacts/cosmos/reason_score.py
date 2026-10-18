#!/usr/bin/env python3
"""Score pick-and-place clips with Cosmos Reason and write curation.json.

Default backend: our own vLLM server on so101-gpu (nvidia/Cosmos-Reason2-2B, served as `cosmos-reason2`,
bound to 127.0.0.1:8000). Token Factory lists no Cosmos model for our key (checked 2026-10-18).

    python3 reason_score.py --videos 'datasets/local/so101_sim_pickplace/videos/chunk-000/observation.images.front/*.mp4' \
        --source sim --out curation.json [--limit 3]
    # other OpenAI-compatible backend: --base-url URL --model ID, key in $REASON_API_KEY

Needs ffmpeg/ffprobe. Re-running skips clips already in --out, so it is safe to resume.
Stated curation rule (keep): task_success AND cube_and_gripper_visible AND overall_score >= MIN_SCORE.
"""
import argparse, base64, glob, json, os, re, subprocess, sys, tempfile, time, urllib.error, urllib.request

BASE_URL = "http://127.0.0.1:8000/v1"
MODEL = "cosmos-reason2"
MIN_SCORE = 6
PROMPT = (
    "These are {n} frames, in time order, from one episode of a SO-101 robot arm. "
    "Task: pick up the red cube and place it on the green pad, then release it.\n"
    "Answer with JSON only, no prose:\n"
    '{{"task_success": true|false, "cube_and_gripper_visible": true|false, '
    '"released_on_pad": true|false, "overall_score": <0-10 visual realism and demo quality>, '
    '"issues": "<short>"}}'
)


def sample_frames(video, n):
    dur = float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video]).strip())
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(n):
            t = dur * (i + 0.5) / n
            out = os.path.join(tmp, f"{i}.jpg")
            subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", video, "-frames:v", "1",
                            "-vf", "scale=640:-2", "-q:v", "4", out], check=True)
            with open(out, "rb") as f:
                frames.append(base64.b64encode(f.read()).decode())
    return frames


def ask(frames, base_url, model, key, retries=5):
    content = [{"type": "text", "text": PROMPT.format(n=len(frames))}]
    content += [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b}} for b in frames]
    body = json.dumps({"model": model, "temperature": 0.2, "max_tokens": 2048,
                       "messages": [{"role": "user", "content": content}]}).encode()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    for attempt in range(retries):
        req = urllib.request.Request(base_url + "/chat/completions", data=body, method="POST", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r)["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(float(e.headers.get("Retry-After") or 2 ** (attempt + 1)))
                continue
            raise


def parse(text):
    text = text.rsplit("</think>", 1)[-1]                      # drop visible reasoning
    m = re.search(r"\{.*\}", text, re.S)                       # strip ```json fences / prose
    return json.loads(m.group(0)) if m else {"parse_error": text[:300]}


def keep(s):
    return bool(s.get("task_success") and s.get("cube_and_gripper_visible")
                and (s.get("overall_score") or 0) >= MIN_SCORE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="glob of mp4 clips")
    ap.add_argument("--source", required=True, help="label stored per clip: sim | cosmos | real")
    ap.add_argument("--out", default="curation.json")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N clips (smoke test)")
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    key = os.environ.get("REASON_API_KEY", "")

    cur = json.load(open(a.out)) if os.path.exists(a.out) else {"model": a.model, "rule": None, "clips": {}}
    cur["rule"] = f"keep if task_success and cube_and_gripper_visible and overall_score >= {MIN_SCORE}"
    videos = sorted(glob.glob(a.videos))[: a.limit or None]
    for v in videos:
        if v in cur["clips"]:
            continue
        t0 = time.time()
        s = parse(ask(sample_frames(v, a.frames), a.base_url, a.model, key))
        s.update(source=a.source, keep=keep(s), latency_s=round(time.time() - t0, 1))
        cur["clips"][v] = s
        json.dump(cur, open(a.out, "w"), indent=1)              # checkpoint after every clip
        print(f"{'KEEP' if s['keep'] else 'DROP'}  {os.path.basename(v)}  {s.get('overall_score')}  {s.get('issues', '')}")

    clips = cur["clips"].values()
    m, kept = len(cur["clips"]), sum(c["keep"] for c in clips)
    cur["summary"] = f"{m - kept} of {m} dropped ({cur['rule']})"
    json.dump(cur, open(a.out, "w"), indent=1)
    print(cur["summary"])


if __name__ == "__main__":
    main()
