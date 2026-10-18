#!/usr/bin/env bash
# Recalibrate BOTH arms from scratch (old files backed up), then a live leader check that the gripper spans 0..100.
set -euo pipefail
cd "$(dirname "$0")"
source robot.env
PY="$VENV/bin/python"; LR="$VENV/bin"
CAL=$HOME/.cache/huggingface/lerobot/calibration
BK=$HOME/calibration_backup_$(date +%H%M%S)
mkdir -p "$BK"; cp -r "$CAL" "$BK/" 2>/dev/null || true
rm -f "$CAL/robots/so_follower/$FOLLOWER_ID.json" "$CAL/teleoperators/so_leader/$LEADER_ID.json"
echo "Old calibration backed up to $BK"

echo; echo "=== FOLLOWER ($FOLLOWER_ID) ==="
echo "Put the follower in its MIDDLE pose: every joint half-way, GRIPPER HALF OPEN. Joints marked ! sit at the encoder wrap point: turn them a bit."
"$PY" wrist_guard.py "$FOLLOWER_PORT"
echo "Then: ENTER at the middle pose -> move EVERY joint end to end (gripper fully open AND fully closed) -> ENTER."
"$LR/lerobot-calibrate" --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID"

echo; echo "=== LEADER ($LEADER_ID) ==="
echo "Put the leader in its MIDDLE pose: every joint half-way, TRIGGER/GRIPPER HALF-WAY (hold it there, don't let it spring to an end)."
"$PY" wrist_guard.py "$LEADER_PORT"
echo "Then: ENTER at the middle pose -> move EVERY joint end to end: trigger FULLY open AND FULLY closed, shoulder pan FULLY left AND right -> ENTER."
"$LR/lerobot-calibrate" --teleop.type=so101_leader --teleop.port="$LEADER_PORT" --teleop.id="$LEADER_ID"

echo; echo "=== Saved ranges (span in ticks; a joint whose min/max sits at ~2047 was only swept to one side) ==="
"$PY" - "$CAL" <<'PY'
import json, sys, glob
for f in sorted(glob.glob(sys.argv[1] + "/*/*/*.json")):
    print(f.split("calibration/")[1])
    for k, v in json.load(open(f)).items():
        span = v["range_max"] - v["range_min"]
        flag = "  <-- one-sided?" if k != "wrist_roll" and (abs(v["range_max"] - 2047) < 15 or abs(v["range_min"] - 2047) < 15) else ""
        flag += "  <-- small range" if k != "wrist_roll" and span < 700 else ""
        print(f"  {k:14} min={v['range_min']:5} max={v['range_max']:5} span={span:5}{flag}")
PY

echo; echo "=== LIVE CHECK (20 s): squeeze the leader trigger open/closed and pan the shoulder ==="
"$PY" - "$LEADER_PORT" "$LEADER_ID" <<'PY'
import sys, time
from lerobot.teleoperators.so_leader import SOLeader
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
t = SOLeader(SOLeaderTeleopConfig(port=sys.argv[1], id=sys.argv[2]))
t.connect(calibrate=False)
lo = {}; hi = {}
t0 = time.time()
while time.time() - t0 < 20:
    a = t.get_action()
    for k, v in a.items(): lo[k] = min(lo.get(k, v), v); hi[k] = max(hi.get(k, v), v)
    print(f"\r gripper={a['gripper.pos']:6.1f} (seen {lo['gripper.pos']:5.1f}..{hi['gripper.pos']:5.1f})   shoulder_pan={a['shoulder_pan.pos']:6.1f}   ", end="", flush=True)
    time.sleep(0.05)
t.disconnect()
g = hi["gripper.pos"] - lo["gripper.pos"]
print(f"\nGripper span {g:.0f} -> {'OK' if g > 60 else 'STILL BROKEN: recalibrate the leader, sweep the trigger fully both ways'}")
PY
echo; echo "Recalibration done. Press ENTER to close."; read -r
