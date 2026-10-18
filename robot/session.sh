#!/usr/bin/env bash
# Guided first session: identify leader port -> calibrate -> record N episodes -> upload to the VM.
set -euo pipefail
cd "$(dirname "$0")"
N="${1:-1}"
A=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82109791-if00
B=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82110822-if00
source robot.env
PY="$VENV/bin/python"

if grep -q 'PORT=/dev/serial/by-id/CHANGE_ME' robot.env; then
    echo "=== STEP 1/4: which arm is the leader? ==="
    echo "Leave the FOLLOWER alone. When you press ENTER, MOVE THE LEADER ARM (all joints + gripper) for 45 s."
    read -r
    out=$("$PY" which_leader.py "$A" "$B" | tee /dev/tty)
    ma=$(grep -F "$A" <<<"$out" | sed -E 's/.*movement +([0-9]+).*/\1/')
    mb=$(grep -F "$B" <<<"$out" | sed -E 's/.*movement +([0-9]+).*/\1/')
    if (( ma > 3 * mb + 50 )); then L=$A F=$B
    elif (( mb > 3 * ma + 50 )); then L=$B F=$A
    else echo "Couldn't tell (A=$ma B=$mb). Run this script again and move the leader more."; read -r; exit 1; fi
    sed -i "s|^FOLLOWER_PORT=[^ ]*|FOLLOWER_PORT=$F|; s|^LEADER_PORT=[^ ]*|LEADER_PORT=$L|" robot.env
    echo "Leader = $L"; echo "Follower = $F"
    source robot.env
fi

CAL=$HOME/.cache/huggingface/lerobot/calibration
if [ ! -f "$CAL/robots/so_follower/$FOLLOWER_ID.json" ] && [ ! -f "$CAL/robots/so101_follower/$FOLLOWER_ID.json" ]; then
    echo; echo "=== STEP 2/4: calibration (follower, then leader) ==="
    echo "For each arm: put it in the MIDDLE of its range, press ENTER; then move EVERY joint through its full range, press ENTER."
    echo "First, put the FOLLOWER in its middle pose. Joints marked ! sit at the encoder wrap point: turn them a bit (wrist roll ~20-30 deg)."
    "$PY" wrist_guard.py "$FOLLOWER_PORT"
    ./robot-ops.sh calibrate
fi

echo; echo "=== STEP 3/4: record $N episode(s) ==="
pkill -u "$USER" -f 'ffplay .*camera' 2>/dev/null || true; sleep 1   # free the cameras
./robot-ops.sh check
./robot-ops.sh record "$N"

echo; echo "=== STEP 4/4: upload to the VM ==="
./robot-ops.sh upload
echo; echo "DONE. Press ENTER to close."; read -r
