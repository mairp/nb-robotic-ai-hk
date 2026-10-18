#!/usr/bin/env bash
# SO-101 leader/follower + two cameras on the robot laptop (Linux): setup, discovery, calibration,
# teleop check, recording and upload to the team GPU VM.
#
#   ./robot-ops.sh setup        install LeRobot + system deps, serial-port access (once)
#   ./robot-ops.sh ports        list arm serial ports (stable by-id paths)
#   ./robot-ops.sh cameras      list cameras and save one test frame per camera
#   ./robot-ops.sh check        verify robot.env: ports, cameras, resolution
#   ./robot-ops.sh calibrate    calibrate follower then leader (interactive)
#   ./robot-ops.sh teleop       live teleop with both cameras shown, nothing recorded
#   ./robot-ops.sh record [N]   record N more episodes (default 5), resumes the same dataset
#   ./robot-ops.sh status       episodes recorded so far
#   ./robot-ops.sh upload       rsync the dataset to the VM (~/work/datasets/real/)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=robot.env
source "$HERE/robot.env"
LR="$VENV/bin"
export PATH="$LR:$PATH"   # rerun viewer (--display_data) lives in the venv

die() { echo "ERROR: $*" >&2; exit 1; }
need_conf() { grep -q CHANGE_ME <<<"$FOLLOWER_PORT$LEADER_PORT$FRONT_CAM$WRIST_CAM" && die "fill in ports/cameras in robot.env first (run: ports, cameras)"; return 0; }
need_lerobot() { [ -x "$LR/lerobot-record" ] || die "LeRobot not installed; run: $0 setup"; }
cameras_arg() {
    echo "{front: {type: opencv, index_or_path: $FRONT_CAM, width: $CAM_W, height: $CAM_H, fps: $CAM_FPS}, wrist: {type: opencv, index_or_path: $WRIST_CAM, width: $CAM_W, height: $CAM_H, fps: $CAM_FPS}}"
}
episodes_done() {
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("total_episodes", 0))' "$DATASET_ROOT/meta/info.json" 2>/dev/null || echo 0
}

cmd_setup() {
    [ "$(uname -s)" = Linux ] || die "this script targets Linux"
    sudo apt-get update
    sudo apt-get install -y ffmpeg v4l-utils curl git rsync python3
    command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
    uv venv --python 3.12 "$VENV"
    uv pip install --python "$LR/python" "lerobot[feetech,core-scripts]==$LEROBOT_VERSION"
    # Serial access without sudo chmod on every replug; takes effect after logging out and back in.
    if ! id -nG "$USER" | grep -qw dialout; then
        sudo usermod -aG dialout "$USER"
        echo ">>> Added $USER to 'dialout'. Log out and back in (or reboot) before using the arms."
    fi
    "$LR/python" -c "import lerobot, cv2; print('lerobot', lerobot.__version__, '| opencv', cv2.__version__)"
    echo "Setup done. Next: $0 ports && $0 cameras, then fill in robot.env"
}

cmd_ports() {
    echo "== Arm serial ports (use the by-id path in robot.env) =="
    ls -l /dev/serial/by-id/ 2>/dev/null || echo "no /dev/serial/by-id entries: are the arms plugged in and powered?"
    echo
    echo "Which one is which? Run 'lerobot-find-port' and unplug one arm when it asks:"
    echo "  $LR/lerobot-find-port"
}

cmd_cameras() {
    echo "== Video devices =="
    v4l2-ctl --list-devices 2>/dev/null || true
    echo "== Stable paths (use these in robot.env) =="
    ls -l /dev/v4l/by-id/ 2>/dev/null || echo "no /dev/v4l/by-id entries"
    need_lerobot
    echo "== Saving one frame per camera to ./camera_check/ (open them to see which is front/wrist) =="
    "$LR/lerobot-find-cameras" opencv --output-dir "$HERE/camera_check" || true
}

cmd_check() {
    need_conf; need_lerobot
    for p in "$FOLLOWER_PORT" "$LEADER_PORT"; do
        [ -e "$p" ] || die "port not found: $p"
        if [ ! -r "$p" ] || [ ! -w "$p" ]; then
            die "no read/write access to $p (log out/in after setup, or: sudo chmod 666 $(readlink -f "$p"))"
        fi
    done
    [ "$(readlink -f "$FOLLOWER_PORT")" != "$(readlink -f "$LEADER_PORT")" ] || die "leader and follower point to the same port"
    for c in "$FRONT_CAM" "$WRIST_CAM"; do [ -e "$c" ] || die "camera not found: $c"; done
    [ "$(readlink -f "$FRONT_CAM")" != "$(readlink -f "$WRIST_CAM")" ] || die "front and wrist point to the same camera"
    "$LR/python" - "$FRONT_CAM" "$WRIST_CAM" "$CAM_W" "$CAM_H" "$CAM_FPS" <<'PY'
import sys, cv2
w, h, fps = map(int, sys.argv[3:6])
for name, dev in zip(("front", "wrist"), sys.argv[1:3]):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h); cap.set(cv2.CAP_PROP_FPS, fps)
    ok, frame = cap.read()
    got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    status = "OK" if ok and got[:2] == (w, h) else "PROBLEM"
    print(f"{name:5} {dev}: read={ok} size={got[0]}x{got[1]} fps={got[2]:.0f} -> {status}")
    if status != "OK":
        sys.exit(f"{name} camera can't deliver {w}x{h}")
PY
    echo "check passed: 2 arms, 2 cameras at ${CAM_W}x${CAM_H}"
}

cmd_calibrate() {
    need_conf; need_lerobot
    echo ">>> Follower ($FOLLOWER_ID): follow the prompts (middle pose, then move every joint through its full range)"
    "$LR/lerobot-calibrate" --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID"
    echo ">>> Leader ($LEADER_ID)"
    "$LR/lerobot-calibrate" --teleop.type=so101_leader --teleop.port="$LEADER_PORT" --teleop.id="$LEADER_ID"
    echo "Calibration saved under ~/.cache/huggingface/lerobot/calibration/. Don't recalibrate after recording starts."
}

cmd_teleop() {
    need_conf; need_lerobot
    exec "$LR/lerobot-teleoperate" \
        --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID" \
        --robot.cameras="$(cameras_arg)" \
        --teleop.type=so101_leader --teleop.port="$LEADER_PORT" --teleop.id="$LEADER_ID" \
        --display_data=true
}

cmd_record() {
    need_conf; need_lerobot
    local n="${1:-5}" done_n
    done_n=$(episodes_done)
    (( done_n + n <= MAX_REAL_EPISODES )) || die "$done_n recorded + $n requested > $MAX_REAL_EPISODES real-episode limit"
    local resume=false
    [ -f "$DATASET_ROOT/meta/info.json" ] && resume=true
    mkdir -p "$(dirname "$DATASET_ROOT")"
    echo ">>> Recording $n episodes (already $done_n), ${EPISODE_TIME_S}s each + ${RESET_TIME_S}s reset, resume=$resume"
    echo "    Keys: → (right arrow) end episode early | ← re-record this episode | Esc stop and save"
    echo "    Every episode must end with the cube RELEASED on the pad. Rotate grid spots every 4-5 episodes."
    "$LR/lerobot-record" \
        --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID" \
        --robot.cameras="$(cameras_arg)" \
        --teleop.type=so101_leader --teleop.port="$LEADER_PORT" --teleop.id="$LEADER_ID" \
        --dataset.repo_id="$DATASET_NAME" --dataset.root="$DATASET_ROOT" \
        --dataset.single_task="$TASK" --dataset.fps="$CAM_FPS" \
        --dataset.num_episodes="$n" \
        --dataset.episode_time_s="$EPISODE_TIME_S" --dataset.reset_time_s="$RESET_TIME_S" \
        --dataset.push_to_hub=false --display_data=true --resume="$resume"
    cmd_status
}

cmd_status() {
    echo "episodes recorded: $(episodes_done) / $MAX_REAL_EPISODES  ($DATASET_ROOT)"
}

cmd_upload() {
    [ -f "$DATASET_ROOT/meta/info.json" ] || die "no dataset at $DATASET_ROOT"
    ssh "$VM" "mkdir -p ~/work/datasets/real"
    rsync -av --partial "$DATASET_ROOT/" "$VM:work/datasets/real/$(basename "$DATASET_ROOT")/"
    echo "uploaded $(episodes_done) episodes to $VM:~/work/datasets/real/$(basename "$DATASET_ROOT")"
}

case "${1:-}" in
    setup|ports|cameras|check|calibrate|teleop|status|upload) "cmd_$1" ;;
    record) shift; cmd_record "$@" ;;
    *) sed -n '2,15p' "$0"; exit 1 ;;
esac
