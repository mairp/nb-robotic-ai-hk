# Live readout of the follower's raw joint positions before calibration. Every joint must be away from the
# encoder wrap-around (raw ~0/4095), or calibrate fails with "Magnitude ... exceeds 2047". Exits once all are safe.
import sys, time
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
b = FeetechMotorsBus(port=sys.argv[1], motors={n: Motor(i + 1, "sts3215", MotorNormMode.RANGE_M100_100) for i, n in enumerate(names)})
b.connect(handshake=False)
for n in names:
    b.write("Homing_Offset", n, 0)   # read true encoder positions, as calibrate does
ok_since = None
while True:
    pos = b.sync_read("Present_Position", normalize=False)
    bad = [n for n in names if not 300 <= pos[n] % 4096 <= 3800]
    line = "  ".join(f"{n}={pos[n] % 4096:4d}{'!' if n in bad else ''}" for n in names)
    print(f"\r{line}  -> {'OK, hold still' if not bad else 'TURN: ' + ', '.join(bad)}      ", end="", flush=True)
    ok_since = (ok_since or time.time()) if not bad else None
    if ok_since and time.time() - ok_since > 2:
        break
    time.sleep(0.1)
print("\nAll joints safe. Keep this pose: it is the 'middle' you confirm with ENTER next.")
b.disconnect(disable_torque=False)
