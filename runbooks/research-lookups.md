# Hackathon quick lookups: SO-101 + LeRobot + Nebius

## 1. SO-101 ports and calibration (LeRobot)
Source: https://huggingface.co/docs/lerobot/so101

Find ports (run once, unplug the arm when prompted, then reconnect):
```bash
lerobot-find-port
```
Linux only, if access is denied:
```bash
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
```

Calibrate follower:
```bash
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm
```

Calibrate leader:
```bash
lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm
```
Note: the `id` is used to store the calibration file. Reuse the same ids for teleop, record and eval.

## 2. Teleoperate and record (lerobot-record)
Sources: https://huggingface.co/docs/lerobot/il_robots and https://huggingface.co/docs/lerobot/so101

Teleoperate:
```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm
```

Teleoperate with one front camera (rerun display):
```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true
```

Record (5 episodes, 60 s each, 10 s reset, with task string; log in first with `hf auth login`):
```bash
HF_USER=$(NO_COLOR=1 hf auth whoami | awk -F': *' 'NR==1 {print $2}')

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/record-test \
    --dataset.num_episodes=5 \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=10 \
    --dataset.single_task="Grab the black cube" \
    --dataset.push_to_hub=False
```
(Drop `--dataset.push_to_hub=False` to upload to the Hub after recording. Local data lands in `~/.cache/huggingface/lerobot/{repo-id}`.)

Resume recording into the same dataset: re-run the same command with these changes.
```bash
lerobot-record \
    ...same flags as above... \
    --resume=true \
    --dataset.num_episodes=3 \
    --dataset.root="local_path"
```
- With `--resume=true`, `--dataset.num_episodes` is the number of ADDITIONAL episodes, not the total.
- `--dataset.root` must be a local path for the new part. To start over, delete the dataset directory manually.

Keys during recording: Right arrow or `n` = next/early stop; Left arrow or `r` = re-record; ESC or `q` = stop and finish.

## 3. Nebius Object Storage with AWS CLI
Source: https://docs.nebius.com/object-storage/interfaces/aws-cli.md
Access keys: https://docs.nebius.com/iam/service-accounts/access-keys.md

Create a service account with edit access and an access key:
```bash
export SA_ID=$(nebius iam service-account create \
  --name object-storage-sa \
  --format jsonpath='{.metadata.id}')

export PROJECT_ID=$(nebius config get parent-id)
export TENANT_ID=$(nebius iam project get $PROJECT_ID --format jsonpath='{.metadata.parent_id}')

export EDITORS_GROUP_ID=$(nebius iam group get-by-name \
  --name editors --parent-id $TENANT_ID \
  --format jsonpath='{.metadata.id}')

nebius iam group-membership create \
  --parent-id $EDITORS_GROUP_ID \
  --member-id $SA_ID

export ACCESS_KEY_ID=$(nebius iam v2 access-key create \
  --account-service-account-id $SA_ID \
  --description 'AWS CLI' \
  --format json | jq -r '.metadata.id')

export ACCESS_KEY_AWS_ID=$(nebius iam v2 access-key get --id $ACCESS_KEY_ID --format json | jq -r '.status.aws_access_key_id')
export SECRET_ACCESS_KEY=$(nebius iam v2 access-key get --id $ACCESS_KEY_ID --format json | jq -r '.status.secret')
```

Configure AWS CLI (writes to the default profile; replace region if your project is elsewhere):
```bash
aws configure set aws_access_key_id $ACCESS_KEY_AWS_ID
aws configure set aws_secret_access_key $SECRET_ACCESS_KEY
aws configure set region eu-north1
aws configure set endpoint_url https://storage.eu-north1.nebius.cloud
```

Upload and sync:
```bash
aws s3 cp --recursive lorem-ipsum/ s3://example-bucket/lorem-ipsum/
aws s3 sync lorem-ipsum/ s3://example-bucket/lorem-ipsum/
```

Verify the upload landed:
```bash
aws s3 ls
aws s3 ls s3://example-bucket/lorem-ipsum/ --recursive --human-readable --summarize
```

Notes:
- If you pass the endpoint per command instead of configuring it, use `--endpoint-url https://storage.eu-north1.nebius.cloud`.
- Unverified: a search summary mentioned a `get-secret-once` command and a global endpoint `https://storage.ai.nebius.cloud`. I could not confirm either on the fetched pages, so do not rely on them without checking.
- The key-creation `access-key get` pattern above is from the fetched pages.
