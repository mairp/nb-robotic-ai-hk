#!/usr/bin/env bash
# Point the starter repo's env.sh at the setup_vm.sh layout (~/work) on so101-gpu and drop the
# DGX-Spark-only lines in gr00t_activate, as README "Running on Nebius AI Cloud" asks.
# Run on the VM:  ssh ubuntu@REDACTED_VM_IP 'bash -s' < infra/vm-fix-env.sh
set -euo pipefail
W="$HOME/work"
F="$W/so101_sim_pickplace/env.sh"
[ -f "$F.orig" ] || cp "$F" "$F.orig"

sed -i \
  -e "s|^export PROJECT_DIR=.*|export PROJECT_DIR=$W/so101_sim_pickplace|" \
  -e "s|^export GR00T_REPO=.*|export GR00T_REPO=$W/Isaac-GR00T|" \
  -e "s|^export ISAACLAB_DIR=.*|export ISAACLAB_DIR=$W/IsaacLab|" \
  -e "s|^export ISAACLAB_PY=.*|export ISAACLAB_PY=$W/env_isaaclab/bin/python|" \
  -e "s|^export DATASETS_DIR=.*|export DATASETS_DIR=$W/datasets|" \
  -e "s|^export FINETUNE_OUTPUT=.*|export FINETUNE_OUTPUT=\"\${FINETUNE_OUTPUT:-$W/finetune/\$DATASET_NAME}\"|" \
  -e 's|^export POLICY_PORT=.*|export POLICY_PORT=5556    # 5555 is taken by nv-hostengine (NVIDIA DCGM) on Nebius images|' \
  -e '/source scripts\/activate_spark.sh/d' \
  -e '/site-packages\/nvpl\/lib/d' \
  "$F"

diff "$F.orig" "$F" || true
source "$F"
for p in "$PROJECT_DIR" "$GR00T_REPO" "$ISAACLAB_DIR" "$ISAACLAB_PY" "$BASE_MODEL"; do
  [ -e "$p" ] && echo "ok      $p" || echo "MISSING $p"
done
