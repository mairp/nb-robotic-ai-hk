set -euo pipefail
export PATH="$HOME/.local/bin:$PATH" OMNI_KIT_ACCEPT_EULA=YES
export UV_PROJECT_ENVIRONMENT=$HOME/work/env_isaaclab
cd ~/work/IsaacLab
uv sync --extra isaacsim --inexact
uv pip install --python $UV_PROJECT_ENVIRONMENT/bin/python pyzmq msgpack msgpack-numpy h5py imageio pyarrow
$UV_PROJECT_ENVIRONMENT/bin/python -c "import isaaclab, isaacsim; print(\"isaaclab import OK\", isaaclab.__file__)"
$UV_PROJECT_ENVIRONMENT/bin/pip list 2>/dev/null | grep -iE "^(isaacsim|isaaclab|torch) " || true
