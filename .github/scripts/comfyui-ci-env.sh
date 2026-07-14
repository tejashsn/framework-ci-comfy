#!/usr/bin/env bash
# Shared ComfyUI CI environment resolution for comfyui-ci.yml.
# Usage: source .github/scripts/comfyui-ci-env.sh && resolve_comfyui_env
#
# Resolves COMFYUI_PATH and the ComfyUI Python, exporting both and appending
# them to $GITHUB_ENV so later steps inherit them. Path discovery order:
# known install dirs (Linux + Windows, incl. C:/ComfyUI), then the suite's
# comfyui_runtime.discover_comfyui_path() as a fallback.
resolve_comfyui_env() {
  local py="${COMFYUI_PYTHON:-}"
  if [ -z "$py" ]; then
    py="${CI_PYTHON:-$(command -v python3 2>/dev/null || command -v python)}"
  fi
  if [ -z "${COMFYUI_PATH:-}" ]; then
    if [ -d /home/master/ComfyUI ]; then
      echo "COMFYUI_PATH=/home/master/ComfyUI" >> "$GITHUB_ENV"
      export COMFYUI_PATH=/home/master/ComfyUI
    elif [ -d /home/taccuser/ComfyUI ]; then
      echo "COMFYUI_PATH=/home/taccuser/ComfyUI" >> "$GITHUB_ENV"
      export COMFYUI_PATH=/home/taccuser/ComfyUI
    elif [ -d /c/TheRock/ComfyUI ]; then
      echo "COMFYUI_PATH=/c/TheRock/ComfyUI" >> "$GITHUB_ENV"
      export COMFYUI_PATH=/c/TheRock/ComfyUI
    elif [ -d /c/ComfyUI ]; then
      echo "COMFYUI_PATH=/c/ComfyUI" >> "$GITHUB_ENV"
      export COMFYUI_PATH=/c/ComfyUI
    else
      local discovered
      discovered=$("$py" -c "
import sys
sys.path.insert(0, 'tests/src/inference/comfyui/executors')
import comfyui_runtime
p = comfyui_runtime.discover_comfyui_path({})
print(p or '', end='')
" 2>/dev/null || true)
      if [ -n "$discovered" ]; then
        echo "COMFYUI_PATH=$discovered" >> "$GITHUB_ENV"
        export COMFYUI_PATH="$discovered"
      fi
    fi
  fi
  if [ "$py" = "python3" ] || [ -z "$py" ]; then
    if [ -x /home/master/comfyui-host-venv/bin/python ]; then
      py=/home/master/comfyui-host-venv/bin/python
    elif [ -x /home/taccuser/comfyui-host-venv/bin/python ]; then
      py=/home/taccuser/comfyui-host-venv/bin/python
    elif [ -x /c/TheRock/.venv314/Scripts/python.exe ]; then
      py=/c/TheRock/.venv314/Scripts/python.exe
    elif [ -x /c/TheRock/.venv/Scripts/python.exe ]; then
      py=/c/TheRock/.venv/Scripts/python.exe
    fi
  fi
  echo "Using Python: $($py --version 2>&1)"
  echo "COMFYUI_PYTHON=$py" >> "$GITHUB_ENV"
  export COMFYUI_PYTHON="$py"
  echo "ComfyUI path: ${COMFYUI_PATH:-(not set — discovery failed)}"
}
