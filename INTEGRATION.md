# Fleet integration (`frameworks-qa-ci`)

This repo is a **migration slice**: correct suite placement, incomplete fleet shell.

## Recommended path (drop-in)

Copy into [frameworks-qa-ci](https://github.com/AMD-ROCm-Internal/frameworks-qa-ci):

| Source (this repo) | Destination (fleet) |
|--------------------|---------------------|
| `tests/src/inference/comfyui/` | `tests/src/inference/comfyui/` |
| `.github/workflows/comfyui-ci.yml` | `.github/workflows/comfyui-ci.yml` |

Then **delete** the vendored copies of:

- `tests/utils/results/` (use fleet's full stack)
- `tests/utils/system/` (use fleet's)

The workflow's `uses: ./.github/actions/*` paths resolve automatically.

## Standalone path

To run CI from this repo alone you must also vendor:

- `.github/actions/` from the fleet (13 composite actions)
- `scripts/capture_system_info.py`, `scripts/upload_to_artifactory.py`
- Optionally `tests/utils/consolidation/` if you switch consolidate back to `config_consolidator`

Or run benchmarks only:

```bash
python tests/src/inference/comfyui/comfyui_benchmark.py --docker-image <image> ...
```

## Workflow caveats

1. **Double container:** `comfyui-ci.yml` calls `provision-runtime` *and* `docker run` for the benchmark. When integrated into the fleet, align with `xdit-ci.yml` (single execution path) before production use.
2. **`upload_results`:** requires `vars.RASTRA_API_URL` + `secrets.RASTRA_API_KEY` on the runner org/repo.
3. **Consolidate job:** uses `consolidate_artifacts.py` (ComfyUI-shaped JSON), not fleet `config_consolidator` (vLLM-shaped YAML).
