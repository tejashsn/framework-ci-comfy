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

This repo now includes:

- `scripts/upload_to_artifactory.py` — PUT `logs/` to Artifactory (reads `logs/artifactory_path.txt`)
- `.github/actions/upload-to-artifactory` — same contract as the fleet action

To run full CI standalone you still need these fleet actions from `frameworks-qa-ci`:

- `setup-test-environment`, `capture-host-info`, `upload-artifacts`, `fix-permissions`
- `scripts/capture_system_info.py` (or write `logs/bm_config.json` by hand)

ComfyUI CI is **bare metal** — no `docker-pull`, `docker-cleanup`, or container runs.
The runner hosts ComfyUI + ROCm torch directly.

**Gated models (SD3/Flux):** `fetch_models.py` tries AMD Artifactory first
(`artifactory/rocm-qa-model-cache/comfyui/...`) using `ARTIFACTORY_USER` /
`ARTIFACTORY_PASSWORD` — no `HF_TOKEN` when the mirror is populated. Seed once:

```bash
python scripts/seed_comfyui_models_to_artifactory.py --all-gated
```

Manual Artifactory upload after a benchmark:

```bash
export ARTIFACTORY_USER=...
export ARTIFACTORY_PASSWORD=...
python tests/src/inference/comfyui/scripts/upload_from_artifacts.py \
  --results-dir logs/benchmark_results --dry-run   # writes artifactory_path.txt
python scripts/upload_to_artifactory.py \
  --source-path logs/ --test-name comfyui_stable_diffusion_2_1
```

Or run benchmarks on the host (no Docker):

```bash
export COMFYUI_PATH=~/ComfyUI
python tests/src/inference/comfyui/comfyui_benchmark.py \
  --model comfyui_stable_diffusion_2_1 --comfyui-url http://127.0.0.1:8188 ...
```

## Workflow caveats

1. **Bare metal:** ComfyUI must be running on the runner; set `COMFYUI_PATH` (repo variable or workflow input) for model fetch/check.
2. **`upload_results`:** requires `vars.RASTRA_API_URL` + `secrets.RASTRA_API_KEY` on the runner org/repo.
3. **Consolidate job:** uses `consolidate_artifacts.py` (ComfyUI-shaped JSON), not fleet `config_consolidator` (vLLM-shaped YAML).
