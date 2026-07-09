# ComfyUI Validation Suite

Standalone ComfyUI image/video validation on AMD GPUs. Mirrors the xDiT / vLLM
fleet layout under `tests/src/inference/comfyui/`.

## Layout

```
tests/src/inference/comfyui/
  comfyui_benchmark.py          # Suite entry (CI + bare node)
  config/
    suite_manifest.json         # Authoritative test list
    models_config.yaml          # Derived fleet view (regenerate with create_config.py)
    models.json                 # Model weight expectations for SKIP logic
    consolidation.yaml            # Legacy — consolidator uses consolidate_artifacts.py
  executors/                    # ComfyUI runtime + validator (ported engine)
  scripts/
    create_config.py            # Matrix + models_config regeneration
    generate_summary.py         # Per-job GitHub Step Summary table
    upload_from_artifacts.py    # RASTRA payload build + POST
    consolidate_artifacts.py    # Multi-artifact rollup (CI consolidate job)
  workflows/                    # ComfyUI graph JSON per test
```

## Run locally

```bash
python tests/src/inference/comfyui/comfyui_benchmark.py \
  --model comfyui_stable_diffusion_2_1 \
  --arch gfx942 --rocm-version 7.15.0 \
  --comfyui-url http://127.0.0.1:8188 \
  --results-dir logs/benchmark_results
```

## CI

Workflow: `.github/workflows/comfyui-ci.yml` (`ComfyUI Validation`).

**Fleet integration:** drop this folder + the workflow into `frameworks-qa-ci`.
Composite actions (`provision-runtime`, `capture-host-info`, etc.) resolve from
the fleet repo. See `INTEGRATION.md` at repo root.

## Results

- Per test: `results_<test_name>.json` (RASTRA item shape + `result_status` for SKIP/INFRA)
- Roll-up: `benchmark_summary.json`
- DB payload: `logs/payload.json` via `upload_from_artifacts.py`
