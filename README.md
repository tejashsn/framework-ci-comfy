# framework-ci-comfy

ComfyUI validation as a **framework-ci-aligned suite** — a standalone repo whose
layout, result schema, catalog, RASTRA payload, and CI mirror the AMD
`frameworks-qa-ci` fleet, so it drops into that repo's `tests/src/inference/`
model or runs on its own.

It carries the hardened ComfyUI validation engine (provenance stamping, model
presence + identity SKIP, IO→INFRA classification, per-test server-log capture,
frozen 0/1/2/3 exit codes, `--doctor`) as the suite's guts, wrapped in the
fleet's conventions.

## Layout (matches frameworks-qa-ci)

```
tests/src/inference/comfyui/
  comfyui_benchmark.py        # standalone entry (xDiT/vLLM-style), self-orchestrates a container
  config/
    suite_manifest.json       # AUTHORITATIVE test catalog (name/tags/arch/os/timeout)
    models_config.yaml        # fleet-facing view, DERIVED from the manifest
    models.json               # model filename -> sha256/size/source (identity check)
    consolidation.yaml        # fleet consolidation config
  executors/                  # hardened engine (validator, protocol, runtime, model_check, preflight)
  workflows/                  # ComfyUI API-format graphs (one per test)
  scripts/
    create_config.py          # regenerate models_config.yaml; --parse-inputs for CI matrix
    generate_summary.py       # markdown Step Summary
    upload_from_artifacts.py  # build RASTRA payload (dry-run)
tests/utils/results/handler.py  # RASTRA payload (fleet shape), dry-run
.github/workflows/
  comfyui-ci.yml              # 4-job: setup -> benchmark -> consolidate -> summary (fleet actions)
  self-check.yml              # unit tests only (no GPU) — NOT validation
```

## Two kinds of tests (do not conflate)

| Name | Command | What | GPU/ComfyUI? |
|---|---|---|---|
| **Framework self-check** | `pytest tests/unit` | harness plumbing (schema, config sync, adapters) | No |
| **ComfyUI/ROCm validation** | `comfyui-ci.yml` / `comfyui_benchmark.py` | the real ~22 GPU generation tests | Yes |

A green self-check does **not** mean validation passed — that is proven only by
`results_<test>.json` (RASTRA shape) from a real GPU run.

## Run one test on a node (self-orchestrating container)

```bash
python tests/src/inference/comfyui/comfyui_benchmark.py \
  --model comfyui_stable_diffusion_2_1 --arch gfx942 \
  --rocm-version 7.15.0 --docker-image rocm/pytorch:latest \
  --results-dir logs/benchmark_results
```

Or by priority/tags: `--model P0`, or `--tags smoke`. Output:
`logs/benchmark_results/results_<test>.json` + `benchmark_summary.json`.

## Result shape (framework-ci RASTRA)

Each `results_<test>.json` carries a single `results` list with
`test_config` (test_name, detected_gpu_arch vs requested, tms_key, rocm_version,
git_sha, workflow_hash, canonical), `test_metrics` (`[{metric_name, score, unit,
flag, primary}]`), `test_result` (PASS/FAIL), `result_status` (PASS/FAIL/SKIP/
INFRA_ERROR — the honest verdict is never lost), and `test_execution_time`
(minutes). `handler.upload_results()` aggregates these into the fleet payload
with `deployment_info.execution_label`.

## Config is derived, never hand-edited

`suite_manifest.json` is authoritative. Regenerate the fleet view:

```bash
python tests/src/inference/comfyui/scripts/create_config.py --regenerate
python tests/src/inference/comfyui/scripts/create_config.py --check   # drift guard (CI)
```

## What is stubbed (wire later)

- **RASTRA POST** — `handler.upload_results` is dry-run only; it writes
  `logs/payload.json` and makes no network call. Add the real POST + creds.
- **Composite actions** — `comfyui-ci.yml` references the fleet's
  `provision-runtime`, `upload-to-artifactory`, etc. They live in
  frameworks-qa-ci, not here; the workflow is drop-in there.

See [MIGRATION_STATUS.md](MIGRATION_STATUS.md) for the full manual-steps list.
