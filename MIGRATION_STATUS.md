# MIGRATION_STATUS.md — framework-ci-comfy

New standalone repo aligning ComfyUI validation with the AMD `frameworks-qa-ci`
fleet, built against the **real** fleet repo (not a description). Local only —
**not pushed, no remote**. Suite self-check: 8 unit tests pass.

> Just pulled this and want to run tests? See **[GETTING_STARTED.md](GETTING_STARTED.md)**
> for a step-by-step walkthrough, or [README.md](README.md) for the reference. This
> file is the migration/hand-off checklist for wiring the suite into the fleet.

## Built to the real fleet conventions

Aligned by reading the actual frameworks-qa-ci repo:
- **Suite layout** = `tests/src/inference/comfyui/` with `<suite>_benchmark.py`,
  `config/models_config.yaml`, `config/consolidation.yaml`,
  `scripts/{create_config,generate_summary,upload_from_artifacts}.py` — same as
  vLLM/xDiT.
- **Result shape** = `results_<test>.json` with the RASTRA item shape
  (`test_config`, `test_metrics: [{metric_name,score,unit,flag,primary}]`,
  `test_result` PASS/FAIL, `test_execution_time` in **minutes**), plus a
  `result_status` field preserving SKIP/INFRA_ERROR (honest verdict never lost).
- **Payload** = `tests/utils/results/handler.py` builds the fleet
  `build_results_payload` top-level shape with
  `deployment_info.execution_label`; framework auto-detected by keyword.
- **CI** = `comfyui-ci.yml` mirrors vLLM's 4 jobs (setup → benchmark →
  consolidate → summary), `fail-fast: false`, calling the fleet composite
  actions (`provision-runtime`, `capture-host-info`, `docker-pull/cleanup`,
  `upload-to-artifactory`, `upload-artifacts`, `fix-permissions`).
- **Runtime** = `comfyui_benchmark.py --docker-image` self-orchestrates a ROCm
  container (bind-mounts cwd, `COMFYUI_IN_CONTAINER=1` guards recursion), same
  pattern as `xdit_benchmark.py`.

## Hardened engine carried in (this repo's edge)

The ported executors keep every trust property built earlier and flow it into
the fleet result shape:
- Provenance stamping (git_sha / canonical / workflow_hash in `test_config`).
- Detected-vs-requested GPU arch (`detected_gpu_arch` vs `requested_gpu_arch`).
- Model presence + identity SKIP (`config/models.json`); missing/mismatch = SKIP.
- IO/save errors → INFRA_ERROR (not a content FAIL).
- Per-test ComfyUI server-log capture on every non-PASS.
- Frozen exit codes 0=PASS/1=FAIL/2=SKIP/3=INFRA_ERROR.

## Stubbed (intentionally — wire manually)

| Stub | Where | Activate |
|---|---|---|
| RASTRA POST | `tests/utils/results/handler.py` (`upload_results`, dry_run=True) | implement `requests.post(api_url, json=payload, headers={'Authorization': api_key})`; set API_URL/API_KEY. Payload shape already correct. |
| Composite actions | `.github/workflows/comfyui-ci.yml` (`uses: ./.github/actions/*`) | these live in frameworks-qa-ci. Drop this suite into that repo, or copy the actions here. |
| Consolidator | `tests.utils.consolidation.config_consolidator` (referenced in consolidate job) | provided by the fleet; not vendored here. |
| GPU telemetry | benchmark metrics beyond latency (vram/throughput) | optional; add rocm-smi capture like the fleet's `rocm_gpuwrap.py`. |

## Manual steps left for you (ordered)

1. **Review** the tree (`README.md`, this file) and the 8 unit tests.
2. **git remote + push** when satisfied — I created no remote and did not push.
3. **Decide integration model:**
   - (a) Drop `tests/src/inference/comfyui/` + `.github/workflows/comfyui-ci.yml`
     into the real frameworks-qa-ci repo — the composite actions + consolidator
     resolve there automatically; OR
   - (b) Keep standalone: copy the fleet composite actions into `.github/actions/`
     and vendor `tests/utils/consolidation/`.
4. **Wire RASTRA creds** — implement the POST in `handler.upload_results`.
5. **Real GPU run** — validate `comfyui_benchmark.py` end-to-end on a gfx942/950
   box and confirm `results_<test>.json` + payload look right. (Self-check green
   ≠ GPU-validated.)

## Guarantees
- Ported executor