# ComfyUI validation — smoke test guide

## Quick smoke (manual workflow_dispatch)

Use these inputs on a self-hosted runner with ROCm + ComfyUI installed:

| Input | Value |
|-------|-------|
| `model_names` | `comfyui_stable_diffusion_2_1,comfyui_stable_diffusion_xl_base_1.0` |
| `runner_label` | Your GPU label (e.g. `gfx1150`) |
| `comfyui_url` | `http://127.0.0.1:8188` |

## Expected outcomes

- **Benchmark jobs:** Each matrix cell restarts ComfyUI, runs one model, then stops the server.
- **PASS:** SD2.1 / SDXL complete within manifest timeouts on capable hardware.
- **Consolidate:** Exit 0 when no FAIL/INFRA_ERROR rows.
- **Done job:** Success when consolidate succeeds.

## Low-VRAM runners (e.g. gfx1150 / 880M)

Heavy models (Wan, LTX, Flux) should **SKIP** with `insufficient_vram` via `min_vram_mb`
gating — not run for 45+ minutes then timeout. Consolidate may show:

```
## ALL RESULTS SKIPPED — check runner capability / gating config
```

That is exit 0 with a loud warning (by design).

## Server start failure

If ComfyUI fails to start, the job writes `results_<test>.json` with
`result_status: INFRA_ERROR` and `failure_reason: INFRA_ERROR server_start_failed`
so consolidate is not blind to the matrix cell.
