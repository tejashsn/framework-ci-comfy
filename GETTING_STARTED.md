# Getting Started — "I just pulled this repo, now what?"

This is a step-by-step walkthrough for someone who has **never seen this repo**
and knows **nothing about how it works**. Follow it top to bottom. By the end you
will have run real ComfyUI tests on an AMD GPU and have result files to show for it.

> **What is this repo, in one sentence?**
> It runs a list of AI image/video generations (Stable Diffusion, SDXL, LTX
> video, etc.) on an AMD GPU through ComfyUI, and writes a PASS/FAIL/SKIP score
> for each one.

There are only **two** commands you really need. Everything else below is just
checking that things are ready before you run them:

1. `pytest tests/unit` — proves the repo itself is intact (no GPU needed)
2. `python comfyui_benchmark.py ...` — runs the actual GPU tests

---

## The big picture (read this once)

```
   YOU run:  comfyui_benchmark.py
                    │
                    │  1. reads the test list (suite_manifest.json)
                    │  2. keeps only tests meant for YOUR GPU
                    │  3. for each test, checks the model file is present
                    │        └─ missing?  → SKIP (not an error)
                    │  4. sends the workflow to a running ComfyUI server
                    ▼
        ComfyUI generates the image/video on the GPU
                    │
                    ▼
   YOU get:  results_<test>.json   (PASS/FAIL/SKIP + how long it took)
             benchmark_summary.json (totals)
```

Two things must exist on the machine **before** you run tests:
- **A running ComfyUI server** (this repo does not install ComfyUI — it talks to one)
- **The model weight files**, placed under `ComfyUI/models/` (missing ones just SKIP)

---

## Step 1 — Prove the repo is intact (30 seconds, no GPU)

Right after pulling, run the self-check. This does **not** touch the GPU — it just
confirms the code and config are consistent.

```bash
cd framework-ci-comfy
pip install -r requirements.txt
python -m pytest tests/unit -q
```

✅ **Expected:** `8 passed`
❌ **If it fails:** the repo didn't come across cleanly — re-pull it. Don't go further.

Also confirm the config is internally consistent:

```bash
python tests/src/inference/comfyui/scripts/create_config.py --check
```

✅ **Expected:** `[ok] models_config.yaml in sync with manifest`

At this point you know the **repo** is healthy. You have **not** tested any model yet.

---

## Step 2 — Check the machine is GPU-ready (1 minute, no generation)

This repo runs models **inside a ComfyUI server** that uses the AMD GPU via
PyTorch+ROCm. Before running tests, ask the built-in "doctor" whether everything
it needs is actually there:

```bash
cd tests/src/inference/comfyui
python executors/preflight_check.py \
  --expected-rocm '7.15.*' --expected-gpu-arch gfx908 --doctor
```

(Replace `gfx908` with your GPU — see Step 3 to find it — and `7.15.*` with your
ROCm version.)

The doctor prints, in plain English, four checks:

| Check | What it means | If it's bad |
|---|---|---|
| `[git]` | is this a clean checkout | harmless warning; runs still work |
| `[profile]` | optional machine config | harmless warning; ignore unless you use it |
| `[comfyui]` | **is a ComfyUI server reachable** | you must start one — see Step 4 |
| `[models]` | which model weights are present vs missing | missing ones will SKIP, not fail |

✅ **Expected bottom line:** `DOCTOR: no infrastructure blockers.`
❌ **If it says `infrastructure problem(s) found`:** read the `FIX:` line it prints
— it tells you exactly what to do (usually: start ComfyUI, or install ROCm torch).

---

## Step 3 — Find out which GPU you have

Which tests run depends on your GPU architecture (the `gfxNNNN` code). Find it:

```bash
rocminfo | grep -m1 -o 'gfx[0-9a-f]*'
```

Example output: `gfx908` (that's an AMD Instinct MI100). Common ones:
`gfx908` = MI100, `gfx90a` = MI200, `gfx942` = MI300.

You'll pass this as `--arch` when running tests. Tests that don't list your GPU in
the manifest are simply not run — that's expected, not a problem.

> No `rocminfo` on the host? That's fine — it often lives inside the ROCm
> container/venv, and the benchmark auto-detects your GPU if you omit `--arch`.

---

## Step 4 — Make sure a ComfyUI server is running

This repo **talks to** ComfyUI; it doesn't start it for you. Check if one is
already up:

```bash
curl -sf http://127.0.0.1:8188/system_stats && echo "  <- ComfyUI is UP"
```

- **If you get JSON back:** great, ComfyUI is running. Note the URL (`http://127.0.0.1:8188`).
- **If you get nothing:** start your ComfyUI (however it's installed on that box,
  e.g. `cd ~/ComfyUI && python main.py --listen 127.0.0.1 --port 8188`), then
  re-check.

Also note **where ComfyUI keeps its models** — the folder that contains a
`models/` subdirectory (often `~/ComfyUI`). You'll point the tests at it so they
can see which weights are present.

---

## Step 5 — Run the tests 🎉

Point the benchmark at your ComfyUI, tell it your GPU, and pick where results go.

```bash
cd tests/src/inference/comfyui
export COMFYUI_PATH=~/ComfyUI          # the folder that has a models/ subdir

# Start small — run ONE quick test to prove the pipeline end to end:
python comfyui_benchmark.py \
  --model comfyui_stable_diffusion_2_1 \
  --arch gfx908 --rocm-version 7.15.0 \
  --comfyui-url http://127.0.0.1:8188 \
  --results-dir ~/results
```

✅ **Expected:** it prints `[PASS] comfyui_stable_diffusion_2_1` and writes files
into `~/results`.

Once that works, run **everything eligible for your GPU** (just drop `--model`):

```bash
python comfyui_benchmark.py \
  --arch gfx908 --rocm-version 7.15.0 \
  --comfyui-url http://127.0.0.1:8188 \
  --results-dir ~/results
```

> **Use the right Python.** Run these with the same Python that your ComfyUI uses
> (the one that can `import torch` and see the GPU). If you're unsure, use
> ComfyUI's own venv, e.g. `~/comfyui-host-venv/bin/python comfyui_benchmark.py ...`.

---

## Step 6 — Read your results

Everything is under the `--results-dir` you chose (`~/results` above).

**The quick answer** — did everything pass?

```bash
cat ~/results/benchmark_summary.json
```

```json
{
  "success": true,
  "arch": "gfx908",
  "counts": { "PASS": 5, "FAIL": 0, "SKIP": 1, "INFRA_ERROR": 0 }
}
```

**Per-test scores** — one file each:

```bash
cat ~/results/results_comfyui_stable_diffusion_2_1.json
```

Look for `test_result` (PASS/FAIL) and the `latency` metric (seconds).

**The generated images/videos** are in ComfyUI's own output folder
(`~/ComfyUI/output/`), not in the results folder.

**If a test didn't PASS**, its evidence folder has the logs:
`~/results/<test>_<timestamp>/.../` contains `protocol_stderr.log` and
`comfyui_server_tail.log` (ComfyUI's own output — that's where a GPU/OOM error
would show up).

---

## What the results mean

| Verdict | What happened | Is it bad? |
|---|---|---|
| **PASS** | model generated valid output in time | 🎉 no |
| **SKIP** | on purpose — weight file missing, or test not for your GPU | no, expected |
| **FAIL** | it ran but output was bad, or it timed out | yes — investigate |
| **INFRA_ERROR** | environment broke (ComfyUI down, GPU gone, disk full) | yes — fix the box |

The whole design philosophy: **a missing model is a SKIP, not a failure**, and a
**broken environment is an INFRA_ERROR, not a failure**. A FAIL means a genuine
model/output problem — the thing you actually care about.

---

## "A test SKIPped — how do I make it run?"

A SKIP for a missing model tells you exactly what file it wanted:

```
SKIP: model not found: models/checkpoints/sd3_medium_incl_clips_t5xxlfp8.safetensors.
Place the file(s) under ComfyUI/models/ or set HF_TOKEN for gated models.
```

Put that file in that folder under your ComfyUI install and re-run. Some models are
license-gated (you must accept terms on Hugging Face and use an `HF_TOKEN`) — those
stay SKIPped until you provide the weights. The suite never auto-downloads weights.

---

## Common problems, quick fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| `pytest` fails right after pull | bad/incomplete checkout | re-pull the repo |
| doctor says ComfyUI not reachable | no server running | start ComfyUI, re-check with the `curl` in Step 4 |
| every test SKIPs with "not found" | wrong/undetected models path | `export COMFYUI_PATH=<folder with models/>` |
| test FAILs with a torch/CUDA error | ran with the wrong Python | use ComfyUI's venv Python |
| "0 tests selected" | your GPU isn't in those tests' arch list | check `--arch`; that's expected for some tests |

---

## The absolute minimum (TL;DR)

```bash
# 1. repo intact?
python -m pytest tests/unit -q                    # -> 8 passed

# 2. box ready?  (uses your GPU arch + rocm version)
cd tests/src/inference/comfyui
python executors/preflight_check.py --expected-rocm '7.15.*' --expected-gpu-arch gfx908 --doctor

# 3. ComfyUI up?
curl -sf http://127.0.0.1:8188/system_stats && echo UP

# 4. run it
export COMFYUI_PATH=~/ComfyUI
python comfyui_benchmark.py --arch gfx908 --rocm-version 7.15.0 \
  --comfyui-url http://127.0.0.1:8188 --results-dir ~/results

# 5. read it
cat ~/results/benchmark_summary.json
```

For the deeper reference (result schema, config internals, what's stubbed), see
[README.md](README.md) and [MIGRATION_STATUS.md](MIGRATION_STATUS.md).
