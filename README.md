# framework-ci-comfy

A test suite that checks whether **image and video AI models run correctly on
AMD GPUs**, using [ComfyUI](https://github.com/comfyanonymous/ComfyUI) as the
engine. It runs a set of real image/video generations (Stable Diffusion, SDXL,
SD3, Flux, LTX video, Wan, etc.), decides PASS / FAIL / SKIP for each, and writes
the results in a standard format the AMD `frameworks-qa-ci` fleet understands.

If you just want to answer *"does model X actually generate a picture on this AMD
GPU, and how fast?"* — that's what this does.

> 🚀 **New here / just pulled this onto a server?** Start with
> **[GETTING_STARTED.md](GETTING_STARTED.md)** — a step-by-step, no-background-needed
> walkthrough from "I just cloned this" to "I have test results."

---

## The 30-second mental model

```
  suite_manifest.json          the list of tests (name, which workflow, which GPUs, timeout)
        │
        ▼
  comfyui_benchmark.py          you run this; it picks the eligible tests and runs each one
        │
        ▼
  a running ComfyUI server      does the actual generation on the GPU
        │
        ▼
  results_<test>.json           one score file per test: PASS/FAIL/SKIP + how long it took
  benchmark_summary.json        the roll-up (5 passed, 1 skipped, ...)
```

Each test = **one ComfyUI workflow** (a `.json` graph in `workflows/`) run with a
fixed prompt, on a real GPU. The prompt lives *inside* that workflow file.

---

## Two things called "tests" — don't mix them up

| | Command | What it checks | Needs a GPU? |
|---|---|---|---|
| **Self-check** | `pytest tests/unit` | that the suite's own plumbing works (config in sync, schema valid) | **No** |
| **Real validation** | `comfyui_benchmark.py` | that models actually generate on the AMD GPU | **Yes** |

A green self-check only means the harness is wired correctly. It does **not** mean
any model passed — that's only proven by a `results_<test>.json` from a real GPU run.

---

## Quick start

### 1. Self-check (no GPU, ~1 second)

```bash
pip install -r requirements.txt        # pytest, pyyaml, jsonschema
python -m pytest tests/unit -q         # expect: 8 passed
```

### 2. Real validation on a GPU box

You need a ComfyUI server running and reachable (default `http://127.0.0.1:8188`),
with the model weights placed under `ComfyUI/models/`.

```bash
cd tests/src/inference/comfyui
export COMFYUI_PATH=~/ComfyUI          # so it can find your models/ folder

# Run ONE test:
python comfyui_benchmark.py \
  --model comfyui_stable_diffusion_2_1 \
  --arch gfx908 --rocm-version 7.15.0 \
  --comfyui-url http://127.0.0.1:8188 \
  --results-dir ~/results

# Run EVERY test eligible for this GPU (just drop --model):
python comfyui_benchmark.py --arch gfx908 --rocm-version 7.15.0 \
  --comfyui-url http://127.0.0.1:8188 --results-dir ~/results

# Run a subset by tag:
python comfyui_benchmark.py --tags smoke --arch gfx908 \
  --comfyui-url http://127.0.0.1:8188 --results-dir ~/results
```

> **Tip:** run it with the same Python that your ComfyUI uses (the one that can
> `import torch` and see the GPU). If you omit `--arch`, it auto-detects the GPU.

### Just want to know if the box is ready? (no generation)

```bash
python executors/preflight_check.py \
  --expected-rocm '7.15.*' --expected-gpu-arch gfx908 --doctor
```

This tells you, in plain English, whether the GPU, PyTorch, and ComfyUI are all
good — and which model weights are present vs missing — without running any tests.

---

## Where the results and logs go

Everything lands under the `--results-dir` you pass (e.g. `~/results`):

```
~/results/
├── results_<test>.json               ← THE SCORE for each test (see below)
├── benchmark_summary.json            ← roll-up: PASS/FAIL/SKIP counts, GPU, git commit
└── <test>_<timestamp>/               ← 