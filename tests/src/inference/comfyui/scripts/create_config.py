#!/usr/bin/env python3
"""
create_config.py — framework-ci suite config helper for the ComfyUI suite.

Roles (mirroring vLLM/xDiT scripts of the same name):

  1. --regenerate : rebuild config/models_config.yaml FROM suite_manifest.json
  2. --parse-inputs : expand model selection + runner labels for CI setup job
  3. --detect-gpu-arch : write detected GPU arch for CI (no arch gating)

Also --check verifies models_config.yaml is in sync with the manifest.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import yaml  # type: ignore[import-not-found]

SUITE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = SUITE_DIR / "config"
MANIFEST = CONFIG_DIR / "suite_manifest.json"
MODELS_CONFIG = CONFIG_DIR / "models_config.yaml"
_FRAMEWORK_ROOT = Path(__file__).resolve().parents[5]


def detect_gpu_arch() -> str:
    """Detect GPU architecture via fleet hardware.py (importlib, no __init__ chain)."""
    try:
        hw_path = _FRAMEWORK_ROOT / "tests" / "utils" / "system" / "hardware.py"
        spec = importlib.util.spec_from_file_location("hardware", str(hw_path))
        if spec is None or spec.loader is None:
            return "unknown"
        hw_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hw_module)
        arch = hw_module.detect_gpu_arch()
        return (arch or "unknown").strip().lower()
    except Exception:
        return "unknown"


def load_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def load_models_config(path: Path | str | None = None) -> dict:
    cfg_path = Path(path) if path else MODELS_CONFIG
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def build_models_config_text(manifest):
    tests = manifest.get("tests", [])
    L = []
    L.append("# ComfyUI validation suite config (framework-ci suite style).")
    L.append("# DERIVED from config/suite_manifest.json via scripts/create_config.py.")
    L.append("# The manifest stays authoritative; this is the fleet-facing view.")
    L.append("# CI selects tests by comma-separated test_name list (fleet workflow_id style).")
    L.append("tests:")
    for t in tests:
        L.append(f'  - name: "{t["test_name"]}"')
        L.append(f'    tms_key: {json.dumps(t.get("tms_key"))}')
        L.append(f'    tags: [{", ".join(chr(34)+x+chr(34) for x in t.get("tags", []))}]')
        L.append(f'    os: [{", ".join(chr(34)+x+chr(34) for x in t.get("os", []))}]')
        L.append(f'    timeout_minutes: {t.get("timeout_minutes")}')
        L.append(f'    expected_output_type: "{t.get("expected_output_type", "image")}"')
        L.append(f'    enabled: {str(t.get("enabled", True)).lower()}')
        L.append('    command: "python3 tests/src/inference/comfyui/comfyui_benchmark.py '
                 f'--model {t["test_name"]}"')
    return "\n".join(L) + "\n"


def expand_selection(selection, models_cfg):
    """Comma-separated test names (fleet workflow_id style)."""
    sel = (selection or "").strip()
    if not sel:
        raise ValueError(
            "test_names required: comma-separated manifest test_name values "
            "(e.g. comfyui_stable_diffusion_2_1,comfyui_flux1_dev)"
        )
    names = [s.strip() for s in sel.split(",") if s.strip()]
    known = {t["name"]: t for t in models_cfg.get("tests", [])}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"Unknown test name(s): {unknown}")
    disabled = [n for n in names if not known[n].get("enabled", True)]
    if disabled:
        raise ValueError(f"Disabled test(s) in manifest: {disabled}")
    return names


def detect_gpu_arch_for_ci(output_file: str = "") -> int:
    """Detect GPU arch for CI metadata. Tests are not gated by architecture."""
    gpu_arch = detect_gpu_arch()
    print(f"Detected GPU arch: {gpu_arch or 'unknown'}")
    if output_file:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"gpu_arch={gpu_arch}\n")
    return 0


def parse_inputs(args) -> int:
    """Expand models + runner labels for GitHub Actions matrix."""
    models_cfg = load_models_config(args.config_file)
    try:
        selected = expand_selection(args.models, models_cfg)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    labels = [x.strip() for x in (args.runner_label or "self-hosted").split(",") if x.strip()]
    if not labels:
        labels = ["self-hosted"]

    trigger_type = args.trigger or "workflow_dispatch"
    upload_results = str(args.upload).lower() if args.upload is not None else "true"

    model_count = len(selected)
    label_count = len(labels)
    total_jobs = model_count * label_count

    print(f"Trigger: {trigger_type}")
    print(f"Models ({model_count}): {json.dumps(selected)}")
    print(f"Runner labels ({label_count}): {json.dumps(labels)}")
    print(f"Total jobs: {model_count} models × {label_count} labels = {total_jobs} jobs")
    print(f"Upload results: {upload_results}")
    print(f"Execution mode: baremetal (no Docker)")

    if args.output_file:
        with open(args.output_file, "a", encoding="utf-8") as f:
            f.write(f"models={json.dumps(selected)}\n")
            f.write(f"model_count={model_count}\n")
            f.write(f"trigger_type={trigger_type}\n")
            f.write(f"runner_labels={json.dumps(labels)}\n")
            f.write(f"runner_label_count={label_count}\n")
            f.write(f"upload_results={upload_results}\n")
    return 0


def main():
    p = argparse.ArgumentParser(description="ComfyUI suite config helper")
    p.add_argument("--regenerate", action="store_true",
                   help="Rebuild models_config.yaml from the manifest.")
    p.add_argument("--check", action="store_true",
                   help="Exit 1 if models_config.yaml is out of sync.")
    p.add_argument("--parse-inputs", action="store_true",
                   help="Expand --models into a CI matrix (writes $GITHUB_OUTPUT).")
    p.add_argument("--detect-gpu-arch", action="store_true",
                   help="Detect GPU arch and write gpu_arch= to --output-file.")
    p.add_argument("--models", default="",
                   help="Comma-separated test names for --parse-inputs")
    p.add_argument("--runner-label", default="self-hosted")
    p.add_argument("--trigger", default="workflow_dispatch")
    p.add_argument("--upload", default=None,
                   help="Upload results flag (true/false) for setup job output")
    p.add_argument("--config-file", default=str(MODELS_CONFIG))
    p.add_argument("--output-file", default=None)
    args = p.parse_args()

    manifest = load_manifest()
    text = build_models_config_text(manifest)

    if args.check:
        cur = MODELS_CONFIG.read_text(encoding="utf-8") if MODELS_CONFIG.exists() else ""
        if cur != text:
            print("[drift] models_config.yaml out of sync; run --regenerate",
                  file=sys.stderr)
            sys.exit(1)
        print("[ok] models_config.yaml in sync with manifest")
        sys.exit(0)

    if args.regenerate:
        MODELS_CONFIG.write_text(text, encoding="utf-8")
        print(f"Wrote {MODELS_CONFIG}")
        sys.exit(0)

    if args.parse_inputs:
        sys.exit(parse_inputs(args))

    if args.detect_gpu_arch:
        sys.exit(detect_gpu_arch_for_ci(args.output_file or ""))

    p.error("choose one of --regenerate / --check / --parse-inputs / --detect-gpu-arch")


if __name__ == "__main__":
    main()
