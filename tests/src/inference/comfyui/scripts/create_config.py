#!/usr/bin/env python3
"""
create_config.py — framework-ci suite config helper for the ComfyUI suite.

Roles (mirroring vLLM/xDiT scripts of the same name):

  1. --regenerate : rebuild config/models_config.yaml FROM suite_manifest.json
  2. --parse-inputs : expand model selection + runner labels for CI setup job
  3. --check-arch-only : decide whether a test supports the current GPU arch

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
    p0 = [t["test_name"] for t in tests
          if "smoke" in t.get("tags", []) and t.get("enabled", True)]
    p1 = [t["test_name"] for t in tests
          if t.get("automation") == "Automated" and t.get("enabled", True)
          and t["test_name"] not in p0]
    L = []
    L.append("# ComfyUI validation suite config (framework-ci suite style).")
    L.append("# DERIVED from config/suite_manifest.json via scripts/create_config.py.")
    L.append("# The manifest stays authoritative; this is the fleet-facing view.")
    L.append("# Priorities group tests for P0/P1 selection (P0 = smoke/regression).")
    L.append("priorities:")
    L.append("  P0: [" + ", ".join(f'"{n}"' for n in p0) + "]")
    L.append("  P1: [" + ", ".join(f'"{n}"' for n in p1) + "]")
    L.append("")
    L.append("tests:")
    for t in tests:
        L.append(f'  - name: "{t["test_name"]}"')
        L.append(f'    tms_key: {json.dumps(t.get("tms_key"))}')
        L.append(f'    tags: [{", ".join(chr(34)+x+chr(34) for x in t.get("tags", []))}]')
        L.append(f'    gpu_arch: [{", ".join(chr(34)+x+chr(34) for x in t.get("gpu_arch", []))}]')
        L.append(f'    os: [{", ".join(chr(34)+x+chr(34) for x in t.get("os", []))}]')
        L.append(f'    timeout_minutes: {t.get("timeout_minutes")}')
        L.append(f'    expected_output_type: "{t.get("expected_output_type", "image")}"')
        L.append(f'    enabled: {str(t.get("enabled", True)).lower()}')
        L.append('    command: "python3 tests/src/inference/comfyui/comfyui_benchmark.py '
                 f'--model {t["test_name"]}"')
    return "\n".join(L) + "\n"


def expand_selection(selection, models_cfg):
    """Turn 'P0'/'P1'/'ALL' or comma-separated names into a list of test names."""
    sel = (selection or "P0").strip()
    prios = models_cfg.get("priorities", {})
    if sel.upper() in ("P0", "P1", "P2"):
        return list(prios.get(sel.upper(), []))
    if sel.upper() == "ALL":
        return [t["name"] for t in models_cfg.get("tests", [])
                if t.get("enabled", True)]
    return [s.strip() for s in sel.split(",") if s.strip()]


def lookup_test_config(test_name: str, models_cfg: dict) -> dict | None:
    for t in models_cfg.get("tests", []):
        if t.get("name") == test_name:
            return t
    return None


def check_arch_compatibility(
    test_name: str,
    config_file: str,
    output_file: str = "",
) -> int:
    """Check if test is compatible with current GPU architecture."""
    gpu_arch = detect_gpu_arch()
    models_cfg = load_models_config(config_file)
    test_cfg = lookup_test_config(test_name, models_cfg)
    supported_arch = (test_cfg or {}).get("gpu_arch", [])

    if supported_arch and gpu_arch not in supported_arch and gpu_arch != "unknown":
        supported = "false"
        skip_reason = f"Test requires {supported_arch}, current GPU is {gpu_arch}"
        print(f"⏭️  {test_name} requires {supported_arch}, current GPU: {gpu_arch}")
    else:
        supported = "true"
        skip_reason = ""
        if supported_arch:
            print(f"✓ {test_name} supports {gpu_arch} (allowed: {supported_arch})")
        else:
            print(f"✓ {test_name} has no arch filter (gpu={gpu_arch})")

    if output_file:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"supported={supported}\n")
            f.write(f"gpu_arch={gpu_arch}\n")
            f.write(f"skip_reason={skip_reason}\n")
    return 0


def parse_inputs(args) -> int:
    """Expand models + runner labels for GitHub Actions matrix."""
    models_cfg = load_models_config(args.config_file)
    selected = expand_selection(args.models, models_cfg)
    labels = [x.strip() for x in (args.runner_label or "self-hosted").split(",") if x.strip()]
    if not labels:
        labels = ["self-hosted"]

    trigger_type = args.trigger or "workflow_dispatch"
    docker_image = args.docker_image or "rocm/pytorch:latest"
    upload_results = str(args.upload).lower() if args.upload is not None else "true"

    model_count = len(selected)
    label_count = len(labels)
    total_jobs = model_count * label_count

    print(f"Trigger: {trigger_type}")
    print(f"Models ({model_count}): {json.dumps(selected)}")
    print(f"Runner labels ({label_count}): {json.dumps(labels)}")
    print(f"Total jobs: {model_count} models × {label_count} labels = {total_jobs} jobs")
    print(f"Upload results: {upload_results}")
    print(f"Docker image: {docker_image}")

    if args.output_file:
        with open(args.output_file, "a", encoding="utf-8") as f:
            f.write(f"models={json.dumps(selected)}\n")
            f.write(f"model_count={model_count}\n")
            f.write(f"trigger_type={trigger_type}\n")
            f.write(f"runner_labels={json.dumps(labels)}\n")
            f.write(f"runner_label_count={label_count}\n")
            f.write(f"docker_image={docker_image}\n")
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
    p.add_argument("--check-arch-only", action="store_true",
                   help="Only check arch compatibility and exit.")
    p.add_argument("--models", default="P0")
    p.add_argument("--model", default="", help="Test name for --check-arch-only")
    p.add_argument("--runner-label", default="self-hosted")
    p.add_argument("--trigger", default="workflow_dispatch")
    p.add_argument("--docker-image", default="rocm/pytorch:latest")
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

    if args.check_arch_only:
        if not args.model:
            print("Error: --model is required for --check-arch-only", file=sys.stderr)
            sys.exit(1)
        sys.exit(check_arch_compatibility(args.model, args.config_file, args.output_file or ""))

    p.error("choose one of --regenerate / --check / --parse-inputs / --check-arch-only")


if __name__ == "__main__":
    main()
