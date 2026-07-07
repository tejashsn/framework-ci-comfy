#!/usr/bin/env python3
"""
create_config.py — framework-ci suite config helper for the ComfyUI suite.

Two roles, mirroring the vLLM/xDiT scripts of the same name:

  1. --regenerate : rebuild config/models_config.yaml FROM the authoritative
     config/suite_manifest.json (derived view; manifest stays source of truth).
  2. --parse-inputs : expand a model selection string (P0/P1/ALL or comma names)
     into a JSON matrix for the CI `setup` job, writing to $GITHUB_OUTPUT.

Also --check verifies models_config.yaml is in sync with the manifest (drift
guard used by the unit tests).
"""

import argparse
import json
import sys
from pathlib import Path

import yaml  # type: ignore[import-not-found]

SUITE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = SUITE_DIR / "config"
MANIFEST = CONFIG_DIR / "suite_manifest.json"
MODELS_CONFIG = CONFIG_DIR / "models_config.yaml"


def load_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


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


def main():
    p = argparse.ArgumentParser(description="ComfyUI suite config helper")
    p.add_argument("--regenerate", action="store_true",
                   help="Rebuild models_config.yaml from the manifest.")
    p.add_argument("--check", action="store_true",
                   help="Exit 1 if models_config.yaml is out of sync.")
    p.add_argument("--parse-inputs", action="store_true",
                   help="Expand --models into a CI matrix (writes $GITHUB_OUTPUT).")
    p.add_argument("--models", default="P0")
    p.add_argument("--runner-label", default="self-hosted")
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
        models_cfg = yaml.safe_load(Path(args.config_file).read_text(encoding="utf-8"))
        names = expand_selection(args.models, models_cfg)
        labels = [x.strip() for x in args.runner_label.split(",") if x.strip()]
        out = {
            "models": json.dumps(names),
            "model_count": str(len(names)),
            "runner_labels": json.dumps(labels),
            "runner_label_count": str(len(labels)),
        }
        target = args.output_file
        if target:
            with open(target, "a", encoding="utf-8") as f:
                for k, v in out.items():
                    f.write(f"{k}={v}\n")
        else:
            print(json.dumps(out, indent=2))
        sys.exit(0)

    p.error("choose one of --regenerate / --check / --parse-inputs")


if __name__ == "__main__":
    main()
