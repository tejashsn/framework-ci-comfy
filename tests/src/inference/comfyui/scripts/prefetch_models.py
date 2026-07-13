#!/usr/bin/env python3
"""Bulk-prefetch model weights for the selected tests, once per runner.

The benchmark matrix downloads each test's weights inside its own job. For large
multi-file models (Flux, Wan, LTX) that cold-start per job can fail fast or time
out. This script runs once (in the prefetch job) BEFORE the matrix: it unions the
missing weights across all selected tests and downloads them in one pass, warming
the shared ``COMFYUI_PATH/models`` cache so matrix jobs reuse them.

It reuses the existing infrastructure and does not change fetch logic:
  * model_check.missing_models() to find what's absent per workflow
  * fetch_models.ensure_missing() to download (HF / Artifactory, same as per-test)

Usage:
  python prefetch_models.py --models-json '["comfyui_stable_diffusion_2_1", ...]'
  python prefetch_models.py --models comfyui_sd3_medium_fp16,comfyui_flux1_dev
  python prefetch_models.py --models P0 --comfyui_path ~/ComfyUI
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SUITE_DIR = Path(__file__).resolve().parent.parent
EXECUTORS_DIR = SUITE_DIR / "executors"
CONFIG_DIR = SUITE_DIR / "config"
sys.path.insert(0, str(EXECUTORS_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import create_config  # noqa: E402
import fetch_models  # noqa: E402
import fetch_workflow_inputs  # noqa: E402
import model_check  # noqa: E402
import workflow_input_check  # noqa: E402


def _log(msg: str, *, error: bool = False) -> None:
    """Print with a [prefetch] prefix; emit a GH ::error:: annotation on error."""
    if error:
        print(f"::error::[prefetch] {msg}", flush=True)
    else:
        print(f"[prefetch] {msg}", flush=True)


def _load_manifest() -> dict:
    return json.loads((CONFIG_DIR / "suite_manifest.json").read_text(encoding="utf-8"))


def _workflow_for(test_name: str, manifest: dict) -> str | None:
    for t in manifest.get("tests", []):
        if t.get("test_name") == test_name:
            return t.get("workflow")
    return None


def resolve_comfyui_path(explicit: str = "") -> str:
    """arg -> $COMFYUI_PATH -> comfyui_runtime.discover_comfyui_path()."""
    if explicit:
        return explicit
    if os.environ.get("COMFYUI_PATH"):
        return os.environ["COMFYUI_PATH"]
    try:
        import comfyui_runtime
        found = comfyui_runtime.discover_comfyui_path({})
        return str(found) if found else ""
    except Exception:
        return ""


def collect_missing_for_tests(test_names, comfy_path):
    """Union of (filename, subdirs) missing across every selected test.

    Deduplicated by filename so a weight shared by several workflows (e.g.
    t5xxl_fp16.safetensors) is fetched once."""
    manifest = _load_manifest()
    seen: dict[str, list[str]] = {}
    for name in test_names:
        workflow = _workflow_for(name, manifest)
        if not workflow:
            _log(f"skip {name}: not in manifest")
            continue
        # Manifest workflow paths are relative to the suite dir; make absolute so
        # the check works regardless of the caller's cwd.
        wf_path = workflow
        if not Path(wf_path).is_absolute():
            wf_path = str(SUITE_DIR / workflow)
        try:
            missing = model_check.missing_models(wf_path, comfy_path)
        except Exception as e:  # never let one bad workflow abort the union
            _log(f"warn {name}: could not check models ({e})")
            continue
        for filename, subdirs in missing:
            if filename not in seen:
                seen[filename] = subdirs
    return [(fn, sd) for fn, sd in seen.items()]


def collect_missing_inputs_for_tests(test_names, comfy_path):
    """Union of input filenames missing across every selected test."""
    manifest = _load_manifest()
    seen: set[str] = set()
    for name in test_names:
        workflow = _workflow_for(name, manifest)
        if not workflow:
            continue
        wf_path = workflow
        if not Path(wf_path).is_absolute():
            wf_path = str(SUITE_DIR / workflow)
        try:
            missing = workflow_input_check.missing_inputs(wf_path, comfy_path)
        except Exception as e:
            _log(f"warn {name}: could not check workflow inputs ({e})")
            continue
        seen.update(missing)
    return sorted(seen)


def prefetch_inputs(test_names, comfy_path, fail_on_error: bool = True) -> int:
    """Prefetch workflow input assets. Returns exit code (0 = ok)."""
    if not fetch_workflow_inputs.auto_fetch_enabled():
        _log("AUTO_FETCH_MODELS disabled — skipping workflow input prefetch")
        return 0
    if not comfy_path:
        _log("COMFYUI_PATH not resolved — cannot prefetch workflow inputs",
             error=fail_on_error)
        return 1 if fail_on_error else 0

    missing = collect_missing_inputs_for_tests(test_names, comfy_path)
    if not missing:
        _log(f"all workflow inputs present for {len(test_names)} test(s)")
        return 0

    _log(f"{len(missing)} workflow input(s) to fetch: "
         + ", ".join(missing))
    result = fetch_workflow_inputs.ensure_missing(missing, comfy_path)
    still = collect_missing_inputs_for_tests(test_names, comfy_path)
    if still:
        _log(f"workflow inputs still missing: {', '.join(still)}",
             error=fail_on_error)
        for err in result.errors:
            _log(f"  reason: {err}")
        return 1 if fail_on_error else 0
    _log(f"workflow input prefetch complete — "
         f"{len(result.downloaded)} downloaded")
    return 0


def prefetch_tests(test_names, comfy_path, fail_on_error: bool = True) -> int:
    """Prefetch all missing weights for the given tests. Returns an exit code.

    0 = nothing to do / all present after fetch. 1 = weights still missing (only
    when fail_on_error). Honours AUTO_FETCH_MODELS: if disabled, this is a no-op.
    """
    if not fetch_models.auto_fetch_enabled():
        _log("AUTO_FETCH_MODELS disabled — nothing to prefetch")
        return 0

    if not comfy_path:
        _log("COMFYUI_PATH not resolved — cannot prefetch", error=fail_on_error)
        return 1 if fail_on_error else 0

    missing = collect_missing_for_tests(test_names, comfy_path)
    if not missing:
        _log(f"all weights present for {len(test_names)} test(s) — nothing to fetch")
        return 0

    _log(f"{len(missing)} unique weight(s) to fetch for {len(test_names)} test(s): "
         + ", ".join(sorted(m[0] for m in missing)))
    result = fetch_models.ensure_missing(missing, comfy_path)

    for fn in result.downloaded:
        _log(f"OK {fn}")
    for fn in result.already_present:
        _log(f"present {fn}")

    # Re-check to confirm nothing is still missing after the fetch attempt.
    still = collect_missing_for_tests(test_names, comfy_path)
    if still:
        names = ", ".join(sorted(m[0] for m in still))
        _log(f"still missing after fetch: {names}", error=fail_on_error)
        for err in result.errors:
            _log(f"  reason: {err}")
        return 1 if fail_on_error else 0

    _log(f"prefetch complete — {len(result.downloaded)} downloaded, "
         f"{len(result.already_present)} already present")

    input_rc = prefetch_inputs(test_names, comfy_path, fail_on_error=fail_on_error)
    if input_rc != 0:
        return input_rc
    return 0


def _parse_names(args) -> list[str]:
    if args.models_json:
        data = json.loads(args.models_json)
        if not isinstance(data, list):
            raise ValueError("--models-json must be a JSON array of test names")
        return [str(x) for x in data]
    if args.models:
        models_cfg = create_config.load_models_config(
            str(CONFIG_DIR / "models_config.yaml")
        ) if hasattr(create_config, "load_models_config") else None
        if models_cfg is not None:
            return create_config.expand_selection(args.models, models_cfg)
        # Fallback: plain comma split (expand_selection needs the config).
        return [s.strip() for s in args.models.split(",") if s.strip()]
    raise ValueError("provide --models-json or --models")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Prefetch ComfyUI model weights")
    p.add_argument("--models-json", default="",
                   help="JSON array of manifest test_name values (from setup job)")
    p.add_argument("--models", default="",
                   help="Comma-separated test names (or a selector like P0)")
    p.add_argument("--comfyui_path", default="",
                   help="ComfyUI install dir (else $COMFYUI_PATH / discovery)")
    p.add_argument("--no-fail", action="store_true",
                   help="Do not exit 1 if weights remain missing (warn only)")
    args = p.parse_args(argv)

    try:
        names = _parse_names(args)
    except (ValueError, json.JSONDecodeError) as e:
        _log(str(e), error=True)
        return 1

    comfy_path = resolve_comfyui_path(args.comfyui_path)
    _log(f"ComfyUI path: {comfy_path or '(unresolved)'}")
    return prefetch_tests(names, comfy_path, fail_on_error=not args.no_fail)


if __name__ == "__main__":
    sys.exit(main())
