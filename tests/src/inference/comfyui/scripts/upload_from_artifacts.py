#!/usr/bin/env python3
"""ComfyUI result upload: read results_*.json and push them to the RASTRA DB.

Mirrors tests/src/inference/xdit/scripts/upload_from_artifacts.py. ComfyUI's
``comfyui_benchmark.py`` writes each result in the ROCm-QA shape; this script
maps those entries onto the shared ``tests/utils/results`` stack with schema
validation and optional Artifactory log URLs.

Usage:
    python tests/src/inference/comfyui/scripts/upload_from_artifacts.py \
        --results-dir logs/benchmark_results
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: requests not installed. Run: pip install requests")
    sys.exit(1)

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

PERF_INGEST_URL = "http://api.rastra-test-results.amd.com/api/v2/perf/rastra-test-results"

try:
    from tests.utils.results.api import build_results_payload, validate_payload, ResultsAPI
    from tests.utils.results.handler import ResultsHandler, build_artifactory_info

    _HAVE_FRAMEWORK = True
except Exception as _import_err:  # noqa: BLE001
    _HAVE_FRAMEWORK = False
    _FRAMEWORK_IMPORT_ERR = _import_err


def load_json(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {path}: {e}")
        return None


def find_result_files(results_dir: Path):
    skip = {
        "bm_config.json", "host_bm_config.json", "payload.json",
        "system_snapshot.json", "container_info.json", "benchmark_summary.json",
        "benchmark_config.yaml",
    }
    return [
        f for f in results_dir.glob("**/results_*.json")
        if f.name not in skip and ".git" not in f.parts
    ]


def _find_metadata_file(results_dir: Path, names) -> dict:
    search_roots = [results_dir, results_dir.parent, Path("logs"), Path(".")]
    for name in names:
        for root in search_roots:
            if not root.exists():
                continue
            matches = sorted(root.glob(f"**/{name}"))
            matches = [m for m in matches if ".git" not in m.parts]
            if matches:
                data = load_json(matches[0])
                if data:
                    print(f"  metadata: {name} <- {matches[0]}")
                    return data
    return {}


def _clean_metrics(test_metrics) -> list:
    cleaned = []
    for m in test_metrics or []:
        if not isinstance(m, dict):
            continue
        score = m.get("score")
        if score is None:
            continue
        try:
            score = float(score)
        except (TypeError, ValueError):
            continue
        cleaned.append({
            "metric_name": m.get("metric_name", "latency"),
            "score": round(score, 4),
            "unit": m.get("unit", ""),
            "flag": m.get("flag", "L"),
            "primary": bool(m.get("primary", True)),
        })
    return cleaned


def collect_test_results(results_dir: Path, env_deps: list) -> list:
    files = find_result_files(results_dir)
    print(f"Found {len(files)} result file(s)")

    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    log_path = os.environ.get("ARTIFACT_URL", "")
    test_results = []

    for f in files:
        data = load_json(f)
        if not data:
            continue
        for entry in data.get("results", []):
            if not isinstance(entry, dict):
                continue
            test_config = dict(entry.get("test_config", {}) or {})
            if not test_config.get("python_version"):
                test_config["python_version"] = python_version
            if not test_config.get("environment_dependencies"):
                test_config["environment_dependencies"] = env_deps
            if entry.get("result_status"):
                test_config["result_status"] = entry["result_status"]
            if entry.get("failure_reason"):
                test_config["failure_reason"] = entry["failure_reason"]
            params = test_config.pop("test_parameters", None)
            if isinstance(params, dict):
                for key, value in params.items():
                    test_config.setdefault(key, value)
            for drop_key in ("ngpu", "device", "args"):
                test_config.pop(drop_key, None)
            deps = test_config.get("environment_dependencies")
            if isinstance(deps, list):
                test_config["environment_dependencies"] = [
                    d for d in deps
                    if isinstance(d, dict)
                    and str(d.get("version", "")).strip().lower() not in ("", "n/a", "unknown")
                ]

            success = str(entry.get("test_result", "PASS")).upper() == "PASS"
            metrics = _clean_metrics(entry.get("test_metrics")) if success else []
            duration_seconds = float(entry.get("test_execution_time", 0) or 0) * 60.0

            test_results.append({
                "success": success,
                "duration": duration_seconds,
                "start_time": entry.get("test_start_time") or datetime.now().isoformat(),
                "log_path": entry.get("test_log") or log_path,
                "test_config": test_config,
                "metrics": metrics,
            })
        print(f"  ✓ {f.name}")
    return test_results


def _build_payload_framework(test_results: list, bm_config: dict, container_info: dict) -> dict:
    docker_info = (
        ResultsHandler.build_docker_info_dict(container_info=container_info)
        if container_info else None
    )
    rocm_info = ResultsHandler.build_rocm_info(
        bm_config=bm_config or None, container_info=container_info or None
    )
    deployment_info = ResultsHandler.build_deployment_info(
        deployed_by=os.environ.get("GITHUB_ACTOR", "ci"),
        execution_label=os.environ.get("EXECUTION_LABEL", ""),
    )
    return build_results_payload(
        system_info=bm_config or {},
        test_results=test_results,
        execution_time=datetime.now().isoformat(),
        test_environment=ResultsHandler.detect_environment(),
        build_info=rocm_info,
        deployment_info=deployment_info,
        docker_info=docker_info,
    )


def build_payload_from_dir(
    results_dir: str | Path,
    *,
    execution_label: str = "",
    logs_dir: str | Path = "logs",
) -> tuple[dict, list]:
    """Build a validated RASTRA payload from a results directory (no POST)."""
    results_dir = Path(results_dir)
    logs_dir = Path(logs_dir)

    bm_config = _find_metadata_file(
        results_dir, ["system_snapshot.json", "bm_config.json", "host_bm_config.json"]
    )
    if bm_config.get("host"):
        container_info = bm_config.get("container") or {}
        bm_config = bm_config["host"]
    else:
        container_info = {}
    if not container_info:
        container_info = _find_metadata_file(results_dir, ["container_info.json"])
        if not container_info:
            container_info = _find_metadata_file(logs_dir, ["container_info.json"])

    env_deps = container_info.get("environment_dependencies", []) if container_info else []
    test_results = collect_test_results(results_dir, env_deps=env_deps)
    if not test_results:
        raise ValueError(f"No results_*.json found under {results_dir}")

    if _HAVE_FRAMEWORK:
        first_test_name = test_results[0]["test_config"].get("test_name", "")
        try:
            artifactory_url = build_artifactory_info(test_name=first_test_name, framework="comfyui")
        except Exception as e:
            print(f"Warning: could not build Artifactory URL: {e}")
            artifactory_url = ""
        if artifactory_url:
            for entry in test_results:
                entry["log_path"] = artifactory_url

    if execution_label:
        os.environ.setdefault("EXECUTION_LABEL", execution_label)

    if not _HAVE_FRAMEWORK:
        raise RuntimeError(f"Results stack unavailable: {_FRAMEWORK_IMPORT_ERR}")

    payload = _build_payload_framework(test_results, bm_config, container_info)
    if not validate_payload(payload):
        raise ValueError("Payload validation failed (see logs/failed_payload.json if written)")
    return payload, test_results


def save_payload(payload: dict, out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="ComfyUI result upload from artifacts")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--execution-label", default="")
    parser.add_argument("--force", action="store_true", help="Upload even if payload.json exists")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-payload", help="Save payload to file")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    logs_dir = results_dir.parent if results_dir.name == "benchmark_results" else Path("logs")

    if not args.force and list(Path("logs").glob("**/payload.json")):
        print("✅ payload.json found - upload already happened. Use --force to override.")
        return 0

    if not _HAVE_FRAMEWORK:
        print(f"Shared results stack unavailable ({_FRAMEWORK_IMPORT_ERR})")
        return 1

    print("Using shared results stack (tests.utils.results) for payload build.")

    try:
        payload, test_results = build_payload_from_dir(
            results_dir,
            execution_label=args.execution_label,
            logs_dir=logs_dir,
        )
    except ValueError as e:
        print(f"❌ {e}")
        return 1

    print("✓ Payload validated")
    print(f"\nParsed {len(test_results)} result row(s)")

    payload_name = "logs/payload.json" if Path("logs").is_dir() else "payload.json"
    for target in ([args.save_payload] if args.save_payload else []) + [payload_name]:
        save_payload(payload, target)

    if args.dry_run:
        print(f"[DRY RUN] Would upload {len(test_results)} results")
        return 0

    api_url = os.environ.get("API_URL") or PERF_INGEST_URL
    api_key = os.environ.get("API_KEY", "") or os.environ.get("API_TOKEN", "")
    print(f"Uploading to perf API: {api_url}")

    ok = ResultsAPI(api_url, api_key).submit_results(payload)
    print("✅ Upload successful!" if ok else "❌ Upload failed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
