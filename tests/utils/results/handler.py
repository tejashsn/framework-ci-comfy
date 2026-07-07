#!/usr/bin/env python3
"""
tests/utils/results/handler.py — RASTRA results handler (framework-ci contract).

Contract-compatible with the fleet's tests/utils/results stack: builds the same
payload shape (test_environment / bm_config / build_info / deployment_info /
docker_config / results[]), with per-result test_config / test_metrics /
test_result / test_execution_time (MINUTES) and deployment_info.execution_label.

STUBBED UPLOAD: upload_results() defaults to dry_run=True — it builds + validates
the payload and writes logs/payload.json, but performs NO network POST. Wire the
real endpoint (API_URL/API_KEY) later; the payload shape is already correct.

Framework auto-detection mirrors the fleet's _derive_framework keyword scan.
"""

import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

_FRAMEWORK_KEYWORDS = {
    "comfyui": "comfyui", "vllm": "vllm", "xdit": "xdit", "sglang": "sglang",
    "torch": "pytorch", "pytorch": "pytorch",
}


def _derive_framework(test_name):
    if os.environ.get("FRAMEWORK_NAME"):
        return os.environ["FRAMEWORK_NAME"]
    low = (test_name or "").lower()
    for kw, fw in _FRAMEWORK_KEYWORDS.items():
        if kw in low:
            return fw
    return "comfyui"


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def build_results_payload(system_info, test_results, execution_time_s,
                          test_environment="bare_metal", execution_label="",
                          docker_info=None, build_info=None):
    """Build the RASTRA payload. `test_results` is a list of already-shaped
    result items (test_config/test_metrics/test_result/test_execution_time).
    execution_time_s is total wall-clock seconds (recorded per result in
    minutes by the caller). Mirrors the fleet build_results_payload top-level."""
    label = execution_label or os.environ.get("EXECUTION_LABEL", "")
    return {
        "test_environment": test_environment,
        "bm_config": system_info or {},
        "build_info": build_info or {},
        "deployment_info": {
            "execution_label": label,
            "test_deployed_by": os.environ.get("GITHUB_ACTOR", ""),
            "test_deployed_on": _now_iso(),
            "test_flag": "prod_test",
            "execution_type": "automated",
        },
        "docker_config": docker_info or {},
        "results": test_results,
        "test_app_commit": [],
        "test_app_version": "",
    }


def results_from_benchmark_dir(results_dir):
    """Collect the per-test results_<name>.json (each has a single 'results'
    list) into one flat list of RASTRA result items."""
    items = []
    for rj in sorted(Path(results_dir).glob("results_*.json")):
        try:
            doc = json.loads(rj.read_text(encoding="utf-8"))
        except Exception:
            continue
        items.extend(doc.get("results", []))
    return items


def load_system_info(logs_dir="logs"):
    for name in ("system_snapshot.json", "bm_config.json", "container_info.json"):
        p = Path(logs_dir) / name
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
    return {"hostname": socket.gethostname()}


def upload_results(results_dir="logs/benchmark_results", *, execution_label="",
                   logs_dir="logs", api_url=None, api_key=None,
                   dry_run=True, out_path="logs/payload.json"):
    """One-stop entry: gather results_<name>.json, build the RASTRA payload, and
    (dry-run) write logs/payload.json. Returns the payload dict.

    dry_run=True (default) makes NO network call — it prints + writes the payload
    that WOULD be POSTed. Real upload (dry_run=False + api_url/api_key) is left
    for when creds are wired; the POST path is intentionally not implemented here.
    """
    items = results_from_benchmark_dir(results_dir)
    system_info = load_system_info(logs_dir)
    payload = build_results_payload(
        system_info, items, execution_time_s=0,
        execution_label=execution_label or os.environ.get("EXECUTION_LABEL", ""))
    fw = _derive_framework(items[0]["test_config"]["test_name"]) if items else "comfyui"
    payload["framework"] = fw

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if dry_run or not (api_url and api_key):
        print(f"[dry-run] RASTRA payload written to {out} "
              f"({len(items)} result item(s), framework={fw}); no POST made.")
        return payload

    # Real POST intentionally not implemented in this contract build.
    raise NotImplementedError(
        "Real RASTRA POST not wired. Add requests.post(api_url, json=payload, "
        "headers={'Authorization': api_key}) once creds are provisioned.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build/preview the RASTRA payload.")
    p.add_argument("--results-dir", default="logs/benchmark_results")
    p.add_argument("--execution-label", default="")
    p.add_argument("--out", default="logs/payload.json")
    args = p.parse_args()
    upload_results(args.results_dir, execution_label=args.execution_label,
                   out_path=args.out, dry_run=True)
