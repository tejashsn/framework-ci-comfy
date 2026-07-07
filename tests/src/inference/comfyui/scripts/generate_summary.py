#!/usr/bin/env python3
"""
generate_summary.py — write a markdown summary table from the ComfyUI suite's
results_<name>.json + benchmark_summary.json, appended to $GITHUB_STEP_SUMMARY
(or stdout). Mirrors the vLLM/xDiT generate_summary role.
"""

import argparse
import json
import os
from pathlib import Path


def collect(results_dir):
    rows = []
    for rj in sorted(Path(results_dir).glob("results_*.json")):
        try:
            doc = json.loads(rj.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in doc.get("results", []):
            cfg = item.get("test_config", {})
            metrics = item.get("test_metrics", [])
            primary = next((m for m in metrics if m.get("primary")), None)
            rows.append({
                "name": cfg.get("test_name", rj.stem),
                "status": item.get("result_status", item.get("test_result", "?")),
                "arch": cfg.get("detected_gpu_arch") or cfg.get("requested_gpu_arch", ""),
                "minutes": item.get("test_execution_time"),
                "metric": (f'{primary.get("score")} {primary.get("unit")}'
                           if primary and primary.get("score") is not None else "-"),
                "reason": item.get("failure_reason", ""),
            })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="logs/benchmark_results")
    p.add_argument("--mode", default="github", choices=["github", "stdout"])
    args = p.parse_args()

    rows = collect(args.results_dir)
    lines = ["| Test | Status | Arch | Minutes | Metric | Reason |",
             "|------|--------|------|---------|--------|--------|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['status']} | {r['arch']} | "
                     f"{r['minutes']} | {r['metric']} | {r['reason'][:60]} |")
    table = "\n".join(lines) + "\n"

    target = os.environ.get("GITHUB_STEP_SUMMARY") if args.mode == "github" else None
    if target:
        with open(target, "a", encoding="utf-8") as f:
            f.write(f"## ComfyUI Validation ({len(rows)} test(s))\n\n{table}\n")
    else:
        print(table)


if __name__ == "__main__":
    main()
