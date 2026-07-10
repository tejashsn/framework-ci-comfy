#!/usr/bin/env python3
"""Consolidate downloaded CI artifacts into results_summary.md.

The fleet's ``config_consolidator`` expects vLLM-style flat benchmark JSON and a
large YAML schema. ComfyUI writes nested ``results_*.json`` files instead, so
this script provides a suite-local consolidator with the same job contract:
scan ``all_results/``, emit ``results_summary.md``, exit non-zero on hard fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FAIL_STATUSES = {"FAIL", "INFRA_ERROR"}
SKIP_STATUSES = {"SKIP"}
ALL_SKIP_BANNER = (
    "## ALL RESULTS SKIPPED — check runner capability / gating config"
)


def _collect_rows(results_root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(results_root.rglob("results_*.json")):
        if ".git" in path.parts:
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for item in doc.get("results", []):
            cfg = item.get("test_config", {}) or {}
            status = item.get("result_status") or item.get("test_result", "?")
            metrics = item.get("test_metrics") or []
            primary = next((m for m in metrics if m.get("primary")), None)
            metric_s = "-"
            if primary and primary.get("score") is not None:
                metric_s = f"{primary.get('score')} {primary.get('unit', '')}".strip()
            rows.append({
                "artifact": path.parent.name,
                "test_name": cfg.get("test_name", path.stem),
                "status": status,
                "arch": cfg.get("detected_gpu_arch") or cfg.get("requested_gpu_arch", ""),
                "minutes": item.get("test_execution_time", ""),
                "metric": metric_s,
                "reason": (item.get("failure_reason") or "")[:120],
            })
    return rows


def _write_summary(rows: list[dict], out_path: Path, *, all_skip: bool = False) -> str:
    lines = [
        "# ComfyUI Validation — consolidated results",
        "",
    ]
    if all_skip:
        lines.extend([ALL_SKIP_BANNER, ""])
    lines.extend([
        f"**Total rows:** {len(rows)}",
        "",
        "| Artifact | Test | Status | Arch | Minutes | Metric | Reason |",
        "|----------|------|--------|------|---------|--------|--------|",
    ])
    for r in rows:
        lines.append(
            f"| {r['artifact']} | {r['test_name']} | {r['status']} | {r['arch']} | "
            f"{r['minutes']} | {r['metric']} | {r['reason']} |"
        )
    text = "\n".join(lines) + "\n"
    out_path.write_text(text, encoding="utf-8")
    return text


def main() -> int:
    p = argparse.ArgumentParser(description="Consolidate ComfyUI CI artifacts")
    p.add_argument("--results-dir", default="all_results")
    p.add_argument("--output", default="results_summary.md")
    p.add_argument("--fail-on-error", action="store_true")
    args = p.parse_args()

    root = Path(args.results_dir)
    if not root.is_dir():
        print(f"No results directory: {root}")
        _write_summary([], Path(args.output))
        return 0

    rows = _collect_rows(root)
    all_skip = bool(rows) and all(
        str(r["status"]).upper() in SKIP_STATUSES for r in rows
    )
    _write_summary(rows, Path(args.output), all_skip=all_skip)
    print(f"Wrote {args.output} ({len(rows)} row(s))")
    if all_skip:
        print(f"WARNING: {ALL_SKIP_BANNER}")

    hard_fails = [r for r in rows if str(r["status"]).upper() in FAIL_STATUSES]
    if hard_fails:
        print(f"Found {len(hard_fails)} FAIL/INFRA_ERROR row(s)")
        if args.fail_on_error:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
