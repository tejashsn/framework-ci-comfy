#!/usr/bin/env python3
"""Print fleet-style ASCII execution summary tables to stdout.

Mirrors the ROCmTest end-of-run reporter (Execution Result, System Information,
DETAILED TESTCASES, TESTCASES, Summary Table) so ComfyUI CI logs match the
frameworks-qa-ci / therock_rocm_install format.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from prettytable import PrettyTable

_SUITE_NAME = "comfyui_validation"
_FAIL_STATUSES = {"FAIL", "FAILED"}
_ERROR_STATUSES = {"INFRA_ERROR", "ERROR", "ERRORED"}
_SKIP_STATUSES = {"SKIP", "SKIPPED"}
_PASS_STATUSES = {"PASS", "PASSED"}
_UNSUPPORTED_STATUSES = {"UNSUPPORTED", "NOT_SUPPORTED"}


def _setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("summary")


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _find_metadata(results_root: Path, names: list[str]) -> dict:
    search_roots = [results_root, results_root.parent, Path("logs"), Path(".")]
    for name in names:
        for root in search_roots:
            if not root.exists():
                continue
            for match in sorted(root.glob(f"**/{name}")):
                if ".git" in match.parts:
                    continue
                data = _load_json(match)
                if data:
                    return data
    return {}


def _collect_rows(results_root: Path) -> list[dict]:
    rows: list[dict] = []
    if not results_root.is_dir():
        return rows

    for path in sorted(results_root.rglob("results_*.json")):
        if ".git" in path.parts:
            continue
        doc = _load_json(path)
        if not doc:
            continue
        for item in doc.get("results", []):
            cfg = item.get("test_config", {}) or {}
            status = str(item.get("result_status") or item.get("test_result", "?"))
            metrics = item.get("test_metrics") or []
            primary = next((m for m in metrics if m.get("primary")), None)
            metric_s = "-"
            if primary and primary.get("score") is not None:
                metric_s = f"{primary.get('score')} {primary.get('unit', '')}".strip()
            rows.append({
                "test_name": cfg.get("test_name", path.stem.replace("results_", "")),
                "status": status,
                "arch": cfg.get("detected_gpu_arch") or cfg.get("requested_gpu_arch", ""),
                "minutes": float(item.get("test_execution_time") or 0),
                "metric": metric_s,
                "reason": (item.get("failure_reason") or "").strip(),
                "rocm_version": cfg.get("rocm_version", ""),
            })
    return rows


def _normalize_status(status: str) -> str:
    s = (status or "?").upper()
    if s in _PASS_STATUSES:
        return "PASS"
    if s in _FAIL_STATUSES:
        return "FAIL"
    if s in _ERROR_STATUSES:
        return "ERROR"
    if s in _SKIP_STATUSES:
        return "SKIP"
    if s in _UNSUPPORTED_STATUSES:
        return "UNSUPPORTED"
    return s


def _print_execution_result(logger: logging.Logger, total_mins: float) -> None:
    table = PrettyTable()
    table.field_names = ["total execution time (in mins)"]
    table.add_row([total_mins])
    logger.info("\n" + table.get_string(title="Execution Result"))


def _gpu_arch(bm_config: dict, rows: list[dict]) -> str:
    gpu = bm_config.get("gpu") or {}
    devices = gpu.get("devices") or []
    if devices:
        return "\n".join(str(d) for d in devices)
    for row in rows:
        if row.get("arch"):
            return str(row["arch"])
    return gpu.get("name") or "Unknown"


def _test_environment() -> str:
    try:
        _repo = Path(__file__).resolve().parents[5]
        if str(_repo) not in sys.path:
            sys.path.insert(0, str(_repo))
        from tests.utils.results.handler import ResultsHandler

        return str(ResultsHandler.detect_environment()).upper()
    except Exception:
        return "BM"


def _print_system_information(
    logger: logging.Logger,
    bm_config: dict,
    rows: list[dict],
) -> None:
    table = PrettyTable()
    table.field_names = ["Component", "Information"]

    gpu = bm_config.get("gpu") or {}
    cpu = bm_config.get("cpu") or {}
    os_name = bm_config.get("os") or "Unknown"
    os_version = bm_config.get("os_version") or "Unknown"
    rocm_version = (
        bm_config.get("rocm_version")
        or (rows[0].get("rocm_version") if rows else "")
        or os.environ.get("ROCM_VERSION", "Unknown")
    )

    partition = gpu.get("partition_mode") or "Unknown"
    if partition in ("", "Unknown"):
        partition = "Compute: N/A, Mem: N/A"
    elif not str(partition).startswith("Compute:"):
        partition = f"Compute: {partition}, Mem: N/A"

    rows_to_add = [
        ("rocm version", rocm_version),
        ("OS", os_name.split()[0] if os_name else "Unknown"),
        ("OS Version", os_version),
        ("OS Pretty Name", os_name),
        ("vbios version", gpu.get("vbios") or "Unknown"),
        ("", ""),
        ("sbios version", bm_config.get("sbios") or "Unknown"),
        ("GPU marketing name", gpu.get("marketing_name") or gpu.get("name") or "Unknown"),
        ("GPU ASIC name", gpu.get("name") or "Unknown"),
        ("Device ID", gpu.get("device_id") or "Unknown"),
        ("Revision ID", gpu.get("revision_id") or "Unknown"),
        ("GPU arch", _gpu_arch(bm_config, rows)),
        ("Number of GPUs", str(gpu.get("count") or 0)),
        ("GPU partition", partition),
        ("CPU model name", cpu.get("model") or "Unknown"),
        ("hostname", bm_config.get("hostname") or os.environ.get("RUNNER_HOSTNAME", "Unknown")),
        ("Test Environment Type", _test_environment()),
        ("SMC Firmware version", gpu.get("smc") or "Unknown"),
        ("", ""),
        ("SDMA Firmware version", gpu.get("sdma") or "Unknown"),
        ("", ""),
        ("MEC Firmware version", gpu.get("mec") or "Unable to get mec version"),
        ("TA_XGMI Firmware version", gpu.get("ta_xgmi") or "Unable to get ta_xgmi version"),
        ("RLC Firmware version", gpu.get("rlc") or "Unknown"),
        ("SOS Firmware version", gpu.get("sos") or "Unknown"),
        ("TA RAS Firmware version", gpu.get("ta_ras") or "Unknown"),
        ("VCN Firmware version", gpu.get("vcn") or "Unknown"),
        ("PLDM_BUNDLE Firmware version", gpu.get("plfm_bundle") or "Unable to get PLDM_BUNDLE version"),
        ("IFWI version", "Unable to get IFWI version"),
        ("XGMI setup?", gpu.get("xgmi_type") or "No"),
    ]
    for component, info in rows_to_add:
        table.add_row([component, info])

    logger.info("\n" + table.get_string(title="System Information"))


def _print_detailed_testcases(logger: logging.Logger, rows: list[dict]) -> None:
    need_detail = [r for r in rows if r.get("reason") or _normalize_status(r["status"]) != "PASS"]
    if not need_detail:
        return
    table = PrettyTable()
    table.field_names = ["Test Suite Name", "Test Case", "Test Result", "message"]
    for row in need_detail:
        msg = row.get("reason") or ""
        if len(msg) > 80:
            msg = msg[:77] + "..."
        table.add_row([
            _SUITE_NAME,
            row["test_name"],
            _normalize_status(row["status"]),
            msg,
        ])
    try:
        table.max_width = 80
    except AttributeError:
        pass
    logger.info("\n" + table.get_string(title="DETAILED TESTCASES"))


def _print_testcases(logger: logging.Logger, rows: list[dict]) -> None:
    if not rows:
        return
    table = PrettyTable()
    if len({_SUITE_NAME}) == 1 and len(rows) <= 80:
        table.field_names = ["Test Suite Name", "Test Case", "Test Result"]
        for row in rows:
            table.add_row([_SUITE_NAME, row["test_name"], _normalize_status(row["status"])])
    else:
        table.field_names = ["Test Case", "Test Result"]
        for row in rows:
            table.add_row([row["test_name"], _normalize_status(row["status"])])
    logger.info("\n" + table.get_string(title="TESTCASES"))


def _print_summary_table(logger: logging.Logger, rows: list[dict]) -> None:
    n_cases = len(rows)
    n_passed = sum(1 for r in rows if _normalize_status(r["status"]) == "PASS")
    n_failed = sum(1 for r in rows if _normalize_status(r["status"]) == "FAIL")
    n_errored = sum(1 for r in rows if _normalize_status(r["status"]) == "ERROR")
    n_skipped = sum(1 for r in rows if _normalize_status(r["status"]) == "SKIP")
    n_unsupported = sum(1 for r in rows if _normalize_status(r["status"]) == "UNSUPPORTED")

    if n_cases == 0:
        final = "FAIL"
    elif n_failed or n_errored:
        final = "FAIL"
    elif n_cases > 0 and n_cases == n_skipped:
        final = "SKIP"
    elif n_cases > 0 and n_cases == n_unsupported:
        final = "UNSUPPORTED"
    else:
        final = "PASS"

    table = PrettyTable()
    table.field_names = [
        "No of TestSuites",
        "No of TestCases",
        "Number of Passed testcases",
        "Number of Failed testcases",
        "Number of Errored testcases",
        "Number of Skipped/Not run testcases",
        "Number of Not Supported testcases",
        "Final Result",
    ]
    table.add_row([
        1,
        n_cases,
        n_passed,
        n_failed,
        n_errored,
        n_skipped,
        n_unsupported,
        final,
    ])
    logger.info("\n" + table.get_string(title="Summary Table"))


def print_summary(results_dir: str | Path, *, logs_dir: str | Path = "logs") -> int:
    logger = _setup_logging()
    results_root = Path(results_dir)
    logs_root = Path(logs_dir)

    rows = _collect_rows(results_root)
    bm_config = _find_metadata(results_root, ["system_snapshot.json", "bm_config.json", "host_bm_config.json"])
    if bm_config.get("host"):
        bm_config = bm_config["host"]

    if not bm_config:
        bm_config = _find_metadata(logs_root, ["system_snapshot.json", "bm_config.json", "host_bm_config.json"])
        if bm_config.get("host"):
            bm_config = bm_config["host"]

    total_mins = sum(r.get("minutes", 0) for r in rows)
    if total_mins <= 0 and rows:
        total_mins = sum(r.get("minutes", 0) for r in rows) or 0.0

    _print_execution_result(logger, total_mins)
    if bm_config or rows:
        _print_system_information(logger, bm_config, rows)
    _print_detailed_testcases(logger, rows)
    _print_testcases(logger, rows)
    _print_summary_table(logger, rows)
    return 0 if rows else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Print fleet-style ComfyUI execution summary tables")
    parser.add_argument("--results-dir", default="logs/benchmark_results")
    parser.add_argument("--logs-dir", default="logs", help="Fallback search path for bm_config.json")
    args = parser.parse_args()
    return print_summary(args.results_dir, logs_dir=args.logs_dir)


if __name__ == "__main__":
    raise SystemExit(main())
