#!/usr/bin/env python3
"""
Classify ComfyUI test failures into actionable categories across platforms.

Used by comfyui_validator.py to refine failure_reason and optionally reclassify
poll timeouts that were caused by server crashes (GPU/OOM/pin errors) as
INFRA_ERROR instead of model FAIL.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# Filesystem / output-path problems (already handled in single_test_protocol).
_IO_ERROR_MARKERS = (
    "permission denied", "errno 13", "eacces",
    "operation not permitted", "errno 1", "eperm",
    "no space left", "errno 28", "enospc",
    "read-only file system", "errno 30", "erofs",
)

# ComfyUI server died or GPU stack aborted — environment, not model content.
_SERVER_CRASH_MARKERS = (
    "fatal python error",
    "pin error",
    "out of memory",
    "oom",
    "cuda error",
    "hip error",
    "hip out of memory",
    "illegal memory access",
    "device-side assert",
    "nccl error",
    "abort",
    "segmentation fault",
    "stack overflow",
)

# Client lost the server mid-poll.
_CONNECTION_LOST_MARKERS = (
    "connection refused",
    "winerror 10061",
    "errno 111",
    "errno 61",
    "remote end closed",
    "broken pipe",
    "connection reset",
    "failed to establish a new connection",
    "urlopen error",
)

# Prompt rejected before execution — usually missing/invalid input assets.
_MISSING_INPUT_MARKERS = (
    "prompt outputs failed validation",
    "invalid image file",
    "loadimage",
    "loadvideo",
    "custom validation failed",
)

_POLL_TIMEOUT_RE = re.compile(
    r"did not complete within \d+s", re.IGNORECASE
)


def _contains_any(text: str, markers: tuple) -> bool:
    low = (text or "").lower()
    return any(m in low for m in markers)


def scan_server_tail(server_tail: str) -> Optional[str]:
    """Return a short crash signature from the server log tail, if any."""
    if not server_tail:
        return None
    low = server_tail.lower()
    for line in reversed(server_tail.splitlines()):
        s = line.strip()
        if not s:
            continue
        ls = s.lower()
        if any(m in ls for m in _SERVER_CRASH_MARKERS):
            return s[:300]
    if _contains_any(low, _SERVER_CRASH_MARKERS):
        for m in _SERVER_CRASH_MARKERS:
            if m in low:
                return f"server crash marker: {m}"
    return None


def classify(
    failure_reason: str = "",
    server_tail: str = "",
    run_errors: Optional[list] = None,
) -> Tuple[str, str, Optional[str]]:
    """Return (category, refined_failure_reason, verdict_override).

    category is one of:
      missing_input, server_crash, connection_lost, poll_timeout,
      io_error, validation_error, model_execution, unknown

    verdict_override is 'INFRA_ERROR' when the failure is environmental, else None.
    """
    combined = failure_reason or ""
    if run_errors:
        combined = combined + " | " + " | ".join(str(e) for e in run_errors)

    if _contains_any(combined, _IO_ERROR_MARKERS):
        return (
            "io_error",
            f"io_error: {failure_reason or combined}",
            "INFRA_ERROR",
        )

    if _contains_any(combined, _MISSING_INPUT_MARKERS):
        return (
            "missing_input",
            f"missing_input_asset: {failure_reason or combined}",
            None,
        )

    crash_sig = scan_server_tail(server_tail)
    if crash_sig:
        refined = f"server_crash: {crash_sig}"
        if failure_reason:
            refined += f" | poll_context: {failure_reason}"
        return "server_crash", refined, "INFRA_ERROR"

    if _contains_any(combined, _CONNECTION_LOST_MARKERS):
        return (
            "connection_lost",
            f"comfyui_server_unreachable: {failure_reason or combined}",
            "INFRA_ERROR",
        )

    if _POLL_TIMEOUT_RE.search(combined):
        return (
            "poll_timeout",
            f"poll_timeout: {failure_reason or combined}",
            None,
        )

    if "http 400" in combined.lower():
        return (
            "validation_error",
            f"prompt_validation: {failure_reason or combined}",
            None,
        )

    if failure_reason:
        return "model_execution", failure_reason, None
    return "unknown", combined or "unknown", None
