#!/usr/bin/env python3
"""
Manifest-driven test suitability checks (any arch / OS / platform).

Returns SKIP when the current machine cannot honestly run a test, instead of
burning the full timeout and recording a misleading FAIL.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple


def _normalize_arch(arch: str) -> str:
    return (arch or "").strip().lower()


def effective_min_vram_mb(test: dict, gpu_arch: str) -> int | None:
    """Resolve the VRAM gate for this test on the given GPU arch.

    Per-arch overrides (min_vram_mb_by_arch) take precedence over the base
    min_vram_mb, mirroring timeout_minutes_by_arch. Returns None when no gate.
    """
    by_arch = test.get("min_vram_mb_by_arch") or {}
    arch = _normalize_arch(gpu_arch)
    if arch and arch in by_arch:
        try:
            return int(by_arch[arch])
        except (TypeError, ValueError):
            pass
    min_mb = test.get("min_vram_mb")
    if min_mb is None:
        return None
    try:
        return int(min_mb)
    except (TypeError, ValueError):
        return None


def check_arch(test: dict, gpu_arch: str) -> Tuple[bool, str]:
    """True if the test may run on this GPU arch."""
    arch = _normalize_arch(gpu_arch)
    require = [_normalize_arch(a) for a in (test.get("require_archs") or []) if a]
    exclude = [_normalize_arch(a) for a in (test.get("exclude_archs") or []) if a]
    if require and arch and arch not in require:
        return False, f"arch {gpu_arch} not in require_archs {require}"
    if arch and arch in exclude:
        return False, f"arch {gpu_arch} listed in exclude_archs"
    return True, ""


def check_os(test: dict, os_family: str) -> Tuple[bool, str]:
    allowed = [o.lower() for o in test.get("os", [])]
    fam = (os_family or "").lower()
    if allowed and fam and fam not in allowed:
        return False, f"os {os_family} not in {allowed}"
    return True, ""


def check_vram(test: dict, gpu_arch: str, python_exe: str) -> Tuple[bool, str]:
    """SKIP when free VRAM is below manifest min_vram_mb (best-effort)."""
    if os.environ.get("COMFYUI_IGNORE_VRAM", "").strip().lower() in ("1", "true", "yes", "on"):
        return True, ""
    min_mb = effective_min_vram_mb(test, gpu_arch)
    if not min_mb:
        return True, ""
    try:
        from preflight_check import check_vram
        result = check_vram(int(min_mb), python_exe)
    except Exception as e:
        # Cannot measure VRAM on this host — do not block the run.
        return True, f"vram check skipped ({e})"

    if result.get("status") == "PASS":
        return True, ""
    free_mb = result.get("free_mb", "?")
    total_mb = result.get("total_mb", "?")
    return (
        False,
        f"needs >= {min_mb} MB free VRAM (detected {free_mb} free / {total_mb} total)",
    )


def check_requirements(
    test: dict,
    *,
    gpu_arch: str,
    os_family: str,
    python_exe: Optional[str] = None,
) -> Tuple[bool, str]:
    """Return (supported, skip_reason). supported=False => honest SKIP."""
    ok, reason = check_os(test, os_family)
    if not ok:
        return False, reason
    ok, reason = check_arch(test, gpu_arch)
    if not ok:
        return False, reason
    if python_exe:
        ok, reason = check_vram(test, gpu_arch, python_exe)
        if not ok:
            return False, reason
    return True, ""

