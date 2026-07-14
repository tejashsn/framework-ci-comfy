#!/usr/bin/env python3
"""Unit tests for per-arch VRAM gate resolution in test_requirements."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests" / "src" / "inference" / "comfyui" / "executors"))

import test_requirements  # noqa: E402

_ARCH_OVERRIDES = {
    "gfx1201": 15360,
    "gfx1100": 15360,
    "gfx1101": 15360,
    "gfx1102": 15360,
    "gfx1151": 15360,
    "gfx1152": 15360,
    "gfx1153": 15360,
}


def test_effective_min_vram_mb_base_only():
    test = {"min_vram_mb": 24576}
    assert test_requirements.effective_min_vram_mb(test, "gfx942") == 24576


def test_effective_min_vram_mb_arch_override_wins():
    test = {"min_vram_mb": 24576, "min_vram_mb_by_arch": _ARCH_OVERRIDES}
    assert test_requirements.effective_min_vram_mb(test, "gfx1201") == 15360
    assert test_requirements.effective_min_vram_mb(test, "GFX1201") == 15360


def test_effective_min_vram_mb_unknown_arch_falls_back_to_base():
    test = {"min_vram_mb": 16384, "min_vram_mb_by_arch": _ARCH_OVERRIDES}
    assert test_requirements.effective_min_vram_mb(test, "gfx942") == 16384


def test_effective_min_vram_mb_no_gate():
    assert test_requirements.effective_min_vram_mb({}, "gfx1201") is None


def test_check_vram_ignore_env_bypasses_gate(monkeypatch):
    monkeypatch.setenv("COMFYUI_IGNORE_VRAM", "1")
    test = {"min_vram_mb": 999999}
    ok, reason = test_requirements.check_vram(test, "gfx1201", sys.executable)
    assert ok and reason == ""
