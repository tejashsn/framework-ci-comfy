#!/usr/bin/env python3
"""Unit tests for the workflow parameter extractor (no GPU/network).

extract_params returns only {prompt, workflow} -- the identifying info for a
RASTRA row. These verify the prompt is resolved correctly (positive, not
negative) across model families and that malformed/missing inputs are safe.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SUITE = REPO / "tests" / "src" / "inference" / "comfyui"
WORKFLOWS = SUITE / "workflows"
sys.path.insert(0, str(SUITE / "executors"))

import workflow_params as wp  # noqa: E402


def _params(name):
    return wp.extract_params(WORKFLOWS / name)


def test_returns_prompt_and_workflow_only():
    p = _params("sd2_1.json")
    assert set(p.keys()) <= {"prompt", "workflow"}
    assert p["workflow"] == "sd2_1.json"
    assert "red fox" in p["prompt"]


def test_prompt_is_positive_not_negative():
    # Must follow the sampler's positive link, never grab the negative prompt.
    p = _params("ltx_t2v.json")
    assert "low quality" not in p["prompt"].lower()
    assert "woman" in p["prompt"].lower()


def test_decomposed_sampler_prompt_resolves():
    # Flux2/Chroma route positive through guider nodes; link-following must reach
    # the CLIPTextEncode.
    assert "red fox" in _params("flux2_dev.json")["prompt"].lower()
    assert "tiger" in _params("chroma.json")["prompt"].lower()


def test_missing_workflow_returns_workflow_key_only():
    p = wp.extract_params(WORKFLOWS / "does_not_exist.json")
    assert p == {"workflow": "does_not_exist.json"}


def test_all_workflows_extract_safely():
    # Every workflow yields a dict with the workflow name; never raises.
    for wf in WORKFLOWS.glob("*.json"):
        p = wp.extract_params(wf)
        assert isinstance(p, dict)
        assert p.get("workflow") == wf.name
        assert set(p.keys()) <= {"prompt", "workflow"}
