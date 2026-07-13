#!/usr/bin/env python3
"""
Pre-run gate for ComfyUI workflow *input* assets (LoadImage / LoadVideo nodes).

Model weights live under ComfyUI/models/ and are handled by model_check.py.
Workflow input images/videos live under ComfyUI/input/ and are referenced by
filename in loader nodes. Missing inputs cause immediate HTTP 400 validation
failures that look like model regressions unless caught early.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# class_type -> input field that names a file under ComfyUI/input/
INPUT_NODE_MAP = {
    "LoadImage": "image",
    "LoadImageMask": "image",
    "LoadImageOutput": "image",
    "LoadVideo": "video",
    "VHS_LoadVideo": "video",
    "VHS_LoadVideoPath": "video",
}


def _load_graph(workflow_path):
    data = json.loads(Path(workflow_path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "prompt" in data and isinstance(data["prompt"], dict):
        return data["prompt"]
    return data if isinstance(data, dict) else {}


def referenced_inputs(workflow_path):
    """Return sorted unique filenames referenced by input-loader nodes."""
    graph = _load_graph(workflow_path)
    names = set()
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        field = INPUT_NODE_MAP.get(node.get("class_type"))
        if not field:
            continue
        val = node.get("inputs", {}).get(field)
        if isinstance(val, str) and val.strip():
            names.add(val.strip())
    return sorted(names)


def input_dir(comfy_path: str) -> Path:
    return Path(comfy_path) / "input"


def missing_inputs(workflow_path, comfy_path):
    """Filenames referenced by the workflow but absent under ComfyUI/input/."""
    if not comfy_path:
        return []
    root = input_dir(comfy_path)
    if not root.exists():
        return referenced_inputs(workflow_path)
    missing = []
    for name in referenced_inputs(workflow_path):
        if not (root / name).is_file():
            missing.append(name)
    return missing


def skip_message(missing):
    joined = ", ".join(f"input/{n}" for n in missing)
    return (f"SKIP: workflow input asset not found: {joined}. "
            "Place file(s) under ComfyUI/input/ or enable AUTO_FETCH_MODELS "
            "so workflow_inputs.json sources can be fetched.")
