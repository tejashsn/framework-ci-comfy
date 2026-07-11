#!/usr/bin/env python3
"""Extract identifying info from a ComfyUI API-format workflow graph.

A ComfyUI workflow is a node graph: ``{node_id: {"class_type", "inputs"}}``.
For RASTRA rows we only need what identifies *what was run*: the positive prompt
and the workflow filename. The sampler internals (steps/cfg/seed/resolution) are
deliberately omitted -- this suite validates that a model generates the right
output, not its tuning knobs.

Used by comfyui_benchmark.py to stamp ``test_config["test_parameters"]`` so a
RASTRA row shows the prompt + workflow. Pure stdlib; never raises on a malformed
graph (returns what it can).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_graph(workflow_path: str | Path) -> Dict[str, Any]:
    """Load a workflow JSON and return its node graph.

    Handles both the bare ``{node_id: node}`` shape and the wrapped
    ``{"prompt": {node_id: node}}`` shape. Returns {} on any error.
    """
    try:
        data = json.loads(Path(workflow_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if isinstance(data, dict) and isinstance(data.get("prompt"), dict):
        data = data["prompt"]
    return data if isinstance(data, dict) else {}


def _positive_prompt(graph: Dict[str, Any]) -> Optional[str]:
    """Resolve the positive prompt by following a sampler/guider 'positive' link
    back to the source CLIPTextEncode.text. Falls back to the first
    CLIPTextEncode whose text doesn't look like a negative prompt."""

    def _text_of(node_id: str, depth: int = 0) -> Optional[str]:
        if depth > 6:
            return None
        node = graph.get(str(node_id))
        if not isinstance(node, dict):
            return None
        inputs = node.get("inputs", {})
        text = inputs.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        # Follow the first link input (guider/conditioning chains).
        for v in inputs.values():
            if isinstance(v, list) and len(v) == 2:
                found = _text_of(v[0], depth + 1)
                if found:
                    return found
        return None

    # Preferred: whatever the sampler/guider calls 'positive'.
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        pos = node.get("inputs", {}).get("positive")
        if isinstance(pos, list) and len(pos) == 2:
            text = _text_of(pos[0])
            if text:
                return text
    # Fallback: first CLIPTextEncode that doesn't look like a negative.
    neg_markers = ("low quality", "worst quality", "watermark", "deformed",
                   "bad anatomy", "blurry")
    for node in graph.values():
        if isinstance(node, dict) and "CLIPTextEncode" in str(node.get("class_type", "")):
            t = node.get("inputs", {}).get("text")
            if isinstance(t, str) and t.strip() \
                    and not any(m in t.lower() for m in neg_markers):
                return t.strip()
    return None


def extract_params(workflow_path: str | Path) -> Dict[str, Any]:
    """Return {prompt, workflow} for the workflow (best-effort).

    Only includes ``prompt`` if one was found; always includes ``workflow`` (the
    file basename). Never raises.
    """
    params: Dict[str, Any] = {"workflow": Path(workflow_path).name}
    graph = load_graph(workflow_path)
    if graph:
        prompt = _positive_prompt(graph)
        if prompt:
            params["prompt"] = prompt
    return params


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    import sys
    for wf in sys.argv[1:]:
        print(wf, json.dumps(extract_params(wf), ensure_ascii=False))
