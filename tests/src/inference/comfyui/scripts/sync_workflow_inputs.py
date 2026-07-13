#!/usr/bin/env python3
"""Refresh bundled workflow input images from Comfy-Org/workflow_templates.

Official assets are catalogued in workflow_inputs.json (upstream_url per file).
Run from repo root:

  python tests/src/inference/comfyui/scripts/sync_workflow_inputs.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

SUITE_DIR = Path(__file__).resolve().parent.parent
CONFIG = SUITE_DIR / "config" / "workflow_inputs.json"
ASSETS = SUITE_DIR / "assets" / "input"


def main() -> int:
    manifest = json.loads(CONFIG.read_text(encoding="utf-8"))
    ASSETS.mkdir(parents=True, exist_ok=True)
    ok, fail = 0, 0
    for name, entry in manifest.items():
        if name.startswith("_") or not isinstance(entry, dict):
            continue
        url = entry.get("upstream_url")
        if not url:
            print(f"skip {name}: no upstream_url")
            continue
        dest = ASSETS / name
        print(f"fetch {name} ...")
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                dest.write_bytes(resp.read())
            print(f"  OK {dest} ({dest.stat().st_size} bytes)")
            ok += 1
        except Exception as e:
            print(f"  FAIL {name}: {e}", file=sys.stderr)
            fail += 1
    print(f"done: {ok} ok, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
