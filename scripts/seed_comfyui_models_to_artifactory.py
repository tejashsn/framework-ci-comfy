#!/usr/bin/env python3
"""One-time seed: upload ComfyUI model weights from disk to the Artifactory cache.

Populates the mirror used by fetch_models.py so gated models download without
HF_TOKEN. Run once per weight (or --all-gated) from a machine that already has
the files under COMFYUI_PATH/models/.

Example:
  export ARTIFACTORY_USER=...
  export ARTIFACTORY_PASSWORD=...
  export COMFYUI_PATH=~/ComfyUI
  python scripts/seed_comfyui_models_to_artifactory.py --all-gated
  python scripts/seed_comfyui_models_to_artifactory.py --file ae.safetensors
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS_JSON = REPO / "tests/src/inference/comfyui/config/models.json"
sys.path.insert(0, str(REPO / "tests/src/inference/comfyui/executors"))

import fetch_models  # noqa: E402


def _put(local: Path, dest_url: str, user: str, password: str, dry_run: bool) -> bool:
    if dry_run:
        print(f"[dry-run] PUT {local} -> {dest_url}")
        return True
    r = subprocess.run(
        ["curl", "-sf", "-u", f"{user}:{password}", "-X", "PUT", "-T", str(local), dest_url],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(f"FAIL {local.name}: {(r.stderr or r.stdout).strip()}", file=sys.stderr)
        return False
    print(f"OK {local.name} -> {dest_url}")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description="Seed ComfyUI model cache on Artifactory")
    p.add_argument("--comfyui-path", default=os.environ.get("COMFYUI_PATH", ""))
    p.add_argument("--file", action="append", default=[], help="Model filename in models.json")
    p.add_argument("--all-gated", action="store_true", help="Upload every entry with artifactory+source.gated")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    comfy = Path(os.path.expanduser(args.comfyui_path or ""))
    if not comfy.is_dir():
        print("Error: set --comfyui-path or COMFYUI_PATH", file=sys.stderr)
        return 1

    user, password = fetch_models.artifactory_credentials()
    if not args.dry_run and (not user or not password):
        print("Error: ARTIFACTORY_USER and ARTIFACTORY_PASSWORD required", file=sys.stderr)
        return 1

    manifest = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    targets = list(args.file)
    if args.all_gated:
        for name, entry in manifest.items():
            src = entry.get("source") or {}
            if entry.get("artifactory") and src.get("gated"):
                targets.append(name)
    targets = sorted(set(targets))
    if not targets:
        print("No files selected", file=sys.stderr)
        return 1

    ok = fail = 0
    for name in targets:
        entry = manifest.get(name)
        if not entry:
            print(f"skip unknown: {name}", file=sys.stderr)
            fail += 1
            continue
        subdirs = entry.get("subdirs") or ["checkpoints"]
        local = fetch_models._resolve_dest(comfy / "models", name, subdirs)
        if not local.is_file():
            print(f"skip missing on disk: {local}", file=sys.stderr)
            fail += 1
            continue
        url = fetch_models.artifactory_model_url(entry, name, subdirs)
        if not url:
            print(f"skip no artifactory.path: {name}", file=sys.stderr)
            fail += 1
            continue
        if _put(local, url, user, password, args.dry_run):
            ok += 1
        else:
            fail += 1

    print(f"done: {ok} uploaded, {fail} failed/skipped")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
