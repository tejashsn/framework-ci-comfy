#!/usr/bin/env python3
"""
Auto-fetch missing ComfyUI workflow input assets (images/videos under input/).

Mirrors fetch_models.py: sources are declared in config/workflow_inputs.json.
When AUTO_FETCH_MODELS is enabled (same gate as model weights), missing input
files are downloaded or generated before validation runs.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import fetch_models
import workflow_input_check

SUITE_DIR = Path(__file__).resolve().parent.parent
INPUTS_JSON = SUITE_DIR / "config" / "workflow_inputs.json"
BUNDLED_INPUT_DIR = SUITE_DIR / "assets" / "input"


@dataclass
class FetchResult:
    downloaded: List[str] = field(default_factory=list)
    already_present: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed


def load_manifest(path: Path | None = None) -> dict:
    p = path or INPUTS_JSON
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _log(msg: str) -> None:
    print(f"[fetch-input] {msg}", flush=True)


def bundled_asset_path(filename: str) -> Path:
    """Shipped copy under tests/src/inference/comfyui/assets/input/."""
    return BUNDLED_INPUT_DIR / filename


def _copy_bundled(filename: str, dest: Path) -> Tuple[bool, str]:
    src = bundled_asset_path(filename)
    if not src.is_file():
        return False, f"bundled asset missing: {src}"
    shutil.copy2(src, dest)
    return True, f"copied bundled asset from {src}"


def _write_generated_image(dest: Path, spec: dict) -> None:
    """Write a solid-color placeholder image (any valid file passes LoadImage)."""
    width = int(spec.get("width") or 512)
    height = int(spec.get("height") or 512)
    fmt = (spec.get("format") or dest.suffix.lstrip(".") or "png").lower()
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        img = Image.new("RGB", (width, height), color=(96, 96, 96))
        if fmt in ("jpg", "jpeg"):
            img.save(dest, "JPEG", quality=85)
        else:
            img.save(dest, "PNG")
        return
    except ImportError:
        pass
    # Minimal 1x1 PNG fallback (only for .png destinations).
    if dest.suffix.lower() not in (".png",):
        raise RuntimeError(
            f"Pillow not available; cannot generate {dest.name} "
            "(install Pillow or configure a url/huggingface source)"
        )
    minimal_png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf"
        b"\xc0\x00\x00\x00\x03\x00\x01\x00\x05\x00\x01\x0d\n-\xb4\x00\x00"
        b"\x00\x00IEND\xaeB`\x82"
    )
    dest.write_bytes(minimal_png)


def download_one(
    filename: str,
    comfy_path: str,
    manifest: dict | None = None,
    *,
    force: bool = False,
) -> Tuple[bool, str]:
    """Fetch a single input asset into ComfyUI/input/. Returns (success, detail)."""
    manifest = manifest if manifest is not None else load_manifest()
    dest = workflow_input_check.input_dir(comfy_path) / filename

    if dest.is_file() and not force:
        return True, "already present"

    entry = manifest.get(filename) or {}
    source = entry.get("source")
    if not source or not isinstance(source, dict):
        if bundled_asset_path(filename).is_file():
            source = {"type": "bundled"}
        else:
            # Generic fallback: generated placeholder so CI can proceed on any platform.
            source = {"type": "generated", "width": 512, "height": 512}

    stype = (source.get("type") or "generated").lower()
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        if stype == "bundled":
            ok, detail = _copy_bundled(filename, dest)
            if ok:
                return True, detail
            # Fall back to upstream URL declared beside the bundled entry.
            upstream = entry.get("upstream_url")
            if upstream:
                fetch_models._try_source({"type": "url", "url": upstream}, filename, dest)
                return True, f"downloaded upstream fallback to {dest}"
            return False, detail
        if stype == "generated":
            _write_generated_image(dest, source)
            return True, f"generated placeholder at {dest}"
        if stype in ("url", "huggingface", "artifactory"):
            tmp = dest.with_suffix(dest.suffix + ".part")
            fetch_models._try_source(source, filename, tmp)
            if stype == "url" and filename.lower().endswith((".jpg", ".jpeg")):
                # URL may serve PNG bytes; normalize to JPEG when required.
                try:
                    from PIL import Image
                    with Image.open(tmp) as img:
                        img.convert("RGB").save(dest, "JPEG", quality=85)
                    tmp.unlink(missing_ok=True)
                    return True, f"downloaded and converted to {dest}"
                except ImportError:
                    pass
            tmp.replace(dest)
            return True, f"downloaded to {dest}"
        return False, f"unsupported source type '{stype}' for {filename}"
    except Exception as e:
        dest.unlink(missing_ok=True)
        return False, str(e)


def ensure_missing(
    missing: List[str],
    comfy_path: str,
    manifest: dict | None = None,
) -> FetchResult:
    result = FetchResult()
    if not missing:
        return result
    if not comfy_path:
        result.failed = list(missing)
        result.errors.append("COMFYUI_PATH not set — cannot fetch workflow inputs")
        return result

    manifest = manifest if manifest is not None else load_manifest()
    for filename in missing:
        ok, detail = download_one(filename, comfy_path, manifest)
        if ok:
            if "already" in detail:
                result.already_present.append(filename)
            else:
                result.downloaded.append(filename)
                _log(f"OK {filename}: {detail}")
        else:
            result.failed.append(filename)
            result.errors.append(f"{filename}: {detail}")
            _log(f"FAIL {filename}: {detail}")
    return result


def auto_fetch_enabled() -> bool:
    return fetch_models.auto_fetch_enabled()
