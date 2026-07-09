#!/usr/bin/env python3
"""
Auto-download missing ComfyUI model weights before validation runs.

When AUTO_FETCH_MODELS is enabled (default), missing files referenced by a
workflow are fetched from sources declared in config/models.json, with progress
logged to stdout (and GitHub Actions notices in CI). A file lock prevents
parallel matrix jobs from downloading the same weight twice.

Download order (gated-model workaround):
  1. AMD Artifactory mirror when ``artifactory.path`` is set and
     ARTIFACTORY_USER/PASSWORD are available (no HF_TOKEN).
  2. Fallback to ``source`` (HuggingFace or direct URL) when Artifactory misses.

Source schema (models.json):
  "source": {
    "type": "huggingface",
    "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
    "filename": "sd_xl_base_1.0.safetensors",
    "gated": false
  }
  "artifactory": {
    "path": "checkpoints/sd_xl_base_1.0.safetensors"
  }

Environment:
  ARTIFACTORY_URL              default https://compute-artifactory.amd.com
  COMFYUI_MODELS_ARTIFACTORY_REPO  default artifactory/rocm-qa-model-cache/comfyui
  ARTIFACTORY_USER / ARTIFACTORY_PASSWORD
  COMFYUI_PREFER_ARTIFACTORY   default true
  COMFYUI_ARTIFACTORY_ONLY     default false (skip HF fallback when true)
  HF_TOKEN                     HuggingFace fallback for gated models

CLI:
  python executors/fetch_models.py --workflow workflows/sd2_1.json \\
      --comfyui_path ~/ComfyUI
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import model_check

SUITE_DIR = Path(__file__).resolve().parent.parent
MODELS_JSON = SUITE_DIR / "config" / "models.json"

DEFAULT_ARTIFACTORY_URL = "https://compute-artifactory.amd.com"
DEFAULT_ARTIFACTORY_REPO = "artifactory/rocm-qa-model-cache/comfyui"


def auto_fetch_enabled() -> bool:
    """True unless AUTO_FETCH_MODELS is explicitly disabled."""
    return os.environ.get("AUTO_FETCH_MODELS", "true").strip().lower() not in (
        "0", "false", "no", "off",
    )


def prefer_artifactory() -> bool:
    return os.environ.get("COMFYUI_PREFER_ARTIFACTORY", "true").strip().lower() not in (
        "0", "false", "no", "off",
    )


def artifactory_only() -> bool:
    return os.environ.get("COMFYUI_ARTIFACTORY_ONLY", "false").strip().lower() in (
        "1", "true", "yes", "on",
    )


def artifactory_credentials() -> Tuple[str, str]:
    user = (
        os.environ.get("ARTIFACTORY_USER", "")
        or os.environ.get("ARTIFACTORY_USERNAME", "")
    )
    password = os.environ.get("ARTIFACTORY_PASSWORD", "")
    return user, password


def artifactory_model_url(
    entry: dict,
    filename: str,
    subdirs: List[str],
) -> Optional[str]:
    """Build full Artifactory GET URL for a models.json entry, if configured."""
    art = entry.get("artifactory")
    if not art:
        return None
    if isinstance(art, str):
        rel_path = art.strip()
    elif isinstance(art, dict):
        rel_path = (art.get("path") or "").strip()
    else:
        return None
    if not rel_path:
        primary = subdirs[0] if subdirs else "checkpoints"
        rel_path = f"{primary}/{filename}"
    base = os.environ.get("ARTIFACTORY_URL", DEFAULT_ARTIFACTORY_URL).rstrip("/")
    repo = os.environ.get(
        "COMFYUI_MODELS_ARTIFACTORY_REPO", DEFAULT_ARTIFACTORY_REPO
    ).strip("/")
    return f"{base}/{repo}/{rel_path.lstrip('/')}"


def load_manifest(path: Path | None = None) -> dict:
    p = path or MODELS_JSON
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except OSError:
        return {}


@dataclass
class FetchResult:
    downloaded: List[str] = field(default_factory=list)
    already_present: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed


def _log(msg: str, *, notice: bool = False) -> None:
    print(msg, flush=True)
    if notice and os.environ.get("GITHUB_ACTIONS"):
        safe = msg.replace("%", "").replace("\n", " ")[:200]
        print(f"::notice::{safe}", flush=True)


def _log_progress(filename: str, pct: float) -> None:
    pct_i = int(pct)
    _log(f"[fetch] {filename}: {pct_i}%")
    if os.environ.get("GITHUB_ACTIONS") and pct_i % 10 == 0:
        _log(f"[fetch] {filename}: {pct_i}%", notice=True)


@contextlib.contextmanager
def _download_lock(lock_path: Path):
    """Exclusive lock so parallel CI matrix jobs don't race the same file."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl  # type: ignore[import-not-found]
    except ImportError:
        yield
        return
    with open(lock_path, "w", encoding="utf-8") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


def _resolve_dest(models_root: Path, filename: str, subdirs: List[str]) -> Path:
    for sub in subdirs:
        cand = models_root / sub / filename
        if cand.is_file():
            return cand
    primary = subdirs[0] if subdirs else "checkpoints"
    return models_root / primary / filename


def _verify_after_download(filename: str, path: Path, manifest: dict) -> Tuple[bool, str]:
    ok, reason = model_check.verify_identity(filename, path, manifest)
    return ok, reason


def _download_huggingface(
    repo_id: str,
    remote_name: str,
    dest: Path,
    *,
    revision: str = "main",
    token: Optional[str],
) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is required for HF downloads "
            "(pip install huggingface_hub)"
        ) from e

    dest.parent.mkdir(parents=True, exist_ok=True)
    _log(f"[fetch] HF {repo_id}/{remote_name} -> {dest}")
    cached = hf_hub_download(
        repo_id=repo_id,
        filename=remote_name,
        revision=revision or "main",
        token=token or None,
        resume_download=True,
    )
    tmp = dest.with_suffix(dest.suffix + ".part")
    shutil.copy2(cached, tmp)
    tmp.replace(dest)


def _download_http(
    url: str,
    dest: Path,
    *,
    user: str = "",
    password: str = "",
    bearer: str = "",
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "framework-ci-comfy/1.0"}
    if user and password:
        token = base64.b64encode(f"{user}:{password}".encode()).decode("ascii")
        headers["Authorization"] = f"Basic {token}"
    elif bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    req = urllib.request.Request(url, headers=headers)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            if resp.status and int(resp.status) >= 400:
                raise urllib.error.HTTPError(
                    url, int(resp.status), resp.reason, resp.headers, None
                )
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            chunk = 1024 * 1024
            last_pct = -1
            with open(tmp, "wb") as out:
                while True:
                    block = resp.read(chunk)
                    if not block:
                        break
                    out.write(block)
                    done += len(block)
                    if total > 0:
                        pct = 100.0 * done / total
                        pct_i = int(pct)
                        if pct_i >= last_pct + 5 or pct_i == 100:
                            _log_progress(dest.name, pct)
                            last_pct = pct_i
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _try_artifactory(entry: dict, filename: str, subdirs: List[str], dest: Path) -> None:
    url = artifactory_model_url(entry, filename, subdirs)
    if not url:
        raise RuntimeError("no artifactory.path configured")
    user, password = artifactory_credentials()
    if not user or not password:
        raise RuntimeError(
            "ARTIFACTORY_USER and ARTIFACTORY_PASSWORD required for Artifactory fetch"
        )
    _log(f"[fetch] Artifactory {url} -> {dest}")
    _download_http(url, dest, user=user, password=password)


def _try_source(source: dict, filename: str, dest: Path) -> None:
    stype = (source.get("type") or "huggingface").lower()
    if stype == "huggingface":
        repo_id = source.get("repo_id") or source.get("repo")
        remote = source.get("filename") or filename
        if not repo_id:
            raise RuntimeError("source missing repo_id")
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if source.get("gated") and not token:
            raise RuntimeError("HF_TOKEN required for gated HuggingFace model")
        _download_huggingface(
            repo_id,
            remote,
            dest,
            revision=source.get("revision") or "main",
            token=token,
        )
    elif stype in ("url", "artifactory"):
        url = source.get("url")
        if not url:
            raise RuntimeError("source missing url")
        _log(f"[fetch] URL {url} -> {dest}")
        user, password = artifactory_credentials()
        bearer = os.environ.get("HF_TOKEN", "")
        if user and password:
            _download_http(url, dest, user=user, password=password)
        else:
            _download_http(url, dest, bearer=bearer)
    else:
        raise RuntimeError(f"unsupported source type '{stype}'")


def download_one(
    filename: str,
    subdirs: List[str],
    comfy_path: str,
    manifest: dict,
    *,
    force: bool = False,
) -> Tuple[bool, str]:
    """Download a single model file. Returns (success, detail)."""
    models_root = Path(comfy_path) / "models"
    dest = _resolve_dest(models_root, filename, subdirs)

    if dest.is_file() and not force:
        return True, "already present"

    entry = manifest.get(filename) or {}
    source = entry.get("source")
    has_art = bool(artifactory_model_url(entry, filename, subdirs))

    if not has_art and (not source or not isinstance(source, dict)):
        return False, f"no download source configured for {filename}"

    lock_path = dest.with_suffix(dest.suffix + ".lock")
    with _download_lock(lock_path):
        if dest.is_file() and not force:
            return True, "already present (after lock)"

        errors: List[str] = []
        tried: List[str] = []

        if has_art and prefer_artifactory():
            tried.append("artifactory")
            try:
                _try_artifactory(entry, filename, subdirs, dest)
            except Exception as e:
                errors.append(f"artifactory: {e}")
            else:
                ok, reason = _verify_after_download(filename, dest, manifest)
                if ok:
                    return True, f"downloaded from Artifactory to {dest}"
                dest.unlink(missing_ok=True)
                errors.append(f"artifactory verify: {reason}")

        if not artifactory_only() and source and isinstance(source, dict):
            tried.append("source")
            try:
                _try_source(source, filename, dest)
            except Exception as e:
                errors.append(f"source: {e}")
            else:
                ok, reason = _verify_after_download(filename, dest, manifest)
                if ok:
                    return True, f"downloaded to {dest}"
                dest.unlink(missing_ok=True)
                errors.append(f"source verify: {reason}")

        if not tried:
            return False, "no fetch method available (configure artifactory.path or source)"
        return False, f"download failed for {filename} ({'; '.join(errors)})"


def ensure_missing(
    missing: List[Tuple[str, List[str]]],
    comfy_path: str,
    manifest: dict | None = None,
) -> FetchResult:
    """Download every missing (filename, subdirs) tuple."""
    result = FetchResult()
    manifest = manifest if manifest is not None else load_manifest()
    if not comfy_path:
        result.failed = [m[0] for m in missing]
        result.errors.append("COMFYUI_PATH not set — cannot download models")
        return result

    for filename, subdirs in missing:
        ok, detail = download_one(filename, subdirs, comfy_path, manifest)
        if ok and detail.startswith("already"):
            result.already_present.append(filename)
        elif ok:
            result.downloaded.append(filename)
            _log(f"[fetch] OK {filename}: {detail}", notice=True)
        else:
            result.failed.append(filename)
            result.errors.append(detail)
            _log(f"[fetch] FAIL {filename}: {detail}")
    return result


def ensure_for_workflow(
    workflow_path: str | Path,
    comfy_path: str,
    manifest: dict | None = None,
) -> FetchResult:
    missing = model_check.missing_models(str(workflow_path), comfy_path)
    if not missing:
        return FetchResult()
    _log(f"[fetch] {len(missing)} model(s) missing for {workflow_path}")
    return ensure_missing(missing, comfy_path, manifest)


def main() -> int:
    p = argparse.ArgumentParser(description="Fetch missing ComfyUI model weights")
    p.add_argument("--workflow", help="Workflow JSON to resolve required models")
    p.add_argument("--comfyui_path", required=True, help="ComfyUI install (has models/)")
    p.add_argument("--models-json", default=str(MODELS_JSON))
    p.add_argument("--force", action="store_true", help="Re-download even if present")
    args = p.parse_args()

    manifest = load_manifest(Path(args.models_json))
    comfy = os.path.expanduser(args.comfyui_path)

    if args.workflow:
        missing = model_check.missing_models(args.workflow, comfy)
        if not missing:
            print("[fetch] all models already present")
            return 0
        result = ensure_missing(missing, comfy, manifest)
    else:
        print("Error: --workflow is required", file=sys.stderr)
        return 1

    if result.downloaded:
        print(f"Downloaded: {', '.join(result.downloaded)}")
    if result.failed:
        for err in result.errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
