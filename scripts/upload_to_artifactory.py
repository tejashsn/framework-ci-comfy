#!/usr/bin/env python3
"""Upload a local directory to Artifactory at the fleet-agreed path.

Fleet contract:
  1. ``build_artifactory_info()`` (tests/utils/results/handler.py) writes
     ``logs/artifactory_path.txt`` with the destination path.
  2. This script reads that file and PUTs every file under ``--source-path``.

Generated media (the point of a ComfyUI run) rides along in the same upload so a
result's Artifactory folder shows the real image/video. Most images (~1 MB) and
short videos (~0.1 MB) upload raw so they're viewable in the browser; a large
video (long/high-res render, hundreds of MB) is zipped first to save bandwidth.
The threshold is configurable via ARTIFACTORY_ZIP_THRESHOLD_MB or
--zip-threshold-mb (0 disables zipping).

Manual example (after a benchmark):

    export ARTIFACTORY_USER=...
    export ARTIFACTORY_PASSWORD=...
  python scripts/upload_to_artifactory.py \\
      --source-path logs/ \\
      --framework comfyui \\
      --test-name comfyui_stable_diffusion_2_1

Use ``--dry-run`` to print destinations without uploading.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "https://compute-artifactory.amd.com"
ARTIFACTORY_PATH_FILE = "artifactory_path.txt"

# Generated media that gets zipped before upload when it exceeds the threshold.
MEDIA_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif",
    ".mp4", ".webm", ".mkv", ".mov", ".avi",
}
DEFAULT_ZIP_THRESHOLD_MB = 50


def _zip_threshold_bytes() -> int:
    """Size (bytes) at/above which a media file is zipped before upload.

    Read from ARTIFACTORY_ZIP_THRESHOLD_MB (in MB). A value of 0 disables
    zipping (every file uploads raw). Falls back to DEFAULT_ZIP_THRESHOLD_MB on
    any parse error so a bad env var never blocks the upload.
    """
    raw = os.environ.get("ARTIFACTORY_ZIP_THRESHOLD_MB", "").strip()
    if raw == "":
        mb: float = DEFAULT_ZIP_THRESHOLD_MB
    else:
        try:
            mb = float(raw)
        except ValueError:
            mb = DEFAULT_ZIP_THRESHOLD_MB
    if mb <= 0:
        return 0  # zipping disabled
    return int(mb * 1024 * 1024)


def _should_zip(path: Path, threshold_bytes: int) -> bool:
    """True if ``path`` is a media file large enough to warrant zipping."""
    if threshold_bytes <= 0:
        return False
    if path.suffix.lower() not in MEDIA_EXTENSIONS:
        return False
    try:
        return path.stat().st_size >= threshold_bytes
    except OSError:
        return False


def _zip_file(src: Path, tmp_dir: Path) -> Path:
    """Zip a single file into ``tmp_dir/<name>.zip`` (stored under its basename).

    Returns the path to the created archive. The file is stored under its
    original name (not its full path) so it extracts cleanly.
    """
    archive = tmp_dir / f"{src.name}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(src, arcname=src.name)
    return archive


def _resolve_credentials(user: str, password: str) -> tuple[str, str]:
    user = (
        user
        or os.environ.get("ARTIFACTORY_USER", "")
        or os.environ.get("ARTIFACTORY_USERNAME", "")
    )
    password = password or os.environ.get("ARTIFACTORY_PASSWORD", "")
    return user, password


def _read_artifactory_path(logs_dir: Path) -> str:
    path_file = logs_dir / ARTIFACTORY_PATH_FILE
    if path_file.exists():
        return path_file.read_text(encoding="utf-8").strip()
    return ""


def _ensure_artifactory_path(
    logs_dir: Path,
    *,
    framework: str,
    test_name: str,
) -> str:
    """Return repo-relative Artifactory path, building it if missing."""
    logs_dir = logs_dir.resolve()
    existing = _read_artifactory_path(logs_dir)
    if existing:
        return existing

    if logs_dir.name != "logs":
        raise ValueError(
            f"logs_dir must be named 'logs' (got {logs_dir.name!r}); "
            "build_artifactory_info writes to ./logs/artifactory_path.txt"
        )

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from tests.utils.results.handler import build_artifactory_info

    prev_cwd = os.getcwd()
    try:
        os.chdir(logs_dir.parent)
        build_artifactory_info(test_name=test_name, framework=framework)
    finally:
        os.chdir(prev_cwd)

    return _read_artifactory_path(logs_dir)


def resolve_upload_base_url(
    artifactory_path: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
) -> str:
    """Turn ``/artifactory/...`` path file content into a full upload base URL."""
    path = (artifactory_path or "").strip()
    if not path:
        raise ValueError(
            f"No {ARTIFACTORY_PATH_FILE} found. Run upload_from_artifacts.py first "
            "or pass --framework and --test-name to generate the path."
        )
    base = base_url.rstrip("/")
    if path.startswith("http://") or path.startswith("https://"):
        return path.rstrip("/")
    if not path.startswith("/"):
        path = f"/{path}"
    return f"{base}{path}".rstrip("/")


def iter_files(source: Path) -> list[Path]:
    if not source.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")
    if source.is_file():
        return [source]
    files = [
        p for p in sorted(source.rglob("*"))
        if p.is_file() and ".git" not in p.parts
    ]
    if not files:
        raise ValueError(f"No files found under {source}")
    return files


def put_file(
    dest_url: str,
    local_path: Path,
    user: str,
    password: str,
    *,
    dry_run: bool = False,
) -> bool:
    if dry_run:
        print(f"[dry-run] PUT {local_path} -> {dest_url}")
        return True
    result = subprocess.run(
        [
            "curl", "-sf",
            "-u", f"{user}:{password}",
            "-X", "PUT",
            "-T", str(local_path),
            dest_url,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "curl failed").strip()
        print(f"Failed: {local_path.name} -> {dest_url}\n  {err}", file=sys.stderr)
        return False
    return True


def upload_directory(
    source_path: str | Path,
    *,
    logs_dir: str | Path = "logs",
    framework: str = "comfyui",
    test_name: str = "",
    base_url: str = DEFAULT_BASE_URL,
    user: str = "",
    password: str = "",
    dry_run: bool = False,
) -> tuple[str, int, int]:
    """Upload ``source_path`` under the Artifactory path. Returns (base_url, ok, fail).

    Media files at/above the zip threshold are compressed to ``<name>.zip``
    before upload (see ``_zip_threshold_bytes``); everything else uploads raw.
    """
    source = Path(source_path).resolve()
    logs = Path(logs_dir).resolve()
    art_path = _ensure_artifactory_path(
        logs, framework=framework, test_name=test_name
    )
    upload_base = resolve_upload_base_url(
        art_path, base_url=os.environ.get("ARTIFACTORY_BASE_URL", base_url)
    )

    user, password = _resolve_credentials(user, password)
    if not dry_run and (not user or not password):
        raise ValueError(
            "Artifactory credentials required. Set ARTIFACTORY_USER and "
            "ARTIFACTORY_PASSWORD or pass --user/--password."
        )

    threshold_bytes = _zip_threshold_bytes()
    ok = fail = 0
    root = source if source.is_dir() else source.parent
    # Temp dir for any on-the-fly zips of oversized media; auto-cleaned on exit.
    with tempfile.TemporaryDirectory(prefix="artifactory_zip_") as tmp:
        tmp_dir = Path(tmp)
        for file_path in iter_files(source):
            rel = file_path.relative_to(root).as_posix()
            upload_path = file_path

            if _should_zip(file_path, threshold_bytes):
                size_mb = file_path.stat().st_size / (1024 * 1024)
                rel = f"{rel}.zip"  # destination gets the .zip suffix
                if dry_run:
                    print(
                        f"[dry-run] ZIP (large media {size_mb:.1f} MB) "
                        f"{file_path.name} -> {file_path.name}.zip"
                    )
                else:
                    upload_path = _zip_file(file_path, tmp_dir)
                    zipped_mb = upload_path.stat().st_size / (1024 * 1024)
                    print(
                        f"  zipped {file_path.name} "
                        f"({size_mb:.1f} MB -> {zipped_mb:.1f} MB)"
                    )

            dest = f"{upload_base}/{rel}"
            if put_file(dest, upload_path, user, password, dry_run=dry_run):
                ok += 1
                if not dry_run:
                    print(f"  ✓ {rel}")
            else:
                fail += 1

    return upload_base, ok, fail


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Upload logs to Artifactory")
    parser.add_argument("--source-path", default="logs", help="Directory or file to upload")
    parser.add_argument("--logs-dir", default="logs", help="Dir containing artifactory_path.txt")
    parser.add_argument("--framework", default="comfyui")
    parser.add_argument("--test-name", default="", help="Used if artifactory_path.txt is missing")
    parser.add_argument("--user", default="")
    parser.add_argument("--password", default="")
    parser.add_argument(
        "--zip-threshold-mb", default=None, type=float,
        help="Zip media files at/above this size (MB) before upload. "
             "0 disables zipping. Overrides ARTIFACTORY_ZIP_THRESHOLD_MB. "
             "Default: 50.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    # CLI flag overrides the env var for this invocation.
    if args.zip_threshold_mb is not None:
        os.environ["ARTIFACTORY_ZIP_THRESHOLD_MB"] = str(args.zip_threshold_mb)

    try:
        upload_base, ok, fail = upload_directory(
            args.source_path,
            logs_dir=args.logs_dir,
            framework=args.framework,
            test_name=args.test_name,
            user=args.user,
            password=args.password,
            dry_run=args.dry_run,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    label = "Would upload" if args.dry_run else "Uploaded"
    print(f"{label} {ok} file(s) to {upload_base}")
    if fail:
        print(f"{fail} file(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
