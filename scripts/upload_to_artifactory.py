#!/usr/bin/env python3
"""Upload a local directory to Artifactory at the fleet-agreed path.

Fleet contract:
  1. ``build_artifactory_info()`` (tests/utils/results/handler.py) writes
     ``logs/artifactory_path.txt`` with the destination path.
  2. This script reads that file and PUTs every file under ``--source-path``.

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
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "https://compute-artifactory.amd.com"
ARTIFACTORY_PATH_FILE = "artifactory_path.txt"


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
    """Upload ``source_path`` under the Artifactory path. Returns (base_url, ok, fail)."""
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

    ok = fail = 0
    root = source if source.is_dir() else source.parent
    for file_path in iter_files(source):
        rel = file_path.relative_to(root).as_posix()
        dest = f"{upload_base}/{rel}"
        if put_file(dest, file_path, user, password, dry_run=dry_run):
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

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
