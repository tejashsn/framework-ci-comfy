#!/usr/bin/env python3
"""
Capture the ROCm + Python/ML runtime stack on a CI runner.

Writes JSON and an optional Markdown table (GitHub step summary) so each workflow
run records which TheRock tarball, commit, ROCm paths, and torch wheels are
active — versions change every build.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse


def _read_ini(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.is_file():
        return data
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        data[key.strip()] = val.strip().strip('"').strip("'")
    return data


def _first_key(meta: dict[str, str], *keys: str) -> str:
    for key in keys:
        val = meta.get(key, "").strip()
        if val:
            return val
    return ""


def _discover_metadata_files() -> list[Path]:
    found: list[Path] = []
    linux_meta = Path("/etc/rocm_metadata.conf")
    if linux_meta.is_file():
        found.append(linux_meta)

    for root in (Path("/c/TheRock"), Path("C:/TheRock")):
        if not root.is_dir():
            continue
        build_meta = root / "build" / "metadata.conf"
        if build_meta.is_file():
            found.append(build_meta)
        found.extend(sorted(root.glob("rocm-*/metadata.conf")))
    return found


def _pick_active_metadata(files: list[Path], rocm_path: str) -> tuple[dict[str, str], str]:
    if not files:
        return {}, ""

    norm = lambda p: str(p).replace("\\", "/").rstrip("/").lower()  # noqa: E731
    target = norm(rocm_path) if rocm_path else ""

    if target:
        for path in files:
            meta = _read_ini(path)
            meta_path = _first_key(meta, "ROCM_PATH", "HIP_PATH")
            if meta_path and norm(meta_path) == target:
                return meta, str(path)

    # Prefer versioned rocm-* metadata over build/metadata.conf
    versioned = [p for p in files if "rocm-" in p.as_posix().lower()]
    if versioned:
        return _read_ini(versioned[-1]), str(versioned[-1])
    return _read_ini(files[0]), str(files[0])


def _resolve_rocm_path() -> str:
    for key in ("ROCM_PATH", "HIP_PATH"):
        val = os.environ.get(key, "").strip()
        if val and Path(val).is_dir():
            return val

    for candidate in (
        Path("/opt/rocm"),
        Path("/opt/rocm/core"),
    ):
        if candidate.is_dir():
            return str(candidate)

    therock = Path("/c/TheRock") if Path("/c/TheRock").is_dir() else Path("C:/TheRock")
    if therock.is_dir():
        versioned = sorted(therock.glob("rocm-*"), key=lambda p: p.name)
        for p in reversed(versioned):
            if p.is_dir():
                return str(p)
        build = therock / "build"
        if build.is_dir():
            return str(build)
    return ""


def _machine_hip_path(files: list[Path]) -> str:
    for path in files:
        if path.as_posix().endswith("/build/metadata.conf") or "\\build\\metadata.conf" in str(path):
            meta = _read_ini(path)
            return _first_key(meta, "ROCM_PATH", "HIP_PATH")
    return str(Path("/c/TheRock/build")) if Path("/c/TheRock/build").is_dir() else ""


def _venv_display(python: str) -> str:
    py = Path(python)
    if not py.is_file():
        return python
    try:
        import stat

        if hasattr(os, "lstat"):
            mode = os.lstat(py).st_mode
            if stat.S_ISLNK(mode):
                return f"{py.parent.parent} (symlink -> {os.readlink(py)})"
        if sys.platform == "win32":
            # Junction/symlink hint without extra deps
            resolved = py.resolve()
            if resolved != py:
                return f"{py.parent.parent} (junction: {resolved.parent.parent.name})"
    except OSError:
        pass
    return str(py.parent.parent)


def _probe_ml_stack(python: str) -> dict:
    probe = r"""
import importlib.metadata as im
import json

def pkg(name):
    try:
        return im.version(name)
    except Exception:
        return ""

out = {}
try:
    import torch
    out["torch"] = torch.__version__
    out["cuda_avail"] = bool(torch.cuda.is_available())
    out["hip_torch"] = getattr(torch.version, "hip", None) or ""
    if out["cuda_avail"]:
        out["gpu_name"] = torch.cuda.get_device_name(0)
        try:
            out["gpu_arch"] = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        except Exception:
            out["gpu_arch"] = ""
    else:
        out["gpu_name"] = ""
        out["gpu_arch"] = ""
except Exception as exc:
    out["torch_error"] = str(exc)

for mod, key in (("torchvision", "torchvision"), ("torchaudio", "torchaudio")):
    try:
        m = __import__(mod)
        out[key] = getattr(m, "__version__", "")
    except Exception:
        out[key] = ""

out["rocm_sdk"] = pkg("rocm-sdk")
out["python_version"] = ".".join(map(str, __import__("sys").version_info[:3]))
print(json.dumps(out))
"""
    try:
        result = subprocess.run(
            [python, "-c", probe],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0 and result.stdout.strip():
            line = result.stdout.strip().splitlines()[-1]
            return json.loads(line)
    except Exception as exc:
        return {"probe_error": str(exc)}
    return {"probe_error": (result.stderr or "torch probe failed").strip()}


def _device_wheels(python: str) -> str:
    try:
        result = subprocess.run(
            [python, "-m", "pip", "list", "--format=freeze"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return ""
        names = []
        for line in result.stdout.splitlines():
            name = line.split("==", 1)[0].strip().lower()
            if name.startswith("amd-torch-device-") or name.startswith("rocm-sdk-device-"):
                names.append(line.split("==", 1)[0])
        return ", ".join(names)
    except Exception:
        return ""


def _rocm_active_label(rocm_path: str, installed_version: str) -> str:
    if installed_version and rocm_path:
        return f"{installed_version} -> {rocm_path}"
    if rocm_path:
        return rocm_path
    return installed_version or "unknown"


def _match_expected(detected: str, pattern: str) -> bool | None:
    if not pattern or not detected or detected == "unknown":
        return None
    regex = "^" + re.escape(pattern).replace(r"\*", r".*") + "$"
    return bool(re.match(regex, detected))


def _wheel_filename(url: str) -> str:
    if not url:
        return ""
    path = unquote(urlparse(url.strip()).path)
    return path.rsplit("/", 1)[-1]


def _version_from_wheel_url(url: str, package: str) -> str:
    """Parse torch-2.12.0+rocm7.14.0rc2-cp314-...whl from a wheel URL."""
    name = _wheel_filename(url)
    if not name:
        return ""
    patterns = (
        rf"^{re.escape(package)}-(.+?)-cp\d+",
        rf"^{re.escape(package)}-(.+?)-py\d",
        rf"^{re.escape(package)}-(.+?)\.whl$",
    )
    for pattern in patterns:
        match = re.match(pattern, name, re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def _normalize_pkg_version(version: str) -> str:
    return version.strip().lower().replace("_", "-")


def _wheel_versions_match(installed: str, expected: str) -> bool | None:
    if not expected:
        return None
    if not installed:
        return False
    return _normalize_pkg_version(installed) == _normalize_pkg_version(expected)


def build_expected_wheels(
    torch_url: str = "",
    torchvision_url: str = "",
    torchaudio_url: str = "",
) -> dict[str, dict[str, str]]:
    specs = {
        "torch": ("torch", torch_url),
        "torchvision": ("torchvision", torchvision_url),
        "torchaudio": ("torchaudio", torchaudio_url),
    }
    out: dict[str, dict[str, str]] = {}
    for key, (package, url) in specs.items():
        url = (url or "").strip()
        if not url:
            continue
        out[key] = {
            "url": url,
            "filename": _wheel_filename(url),
            "version": _version_from_wheel_url(url, package),
        }
    return out


def attach_wheel_expectations(report: dict, expected_wheels: dict[str, dict[str, str]]) -> dict:
    ml = report.setdefault("python_ml_runtime", {})
    comparisons: dict[str, dict] = {}
    for key, spec in expected_wheels.items():
        installed = ml.get(key, "")
        expected_version = spec.get("version", "")
        comparisons[key] = {
            "installed": installed,
            "expected_version": expected_version,
            "expected_url": spec.get("url", ""),
            "expected_filename": spec.get("filename", ""),
            "match": _wheel_versions_match(installed, expected_version),
        }
    report["expected_wheels"] = expected_wheels
    report["wheel_comparisons"] = comparisons
    return report


def collect(python: str | None = None) -> dict:
    python = python or os.environ.get("COMFYUI_PYTHON") or sys.executable
    meta_files = _discover_metadata_files()
    rocm_path = _resolve_rocm_path()
    active_meta, active_meta_path = _pick_active_metadata(meta_files, rocm_path)
    if not rocm_path:
        rocm_path = _first_key(active_meta, "ROCM_PATH", "HIP_PATH")

    installed = _first_key(
        active_meta,
        "INSTALLED_VERSION",
        "ROCM_VERSION",
        "VERSION",
    )
    ml = _probe_ml_stack(python)
    hip_user = os.environ.get("HIP_PATH", "").strip() or rocm_path

    report = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "runner_name": os.environ.get("RUNNER_NAME", ""),
        "hostname": os.environ.get("RUNNER_HOST", os.environ.get("COMPUTERNAME", "")),
        "metadata_file": active_meta_path,
        "metadata_files_found": [str(p) for p in meta_files],
        "rocm_stack": {
            "rocm_active": _rocm_active_label(rocm_path, installed),
            "installed_version": installed,
            "rocm_path": rocm_path,
            "rocm_path_env": os.environ.get("ROCM_PATH", "").strip(),
            "hip_path_user": hip_user,
            "hip_path_machine": _machine_hip_path(meta_files),
            "source_tarball": _first_key(
                active_meta,
                "SOURCE_TARBALL",
                "TARBALL",
                "SOURCE_ARCHIVE",
                "DIST_TARBALL",
            ),
            "therock_commit": _first_key(
                active_meta,
                "THEROCK_COMMIT",
                "GIT_COMMIT",
                "COMMIT",
                "THEROCK_GIT_COMMIT",
            ),
            "installed_gfx_target": _first_key(
                active_meta,
                "INSTALLED_GFX_TARGET",
                "GFX_TARGET",
                "GPU_ARCH",
            ),
        },
        "python_ml_runtime": {
            "venv": _venv_display(python),
            "python": python,
            "python_version": ml.get("python_version", ""),
            "torch": ml.get("torch", ""),
            "torchvision": ml.get("torchvision", ""),
            "torchaudio": ml.get("torchaudio", ""),
            "hip_torch": ml.get("hip_torch", ""),
            "rocm_sdk": ml.get("rocm_sdk", ""),
            "device_wheels": _device_wheels(python),
            "gpu_detected": ml.get("cuda_avail", False),
            "gpu_name": ml.get("gpu_name", ""),
            "gpu_arch": ml.get("gpu_arch", "")
            or _first_key(active_meta, "INSTALLED_GFX_TARGET", "GFX_TARGET"),
            "probe_error": ml.get("probe_error") or ml.get("torch_error"),
        },
    }
    return report


def format_markdown(
    report: dict,
    expected_rocm: str = "",
    expected_wheels: dict[str, dict[str, str]] | None = None,
) -> str:
    rocm = report["rocm_stack"]
    ml = report["python_ml_runtime"]

    lines = [
        "## Runtime environment",
        "",
        f"Captured: `{report.get('captured_at', '')}`",
        f"Runner: `{report.get('runner_name') or report.get('hostname') or 'unknown'}`",
        "",
        "### ROCm stack",
        "",
        "| Component | Version / Path |",
        "|-----------|----------------|",
        f"| ROCm (active) | `{rocm.get('rocm_active', '')}` |",
        f"| ROCM_PATH | `{rocm.get('rocm_path') or rocm.get('rocm_path_env', '')}` |",
        f"| HIP_PATH (user) | `{rocm.get('hip_path_user', '')}` |",
        f"| HIP_PATH (machine) | `{rocm.get('hip_path_machine', '')}` |",
        f"| Source tarball | `{rocm.get('source_tarball', '')}` |",
        f"| TheRock commit | `{rocm.get('therock_commit', '')}` |",
        f"| Metadata | `{report.get('metadata_file', '')}` |",
        "",
        "### Python / ML runtime (ComfyUI)",
        "",
        "| Component | Installed | Expected (workflow URL) | Status |",
        "|-----------|-----------|-------------------------|--------|",
    ]

    comparisons = report.get("wheel_comparisons") or {}
    for key, label in (
        ("torch", "PyTorch"),
        ("torchvision", "torchvision"),
        ("torchaudio", "torchaudio"),
    ):
        installed = ml.get(key, "")
        comp = comparisons.get(key, {})
        expected_version = comp.get("expected_version", "")
        expected_url = comp.get("expected_url", "")
        if expected_url:
            expected_cell = f"[{expected_version or comp.get('expected_filename', 'wheel')}]({expected_url})"
            match = comp.get("match")
            if match is True:
                status = "match"
            elif match is False:
                status = "**MISMATCH**"
            else:
                status = "could not verify"
        else:
            expected_cell = "—"
            status = "—"
        lines.append(f"| {label} | `{installed}` | {expected_cell} | {status} |")

    lines.extend([
        "",
        "| Component | Version / Path |",
        "|-----------|----------------|",
        f"| Venv | `{ml.get('venv', '')}` |",
        f"| Python | `{ml.get('python_version', '')}` |",
        f"| HIP (torch) | `{ml.get('hip_torch', '')}` |",
        f"| rocm-sdk | `{ml.get('rocm_sdk', '')}` |",
        f"| Device wheels | `{ml.get('device_wheels', '')}` |",
    ])
    gpu_line = "Yes" if ml.get("gpu_detected") else "No"
    arch = ml.get("gpu_arch", "")
    if arch:
        gpu_line += f" (cuda_avail {ml.get('gpu_detected')}, arch {arch})"
    lines.append(f"| GPU detected | {gpu_line} |")
    if ml.get("probe_error"):
        lines.append(f"| Probe error | `{ml['probe_error']}` |")

    detected = rocm.get("installed_version") or rocm.get("rocm_active", "").split(" ->")[0]
    match = _match_expected(detected, expected_rocm)
    if expected_rocm:
        status = "match" if match else "MISMATCH"
        if match is None:
            status = "could not verify"
        lines.extend(["", f"**Expected ROCm (workflow input):** `{expected_rocm}` — **{status}**"])

    wheels = expected_wheels or report.get("expected_wheels") or {}
    if wheels:
        lines.extend(["", "### Wheel URLs (workflow inputs)", ""])
        for key in ("torch", "torchvision", "torchaudio"):
            spec = wheels.get(key)
            if spec and spec.get("url"):
                label = {"torch": "PyTorch", "torchvision": "torchvision", "torchaudio": "torchaudio"}[key]
                lines.append(f"- **{label}:** [{spec.get('filename') or spec.get('url')}]({spec['url']})")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture ROCm + ML runtime environment")
    parser.add_argument("--python", help="ComfyUI / torch Python interpreter")
    parser.add_argument("--output", type=Path, help="Write JSON report to this path")
    parser.add_argument("--markdown", type=Path, help="Write Markdown table to this path")
    parser.add_argument("--github-summary", type=Path, help="Alias for --markdown (step summary file)")
    parser.add_argument(
        "--expected-rocm",
        default="",
        help="Workflow ROCm version pattern (e.g. 7.14.*); warns on mismatch",
    )
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="Exit 1 when --expected-rocm is set and installed ROCm does not match",
    )
    parser.add_argument("--expected-torch-url", default="", help="Expected PyTorch wheel URL")
    parser.add_argument("--expected-torchvision-url", default="", help="Expected torchvision wheel URL")
    parser.add_argument("--expected-torchaudio-url", default="", help="Expected torchaudio wheel URL")
    parser.add_argument(
        "--fail-on-wheel-drift",
        action="store_true",
        help="Exit 1 when wheel URLs are set and installed versions do not match",
    )
    args = parser.parse_args()

    expected_wheels = build_expected_wheels(
        torch_url=args.expected_torch_url,
        torchvision_url=args.expected_torchvision_url,
        torchaudio_url=args.expected_torchaudio_url,
    )

    report = collect(args.python)
    attach_wheel_expectations(report, expected_wheels)
    md_path = args.markdown or args.github_summary

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")

    md = format_markdown(report, expected_rocm=args.expected_rocm, expected_wheels=expected_wheels)
    print(md)

    if md_path:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md, encoding="utf-8")

    rocm = report["rocm_stack"]
    detected = rocm.get("installed_version") or ""
    if not detected and " -> " in rocm.get("rocm_active", ""):
        detected = rocm["rocm_active"].split(" ->", 1)[0].strip()

    match = _match_expected(detected, args.expected_rocm)
    if args.expected_rocm and match is False:
        print(f"::warning::ROCm drift: detected '{detected}' does not match expected '{args.expected_rocm}'")
        if args.fail_on_drift:
            return 1

    ml = report["python_ml_runtime"]
    if ml.get("probe_error"):
        print(f"::warning::ML runtime probe: {ml['probe_error']}")
    if not ml.get("gpu_detected"):
        print("::warning::PyTorch does not see a GPU (cuda/HIP unavailable)")

    wheel_fail = False
    for key, comp in report.get("wheel_comparisons", {}).items():
        if comp.get("match") is False:
            wheel_fail = True
            print(
                f"::warning::{key} drift: installed '{comp.get('installed')}' "
                f"!= expected '{comp.get('expected_version')}' "
                f"from {comp.get('expected_filename') or comp.get('expected_url')}"
            )
    if wheel_fail and args.fail_on_wheel_drift:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
