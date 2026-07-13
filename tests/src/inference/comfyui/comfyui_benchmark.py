#!/usr/bin/env python3
"""Standalone ComfyUI validation/benchmark runner (framework-ci suite entry).

Mirrors the xDiT/vLLM standalone pattern: one script, no BaseTest, that CI and a
bare node both invoke the same way. It does four things:

  1. Resolve which ComfyUI tests to run (from config/models_config.yaml,
     filtered by --model/--tags; GPU arch does not gate selection).
  2. Drive each test through the hardened ComfyUI executor
     (executors/comfyui_validator.py -> single_test_protocol.py), which talks to
     a running ComfyUI over its HTTP API on real GPU hardware.
  3. Convert each test's native evidence (summary.json/results.json) into a
     framework-ci `results_<test>.json` (the ROCm-QA RASTRA result shape:
     test_config / test_metrics / test_result / test_execution_time in minutes),
     carrying provenance + detected arch.
  4. Write benchmark_summary.json.

Two ways to run:
  * CI / bare-metal node -> auto-discovers, clones (if needed), and starts ComfyUI.
  * Optional: pass --docker-image to self-wrap in a container (not used by CI).

Examples:
  python tests/src/inference/comfyui/comfyui_benchmark.py \
      --model comfyui_stable_diffusion_2_1 --arch gfx942 \
      --rocm-version 7.15.0 --results-dir logs/benchmark_results

  python tests/src/inference/comfyui/comfyui_benchmark.py \
      --tags smoke --arch gfx942 --results-dir logs/benchmark_results

The heavy trust properties (provenance stamping, model-identity SKIP, IO->INFRA
classification, per-test server-log capture, frozen 0/1/2/3 exit codes) live in
the ported executors and flow into the results this script writes.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml  # type: ignore[import-not-found]

SUITE_DIR = Path(__file__).resolve().parent
EXECUTORS_DIR = SUITE_DIR / "executors"
CONFIG_DIR = SUITE_DIR / "config"
sys.path.insert(0, str(EXECUTORS_DIR))

CODE_TO_STATUS = {0: "PASS", 1: "FAIL", 2: "SKIP", 3: "INFRA_ERROR"}

DOCKER_GPU_FLAGS_DEFAULT = (
    "--network=host --device=/dev/kfd --device=/dev/dri --group-add video "
    "--ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined "
    "--shm-size=16g -v /sys:/sys:ro"
)


# --------------------------------------------------------------------------- #
# console (shared-helper style with plain fallback)
# --------------------------------------------------------------------------- #
def status(msg, kind="info"):
    print(f"[{kind.upper()}] {msg}", flush=True)


def phase(title, kind="phase"):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}", flush=True)


# --------------------------------------------------------------------------- #
# arch / device detection (delegates to the ported executor's logic)
# --------------------------------------------------------------------------- #
def detect_arch():
    try:
        import comfyui_validator
        return comfyui_validator.detect_gpu_arch()
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def load_yaml(path):
    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return {}


def load_manifest(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def select_tests(models_cfg, manifest, *, model, tags, arch, os_family):
    """Pick tests from models_config.yaml, cross-checked against the manifest.
    Selection precedence: explicit --model > --tags > all enabled.
    GPU arch is not used for filtering (any arch may run any test)."""
    by_name = {t["test_name"]: t for t in manifest.get("tests", [])}
    want_tags = {t.strip() for t in (tags or "").split(",") if t.strip()}
    selected = []
    for entry in models_cfg.get("tests", []):
        name = entry["name"]
        mt = by_name.get(name, {})
        if not mt.get("enabled", True):
            continue
        if os_family and os_family not in mt.get("os", entry.get("os", [])):
            continue
        if model and name != model:
            continue
        if want_tags and not (want_tags & set(entry.get("tags", mt.get("tags", [])))):
            continue
        selected.append({**entry, "_manifest": mt})
    return selected


# --------------------------------------------------------------------------- #
# run one test via the hardened executor
# --------------------------------------------------------------------------- #
def run_one(entry, args):
    """Invoke comfyui_validator.py for one test into a per-test evidence dir.
    Returns (status_str, evidence_dir)."""
    name = entry["name"]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.results_dir) / f"{name}_{ts}"
    cmd = [
        sys.executable, str(EXECUTORS_DIR / "comfyui_validator.py"),
        "--execute", name,
        "--execution_label", args.execution_label or "manual",
        "--rocm_version", args.rocm_version or "unknown",
        "--gpu_arch", args.arch or "unknown",
        "--os_version", args.os_version or "linux",
        "--comfyui_url", args.comfyui_url,
        "--manifest", str(CONFIG_DIR / "suite_manifest.json"),
        "--output_dir", str(out_dir),
    ]
    status(f"run {name}", kind="run")
    rc = subprocess.run(cmd).returncode
    return CODE_TO_STATUS.get(rc, "FAIL"), out_dir


# --------------------------------------------------------------------------- #
# convert native evidence -> framework-ci results_<test>.json
# --------------------------------------------------------------------------- #
def _minutes(seconds):
    try:
        return round(float(seconds) / 60.0, 4)
    except (TypeError, ValueError):
        return None


# Metrics where a HIGHER score is better (throughput-like); everything else
# (latency, time) is lower-is-better. Lets us stamp the RASTRA flag correctly so
# future non-latency metrics aren't mislabeled.
_HIGHER_IS_BETTER = {"throughput", "images_per_s", "frames_per_s", "tokens_per_s"}


def _make_metric(metric_name, score, unit, *, primary=False):
    """Build a RASTRA test_metric with the correct H/L flag for its name."""
    flag = "H" if metric_name in _HIGHER_IS_BETTER else "L"
    return {
        "flag": flag, "metric_name": metric_name, "primary": primary,
        "score": score, "unit": unit,
    }


def _extract_workflow_params(entry):
    """Extract generation params from this test's workflow graph (best-effort).

    Returns a flat dict (steps/cfg/sampler/resolution/frames/prompt/...) or {}
    if the workflow can't be located or parsed. Never raises."""
    mt = entry.get("_manifest", {}) or {}
    workflow_rel = mt.get("workflow") or entry.get("workflow")
    if not workflow_rel:
        return {}
    workflow_path = SUITE_DIR / workflow_rel
    if not workflow_path.exists():
        return {}
    try:
        import workflow_params
        return workflow_params.extract_params(workflow_path)
    except Exception:
        return {}


def write_result_json(entry, status_str, evidence_dir, results_dir, meta):
    """Write results_<test>.json in the framework-ci RASTRA result shape from the
    executor's native summary.json/results.json. Latency (if any) becomes the
    primary metric. Provenance + detected arch travel in test_config."""
    name = entry["name"]
    summary = {}
    results = {}
    # The validator writes into evidence_dir; find its summary/results.
    sj = next(iter(evidence_dir.glob("summary.json")), None) if evidence_dir.exists() else None
    if sj is None and evidence_dir.exists():
        # nested (validator makes <execute>_<ts>/ inside our out_dir)
        sj = next(iter(evidence_dir.glob("*/summary.json")), None)
    if sj and sj.exists():
        summary = json.loads(sj.read_text(encoding="utf-8"))
        rj = sj.parent / "results.json"
        if rj.exists():
            results = json.loads(rj.read_text(encoding="utf-8"))

    latency = results.get("latency_avg_s")
    metrics = []
    if latency is not None:
        metrics.append(_make_metric("latency", latency, "seconds", primary=True))

    # Generation params (steps/cfg/sampler/resolution/frames/prompt) pulled from
    # the workflow graph so RASTRA rows are queryable/comparable. Flattened into
    # test_config by upload_from_artifacts.py's test_parameters handling.
    test_params = _extract_workflow_params(entry)

    result = {
        "results": [
            {
                "test_config": {
                    "test_name": name,
                    "sub_test_name": name,
                    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                    "environment_dependencies": [],
                    "device": meta.get("device", ""),
                    "detected_gpu_arch": meta.get("detected_arch", ""),
                    "requested_gpu_arch": meta.get("arch", ""),
                    "test_domain": "ml",
                    "metric": "latency",
                    "tags": ",".join(entry.get("tags", [])),
                    "tms_key": summary.get("tms_key") or entry.get("tms_key"),
                    "rocm_version": meta.get("rocm_version", ""),
                    "git_sha": meta.get("git_sha"),
                    "workflow_hash": meta.get("workflow_hash"),
                    "canonical": meta.get("canonical"),
                    **({"test_parameters": test_params} if test_params else {}),
                },
                "test_execution_time": _minutes(summary.get("duration_s")),
                "test_log": str((sj.parent / "comfyui_server_tail.log")) if sj else "",
                "test_metrics": metrics,
                # RASTRA is binary PASS/FAIL; SKIP/INFRA_ERROR are recorded in
                # test_config.result_status so the honest verdict is never lost.
                "test_result": "PASS" if status_str == "PASS" else "FAIL",
                "result_status": status_str,
                "failure_reason": summary.get("failure_reason", ""),
                "test_start_time": summary.get("timestamp")
                or datetime.now(timezone.utc).isoformat(),
            }
        ]
    }
    out = Path(results_dir) / f"results_{name}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
def _git(a):
    try:
        p = subprocess.run(["git", "-C", str(SUITE_DIR), *a],
                           capture_output=True, text=True, timeout=15)
        return p.stdout.strip() if p.returncode == 0 else None
    except Exception:
        return None


def provenance():
    sha = _git(["rev-parse", "--short", "HEAD"])
    st = _git(["status", "--porcelain"])
    dirty = None if st is None else (st != "")
    return {
        "git_sha": sha,
        "canonical": (not dirty) if isinstance(dirty, bool) else None,
        "workflow_hash": None,  # populated by the fleet's hashing if desired
    }


# --------------------------------------------------------------------------- #
# docker self-orchestration (node convenience, mirrors xdit)
# --------------------------------------------------------------------------- #
def _docker_prefix():
    try:
        subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True)
        return ["docker"]
    except Exception:
        return ["sudo", "docker"]


def run_in_docker(args):
    image = args.docker_image
    phase(f"COMFYUI IN DOCKER: {image}", kind="run")
    cwd = Path.cwd().resolve()
    script = Path(__file__).resolve()
    try:
        inner = f"/workspace/{script.relative_to(cwd).as_posix()}"
        mounts = []
    except ValueError:
        mounts = ["-v", f"{script.parent}:/comfy_runner:ro"]
        inner = f"/comfy_runner/{script.name}"
    inner_args = _inner_args(args)
    inner_cmd = (f"pip install -q pyyaml 2>/dev/null || true; "
                 f"python3 {inner} {inner_args}; exit $?")
    docker_cmd = _docker_prefix() + [
        "run", "--rm", "--entrypoint", "/bin/bash",
        *shlex.split(args.docker_gpu_flags or DOCKER_GPU_FLAGS_DEFAULT),
        "-e", "COMFYUI_IN_CONTAINER=1", "-e", "PYTHONUNBUFFERED=1",
    ]
    for var in ("EXECUTION_LABEL", "HF_TOKEN", "COMFYUI_PATH"):
        if os.environ.get(var):
            docker_cmd += ["-e", f"{var}={os.environ[var]}"]
    docker_cmd += mounts + ["-v", f"{cwd}:/workspace", "-w", "/workspace",
                            image, "-c", inner_cmd]
    return subprocess.run(docker_cmd).returncode


def _inner_args(args):
    parts = []
    for flag, val in (("--model", args.model), ("--tags", args.tags),
                      ("--arch", args.arch), ("--rocm-version", args.rocm_version),
                      ("--os-version", args.os_version),
                      ("--comfyui-url", args.comfyui_url),
                      ("--results-dir", args.results_dir),
                      ("--execution-label", args.execution_label)):
        if val:
            parts.append(f"{flag} {shlex.quote(str(val))}")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="ComfyUI validation benchmark (standalone)")
    p.add_argument("--model", default="", help="single test_name to run")
    p.add_argument("--tags", default="", help="comma-separated tags to select")
    p.add_argument("--arch", default="", help="GPU arch (else auto-detected)")
    p.add_argument("--os-version", dest="os_version", default="linux")
    p.add_argument("--rocm-version", dest="rocm_version", default="")
    p.add_argument("--comfyui-url", dest="comfyui_url", default="http://127.0.0.1:8188")
    p.add_argument("--results-dir", dest="results_dir", default="logs/benchmark_results")
    p.add_argument("--execution-label", dest="execution_label",
                   default=os.environ.get("EXECUTION_LABEL", "manual"))
    p.add_argument("--config-file", dest="config_file",
                   default=str(CONFIG_DIR / "models_config.yaml"))
    p.add_argument("--docker-image", dest="docker_image", default="")
    p.add_argument("--no-autostart", dest="no_autostart", action="store_true",
                   help="Fail if ComfyUI is not already running (do not launch).")
    p.add_argument("--no-bootstrap", dest="no_bootstrap", action="store_true",
                   help="Do not git-clone ComfyUI; only start an existing install.")
    p.add_argument("--startup-timeout-s", dest="startup_timeout_s", type=int, default=300,
                   help="Seconds to wait for ComfyUI to become reachable.")
    return p.parse_args()


def load_profile():
    """Build a machine profile from config file + workflow env overrides."""
    profile = {}
    prof_path = CONFIG_DIR / "machine_profile.json"
    if prof_path.exists():
        try:
            profile = json.loads(prof_path.read_text(encoding="utf-8"))
        except Exception:
            profile = {}
    if os.environ.get("COMFYUI_PATH"):
        profile["comfyui_path"] = os.environ["COMFYUI_PATH"]
    if os.environ.get("COMFYUI_PYTHON"):
        profile["python_bin"] = os.environ["COMFYUI_PYTHON"]
    return profile


def ensure_server(args):
    """Locate, bootstrap, and start ComfyUI when not already reachable.

    Autostart is the default on all platforms (Linux and Windows). Pass
    --no-autostart or set COMFYUI_NO_AUTOSTART=1 to opt out.
    """
    import comfyui_runtime

    if comfyui_runtime.comfyui_up(args.comfyui_url):
        status(f"ComfyUI already up at {args.comfyui_url}", kind="info")
        return True

    if args.no_autostart or os.environ.get("COMFYUI_NO_AUTOSTART", "").lower() in ("1", "true", "yes"):
        status(f"ComfyUI not reachable at {args.comfyui_url} and autostart disabled",
               kind="error")
        return False

    status(f"ComfyUI not reachable at {args.comfyui_url}; bootstrapping/starting...",
           kind="info")
    allow_bootstrap = not args.no_bootstrap and \
        os.environ.get("COMFYUI_NO_BOOTSTRAP", "").lower() not in ("1", "true", "yes")
    return comfyui_runtime.ensure_comfyui_running(
        load_profile(), args.comfyui_url,
        allow_bootstrap=allow_bootstrap,
        timeout_s=args.startup_timeout_s,
    )


def detect_os_version():
    if os.name == "nt":
        return "windows"
    return "linux"


def os_family(os_version):
    return "windows" if "windows" in (os_version or "").lower() else "linux"


def recover_server(args):
    """Restart ComfyUI after a failed test so the next test starts clean."""
    import comfyui_runtime
    profile = load_profile()
    status("recovering ComfyUI server after non-PASS test", kind="info")
    try:
        if comfyui_runtime.comfyui_up(args.comfyui_url):
            return comfyui_runtime.restart_comfyui(
                profile, args.comfyui_url, timeout_s=args.startup_timeout_s)
        return comfyui_runtime.ensure_comfyui_running(
            profile, args.comfyui_url,
            allow_bootstrap=not args.no_bootstrap,
            timeout_s=args.startup_timeout_s,
        )
    except Exception as e:
        status(f"ComfyUI recovery failed: {e}", kind="error")
        return False


def main():
    args = parse_args()
    if args.docker_image and not os.environ.get("COMFYUI_IN_CONTAINER"):
        return run_in_docker(args)

    phase("COMFYUI BENCHMARK", kind="run")
    if not args.os_version or args.os_version == "linux":
        if os.name == "nt":
            args.os_version = detect_os_version()
    if not args.arch:
        args.arch = detect_arch()
        status(f"detected arch: {args.arch or '(unknown)'}", kind="info")

    if not ensure_server(args):
        status("could not reach or start ComfyUI", kind="error")
        return 1

    models_cfg = load_yaml(args.config_file)
    manifest = load_manifest(CONFIG_DIR / "suite_manifest.json")
    fam = os_family(args.os_version)
    selected = select_tests(models_cfg, manifest, model=args.model,
                            tags=args.tags, arch=args.arch, os_family=fam)
    status(f"selected {len(selected)} test(s) for arch={args.arch}/os={fam}",
           kind="info")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    prov = provenance()
    meta = {"arch": args.arch, "detected_arch": detect_arch(),
            "rocm_version": args.rocm_version, "device": "",
            **prov}

    written, counts = 0, {"PASS": 0, "FAIL": 0, "SKIP": 0, "INFRA_ERROR": 0}
    for entry in selected:
        st, evidence = run_one(entry, args)
        counts[st] = counts.get(st, 0) + 1
        write_result_json(entry, st, evidence, results_dir, meta)
        written += 1
        if st in ("FAIL", "INFRA_ERROR"):
            recover_server(args)

    summary = {
        "success": counts["FAIL"] == 0 and counts["INFRA_ERROR"] == 0,
        "suite": "comfyui",
        "arch": args.arch,
        "detected_arch": meta["detected_arch"],
        "rocm_version": args.rocm_version,
        "results_written": written,
        "counts": counts,
        "git_sha": prov["git_sha"],
        "canonical": prov["canonical"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (results_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    status(f"done: {counts} -> {results_dir}", kind="ok")
    # Exit non-zero if any real failure (FAIL/INFRA); SKIP is fine.
    return 1 if (counts["FAIL"] or counts["INFRA_ERROR"]) else 0


if __name__ == "__main__":
    sys.exit(main())
