#!/usr/bin/env python3
"""
Layer    : Gate
Equiv    : ROCm env setup step in reusable-test-executor.yml (production)
Purpose  : Validates ROCm stack, GPU visibility, VRAM, and ComfyUI server
           before any test runs. Always writes a structured JSON report.

Exit codes:
  0  READY      - all checks passed
  1  NOT_READY  - critical check failed, abort suite
  2  PARTIAL    - non-critical warning, suite can continue (unless --strict)
"""

import argparse, json, os, platform, re, socket, subprocess, sys, urllib.request
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--expected-rocm",     required=True, help="e.g. 7.14.0 or 7.14.*")
    p.add_argument("--expected-gpu-arch", required=True, help="e.g. gfx1150")
    p.add_argument("--comfyui-url",       default="http://127.0.0.1:8188")
    p.add_argument("--min-vram-mb",       type=int, default=2048)
    p.add_argument("--output",            default="preflight_report.json")
    p.add_argument("--python",            default=None,
                   help="Interpreter to run the torch/GPU checks with. Defaults "
                        "to this process. On a TheRock multi-arch box the GPU "
                        "torch lives in the ComfyUI venv, so the runner passes "
                        "that interpreter here.")
    p.add_argument("--strict",            action="store_true",
                   help="Treat PARTIAL as NOT_READY")
    p.add_argument("--doctor",            action="store_true",
                   help="Diagnose this machine end-to-end (git freshness, "
                        "machine profile + ComfyUI venv torch, model coverage) "
                        "and print a concrete FIX for every problem. Exits 3 "
                        "only for infra breakage, never for missing models.")
    return p.parse_args()


# framework-ci-comfy: preflight lives in the suite executors/ dir. REPO_ROOT is
# the suite root; sibling modules (model_check, comfyui_runtime) import from the
# executors dir, and configs live under config/.
EXECUTORS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXECUTORS_DIR.parent
CONFIG_DIR = REPO_ROOT / "config"


# --------------------------------------------------------------------------- #
# doctor mode - end-to-end machine diagnosis with actionable fixes
# --------------------------------------------------------------------------- #
def _run(cmd, timeout=15):
    """Run a command; return (returncode, stdout, stderr). Never raises."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except (OSError, subprocess.SubprocessError) as e:
        return 127, "", str(e)


def _venv_python(venv_dir):
    """Resolve the interpreter inside a venv dir, cross-platform, or None."""
    venv_dir = Path(os.path.expandvars(os.path.expanduser(str(venv_dir))))
    cand = (venv_dir / "Scripts" / "python.exe") if os.name == "nt" \
        else (venv_dir / "bin" / "python")
    return cand if cand.exists() else None


def doctor_git_freshness(repo_root=REPO_ROOT):
    """Is the checkout clean and up to date with origin? Returns a check dict
    {status, detail, fix}. status: PASS | WARN | INFRA (INFRA only if git is
    entirely unavailable, which blocks reproducibility)."""
    rc, _, _ = _run(["git", "-C", str(repo_root), "rev-parse", "--git-dir"])
    if rc != 0:
        return {"status": "WARN", "detail": "not a git checkout / git unavailable",
                "fix": "Clone via git so runs are reproducible and stamped."}
    _, status, _ = _run(["git", "-C", str(repo_root), "status", "--porcelain"])
    dirty = status != ""
    # Behind origin? Best-effort; skip network fetch, compare against cached ref.
    _, counts, _ = _run(["git", "-C", str(repo_root), "rev-list",
                         "--left-right", "--count", "HEAD...@{u}"])
    behind = None
    if counts and "\t" in counts:
        try:
            behind = int(counts.split("\t")[1])
        except (ValueError, IndexError):
            behind = None
    problems, fixes = [], []
    if dirty:
        problems.append("tree is dirty (uncommitted changes)")
        fixes.append("git stash  # or commit, then re-run")
    if behind:
        problems.append(f"{behind} commit(s) behind origin")
        fixes.append("git pull --ff-only")
    if problems:
        return {"status": "WARN", "detail": "; ".join(problems),
                "fix": " ; ".join(fixes)}
    return {"status": "PASS", "detail": "clean and up to date", "fix": ""}


def doctor_machine_profile(repo_root=REPO_ROOT):
    """Does the machine profile parse, and does its comfyui_venv have a Python
    that imports torch with HIP/CUDA available? Returns {status, detail, fix}.
    status INFRA if the venv exists but its torch cannot see the GPU (that truly
    blocks real runs); WARN if the profile/venv is simply not configured yet."""
    prof_path = Path(repo_root) / "config" / "machine_profile.json"
    if not prof_path.exists():
        return {"status": "WARN", "detail": "configs/machine_profile.json missing",
                "fix": "cp configs/machine_profile.example.json "
                       "configs/machine_profile.json  # then edit comfyui_venv"}
    try:
        profile = json.loads(prof_path.read_text())
    except Exception as e:
        return {"status": "INFRA", "detail": f"machine_profile.json invalid: {e}",
                "fix": "Fix the JSON syntax in configs/machine_profile.json"}

    venv = profile.get("comfyui_venv")
    if not venv:
        return {"status": "WARN", "detail": "comfyui_venv not set in profile",
                "fix": "Set 'comfyui_venv' to a venv containing ROCm torch."}
    py = _venv_python(venv)
    if py is None:
        return {"status": "INFRA",
                "detail": f"comfyui_venv python not found under {venv}",
                "fix": f"Create the venv or fix comfyui_venv path ({venv})."}
    rc, out, err = _run([str(py), "-c",
                         "import torch; print(bool(torch.cuda.is_available()))"])
    if rc != 0:
        return {"status": "INFRA",
                "detail": f"{py} cannot import torch ({(err or out)[:120]})",
                "fix": "Install ROCm torch into the comfyui_venv (AMD TheRock "
                       "index); see README 'Fresh machine setup'."}
    if out.strip() != "True":
        return {"status": "INFRA",
                "detail": f"{py} has torch but GPU not available (cuda/HIP False)",
                "fix": "Check ROCm/driver; torch.cuda.is_available() must be True."}
    return {"status": "PASS", "detail": f"{py} torch sees the GPU", "fix": ""}


def doctor_model_coverage(gpu_arch, repo_root=REPO_ROOT, comfy_path=None):
    """Per enabled manifest test eligible for this arch, report model present or
    missing (content, never infra). Returns {status, rows, detail}. status is
    always PASS or WARN - missing models are a content concern, not infra."""
    rows = []
    try:
        sys.path.insert(0, str(Path(repo_root) / "executors"))
        import model_check
    except Exception as e:
        return {"status": "WARN", "rows": [],
                "detail": f"model_check unavailable: {e}"}

    manifest_path = Path(repo_root) / "config" / "suite_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as e:
        return {"status": "WARN", "rows": [], "detail": f"manifest unreadable: {e}"}

    if comfy_path is None:
        try:
            import comfyui_runtime
            profile = {}
            pp = Path(repo_root) / "config" / "machine_profile.json"
            if pp.exists():
                try:
                    profile = json.loads(pp.read_text())
                except Exception:
                    profile = {}
            found = comfyui_runtime.discover_comfyui_path(profile)
            comfy_path = str(found) if found else None
        except Exception:
            comfy_path = None

    if not comfy_path:
        for t in manifest.get("tests", []):
            if not t.get("enabled", True):
                continue
            rows.append((t["test_name"], "UNKNOWN", "no ComfyUI path to check"))
        return {"status": "WARN", "rows": rows,
                "detail": "ComfyUI install not found - cannot check model "
                          "presence (set comfyui_path/$COMFYUI_PATH)"}

    missing_total = 0
    for t in manifest.get("tests", []):
        if not t.get("enabled", True):
            continue
        name = t["test_name"]
        try:
            missing = model_check.missing_models(t["workflow"], comfy_path)
        except Exception as e:
            rows.append((name, "ERROR", str(e)[:80]))
            continue
        if missing:
            missing_total += 1
            files = ", ".join(m[0] for m in missing)
            rows.append((name, "MISSING", files))
        else:
            rows.append((name, "PRESENT", ""))
    status = "WARN" if missing_total else "PASS"
    detail = (f"{missing_total} eligible test(s) missing model weights"
              if missing_total else "all eligible tests have their models")
    return {"status": status, "rows": rows, "detail": detail}


def run_doctor(gpu_arch, comfyui_url, repo_root=REPO_ROOT):
    """Run the doctor checks, print human-readable results + FIX lines, and
    return an exit code. Exit 3 ONLY for infra breakage (git fatal, venv/torch
    unusable, ComfyUI down); missing models never change the exit code."""
    print("=" * 66)
    print("ComfyUI AMD validation - DOCTOR")
    print("=" * 66)

    infra_broken = False

    git = doctor_git_freshness(repo_root)
    print(f"\n[git]      {git['status']}: {git['detail']}")
    if git["status"] == "INFRA":
        infra_broken = True
    if git.get("fix"):
        print(f"           FIX: {git['fix']}")

    prof = doctor_machine_profile(repo_root)
    print(f"\n[profile]  {prof['status']}: {prof['detail']}")
    if prof["status"] == "INFRA":
        infra_broken = True
    if prof.get("fix"):
        print(f"           FIX: {prof['fix']}")

    comfy = check_comfyui(comfyui_url)
    print(f"\n[comfyui]  {comfy['status']}: {comfy['detail']}")
    if comfy["status"] != "PASS":
        infra_broken = True
        print(f"           FIX: start ComfyUI (see RUNBOOK) or set comfyui_url; "
              f"tried {comfyui_url}")

    models = doctor_model_coverage(gpu_arch, repo_root)
    print(f"\n[models]   {models['status']}: {models['detail']}")
    for name, state, extra in models["rows"]:
        line = f"           - {name:52} {state}"
        if extra:
            line += f"  ({extra})"
        print(line)
    if models["status"] == "WARN":
        print("           FIX: missing weights are auto-downloaded when "
              "AUTO_FETCH_MODELS=true (default); set HF_TOKEN for gated models "
              "or run: python executors/fetch_models.py --workflow <wf> "
              "--comfyui_path $COMFYUI_PATH")

    print("\n" + "=" * 66)
    if infra_broken:
        print("DOCTOR: infrastructure problem(s) found - fix the FIX lines above.")
        return 3
    print("DOCTOR: no infrastructure blockers. (Missing models, if any, only "
          "cause SKIPs.)")
    return 0


def check_rocminfo(expected_arch):
    """Run rocminfo and extract detected GPU arch. Informational only: on a
    TheRock multi-arch install rocminfo / /opt/rocm may be absent because ROCm
    ships inside the torch wheels, so a missing rocminfo is a WARN, not a FAIL
    (torch is the authoritative GPU source - see check_pytorch_rocm)."""
    try:
        out = subprocess.check_output(["rocminfo"], text=True, timeout=30)
        match = re.search(r"Name:\s+(gfx\w+)", out)
        arch = match.group(1) if match else "unknown"
        return {
            "status":   "PASS" if arch == expected_arch else "WARN",
            "detected": arch,
            "detail":   f"Detected {arch}, expected {expected_arch}",
        }
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        return {"status": "WARN", "detected": "none",
                "detail": f"rocminfo unavailable ({e}); using torch for GPU arch."}


def check_rocm_version(expected_pattern, hip_version=None):
    """Determine the ROCm version from /opt/rocm/.info/version, then rocminfo,
    then torch.version.hip (passed in as hip_version). Informational: if it can't
    be determined (common on multi-arch), report WARN rather than FAIL."""
    version_file = Path("/opt/rocm/.info/version")
    version = "unknown"
    try:
        if version_file.exists():
            version = version_file.read_text().strip()
        else:
            out = subprocess.check_output(["rocminfo"], text=True, timeout=30)
            m = re.search(r"ROCm Version:\s+([\d.]+)", out)
            version = m.group(1) if m else "unknown"
    except Exception:
        version = "unknown"

    if version == "unknown" and hip_version:
        version = hip_version

    if version == "unknown":
        return {"status": "WARN", "detected": "unknown",
                "detail": "ROCm version not determinable (no /opt/rocm, no "
                          "rocminfo, no torch.version.hip)."}

    pattern = expected_pattern.replace("*", r"\d+")
    match = bool(re.fullmatch(pattern.replace(".", r"\."), version))
    return {
        "status":   "PASS" if match else "WARN",
        "detected": version,
        "detail":   f"Detected {version}, pattern {expected_pattern}",
    }


def check_pytorch_rocm(python=None):
    """Verify PyTorch can see the GPU via ROCm HIP, in the given interpreter.
    Returns the device name plus the detected gfx arch and HIP version, which the
    rest of preflight treats as the authoritative GPU facts."""
    python = python or sys.executable
    probe = (
        "import json, torch\n"
        "ok = torch.cuda.is_available()\n"
        "name = torch.cuda.get_device_name(0) if ok else ''\n"
        "arch = ''\n"
        "try:\n"
        "    arch = torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]\n"
        "except Exception: pass\n"
        "print(json.dumps({'ok': ok, 'name': name, 'arch': arch, "
        "'hip': getattr(torch.version, 'hip', None)}))"
    )
    try:
        result = subprocess.run([python, "-c", probe],
                                capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and result.stdout.strip():
            info = json.loads(result.stdout.strip().splitlines()[-1])
            ok = bool(info.get("ok"))
            return {"status": "PASS" if ok else "FAIL",
                    "detail": info.get("name") or "CUDA/HIP not available",
                    "gpu_arch": info.get("arch") or "unknown",
                    "hip_version": info.get("hip")}
        return {"status": "FAIL",
                "detail": result.stderr.strip() or "torch probe failed",
                "gpu_arch": "unknown", "hip_version": None}
    except Exception as e:
        return {"status": "FAIL", "detail": str(e),
                "gpu_arch": "unknown", "hip_version": None}


def check_vram(min_mb, python=None):
    """Get free VRAM via PyTorch."""
    python = python or sys.executable
    try:
        result = subprocess.run(
            [python, "-c",
             "import torch; "
             "free, total = torch.cuda.mem_get_info(0); "
             "print(free // 1024**2, total // 1024**2)"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split()
            free_mb, total_mb = int(parts[0]), int(parts[1])
            ok = free_mb >= min_mb
            return {"status": "PASS" if ok else "WARN",
                    "free_mb": free_mb, "total_mb": total_mb,
                    "detail": f"{free_mb} MB free of {total_mb} MB total"}
        return {"status": "WARN", "free_mb": 0, "total_mb": 0,
                "detail": result.stderr.strip()}
    except Exception as e:
        return {"status": "WARN", "free_mb": 0, "total_mb": 0, "detail": str(e)}


def check_matmul(python=None):
    """Run a 512x512 fp16 matmul and measure latency."""
    python = python or sys.executable
    try:
        result = subprocess.run(
            [python, "-c",
             "import torch, time; "
             "a = torch.randn(512,512,dtype=torch.float16,device='cuda'); "
             "b = torch.randn(512,512,dtype=torch.float16,device='cuda'); "
             "torch.cuda.synchronize(); "
             "t0 = time.perf_counter(); "
             "c = torch.matmul(a, b); "
             "torch.cuda.synchronize(); "
             "print(round((time.perf_counter()-t0)*1000, 1))"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            ms = float(result.stdout.strip())
            return {"status": "PASS" if ms < 5000 else "WARN",
                    "latency_ms": ms,
                    "detail": f"fp16 matmul 512x512 = {ms}ms"}
        return {"status": "FAIL", "latency_ms": -1, "detail": result.stderr.strip()}
    except Exception as e:
        return {"status": "FAIL", "latency_ms": -1, "detail": str(e)}


def check_comfyui(url):
    """Check ComfyUI server health."""
    try:
        with urllib.request.urlopen(f"{url}/system_stats", timeout=5) as r:
            data = json.loads(r.read())
            return {"status": "PASS", "detail": str(data.get("system", {}).get("comfyui_version", "unknown"))}
    except Exception as e:
        return {"status": "FAIL", "detail": str(e)}


def main():
    args = parse_args()

    # Doctor mode: end-to-end machine diagnosis with actionable fixes. Exits
    # here (3 only for infra breakage, 0 otherwise - missing models never fail).
    if args.doctor:
        sys.exit(run_doctor(args.expected_gpu_arch, args.comfyui_url))

    # On a TheRock multi-arch box the GPU torch lives in the ComfyUI venv, not in
    # this process. The runner passes --python so the torch checks probe the
    # interpreter that ComfyUI actually uses; torch is the authoritative GPU
    # source, while rocminfo / /opt/rocm are informational (often absent).
    py = args.python or sys.executable

    checks = {}
    checks["pytorch_rocm"] = check_pytorch_rocm(py)
    hip = checks["pytorch_rocm"].get("hip_version")
    checks["rocminfo"]     = check_rocminfo(args.expected_gpu_arch)
    checks["rocm_version"] = check_rocm_version(args.expected_rocm, hip_version=hip)
    checks["vram"]         = check_vram(args.min_vram_mb, py)
    checks["matmul_fp16"]  = check_matmul(py)
    checks["comfyui"]      = check_comfyui(args.comfyui_url)

    # torch sees the real GPU -> cross-check the arch torch reports against the
    # expected arch (this REPLACES the rocminfo arch gate, which is unreliable on
    # multi-arch). A mismatch is a warning, not a hard fail.
    torch_arch = checks["pytorch_rocm"].get("gpu_arch", "unknown")
    if checks["pytorch_rocm"]["status"] == "PASS" and torch_arch not in ("", "unknown"):
        if torch_arch != args.expected_gpu_arch:
            checks["pytorch_rocm"]["detail"] += (
                f" | WARN: torch reports {torch_arch}, expected "
                f"{args.expected_gpu_arch}")

    # Critical = the things that truly block real tests: the GPU must be usable
    # via torch, and ComfyUI must be reachable. rocm_version / rocminfo are
    # informational (multi-arch installs frequently lack both).
    critical = ["pytorch_rocm", "comfyui"]
    warnings = ["rocminfo", "rocm_version", "vram", "matmul_fp16"]

    critical_fail = any(checks[k]["status"] == "FAIL" for k in critical)
    warn_only     = any(checks[k]["status"] in ("FAIL", "WARN") for k in warnings)

    if critical_fail:
        overall = "NOT_READY"
    elif warn_only:
        overall = "PARTIAL"
    else:
        overall = "READY"

    if args.strict and overall == "PARTIAL":
        overall = "NOT_READY"

    # Prefer the arch torch actually reports; fall back to rocminfo's.
    gpu_arch_detected = torch_arch if torch_arch not in ("", "unknown") \
        else checks["rocminfo"].get("detected", "unknown")

    report = {
        "timestamp":              datetime.now(timezone.utc).isoformat(),
        "overall":                overall,
        "rocm_version_detected":  checks["rocm_version"].get("detected", "unknown"),
        "rocm_version_expected":  args.expected_rocm,
        "gpu_arch_detected":      gpu_arch_detected,
        "gpu_arch_expected":      args.expected_gpu_arch,
        "gpu_vram_free_mb":       checks["vram"].get("free_mb", -1),
        "gpu_vram_total_mb":      checks["vram"].get("total_mb", -1),
        "matmul_latency_ms":      checks["matmul_fp16"].get("latency_ms", -1),
        "comfyui_url":            args.comfyui_url,
        "comfyui_reachable":      checks["comfyui"]["status"] == "PASS",
        "hostname":               socket.gethostname(),
        "os_platform":            platform.system().lower(),
        "checks":                 checks,
    }

    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    gha_output = os.environ.get("GITHUB_OUTPUT")
    if gha_output:
        with open(gha_output, "a") as f:
            f.write(f"ready={overall}\n")
            f.write(f"rocm_ver_detected={report['rocm_version_detected']}\n")
            f.write(f"gpu_arch_detected={report['gpu_arch_detected']}\n")
            f.write(f"gpu_uuid={report['hostname']}\n")

    exit_map = {"READY": 0, "PARTIAL": 2, "NOT_READY": 1}
    sys.exit(exit_map.get(overall, 1))


if __name__ == "__main__":
    main()
