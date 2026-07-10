#!/usr/bin/env python3
"""
Layer    : Execution bridge
Equiv    : rocm_test_executor.py (production)
Purpose  : Wraps single_test_protocol.py with the production CLI contract so
           GHA steps can call this identically to the prod executor.

Exit codes - IMMUTABLE, do not change:
  0  PASS
  1  FAIL
  2  SKIP        (disabled, os mismatch, skip-cache hit)
  3  INFRA_ERROR (ComfyUI unreachable, GPU not visible, manifest missing)
"""

import argparse, importlib.util, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

# framework-ci-comfy layout: all comfyui executors live together under
# tests/src/inference/comfyui/executors/. SUITE_DIR is the suite root; sibling
# modules (comfyui_runtime, model_check) import from this same executors dir.
EXECUTORS_DIR = Path(__file__).resolve().parent
SUITE_DIR = EXECUTORS_DIR.parent
REPO_ROOT = SUITE_DIR
sys.path.insert(0, str(EXECUTORS_DIR))


def parse_args():
    p = argparse.ArgumentParser(description="ComfyUI AMD Validator")
    p.add_argument("--execute",          required=True,  help="test_name from suite_manifest.json")
    p.add_argument("--execution_label",  required=True,  help="Run label for traceability")
    p.add_argument("--rocm_version",     required=True,  help="ROCm version, e.g. 7.14.0")
    p.add_argument("--gpu_arch",         required=True,  help="GPU arch, e.g. gfx1100")
    p.add_argument("--os_version",       required=True,  help="OS string, e.g. linux-ubuntu-24.04")
    p.add_argument("--manifest",         default="tests/src/inference/comfyui/config/suite_manifest.json")
    p.add_argument("--output_dir",       default="protocol_runs")
    p.add_argument("--comfyui_url",      default="http://127.0.0.1:8188")
    p.add_argument("--comfyui_path",     default=None,
                   help="ComfyUI install dir (the one containing models/). "
                        "Auto-discovered if omitted; used for the model-presence "
                        "SKIP check.")
    p.add_argument("--dry_run",          action="store_true")
    p.add_argument("--skip_warmup",      action="store_true")
    p.add_argument("--max_timed_runs",   type=int, default=0,
                   help="Cap timed runs (0 = use manifest perf_targets value). "
                        "Useful on slow hardware to avoid multi-pass timeouts.")
    p.add_argument("--comfyui_pid",      default=None,
                   help="ComfyUI server PID from CI lifecycle (for crash detection)")
    return p.parse_args()


def write_gha_output(key, value):
    """Write key=value to $GITHUB_OUTPUT if running inside GHA."""
    gha_output = os.environ.get("GITHUB_OUTPUT")
    if gha_output:
        with open(gha_output, "a") as f:
            f.write(f"{key}={value}\n")


def load_manifest(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        print(f"[INFRA_ERROR] Manifest not found: {path}", file=sys.stderr)
        sys.exit(3)
    except json.JSONDecodeError as e:
        print(f"[INFRA_ERROR] Manifest JSON invalid: {e}", file=sys.stderr)
        sys.exit(3)


def find_test(manifest, test_name):
    for t in manifest.get("tests", []):
        if t["test_name"] == test_name:
            return t
    return None


def os_family(os_version_string):
    """Extract 'linux' or 'windows' from os_version string."""
    s = os_version_string.lower()
    if "windows" in s:
        return "windows"
    return "linux"


def effective_timeout_minutes(test, gpu_arch):
    """Per-arch timeout, falling back to the base timeout_minutes.

    Slow architectures (e.g. APU/iGPU like gfx1153, where heavy diffusion models
    run heavily offloaded at minutes-per-step) can need far more time than a fast
    discrete GPU. An optional manifest field lets a test carry per-arch overrides
    without inflating the timeout ceiling for fast hardware:

        "timeout_minutes": 25,
        "timeout_minutes_by_arch": {"gfx1153": 60}
    """
    by_arch = test.get("timeout_minutes_by_arch") or {}
    try:
        return int(by_arch.get(gpu_arch, test["timeout_minutes"]))
    except (TypeError, ValueError):
        return int(test["timeout_minutes"])


def resolve_workflow_path(workflow_rel: str) -> Path:
    """Resolve manifest workflow paths (workflows/foo.json) against the suite root."""
    p = Path(workflow_rel)
    if p.is_file():
        return p.resolve()
    candidate = SUITE_DIR / workflow_rel
    if candidate.is_file():
        return candidate.resolve()
    return p


def resolve_comfyui_path(explicit):
    """Resolve the ComfyUI install dir for the model-presence check. Honour an
    explicit --comfyui_path / $COMFYUI_PATH, else use the shared runtime's
    discovery against the machine profile. Returns a path string or None."""
    if explicit:
        return explicit
    if os.environ.get("COMFYUI_PATH"):
        return os.environ["COMFYUI_PATH"]
    try:
        import comfyui_runtime
        profile = {}
        prof_path = SUITE_DIR / "config" / "machine_profile.json"
        if prof_path.exists():
            try:
                profile = json.loads(prof_path.read_text())
            except Exception:
                profile = {}
        found = comfyui_runtime.discover_comfyui_path(profile)
        return str(found) if found else None
    except Exception:
        return None


def _fleet_hardware_module():
    """Load tests/utils/system/hardware.py without importing tests package."""
    for parent in Path(__file__).resolve().parents:
        hw_path = parent / "tests" / "utils" / "system" / "hardware.py"
        if hw_path.is_file():
            spec = importlib.util.spec_from_file_location("hardware", str(hw_path))
            if spec is None or spec.loader is None:
                return None
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None


def detect_gpu_arch():
    """Best-effort detection of the ACTUAL hardware arch (gfxNNNN) this run
    executed on, independent of the requested --gpu_arch selection.

    Uses fleet hardware.py first (/opt/rocm*/bin/rocminfo, bm_config, device_id,
    amd-smi) so CI matches install-job host capture. Falls back to torch, then
    rocminfo on PATH. Returns "" if undetectable."""
    try:
        hw = _fleet_hardware_module()
        if hw is not None:
            arch = (hw.detect_gpu_arch() or "").strip().lower()
            if arch and arch != "unknown":
                return arch
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_properties(0).gcnArchName
            return name.split(":")[0].strip()
    except Exception:
        pass
    try:
        import re as _re
        out = subprocess.check_output(["rocminfo"], text=True, timeout=20)
        m = _re.search(r"Name:\s+(gfx\w+)", out)
        if m:
            return m.group(1)
    except Exception:
        pass
    return ""


def check_output_writable(comfy_path):
    """Preflight: verify ComfyUI's output dir is writable. An unwritable output
    dir (EACCES) or a full disk (ENOSPC) is an INFRA problem - it would surface
    later as a misleading 'no media' FAIL on every test. Returns (ok, detail):
    ok False means infrastructure breakage. If the path can't be resolved we
    return ok True (nothing to assert)."""
    if not comfy_path:
        return True, ""
    out_dir = Path(comfy_path) / "output"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        probe = out_dir / ".validation_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True, ""
    except OSError as e:
        return False, f"{out_dir} not writable: {e.__class__.__name__}: {e}"




def detect_vram_mb(python=None):
    """Return (total_mb, free_mb) via torch.cuda.mem_get_info, or (0, 0) on failure."""
    python = python or sys.executable
    try:
        result = subprocess.run(
            [python, "-c",
             "import torch; "
             "free, total = torch.cuda.mem_get_info(0); "
             "print(free // 1024**2, total // 1024**2)"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split()
            if len(parts) >= 2:
                return int(parts[1]), int(parts[0])
    except Exception:
        pass
    return 0, 0


def check_capability_gate(test, gpu_arch_detected):
    """Return (should_skip, skip_reason, details) or (False, None, {})."""
    details = {}
    arch = (gpu_arch_detected or "").lower()
    allow = test.get("gpu_arch_allow") or []
    deny = test.get("gpu_arch_deny") or []
    if allow and arch and arch not in [a.lower() for a in allow]:
        return True, "gpu_arch_unsupported", {
            "required_arch": ",".join(allow), "detected_arch": arch,
        }
    if deny and arch in [d.lower() for d in deny]:
        return True, "gpu_arch_unsupported", {
            "denied_arch": arch, "detected_arch": arch,
        }

    required = test.get("min_vram_mb")
    if required:
        total_mb, free_mb = detect_vram_mb()
        detected = total_mb or free_mb
        if detected and detected < int(required):
            return True, "insufficient_vram", {
                "required_mb": int(required),
                "detected_mb": detected,
                "detected_gb": round(detected / 1024, 2),
                "required_gb": round(int(required) / 1024, 2),
            }
    return False, None, {}


def format_skip_failure_reason(skip_reason, details):
    parts = [f"SKIP {skip_reason}"]
    for k, val in details.items():
        parts.append(f"{k}={val}")
    return ": ".join(parts)

def capture_server_tail(evidence_dir, max_lines=80, always_write=False):
    """Copy the tail of the shared ComfyUI server log into this test's evidence
    dir as comfyui_server_tail.log, so a failure carries the server-side output
    (GPU/HIP crash, OOM, node/save traceback) that caused it. Returns the tail
    text (may be ""). Best-effort - never raises.

    always_write=True writes the file even when the server log is absent/empty
    (with a short note explaining why), so for EVERY non-PASS outcome the
    evidence dir deterministically contains a comfyui_server_tail.log - including
    the common host-native/container case where ComfyUI was started outside this
    framework and no shared server log exists."""
    try:
        import comfyui_runtime
        server_log = comfyui_runtime.get_server_log()
        tail = comfyui_runtime.tail_file(server_log, max_lines)
    except Exception:
        server_log, tail = None, ""
    dest = Path(evidence_dir) / "comfyui_server_tail.log"
    if tail:
        try:
            dest.write_text(tail, encoding="utf-8")
        except OSError:
            pass
    elif always_write:
        note = (f"[no ComfyUI server-log tail available]\n"
                f"Expected at: {server_log}\n"
                f"This is normal when ComfyUI was started OUTSIDE this framework "
                f"(host-native / container / manual launch): its stdout/stderr\n"
                f"is not captured here. The failure's root cause, if any, is in\n"
                f"results.json (ComfyUI /history execution errors) and "
                f"protocol_stderr.log.\n")
        try:
            dest.write_text(note, encoding="utf-8")
        except OSError:
            pass
    return tail


def check_comfyui(url):
    import urllib.request
    try:
        with urllib.request.urlopen(f"{url}/system_stats", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def main():
    args = parse_args()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    manifest = load_manifest(args.manifest)
    test = find_test(manifest, args.execute)

    if test is None:
        print(f"[INFRA_ERROR] Test '{args.execute}' not found in manifest.", file=sys.stderr)
        write_gha_output("result", "INFRA_ERROR")
        sys.exit(3)

    # Skip checks
    if not test.get("enabled", True):
        print(f"[SKIP] {args.execute} is disabled in manifest.")
        write_gha_output("result", "SKIP")
        write_gha_output("skip_reason", "disabled")
        sys.exit(2)

    os_fam = os_family(args.os_version)
    if os_fam not in test.get("os", []):
        print(f"[SKIP] {args.execute} does not support OS family '{os_fam}'.")
        write_gha_output("result", "SKIP")
        write_gha_output("skip_reason", f"os_mismatch:{os_fam}")
        sys.exit(2)

    # Dry run: validate paths, no GPU calls
    if args.dry_run:
        workflow_path = resolve_workflow_path(test["workflow"])
        if not workflow_path.exists():
            print(f"[DRY_RUN] WARNING: workflow file not found: {workflow_path}")
        else:
            print(f"[DRY_RUN] Workflow found: {workflow_path}")
        print(f"[DRY_RUN] OK - {args.execute} ({args.gpu_arch} / {args.os_version})")
        write_gha_output("result", "DRY_RUN_OK")
        sys.exit(0)

    detected_arch = detect_gpu_arch()
    should_skip, skip_reason, skip_details = check_capability_gate(test, detected_arch)
    if should_skip:
        failure_reason = format_skip_failure_reason(skip_reason, skip_details)
        evidence_dir = Path(args.output_dir) / f"{args.execute}_{ts}"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        print(f"[SKIP] {args.execute} - {failure_reason}")
        (evidence_dir / "summary.json").write_text(json.dumps({
            "test_name": args.execute,
            "tms_key": test.get("tms_key"),
            "execution_label": args.execution_label,
            "rocm_version": args.rocm_version,
            "gpu_arch": args.gpu_arch,
            "detected_gpu_arch": detected_arch,
            "os_version": args.os_version,
            "verdict": "SKIP",
            "skip_reason": skip_reason,
            "failure_reason": failure_reason,
            "duration_s": 0,
            "timestamp": ts,
            "evidence_dir": str(evidence_dir),
        }, indent=2), encoding="utf-8")
        write_gha_output("result", "SKIP")
        write_gha_output("skip_reason", skip_reason)
        sys.exit(2)

    # Check ComfyUI reachable
    if not check_comfyui(args.comfyui_url):
        print(f"[INFRA_ERROR] ComfyUI not reachable at {args.comfyui_url}", file=sys.stderr)
        write_gha_output("result", "INFRA_ERROR")
        write_gha_output("infra_error", f"comfyui_unreachable:{args.comfyui_url}")
        sys.exit(3)

    # Model-presence gate: auto-download missing weights when configured, then
    # re-check. Still missing after fetch -> FAIL (content_fetch_failed).
    comfy_path = resolve_comfyui_path(args.comfyui_path)
    workflow_file = str(resolve_workflow_path(test["workflow"]))
    if comfy_path:
        import fetch_models
        try:
            import model_check
            missing = model_check.missing_models(workflow_file, comfy_path)
        except Exception:
            missing = []
        if missing and fetch_models.auto_fetch_enabled():
            print(f"[fetch] {len(missing)} model(s) missing — starting auto-download")
            fr = fetch_models.ensure_missing(missing, comfy_path)
            for err in fr.errors:
                print(f"[fetch] {err}")
            try:
                missing = model_check.missing_models(workflow_file, comfy_path)
            except Exception:
                missing = list(missing)
        if missing:
            msg = model_check.skip_message(missing)
            if fetch_models.auto_fetch_enabled():
                print(f"[FAIL] {args.execute} - still missing after auto-fetch: {msg}",
                      file=sys.stderr)
                write_gha_output("result", "FAIL")
                write_gha_output("failure_reason", "content_fetch_failed")
                sys.exit(1)
            print(f"[SKIP] {args.execute} - {msg}")
            write_gha_output("result", "SKIP")
            write_gha_output("skip_reason", "model_missing")
            sys.exit(2)
        # Identity check: wrong file on disk — try re-fetch when auto-fetch on.
        try:
            mismatches = model_check.identity_mismatches(workflow_file, comfy_path)
        except Exception:
            mismatches = []
        if mismatches and fetch_models.auto_fetch_enabled():
            refs = model_check.referenced_models(workflow_file)
            by_name = {fn: subs for fn, subs in refs}
            to_refetch = [(fn, by_name[fn]) for fn in by_name
                          if any(fn in m for m in mismatches)]
            for fn, subs in to_refetch:
                fetch_models.download_one(fn, subs, comfy_path,
                                          fetch_models.load_manifest(), force=True)
            try:
                mismatches = model_check.identity_mismatches(workflow_file, comfy_path)
            except Exception:
                pass
        if mismatches:
            detail = "; ".join(mismatches)
            if fetch_models.auto_fetch_enabled():
                print(f"[FAIL] {args.execute} - model identity mismatch after "
                      f"re-fetch: {detail}", file=sys.stderr)
                write_gha_output("result", "FAIL")
                write_gha_output("failure_reason", "model_identity_mismatch")
                sys.exit(1)
            print(f"[SKIP] {args.execute} - model identity mismatch: {detail}")
            write_gha_output("result", "SKIP")
            write_gha_output("skip_reason", "model_identity_mismatch")
            sys.exit(2)

    # Output-dir writability preflight: an unwritable ComfyUI output dir is an
    # environment problem (INFRA_ERROR), not a model FAIL. Catch it up-front with
    # a clear FIX instead of letting every test fail later with "no media".
    if comfy_path:
        ok, detail = check_output_writable(comfy_path)
        if not ok:
            print(f"[INFRA_ERROR] {args.execute} - {detail}", file=sys.stderr)
            print(f"        FIX: make the ComfyUI output dir writable, e.g. "
                  f"chmod u+rwx '{Path(comfy_path) / 'output'}' (or free disk "
                  f"space / fix ownership).", file=sys.stderr)
            write_gha_output("result", "INFRA_ERROR")
            write_gha_output("infra_error", "output_dir_not_writable")
            sys.exit(3)

    # Prepare evidence directory
    evidence_dir = Path(args.output_dir) / f"{args.execute}_{ts}"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # Invoke single_test_protocol.py
    timed_runs = test["perf_targets"]["timed_runs"]
    if args.max_timed_runs and args.max_timed_runs > 0:
        timed_runs = min(timed_runs, args.max_timed_runs)
    timeout_min = effective_timeout_minutes(test, args.gpu_arch)
    cmd = [
        sys.executable,
        str(EXECUTORS_DIR / "single_test_protocol.py"),
        "--test_name",      args.execute,
        "--workflow",       workflow_file,
        "--comfyui_url",    args.comfyui_url,
        "--output_dir",     str(evidence_dir),
        "--warmup_runs",    str(0 if args.skip_warmup else test["perf_targets"]["warmup_runs"]),
        "--timed_runs",     str(timed_runs),
        "--timeout_s",      str(timeout_min * 60),
        "--expected_type",  test["expected_output_type"],
    ]
    # Perf gate: forward the manifest's max_latency_s so a too-slow (but
    # otherwise valid) run is FAILed by single_test_protocol.py. Absent field =
    # no gate. A slow pass is a FAIL, never a SKIP.
    max_latency_s = test.get("perf_targets", {}).get("max_latency_s")
    if max_latency_s is not None:
        cmd += ["--max_latency_s", str(max_latency_s)]

    start = datetime.now(timezone.utc)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_min * 60 + 30)
    except subprocess.TimeoutExpired:
        import comfyui_runtime
        dur = round((datetime.now(timezone.utc) - start).total_seconds(), 1)
        pid = args.comfyui_pid or os.environ.get("COMFYUI_SERVER_PID")
        alive = comfyui_runtime.pid_alive(pid) if pid else None
        tail = capture_server_tail(evidence_dir, max_lines=100, always_write=True)
        last = [ln for ln in tail.splitlines() if ln.strip()]
        if alive is False:
            verdict = "INFRA_ERROR"
            reason = f"server_died_mid_test after {timeout_min} min"
            fail_key, fail_val = "infra_error", "server_died_mid_test"
            exit_code = 3
        else:
            verdict = "FAIL"
            reason = f"timeout after {timeout_min} min (inference too slow)"
            fail_key, fail_val = "fail_reason", "timeout"
            exit_code = 1
        if last:
            reason += f" | comfyui_server last line: {last[-1][:200]}"
        print(f"[{verdict}] {args.execute} timed out after {timeout_min} min "
              f"-- reason: {reason}", file=sys.stderr)
        (evidence_dir / "summary.json").write_text(json.dumps({
            "test_name": args.execute, "tms_key": test["tms_key"],
            "execution_label": args.execution_label, "rocm_version": args.rocm_version,
            "gpu_arch": args.gpu_arch, "detected_gpu_arch": detect_gpu_arch(),
            "os_version": args.os_version,
            "verdict": verdict, "failure_reason": reason, "io_error": False,
            "server_log_tail": "comfyui_server_tail.log", "duration_s": dur,
            "timestamp": ts, "evidence_dir": str(evidence_dir),
        }, indent=2))
        write_gha_output("result", verdict)
        write_gha_output(fail_key, fail_val)
        sys.exit(exit_code)

    duration_s = round((datetime.now(timezone.utc) - start).total_seconds(), 1)

    # Write stdout/stderr to evidence
    (evidence_dir / "protocol_stdout.log").write_text(proc.stdout)
    (evidence_dir / "protocol_stderr.log").write_text(proc.stderr)

    # Parse result
    results_file = evidence_dir / "results.json"
    failure_reason = ""
    io_error = False
    if proc.returncode == 0 and results_file.exists():
        results = json.loads(results_file.read_text())
        verdict = results.get("verdict", "FAIL")
        failure_reason = results.get("failure_reason", "")
        io_error = bool(results.get("io_error"))
    else:
        verdict = "FAIL"
        failure_reason = (f"protocol did not produce results.json "
                          f"(exit {proc.returncode}); see protocol_stderr.log")

    # Reclassify filesystem/IO save failures as INFRA_ERROR (exit 3): an
    # unwritable output dir / full disk is an environment problem, not a model
    # regression. This must not fake-green - it stays a non-PASS, just a more
    # honest category than FAIL.
    if verdict == "FAIL" and io_error:
        verdict = "INFRA_ERROR"
        print(f"        FIX: filesystem/IO error on the ComfyUI output path - "
              f"fix permissions/space on the output dir, then re-run.",
              file=sys.stderr)

    # On ANY non-PASS outcome, snapshot the ComfyUI server-log tail into this
    # test's evidence dir. The root cause of a failure can live in the server's
    # own output (GPU/HIP crash, OOM, node/save traceback); preserve it per-test
    # so it isn't lost to the next relaunch or buried in the shared log. Always
    # write the file (even if empty/absent, note that) so its presence is
    # deterministic for every failure path, not just hard crashes.
    if verdict != "PASS":
        server_tail = capture_server_tail(evidence_dir, always_write=True)
        if server_tail:
            last = [ln for ln in server_tail.splitlines() if ln.strip()]
            if last:
                failure_reason += f" | comfyui_server last line: {last[-1][:200]}"

    # Write summary record
    summary = {
        "test_name":       args.execute,
        "tms_key":         test["tms_key"],
        "execution_label": args.execution_label,
        "rocm_version":    args.rocm_version,
        "gpu_arch":        args.gpu_arch,
        "detected_gpu_arch": detect_gpu_arch(),
        "os_version":      args.os_version,
        "verdict":         verdict,
        "failure_reason":  failure_reason,
        "io_error":        io_error,
        "server_log_tail": "comfyui_server_tail.log",
        "duration_s":      duration_s,
        "timestamp":       ts,
        "evidence_dir":    str(evidence_dir),
    }
    (evidence_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # GHA outputs
    write_gha_output("result",      verdict)
    write_gha_output("duration_s",  str(duration_s))
    write_gha_output("tms_key",     test["tms_key"])
    write_gha_output("evidence_dir", str(evidence_dir))

    print(f"[{verdict}] {args.execute} - {duration_s}s")
    exit_by_verdict = {"PASS": 0, "FAIL": 1, "SKIP": 2, "INFRA_ERROR": 3}
    sys.exit(exit_by_verdict.get(verdict, 1))


if __name__ == "__main__":
    main()
