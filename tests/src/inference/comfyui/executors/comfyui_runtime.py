#!/usr/bin/env python3
"""
Layer    : Runtime lifecycle (shared)
Purpose  : Single source of truth for the ComfyUI server lifecycle - discovery,
           bootstrap (git clone), dependency install, import verification,
           interpreter resolution, start / ensure-running / restart. Both
           scripts/_run.py and scripts/run_all_resilient.py import from here so
           there is exactly ONE cross-platform implementation (no Windows-only
           PowerShell launchers, no hard-coded paths, no duplicated logic).

Design rules enforced here:
  * NEVER install/replace torch / torchvision / torchaudio - the ROCm wheels
    (from AMD's TheRock multi-arch index) carry the GPU runtime and pip would
    clobber them with a CPU/CUDA build. We PIN the trio via a pip constraints
    file built from the versions already installed in the launch interpreter,
    then run ComfyUI's full requirements.txt under that constraint. Transitive
    deps (e.g. typing_extensions) resolve normally; the ROCm trio is untouched;
    and because torch is pinned the resolver cannot back-track for minutes.
  * Fail fast if the launch interpreter has no torch (printing AMD's install
    command) rather than launching ComfyUI and timing out for 300s.
  * Verify `import yaml, aiohttp` actually succeed before launching ComfyUI, so
    a broken install fails loudly instead of timing out on a dead server.
  * restart_comfyui() is cross-platform: it finds the process on the port and
    kills it (psutil if present, else netstat/taskkill on Windows or the /proc
    scan / fuser / lsof on Linux) and then relaunches via the resolved Python.
    It only uses a PowerShell launcher when os.name == "nt" AND the profile
    explicitly configures one.

All logging goes to protocol_runs/bootstrap.log (setup) and
protocol_runs/comfyui_server.log (the server's own stdout/stderr) unless
$COMFYUI_SERVER_LOG overrides the server log path (CI uses logs/comfyui_server.log).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "protocol_runs"
BOOTSTRAP_LOG = LOG_DIR / "bootstrap.log"
SERVER_LOG = LOG_DIR / "comfyui_server.log"
DEPS_MARKER = ".validation_deps_installed"

# PID of the most recently launched ComfyUI process (main.py or port listener).
_last_server_pid: int | None = None


@dataclass(frozen=True)
class LaunchResult:
    """Outcome of start/restart_comfyui."""
    ok: bool
    pid: int | None = None


def get_server_log():
    """Server stdout/stderr log path; honour $COMFYUI_SERVER_LOG in CI."""
    override = os.environ.get("COMFYUI_SERVER_LOG", "").strip()
    if override:
        return Path(override)
    return SERVER_LOG


def get_last_server_pid():
    """PID recorded at the last successful _launch_process / port discovery."""
    return _last_server_pid


def set_last_server_pid(pid):
    global _last_server_pid
    _last_server_pid = int(pid) if pid else None

# Minimal pinned core that ComfyUI needs to import + boot. Installing this first
# (and fast) avoids the common "ModuleNotFoundError: No module named 'yaml'"
# crash before we run the heavier requirements install.
CORE_DEPS = [
    "pyyaml>=6.0",
    "aiohttp>=3.9",
    "einops",
    "safetensors",
    "psutil",
]

# The torch trio carries the ROCm runtime on TheRock multi-arch wheels, so it
# must never be re-resolved from PyPI (that would pull a CPU/CUDA build and break
# the GPU). We do NOT strip these from requirements.txt; instead we PIN them via
# a pip constraints file built from whatever versions the launch interpreter
# already has. That lets `pip install -r requirements.txt` resolve transitive
# deps normally (e.g. typing_extensions) while pip is forbidden from touching
# the ROCm trio - and, because torch is pinned, the resolver cannot back-track
# for minutes on transformers either.
TORCH_PKGS = ("torch", "torchvision", "torchaudio")


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
def _log(msg, *, stream=None):
    """Print to console and append to bootstrap.log with a timestamp."""
    line = f"[runtime] {msg}"
    print(line, file=stream or sys.stdout, flush=True)
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with open(BOOTSTRAP_LOG, "a", encoding="utf-8") as f:
            f.write(f"{ts} {msg}\n")
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def port_from_url(url):
    """Extract the port from a comfyui_url; default 8188."""
    try:
        port = url.rstrip("/").rsplit(":", 1)[1].split("/")[0]
        return port if port.isdigit() else "8188"
    except Exception:
        return "8188"


def comfyui_up(url, timeout=10):
    """True if ComfyUI answers /system_stats with HTTP 200."""
    try:
        with urllib.request.urlopen(
                f"{url.rstrip('/')}/system_stats", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def wait_comfyui_up(url, *, timeout_s=120, poll_interval_s=5):
    """Poll /system_stats until reachable or timeout. Returns True when up."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if comfyui_up(url):
            return True
        time.sleep(max(1, poll_interval_s))
    return False


def load_machine_profile():
    """Load optional configs/machine_profile.json (gitignored on runners)."""
    prof_path = REPO_ROOT / "config" / "machine_profile.json"
    if not prof_path.exists():
        return {}
    try:
        return json.loads(prof_path.read_text(encoding="utf-8")) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def build_profile_from_env():
    """Merge machine_profile.json with workflow env overrides."""
    profile = load_machine_profile()
    if os.environ.get("COMFYUI_PATH"):
        profile = {**profile, "comfyui_path": os.environ["COMFYUI_PATH"]}
    if os.environ.get("COMFYUI_PYTHON"):
        profile = {**profile, "comfyui_venv": os.environ["COMFYUI_PYTHON"]}
    return profile


def find_listener_pid(port):
    """Return PID listening on TCP `port`, or None."""
    port = str(port)
    try:
        import psutil
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr and str(conn.laddr.port) == port and conn.pid:
                return int(conn.pid)
    except Exception:
        pass
    try:
        if os.name == "nt":
            out = subprocess.check_output(
                ["netstat", "-ano", "-p", "TCP"], text=True, timeout=15)
            for line in out.splitlines():
                if f":{port} " in line and "LISTENING" in line.upper():
                    return int(line.split()[-1])
        else:
            out = subprocess.check_output(
                ["bash", "-c", f"lsof -t -i:{port} -sTCP:LISTEN 2>/dev/null | head -1"],
                text=True, timeout=15)
            pid = out.strip().splitlines()[0] if out.strip() else ""
            if pid.isdigit():
                return int(pid)
    except Exception:
        pass
    return None


def pid_alive(pid):
    """True if live, False if dead, None if pid unknown/invalid."""
    if pid is None:
        return None
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:
        pass
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _expand(p):
    """Expand ~ and environment variables in a path-like string."""
    return Path(os.path.expandvars(os.path.expanduser(str(p))))


def _is_comfyui_dir(p):
    """A directory is a ComfyUI install if it has main.py at its root."""
    try:
        return p.is_dir() and (p / "main.py").is_file()
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# discovery / bootstrap target
# --------------------------------------------------------------------------- #
def discover_comfyui_path(profile):
    """Find a ComfyUI install without relying on a hard-coded path.

    Resolution order (first existing wins):
      1. $COMFYUI_PATH environment variable
      2. comfyui_path from the machine profile
      3. Common locations under the user's home / repo / system roots
      4. A shallow scan of the home dir and the repo's parent for ComfyUI*
    Works on any machine, user, or OS without editing configs first.
    """
    profile = profile or {}
    candidates = []
    for v in (os.environ.get("COMFYUI_PATH"), profile.get("comfyui_path")):
        if v:
            candidates.append(_expand(v))

    home = Path.home()
    candidates += [
        home / "ComfyUI",
        home / "comfyui",
        home / "Documents" / "ComfyUI",
        home / "Desktop" / "ComfyUI",
        home / "ComfyUI_windows_portable" / "ComfyUI",
        REPO_ROOT.parent / "ComfyUI",
        REPO_ROOT / "ComfyUI",
        Path("C:/ComfyUI"),
        Path("C:/ComfyUI_windows_portable/ComfyUI"),
        Path("/opt/ComfyUI"),
    ]
    for c in candidates:
        if _is_comfyui_dir(c):
            return c

    # Shallow scan: any top-level "comfyui*" folder under home or repo parent.
    for root in (home, REPO_ROOT.parent):
        try:
            for child in sorted(root.iterdir()):
                if child.is_dir() and "comfyui" in child.name.lower():
                    if _is_comfyui_dir(child):
                        return child
                    nested = child / "ComfyUI"
                    if _is_comfyui_dir(nested):
                        return nested
        except OSError:
            continue
    return None


def _bootstrap_target(profile):
    """Where to install ComfyUI when bootstrapping. Honour a configured
    comfyui_path / $COMFYUI_PATH if given, otherwise install beside this repo."""
    profile = profile or {}
    configured = profile.get("comfyui_path") or os.environ.get("COMFYUI_PATH")
    if configured:
        return _expand(configured)
    return REPO_ROOT.parent / "ComfyUI"


# --------------------------------------------------------------------------- #
# interpreter resolution
# --------------------------------------------------------------------------- #
def resolve_comfyui_python(profile, comfy_path):
    """Pick the best Python to launch / install ComfyUI with, in priority order:
      1. comfyui_venv from the profile (if it exists)
      2. a venv living next to the ComfyUI install (venv/.venv)
      3. the embedded python from a Windows portable build
      4. this validation venv's interpreter (already has torch/ROCm)
      5. profile python_bin, then a plain 'python'/'python3'
    """
    profile = profile or {}
    comfy_path = Path(comfy_path)

    def _venv_python(venv_dir):
        venv_dir = _expand(venv_dir)
        cand = (venv_dir / "Scripts" / "python.exe") if os.name == "nt" \
            else (venv_dir / "bin" / "python")
        return str(cand) if cand.exists() else None

    venv = profile.get("comfyui_venv")
    if venv:
        py = _venv_python(venv)
        if py:
            return py

    for sub in ("venv", ".venv"):
        py = _venv_python(comfy_path / sub)
        if py:
            return py

    portable = comfy_path.parent / "python_embeded" / \
        ("python.exe" if os.name == "nt" else "python")
    if portable.exists():
        return str(portable)

    # The validation venv already has a working torch/ROCm stack, so it is a
    # safe default that needs zero extra setup on the runner.
    if sys.executable:
        return sys.executable

    explicit = profile.get("python_bin")
    if explicit:
        return explicit
    return "python" if os.name == "nt" else "python3"


# --------------------------------------------------------------------------- #
# dependency install + verification
# --------------------------------------------------------------------------- #
def _pip(python, args, timeout=None):
    """Run `python -m pip ...`, streaming to bootstrap.log. Returns returncode."""
    cmd = [python, "-m", "pip", *args]
    _log(f"$ {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        _log(f"pip timed out after {timeout}s: {' '.join(args)}", stream=sys.stderr)
        return 124
    try:
        with open(BOOTSTRAP_LOG, "a", encoding="utf-8") as f:
            if proc.stdout:
                f.write(proc.stdout)
            if proc.stderr:
                f.write(proc.stderr)
    except OSError:
        pass
    return proc.returncode


def torch_versions(python):
    """Return {pkg: version} for the torch trio installed in `python`, e.g.
    {'torch': '2.12.0+rocm7.14.0a20260620', ...}. Missing packages are omitted.
    Used to build a pip constraints file so the ROCm wheels are pinned (never
    re-resolved from PyPI) while ComfyUI's other requirements install normally."""
    pkgs = ",".join(repr(p) for p in TORCH_PKGS)
    code = (
        "import importlib.metadata as m, json\n"
        "out={}\n"
        f"for p in ({pkgs},):\n"
        "    try: out[p]=m.version(p)\n"
        "    except Exception: pass\n"
        "print(json.dumps(out))"
    )
    try:
        proc = subprocess.run([python, "-c", code],
                              capture_output=True, text=True, timeout=60)
        if proc.returncode == 0 and proc.stdout.strip():
            import json
            return json.loads(proc.stdout.strip())
    except Exception:
        pass
    return {}


def _write_torch_constraints(comfy_path, versions):
    """Write a pip constraints file pinning the installed torch trio. Returns the
    path, or None if no torch is installed."""
    if not versions:
        return None
    path = Path(comfy_path) / ".validation_constraints.txt"
    path.write_text(
        "".join(f"{pkg}=={ver}\n" for pkg, ver in versions.items()),
        encoding="utf-8")
    return path


def install_comfyui_deps(comfy_path, python):
    """Install ComfyUI's runtime deps into the given interpreter WITHOUT touching
    the ROCm torch trio.

    Strategy (matches AMD's TheRock flow - constraints, not --no-deps):
      1. Detect the torch trio already installed in `python`. If torch is
         absent, fail fast with the AMD install command (launching ComfyUI from
         an interpreter that has no ROCm torch only times out for 300s).
      2. Install a small pinned CORE (pyyaml/aiohttp/einops/safetensors/psutil).
      3. Install ComfyUI's full requirements.txt with `-c <constraints>` pinning
         the torch trio. Transitive deps (e.g. typing_extensions) resolve
         normally, but pip is forbidden from replacing the ROCm wheels, and -
         because torch is pinned - it cannot back-track for minutes either.
    Writes a marker file on success so re-runs skip pip entirely.
    Returns True on success, False on any pip failure or missing torch.
    """
    comfy_path = Path(comfy_path)
    marker = comfy_path / DEPS_MARKER
    if marker.exists():
        _log(f"deps already installed (marker {marker.name} present); skipping pip.")
        return True

    versions = torch_versions(python)
    if "torch" not in versions:
        _log(f"interpreter {python} has no ROCm torch installed. ComfyUI needs "
             "the GPU torch trio. Install it (per AMD's TheRock flow), e.g.:",
             stream=sys.stderr)
        _log("  pip install --pre --index-url "
             "https://rocm.nightlies.amd.com/whl-multi-arch/ "
             "'torch[device-gfxXXXX]==<ver>+rocm<train>' "
             "'torchvision[device-gfxXXXX]==<ver>+rocm<train>' "
             "'torchaudio==<ver>+rocm<train>'", stream=sys.stderr)
        _log("  then set 'comfyui_venv' in configs/machine_profile.json to that "
             "venv so the suite launches ComfyUI with it.", stream=sys.stderr)
        return False
    _log(f"detected ROCm torch in launch interpreter: "
         f"{', '.join(f'{k}={v}' for k, v in versions.items())}")

    _log("installing ComfyUI core deps (torch pinned, untouched)...")
    rc = _pip(python, ["install", *CORE_DEPS], timeout=600)
    if rc != 0:
        _log(f"core dep install failed (pip exit {rc}); see {BOOTSTRAP_LOG}.",
             stream=sys.stderr)
        return False

    req = comfy_path / "requirements.txt"
    if req.exists():
        constraints = _write_torch_constraints(comfy_path, versions)
        _log("installing ComfyUI requirements.txt with torch pinned via "
             "constraints (transitive deps resolve normally)...")
        rc = _pip(python, ["install", "-c", str(constraints),
                           "-r", str(req)], timeout=1200)
        if rc != 0:
            _log(f"ComfyUI requirements install failed (pip exit {rc}); "
                 f"see {BOOTSTRAP_LOG}.", stream=sys.stderr)
            return False
    else:
        _log(f"no requirements.txt at {req}; core deps only.")

    ok, missing = verify_comfyui_imports(python)
    if not ok:
        _log(f"post-install import check failed; missing: {missing}",
             stream=sys.stderr)
        return False

    try:
        marker.write_text(
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") + "\n",
            encoding="utf-8")
    except OSError:
        pass
    _log("ComfyUI deps installed + verified.")
    return True


def verify_comfyui_imports(python, modules=("yaml", "aiohttp")):
    """Try to import the given modules in `python` via a subprocess.
    Returns (all_ok, list_of_missing_modules)."""
    missing = []
    for mod in modules:
        try:
            rc = subprocess.run(
                [python, "-c", f"import {mod}"],
                capture_output=True, text=True, timeout=60).returncode
        except (subprocess.TimeoutExpired, OSError):
            rc = 1
        if rc != 0:
            missing.append(mod)
    return (not missing), missing


# --------------------------------------------------------------------------- #
# bootstrap (git clone + deps)
# --------------------------------------------------------------------------- #
def bootstrap_comfyui(profile, *, force=False):
    """Clone ComfyUI and install its deps when no install is found.
    Returns the install Path on success, else None. Override the source with
    $COMFYUI_REPO. With force=True, clones even if a discovery would succeed
    (still no-op if the target dir already has main.py)."""
    profile = profile or {}
    target = _bootstrap_target(profile)
    if _is_comfyui_dir(target):
        _log(f"ComfyUI already present at {target}; ensuring deps.")
        py = resolve_comfyui_python(profile, target)
        install_comfyui_deps(target, py)
        return target

    if not force:
        existing = discover_comfyui_path(profile)
        if existing is not None:
            py = resolve_comfyui_python(profile, existing)
            install_comfyui_deps(existing, py)
            return existing

    if shutil.which("git") is None:
        _log("git not found on PATH; cannot auto-install ComfyUI. Install git "
             "or set comfyui_path to an existing install.", stream=sys.stderr)
        return None

    repo = os.environ.get("COMFYUI_REPO",
                          "https://github.com/comfyanonymous/ComfyUI.git")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _log(f"cannot create {target.parent}: {e}", stream=sys.stderr)
        return None

    _log(f"cloning ComfyUI ({repo}) -> {target}")
    rc = subprocess.run(["git", "clone", "--depth", "1", repo,
                         str(target)]).returncode
    if rc != 0 or not _is_comfyui_dir(target):
        _log("ComfyUI clone failed.", stream=sys.stderr)
        return None

    py = resolve_comfyui_python(profile, target)
    if not install_comfyui_deps(target, py):
        _log("ComfyUI cloned but dependency install failed; the server will "
             "likely not start. See bootstrap.log.", stream=sys.stderr)
        return None
    return target


# --------------------------------------------------------------------------- #
# start / ensure / restart
# --------------------------------------------------------------------------- #
def _launch_process(profile, comfy_path, url):
    """Spawn ComfyUI as a background process. Returns the Popen or None."""
    profile = profile or {}
    py = resolve_comfyui_python(profile, comfy_path)
    cmd = [py, "main.py", "--listen", "127.0.0.1", "--port", str(port_from_url(url))]
    cmd += list(profile.get("comfyui_extra_args", []))

    env = dict(os.environ)
    env.update({k: str(v) for k, v in profile.get("env", {}).items()})

    server_log = get_server_log()
    server_log.parent.mkdir(parents=True, exist_ok=True)
    _log(f"starting ComfyUI: {' '.join(cmd)} (cwd={comfy_path})")
    _log(f"server log -> {server_log}")
    try:
        logf = open(server_log, "a", encoding="utf-8")
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        logf.write(f"\n{'=' * 70}\n== ComfyUI launch {ts} :: {' '.join(cmd)}\n"
                   f"{'=' * 70}\n")
        logf.flush()
        proc = subprocess.Popen(cmd, cwd=str(comfy_path),
                                stdout=logf, stderr=logf, env=env)
        set_last_server_pid(proc.pid)
        return proc
    except OSError as e:
        _log(f"failed to spawn ComfyUI: {e}", stream=sys.stderr)
        set_last_server_pid(None)
        return None


def tail_file(path, max_lines=80):
    """Return the last `max_lines` lines of a text file as a string, or "" if the
    file is absent/unreadable. Stdlib-only; reads the whole file (server logs are
    small). Used to attach ComfyUI's server-side output to a failing test."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def start_comfyui(profile, url, *, allow_bootstrap=True, timeout_s=300,
                  poll_interval_s=5):
    """Locate (or bootstrap) ComfyUI, launch it, wait until /system_stats answers."""
    profile = profile or {}
    comfy_path = discover_comfyui_path(profile)
    if comfy_path is None and allow_bootstrap:
        _log("no ComfyUI install found; bootstrapping one...")
        comfy_path = bootstrap_comfyui(profile)
    if comfy_path is None:
        configured = profile.get("comfyui_path") or os.environ.get("COMFYUI_PATH")
        _log("could not locate or bootstrap a ComfyUI install.", stream=sys.stderr)
        if configured:
            _log(f"checked configured path: {_expand(configured)}", stream=sys.stderr)
        _log("set $COMFYUI_PATH or 'comfyui_path' in "
             "configs/machine_profile.json to your ComfyUI folder "
             "(the one containing main.py).", stream=sys.stderr)
        return LaunchResult(False, None)

    _log(f"using ComfyUI install at: {comfy_path}")

    py = resolve_comfyui_python(profile, comfy_path)
    ok, missing = verify_comfyui_imports(py)
    if not ok:
        _log(f"interpreter {py} is missing {missing}; installing ComfyUI deps...")
        if not install_comfyui_deps(comfy_path, py):
            _log("dependency install failed; ComfyUI will not start cleanly.",
                 stream=sys.stderr)
            return LaunchResult(False, None)

    if _launch_process(profile, comfy_path, url) is None:
        return LaunchResult(False, None)

    if wait_comfyui_up(url, timeout_s=timeout_s, poll_interval_s=poll_interval_s):
        port_pid = find_listener_pid(port_from_url(url))
        if port_pid:
            set_last_server_pid(port_pid)
        _log(f"ComfyUI is up (pid={get_last_server_pid()}).")
        return LaunchResult(True, get_last_server_pid())

    _log(f"ComfyUI did not become reachable within {timeout_s}s. "
         f"See {get_server_log()}.", stream=sys.stderr)
    return LaunchResult(False, get_last_server_pid())


def ensure_comfyui_running(profile, url, *, allow_bootstrap=True, timeout_s=300,
                           poll_interval_s=5):
    """Start ComfyUI if not reachable. Returns LaunchResult."""
    if comfyui_up(url):
        set_last_server_pid(find_listener_pid(port_from_url(url)))
        return LaunchResult(True, get_last_server_pid())
    return start_comfyui(profile, url, allow_bootstrap=allow_bootstrap,
                         timeout_s=timeout_s, poll_interval_s=poll_interval_s)


def _kill_process_on_port(port):
    """Cross-platform: kill whatever process is listening on `port`.
    Tries psutil first, then OS-specific fallbacks. Best-effort."""
    port = str(port)

    # 1. psutil (clean + cross-platform) if available.
    try:
        import psutil
        killed = False
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr and str(conn.laddr.port) == port and conn.pid:
                try:
                    psutil.Process(conn.pid).terminate()
                    killed = True
                    _log(f"terminated PID {conn.pid} on port {port} (psutil).")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        if killed:
            time.sleep(2)
            return True
    except Exception:
        pass

    # 2. OS-specific fallbacks.
    try:
        if os.name == "nt":
            out = subprocess.check_output(
                ["netstat", "-ano", "-p", "TCP"], text=True, timeout=15)
            pids = set()
            for line in out.splitlines():
                if f":{port} " in line and "LISTENING" in line.upper():
                    pids.add(line.split()[-1])
            for pid in pids:
                subprocess.run(["taskkill", "/F", "/PID", pid],
                               capture_output=True)
                _log(f"killed PID {pid} on port {port} (taskkill).")
            if pids:
                time.sleep(2)
                return True
        else:
            # Prefer fuser, then lsof; both are best-effort.
            for tool in (["fuser", "-k", f"{port}/tcp"],
                         ["bash", "-c",
                          f"kill $(lsof -t -i:{port}) 2>/dev/null"]):
                try:
                    subprocess.run(tool, capture_output=True, timeout=15)
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    continue
            time.sleep(2)
            return True
    except Exception as e:
        _log(f"could not kill process on port {port}: {e}", stream=sys.stderr)
    return False


def stop_comfyui(*, pid=None, url=None):
    """Terminate ComfyUI PID and kill anything listening on the ComfyUI port."""
    if pid is not None:
        if pid_alive(pid):
            try:
                import psutil
                psutil.Process(int(pid)).terminate()
                _log(f"terminated ComfyUI pid {pid}.")
            except Exception:
                try:
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                                       capture_output=True)
                    else:
                        os.kill(int(pid), 15)
                except OSError as e:
                    _log(f"could not terminate pid {pid}: {e}", stream=sys.stderr)
            time.sleep(2)
    if url:
        _kill_process_on_port(port_from_url(url))
    set_last_server_pid(None)
    return True


def restart_comfyui(profile, url, *, timeout_s=300, poll_interval_s=5):
    """Kill :port and start a fresh ComfyUI. Never reuse-if-healthy."""
    profile = profile or {}
    port = port_from_url(url)
    _log(f"restarting ComfyUI on port {port}...")
    _kill_process_on_port(port)
    set_last_server_pid(None)

    launcher = profile.get("comfyui_launcher")
    if os.name == "nt" and launcher and Path(_expand(launcher)).exists():
        _log(f"using configured Windows launcher: {launcher}")
        server_log = get_server_log()
        server_log.parent.mkdir(parents=True, exist_ok=True)
        try:
            logf = open(server_log, "a", encoding="utf-8")
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            logf.write(f"\n{'=' * 70}\n== ComfyUI launcher {ts} :: {launcher}\n"
                       f"{'=' * 70}\n")
            logf.flush()
            proc = subprocess.Popen(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                 "-File", str(_expand(launcher))],
                stdout=logf, stderr=logf)
            set_last_server_pid(proc.pid)
        except OSError as e:
            _log(f"launcher failed: {e}", stream=sys.stderr)
            return LaunchResult(False, None)
        if wait_comfyui_up(url, timeout_s=timeout_s, poll_interval_s=poll_interval_s):
            port_pid = find_listener_pid(port)
            if port_pid:
                set_last_server_pid(port_pid)
            _log(f"ComfyUI is up via launcher (pid={get_last_server_pid()}).")
            return LaunchResult(True, get_last_server_pid())
        return LaunchResult(False, get_last_server_pid())

    return start_comfyui(profile, url, allow_bootstrap=False, timeout_s=timeout_s,
                         poll_interval_s=poll_interval_s)


def write_gha_output(key, value):
    gha_output = os.environ.get("GITHUB_OUTPUT")
    if gha_output:
        with open(gha_output, "a", encoding="utf-8") as f:
            f.write(f"{key}={value}\n")


def _cli_restart(args):
    profile = build_profile_from_env()
    if args.server_log:
        os.environ["COMFYUI_SERVER_LOG"] = args.server_log
    result = restart_comfyui(
        profile, args.url,
        timeout_s=args.health_timeout,
        poll_interval_s=args.poll_interval,
    )
    if result.ok and result.pid:
        write_gha_output("comfyui_pid", str(result.pid))
        print(f"COMFYUI_SERVER_PID={result.pid}")
    if not result.ok:
        print("[INFRA_ERROR] server_start_failed", file=sys.stderr)
        write_gha_output("result", "INFRA_ERROR")
        write_gha_output("infra_error", "server_start_failed")
        return 3
    return 0


def _cli_stop(args):
    pid = args.pid or os.environ.get("COMFYUI_SERVER_PID")
    stop_comfyui(pid=pid, url=args.url)
    return 0


def _cli_main():
    p = argparse.ArgumentParser(description="ComfyUI server lifecycle (CI + local)")
    p.add_argument("--url", default=os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188"))
    p.add_argument("--server-log", default=os.environ.get("COMFYUI_SERVER_LOG", ""))
    p.add_argument("--health-timeout", type=int, default=int(
        os.environ.get("COMFYUI_HEALTH_TIMEOUT_S", "120")))
    p.add_argument("--poll-interval", type=int, default=int(
        os.environ.get("COMFYUI_HEALTH_POLL_S", "5")))
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("restart", help="kill :port and start fresh ComfyUI")
    stop_p = sub.add_parser("stop", help="stop ComfyUI pid and/or port listener")
    stop_p.add_argument("--pid", default=None)

    args = p.parse_args()
    if args.command == "restart":
        return _cli_restart(args)
    if args.command == "stop":
        return _cli_stop(args)
    return 1


if __name__ == "__main__":
    sys.exit(_cli_main())
