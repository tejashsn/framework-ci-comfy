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
protocol_runs/comfyui_server.log (the server's own stdout/stderr).
"""

import os
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "protocol_runs"
BOOTSTRAP_LOG = LOG_DIR / "bootstrap.log"
SERVER_LOG = LOG_DIR / "comfyui_server.log"
DEPS_MARKER = ".validation_deps_installed"

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
        Path("C:/TheRock/ComfyUI"),
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
    comfyui_path / $COMFYUI_PATH if given, otherwise use a sensible default."""
    profile = profile or {}
    configured = profile.get("comfyui_path") or os.environ.get("COMFYUI_PATH")
    if configured:
        return _expand(configured)
    if os.name == "nt" and Path("C:/TheRock").is_dir():
        return Path("C:/TheRock/ComfyUI")
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

    env_py = os.environ.get("COMFYUI_PYTHON", "").strip()
    if env_py and Path(env_py).exists():
        return env_py

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

    # Free the port before spawning. ComfyUI runs as parent+child; if a previous
    # instance's parent exited but its child still owns the port, a fresh launch
    # would fail with "port already in use" and never become reachable. Killing
    # whatever holds the port first makes every relaunch reliable.
    _kill_process_on_port(port_from_url(url))

    SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    _log(f"starting ComfyUI: {' '.join(cmd)} (cwd={comfy_path})")
    _log(f"server log -> {SERVER_LOG}")
    try:
        # APPEND (not truncate): during a self-healing sweep ComfyUI is relaunched
        # after a crash, and truncating here would wipe the crash traceback of the
        # server session that just died - exactly the evidence we need. A dated
        # separator marks each launch so sessions stay distinguishable.
        logf = open(SERVER_LOG, "a", encoding="utf-8")
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        logf.write(f"\n{'=' * 70}\n== ComfyUI launch {ts} :: {' '.join(cmd)}\n"
                   f"{'=' * 70}\n")
        logf.flush()
        return subprocess.Popen(cmd, cwd=str(comfy_path),
                                stdout=logf, stderr=logf, env=env)
    except OSError as e:
        _log(f"failed to spawn ComfyUI: {e}", stream=sys.stderr)
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


def start_comfyui(profile, url, *, allow_bootstrap=True, timeout_s=300):
    """Locate (or bootstrap) ComfyUI and launch it, waiting until reachable.
    Returns True once /system_stats answers, else False."""
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
        return False

    _log(f"using ComfyUI install at: {comfy_path}")

    # Make sure deps are present even when we discovered an existing checkout
    # that was never set up for our venv (the marker keeps this cheap on reruns).
    py = resolve_comfyui_python(profile, comfy_path)
    ok, missing = verify_comfyui_imports(py)
    if not ok:
        _log(f"interpreter {py} is missing {missing}; installing ComfyUI deps...")
        if not install_comfyui_deps(comfy_path, py):
            _log("dependency install failed; ComfyUI will not start cleanly.",
                 stream=sys.stderr)
            return False

    if _launch_process(profile, comfy_path, url) is None:
        return False

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if comfyui_up(url):
            _log("ComfyUI is up.")
            return True
        time.sleep(3)
    _log(f"ComfyUI did not become reachable within {timeout_s}s. See {SERVER_LOG}.",
         stream=sys.stderr)
    return False


def ensure_comfyui_running(profile, url, *, allow_bootstrap=True, timeout_s=300):
    """If ComfyUI is already reachable, return True immediately; otherwise try
    to start it (bootstrapping if allowed). The one entry point callers should
    use before running tests."""
    if comfyui_up(url):
        return True
    return start_comfyui(profile, url, allow_bootstrap=allow_bootstrap,
                         timeout_s=timeout_s)


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


def restart_comfyui(profile, url, *, timeout_s=300):
    """Kill any ComfyUI on the port and start a fresh one. Cross-platform.

    Only uses a PowerShell launcher when os.name == 'nt' AND the profile
    explicitly sets 'comfyui_launcher' (back-compat for a pinned-memory launcher
    script). Otherwise it always uses resolve_comfyui_python() + main.py.
    """
    profile = profile or {}
    port = port_from_url(url)
    _log(f"restarting ComfyUI on port {port}...")
    _kill_process_on_port(port)

    launcher = profile.get("comfyui_launcher")
    if os.name == "nt" and launcher and Path(_expand(launcher)).exists():
        _log(f"using configured Windows launcher: {launcher}")
        try:
            subprocess.Popen(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                 "-File", str(_expand(launcher))],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as e:
            _log(f"launcher failed: {e}", stream=sys.stderr)
            return False
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if comfyui_up(url):
                _log("ComfyUI is up (launcher).")
                time.sleep(3)
                return True
            time.sleep(3)
        return False

    return start_comfyui(profile, url, allow_bootstrap=False, timeout_s=timeout_s)
