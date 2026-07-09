"""Hardware detection for CPU and GPU using system commands and ROCm tools."""

import subprocess
import re
import os
import glob
import shutil
import logging
import json
from typing import List, Optional, Dict
from dataclasses import dataclass, field


@dataclass
class CpuInfo:
    """CPU information dataclass with model, cores, cache, and clock details."""

    model_name: str = "Unknown"
    cores: int = 0
    sockets: int = 1
    ram_size_gb: int = 0
    numa_nodes: int = 1
    clock_speed_mhz: int = 0
    l1_cache_kb: int = 0
    l2_cache_kb: int = 0
    l3_cache_kb: int = 0

    def getCpuModelName(self) -> str:
        """Get CPU model name."""
        return self.model_name

    def getCpuCores(self) -> int:
        """Get CPU cores count."""
        return self.cores

    def getCpuSockets(self) -> int:
        """Get CPU sockets count."""
        return self.sockets

    def getCpuRamSize(self) -> int:
        """Get RAM size in GB."""
        return self.ram_size_gb

    def getCpuNumaNodes(self) -> int:
        """Get NUMA nodes count."""
        return self.numa_nodes

    def getCpuClockSpeed(self) -> int:
        """Get CPU clock speed in MHz."""
        return self.clock_speed_mhz

    def getCpuL1Cache(self) -> int:
        """Get L1 cache size in KB."""
        return self.l1_cache_kb

    def getCpuL2Cache(self) -> int:
        """Get L2 cache size in KB."""
        return self.l2_cache_kb

    def getCpuL3Cache(self) -> int:
        """Get L3 cache size in KB."""
        return self.l3_cache_kb

    def __str__(self):
        return f"{self.model_name} ({self.cores} cores, {self.sockets} sockets, {self.ram_size_gb}GB RAM)"


@dataclass
class GpuInfo:
    """GPU information."""

    device_id: str = ""
    revision_id: str = ""
    product_name: str = "Unknown"
    marketing_name: str = "Unknown"  # e.g., "AMD Instinct MI300X"
    vendor: str = "AMD"
    vram_size_gb: int = 0
    sys_clock_mhz: int = 0
    mem_clock_mhz: int = 0
    pci_address: str = ""
    vbios: str = "Unknown"
    partition_mode: str = "Unknown"
    xgmi_type: str = "Unknown"
    host_driver: str = "Unknown"
    firmwares: List[Dict[str, str]] = field(default_factory=list)

    def __str__(self):
        name = self.marketing_name if self.marketing_name != "Unknown" else self.product_name
        return f"{name} (Device ID: {self.device_id}, VRAM: {self.vram_size_gb}GB)"


def _get_rocm_tool_path(tool_name: str) -> str:
    """Get full path to ROCm tool by searching known locations and PATH.

    Search order: THEROCK_BIN_DIR → ROCM_PATH → /opt/rocm/bin →
    /opt/rocm*/bin (multi-version) → PATH (shutil.which) → bare name.
    """
    therock_bin_dir = os.getenv("THEROCK_BIN_DIR")
    if therock_bin_dir:
        full = os.path.join(therock_bin_dir, tool_name)
        if os.path.exists(full):
            return full

    rocm_path = os.getenv("ROCM_PATH", "")
    if rocm_path:
        full = os.path.join(rocm_path, "bin", tool_name)
        if os.path.exists(full):
            return full

    std = os.path.join("/opt/rocm/bin", tool_name)
    if os.path.exists(std):
        return std

    for path in sorted(glob.glob(os.path.join("/opt/rocm*/bin", tool_name)), reverse=True):
        if os.path.exists(path):
            return path

    found = shutil.which(tool_name)
    if found:
        return found

    return tool_name


# AMD GPU Device ID to Name mapping (fallback when amd-smi fails)
AMD_DEVICE_ID_MAP = {
    # MI300 Series
    "74a1": "MI300X",
    "74a0": "MI300A", 
    "74a2": "MI300",
    "74a3": "MI300X",  # Alternate MI300X device ID
    # MI325/MI350 Series
    "74b5": "MI325X",
    "74b0": "MI350",
    # MI200 Series
    "740f": "MI250X",
    "740c": "MI250",
    "7408": "MI210",
    # MI100 Series
    "738c": "MI100",
    "738e": "MI100",
    # Radeon Pro
    "73a1": "W7900",
    "73a5": "W7800",
    # Consumer GPUs
    "744c": "RX 7900 XTX",
    "7480": "RX 9070 XT",
}

# AMD GPU Device ID to Architecture (gfx) mapping
AMD_DEVICE_ID_TO_ARCH = {
    # MI300 Series - gfx942
    "74a1": "gfx942",  # MI300X
    "74a0": "gfx942",  # MI300A
    "74a2": "gfx942",  # MI300
    "74a3": "gfx942",  # MI300X (alternate)
    "75a3": "gfx942",  # MI300X variant
    # MI350 Series - gfx950
    "74b0": "gfx950",  # MI350
    "74b1": "gfx950",  # MI350X
    "74b5": "gfx950",  # MI325X / MI355X
    # MI200 Series - gfx90a
    "740f": "gfx90a",  # MI250X
    "740c": "gfx90a",  # MI250
    "7408": "gfx90a",  # MI210
    # MI100 Series - gfx908
    "738c": "gfx908",  # MI100
    "738e": "gfx908",  # MI100
    # RDNA3 - gfx1100/gfx1101
    "73a1": "gfx1100",  # W7900
    "73a5": "gfx1100",  # W7800
    "744c": "gfx1100",  # RX 7900 XTX
    # RDNA4 - gfx1200
    "7480": "gfx1200",  # RX 9070 XT
}

logger = logging.getLogger(__name__)

def normalize_gpu_name(raw_name: str) -> str:
    """
    Normalize GPU name to short format (e.g., "AMD Instinct MI300X" -> "MI300X").
    
    Args:
        raw_name: Raw GPU name from amd-smi, rocm-smi, or other sources
        
    Returns:
        Normalized short name (e.g., "MI300X", "RX 7900 XTX")
    """
    if not raw_name:
        return "Unknown"
    
    name = str(raw_name).strip()
    
    # Skip invalid values
    if name.lower() in ("n/a", "none", "", "unknown"):
        return "Unknown"
    
    # Try MI series pattern: "AMD Instinct MI300X" -> "MI300X"
    mi_match = re.search(r'\b(MI\d+[A-Z]?)\b', name, re.IGNORECASE)
    if mi_match:
        normalized = mi_match.group(1).upper()
        # Normalize known variants where X suffix is sometimes missing
        # Only for GPUs that don't have distinct non-X variants
        # MI355/MI325 are only available as X variants
        MI_X_ONLY_VARIANTS = {
            "MI355": "MI355X",
            "MI325": "MI325X",
        }
        return MI_X_ONLY_VARIANTS.get(normalized, normalized)
    
    # Try RX series pattern: "Radeon RX 7900 XTX" -> "RX 7900 XTX"
    rx_match = re.search(r'(RX\s*\d+[^\[,]*)', name, re.IGNORECASE)
    if rx_match:
        return rx_match.group(1).strip()
    
    # Try Radeon Pro pattern: "AMD Radeon Pro W7900" -> "W7900"
    pro_match = re.search(r'(W\d+)', name, re.IGNORECASE)
    if pro_match:
        return pro_match.group(1).upper()
    
    # Try gfx pattern as last resort
    gfx_match = re.search(r'\b(gfx\d+[a-z]?)\b', name, re.IGNORECASE)
    if gfx_match:
        return gfx_match.group(1).lower()
    
    # Fallback: remove common prefixes
    for prefix in ["AMD ", "Advanced Micro Devices ", "Instinct ", "Radeon "]:
        if name.upper().startswith(prefix.upper()):
            name = name[len(prefix):]
    
    return name.strip()


def detect_gpu_arch() -> str:
    """
    Detect GPU architecture (e.g., gfx942, gfx950, gfx1100).
    Uses multiple fallback methods: rocminfo -> amd-smi -> lspci -> GPU name mapping.
    
    Returns:
        GPU architecture string (e.g., "gfx942") or "unknown"
    """
    GPU_NAME_TO_ARCH = {
        "MI300X": "gfx942", "MI300A": "gfx942", "MI300": "gfx942",
        "MI325X": "gfx950", "MI350X": "gfx950", "MI350": "gfx950", "MI355X": "gfx950",
        "MI250X": "gfx90a", "MI250": "gfx90a", "MI210": "gfx90a",
        "MI100": "gfx908",
        "W7900": "gfx1100", "W7800": "gfx1100",
        "RX 7900 XTX": "gfx1100", "RX 7900 XT": "gfx1100",
        "RX 9070 XT": "gfx1200", "RX 9070": "gfx1200",
    }

    # Method 1: Check environment variables (instant, no subprocess, most reliable in CI)
    for env_var in ("GPU_ARCH", "GPU_FAMILY"):
        env_val = os.environ.get(env_var, "").strip()
        if env_val:
            if env_val.lower().startswith("gfx"):
                logger.debug(f"Detected arch via {env_var} env var: {env_val.lower()}")
                return env_val.lower()
            normalized = env_val.upper()
            if normalized in GPU_NAME_TO_ARCH:
                logger.debug(f"Detected arch via {env_var} env var mapping: {env_val} -> {GPU_NAME_TO_ARCH[normalized]}")
                return GPU_NAME_TO_ARCH[normalized]

    # Method 2: Check bm_config.json (pre-captured on host, works inside Docker)
    bm_paths = ["logs/bm_config.json", "/workspace/logs/bm_config.json",
                "logs/host_bm_config.json"]
    bm_env = os.environ.get("BM_CONFIG_PATH", "").strip()
    if bm_env:
        bm_paths.insert(0, bm_env)
    for bm_path in bm_paths:
        try:
            if os.path.exists(bm_path):
                with open(bm_path) as f:
                    bm_data = json.load(f)
                gpu_info = bm_data.get("gpu", {})
                if isinstance(gpu_info, dict):
                    arch = gpu_info.get("arch", "")
                    if arch and arch.lower().startswith("gfx"):
                        logger.debug(f"Detected arch via {bm_path}: {arch.lower()}")
                        return arch.lower()
        except Exception as e:
            logger.debug(f"Failed to read {bm_path}: {e}")

    # Method 3: Try rocminfo (most reliable live detection)
    try:
        rocminfo_cmd = _get_rocm_tool_path("rocminfo")
        result = subprocess.run(
            [rocminfo_cmd],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'gfx' in line.lower():
                    parts = line.split()
                    for part in parts:
                        if part.lower().startswith('gfx'):
                            logger.debug(f"Detected arch via rocminfo: {part.lower().strip()}")
                            return part.lower().strip()
            logger.debug("rocminfo ran but no gfx arch found in output")
        else:
            logger.debug(f"rocminfo returned exit code {result.returncode}: {result.stderr.strip()[:200]}")
    except FileNotFoundError:
        logger.debug("rocminfo not found")
    except Exception as e:
        logger.debug(f"rocminfo failed: {e}")
    
    # Method 4: Use target_graphics_version from amd-smi (direct gfx value)
    try:
        amd_smi_cmd = _get_rocm_tool_path("amd-smi")
        result = subprocess.run(
            [amd_smi_cmd, 'static', '--json'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if isinstance(data, dict) and 'gpu_data' in data:
                data = data['gpu_data']
            if isinstance(data, list) and len(data) > 0:
                asic = data[0].get('asic', {})
                target_gfx = asic.get('target_graphics_version', '')
                if target_gfx and target_gfx.lower().startswith('gfx'):
                    logger.debug(f"Detected arch via amd-smi target_graphics_version: {target_gfx.lower()}")
                    return target_gfx.lower()
                device_id = str(asic.get("device_id", "")).replace("0x", "").lower()
                if device_id in AMD_DEVICE_ID_TO_ARCH:
                    logger.debug(f"Detected arch via amd-smi device_id {device_id}: {AMD_DEVICE_ID_TO_ARCH[device_id]}")
                    return AMD_DEVICE_ID_TO_ARCH[device_id]
                logger.debug(f"amd-smi ran but no gfx/device_id match (asic: {asic})")
        else:
            logger.debug(f"amd-smi static returned exit code {result.returncode}: {result.stderr.strip()[:200]}")
    except FileNotFoundError:
        logger.debug("amd-smi not found")
    except Exception as e:
        logger.debug(f"amd-smi static failed: {e}")
    
    # Method 5: sysfs DRM device ID (always available on Linux, no tools, no permissions)
    try:
        import glob as _glob
        for card_dir in sorted(_glob.glob('/sys/class/drm/card[0-9]*/device')):
            try:
                vendor = open(f'{card_dir}/vendor').read().strip()
                if vendor != '0x1002':
                    continue
                device_id = open(f'{card_dir}/device').read().strip().replace('0x', '').lower()
                if device_id in AMD_DEVICE_ID_TO_ARCH:
                    logger.debug(f"Detected arch via sysfs device_id {device_id}: {AMD_DEVICE_ID_TO_ARCH[device_id]}")
                    return AMD_DEVICE_ID_TO_ARCH[device_id]
                logger.debug(f"sysfs found AMD GPU device_id {device_id} but not in mapping")
            except (IOError, OSError):
                continue
    except Exception as e:
        logger.debug(f"sysfs DRM detection failed: {e}")

    # Method 6: lspci device ID (no GPU permissions needed)
    # PCI classes for AMD GPUs: VGA [0300], Display [0380], 3D [0302],
    # Processing accelerators [1200] (MI300/MI350 compute GPUs), Co-processor [0b40]
    try:
        result = subprocess.run(
            ['lspci', '-nn'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            gpu_classes = ('VGA', 'Display', '3D', 'Processing', 'Accelerator', 'Co-processor')
            for line in result.stdout.split('\n'):
                if '1002:' in line and any(cls in line for cls in gpu_classes):
                    match = re.search(r'\[1002:([0-9a-fA-F]{4})\]', line)
                    if match:
                        device_id = match.group(1).lower()
                        if device_id in AMD_DEVICE_ID_TO_ARCH:
                            logger.debug(f"Detected arch via lspci device_id {device_id}: {AMD_DEVICE_ID_TO_ARCH[device_id]}")
                            return AMD_DEVICE_ID_TO_ARCH[device_id]
                        logger.debug(f"lspci found AMD GPU device_id {device_id} but not in mapping")
    except FileNotFoundError:
        logger.debug("lspci not found")
    except Exception as e:
        logger.debug(f"lspci failed: {e}")
    
    # Method 7: Fallback - map GPU name from amd-smi marketing name
    try:
        amd_smi_cmd = _get_rocm_tool_path("amd-smi")
        result = subprocess.run(
            [amd_smi_cmd, 'static', '--json'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if isinstance(data, list) and len(data) > 0:
                asic = data[0].get('asic', {})
                gpu_name = asic.get('market_name') or asic.get('marketing_name', '')
                
                mi_match = re.search(r'\b(MI\d+[A-Z]?)\b', gpu_name, re.IGNORECASE)
                if mi_match:
                    normalized = mi_match.group(1).upper()
                    if normalized in GPU_NAME_TO_ARCH:
                        logger.debug(f"Detected arch via GPU name mapping: {gpu_name} -> {GPU_NAME_TO_ARCH[normalized]}")
                        return GPU_NAME_TO_ARCH[normalized]
                
                for name_pattern, arch in GPU_NAME_TO_ARCH.items():
                    if name_pattern.lower() in gpu_name.lower():
                        logger.debug(f"Detected arch via GPU name mapping: {gpu_name} -> {arch}")
                        return arch
    except Exception as e:
        logger.debug(f"amd-smi name fallback failed: {e}")
    
    logger.warning("Could not detect GPU architecture via any method (env vars, bm_config, rocminfo, amd-smi, sysfs, lspci)")
    return "unknown"


def detect_bkc_version() -> str:
    """Detect BKC (Best Known Configuration) firmware version.

    Checks BKC_VERSION env var first (set by workflow), then tries
    to run platypi locally if available. Never installs or blocks.

    Returns:
        BKC version string or empty string if not available.
    """
    bkc = os.environ.get("BKC_VERSION", "").strip()
    if bkc:
        return bkc

    # Try reading from saved file
    saved_paths = [
        os.path.expanduser("~/.platypi_venv/systemBKC"),
        "logs/bkc.txt",
    ]
    for path in saved_paths:
        try:
            if os.path.exists(path):
                with open(path) as f:
                    val = f.read().strip()
                if val:
                    return val
        except Exception:
            pass

    # Try running platypi if available (no install)
    import shutil
    import socket

    platypi = shutil.which("platypi")
    if not platypi:
        for p in [
            "/home/orch/error-scraper-venv/bin/platypi",
            os.path.expanduser("~/.platypi_venv/bin/platypi"),
            os.path.expanduser("~/.local/bin/platypi"),
        ]:
            if os.path.exists(p):
                platypi = p
                break

    if not platypi:
        return ""

    try:
        hostname = socket.gethostname().split(".")[0]
        result = subprocess.run(
            [platypi, hostname, "--platform_fw_get"],
            capture_output=True, text=True, timeout=60,
        )
        output = result.stdout + result.stderr
        for field in ["'pldm':", "'bkc':", "'bundle':"]:
            for line in output.split("\n"):
                if field in line.lower():
                    try:
                        ver = line.split("'")[-2]
                        return ver[3:] if ver.startswith("00.") else ver
                    except (IndexError, ValueError):
                        continue
        match = re.search(r"pldm['\"]?\s*[:=]\s*['\"]?([\d.]+)", output, re.I)
        if match:
            ver = match.group(1)
            return ver[3:] if ver.startswith("00.") else ver
    except Exception as e:
        logger.debug(f"BKC detection via platypi failed: {e}")

    return ""


def _get_gpu_name_from_device_id(device_id: str) -> str:
    """Get GPU name from device ID using lookup table.
    
    Args:
        device_id: PCI device ID (e.g., "74a1")
        
    Returns:
        str: GPU name (e.g., "MI300X") or None if not found
    """
    if not device_id:
        return None
    # Normalize device_id (remove 0x prefix, lowercase)
    device_id = device_id.lower().replace("0x", "")
    return AMD_DEVICE_ID_MAP.get(device_id)


def _get_gpu_name_dynamic(device_id: str = None, gpu_index: int = 0) -> str:
    """Dynamically get GPU name from multiple sources (for unknown device IDs).
    
    Priority order:
    1. amd-smi static (market_name)
    2. sysfs product_name
    3. rocminfo (Marketing Name)
    4. Device ID mapping table
    5. Generic fallback
    
    Args:
        device_id: PCI device ID (e.g., "74a1") - used for mapping fallback
        gpu_index: GPU index for tool queries
        
    Returns:
        str: GPU name (e.g., "MI300X") or generic name if not found
    """
    import json as json_module
    
    # Normalize device_id
    if device_id:
        device_id = device_id.lower().replace("0x", "")
    
    # Try 1: amd-smi static (market_name)
    try:
        amd_smi = _get_rocm_tool_path("amd-smi")
        if amd_smi:
            result = subprocess.run(
                [amd_smi, "static", "--json", "-g", str(gpu_index)],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                data = json_module.loads(result.stdout)
                if isinstance(data, list) and data:
                    asic = data[0].get("asic", {})
                    raw_name = asic.get("market_name") or asic.get("marketing_name")
                    if raw_name:
                        normalized = normalize_gpu_name(raw_name)
                        if normalized != "Unknown":
                            logger.debug(f"_get_gpu_name_dynamic: amd-smi raw='{raw_name}' -> '{normalized}'")
                            return normalized
    except Exception:
        pass
    
    # Try 2: sysfs product_name (works inside containers)
    try:
        drm_cards = [d for d in os.listdir("/sys/class/drm") if d.startswith("card") and d[4:].isdigit()]
        for card in sorted(drm_cards):
            product_path = f"/sys/class/drm/{card}/device/product_name"
            if os.path.exists(product_path):
                with open(product_path, 'r') as f:
                    raw_name = f.read().strip()
                    if raw_name:
                        normalized = normalize_gpu_name(raw_name)
                        if normalized != "Unknown":
                            logger.debug(f"_get_gpu_name_dynamic: sysfs raw='{raw_name}' -> '{normalized}'")
                            return normalized
    except Exception:
        pass
    
    # Try 3: rocminfo (Marketing Name)
    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'Marketing Name' in line and ':' in line:
                    raw_name = line.split(':', 1)[1].strip()
                    if raw_name:
                        normalized = normalize_gpu_name(raw_name)
                        if normalized != "Unknown":
                            logger.debug(f"_get_gpu_name_dynamic: rocminfo raw='{raw_name}' -> '{normalized}'")
                            return normalized
    except Exception:
        pass
    
    # Try 4: Device ID mapping table (static lookup)
    if device_id:
        mapped_name = AMD_DEVICE_ID_MAP.get(device_id)
        if mapped_name:
            logger.debug(f"_get_gpu_name_dynamic: device_id '{device_id}' -> '{mapped_name}'")
            return mapped_name
    
    # Fallback: Return generic name with device ID if available
    if device_id:
        return f"AMD GPU ({device_id})"
    return "AMD GPU"


class HardwareDetector:
    """Simple hardware detector."""

    def __init__(self):
        """Initialize hardware detector."""
        self.cpu_info = None
        self.gpu_list = []

    def detect_all(self):
        """Detect all hardware (CPU and GPU)."""
        self.detect_cpu()
        self.detect_gpu()

    def detect_cpu(self) -> CpuInfo:
        """Detect CPU information from /proc/cpuinfo and lscpu.

        Returns:
            CpuInfo object
        """
        try:
            # Read /proc/cpuinfo
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()

            # Extract model name
            model_match = re.search(r"model name\s*:\s*(.+)", cpuinfo)
            model_name = model_match.group(1).strip() if model_match else "Unknown"

            # Count cores (logical processors)
            cores = len(re.findall(r"^processor\s*:", cpuinfo, re.MULTILINE))

            # Count sockets
            physical_ids = set(re.findall(r"physical id\s*:\s*(\d+)", cpuinfo))
            sockets = len(physical_ids) if physical_ids else 1

            # Get CPU MHz (clock speed)
            clock_match = re.search(r"cpu MHz\s*:\s*([\d.]+)", cpuinfo)
            clock_speed_mhz = int(float(clock_match.group(1))) if clock_match else 0

            # Get cache sizes from first processor
            l1_cache_kb = 0
            l2_cache_kb = 0
            l3_cache_kb = 0
            numa_nodes = 1

            # Try to get cache info from lscpu if available
            try:
                lscpu_output = subprocess.check_output(["lscpu", "-B"], text=True)

                # L1d cache (data)
                l1d_match = re.search(
                    r"L1d cache:\s*(\d+)\s*([KMG]?)", lscpu_output, re.IGNORECASE
                )
                if l1d_match:
                    size = int(l1d_match.group(1))
                    unit = l1d_match.group(2).upper()
                    if unit == "M":
                        size *= 1024
                    elif unit == "G":
                        size *= 1024 * 1024
                    l1_cache_kb = size

                # L1i cache (instruction)
                l1i_match = re.search(
                    r"L1i cache:\s*(\d+)\s*([KMG]?)", lscpu_output, re.IGNORECASE
                )
                if l1i_match:
                    size = int(l1i_match.group(1))
                    unit = l1i_match.group(2).upper()
                    if unit == "M":
                        size *= 1024
                    elif unit == "G":
                        size *= 1024 * 1024
                    l1_cache_kb += size

                # L2 cache
                l2_match = re.search(
                    r"L2 cache:\s*(\d+)\s*([KMG]?)", lscpu_output, re.IGNORECASE
                )
                if l2_match:
                    size = int(l2_match.group(1))
                    unit = l2_match.group(2).upper()
                    if unit == "M":
                        size *= 1024
                    elif unit == "G":
                        size *= 1024 * 1024
                    l2_cache_kb = size

                # L3 cache
                l3_match = re.search(
                    r"L3 cache:\s*(\d+)\s*([KMG]?)", lscpu_output, re.IGNORECASE
                )
                if l3_match:
                    size = int(l3_match.group(1))
                    unit = l3_match.group(2).upper()
                    if unit == "M":
                        size *= 1024
                    elif unit == "G":
                        size *= 1024 * 1024
                    l3_cache_kb = size

                # Get NUMA nodes
                numa_match = re.search(r"NUMA node\(s\):\s*(\d+)", lscpu_output)
                numa_nodes = int(numa_match.group(1)) if numa_match else 1

            except Exception as e:
                # Fallback: try to get cache from /proc/cpuinfo
                cache_match = re.search(r"cache size\s*:\s*(\d+)\s*KB", cpuinfo)
                if cache_match:
                    # This is usually L2 or L3 cache
                    l2_cache_kb = int(cache_match.group(1))

            # Get RAM size from /proc/meminfo
            ram_size_gb = 0
            try:
                with open("/proc/meminfo", "r") as f:
                    meminfo = f.read()
                mem_match = re.search(r"MemTotal:\s*(\d+)\s*kB", meminfo)
                if mem_match:
                    # Convert KB to GB
                    ram_size_gb = int(mem_match.group(1)) // (1024 * 1024)
            except Exception:
                pass

            self.cpu_info = CpuInfo(
                model_name=model_name,
                cores=cores,
                sockets=sockets,
                ram_size_gb=ram_size_gb,
                numa_nodes=numa_nodes,
                clock_speed_mhz=clock_speed_mhz,
                l1_cache_kb=l1_cache_kb,
                l2_cache_kb=l2_cache_kb,
                l3_cache_kb=l3_cache_kb,
            )

        except Exception:
            self.cpu_info = CpuInfo()

        return self.cpu_info

    def detect_gpu(self) -> List[GpuInfo]:
        """Detect GPU information.
        
        Priority order:
        1. amd-smi (most complete, newest)
        2. rocm-smi (fallback)
        3. lspci (final fallback for systems without ROCm)

        Returns:
            List of GpuInfo objects
        """
        self.gpu_list = []

        # 1. Try amd-smi first (PRIMARY - most complete)
        if self._detect_gpu_from_amd_smi():
            return self.gpu_list

        # 2. Fallback to rocm-smi
        if self._detect_gpu_from_rocm_smi():
            return self.gpu_list

        # 3. Final fallback to lspci
        self._detect_gpu_from_lspci()
        return self.gpu_list

    def _detect_gpu_from_amd_smi(self) -> bool:
        """Detect GPUs using amd-smi (primary method).
        
        Returns:
            bool: True if GPUs were detected, False otherwise
        """
        

        try:
            amd_smi_cmd = _get_rocm_tool_path("amd-smi")
            
            # Print amd-smi table output to console
            table_result = subprocess.run(
                [amd_smi_cmd],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if table_result.returncode == 0 and table_result.stdout:
                print(table_result.stdout)
            
            # Get static info (GPU details) as JSON for parsing
            result = subprocess.run(
                [amd_smi_cmd, "static", "--json"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return False

            if not result.stdout.strip():
                return False

            data = json.loads(result.stdout)

            # Handle different amd-smi output structures
            gpu_list = data
            if isinstance(data, dict):
                gpu_list = data.get("gpu_data") or data.get("gpus") or []
                if not gpu_list and len(data) > 0:
                    gpu_list = [data]
            
            if not gpu_list or len(gpu_list) == 0:
                return False

            # Parse each GPU
            for i, gpu_data in enumerate(gpu_list):
                # Extract ASIC info
                asic = gpu_data.get("asic", {}) or gpu_data
                
                # Get GPU name from asic.market_name (primary source)
                raw_name = asic.get("market_name") or asic.get("marketing_name")
                
                # Normalize to short name (e.g., "AMD Instinct MI300X" -> "MI300X")
                product_name = normalize_gpu_name(raw_name) if raw_name else f"AMD GPU {i}"
                
                # Log source and final name
                logger.info(f"GPU[{i}] amd-smi: raw='{raw_name}' -> normalized='{product_name}'")
                
                # Get device/revision ID
                device_id_raw = asic.get("device_id") or asic.get("pci_id") or ""
                device_id = str(device_id_raw).replace("0x", "").replace("0X", "")
                
                revision_id_raw = asic.get("rev_id") or asic.get("revision_id") or ""
                revision_id = str(revision_id_raw).replace("0x", "").replace("0X", "")
                
                # Get VRAM (in MB, convert to GB)
                vram_info = gpu_data.get("vram", {})
                if isinstance(vram_info, dict):
                    vram_total = vram_info.get("total", {})
                    if isinstance(vram_total, dict):
                        vram_mb = vram_total.get("value", 0)
                    else:
                        vram_mb = vram_total
                else:
                    vram_mb = 0
                vram_gb = int(vram_mb) // 1024 if vram_mb else 0
                
                # Get VBIOS info - check multiple locations
                vbios_str = "Unknown"
                
                # Try direct vbios field
                vbios = gpu_data.get("vbios", {})
                if isinstance(vbios, dict):
                    vbios_str = vbios.get("version") or vbios.get("vbios_version") or "Unknown"
                elif vbios:
                    vbios_str = str(vbios)
                
                # Fallback to ifwi.part_number (seen as "113-M3000108-103" in MI300X)
                if vbios_str == "Unknown":
                    ifwi = gpu_data.get("ifwi", {})
                    if isinstance(ifwi, dict):
                        vbios_str = ifwi.get("part_number") or ifwi.get("version") or "Unknown"
                
                # Driver info
                driver = gpu_data.get("driver", {})
                if isinstance(driver, dict):
                    host_driver = driver.get("version", "Unknown")
                else:
                    host_driver = str(driver) if driver else "Unknown"
                
                # Partition mode (SPX, NPX, CPX, TPX, QPX, DPX)
                # From amd-smi partition output: {"accelerator_type": "SPX", "memory": "NPS1", ...}
                partition_mode = "Unknown"
                partition = gpu_data.get("partition", {})
                if isinstance(partition, dict):
                    # Try multiple field names - accelerator_type is the key one for SPX/NPX
                    partition_mode = (
                        partition.get("accelerator_type") or  # SPX, NPX, etc.
                        partition.get("compute_partition") or
                        partition.get("current_compute_partition") or
                        partition.get("partition_mode") or
                        "Unknown"
                    )
                elif partition:
                    partition_mode = str(partition)

                # Also try top-level fields
                if partition_mode == "Unknown":
                    partition_mode = (
                        gpu_data.get("accelerator_type") or
                        gpu_data.get("compute_partition") or
                        gpu_data.get("current_compute_partition") or 
                        gpu_data.get("partition_mode") or
                        "Unknown"
                    )
                
                # XGMI info
                xgmi = gpu_data.get("xgmi", {})
                if isinstance(xgmi, dict):
                    xgmi_type = xgmi.get("xgmi_hive_id", "Unknown")
                else:
                    xgmi_type = str(xgmi) if xgmi else "Unknown"
                
                # Firmware info - check multiple sections
                firmwares = []
                
                # Try fw_info section (common structure)
                fw_data = gpu_data.get("fw_info", {}) or gpu_data.get("firmware", {})
                if isinstance(fw_data, dict):
                    for fw_name, fw_info in fw_data.items():
                        if isinstance(fw_info, dict):
                            fw_ver = fw_info.get("version", fw_info.get("fw_version", "Unknown"))
                        else:
                            fw_ver = str(fw_info)
                        if fw_ver and fw_ver != "Unknown":
                            firmwares.append({"name": fw_name, "version": str(fw_ver)})
                
                # Try ifwi section (seen in MI300X output)
                ifwi_data = gpu_data.get("ifwi", {})
                if isinstance(ifwi_data, dict):
                    if ifwi_data.get("name"):
                        firmwares.append({"name": "IFWI", "version": ifwi_data.get("version", "Unknown")})
                
                # Try individual firmware fields that may exist at top level
                fw_fields = ["smc", "sdma", "sdma2", "mec", "mec2", "rlc", "sos", "vcn", 
                             "ta_ras", "ta_xgmi", "pldm", "ce", "pfp", "me"]
                for fw_field in fw_fields:
                    fw_val = gpu_data.get(fw_field) or gpu_data.get(fw_field.upper())
                    if fw_val:
                        if isinstance(fw_val, dict):
                            fw_ver = fw_val.get("version", str(fw_val))
                        else:
                            fw_ver = str(fw_val)
                        firmwares.append({"name": fw_field.upper(), "version": fw_ver})
                
                # Get marketing name
                # Keep full name for marketing_name field
                marketing_name = raw_name or product_name
                
                # Create GpuInfo object
                gpu = GpuInfo(
                    device_id=device_id,
                    revision_id=revision_id,
                    product_name=str(product_name),
                    marketing_name=str(marketing_name),
                    vendor="AMD",
                    vram_size_gb=vram_gb,
                    vbios=vbios_str,
                    host_driver=host_driver,
                    partition_mode=partition_mode,
                    xgmi_type=str(xgmi_type),
                    firmwares=firmwares,
                )
                self.gpu_list.append(gpu)        
            # Get clock info from amd-smi metric
            try:
                result = subprocess.run(
                    [amd_smi_cmd, "metric", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    metric_data = json.loads(result.stdout)
                    for i, gpu in enumerate(self.gpu_list):
                        if i < len(metric_data):
                            gpu_metric = metric_data[i]
                            clocks = gpu_metric.get("clocks", {}) or gpu_metric.get("clock", {})
                            if isinstance(clocks, dict):
                                sclk = clocks.get("gfx", {}) or clocks.get("sclk", {})
                                mclk = clocks.get("mem", {}) or clocks.get("mclk", {})
                                if isinstance(sclk, dict):
                                    gpu.sys_clock_mhz = int(sclk.get("value", sclk.get("clk", 0)))
                                elif isinstance(sclk, (int, float)):
                                    gpu.sys_clock_mhz = int(sclk)
                                if isinstance(mclk, dict):
                                    gpu.mem_clock_mhz = int(mclk.get("value", mclk.get("clk", 0)))
                                elif isinstance(mclk, (int, float)):
                                    gpu.mem_clock_mhz = int(mclk)
            except Exception:
                pass

            # Get partition info if still Unknown
            if self.gpu_list and self.gpu_list[0].partition_mode == "Unknown":
                try:
                    logger.debug("Running amd-smi partition --json to get partition mode...")
                    result = subprocess.run(
                        [amd_smi_cmd, "partition", "--json"],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if result.returncode == 0:
                        partition_data = json.loads(result.stdout)
                        logger.debug(f"Partition data: {partition_data}")
                        
                        # Helper to extract partition mode from a dict
                        def extract_partition_mode(data):
                            if not isinstance(data, dict):
                                return None
                            return (
                                data.get("current_compute_partition") or
                                data.get("accelerator_type") or
                                data.get("compute_partition") or
                                data.get("partition_mode") or
                                data.get("partition") or
                                None
                            )
                        
                        # Handle various amd-smi output formats
                        if isinstance(partition_data, list):
                            # Format: [{"current_compute_partition": "SPX"}, ...]
                            for i, gpu in enumerate(self.gpu_list):
                                if i < len(partition_data):
                                    mode = extract_partition_mode(partition_data[i])
                                    if mode:
                                        gpu.partition_mode = mode
                                        logger.debug(f"GPU {i} partition_mode: {mode}")
                        elif isinstance(partition_data, dict):
                            # Try to find partition info in various locations
                            mode = None
                            
                            # Format: {"current_compute_partition": "SPX"} (direct)
                            mode = extract_partition_mode(partition_data)
                            
                            # Format: {"current_partition": {"current_compute_partition": "SPX"}}
                            if not mode and "current_partition" in partition_data:
                                cp = partition_data["current_partition"]
                                if isinstance(cp, dict):
                                    mode = extract_partition_mode(cp)
                                elif isinstance(cp, list) and cp:
                                    mode = extract_partition_mode(cp[0])
                            
                            # Format: {"gpu_data": [{"current_compute_partition": "SPX"}]}
                            if not mode:
                                for key in ["gpu_data", "gpus", "gpu"]:
                                    if key in partition_data:
                                        gpu_list = partition_data[key]
                                        if isinstance(gpu_list, list) and gpu_list:
                                            mode = extract_partition_mode(gpu_list[0])
                                        elif isinstance(gpu_list, dict):
                                            mode = extract_partition_mode(gpu_list)
                                        if mode:
                                            break
                            
                            # Apply to all GPUs
                            if mode:
                                for gpu in self.gpu_list:
                                    gpu.partition_mode = mode
                                logger.debug(f"Partition mode for all GPUs: {mode}")
                    else:
                        logger.debug(f"amd-smi partition failed: {result.stderr}")
                except Exception as e:
                    logger.debug(f"Error getting partition info: {e}")

            # Get firmware info from amd-smi firmware --json
            # The static output may include partial firmware (e.g. just IFWI),
            # so re-fetch if we have fewer than 5 entries (a full list has ~13).
            if self.gpu_list and len(self.gpu_list[0].firmwares) < 5:
                try:
                    logger.debug("Running amd-smi firmware --json to get firmware versions...")
                    result = subprocess.run(
                        [amd_smi_cmd, "firmware", "--json"],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if result.returncode == 0:
                        fw_data = json.loads(result.stdout)
                        
                        # Map fw_id names to standardized DB field names
                        FW_ID_MAP = {
                            "CP_MEC1": "mec",
                            "CP_MEC2": "mec2",
                            "RLC": "rlc",
                            "SDMA0": "sdma",
                            "SDMA1": "sdma2",
                            "VCN": "vcn",
                            "PSP_SOSDRV": "sos",
                            "TA_RAS": "ta_ras",
                            "TA_XGMI": "ta_xgmi",
                            "PM": "smc",
                            "PLDM_BUNDLE": "plfm_bundle",
                        }
                        
                        # Parse fw_list for each GPU
                        if isinstance(fw_data, list):
                            for i, gpu in enumerate(self.gpu_list):
                                if i < len(fw_data):
                                    gpu_fw = fw_data[i]
                                    fw_list = gpu_fw.get("fw_list", [])
                                    firmwares = []
                                    for fw in fw_list:
                                        fw_id = fw.get("fw_id", "")
                                        fw_version = fw.get("fw_version", "")
                                        if fw_id and fw_version:
                                            # Use mapped name or original (lowercase)
                                            name = FW_ID_MAP.get(fw_id, fw_id.lower())
                                            firmwares.append({"name": name, "version": fw_version})
                                    gpu.firmwares = firmwares
                                    logger.debug(f"GPU {i}: {len(firmwares)} firmwares detected")
                    else:
                        logger.debug(f"amd-smi firmware failed: {result.stderr}")
                except Exception as e:
                    logger.debug(f"Error getting firmware info: {e}")

            return len(self.gpu_list) > 0

        except FileNotFoundError:
            return False
        except json.JSONDecodeError:
            return False
        except Exception:
            return False

    def _detect_gpu_from_rocm_smi(self) -> bool:
        """Detect GPUs using rocm-smi (fallback method).

        Returns:
            bool: True if GPUs were detected, False otherwise
        """
        import json

        try:
            rocm_smi_cmd = _get_rocm_tool_path("rocm-smi")
            
            # Get GPU count and basic info
            result = subprocess.run(
                [rocm_smi_cmd, "--showid", "--json"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return False

            data = json.loads(result.stdout)
            if not data:
                return False

            # Count GPUs (keys like "card0", "card1", etc.)
            gpu_keys = [k for k in data.keys() if k.startswith("card")]
            if not gpu_keys:
                return False

            # Create GPU entries
            for gpu_key in sorted(gpu_keys):
                gpu_data = data[gpu_key]
                device_id = str(gpu_data.get("Device ID", "")).replace("0x", "").lower()
                revision_id = str(gpu_data.get("Device Rev", "")).replace("0x", "")
                
                # Get GPU name from Device Name (primary source for rocm-smi)
                raw_name = gpu_data.get("Device Name")
                
                # Normalize to short name
                product_name = normalize_gpu_name(raw_name) if raw_name else None
                
                # Fallback: try dynamic name resolution if name is invalid
                if not product_name or product_name == "Unknown":
                    product_name = _get_gpu_name_dynamic(device_id, gpu_index=len(self.gpu_list))
                    logger.info(f"GPU[{gpu_key}] rocm-smi: raw='{raw_name}' -> fallback='{product_name}'")
                else:
                    logger.info(f"GPU[{gpu_key}] rocm-smi: raw='{raw_name}' -> normalized='{product_name}'")
                
                gpu = GpuInfo(
                    device_id=device_id,
                    revision_id=revision_id,
                    product_name=product_name,
                    marketing_name=product_name,
                    vendor="AMD",
                )
                self.gpu_list.append(gpu)

            # Get VRAM info
            try:
                result = subprocess.run(
                    [rocm_smi_cmd, "--showmeminfo", "vram", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    vram_data = json.loads(result.stdout)
                    for i, gpu in enumerate(self.gpu_list):
                        gpu_key = f"card{i}"
                        if gpu_key in vram_data:
                            vram_bytes = vram_data[gpu_key].get("VRAM Total Memory (B)", 0)
                            gpu.vram_size_gb = int(vram_bytes) // (1024 * 1024 * 1024)
            except Exception:
                pass

            # Get VBIOS info
            try:
                result = subprocess.run(
                    [rocm_smi_cmd, "--showvbios", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    vbios_data = json.loads(result.stdout)
                    for i, gpu in enumerate(self.gpu_list):
                        gpu_key = f"card{i}"
                        if gpu_key in vbios_data:
                            gpu.vbios = vbios_data[gpu_key].get("VBIOS version", "Unknown")
            except Exception:
                pass

            # Get driver info
            try:
                result = subprocess.run(
                    [rocm_smi_cmd, "--showdriverversion", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    driver_data = json.loads(result.stdout)
                    driver_ver = driver_data.get("Driver version", "Unknown")
                    for gpu in self.gpu_list:
                        gpu.host_driver = driver_ver
            except Exception:
                pass

            # Get clock info
            try:
                result = subprocess.run(
                    [rocm_smi_cmd, "--showclocks", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    clock_data = json.loads(result.stdout)
                    for i, gpu in enumerate(self.gpu_list):
                        gpu_key = f"card{i}"
                        if gpu_key in clock_data:
                            gpu_clocks = clock_data[gpu_key]
                            sclk = gpu_clocks.get("sclk clock level", "")
                            mclk = gpu_clocks.get("mclk clock level", "")
                            sclk_match = re.search(r"(\d+)Mhz", str(sclk), re.IGNORECASE)
                            mclk_match = re.search(r"(\d+)Mhz", str(mclk), re.IGNORECASE)
                            if sclk_match:
                                gpu.sys_clock_mhz = int(sclk_match.group(1))
                            if mclk_match:
                                gpu.mem_clock_mhz = int(mclk_match.group(1))
            except Exception:
                pass

            # Get partition info via amd-smi partition command
            try:
                amd_smi_cmd = _get_rocm_tool_path("amd-smi")
                result = subprocess.run(
                    [amd_smi_cmd, "partition", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    partition_data = json.loads(result.stdout)
                    
                    # Helper to extract partition mode from a dict
                    def extract_part_mode(data):
                        if not isinstance(data, dict):
                            return None
                        return (
                            data.get("current_compute_partition") or
                            data.get("accelerator_type") or
                            data.get("compute_partition") or
                            data.get("partition_mode") or
                            data.get("partition") or
                            None
                        )
                    
                    mode = None
                    if isinstance(partition_data, list) and partition_data:
                        mode = extract_part_mode(partition_data[0])
                    elif isinstance(partition_data, dict):
                        mode = extract_part_mode(partition_data)
                        if not mode and "current_partition" in partition_data:
                            cp = partition_data["current_partition"]
                            if isinstance(cp, dict):
                                mode = extract_part_mode(cp)
                            elif isinstance(cp, list) and cp:
                                mode = extract_part_mode(cp[0])
                        if not mode:
                            for key in ["gpu_data", "gpus", "gpu"]:
                                if key in partition_data:
                                    gd = partition_data[key]
                                    if isinstance(gd, list) and gd:
                                        mode = extract_part_mode(gd[0])
                                    elif isinstance(gd, dict):
                                        mode = extract_part_mode(gd)
                                    if mode:
                                        break
                    
                    if mode:
                        for gpu in self.gpu_list:
                            gpu.partition_mode = mode
            except Exception:
                pass

            return len(self.gpu_list) > 0

        except FileNotFoundError:
            return False
        except json.JSONDecodeError:
            return False
        except Exception:
            return False

    def _detect_gpu_from_lspci(self) -> bool:
        """Detect GPUs using lspci (final fallback).
        
        Returns:
            bool: True if GPUs were detected, False otherwise
        """
        try:
            # Run lspci to find AMD GPUs
            logger.debug("Running lspci -d 1002: -nn...")
            result = subprocess.run(
                ["lspci", "-d", "1002:", "-nn"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                logger.debug(f"lspci failed with return code {result.returncode}")
                return False

            # Parse output
            for line in result.stdout.splitlines():
                # Only actual GPUs (VGA/Display controller)
                if "VGA compatible controller" in line or "Display controller" in line:
                    # Extract PCI address (XX:XX.X)
                    pci_match = re.match(
                        r"^([0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F])", line
                    )
                    pci_address = pci_match.group(1) if pci_match else ""

                    # Extract device ID from [1002:XXXX]
                    device_id_match = re.search(r"\[1002:([0-9a-fA-F]{4})\]", line)
                    device_id = device_id_match.group(1) if device_id_match else ""

                    # Extract revision ID
                    rev_match = re.search(r"\(rev\s+([0-9a-fA-F]{2})\)", line, re.IGNORECASE)
                    revision_id = rev_match.group(1) if rev_match else ""

                    # Extract product name from lspci output
                    raw_name = None
                    parts = line.split("]:")
                    if len(parts) >= 2:
                        product_part = parts[-1].strip()
                        raw_name = re.sub(r"\s*\([^)]*\)\s*$", "", product_part)
                        raw_name = re.sub(r"\s*\[[^\]]*\]\s*$", "", raw_name)
                        raw_name = raw_name.strip()
                    
                    # Normalize to short name
                    product_name = normalize_gpu_name(raw_name) if raw_name else None
                    
                    # Fallback: try dynamic name resolution
                    if not product_name or product_name == "Unknown":
                        product_name = _get_gpu_name_dynamic(device_id, gpu_index=len(self.gpu_list))
                        logger.info(f"GPU lspci: raw='{raw_name}' -> fallback='{product_name}'")
                    else:
                        logger.info(f"GPU lspci: raw='{raw_name}' -> normalized='{product_name}'")

                    gpu = GpuInfo(
                        device_id=device_id,
                        revision_id=revision_id,
                        product_name=product_name,
                        marketing_name=product_name,
                        vendor="AMD",
                        pci_address=pci_address,
                    )
                    self.gpu_list.append(gpu)

            # Try to get partition info via amd-smi (even if lspci was used for detection)
            if self.gpu_list:
                try:
                    import json
                    amd_smi_cmd = _get_rocm_tool_path("amd-smi")
                    result = subprocess.run(
                        [amd_smi_cmd, "partition", "--json"],
                        capture_output=True, text=True, timeout=30,
                    )
                    if result.returncode == 0:
                        partition_data = json.loads(result.stdout)
                        
                        # Helper to extract partition mode
                        def extract_pmode(data):
                            if not isinstance(data, dict):
                                return None
                            return (
                                data.get("current_compute_partition") or
                                data.get("accelerator_type") or
                                data.get("compute_partition") or
                                data.get("partition_mode") or
                                data.get("partition") or
                                None
                            )
                        
                        mode = None
                        if isinstance(partition_data, list) and partition_data:
                            mode = extract_pmode(partition_data[0])
                        elif isinstance(partition_data, dict):
                            mode = extract_pmode(partition_data)
                            if not mode and "current_partition" in partition_data:
                                cp = partition_data["current_partition"]
                                if isinstance(cp, dict):
                                    mode = extract_pmode(cp)
                                elif isinstance(cp, list) and cp:
                                    mode = extract_pmode(cp[0])
                            if not mode:
                                for key in ["gpu_data", "gpus", "gpu"]:
                                    if key in partition_data:
                                        gd = partition_data[key]
                                        if isinstance(gd, list) and gd:
                                            mode = extract_pmode(gd[0])
                                        elif isinstance(gd, dict):
                                            mode = extract_pmode(gd)
                                        if mode:
                                            break
                        
                        if mode:
                            for gpu in self.gpu_list:
                                gpu.partition_mode = mode
                except Exception:
                    pass

            return len(self.gpu_list) > 0

        except FileNotFoundError:
            logger.debug("lspci command not found")
            return False
        except Exception as e:
            logger.debug(f"lspci error: {e}")
            return False

    def get_cpu(self) -> Optional[CpuInfo]:
        """Get detected CPU information.

        Returns:
            CpuInfo: Detected CPU info or None
        """
        return self.cpu_info

    def get_is_cpu_initialized(self) -> bool:
        """Check if CPU detection completed.

        Returns:
            bool: True if CPU info available
        """
        return self.cpu_info is not None

    def get_is_gpu_initialized(self) -> bool:
        """Check if GPU detection completed.

        Returns:
            bool: True if GPU detection was attempted
        """
        return True  # Always true after detect_all() is called

    def getGpu(self):
        """Get GPU handler (camelCase compatibility alias).

        Returns:
            Self for accessing .adapters attribute
        """
        return self

    @property
    def adapters(self) -> List[GpuInfo]:
        """Get GPU adapters list (compatibility property).

        Returns:
            List[GpuInfo]: List of detected GPUs
        """
        return self.gpu_list
